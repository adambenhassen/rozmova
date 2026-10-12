"""Streams one clip to the app's /ws, like the browser client, and records everything the user would see or hear.

Down: JSON events and binary reply audio (2-byte little-endian utt id, 2 pad bytes, 24 kHz int16 PCM). Events used:
  reply_start {turn, utt}     maps a turn's reply to its audio utterance
  reply_delta {turn, text}    reply text as shown on screen
  reply_end {turn}            generation and TTS finished for that reply
  turn {turn, ...chain}       latency chain; re-sent as the turn updates (kept: the last snapshot per turn)
  word {text}                 the user's own words as shown on screen
  interject / backchannel     the agent's other spoken text
  tool / search / search_done tool cards, the Memory card and web-search snippets shown on screen
  error                       a backend failure: the clip is a failed run, not a "no reply" result
The app logs a turn's chain when the client reports playback, so the client sends {"cmd":"play_start"} on each reply's
first audio frame. The whole clip is always sent; afterwards the client keeps sending silence, as an open mic would,
until the first reply ends (plus DRAIN_S for late output and the final chain snapshot) or `timeout_s` passes."""
from __future__ import annotations

import asyncio
import json
import os
import time

import aiohttp
import numpy as np

from asr_client import CHUNK, stream_pcm
from audio import SR, pcm16

APP_URL = os.environ.get("APP_URL", "ws://app:8080/ws")
DRAIN_S = 1.5
SEARCH_SHOWN = 4                 # the search card renders the first 4 results (app/web/live/live.js cardDone)


class AppError(RuntimeError):
    """The app reported an error or dropped the connection: the clip failed, it is not a model result. `texts` keeps
    what the user had already seen or heard, so the zero-Russian audit still covers it."""

    def __init__(self, msg: str, texts: list[dict] | None = None):
        super().__init__(msg)
        self.texts = texts or []


async def _open(s: aiohttp.ClientSession, url: str, tries: int = 6):
    for _ in range(tries):
        ws = await s.ws_connect(url, heartbeat=20)
        while True:
            m = await ws.receive(timeout=15)
            if m.type != aiohttp.WSMsgType.TEXT:
                break
            ev = json.loads(m.data)
            if ev.get("type") == "ready":
                return ws
            if ev.get("type") == "error":
                break                                    # usually the ASR still held by the previous session
        await ws.close()
        await asyncio.sleep(1.0)
    raise RuntimeError("app /ws never became ready (ASR busy? close the Speakrail browser tab)")


async def converse(x: np.ndarray, timeout_s: float = 20.0, url: str = APP_URL) -> dict:
    """-> reply (first reply's text, "" if none), no_audio (that reply made no sound), early (it started before the
    clip ended), unfinished (no reply_end by the timeout), first (its last chain snapshot), turns (last snapshot per
    turn), decisions, texts (every user-visible text: [{kind, text}]). No reply within the timeout is a result (the
    model kept listening). Raises AppError on an app error or a dropped connection."""
    clip_s = len(x) / SR
    async with aiohttp.ClientSession() as s:
        ws = await _open(s, url)
        t0 = time.monotonic()
        st = {"turns": {}, "decisions": [], "deltas": {}, "texts": [], "utt_of": {}, "started": {}, "pcm": set(),
              "ended": set(), "first": None, "user": {}}
        done, sent = asyncio.Event(), asyncio.Event()

        async def reader():
            async for m in ws:
                if m.type == aiohttp.WSMsgType.BINARY:
                    utt = int.from_bytes(m.data[:2], "little")
                    if utt not in st["pcm"] and len(m.data) > 4:
                        st["pcm"].add(utt)
                        await ws.send_str(json.dumps({"cmd": "play_start", "utt": utt, "delay_ms": 0}))
                    continue
                if m.type != aiohttp.WSMsgType.TEXT:
                    continue
                ev = json.loads(m.data)
                k = ev.get("type")
                if k == "error":
                    st["texts"].append({"kind": "error", "text": json.dumps(ev, ensure_ascii=False)})
                    raise AppError(f"app error: {json.dumps(ev, ensure_ascii=False)[:300]}")
                if k == "decision":
                    st["decisions"].append(ev.get("decision"))
                elif k == "reply_start":
                    st["utt_of"][ev["turn"]] = ev["utt"]
                    st["started"][ev["turn"]] = time.monotonic() - t0
                    if st["first"] is None:
                        st["first"] = ev["turn"]
                elif k == "reply_delta":
                    st["deltas"][ev["turn"]] = st["deltas"].get(ev["turn"], "") + ev.get("text", "")
                elif k == "reply_end":
                    st["ended"].add(ev["turn"])
                    if ev["turn"] == st["first"] and sent.is_set():
                        done.set()
                elif k == "turn":
                    st["turns"][ev["turn"]] = ev
                elif k == "word":
                    st["texts"].append({"kind": "user_word", "text": ev.get("text", "")})
                    st["user"][ev.get("turn")] = st["user"].get(ev.get("turn"), "") + " " + ev.get("text", "")
                elif k in ("interject", "backchannel"):
                    st["texts"].append({"kind": k, "text": ev.get("text", "")})
                elif k in ("tool", "search"):
                    st["texts"].append({"kind": k, "text": json.dumps(
                        {f: ev.get(f) for f in ("name", "args", "result", "query", "store") if f in ev}, ensure_ascii=False)})
                elif k == "search_done":
                    st["texts"] += [{"kind": "search_result", "text": f"{r.get('title', '')}. {r.get('snippet', '')}"}
                                    for r in (ev.get("results") or [])[:SEARCH_SHOWN]]
            if not done.is_set():
                raise AppError("app closed the connection mid-clip")

        async def mic():
            await stream_pcm(ws, x, 1.0)
            sent.set()
            if st["first"] is not None and st["first"] in st["ended"]:
                done.set()                               # the reply finished while the clip was still playing
            silence = pcm16(np.zeros(CHUNK, np.float32))
            t = time.monotonic()
            while not done.is_set():
                await ws.send_bytes(silence)
                t += CHUNK / SR
                await asyncio.sleep(max(0.0, t - time.monotonic()))

        def texts() -> list[dict]:
            # the user's words joined per turn as well: clause-level checks need the whole line, not single words
            return st["texts"] + [{"kind": "user_transcript", "text": " ".join(t.split())} for _, t in st["user"].items()] + \
                [{"kind": "reply", "text": t} for _, t in sorted(st["deltas"].items())]

        rtask, mtask = asyncio.create_task(reader()), asyncio.create_task(mic())
        wait_done = asyncio.ensure_future(done.wait())
        try:
            finished, _ = await asyncio.wait({rtask, mtask, wait_done}, timeout=clip_s + timeout_s,
                                             return_when=asyncio.FIRST_COMPLETED)
            for t in (rtask, mtask):
                if t in finished and t.exception():
                    raise t.exception()
            if wait_done in finished:                    # late words, interjections and the final turn snapshot
                finished, _ = await asyncio.wait({rtask, mtask}, timeout=DRAIN_S, return_when=asyncio.FIRST_EXCEPTION)
                for t in finished:
                    if t.exception():
                        raise t.exception()
        except AppError as e:
            raise AppError(str(e), texts()) from e
        except Exception as e:                           # transport failure: still keep what was already shown
            raise AppError(f"client: {e!r}"[:300], texts()) from e
        finally:
            mtask.cancel(); rtask.cancel(); wait_done.cancel()
            await ws.close()
        f = st["first"]
        st["texts"] = texts()
        return {"reply": st["deltas"].get(f, "") if f is not None else "",
                "no_audio": f is not None and st["utt_of"].get(f) not in st["pcm"],
                "early": f is not None and st["started"][f] < clip_s,
                "unfinished": f is not None and f not in st["ended"],
                "first": st["turns"].get(f, {}) if f is not None else {},
                "turns": list(st["turns"].values()), "decisions": st["decisions"], "texts": st["texts"]}
