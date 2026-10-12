"""Streams one clip to the ASR bridge (services/asr/bridge.py, /v1/transcribe) and collects what it says.

Protocol: binary int16 16 kHz PCM up, then {"type":"input_audio.end"}; JSON down: session.created (with
turn_head.classes), transcript (text + timed words), turn (head probabilities per 80 ms frame, stamped `t` in ms of
session audio), error, session.ended. The bridge takes one session at a time: session_full means someone else holds it."""
from __future__ import annotations

import asyncio
import json
import os
import time

import aiohttp
import numpy as np

from audio import SR, pcm16

ASR_URL = os.environ.get("ASR_URL", "ws://asr:8765/v1/transcribe")
QUERY = "?encoding=pcm_s16le&sample_rate=16000&delay_ms=480&turn_detection=none&turn_probs=true&client_id=eval"
CHUNK = SR * 80 // 1000                      # 80 ms, the bridge's frame


class ASRError(RuntimeError):
    """The bridge failed mid-clip. `text` keeps what it had already transcribed (it was on screen: still audited)."""

    def __init__(self, msg: str, text: str = ""):
        super().__init__(msg)
        self.text = text


class Busy(RuntimeError):
    """The bridge's single session is taken (an open browser tab, or the previous run still closing)."""


async def _open(s: aiohttp.ClientSession, url: str, tries: int = 6):
    for i in range(tries):
        ws = await s.ws_connect(url + QUERY, timeout=aiohttp.ClientWSTimeout(ws_close=5))
        first = json.loads((await ws.receive(timeout=5)).data)
        if first.get("type") == "session.created":
            return ws, first
        await ws.close()
        if first.get("code") != "session_full":
            raise RuntimeError(f"asr: {first}")
        await asyncio.sleep(1.0)
    raise Busy("ASR bridge session_full: close the Speakrail browser tab and retry")


async def stream_pcm(ws, x: np.ndarray, speed: float):
    """Send x in 80 ms chunks on a wall clock running `speed` times realtime."""
    t0 = time.monotonic()
    for i, k in enumerate(range(0, len(x), CHUNK)):
        await ws.send_bytes(pcm16(x[k:k + CHUNK]))
        lag = t0 + (i + 1) * CHUNK / SR / speed - time.monotonic()
        if lag > 0:
            await asyncio.sleep(lag)


async def transcribe(x: np.ndarray, speed: float = 1.0, tail_ms: int = 400, timeout_s: float = 10.0,
                     url: str = ASR_URL) -> dict:
    """-> text, words [{text,start,end}], turn [(t_ms, p)], classes, restarts. Raises on error or timeout."""
    x = np.concatenate([x, np.zeros(SR * tail_ms // 1000, np.float32)])
    async with aiohttp.ClientSession() as s:
        ws, created = await _open(s, url)
        out = {"text": "", "words": [], "turn": [], "classes": created.get("turn_head", {}).get("classes"),
               "restarts": 0}

        async def reader():
            async for m in ws:
                if m.type != aiohttp.WSMsgType.TEXT:
                    continue
                ev = json.loads(m.data)
                k = ev.get("type")
                if k == "transcript":
                    out["text"] += ev["text"]
                    out["words"] += ev["words"]
                elif k == "turn":
                    out["turn"].append((ev["t"], ev["p"]))
                elif k == "asr_restart":
                    out["restarts"] += 1
                elif k == "error":
                    raise RuntimeError(f"asr {ev.get('code')}: {ev.get('message')}")
                elif k == "session.ended":
                    return
            raise RuntimeError("asr closed the connection without session.ended")    # partial text is not a result

        rtask = asyncio.create_task(reader())
        try:
            await stream_pcm(ws, x, speed)
            await ws.send_str(json.dumps({"type": "input_audio.end"}))
            await asyncio.wait_for(rtask, timeout_s)
        except Exception as e:
            raise ASRError(f"asr: {e!r}"[:300], " ".join(out["text"].split())) from e
        finally:
            rtask.cancel()
            await ws.close()
        out["text"] = " ".join(out["text"].split())
        return out


async def preflight(url: str = ASR_URL) -> dict:
    async with aiohttp.ClientSession() as s:
        ws, created = await asyncio.wait_for(_open(s, url, tries=1), 5)
        await ws.send_str(json.dumps({"type": "input_audio.end"}))
        await ws.close()
        return created
