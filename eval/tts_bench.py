"""TTS bench: time to first audio, real-time factor, round-trip CER (synthesize uk_read references, transcribe them
with the ASR bridge) and a 20-sentence listening sheet for the operator. Every clip is also written with a manifest,
sets/tts_<adapter>.jsonl, so `run.sh parakeet offline tts_<adapter>` scores the round trip with an ASR that does not
turn Ukrainian into Russian (Voxtral does, which would make good Ukrainian speech look bad).

An adapter is `async def synth(text) -> AsyncIterator[bytes]` of int16 PCM at adapter.SR. Phase 3 adds candidates;
`breeze` is the deployed Breeze TTS 2 (English and Chinese only), the baseline any candidate must beat."""
from __future__ import annotations

import os
import time

import aiohttp
import numpy as np

import asr_client
import audio
import metrics as M
import sets as S

N_SHEET = 20


class Breeze:
    """The app's TTS endpoint: multipart text + reference voice, streamed raw 24 kHz int16 PCM back."""
    SR = 24000

    def __init__(self, url=os.environ.get("TTS_URL", "http://tts:7860/v1/audio/speech"),
                 voice=os.environ.get("VOICE", "/app/voices/female_a.wav")):
        self.url, self.ref = url, open(voice, "rb").read()
        self.ref_text = open(voice.rsplit(".", 1)[0] + ".txt").read().strip()

    async def synth(self, s: aiohttp.ClientSession, text: str):
        form = aiohttp.FormData()
        for k, v in {"text": text, "ref_text": self.ref_text, "seed": "42", "cfg_scale": "1.0"}.items():
            form.add_field(k, v)
        form.add_field("ref_audio", self.ref, filename="ref.wav", content_type="audio/wav")
        async with s.post(self.url, data=form, timeout=aiohttp.ClientTimeout(total=120)) as r:
            r.raise_for_status()
            async for chunk in r.content.iter_any():
                yield chunk


ADAPTERS = {"breeze": Breeze}


def aggregate(adapter: str, rows: list[dict]) -> dict:
    import cli
    ok = [r for r in rows if r["status"] == "ok"]
    agg = {"adapter": adapter, "n": len(rows), "failed": len(rows) - len(ok),
           "invalid": bool(rows) and (len(rows) - len(ok)) / len(rows) > cli.FAIL_MAX}
    if agg["invalid"] or not ok:
        return agg | {"invalid": True}
    return agg | {"ttfa_ms": M.percentiles([r["ttfa_ms"] for r in ok]), "rtf": M.percentiles([r["rtf"] for r in ok]),
                  "roundtrip_cer": M.corpus_cer([(r["ref"], r["hyp"]) for r in ok])}


async def main(adapter: str, limit: int, speed: float = 1.0):
    import cli
    tts = ADAPTERS[adapter]()
    rows = S.read_manifest(os.path.join(S.SETS, "uk_read.jsonl"))[:limit]
    sheet_dir = os.path.join(cli.OUT, f"{cli.today()}-tts-{adapter}")
    os.makedirs(sheet_dir, exist_ok=True)
    out = []
    async with aiohttp.ClientSession() as s:
        for i, row in enumerate(rows):
            rec = {"id": row["id"], "ref": row["ref"]}
            try:
                t0, t_first, buf = time.monotonic(), None, bytearray()
                async for chunk in tts.synth(s, row["ref"]):
                    if t_first is None and chunk:
                        t_first = time.monotonic()
                    buf += chunk
                total = time.monotonic() - t0
                x = np.frombuffer(bytes(buf[:len(buf) // 2 * 2]), dtype="<i2").astype(np.float32) / 32768
                dur = len(x) / tts.SR
                if not dur:
                    raise RuntimeError("no audio")
                y = audio.resample(x, tts.SR, audio.SR)
                wav = os.path.join(sheet_dir, f"{i:03d}.wav")
                audio.write(wav, y)
                rec["audio"] = os.path.relpath(wav, cli.HERE)
                hyp = (await asr_client.transcribe(y, speed=speed))["text"]
                rec |= {"status": "ok", "ttfa_ms": round((t_first - t0) * 1000), "rtf": total / dur, "hyp": hyp,
                        "seconds": round(dur, 2)}
            except Exception as e:                          # one bad sentence must not end the bench
                rec |= {"status": "failed", "reason": repr(e)[:300]}
            out.append(rec)
            print(f"[tts] {adapter} {i + 1}/{len(rows)} {rec['status']}", flush=True)
    agg = aggregate(adapter, out)
    S.write_manifest(os.path.join(S.SETS, f"tts_{adapter}.jsonl"),
                     [{"id": r["id"], "audio": r["audio"], "ref": r["ref"], "lang": "uk"} for r in out if r.get("audio")])
    with open(os.path.join(sheet_dir, "sheet.md"), "w") as f:
        f.write(f"# Listening sheet: {adapter}\n\nScore 1-5. Naturalness: does it sound like a person? "
                "Accent: does it sound like a native Ukrainian speaker?\n\n| # | text | naturalness | accent |\n|---|---|---|---|\n")
        for i, r in enumerate(out[:N_SHEET]):
            f.write(f"| {i:03d} | {r['ref']} |  |  |\n")
    path = cli.save(f"tts-{adapter}", {"aggregate": agg, "rows": out})
    print(f"[tts] {agg}\n[tts] wrote {path} and {sheet_dir}/", flush=True)
