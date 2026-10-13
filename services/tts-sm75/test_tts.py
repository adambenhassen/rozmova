# Smoke test for Breeze on Turing (BREEZE_DTYPE=float32) against a running TTS: non-empty, finite, faster than real
# time. Run on the compose network, e.g. inside the app container:
#   docker cp services/tts-sm75/test_tts.py speakrail-app-1:/tmp/ && docker exec speakrail-app-1 python /tmp/test_tts.py
import os, sys, time
import numpy as np, requests

URL = os.environ.get("TTS_URL", "http://tts:7860/v1/audio/speech")
VOICE = os.environ.get("VOICE", "female_a.wav")
VOICE = VOICE if os.path.isabs(VOICE) else os.path.join("/app/voices", VOICE)     # the app's VOICE is a file name
SR = 24000
ref = open(VOICE, "rb").read(); ref_text = open(VOICE.rsplit(".", 1)[0] + ".txt").read().strip()
failed = 0
for text in ("Sure, I set a timer for ten minutes.",
             "It's currently seventeen degrees and overcast in London, with light drizzle expected later this evening.",
             "That's a great summary of the two-pool model of genetic variation, hidden and expressed."):
    t0 = time.monotonic()
    r = requests.post(URL, data={"text": text, "ref_text": ref_text, "seed": "42", "cfg_scale": "1.0"},
                      files={"ref_audio": ("ref.wav", ref, "audio/wav")}, timeout=120)
    pcm = np.frombuffer(r.content[:len(r.content) // 2 * 2], "<i2").astype(np.float32) / 32768
    dur, wall = len(pcm) / SR, time.monotonic() - t0
    ok = r.status_code == 200 and dur > 0.5 and np.isfinite(pcm).all() and float(np.abs(pcm).max()) > 0.01 and wall / dur < 1.0
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} status={r.status_code} audio={dur:.1f}s RTF={wall / max(dur, 1e-9):.2f} {text[:40]!r}")
print(f"{failed} failed")
sys.exit(1 if failed else 0)
