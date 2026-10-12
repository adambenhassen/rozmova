"""Audio helpers: float32 mono at 16 kHz everywhere inside the eval; int16 PCM only on the wire."""
from __future__ import annotations

import random
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
TARGET_DBFS = -23.0


def read(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return resample(x.mean(axis=1), sr, SR)


def read_bytes(data: bytes) -> np.ndarray:
    import io
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return resample(x.mean(axis=1), sr, SR)


def write(path: str, x: np.ndarray, sr: int = SR):
    sf.write(path, x, sr, subtype="PCM_16")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def rms_dbfs(x: np.ndarray) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))), 1e-9))


PEAK_MAX = 0.98


def rms_normalize(x: np.ndarray, target: float = TARGET_DBFS) -> np.ndarray:
    """Scale to `target` dBFS RMS, but never past PEAK_MAX: a high-crest clip ends up quieter instead of clipped."""
    g = 10 ** ((target - rms_dbfs(x)) / 20)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak * g > PEAK_MAX:
        g = PEAK_MAX / peak
    return (x * g).astype(np.float32)


def splice(a: np.ndarray, b: np.ndarray, gap_ms: int, fade_ms: int = 10) -> np.ndarray:
    """a, a short fade-out into silence, the gap, then b. Both halves loudness-matched first."""
    a, b = rms_normalize(a), rms_normalize(b)
    n = min(int(SR * fade_ms / 1000), len(a))
    if n:
        a = a.copy()
        a[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return np.concatenate([a, np.zeros(int(SR * gap_ms / 1000), np.float32), b]).astype(np.float32)


def make_pairs(uk: list[dict], en: list[dict], n: int, seed: int) -> list[dict]:
    """n same-gender (uk, en) pairs in random order, each with a gap of 50-150 ms. Rows need id, gender."""
    rnd = random.Random(seed)
    en_by_g: dict[str, list[dict]] = {}
    for r in en:
        en_by_g.setdefault(r["gender"], []).append(r)
    pool = [r for r in uk if en_by_g.get(r["gender"])]
    rnd.shuffle(pool)
    out = []
    for u in pool[:n]:
        e = rnd.choice(en_by_g[u["gender"]])
        a, b = (u, e) if rnd.random() < 0.5 else (e, u)
        out.append({"a": a, "b": b, "gap_ms": rnd.randint(50, 150)})
    return out
