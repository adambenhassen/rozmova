import numpy as np
import pytest

from audio import SR, make_pairs, rms_dbfs, rms_normalize, splice


def tone(seconds, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_rms_normalize():
    assert rms_dbfs(rms_normalize(tone(1.0, 0.02))) == pytest.approx(-23.0, abs=0.1)


def test_rms_normalize_never_clips():
    x = np.zeros(SR, np.float32)
    x[100] = 0.5                                      # one spike: reaching -23 dBFS RMS would need gain far past 1.0
    y = rms_normalize(x)
    assert float(np.max(np.abs(y))) <= 0.98 + 1e-6


def test_splice_length():
    a, b = tone(1.0), tone(0.5)
    out = splice(a, b, gap_ms=100, fade_ms=10)
    assert len(out) == len(a) + int(SR * 0.1) + len(b)
    assert out.dtype == np.float32


def rows(lang, n):
    return [{"id": f"{lang}{i}", "gender": "female" if i % 2 else "male", "ref": f"{lang} {i}"} for i in range(n)]


def test_make_pairs_reproducible_same_gender():
    uk, en = rows("uk", 20), rows("en", 20)
    p1, p2 = make_pairs(uk, en, 10, seed=7), make_pairs(uk, en, 10, seed=7)
    assert p1 == p2 and len(p1) == 10
    for p in p1:
        assert p["a"]["gender"] == p["b"]["gender"]
        assert {p["a"]["id"][:2], p["b"]["id"][:2]} == {"uk", "en"}
        assert 50 <= p["gap_ms"] <= 150
    assert make_pairs(uk, en, 10, seed=8) != p1
