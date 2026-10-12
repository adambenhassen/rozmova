"""Test-set builders: download FLEURS and smart-turn-data, write 16 kHz wavs under data/ and manifests under sets/.

Manifest rows (sets/<name>.jsonl): id, audio (path relative to eval/), lang, plus ref (read sets) or label (turn sets:
True = complete). Sources, verified 2026-10-12:
  FLEURS     https://huggingface.co/datasets/google/fleurs  data/<cfg>/test.tsv + data/<cfg>/audio/test.tar.gz (CC BY 4.0)
  smart-turn https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test  data/*.parquet (CC BY 4.0)"""
from __future__ import annotations

import hashlib
import json
import os
import random
import tarfile

import requests

import audio

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SETS = os.path.join(HERE, "sets")
HF = "https://huggingface.co"
FLEURS = f"{HF}/datasets/google/fleurs/resolve/main/data"
TURN_REPO = "pipecat-ai/smart-turn-data-v3.2-test"
SEED = 20261012
N_MIXED = 300


def download(url: str, dest: str) -> str:
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[sets] get {url}", flush=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest + ".part", "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(dest + ".part", dest)
    return dest


def parse_fleurs_tsv(text: str, lang: str) -> list[dict]:
    """7 tab-separated columns, no header: sentence id (not unique), wav name, raw text, normalized text,
    spelled text, num_samples, gender."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        c = line.split("\t")
        rows.append({"id": c[1].removesuffix(".wav"), "wav": c[1], "ref": c[2], "lang": lang,
                     "gender": c[6].strip().lower(), "seconds": int(c[5]) / 16000})
    return rows


def write_manifest(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_manifest(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _rel(p: str) -> str:
    return os.path.relpath(p, HERE)


def build_fleurs(cfg: str, lang: str) -> list[dict]:
    base = os.path.join(DATA, "fleurs", cfg)
    tsv = download(f"{FLEURS}/{cfg}/test.tsv", os.path.join(base, "test.tsv"))
    rows = parse_fleurs_tsv(open(tsv, encoding="utf-8").read(), lang)
    wav_dir = os.path.join(base, "wav")
    if not all(os.path.exists(os.path.join(wav_dir, r["wav"])) for r in rows):
        tgz = download(f"{FLEURS}/{cfg}/audio/test.tar.gz", os.path.join(base, "test.tar.gz"))
        os.makedirs(wav_dir, exist_ok=True)
        want = {r["wav"] for r in rows}
        with tarfile.open(tgz) as t:
            for m in t:
                name = os.path.basename(m.name)
                if m.isfile() and name in want:
                    with t.extractfile(m) as src:
                        audio.write(os.path.join(wav_dir, name), audio.read_bytes(src.read()))
    out = [{k: v for k, v in r.items() if k != "wav"} | {"audio": _rel(os.path.join(wav_dir, r["wav"]))}
           for r in rows]
    missing = [r["id"] for r in out if not os.path.exists(os.path.join(HERE, r["audio"]))]
    if missing:
        raise RuntimeError(f"FLEURS {cfg}: {len(missing)} wavs missing from the tar, e.g. {missing[:3]}")
    return out


def build_mixed(uk: list[dict], en: list[dict]) -> list[dict]:
    d = os.path.join(DATA, "mixed")
    os.makedirs(d, exist_ok=True)
    rows = []
    for i, p in enumerate(audio.make_pairs(uk, en, N_MIXED, SEED)):
        a, b = p["a"], p["b"]
        key = hashlib.sha1(f"{a['id']}|{b['id']}|{p['gap_ms']}|{audio.TARGET_DBFS}|{audio.PEAK_MAX}".encode()).hexdigest()[:10]
        path = os.path.join(d, f"mix{i:03d}-{key}.wav")      # keyed by its inputs: a changed pairing never reuses audio
        if not os.path.exists(path):
            x = audio.splice(audio.read(os.path.join(HERE, a["audio"])), audio.read(os.path.join(HERE, b["audio"])),
                             p["gap_ms"])
            audio.write(path, x)
        rows.append({"id": f"mix{i:03d}", "audio": _rel(path), "ref": f"{a['ref']} {b['ref']}", "lang": "mixed",
                     "order": f"{a['lang']}+{b['lang']}", "gender": a["gender"], "gap_ms": p["gap_ms"],
                     "parts": [a["id"], b["id"]]})
    return rows


def build_turn() -> dict[str, list[dict]]:
    import pyarrow.parquet as pq
    files = requests.get(f"{HF}/api/datasets/{TURN_REPO}/tree/main/data", timeout=30).json()
    shards = sorted(f["path"] for f in files if f["path"].endswith(".parquet"))
    local = [download(f"{HF}/datasets/{TURN_REPO}/resolve/main/{s}", os.path.join(DATA, "smart_turn", os.path.basename(s)))
             for s in shards]
    # pass 1: light columns only, to pick rows; pass 2: decode audio of the picked rows
    light = []
    for si, path in enumerate(local):
        t = pq.read_table(path, columns=["id", "language", "endpoint_bool"]).to_pylist()
        light += [(si, ri, r) for ri, r in enumerate(t) if r["language"] in ("ukr", "eng")]
    uk = [x for x in light if x[2]["language"] == "ukr"]
    en = [x for x in light if x[2]["language"] == "eng"]
    random.Random(SEED).shuffle(en)
    picked = {"uk_turn": uk, "en_turn": en[:len(uk)]}
    out = {}
    d = os.path.join(DATA, "turn")
    os.makedirs(d, exist_ok=True)
    by_shard: dict[int, list] = {}
    for name, xs in picked.items():
        for si, ri, r in xs:
            by_shard.setdefault(si, []).append((name, ri, r))
        out[name] = []
    for si, xs in sorted(by_shard.items()):
        col = pq.read_table(local[si], columns=["audio"]).column("audio")
        for name, ri, r in xs:
            path = os.path.join(d, f"{r['id']}.wav")
            if not os.path.exists(path):
                audio.write(path, audio.read_bytes(col[ri].as_py()["bytes"]))
            out[name].append({"id": r["id"], "audio": _rel(path), "lang": "uk" if name == "uk_turn" else "en",
                              "label": bool(r["endpoint_bool"])})
    return out


def build_all():
    uk = build_fleurs("uk_ua", "uk")
    en = build_fleurs("en_us", "en")
    write_manifest(os.path.join(SETS, "uk_read.jsonl"), uk)
    write_manifest(os.path.join(SETS, "en_read.jsonl"), en)
    write_manifest(os.path.join(SETS, "mixed_splice.jsonl"), build_mixed(uk, en))
    for name, rows in build_turn().items():
        write_manifest(os.path.join(SETS, f"{name}.jsonl"), rows)
    for f in sorted(os.listdir(SETS)):
        print(f"[sets] {f}: {len(read_manifest(os.path.join(SETS, f)))} rows", flush=True)
