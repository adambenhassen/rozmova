#!/usr/bin/env python3
"""Parakeet v3 (nvidia/parakeet-tdt-0.6b-v3, CC BY 4.0) on a read set: the Gate A comparison for Voxtral.

Two modes, both written in the same shape as `cli.py run` so `cli.py summary` scores them alike:
  offline  whole clips at once (out/<date>-<set>-parakeet.json): an upper bound, not what a live swap would get
  stream   NeMo's buffered streaming script (out/<date>-<set>-parakeet-stream.json), 0.32 s chunks + 0.32 s right
           context: 0.32-0.64 s of lookahead, 0.48 s on average, matching Voxtral's fixed 480 ms delay. Gate A uses this.
Runs in its own NeMo container on GPU1 (run.sh parakeet)."""
import json
import os
import subprocess
import sys
import time

import cli
import sets as S


MODEL = "nvidia/parakeet-tdt-0.6b-v3"
STREAM_SCRIPT = "/nemo/speech_to_text_streaming_infer_rnnt.py"     # NeMo v2.7.3 examples/asr/asr_chunked_inference/rnnt/
CHUNK_S = float(os.environ.get("PARAKEET_CHUNK_S", "0.32"))
RIGHT_S = float(os.environ.get("PARAKEET_RIGHT_S", "0.32"))
LEFT_S = 10.0
BATCH = int(os.environ.get("PARAKEET_BATCH", "16"))
LIMIT = int(os.environ.get("PARAKEET_LIMIT", "0"))        # probe runs: first N clips
TAG = os.environ.get("PARAKEET_TAG", "")                 # probe runs: kept apart from the Gate A result


def offline(rows: list[dict]) -> list[str]:
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL).eval()
    return [h.text for h in m.transcribe([os.path.join(cli.HERE, r["audio"]) for r in rows], batch_size=16)]


def stream(rows: list[dict], name: str) -> list[str]:
    work = os.path.join(cli.OUT, f"parakeet-{name}{TAG}")
    os.makedirs(work, exist_ok=True)
    src, dst = os.path.join(work, "in.jsonl"), os.path.join(work, "out.jsonl")
    with open(src, "w") as f:
        for r in rows:
            f.write(json.dumps({"audio_filepath": os.path.join(cli.HERE, r["audio"]), "duration": r.get("seconds", 0),
                                "text": r["ref"]}, ensure_ascii=False) + "\n")
    subprocess.run([sys.executable, STREAM_SCRIPT, f"pretrained_name={MODEL}", f"dataset_manifest={src}",
                    f"output_filename={dst}", f"chunk_secs={CHUNK_S}", f"right_context_secs={RIGHT_S}",
                    f"left_context_secs={LEFT_S}", f"batch_size={BATCH}"], check=True)
    by_path = {}
    for line in open(dst):
        d = json.loads(line)
        by_path[d["audio_filepath"]] = d.get("pred_text", "")
    return [by_path[os.path.join(cli.HERE, r["audio"])] for r in rows]


def main(name: str, mode: str):
    rows = S.read_manifest(os.path.join(S.SETS, f"{name}.jsonl"))[:LIMIT or None]
    t0 = time.time()
    hyps = offline(rows) if mode == "offline" else stream(rows, name)
    done = [{"id": r["id"], "ref": r["ref"], "lang": r["lang"], "status": "ok", "hyp": h} for r, h in zip(rows, hyps)]
    agg = cli.aggregate(name, done, e2e=False)
    tag = f"{name}-parakeet" + ("-stream" if mode == "stream" else "") + TAG
    path = cli.save(tag, {"set": name, "model": MODEL, "mode": mode, "chunk_s": CHUNK_S, "right_s": RIGHT_S, "batch": BATCH,
                          "aggregate": agg, "rows": done, "seconds": round(time.time() - t0)})
    print(f"[parakeet] {tag}: {json.dumps(agg, ensure_ascii=False)}\n[parakeet] wrote {path}", flush=True)


if __name__ == "__main__":
    mode, names = sys.argv[1], sys.argv[2:] or ["uk_read"]       # parakeet.py offline|stream [set...]
    for n in names:
        main(n, mode)
