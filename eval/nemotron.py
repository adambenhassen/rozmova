#!/usr/bin/env python3
"""Nemotron 3.5 ASR streaming (nvidia/nemotron-3.5-asr-streaming-0.6b, OpenMDW-1.1) on a read set: a Phase 2
candidate that streams natively (cache-aware, no re-computation) and lists uk-UA.

    python nemotron.py <lang> <left,right> [set...]     e.g.  uk-UA 56,6 uk_read   |   auto 56,3 mixed_splice

`lang` is the language-ID prompt (a locale, or `auto` to detect); left/right are attention context in 80 ms frames,
chunk = right + 1 frames: 56,6 = 0.56 s (about Voxtral's 480 ms delay), 56,3 = 0.32 s. Writes
out/<date>-<set>-nemotron-<lang>-l<left>-c<chunk ms>.json in the same shape as `cli.py run`."""
import json
import os
import subprocess
import sys
import time

import cli
import sets as S

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
SCRIPT = "/nemo/cache_aware_streaming_infer.py"


def main(name: str, lang: str, ctx: str):
    left, right = (int(x) for x in ctx.split(","))
    chunk_ms = (right + 1) * 80
    rows = S.read_manifest(os.path.join(S.SETS, f"{name}.jsonl"))
    tag = f"{name}-nemotron-{lang}-l{left}-c{chunk_ms}"
    work = os.path.join(cli.OUT, tag)
    os.makedirs(work, exist_ok=True)
    src = os.path.join(work, "in.jsonl")
    with open(src, "w") as f:
        for r in rows:
            f.write(json.dumps({"audio_filepath": os.path.join(cli.HERE, r["audio"]), "duration": r.get("seconds", 0),
                                "text": r["ref"]}, ensure_ascii=False) + "\n")
    t0 = time.time()
    subprocess.run([sys.executable, SCRIPT, f"pretrained_name={MODEL}", f"dataset_manifest={src}", f"output_path={work}",
                    f"target_lang={lang}", f"att_context_size=[{left},{right}]", "strip_lang_tags=true", "batch_size=32"],
                   check=True)
    outs = [os.path.join(work, x) for x in os.listdir(work) if x.startswith("streaming_out_")]
    if len(outs) != 1:
        raise RuntimeError(f"expected one streaming_out_*.json in {work}, found {outs}")
    hyps = [json.loads(l)["pred_text"] for l in open(outs[0])]
    if len(hyps) != len(rows):                         # the script writes one line per manifest line, in order
        raise RuntimeError(f"{len(hyps)} hypotheses for {len(rows)} clips")
    done = [{"id": r["id"], "ref": r["ref"], "lang": r["lang"], "status": "ok", "hyp": h} for r, h in zip(rows, hyps)]
    agg = cli.aggregate(name, done, e2e=False)
    path = cli.save(tag, {"set": name, "model": MODEL, "lang": lang, "att_context_size": [left, right],
                          "chunk_ms": chunk_ms, "aggregate": agg, "rows": done, "seconds": round(time.time() - t0)})
    print(f"[nemotron] {tag}: wer={agg.get('wer')} russian={len(agg['russian'])}\n[nemotron] wrote {path}", flush=True)


if __name__ == "__main__":
    lang, ctx, names = sys.argv[1], sys.argv[2], sys.argv[3:] or ["uk_read"]
    for n in names:
        main(n, lang, ctx)
