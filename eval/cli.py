#!/usr/bin/env python3
"""Phase 1 Ukrainian eval: build the test sets, run them against a loaded Speakrail stack, summarize, decide Gate A.

    python cli.py build-sets
    python cli.py run --sets uk_read,en_read [--limit N] [--speed 1.0] [--e2e]
    python cli.py tts --adapter breeze [--limit 50]
    python cli.py summary

Runs inside the eval container on the compose network (see run.sh). Results: out/<date>-<set>[-e2e].json."""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import signal
import sys
import time


import audio
import metrics as M
import sets as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FAIL_MAX = 0.02
TURN_SETS = {"uk_turn", "en_turn"}
SET_LANG = {"uk_read": "uk", "en_read": "en", "mixed_splice": None}
GATE_WER_RATIO = 1.25


def today() -> str:
    return time.strftime("%Y-%m-%d")


def save(name: str, doc: dict) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{today()}-{name}.json")
    with open(path, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return path


def russian_audit(rows: list[dict], e2e: bool) -> list[dict]:
    """Every Russian text the user would see or hear, from every row that has output, valid run or not."""
    out = []
    for r in rows:
        texts = r.get("texts", []) if e2e else ([{"kind": "transcript", "text": r["hyp"]}] if r.get("hyp") else [])
        out += [{"id": r["id"], "kind": t["kind"], "text": t["text"][:160]} for t in texts if M.contains_russian(t["text"])]
    return out


def aggregate(name: str, rows: list[dict], e2e: bool) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    agg = {"n": len(rows), "failed": len(rows) - len(ok),
           # zero-Russian requirement: audited on all output, every language, before any validity cut
           "russian": russian_audit(rows, e2e), "russian_detector": "letters+words+lid" if M.lid_available() else
           "letters+words only (language ID missing: Russian may be undercounted)"}
    if name in TURN_SETS:
        unscored = [r for r in ok if r.get("score") is None]
        agg["failed"] += len(unscored)                     # no head frames after the clip end counts as a failure
        agg["unscored"] = len(unscored)
        ok = [r for r in ok if r.get("score") is not None]
    agg["invalid"] = bool(rows) and agg["failed"] / len(rows) > FAIL_MAX
    if agg["invalid"] or not ok:
        return agg
    if e2e:
        want, replied = SET_LANG.get(name), [r for r in ok if r.get("reply")]
        agg["no_reply"] = len(ok) - len(replied)
        agg["no_audio"] = sum(bool(r.get("no_audio")) for r in replied)      # reply text that never became sound
        agg["early"] = sum(bool(r.get("early")) for r in replied)            # reply started before the clip ended
        agg["unfinished"] = sum(bool(r.get("unfinished")) for r in replied)
        langs = [M.reply_language(r["reply"]) for r in replied]
        agg["reply_langs"] = {k or "unclear": langs.count(k) for k in set(langs)}
        agg["reply_lang_match"] = langs.count(want) / len(replied) if want and replied else None
        for k in ("word_end_to_endpoint", "word_end_to_first_pcm"):
            agg[k] = M.percentiles([r["chain"].get(k) for r in replied if not r.get("no_audio")])
        agg["premature"] = sum(bool(r["chain"].get("premature")) for r in ok)
        agg["decisions"] = {d: sum(r["decisions"].count(d) for r in ok) for d in {d for r in ok for d in r["decisions"]}}
    elif name in TURN_SETS:
        res = M.binary_turn([r["label"] for r in ok], [r["score"] > 0.5 for r in ok])
        res["n_scored"] = res.pop("n")
        agg |= res
    else:
        agg["wer"] = M.corpus_wer([(r["ref"], r["hyp"]) for r in ok])
        agg["cer"] = M.corpus_cer([(r["ref"], r["hyp"]) for r in ok])
        agg["russian_rate"] = M.russian_rate([w for r in ok for w in M.normalize(r["hyp"]).split()])
        agg["restarts"] = sum(r.get("restarts", 0) for r in ok)
    return agg


async def run_row(name: str, row: dict, args) -> dict:
    import app_client, asr_client
    rec = {k: row[k] for k in ("id", "ref", "label", "lang") if k in row}
    try:
        x = audio.read(os.path.join(HERE, row["audio"]))
        rec["seconds"] = round(len(x) / audio.SR, 2)
        if args.e2e:
            r = await app_client.converse(x)
            rec |= {"status": "ok", "reply": r["reply"], "no_audio": r["no_audio"], "early": r["early"],
                    "unfinished": r["unfinished"], "chain": r["first"], "decisions": r["decisions"],
                    "n_turns": len(r["turns"]), "texts": r["texts"]}
        else:
            r = await asr_client.transcribe(x, speed=args.speed)
            rec |= {"status": "ok", "hyp": r["text"], "restarts": r["restarts"]}
            if name in TURN_SETS:
                rec["score"] = M.turn_score(r["turn"], r["classes"], end_ms=len(x) / audio.SR * 1000)
    except asyncio.TimeoutError:
        rec |= {"status": "failed", "reason": "timeout"}
    except asr_client.Busy:
        raise
    except Exception as e:                                  # one bad clip must not end the run
        rec |= {"status": "failed", "reason": repr(e)[:300]}
    return rec


async def preflight(e2e: bool):
    import app_client, asr_client
    try:
        created = await asr_client.preflight()
    except asr_client.Busy as e:
        sys.exit(str(e))
    except Exception as e:
        sys.exit(f"ASR bridge unreachable ({e!r}): load speakrail in llama-swap first")
    classes = created.get("turn_head", {}).get("classes") or []
    if not {"complete", "incomplete"} <= set(classes):
        sys.exit(f"unexpected turn head classes {classes}")
    if e2e:
        import aiohttp
        try:
            async with aiohttp.ClientSession() as s:
                ws = await asyncio.wait_for(app_client._open(s, app_client.APP_URL, tries=2), 20)
                await ws.close()
        except Exception as e:
            sys.exit(f"app /ws unreachable ({e!r}): load speakrail in llama-swap first")
    print(f"[run] preflight ok, head classes {classes}", flush=True)


async def cmd_run(args):
    await preflight(args.e2e)
    for name in args.sets.split(","):
        rows = S.read_manifest(os.path.join(S.SETS, f"{name}.jsonl"))[:args.limit or None]
        # a limited run never shadows a full one: summary prefers the result with the most rows
        tag = name + ("-e2e" if args.e2e else "") + (f"-limit{args.limit}" if args.limit else "") + args.tag
        done: list[dict] = []
        stop = False

        def on_int(*_):
            nonlocal stop
            stop = True
        signal.signal(signal.SIGINT, on_int)
        t0 = time.time()
        for i, row in enumerate(rows):
            if stop:
                break
            done.append(await run_row(name, row, args))
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                nf = sum(r["status"] != "ok" for r in done)
                print(f"[run] {tag}: {i + 1}/{len(rows)} ({nf} failed, {time.time() - t0:.0f}s)", flush=True)
        agg = aggregate(name, done, args.e2e)
        path = save(tag, {"set": name, "e2e": args.e2e, "speed": args.speed, "aggregate": agg, "rows": done,
                          "partial": stop})
        print(f"[run] {tag}: {json.dumps(agg, ensure_ascii=False)}\n[run] wrote {path}", flush=True)
        if stop:
            sys.exit(130)


def latest(name: str, suffix: str = "") -> dict | None:
    """The result for `name` + suffix with the most rows (a full run beats a limited one), newest on a tie; custom
    tags (smoke, speedcheck, probe) are ignored. Re-scored from its rows, so scoring fixes apply to old runs."""
    best = None
    for f in sorted(glob.glob(os.path.join(OUT, f"*-{name}{suffix}.json")) +
                    glob.glob(os.path.join(OUT, f"*-{name}{suffix}-limit*.json"))):
        d = json.load(open(f))
        if d.get("partial"):
            continue
        if best is None or len(d.get("rows", [])) >= len(best.get("rows", [])):
            best = d | {"file": os.path.basename(f)}
    if best and "set" in best and "rows" in best:
        best["aggregate"] = aggregate(best["set"], best["rows"], best.get("e2e", False))
    return best


def gate_a(vox: dict | None, para: dict | None, n_set: int) -> tuple[str, str]:
    """Voxtral vs Parakeet in streaming mode (same lookahead; offline Parakeet would tilt the gate to a swap).
    Zero Russian is a hard condition. Incomplete or mismatched evidence is never decided."""
    for label, d in (("Voxtral", vox), ("streaming Parakeet", para)):
        if not d:
            return "undecided", f"no uk_read result for {label}"
        a = d["aggregate"]
        if a.get("invalid") or "wer" not in a:
            return "undecided", f"{label} uk_read result is invalid ({a.get('failed')} of {a.get('n')} failed)"
        if len(d["rows"]) != n_set:
            return "undecided", f"{label} covered {len(d['rows'])} of {n_set} uk_read clips"
    if {r["id"] for r in vox["rows"]} != {r["id"] for r in para["rows"]}:
        return "undecided", "Voxtral and Parakeet results cover different clips"
    v, p = vox["aggregate"], para["aggregate"]
    ru = len({x["id"] for x in v["russian"]})
    if ru:
        return ("FAIL: Voxtral writes Russian; fine-tune or replace it (Phase 2)",
                f"{ru} of {v['n']} Ukrainian clips transcribed with Russian (must be 0); WER {v['wer']:.1%}")
    ratio = v["wer"] / p["wer"] if p["wer"] else float("inf")
    why = f"no Russian; Voxtral WER {v['wer']:.1%} = {ratio:.2f}x streaming Parakeet's {p['wer']:.1%} (limit {GATE_WER_RATIO}x)"
    return ("PASS: keep Voxtral, Phase 2a" if ratio <= GATE_WER_RATIO else "FAIL: swap ASR, Phase 2b"), why


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.1%}"
    return str(v)


def russian_lines(label: str, a: dict) -> list[str]:
    ru = a.get("russian")
    if ru is None:
        return []
    out = [f"- **{label}: {len({x['id'] for x in ru})} clips with Russian output** ({a['russian_detector']})"]
    out += [f"  - `{x['id']}` {x['kind']}: {x['text']}" for x in ru[:10]]
    return out + ([f"  - ... {len(ru) - 10} more in the result file"] if len(ru) > 10 else [])


def cmd_summary(_args):
    lines = [f"# Ukrainian eval summary, {today()}", ""]
    lines += ["## ASR", "", "| set | n | failed | WER | CER | Russian-spelling (words) | restarts | file |",
              "|---|---|---|---|---|---|---|---|"]
    ru_lines: list[str] = []
    for name in ("uk_read", "mixed_splice", "en_read"):
        for suffix, label in (("", "Voxtral"), ("-parakeet-stream", "Parakeet, streaming 0.48 s"),
                              ("-parakeet", "Parakeet, offline upper bound")):
            d = latest(name, suffix)
            if not d:
                continue
            a = d["aggregate"]
            if suffix == "":
                label += f", streamed at {d.get('speed', 1.0):g}x"
            lines.append(f"| {name} ({label}) | {a['n']} | {a['failed']} | " + (
                "invalid |  |  |  |" if a.get("invalid") else
                f"{fmt(a.get('wer'))} | {fmt(a.get('cer'))} | {fmt(a.get('russian_rate'))} | {fmt(a.get('restarts'))} |")
                + f" {d['file']} |")
            ru_lines += russian_lines(f"{name} ({label})", a)
    lines += ["", "Spliced uk/en results are an optimistic upper bound (switches fall between speakers).", ""]
    lines += ["## Turn head (complete vs incomplete, clip end + 400 ms)", "",
              "| set | n | failed (incl. unscored) | accuracy | macro F1 |", "|---|---|---|---|---|"]
    for name in ("uk_turn", "en_turn"):
        d = latest(name)
        if d:
            a = d["aggregate"]
            lines.append(f"| {name} | {a['n']} | {a['failed']} ({a.get('unscored', 0)}) | " + (
                "invalid |  |" if a.get("invalid") else f"{fmt(a.get('accuracy'))} | {fmt(a.get('macro_f1'))} |"))
    lines += ["", "## End to end (uk_dialog = uk_read clips as single turns)", "",
              "| set | n | failed | no reply | text, no audio | early | reply languages | language match | "
              "endpoint p50/p90 ms | first audio p50/p90 ms | premature |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("uk_read", "en_read"):
        d = latest(name, "-e2e")
        if d:
            a = d["aggregate"]
            label = "uk_dialog" if name == "uk_read" else "en_dialog"
            ep, fa = a.get("word_end_to_endpoint") or {}, a.get("word_end_to_first_pcm") or {}
            lines.append(f"| {label} | {a['n']} | {a['failed']} | " + (
                "invalid |  |  |  |  |  |  |  |" if a.get("invalid") else
                f"{a.get('no_reply')} | {a.get('no_audio')} | {a.get('early')} | {a.get('reply_langs')} | "
                f"{fmt(a.get('reply_lang_match'))} | {ep.get('p50')}/{ep.get('p90')} | {fa.get('p50')}/{fa.get('p90')} | "
                f"{a.get('premature')} |"))
            ru_lines += russian_lines(f"{label} (replies, user transcript, interjections, tool cards)", a)
    tts = latest_tts()
    if tts:
        lines += ["", "## TTS", "", "| adapter | n | failed | TTFA p50/p90 ms | RTF p50 | round-trip CER (Voxtral) | "
                  "round-trip CER (Parakeet offline) |", "|---|---|---|---|---|---|---|"]
        for a in tts:
            lines.append(f"| {a['adapter']} | {a['n']} | {a['failed']} | " + (
                "invalid |  |  |  |" if a.get("invalid") else
                f"{a['ttfa_ms']['p50']}/{a['ttfa_ms']['p90']} | {a['rtf']['p50']:.2f} | "
                f"{fmt(a.get('roundtrip_cer'))} | {fmt(a.get('roundtrip_cer_parakeet'))} |"))
    lines += ["", "## Zero-Russian audit (must be 0 everywhere to ship)", ""] + (ru_lines or ["- no results"])
    n_uk = len(S.read_manifest(os.path.join(S.SETS, "uk_read.jsonl")))
    verdict, why = gate_a(latest("uk_read"), latest("uk_read", "-parakeet-stream"), n_uk)
    lines += ["", "## Gate A", "", f"**{verdict}**: {why}", ""]
    path = os.path.join(OUT, f"{today()}-summary.md")
    os.makedirs(OUT, exist_ok=True)
    open(path, "w").write("\n".join(lines))
    print("\n".join(lines))


def latest_tts() -> list[dict]:
    """Newest TTS bench per adapter, re-scored from its rows, plus the Parakeet round trip when it has been run."""
    import tts_bench
    by_adapter: dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(OUT, "*-tts-*.json"))):
        by_adapter[json.load(open(f))["aggregate"]["adapter"]] = f
    out = []
    for adapter, f in sorted(by_adapter.items()):
        a = tts_bench.aggregate(adapter, json.load(open(f))["rows"])
        para = latest(f"tts_{adapter}", "-parakeet")
        if para and not para["aggregate"].get("invalid"):
            a["roundtrip_cer_parakeet"] = para["aggregate"].get("cer")
        out.append(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-sets")
    r = sub.add_parser("run")
    r.add_argument("--sets", required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--speed", type=float, default=1.0)
    r.add_argument("--e2e", action="store_true")
    r.add_argument("--tag", default="")
    t = sub.add_parser("tts")
    t.add_argument("--adapter", default="breeze")
    t.add_argument("--limit", type=int, default=50)
    t.add_argument("--speed", type=float, default=1.0)
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "build-sets":
        S.build_all()
    elif a.cmd == "run":
        if a.e2e and a.speed != 1.0:
            sys.exit("--e2e runs in real time only")
        asyncio.run(cmd_run(a))
    elif a.cmd == "tts":
        import tts_bench
        asyncio.run(tts_bench.main(a.adapter, a.limit, a.speed))
    else:
        cmd_summary(a)


if __name__ == "__main__":
    main()
