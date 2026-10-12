"""Scoring for the Phase 1 eval: text normalization, WER/CER, the Russian-spelling rate, reply script, turn metrics.

WER and CER are micro-averaged: total edits over total reference units across the set, so long utterances weigh more."""
from __future__ import annotations

import os
import re
import unicodedata
from decimal import Decimal

from num2words import num2words

APOSTROPHES = "'’‘`ʹ"                      # all folded to U+02BC (Ukrainian apostrophe)
RU_ONLY = set("ыэъё")                       # letters Russian has and Ukrainian doesn't
UK_ONLY = set("іїєґ")                       # and the reverse
# frequent words found in only one of the two (short replies often have none of the letters above)
RU_WORDS = {"что", "это", "как", "мы", "вы", "да", "нет", "или", "если", "меня", "мне", "тебя", "совсем", "понял", "сейчас",
            "здесь", "только", "конечно", "хорошо", "его", "она", "они", "можно", "пожалуйста",
            # short conversational replies and backchannels (Ukrainian in brackets), caught without language ID:
            "понятно", "спасибо", "привет", "посмотрю", "слушаю", "согласен", "согласна", "поняла", "извините",  # зрозуміло дякую привіт подивлюсь слухаю згоден згодна зрозуміла вибачте
            "ничего", "почему", "где", "когда", "кто", "сколько", "сегодня", "скажите",                          # нічого чому де коли хто скільки сьогодні скажіть
            "минутку", "подождите", "отлично", "здравствуйте"}                                                 # хвилинку зачекайте чудово вітаю
# never add a word Ukrainian also uses ("говорите", "правильно", "ладно"): a marker flags the whole line
UK_WORDS = {"що", "це", "як", "ми", "ви", "так", "ні", "або", "якщо", "мене", "тебе", "зовсім", "зрозумів", "зараз",
            "тут", "тільки", "звичайно", "добре", "його", "вона", "вони", "можна", "будь", "ласка"}
PERCENT = {"en": "percent", "uk": "відсотків"}
_NONWORD = re.compile(r"[^\w\sʼ]|_")
# 1,000 / 3.5 / 100% / 20th: FLEURS references spell numbers out, ASR writes digits
_NUMBER = re.compile(r"(?<!\w)((?<![\w-])-)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)(\s?%|st|nd|rd|th)?")


def _spell_numbers(t: str) -> str:
    out, pos = [], 0
    for m in _NUMBER.finditer(t):
        before = [w for w in re.findall(r"[^\W\d_]+", t[:m.start()])]
        lang = "uk" if majority_script(before[-1] if before else t) == "cyrl" else "en"
        sign, num, suf = m.group(1), m.group(2), (m.group(3) or "").strip()      # "-5" is minus; "pre-2010" is not
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", num):
            num = num.replace(",", "")                   # 1,234.56: thousands separators
        val = Decimal(num.replace(",", ".")) if re.search(r"[.,]", num) else int(num)
        if sign:
            val = -val
        if suf in ("st", "nd", "rd", "th") and isinstance(val, int) and val >= 0:
            words = num2words(val, to="ordinal", lang="en")
        else:
            words = num2words(val, lang=lang) + (f" {PERCENT[lang]}" if suf == "%" else "")
        out += [t[pos:m.start()], f" {words} "]
        pos = m.end()
    return "".join(out) + t[pos:]


def normalize(text: str) -> str:
    t = _spell_numbers(unicodedata.normalize("NFC", text).lower()).lower()
    for a in APOSTROPHES:
        t = t.replace(a, "ʼ")
    t = _NONWORD.sub(" ", t.replace("-", " "))
    return " ".join(w.strip("ʼ") for w in t.split() if w.strip("ʼ"))


def edit_distance(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def _corpus(pairs, unit) -> float:
    errs = total = 0
    for ref, hyp in pairs:
        r, h = unit(normalize(ref)), unit(normalize(hyp))
        errs += edit_distance(r, h)
        total += len(r)
    return errs / total if total else 0.0


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    return _corpus(pairs, str.split)


def corpus_cer(pairs: list[tuple[str, str]]) -> float:
    return _corpus(pairs, lambda s: list(s.replace(" ", "")))


def _is_cyrl(ch: str) -> bool:
    return "CYRILLIC" in unicodedata.name(ch, "")


def russian_rate(words: list[str]) -> float | None:
    """Share of Cyrillic words spelled with a Russian-only letter. A lower bound: `привет` passes."""
    cyr = [w for w in words if any(_is_cyrl(c) for c in w)]
    if not cyr:
        return None
    return sum(1 for w in cyr if RU_ONLY & set(w.lower())) / len(cyr)


def majority_script(text: str) -> str | None:
    cyr = sum(1 for c in text if c.isalpha() and _is_cyrl(c))
    lat = sum(1 for c in text if c.isalpha() and "LATIN" in unicodedata.name(c, ""))
    if not cyr and not lat:
        return None
    return "cyrl" if cyr > lat else "latn"


def turn_score(frames: list[tuple[float, list[float]]], classes: list[str], end_ms: float,
               window_ms: float = 400) -> float | None:
    """Mean p(complete) / (p(complete) + p(incomplete)) over head frames stamped in [end_ms, end_ms + window_ms)."""
    ci, ii = classes.index("complete"), classes.index("incomplete")
    vals = [p[ci] / (p[ci] + p[ii]) for t, p in frames if end_ms <= t < end_ms + window_ms and p[ci] + p[ii] > 0]
    return sum(vals) / len(vals) if vals else None


def percentiles(xs: list[float], qs=(50, 90)) -> dict:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": xs[min(len(xs) - 1, round(q / 100 * (len(xs) - 1)))] for q in qs}


def reply_language(text: str) -> str | None:
    """en | uk | ru, or None when unclear. Cyrillic is split by the letters only one of the two languages has."""
    script = majority_script(text)
    if script != "cyrl":
        return "en" if script == "latn" else None
    low = text.lower()
    words = set(re.findall(r"[^\W\d_]+", low))
    uk = len(UK_ONLY & set(low)) + len(UK_WORDS & words)
    ru = len(RU_ONLY & set(low)) + len(RU_WORDS & words)
    return "uk" if uk > ru else "ru" if ru > uk else None


LID_PATH = os.environ.get("LID_MODEL", "/opt/lid.176.ftz")   # fastText language ID (CC BY-SA 3.0), baked into the image
_lid = None


LID_MIN_WORDS = 3           # 1-2 word clauses are mostly names ("Сан Паркс", "Джонсона"): fastText calls them Russian
LID_MIN_P, LID_MARGIN = 0.6, 2.0


def lang_probs(text: str) -> dict[str, float]:
    """fastText's top languages for the line, {"ru": 0.8, "uk": 0.1, ...}; {} when the model isn't installed."""
    global _lid
    if _lid is None:
        if not os.path.exists(LID_PATH):
            return {}
        import fasttext
        _lid = fasttext.load_model(LID_PATH)
    labels, probs = _lid.predict(" ".join(text.split()), k=3)
    return {l.removeprefix("__label__"): float(p) for l, p in zip(labels, probs)}


def lang_id(text: str) -> str | None:
    p = lang_probs(text)
    return max(p, key=p.get) if p else None


def lid_says_russian(span: str) -> bool:
    if len(re.findall(r"[^\W\d_]+", span)) < LID_MIN_WORDS:
        return False
    p = lang_probs(span)
    return p.get("ru", 0) >= LID_MIN_P and p.get("ru", 0) >= LID_MARGIN * p.get("uk", 0)


_SPANS = re.compile(r"[^.!?;:…\n]+")


def _words(text: str) -> set[str]:
    return set(re.findall(r"[^\W\d_]+", text.lower()))


def russian_verdict(text: str) -> str:
    """"ru" (Russian found), "uncertain" (language ID leans Russian on a clause too short or below threshold to
    certify either way: listed for review) or "clean". Russian markers are never outvoted by Ukrainian words."""
    if RU_ONLY & set(text.lower()) or RU_WORDS & _words(text):
        return "ru"
    spans = [sp for sp in (x.strip() for x in _SPANS.findall(text)) if any(c.isalpha() for c in sp)] or [text]
    verdict = "clean"
    for sp in spans:
        if majority_script(sp) != "cyrl":
            continue
        if lid_says_russian(sp):
            return "ru"
        if len(re.findall(r"[^\W\d_]+", sp)) >= 2 and lang_id(sp) == "ru":
            verdict = "uncertain"            # one word ("Мартелли", "стала") tells language ID nothing
    return verdict


def contains_russian(text: str) -> bool:
    """Zero-Russian check, per sentence so a Russian clause inside a Ukrainian or English line is not outvoted: any
    Russian-only letter (also in names: "Буэнос-Айрес" is the Russian spelling), Russian marker words, or confident
    language ID on clauses of LID_MIN_WORDS+ words."""
    return russian_verdict(text) == "ru"


def lid_available() -> bool:
    return os.path.exists(LID_PATH)


def binary_turn(labels: list[bool], preds: list[bool]) -> dict:
    """labels/preds: True = complete. Accuracy, per-class F1 averaged (macro), confusion with complete as positive."""
    tp = sum(l and p for l, p in zip(labels, preds))
    tn = sum(not l and not p for l, p in zip(labels, preds))
    fp = sum(not l and p for l, p in zip(labels, preds))
    fn = sum(l and not p for l, p in zip(labels, preds))

    def f1(tp_, fp_, fn_):
        return 2 * tp_ / (2 * tp_ + fp_ + fn_) if tp_ else 0.0

    n = len(labels)
    return {"n": n, "accuracy": (tp + tn) / n if n else 0.0, "macro_f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn}}
