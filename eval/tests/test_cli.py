from cli import aggregate, gate_a


def rows(n_ok, n_fail, **kw):
    return [{"id": f"r{i}", "status": "ok", **kw} for i in range(n_ok)] + \
        [{"id": f"f{i}", "status": "failed"} for i in range(n_fail)]


def test_aggregate_invalid_over_2pct():
    a = aggregate("uk_read", rows(97, 3, ref="a", hyp="a"), e2e=False)
    assert a["invalid"] and "wer" not in a


def test_aggregate_read_set():
    a = aggregate("uk_read", rows(99, 1, ref="привіт світ", hyp="вітаю всіх"), e2e=False)     # wrong, but Ukrainian
    assert not a["invalid"] and a["failed"] == 1
    assert a["wer"] == 1.0 and a["russian_rate"] == 0.0
    assert a["russian"] == []
    # a real Voxtral line: Ukrainian letters mixed in, so a uk-vs-ru vote is not enough; any Russian counts
    voxtral = "Комиссия стала відповодью Мартелли на масштабные протесты против режима и кироспочались ужовки."
    b = aggregate("uk_read", rows(1, 0, ref="комісія проти режиму", hyp=voxtral), e2e=False)
    assert len(b["russian"]) == 1


def test_aggregate_e2e_reply_language():
    chain = {"word_end_to_endpoint": 300, "word_end_to_first_pcm": 700, "premature": False}
    a = aggregate("uk_read", rows(2, 0, reply="Привіт, як справи?", chain=chain, decisions=["speak"]), e2e=True)
    assert a["reply_lang_match"] == 1.0 and a["word_end_to_endpoint"]["p50"] == 300
    assert a["decisions"] == {"speak": 2}


def test_aggregate_e2e_no_reply_no_audio_russian():
    chain = {"word_end_to_endpoint": 300, "word_end_to_first_pcm": 700}
    rs = rows(1, 0, reply="Привіт, як справи?", chain=chain, decisions=["speak"]) + \
        rows(1, 0, reply="", chain={}, decisions=["listen"]) + \
        rows(1, 0, reply="Я не совсем понял.", no_audio=True, chain={}, decisions=["speak"])
    a = aggregate("uk_read", rs, e2e=True)
    assert a["failed"] == 0 and a["no_reply"] == 1 and a["no_audio"] == 1
    assert a["reply_langs"] == {"uk": 1, "ru": 1} and a["reply_lang_match"] == 0.5
    assert a["word_end_to_endpoint"]["p50"] == 300
    assert [x["text"] for x in a["russian"]] == ["Я не совсем понял."]      # reply audited even without texts


def test_gate_a():
    def doc(wer, russian=(), ids=("a", "b")):
        return {"rows": [{"id": i} for i in ids],
                "aggregate": {"wer": wer, "n": len(ids), "invalid": False, "russian": [{"id": i} for i in russian]}}
    assert gate_a(doc(0.08), doc(0.07), 2)[0].startswith("PASS")
    assert gate_a(doc(0.10), doc(0.07), 2)[0].startswith("FAIL: swap")           # 1.43x
    assert gate_a(doc(0.05, russian=["a"]), doc(0.07), 2)[0].startswith("FAIL: Voxtral writes Russian")
    assert gate_a(None, doc(0.07), 2)[0] == "undecided"
    assert gate_a(doc(0.08, ids=("a",)), doc(0.07), 2)[0] == "undecided"          # partial coverage
    assert gate_a(doc(0.08), doc(0.07, ids=("a", "c")), 2)[0] == "undecided"      # different clips
    unsure = doc(0.08)
    unsure["aggregate"]["russian_uncertain"] = [{"id": "a"}]
    assert gate_a(unsure, doc(0.07), 2)[0] == "undecided"                      # open review items
    bad = doc(0.07)
    bad["aggregate"] = {"invalid": True, "n": 2, "failed": 1}
    assert gate_a(doc(0.08), bad, 2)[0] == "undecided"


def test_aggregate_turn_unscored_counts_as_failed():
    rs = rows(97, 0, label=True, score=0.9) + rows(3, 0, label=False, score=None)
    a = aggregate("uk_turn", rs, e2e=False)
    assert a["n"] == 100 and a["failed"] == 3 and a["unscored"] == 3 and a["invalid"]


def test_russian_audit_survives_invalid_run_and_covers_english():
    rs = rows(1, 9, hyp="Позвони мне завтра, пожалуйста.", ref="x")
    a = aggregate("en_read", rs, e2e=False)
    assert a["invalid"] and [x["id"] for x in a["russian"]] == ["r0"]


def test_russian_audit_e2e_all_texts():
    # the user's on-screen transcript counts too: Russian there is Russian output
    texts = [{"kind": "user_word", "text": "масштабные"}, {"kind": "reply", "text": "Добре, зараз."},
             {"kind": "interject", "text": "Да, конечно."}, {"kind": "user_word", "text": "Привіт"}]
    a = aggregate("uk_read", rows(1, 0, reply="Добре, зараз.", texts=texts, chain={}, decisions=[]), e2e=True)
    assert sorted(x["kind"] for x in a["russian"]) == ["interject", "user_word"]


def test_russian_audit_failed_and_legacy_rows():
    failed = [{"id": "f", "status": "failed", "texts": [{"kind": "user_word", "text": "масштабные"}]}]
    legacy = [{"id": "l", "status": "ok", "reply": "Хорошо, конечно, сейчас.", "chain": {}, "decisions": []}]
    a = aggregate("uk_read", failed + legacy, e2e=True)
    assert sorted(x["id"] for x in a["russian"]) == ["f", "l"]
    assert a["audit_incomplete"] == 1


def test_asr_failure_partial_transcript_audited():
    rs = [{"id": "x", "status": "failed", "hyp_partial": "Комиссия стала масштабные"}] + rows(60, 0, ref="a", hyp="a")
    a = aggregate("uk_read", rs, e2e=False)
    assert [r["id"] for r in a["russian"]] == ["x"]
