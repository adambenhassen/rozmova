import pytest

from metrics import (binary_turn, corpus_cer, corpus_wer, edit_distance, majority_script, normalize,
                     russian_rate)


def test_normalize_apostrophes_case_punct():
    assert normalize("Пам'ять, ПАМ’ЯТЬ!") == "памʼять памʼять"
    assert normalize("Hello,   World?!") == "hello world"


def test_normalize_splits_hyphens():
    # FLEURS writes "one-hundred"; split hyphens on both sides so spelling choices don't count as errors
    assert normalize("будь-ласка — так") == "будь ласка так"
    assert normalize("one-hundred percent") == "one hundred percent"


def test_normalize_numbers_follow_neighbouring_script():
    assert normalize("predicted with 100% certainty") == "predicted with one hundred percent certainty"
    assert normalize("20th century research") == "twentieth century research"
    assert normalize("у 2010 році") == "у дві тисячі десять році"
    assert normalize("це 50% часу") == "це пʼятдесят відсотків часу"
    assert normalize("1,000 people") == "one thousand people"


def test_edit_distance():
    assert edit_distance("a b c".split(), "a x c".split()) == 1
    assert edit_distance([], ["a"]) == 1
    assert edit_distance(list("kitten"), list("sitting")) == 3


def test_corpus_wer_micro_average():
    assert corpus_wer([("a b c", "a x c")]) == pytest.approx(1 / 3)
    # 1 error over 3 words + 0 over 1 word -> 1/4, not mean(1/3, 0)
    assert corpus_wer([("a b c", "a x c"), ("d", "d")]) == pytest.approx(1 / 4)


def test_corpus_cer_ignores_spaces():
    assert corpus_cer([("ab cd", "ab ce")]) == pytest.approx(1 / 4)


def test_russian_rate():
    assert russian_rate(["привіт"]) == 0.0
    assert russian_rate(["привет"]) == 0.0           # lower bound: no ы э ъ ё
    assert russian_rate(["объект"]) == 1.0
    assert russian_rate(["объект", "hello", "привіт"]) == 0.5   # Latin words excluded
    assert russian_rate(["hello"]) is None


def test_majority_script():
    assert majority_script("Привіт, як справи?") == "cyrl"
    assert majority_script("set a timer на 5") == "latn"
    assert majority_script("123 !!") is None


def test_binary_turn():
    r = binary_turn([True, True, False, False], [True, False, False, False])
    assert r["n"] == 4 and r["accuracy"] == pytest.approx(0.75)
    # complete: P=1 R=0.5 F1=2/3; incomplete: P=2/3 R=1 F1=0.8
    assert r["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert r["confusion"] == {"tp": 1, "fn": 1, "fp": 0, "tn": 2}


def test_turn_score_window():
    from metrics import turn_score
    cls = ["speaking", "complete", "incomplete", "backchannel", "wait"]
    frames = [(880, [0, 0.0, 1.0, 0, 0]),      # before the clip end: ignored
              (960, [0, 0.6, 0.2, 0, 0]),       # 0.75
              (1040, [0.5, 0.1, 0.1, 0, 0]),    # 0.5
              (1360, [0, 1.0, 0.0, 0, 0])]      # at end + 400: outside
    assert turn_score(frames, cls, end_ms=960) == pytest.approx(0.625)
    assert turn_score(frames, cls, end_ms=5000) is None


def test_percentiles():
    from metrics import percentiles
    assert percentiles([5, 1, 3, None, 2, 4]) == {"p50": 3, "p90": 5}
    assert percentiles([]) == {"p50": None, "p90": None}


def test_reply_language():
    from metrics import reply_language
    assert reply_language("Я не совсем понял, о чём речь. Вы имеете в виду новость?") == "ru"
    assert reply_language("Я не зовсім зрозумів, про що йдеться. Ви маєте на увазі новину?") == "uk"
    assert reply_language("Sure, setting a timer.") == "en"
    assert reply_language("Я не совсем понял.") == "ru"           # no Russian-only letters: caught by words
    assert reply_language("Добре, зараз.") == "uk"
    assert reply_language("Мартелли.") is None


def test_contains_russian_without_lid(monkeypatch):
    import metrics
    monkeypatch.setattr(metrics, "LID_PATH", "/nonexistent")
    monkeypatch.setattr(metrics, "_lid", None)
    assert metrics.contains_russian("Комиссия стала відповодью Мартелли на масштабные протесты")   # ы
    assert metrics.contains_russian("Я не совсем понял.")                                          # words
    assert not metrics.contains_russian("Комісія стала відповіддю Мартеллі на масштабні протести.")
    assert not metrics.contains_russian("Sure, setting a timer.")
    # a Russian clause inside a Ukrainian or English line is not outvoted
    assert metrics.contains_russian("Привіт! Будь ласка, зараз. Хорошо, конечно.")
    assert metrics.contains_russian("A long English answer about the weather today. Да, конечно.")


def test_normalize_signed_and_grouped_numbers():
    assert normalize("temperature -5") == "temperature minus five"
    assert normalize("it costs 1,234.56") == "it costs one thousand two hundred and thirty four point five six"
    assert normalize("pre-2010 rules") == "pre two thousand and ten rules"


def test_contains_russian_short_names_vs_short_russian():
    import metrics
    # names and 1-2 word fragments must not trip language ID; short Russian is caught by marker words
    assert not metrics.contains_russian("Фотографи пізніше зайняли місце. Мендозу застрелили.")
    assert not metrics.contains_russian("Це одна з атракцій. Сан Паркс.")
    assert metrics.contains_russian("Позвони мне завтра.")
    assert metrics.contains_russian("від Буэнос-Айреса.")                          # Russian spelling of a name


def test_short_russian_replies_and_markers_not_outvoted():
    import metrics
    for t in ("Понятно.", "Спасибо!", "Привет.", "Посмотрю.", "Я слушаю.", "Согласен.", "Извините, поняла."):
        assert metrics.contains_russian(t), t
    assert metrics.contains_russian("Привіт, як справи? Понятно.")       # a Ukrainian line does not outvote it
    for t in ("Зрозуміло.", "Дякую!", "Привіт.", "Подивлюсь.", "Слухаю.", "Згоден.", "Ага.", "Угу."):
        assert not metrics.contains_russian(t), t


def test_markers_shared_with_ukrainian_do_not_flag():
    import metrics
    # "ладно" is left to language ID (borderline colloquial Ukrainian); these two are plain Ukrainian
    for t in ("Ви говорите українською.", "Це правильно."):
        assert not metrics.contains_russian(t), t


def test_single_word_never_uncertain(monkeypatch):
    import metrics
    monkeypatch.setattr(metrics, "lang_id", lambda t: "ru")
    monkeypatch.setattr(metrics, "lid_says_russian", lambda t: False)
    assert metrics.russian_verdict("Мартелли") == "clean"
    assert metrics.russian_verdict("Мендозу застрелили.") == "uncertain"
