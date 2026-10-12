from sets import parse_fleurs_tsv, read_manifest, write_manifest

TSV = ("1820\t14005473274866252108.wav\tПривіт, світе!\tпривіт світе\tп р и в і т | с в і т е |\t52480\tFEMALE\n"
       "1820\t99.wav\tПривіт, світе!\tпривіт світе\tп р и в і т | с в і т е |\t40000\tMALE\n")


def test_parse_fleurs_tsv():
    rows = parse_fleurs_tsv(TSV, lang="uk")
    assert [r["id"] for r in rows] == ["14005473274866252108", "99"]      # wav name is the key, not column 1
    assert rows[0]["ref"] == "Привіт, світе!"
    assert rows[0]["gender"] == "female" and rows[1]["gender"] == "male"
    assert rows[0]["seconds"] == 52480 / 16000
    assert rows[0]["wav"] == "14005473274866252108.wav" and rows[0]["lang"] == "uk"


def test_manifest_roundtrip(tmp_path):
    p = tmp_path / "s.jsonl"
    rows = [{"id": "a", "audio": "x.wav", "ref": "пам'ять"}, {"id": "b", "audio": "y.wav", "label": True}]
    write_manifest(str(p), rows)
    assert read_manifest(str(p)) == rows
