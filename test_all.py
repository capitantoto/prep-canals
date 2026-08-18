"""
Single test file for matching app
- test normalizer: known corner cases
- test matcher: exact match wins, ...
- test API: ???"""

import pytest
from fastapi.testclient import TestClient

from api import app
from matcher import Matcher, normalize

client = TestClient(app)


### Test API endpoint ###
def test_match_endpoint():
    res = client.post("/match", json={"queries": ["thhn solid red copper wire 500'"], "k": 2})
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1 and len(data[0]) == 2
    top = data[0][0]
    assert "sku" in top and "score" in top
    assert 0 <= top["score"] < 1.0000001


### Test normalizer ###
@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            "THHN Cu 12ga STR BLK 500'",
            "thhn copper 12 awg stranded black 500 ft",
        ),  # abbrevs + ' → ft
        ("COPPER cplg 3/4 in", "copper coupling 3/4 in"),  # more abbrevs, cplg -> coupling
        ("ITM#5783 xhhw wire", "xhhw wire"),  # SKU tag stripped, no fake 'awg' injected
        ('3/4" PVC SCH40 pipe', "3/4 in pvc schedule 40 pipe"),  # " → in, sch expansion
        ("aluminium 6 AWG solid", "aluminum 6 awg solid"),  # spelling variant
        ("250FT", "250 ft"),  # glued unit split
        # ("1.25 in copper tee", "1-1/4 in copper tee"),  # decimal → catalog fraction form
    ],
)
def test_normalize(raw, expected):
    normalized = normalize(raw)
    assert normalized == expected


### Test Matcher ###
CATALOG = ["Toy Story 2", "Toy Story 3", "Despicable Me"]


@pytest.fixture(scope="module")
def m():
    return Matcher().fit(CATALOG)


def test_exact_match(m):
    for idx in range(len(CATALOG)):
        expected = CATALOG[idx]
        actual = m.match([expected], k=1)[0][0].sku
        assert expected == actual


def test_relative_ordering(m):
    expected = CATALOG
    actual = [cand.sku for cand in m.match(expected, k=3)[0]]
    assert expected == actual


def test_roundtrip_preserves_predictions(m, tmp_path):
    query = "Toys R' Us"
    orig_doc = m.match([query], k=1)[0][0].sku
    m.save(tmp_path / "model.dump")
    mm = Matcher.load(tmp_path / "model.dump")
    dump_doc = mm.match([query], k=1)[0][0].sku
    assert orig_doc == dump_doc
