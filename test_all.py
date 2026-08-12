"""
Single test file for matching app
- test normalizer: known corner cases
- test matcher: exact match wins, ...
- test API: ???"""

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


### Test API endpoint ###
def test_match_endpoint():
    res = client.post("/match", json={"quieres": ["thhn solid red copper wire 500'"], "k": 2})
    assert res.status_code == 200
    assert len(res[0]) == 2
    top = res[0][0]
    assert "sku" in top and "score" in top
    assert 0 <= top.score < 1.0000001


### Test normalizer ###
# Parametrize known edge cases from scratch

### Test Matcher ###
# On a basic catalog,
# - exact strings match themselves with score == 1.0
# - similar strings score higher than dissimilar strings
# - roundtrip: matches before and after save/load remain identical
