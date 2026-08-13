from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from matcher import Candidate, Matcher

app = FastAPI()

filepath = Path("data/matcher.dump")
if filepath.exists():
    matcher = Matcher.load(filepath)
else:
    print(f"`{filepath}` should exist. Run python matcher.py first.")


class MatchRequest(BaseModel):
    queries: list[str]
    k: int = 3


@app.post("/match")
def match(req: MatchRequest) -> list[list[Candidate]]:
    results = matcher.match(req.queries, req.k)
    return [
        [Candidate(sku=sku, score=score) for sku, score in candidates] for candidates in results
    ]


# TODO: sugary syntax for single-query matching
