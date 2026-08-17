from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from matcher import Matcher

app = FastAPI()

filepath = Path("data/matcher.dump")
if filepath.exists():
    matcher = Matcher.load(filepath)
else:
    print(f"`{filepath}` should exist. Run python matcher.py first.")


class MatchRequest(BaseModel):
    queries: list[str]
    k: int = 3


class CandidateResponse(BaseModel):
    sku: str
    score: float


@app.post("/match")
def match(req: MatchRequest) -> list[list[CandidateResponse]]:
    results = matcher.match(req.queries, req.k)
    return [
        [CandidateResponse(sku=sku, score=score) for sku, score in candidates]
        for candidates in results
    ]


# TODO: sugary syntax for single-query matching
