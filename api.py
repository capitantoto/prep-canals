from fastapi import FastAPI
from pydantic import BaseModel

from matcher import Matcher, Candidate

app = FastAPI()

matcher = Matcher.load("data/matcher.dump")


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
