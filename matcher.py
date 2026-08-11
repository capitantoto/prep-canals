from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


def normalize(query: str) -> str:
    """Normalize queries to comply with client's document corpus conventions."""
    return query


class Matcher:
    def __init__(self, normalizer=normalize, vectorizer=None) -> "Matcher":
        self.normalizer = normalizer
        self.vectorizer = vectorizer or TfidfVectorizer

    def fit(self, queries: list[str], documents: list[str]):
        self.catalog = {}

    def match(self, queries: list[str], k=1) -> list[list[str]]:
        pass

    def evaluate(
        self, queries: list[str], documents: list[str], ks: tuple[int] = (1, 3, 5)
    ) -> dict[str, float]:
        """HR@K + MRR for K in ks"""

    def save(self, path: Path | None = None) -> None:
        """save (matcher, catalog) artifact"""
        path = path or Path.cwd() / "model.dump"

    @classmethod
    def load(cls, dump) -> "Matcher":
        pass
