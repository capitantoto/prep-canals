import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from typing import NamedTuple


class Candidate(NamedTuple):
    sku: str
    score: float


def normalize(query: str) -> str:
    """Normalize queries to comply with client's document corpus conventions."""
    lowercase = query.lower()
    lowercase = re.sub(r"\bitm#\d+\b", " ", lowercase)  # no item numbers
    abbreviations = {
        "'": "ft",
        # "ft": "feet",  # always ft in our catalog
        '"': "in",
        "#": " awg ",  # TODO: review later on, problematic at 'ITM#1234"
    }
    for abbrev, full in abbreviations.items():
        lowercase = lowercase.replace(abbrev, full)
    simplified = re.sub(r"[^a-z0-9 /]", "", lowercase)
    stripped_words = re.sub(r"^\s", " ", simplified).split()
    replacements = {
        "sol": "solid",
        "rd": "red",
        "ga": "awg",
        "blu": "blue",
        "grn": "green",
        # materials
        "cu": "copper",
        "al": "aluminum",
        "bv": "ball valve",
        "sch": "schedule",
        # A to amp or viceversa, 1p/2p to [1|2]-pole, ...
    }
    # not informative
    removed = ["ea", "bldg"]
    replaced = [replacements.get(word, word) for word in stripped_words if word not in removed]
    # TODO: review more (failing) strings and improve
    return " ".join(replaced)


class Matcher:
    def __init__(self, normalizer=normalize, vectorizer=None) -> "Matcher":
        self.normalizer = normalizer
        # default vectorizer is TF-IDF over trigrams
        self.vectorizer = vectorizer or TfidfVectorizer(ngram_range=(3, 3), analyzer="char_wb")

    def fit(self, queries: list[str], documents: list[str]):
        unique_documents = sorted(set(documents))
        self.catalog = dict(enumerate(unique_documents))
        self.vectorizer.fit([*queries, *unique_documents])
        self.vectorized_catalog = self.vectorizer.transform(unique_documents)
        # self.fitted = True for sklearn duck typing if needed later

    def match(self, queries: list[str], k=1) -> list[list[Candidate]]:
        normalized_queries = [normalize(q) for q in queries]
        vectorized_queries = self.vectorizer.transform(normalized_queries)
        scores = (vectorized_queries @ self.vectorized_catalog.T).toarray()  # densify for argsort
        top_k_indices = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top_k_indices, axis=1)
        results = []
        for row_scores, row_indices in zip(top_scores, top_k_indices, strict=True):
            top_documents = [self.catalog[idx] for idx in row_indices]
            results.append(
                [
                    Candidate(sku=doc, score=sc)
                    for doc, sc in zip(top_documents, row_scores, strict=True)
                ]
            )
        return results

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
