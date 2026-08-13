import re
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


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

    def fit(self, documents: list[str]):
        unique_documents = sorted(set(documents))
        self.catalog = dict(enumerate(unique_documents))
        self.vectorizer.fit(unique_documents)
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
        """Hit Rate @ K for K in ks"""
        max_k = max(ks)
        candidates_lists = self.match(queries, max_k)
        ranks = [
            next((i for i, cand in enumerate(candidates) if cand.sku == document), None)
            for candidates, document in zip(candidates_lists, documents, strict=True)
        ]
        return {k: sum(r is not None and r < k for r in ranks) / len(documents) for k in ks}

    def save(self, path: Path | None = None) -> None:
        """save (matcher, catalog) artifact"""
        path = path or Path.cwd() / "data/matcher.dump"
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "Matcher":
        path = path or Path.cwd() / "data/matcher.dump"
        return joblib.load(path)


if __name__ == "__main__":
    data = pd.read_csv("data/pairs.csv")
    X = data.raw_description
    y = data.canonical_description
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.8,
        random_state=42,
        stratify=y,
    )
    matcher = Matcher()
    matcher.fit(y_train)
    train_hr = matcher.evaluate(X_train, y_train)
    test_hr = matcher.evaluate(X_test, y_test)
    print("=== TRAIN ===", train_hr, "=== TEST ===", test_hr, sep="\n")
    matcher.save()
