import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


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
        # self.fitted = True for sklearn duck typing if needed later

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
