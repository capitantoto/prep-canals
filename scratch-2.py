#!/usr/bin/env python3
"""
SKU-matching code for canals'live coding interview.

Basic plan:
1. Load data, briefly observe & present summary statistics.
2. Normalize & clean raw data
3. Prepare basic evaluation mechanism: hold out 20%. On sampling:
  a. consider stratified sampling separating 20% of obs per known SKU (no "unknown" items in testing), then
  b. consider separating _all samples_, for 20% (or less) of SKUs (some "unknown" items in testing), show GoF results per set.
On goodness of fit, let's consider two metrics:
- accuracy @ 1: id the most likely candidate the correct one?
- acc@k: is the correct candidate relatively highly scored? (top 2.5% for 200 SKUs: acc@5)
- (maybe mrr (mean reciprocal rank))
4. prepare basic algo: ngram analyzer. star w/trigams as baseline (like pg does!), then compare some more ranges.
(5. optional: talk abt alternative algos:
  a. learning algos (candidate features, ground truth features, is_match response variable)
  b. NLP semantic/structured retrieval of dimensions, sizes, known items b4hand
)
6. apify it
7. test it!
"""

import re
from collections import Counter

import pandas as pd

data = pd.read_csv("data/pairs.csv")
print(data.head().to_markdown())

canon = data.canonical_description.value_counts()

# %%
print("canon size: %i\n" % len(canon), canon.head())

# %%
# count words

all_raw = [raw.lower().split(" ") for raw in data.raw_description]
cnt = pd.Series(Counter([w for words in all_raw for w in words])).sort_values(ascending=False)
print(cnt.head(40))
# %%
"""
2. Normalize and clean
- lowercase
(optional in english: remove accents)
- replace frequent acronyms by expanded versions: ft -> feet, ' -> feet, rd -> red... (colors, dimensions, other abbrevs)
- simplify charset: remove everything except a-z0-9, "/"
let's do it scalar (fast ops in strings), and only vectorize later if needed -- easier tests later!
"""


def normalize(string: str) -> str:
    lowercase = string.lower()
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
    print(replaced)
    # TODO: review more (failing) strings and improve
    return " ".join(replaced)


normalize(data.raw_description.iloc[189])

data["normalized_raw_description"] = data.raw_description.apply(normalize)
data["normalized_canonical_description"] = data.canonical_description.apply(normalize)
# %%
"""
4. train test split, stratified 20%
"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data.normalized_raw_description,
    data.normalized_canonical_description,
    train_size=0.8,
    random_state=42,
    stratify=data.canonical_description,
)

# %%
