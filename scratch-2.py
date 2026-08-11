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

# %%
import re
from collections import Counter

import pandas as pd

data = pd.read_csv("data/pairs.csv")
print(data.head().to_markdown())

canon = data.canonical_description.value_counts()

# %%
print(f"canon size: {len(canon)}\n", canon.head())

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
from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(ngram_range=(3, 3), analyzer="char_wb").fit([*X_train, *y_train])
# Q: how often should each canonincal sku and each client description appear for the vectorizer counts?

X_train_vec = tfv.transform(X_train)
categories = set(y_train)
codes = {v: k for k, v in enumerate(categories)}
categories_vec = tfv.transform(categories)

# %%
sims = X_train_vec @ categories_vec.T
print(X_train_vec.shape, categories_vec.shape, sims.shape)
# %%


def tf_repr(string: str, vectorizer: TfidfVectorizer | None = tfv) -> pd.Series:
    scores = vectorizer.transform([string])
    ngrams = vectorizer.get_feature_names_out()
    return pd.Series(scores.data, index=ngrams[scores.indices])


tf_repr(data.normalized_raw_description[189])
# %%

from sklearn.metrics import accuracy_score, top_k_accuracy_score

# train score
# np.matrix.A1's a property that returns a flattened ndarray... quirky
preds = sims.argmax(axis=1).A1
truth = y_train.map(codes)
print(preds.shape, truth.shape)
print(accuracy_score(truth, preds))

import numpy as np

# and top_k accuracy
k = 1
S = sims.toarray()
sorted_labels = np.argsort(-S)
k_preds = sorted_labels[:, :k]

assert all(k_preds.flatten() == preds)

assert accuracy_score(truth, preds) == top_k_accuracy_score(truth, S, k=1)
# %%
# np.array([row[indices] for row, indices in zip(sims.toarray(), k_preds)][:5])
# equivalent: np.take_along_axis(arr, indices, axis)
# %%

for i in (1, 2, 3, 5, 10, 20):
    print(f"Top {i} accuracy: {(100 * top_k_accuracy_score(truth, S, k=i)):.3f}")

# %%
# Now on test data

X_test_vec = tfv.transform(X_test)

sims_test = X_test_vec @ categories_vec.T
print(X_test_vec.shape, categories_vec.shape, sims_test.shape)

preds = sims_test.argmax(axis=1).A1
truth = y_test.map(codes)
print(accuracy_score(truth, preds))

# no need for y_test_vec, same eaxct categories as in training

# %%
S_test = sims_test.toarray()
for i in (1, 2, 3, 5, 10, 20):
    print(f"Top {i} accuracy: {(100 * top_k_accuracy_score(truth, S_test, k=i)):.3f}")

# v. good! or in series aform and them a plot:
hit_rates = pd.Series({i: top_k_accuracy_score(truth, S_test, k=i) for i in range(1, 21)})
print(hit_rates[:5])

hit_rates[:6].plot(kind="bar")
# all but 5 covered at k=5 already
# %%
# %%
# One improvement: Consider more ngrams!
# min length == 2 (ft, in, ...)
# max length == 4 (xhhw, thhn) or 5? (250ft, xhhw2, ...)
# Ideally actual cross-validation, for now just co it.
tfv2 = TfidfVectorizer(ngram_range=(2, 4), analyzer="char_wb").fit([*X_train, *y_train])

# %%
X_train_vec2 = tfv2.transform(X_train)
categories_vec2 = tfv2.transform(categories)
sims2 = X_train_vec2 @ categories_vec2.T
preds2 = sims2.argmax(axis=1).A1
truth = y_train.map(codes)
accuracy_score(truth, preds2)

# %%
# hmm worst train perf is not very good, maybe in test? doubt it.
X_test_vec2 = tfv2.transform(X_test)
sims_test2 = X_test_vec2 @ categories_vec2.T
preds_test2 = sims_test2.argmax(axis=1).A1
accuracy_score(y_test.map(codes), preds_test2)
# %%
# no improvement, really... let's check some misses
fails = y_test.map(codes).ne(preds)  # preds, noit 2, the first vectorizer did good enough
reverse_codes = {code: sku for sku, code in codes.items()}
test_df = pd.DataFrame(
    {
        "X": X_test,
        "y": y_test,
        "pred": pd.Series(preds).map(reverse_codes).values,  # .values so indexes align correctly
    }
)
test_df["match"] = ~fails
"""
y: "copper elbow 45 degree 2 in" ; pred: "copper elbow 45 degree 1 in"
- Marginal difference, single char, top retrieval not [perfect but second-fifth usually are

Others: removing dot '.' killed dfecimal numbers 1.25 -> 125 instead of 1 1/4
Others: mangl;ed up lengths (50ft != 500ft)
Others: cplg -> coupling not identified in abbreviations
...
Can improve, won't do just now. SHould consider dimension and measure extraction specifically probably, and/or char vs char_wb ( to get ngrams across the measure-dimension pair of consecutive 'words
')
"""
