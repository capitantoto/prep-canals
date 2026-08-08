# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# %%
tv = TfidfVectorizer(
    strip_accents="ascii",
    lowercase=True,
    analyzer="char_wb",
    ngram_range=(3, 3),  # only trigrams
)

# %%

data = pd.read_csv("data/pairs.csv")  # raw CSV reade might be faster

raw = data.raw_description.to_list()
canon = data.canonical_description.tolist()
unique_canon = data.canonical_description.unique().tolist()


def standardize(raw: pd.Series) -> pd.Series:
    std = raw.copy()
    replacements = {
        "'": "feet",
        "ft": "feet",
        '"': "in",
        "sch": "schedule ",
        "#": "awg",
        "ga": "awg",
        "cplg": "coupling",
        "cu": "copper",
        # ... and more
    }
    for old, new in replacements.items():
        std = std.str.replace(old, new)
    return std


def normalize(desc: pd.Series) -> pd.Series:
    std = standardize(desc.lower())
    # just alphanums and some specials
    clean = std.str.replace(r"[^a-z0-9 /]+", "", regex=True)
    return clean


# %%
data.raw_description[:20]
# %%
raw_tf = tv.fit_transform(raw)
canon_tf = tv.transform(canon)

# %%
pd.Series(tv.vocabulary_).sort_values(ascending=False)
# %%
from scipy.spatial.distance import cdist

dists = cdist(raw_tf.todense(), canon_tf.todense())
pred = [canon[idx] for idx in dists.argmin(axis=1)]

data["closest_description"] = pred
data["match"] = data.canonical_description == data.closest_description

print(data[~data.match][:10].to_markdown())

# %%
"""
0. NORMALIZE. leave a-zA-Z0-9
1. make sure qtys and dimensions extract cleanly (0.75 -> 3/4, ' -> ft, ...) 
    - and _together_, ideally (NER?)
2. Are we even extracting "40", "cplg", "A" & other short very common things? maybe more ngrams? id # 19 seems very off.
3. honest eval - holdout test 20%, train, eval in test
    - stratification? simple take: stratified holdout first: 20% of each sku
    - later: take 10% skus off at random, then repeat above on remaining 90%. report metrics as before, and then for the 10% of blind SKUs not trained on (thresold & warn even top suggestion)
4. Apify it
"""


# %%
def tf_ngram(str):
    tf_str = tv.transform([str])
    ngrams = tv.get_feature_names_out()[tf_str.indices]
    return pd.Series(tf_str.data, index=ngrams).sort_values(ascending=False)


idx = 19
(
    raw_str,
    canon_str,
    closest_str,
) = data.loc[idx][["raw_description", "canonical_description", "closest_description"]]
tf_ngram(closest_str)
# %%
train_raw, test_raw, train_canon, test_canon = train_test_split(
    raw, canon, stratify=canon, test_size=0.2
)

# %%
assert pd.Series(canon).nunique() == pd.Series(train_canon).nunique()
assert pd.Series(canon).nunique() == pd.Series(test_canon).nunique()
# %%
# accuracy@1 and recall @ 5 to see if they are at least in the neighborhood
