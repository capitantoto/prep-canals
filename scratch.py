# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
canon = data.canonical_description.unique().tolist()

raw_tf = tv.fit_transform(raw)
canon_tf = tv.transform(canon)

# %%
pd.Series(tv.vocabulary_).sort_values(ascending=False)

from scipy.spatial.distance import cdist

dists = cdist(raw_tf.todense(), canon_tf.todense())
pred = [canon[idx] for idx in dists.argmin(axis=1)]

data["closest_description"] = pred
data["match"] = data.canonical_description == data.closest_description
