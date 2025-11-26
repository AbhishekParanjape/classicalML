"""

"""

import base64, pickle, zlib, os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

class StatisticalFeatureExtractor(BaseEstimator, TransformerMixin):
    """100 statistical features: char freq, entropy, ascii stats, etc."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for s in X:
            codes = np.array([ord(ch) for ch in s], dtype=np.uint8)
            freq = np.bincount(codes, minlength=256)
            freq = freq / freq.sum()
            ent = -np.sum(freq * np.log2(freq + 1e-12))
            feats.append(np.hstack([
                freq,                       # 256
                [ent],                      # 1
                [codes.mean(), codes.std()] # 2
            ])[:100])                       # truncate to 100
        return np.vstack(feats)

class CharPositionFeature(BaseEstimator, TransformerMixin):
    """50 position-sensitive features: first/last 25 char codes."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = np.zeros((len(X), 50), dtype=float)
        for i, s in enumerate(X):
            codes = [ord(ch) for ch in (s[:25] + s[-25:])]
            out[i, :len(codes)] = codes
        return out


print("Loading cipher_objective.csv …")
df = pd.read_csv("data/cipher_objective.csv")
X_text, y = df["text"], df["class"]
print(f"   {len(X_text)} samples, {y.nunique()} classes")

# 90 / 10 stratified split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_text, y, test_size=0.10, random_state=71, stratify=y
)

tfidf = TfidfVectorizer(
    analyzer="char", ngram_range=(2, 4), max_features=5000
)
tfidf_vecs = tfidf.fit_transform(X_train_raw)

# statistical features
stat_extr = StatisticalFeatureExtractor()
stat_feats = stat_extr.fit_transform(X_train_raw)
stat_scaler = StandardScaler()
stat_feats = stat_scaler.fit_transform(stat_feats)

# position features
pos_extr = CharPositionFeature()
pos_feats = pos_extr.fit_transform(X_train_raw)
pos_scaler = StandardScaler()
pos_feats = pos_scaler.fit_transform(pos_feats)


from scipy.sparse import hstack
X_train = hstack([tfidf_vecs, stat_feats, pos_feats])
print("   Final train shape:", X_train.shape)


clf = LogisticRegression(
    max_iter=5000,
    C=10,
    random_state=71
)
clf.fit(X_train, y_train)

train_acc = clf.score(X_train, y_train)
print(f"   Train accuracy: {train_acc:.5f}")


X_test_tfidf  = tfidf.transform(X_test_raw)
X_test_stat   = stat_scaler.transform(stat_extr.transform(X_test_raw))
X_test_pos    = pos_scaler.transform(pos_extr.transform(X_test_raw))
X_test        = hstack([X_test_tfidf, X_test_stat, X_test_pos])
test_acc      = clf.score(X_test, y_test)
print(f"   Test  accuracy: {test_acc:.5f}")

def _to_blob(obj):
    return base64.b64encode(
        zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), 9)
    ).decode("ascii")

tfidf_blob   = _to_blob(tfidf)
scaler_blob  = _to_blob((stat_scaler, pos_scaler))
model_blob   = _to_blob(clf)

def _make_getter(name, blob):
    return f'''def get_{name}():
    """
    Reconstruct the {name} object from an embedded compressed blob.
    """
    import base64, pickle, zlib
    raw = zlib.decompress(base64.b64decode("{blob}"))
    return pickle.loads(raw)
'''

os.makedirs("snippets", exist_ok=True)
with open("snippets/get_vectorizer.txt", "w") as f:
    f.write(_make_getter("vectorizer", tfidf_blob))

with open("snippets/get_scaler.txt", "w") as f:
    f.write(_make_getter("scaler", scaler_blob))

with open("snippets/get_model.txt", "w") as f:
    f.write(_make_getter("model", model_blob))
