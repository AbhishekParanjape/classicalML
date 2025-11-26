"""
Complete training script to generate both model and vectorizer blobs.
This ensures both use the same parameters and are compatible.

Run this script to regenerate the embedded functions for your Agent class.
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import base64
import zlib
from sklearn.model_selection import train_test_split


print("="*80)
print("TRAINING CIPHER DECODER MODEL")
print("="*80)

df = pd.read_csv('data/cipher_objective.csv')
X_text = df['text']
y = df['class']
print(f"   ✓ Loaded {len(X_text)} samples")
print(f"   ✓ Classes: {y.unique()}")


vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,5), max_features=1000)

print(f"   ✓ Feature matrix shape: {X.shape}")
print(f"   ✓ Vocabulary size: {len(vectorizer.vocabulary_)}")


X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.20, random_state=71, stratify=y
)
print(f"   ✓ Train size: {len(X_train)}")
print(f"   ✓ Test  size: {len(X_test)}")

X_train_vec = vectorizer.fit_transform(X_train)   # fit on train 
X_test_vec  = vectorizer.transform(X_test)        # apply to test

print(f"   ✓ Train feature matrix shape: {X_train_vec.shape}")
print(f"   ✓ Test  feature matrix shape: {X_test_vec.shape}")


print("\n4. Training logistic regression model...")
model = LogisticRegression(max_iter=1000, random_state=71)
model.fit(X_train_vec, y_train)

train_acc = model.score(X_train_vec, y_train)
test_acc  = model.score(X_test_vec,  y_test)

print(f"   ✓ Training accuracy: {train_acc:.5f}")
print(f"   ✓ Test     accuracy: {test_acc:.5f}")

# Generate get_model() function
print("\n4. Generating get_model() function...")
model_blob = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
model_compressed = zlib.compress(model_blob, level=9)
model_b64 = base64.b64encode(model_compressed).decode("ascii")

get_model_snippet = f'''def get_model():
    """
    Reconstruct and return a scikit-learn model from an embedded, base64-encoded compressed blob.

    Security note:
    This uses pickle-compatible loading. Only use if you trust the source.
    """
    import base64
    import pickle as _p
    import zlib as _z; _decomp = _z.decompress
    _blob_b64 = "{model_b64}"
    _raw = _decomp(base64.b64decode(_blob_b64))
    model = _p.loads(_raw)
    return model
'''

print(f"   ✓ Model blob size: {len(model_b64)} characters")

# Generate get_vectorizer() function
print("\n5. Generating get_vectorizer() function...")
vectorizer_blob = pickle.dumps(vectorizer, protocol=pickle.HIGHEST_PROTOCOL)
vectorizer_compressed = zlib.compress(vectorizer_blob, level=9)
vectorizer_b64 = base64.b64encode(vectorizer_compressed).decode("ascii")

get_vectorizer_snippet = f'''def get_vectorizer():
    """
    Reconstruct and return the fitted CountVectorizer from an embedded, base64-encoded compressed blob.
    """
    import base64
    import pickle as _p
    import zlib as _z
    _blob_b64 = "{vectorizer_b64}"
    _raw = _z.decompress(base64.b64decode(_blob_b64))
    vectorizer = _p.loads(_raw)
    return vectorizer
'''
print(f"   ✓ Vectorizer blob size: {len(vectorizer_b64)} characters")

# Save to files
with open('get_model_snippet.txt', 'w') as f:
    f.write(get_model_snippet)
with open('get_vectorizer_snippet.txt', 'w') as f:
    f.write(get_vectorizer_snippet)


# Print the snippets
print("\n" + "="*80)
print("get_decoder_model() FUNCTION:")
print("="*80)
print(get_model_snippet)

print("\n" + "="*80)
print("get_vectorizer() FUNCTION:")
print("="*80)
print(get_vectorizer_snippet)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Training accuracy: {test_acc:.5f}")
print("="*80)

