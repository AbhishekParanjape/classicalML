"""
Logistic Regression model training on text data converted to ASCII embeddings.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import base64
import zlib
import os

# Load data
df = pd.read_csv("data/cipher_objective.csv")
X_text, y = df["text"], df["class"]

# Clean the data - ensure all text values are strings
X_text = X_text.astype(str)
y = y.astype(str)

# Step 1 & 2: Convert text to ASCII and embed into 100D vectors
def text_to_ascii_embedding(text, max_len=100):
    """Convert text to ASCII values and create fixed-length 100D vector"""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        text = ""
    else:
        text = str(text)
    
    ascii_vals = [ord(char) for char in text]
    
    if len(ascii_vals) < max_len:
        ascii_vals.extend([0] * (max_len - len(ascii_vals)))
    else:
        ascii_vals = ascii_vals[:max_len]
    
    return np.array(ascii_vals, dtype=np.float64)

# Apply embedding to all texts
X_embedded = np.array([text_to_ascii_embedding(text) for text in X_text])

print(f"Embedding shape: {X_embedded.shape}")
print(f"Sample embedding (first 10 values): {X_embedded[0][:10]}")

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_embedded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Step 4: Train logistic regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Test prediction
sample_text = X_text.iloc[0]
embedding = text_to_ascii_embedding(sample_text).reshape(1, -1)
pred = model.predict(embedding)[0]
proba = model.predict_proba(embedding)[0]
print(f"\nSample prediction:")
print(f"Text: {sample_text}")
print(f"Predicted class: {pred}")
print(f"Probabilities: {proba}")

# Save the model
with open('cipher_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved to 'cipher_model.pkl'")

print(f"\nModel expects input shape: (n_samples, 100)")
print(f"Model classes: {model.classes_}")

# Create blob functions
def _to_blob(obj):
    return base64.b64encode(
        zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), 9)
    ).decode("ascii")

model_blob = _to_blob(model)

def _make_getter(name, blob):
    return f'''def get_{name}():
    """
    Reconstruct the {name} object from an embedded compressed blob.
    """
    import base64, pickle, zlib
    raw = zlib.decompress(base64.b64decode("{blob}"))
    return pickle.loads(raw)
'''

# Save model getter
os.makedirs("snippets", exist_ok=True)
with open("snippets/get_model.txt", "w") as f:
    f.write(_make_getter("model", model_blob))

print("\nModel getter saved to 'snippets/get_model.txt'")

