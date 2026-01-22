"""pip install scikit-learn numpy sentence-transformers

"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split


# ============================================================
# 1. FILE PATHS
# ============================================================

dataPath = "./requirements_data.txt"
answerPath = "./requirements_answers.txt"


# ============================================================
# 2. LOAD REQUIREMENTS DATA
#    - Parse NFRs and FRs from text file
#    - NFRs stored as dict: {NFR_key: text}
#    - FRs stored as list: [(FR_id, text), ...]
# ============================================================

nfrs = {}
frs = []

print("Loading requirements data...")

with open(dataPath, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Enforce expected "KEY: value" format
        if ":" not in line:
            raise ValueError(f"Missing ':' at line {line_num}: {line}")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        # Separate NFRs and FRs based on prefix
        if key.upper().startswith("NFR"):
            nfrs[key] = value
        elif key.upper().startswith("FR"):
            frs.append((key, value))

print(f"Loaded {len(nfrs)} NFRs")
print(f"Loaded {len(frs)} FRs\n")


# ============================================================
# 3. LOAD ANSWER LABELS
#    - Format: FRx,0,1,0
#    - Stored as dict: {FR_id: np.array([o, u, s])}
# ============================================================

answers = {}

print("Loading answer labels...")

with open(answerPath, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()

        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]

        fr_id = parts[0]
        labels = np.array([int(x) for x in parts[1:]])

        answers[fr_id] = labels

print(f"Loaded labels for {len(answers)} FRs\n")


# ============================================================
# 4. NORMALIZE NFR ORDER
#    Ensures consistent column ordering:
#    [operational, usability, security]
# ============================================================

NFR_ORDER = ["operational", "usability", "security"]

def normalize_nfrs(nfrs):
    """
    Returns NFR texts in a fixed semantic order:
    operational -> usability -> security
    """
    ordered_texts = []

    for label in NFR_ORDER:
        for key, text in nfrs.items():
            if label.lower() in key.lower():
                ordered_texts.append(text)
                break
        else:
            raise ValueError(f"Missing NFR for label: {label}")

    print("NFR ordering confirmed:")
    for lbl, txt in zip(NFR_ORDER, ordered_texts):
        print(f"  {lbl}: {txt[:60]}...")

    print()
    return ordered_texts


# ============================================================
# 5. TF-IDF SIMILARITY COMPUTATION
#    - Vectorizes FRs and NFRs together
#    - Computes cosine similarity between each FR and each NFR
# ============================================================

def tfidf_similarity(frs, nfrs):
    # Normalize NFRs into fixed order
    nfr_texts = normalize_nfrs(nfrs)

    # Extract FR texts only
    fr_texts = [text for _, text in frs]

    # Combined corpus ensures shared TF-IDF vocabulary
    corpus = fr_texts + nfr_texts

    print("Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf = vectorizer.fit_transform(corpus)

    print(f"TF-IDF matrix shape: {tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}\n")

    # Split vectors back into FR and NFR matrices
    fr_vectors = tfidf[:len(frs)]
    nfr_vectors = tfidf[len(frs):]

    # Compute cosine similarity
    similarities = cosine_similarity(fr_vectors, nfr_vectors)

    print("Similarity statistics:")
    print("  Min:", similarities.min())
    print("  Max:", similarities.max())
    print("  Mean:", similarities.mean(), "\n")

    # Print first few FR similarity scores
    print("Sample similarity rows (first 5 FRs):")
    for i in range(min(5, len(frs))):
        print(f"  {frs[i][0]} -> {similarities[i]}")

    print()
    return similarities

# ============================================================
# 6. TF-IDF + SVM SIMILARITY COMPUTATION
#    - Vectorizes FRs and NFRs together
#    - Computes cosine similarity between each FR and each NFR
# ============================================================

def train_svm(frs, answers):
    texts = [text for _, text in frs]
    y = np.array([answers[fr_id] for fr_id, _ in frs])

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = OneVsRestClassifier(LinearSVC())
    model.fit(X, y)

    return model, vectorizer

def embedding_similarity(frs, nfrs):
    """
    Computes semantic similarity between FRs and NFRs
    using Sentence-BERT embeddings.
    """

    print("Loading Sentence-BERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    fr_texts = [text for _, text in frs]

    # IMPORTANT: enforce same NFR ordering as labels
    nfr_texts = normalize_nfrs(nfrs)

    print("Encoding FR texts...")
    fr_emb = model.encode(fr_texts, normalize_embeddings=True)

    print("Encoding NFR texts...")
    nfr_emb = model.encode(nfr_texts, normalize_embeddings=True)

    similarities = cosine_similarity(fr_emb, nfr_emb)

    print("\nEmbedding similarity statistics:")
    print("  Min:", similarities.min())
    print("  Max:", similarities.max())
    print("  Mean:", similarities.mean())

    print("\nSample embedding similarities (first 5 FRs):")
    for i in range(min(5, len(frs))):
        print(f"  {frs[i][0]} -> {similarities[i]}")

    print()
    return similarities

# ============================================================
# 6. MULTI-LABEL PREDICTION VIA THRESHOLDS
# ============================================================

def predict_labels(similarities, thresholds):
    """
    Applies per-label thresholds to similarity matrix
    """
    preds = np.zeros_like(similarities, dtype=int)

    for i, t in enumerate(thresholds):
        preds[:, i] = similarities[:, i] >= t

    print("Predicted positive counts per label:", preds.sum(axis=0))
    print()
    return preds


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate(preds, frs, answers):
    """
    Prints precision, recall, and F1-score per NFR
    """
    y_true = np.array([answers[fr_id] for fr_id, _ in frs])

    print("True positive counts per label:", y_true.sum(axis=0))
    print()

    print(
        classification_report(
            y_true,
            preds,
            target_names=["operational", "usability", "security"],
            zero_division=0
        )
    )


# ============================================================
# 8. RUN PIPELINE
# ============================================================
"""
sim = tfidf_similarity(frs, nfrs)

for i, name in enumerate(["operational", "usability", "security"]):
    col = sim[:, i]
    print(f"\n{name.upper()} similarity distribution:")
    print("  min:", col.min())
    print("  25%:", np.percentile(col, 25))
    print("  50%:", np.percentile(col, 50))
    print("  75%:", np.percentile(col, 75))
    print("  max:", col.max())

THRESHOLDS_EMBEDDING = [0.39, 0.22, 0.44]
THRESHOLDS_TFIDF = [0.01, 0.005, 0.11]

preds = predict_labels(sim, thresholds=THRESHOLDS_TFIDF)

print(preds)


evaluate(preds, frs, answers)
"""

# FR texts and labels
texts = [text for _, text in frs]
y = np.array([answers[fr_id] for fr_id, _ in frs])

# Split into train/test (e.g., 80/20)
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42
)

# Fit TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

# Train SVM
svm_model = OneVsRestClassifier(LinearSVC())
svm_model.fit(X_train, y_train)

# Predict on unseen test FRs
preds_svm = svm_model.predict(X_test)

# Evaluate
print("True positive counts per label:", y_test.sum(axis=0))
print("Predicted positive counts per label:", preds_svm.sum(axis=0))

print(classification_report(
    y_test, preds_svm,
    target_names=["operational", "usability", "security"],
    zero_division=0
))

