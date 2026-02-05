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

DATAPATH = "./requirements_data.txt"
ANSWERPATH = "./requirements_answers.txt"

FIREFOXDATAPATH = "./firefox_data.txt"
FIREFOXANSWERPATH = "./firefox_data_answers.txt"

# FIREFOX DATA WAS TAKEN FROM THE FOLLOWING LINKS:
# NFRs: https://wiki.mozilla.org/Security/Reviews/Firefox_4_Non-Functional_Requirements
# https://www.firefox.com/en-US/firefox/147.0/releasenotes/
# https://www.firefox.com/en-US/firefox/146.0/releasenotes/
# https://www.firefox.com/en-US/firefox/145.0/releasenotes/


# ============================================================
# 2. LOAD REQUIREMENTS DATA
#    - Parse NFRs and FRs from text file
#    - NFRs stored as dict: {NFR_key: text}
#    - FRs stored as list: [(FR_id, text), ...]
# ============================================================

def load_requirements(file_path):
    nfrs = {}
    frs = []

    print(f"Loading requirements data from {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
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
    return nfrs, frs

# ============================================================
# 3. LOAD ANSWER LABELS
#    - Format: FRx,0,1,0
#    - Stored as dict: {FR_id: np.array([o, u, s])}
# ============================================================

def load_answers(file_path):
    answers = {}

    print(f"Loading answer labels from {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]

            fr_id = parts[0]
            labels = np.array([int(x) for x in parts[1:]])

            answers[fr_id] = labels

    print(f"Loaded labels for {len(answers)} FRs\n")
    return answers

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

# ============================================================
# 7. EMBEDDING SIMILARITY COMPUTATION
#    - Vectorizes FRs and NFRs using Sentence-BERT
#    - Computes cosine similarity between each FR and each NFR
# ============================================================

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
# 7. MULTI-LABEL PREDICTION VIA THRESHOLDS
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
# 8. EVALUATION
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
# 9. RUN SCRIPTS
# ============================================================
def run_option1(filepath, answerpath):
    nfrs, frs = load_requirements(filepath)
    answers = load_answers(answerpath)

    sim = tfidf_similarity(frs, nfrs)

    sort_similarities(frs, sim)

    THRESHOLDS_TFIDF = [0.1, 0.1, 0.1]

    preds = predict_labels(sim, thresholds=THRESHOLDS_TFIDF)

    evaluate(preds, frs, answers)

def run_option2(filepath, answerpath):
    nfrs, frs = load_requirements(filepath)
    answers = load_answers(answerpath)

    sim = embedding_similarity(frs, nfrs)

    sort_similarities(frs, sim)

    THRESHOLDS_EMBEDDING = [0.3, 0.3, 0.3]

    preds = predict_labels(sim, thresholds=THRESHOLDS_EMBEDDING)

    evaluate(preds, frs, answers)

def run_option3(filepath, answerpath):
    nfrs, frs = load_requirements(filepath)
    answers = load_answers(answerpath)

    model, vectorizer = train_svm(frs, answers)

    texts = [text for _, text in frs]
    X = vectorizer.transform(texts)

    preds = model.predict(X)

    evaluate(preds, frs, answers)

# ============================================================
# 10. OPTIONAL MERGE SORTING OF FRs BASED ON SIMILARITY SCORES
# ============================================================
def sort_similarities(frs, similarities):
    """
    Sorts FRs based on max similarity to any NFR
    TODO - compared to each NFR separately
    """

    fr_sim_pairs = [(fr_id, max(similarities[i])) for i, (fr_id, _) in enumerate(frs)]

    # Use manual merge sort instead of sorted()
    sorted_pairs = merge_sort(fr_sim_pairs)

    # TODO - output as text with the actual FR texts
    with open("sorted_frs.txt", "w") as f:
        for fr_id, sim in sorted_pairs:
            f.write(f"{fr_id}: {sim}\n")

    return sorted_pairs

def merge_sort(fr_sim_pairs):
    if len(fr_sim_pairs) <= 1:
        return fr_sim_pairs

    mid = len(fr_sim_pairs) // 2
    left = merge_sort(fr_sim_pairs[:mid])
    right = merge_sort(fr_sim_pairs[mid:])

    return merge(left, right)


def merge(left, right):
    result = []
    i = 0
    j = 0

    # Descending order by similarity (index 1)
    while i < len(left) and j < len(right):
        if left[i][1] >= right[j][1]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    while i < len(left):
        result.append(left[i])
        i += 1

    while j < len(right):
        result.append(right[j])
        j += 1

    return result


# ============================================================
# 11. RUN PIPELINE
# ============================================================
if __name__ == "__main__":
    run_option1(DATAPATH, ANSWERPATH)
    run_option2(DATAPATH, ANSWERPATH)
    run_option3(DATAPATH, ANSWERPATH)

    run_option1(FIREFOXDATAPATH, FIREFOXANSWERPATH)
    run_option2(FIREFOXDATAPATH, FIREFOXANSWERPATH)
    run_option3(FIREFOXDATAPATH, FIREFOXANSWERPATH)

