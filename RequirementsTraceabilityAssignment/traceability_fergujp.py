"""pip install scikit-learn numpy sentence-transformers
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import re

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

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')

# ============================================================
# 2. LOAD REQUIREMENTS DATA
# ============================================================

def load_requirements(file_path):
    nfrs = {}
    frs = []

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

    return nfrs, frs

# ============================================================
# 3. LOAD ANSWER LABELS
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
# 4. PREPROCESSING VARIANTS
# ============================================================

def preprocess_nltk_variant1(text):
    """
    Variant 1: Basic Tokenization + Lowercasing + Stopword Removal + Lemmatization
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    processed = [lemmatizer.lemmatize(token) for token in tokens 
                 if token not in stop_words and len(token) > 2]
    
    return ' '.join(processed)

def preprocess_nltk_variant2(text):
    """
    Variant 2: Aggressive Stemming + POS Tagging + Selective Token Filtering
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # POS tagging to filter for meaningful words
    pos_tagged = pos_tag(tokens)
    
    # Keep only nouns (NN, NNS, NNP), verbs (VB, VBD, VBG, VBN, VBP, VBZ), 
    # adjectives (JJ, JJR, JJS), and adverbs (RB, RBR, RBS)
    meaningful_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 
                      'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
    
    processed = [stemmer.stem(token) for token, pos in pos_tagged 
                 if pos in meaningful_pos and token not in stop_words and len(token) > 2]
    
    return ' '.join(processed)

def preprocess_nltk_variant3(text):
    """
    Variant 3: Sentence-Level Processing + Named Entity Extraction + Lemmatization
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Lowercase
    text = text.lower()

    # Remove special characters but keep periods for sentence tokenization
    text = re.sub(r'[^a-zA-Z0-9\s\.]', ' ', text)
    
    # Sentence tokenization
    sentences = sent_tokenize(text)
    
    processed_tokens = []
    
    for sentence in sentences:
        # Word tokenization
        tokens = word_tokenize(sentence)
        
        # Remove stopwords and lemmatize
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens 
                     if token not in stop_words and len(token) > 2]
        
        # Named entity recognition
        pos_tagged = pos_tag(lemmatized)
        entities = ne_chunk(pos_tagged)
        
        # Extract tokens (including entity-tagged ones)
        for item in entities:
            if hasattr(item, 'label'):
                # It's a named entity
                entity_text = ' '.join([word for word, _ in item.leaves()])
                processed_tokens.append(f"[{item.label()}:{entity_text}]")
            else:
                # Regular token
                processed_tokens.append(item[0])
    
    return ' '.join(processed_tokens)

def preprocess_text(text, variant=1):
    """
    Main preprocessing function that supports 3 NLTK variants.
 
    Variant 1: Basic Tokenization + Lowercasing + Stopword Removal + Lemmatization
    Variant 2: Aggressive Stemming + POS Tagging + Selective Token Filtering
    Variant 3: Sentence-Level Processing + Named Entity Extraction + Lemmatization
    """
    if variant == 1:
        return preprocess_nltk_variant1(text)
    elif variant == 2:
        return preprocess_nltk_variant2(text)
    elif variant == 3:
        return preprocess_nltk_variant3(text)
    else:
        raise ValueError(f"Invalid variant: {variant}. Must be 1, 2, or 3.")
    
# ============================================================
# 5. NORMALIZE NFR ORDER
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

    for lbl, txt in zip(NFR_ORDER, ordered_texts):
        print(f"  {lbl}: {txt[:60]}...")

    return ordered_texts

# ============================================================
# 6. TF-IDF SIMILARITY COMPUTATION
# ============================================================

def tfidf_similarity(frs, nfrs):
    # Normalize NFRs into fixed order
    nfr_texts = normalize_nfrs(nfrs)

    # Extract FR texts only
    fr_texts = [text for _, text in frs]

    # Combined corpus ensures shared TF-IDF vocabulary
    corpus = fr_texts + nfr_texts

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf = vectorizer.fit_transform(corpus)

    # Split vectors back into FR and NFR matrices
    fr_vectors = tfidf[:len(frs)]
    nfr_vectors = tfidf[len(frs):]

    # Compute cosine similarity
    similarities = cosine_similarity(fr_vectors, nfr_vectors)

    return similarities

# ============================================================
# 7. TF-IDF + SVM SIMILARITY COMPUTATION
# ============================================================

def train_svm(frs, answers, test_size=0.2, random_state=42):
    """
    Trains SVM using TF-IDF with proper train/test split to avoid data leakage.
    """
    texts = [text for _, text in frs]
    y = np.array([answers[fr_id] for fr_id, _ in frs])

    # Train/test split
    X_train, X_test, y_train, y_test, frs_train, frs_test = train_test_split(
        texts, y, frs, test_size=test_size, random_state=random_state
    )

    # Vectorize on TRAIN only
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model on TRAIN only
    model = OneVsRestClassifier(LinearSVC())
    model.fit(X_train_vec, y_train)

    # Predict on TEST only
    preds = model.predict(X_test_vec)

    # Build answer dict for test set
    answers_test = dict(zip([fr_id for fr_id, _ in frs_test], y_test))

    return preds, frs_test, answers_test


# ============================================================
# 8. EMBEDDING SIMILARITY COMPUTATION
#    - Vectorizes FRs and NFRs using Sentence-BERT
#    - Computes cosine similarity between each FR and each NFR
# ============================================================

def embedding_similarity(frs, nfrs):
    """
    Computes semantic similarity between FRs and NFRs
    using Sentence-BERT embeddings.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    fr_texts = [text for _, text in frs]

    nfr_texts = normalize_nfrs(nfrs)

    fr_emb = model.encode(fr_texts, normalize_embeddings=True)

    nfr_emb = model.encode(nfr_texts, normalize_embeddings=True)

    similarities = cosine_similarity(fr_emb, nfr_emb)

    return similarities

# ============================================================
# 9. MULTI-LABEL PREDICTION VIA THRESHOLDS
# ============================================================

def predict_labels(similarities, thresholds):
    """
    Applies per-label thresholds to similarity matrix
    """
    preds = np.zeros_like(similarities, dtype=int)

    for i, t in enumerate(thresholds):
        preds[:, i] = similarities[:, i] >= t

    print("Predicted positive counts per label:", preds.sum(axis=0))
    return preds

# ============================================================
# 10. EVALUATION
# ============================================================

def evaluate(preds, frs, answers):
    """
    Prints precision, recall, and F1-score per NFR, returns average F1-score
    """
    y_true = np.array([answers[fr_id] for fr_id, _ in frs])

    print("True positive counts per label:", y_true.sum(axis=0))

    print(
        classification_report(y_true,preds,target_names=["operational", "usability", "security"],zero_division=0)
    )

    # Return average F1-score for overall performance summary
    return classification_report(y_true, preds, target_names=["operational", "usability", "security"], zero_division=0, output_dict=True)["macro avg"]["f1-score"]

# ============================================================
# 11. THRESHOLD OPTIMIZATION
# ============================================================
def find_best_thresholds(sim, y_true):
    thresholds = []
    for i in range(sim.shape[1]):
        best_t, best_f1 = 0, 0
        for t in np.linspace(0,1,50):
            preds = (sim[:,i] >= t).astype(int)
            f1 = f1_score(y_true[:,i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return thresholds

# ============================================================
# 12. RUN SCRIPTS
# ============================================================
def run(filepath, answerpath, variant=1, method=1):
    nfrs, frs = load_requirements(filepath)
    answers = load_answers(answerpath)

    # preprocess FR texts based on selected variant
    preprocessed_frs = []
    for fr_id, fr_text in frs:
        processed_text = preprocess_text(fr_text, variant=variant)
        preprocessed_frs.append((fr_id, processed_text))

    # save preprocessed texts
    with open(f"preprocessed_frs_variant{variant}{filepath.split('/')[-1][:-4]}.txt", "w") as f:
        for fr_id, fr_text in preprocessed_frs:
            f.write(f"{fr_id}: {fr_text}\n")

    if method == 1:
        sim = tfidf_similarity(preprocessed_frs, nfrs)
    elif method == 2:
        sim = embedding_similarity(preprocessed_frs, nfrs)
    elif method == 3:
        preds, frs_test, answers_test = train_svm(preprocessed_frs, answers)
        f1 = evaluate(preds, frs_test, answers_test)
        return f1
    
    print(sim)
    sort_similarities(frs, sim, nfrs, vm=f"{variant}{method}{filepath.split('/')[-1][:-4]}")

    y_true = np.array([answers[fr_id] for fr_id,_ in frs])
    THRESHOLDS = find_best_thresholds(sim, y_true)

    preds = predict_labels(sim, thresholds=THRESHOLDS)

    f1 = evaluate(preds, frs, answers)
    return f1

# ============================================================
# 13. MERGE SORTING OF FRs BASED ON SIMILARITY SCORES
# ============================================================

def sort_similarities(frs, similarities, nfrs, vm=None):
    """
    Outputs similarity of each FR to each NFR separately
    """
    filename = f"sorted_frs_{vm}.txt" if vm else "sorted_frs.txt"
    with open(filename, "w") as f:

        for j, (nfr_id, nfr_text) in enumerate(nfrs.items()):

            f.write(f"\n=== NFR {nfr_id}: {nfr_text} ===\n")

            fr_sim_pairs = [
                (fr_id, fr_text, similarities[i][j])
                for i, (fr_id, fr_text) in enumerate(frs)
            ]

            sorted_pairs = merge_sort(fr_sim_pairs)

            for fr_id, fr_text, sim in sorted_pairs:
                f.write(f"{fr_id}: {sim:.4f} -- {fr_text}\n")

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

    # Descending order by similarity
    while i < len(left) and j < len(right):
        if left[i][2] >= right[j][2]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result

# ============================================================
# 14. RUN PIPELINE
# ============================================================
if __name__ == "__main__":
    v11 = run(DATAPATH, ANSWERPATH, variant=1, method=1)
    v12 = run(DATAPATH, ANSWERPATH, variant=1, method=2)
    v13 = run(DATAPATH, ANSWERPATH, variant=1, method=3)

    v21 = run(DATAPATH, ANSWERPATH, variant=2, method=1)
    v22 = run(DATAPATH, ANSWERPATH, variant=2, method=2)
    v23 = run(DATAPATH, ANSWERPATH, variant=2, method=3)

    v31 = run(DATAPATH, ANSWERPATH, variant=3, method=1)
    v32 = run(DATAPATH, ANSWERPATH, variant=3, method=2)
    v33 = run(DATAPATH, ANSWERPATH, variant=3, method=3)

    firefox_v11 = run(FIREFOXDATAPATH, FIREFOXANSWERPATH, variant=1, method=1)
    firefox_v12 = run(FIREFOXDATAPATH, FIREFOXANSWERPATH, variant=1, method=2)
    firefox_v13 = run(FIREFOXDATAPATH, FIREFOXANSWERPATH, variant=1, method=3)

    firefox_v21 = run(FIREFOXDATAPATH, FIREFOXANSWERPATH, variant=2, method=1)
    firefox_v22 = run(FIREFOXDATAPATH, FIREFOXANSWERPATH, variant=2, method=2)
    firefox_v23 = run(FIREFOXDATAPATH, FIREFOXANSWERPATH, variant=2, method=3)

    firefox_v31 = run(FIREFOXDATAPATH, FIREFOXANSWERPATH, variant=3, method=1)
    firefox_v32 = run(FIREFOXDATAPATH, FIREFOXANSWERPATH, variant=3, method=2)
    firefox_v33 = run(FIREFOXDATAPATH, FIREFOXANSWERPATH, variant=3, method=3)

    with open("results_summary.txt", "w") as f:
        f.write("=== Traceability Results Summary ===\n\n")
        f.write(f"Variant 1 (Basic Preprocessing):\n")
        f.write(f"  TF-IDF Similarity F1-score: {v11}\n")
        f.write(f"  Embedding Similarity F1-score: {v12}\n")
        f.write(f"  SVM Classification F1-score: {v13}\n\n")

        f.write(f"Variant 2 (Aggressive Stemming + POS):\n")
        f.write(f"  TF-IDF Similarity F1-score: {v21}\n")
        f.write(f"  Embedding Similarity F1-score: {v22}\n")
        f.write(f"  SVM Classification F1-score: {v23}\n\n")

        f.write(f"Variant 3 (Sentence-Level + NER):\n")
        f.write(f"  TF-IDF Similarity F1-score: {v31}\n")
        f.write(f"  Embedding Similarity F1-score: {v32}\n")
        f.write(f"  SVM Classification F1-score: {v33}\n\n")

        f.write("=== Firefox Data Results ===\n\n")
        f.write(f"Variant 1 (Basic Preprocessing):\n")
        f.write(f"  TF-IDF Similarity F1-score: {firefox_v11}\n")
        f.write(f"  Embedding Similarity F1-score: {firefox_v12}\n")
        f.write(f"  SVM Classification F1-score: {firefox_v13}\n\n")

        f.write(f"Variant 2 (Aggressive Stemming + POS):\n")
        f.write(f"  TF-IDF Similarity F1-score: {firefox_v21}\n")
        f.write(f"  Embedding Similarity F1-score: {firefox_v22}\n")
        f.write(f"  SVM Classification F1-score: {firefox_v23}\n\n")

        f.write(f"Variant 3 (Sentence-Level + NER):\n")
        f.write(f"  TF-IDF Similarity F1-score: {firefox_v31}\n")
        f.write(f"  Embedding Similarity F1-score: {firefox_v32}\n")
        f.write(f"  SVM Classification F1-score: {firefox_v33}\n\n")
    

