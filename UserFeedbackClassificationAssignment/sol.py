import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from google_play_scraper import reviews, Sort
from datetime import datetime, timedelta

########################################
# PART 1 — TRAINING 3 CLASSIFICATION MODELS
########################################

# Load labeled dataset
df = pd.read_csv('tse_dataset.csv')

# Keep only review text and category
df = df[['body', 'category']]

# Filter to BUG and FEATURE only
df = df[df['category'].isin(['FEATURE', 'BUG'])]

# Convert labels to binary
df['category'] = df['category'].map({'BUG': 0, 'FEATURE': 1})

X = df['body']
y = df['category']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Initialize models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier(random_state=42)

models = {
    "Random Forest": rf,
    "KNN": knn,
    "Decision Tree": dt
}

# Save model evaluation to text file
with open("model_results.txt", "w", encoding="utf-8") as f:

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        f.write(f"\n===== {name} =====\n")
        f.write(f"Accuracy: {accuracy_score(y_test, preds)}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, preds)) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, preds, target_names=["BUG","FEATURE"]))
        f.write("\n\n")

########################################
# PART 2 - A — GOOGLE PLAY REVIEWS CLASSIFICATION AS BUG OR FEATURE
########################################

app_id = "us.zoom.videomeetings"
two_months_ago = datetime.now() - timedelta(days=60)

result, _ = reviews(
    app_id,
    lang='en',
    country='us',
    sort=Sort.NEWEST,
    count=1000
)

reviews_df = pd.DataFrame(result)

# Convert timestamp
reviews_df['at'] = pd.to_datetime(reviews_df['at'])

# Keep only last two months
reviews_df = reviews_df[reviews_df['at'] >= two_months_ago]

# Vectorize new reviews
new_reviews = reviews_df['content']
new_vec = vectorizer.transform(new_reviews)

# Predict using KNN
predictions = knn.predict(new_vec)
probs = knn.predict_proba(new_vec)

reviews_df['predicted'] = predictions
reviews_df['predicted_label'] = reviews_df['predicted'].map({0:'BUG', 1:'FEATURE'})
reviews_df['confidence'] = probs.max(axis=1)

# Save classifications to CSV
reviews_df[['content','predicted_label','confidence']].to_csv("classified_reviews.csv", index=False)

# Output best and worst predictions
good = reviews_df.sort_values('confidence', ascending=False).head(5)
bad = reviews_df.sort_values('confidence').head(5)

with open("well_classified.txt", "w", encoding="utf-8") as f:
    for _, row in good.iterrows():
        f.write(f"\nLabel: {row['predicted_label']}\n")
        f.write(f"Confidence: {round(row['confidence'],2)}\n")
        f.write(row['content'][:500] + "\n\n")

with open("poorly_classified.txt", "w", encoding="utf-8") as f:
    for _, row in bad.iterrows():
        f.write(f"\nLabel: {row['predicted_label']}\n")
        f.write(f"Confidence: {round(row['confidence'],2)}\n")
        f.write(row['content'][:500] + "\n\n")

print("Files generated:")
print("model_results.txt")
print("classified_reviews.xlsx")
print("well_classified.txt")
print("poorly_classified.txt")
