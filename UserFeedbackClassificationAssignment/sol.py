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

#part 1 grab data from tse_dataset.csv
df = pd.read_csv('tse_dataset.csv')

###PREPROCESSING

#get only body and category columns
df = df[['body', 'category']]

# from category column, get only "FEATURE" and "BUG"
df = df[df['category'].isin(['FEATURE', 'BUG'])]

# convert labels to binary
df['category'] = df['category'].map({'BUG': 0, 'FEATURE': 1})

X = df['body']
y = df['category']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier(random_state=42)

models = {
    "Random Forest": rf,
    "KNN": knn,
    "Decision Tree": dt
}

for name, model in models.items():
    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["BUG","FEATURE"]))

"""
We applied TF-IDF vectorization to convert textual user feedback into numerical features. 
The dataset was split into 80% training and 20% testing. 
Three classifiers (Random Forest, K-Nearest Neighbors, and Decision Tree) were trained. 
Performance was evaluated using confusion matrix, accuracy, precision, recall, and F1 score.
"""


#PART2
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

# Keep only last 2 months
reviews_df['at'] = pd.to_datetime(reviews_df['at'])
reviews_df = reviews_df[reviews_df['at'] >= two_months_ago]

print("Total reviews collected:", len(reviews_df))

new_reviews = reviews_df['content']

new_vec = vectorizer.transform(new_reviews)

predictions = rf.predict(new_vec)   # or your best model

reviews_df['predicted'] = predictions
reviews_df['predicted_label'] = reviews_df['predicted'].map({0:'BUG', 1:'FEATURE'})

probs = rf.predict_proba(new_vec)
reviews_df['confidence'] = probs.max(axis=1)

good = reviews_df.sort_values('confidence', ascending=False).head(5)

print("\nWELL CLASSIFIED:")
for _, row in good.iterrows():
    print("\nLabel:", row['predicted_label'])
    print("Confidence:", round(row['confidence'],2))
    print(row['content'][:300])

bad = reviews_df.sort_values('confidence').head(5)

print("\nPOORLY CLASSIFIED:")
for _, row in bad.iterrows():
    print("\nLabel:", row['predicted_label'])
    print("Confidence:", round(row['confidence'],2))
    print(row['content'][:300])
