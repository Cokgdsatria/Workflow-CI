import argparse 
import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def main(data_path):
    df = pd.read_csv(data_path)
    X = df['clean_review'].fillna('')
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if os.getenv("GITHUB_ACTIONS") != "true":
        mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.set_experiment("Amazon_Reviews_Basic_Autolog")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="CLI_Training Run"):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ])
        pipeline.fit(X_train, y_train)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="namadataset_preprocessing/amazon_reviews_ready.csv")
    args = parser.parse_args()
    main(args.data_path)