#!/usr/bin/env python3

"""
train_model.py
Trenira model za klasifikaciju kategorije proizvoda na osnovu naslova
i čuva ga kao product_category_model.pkl
"""

import argparse
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Očisti imena kolona zbog leading/trailing razmaka
    df.columns = df.columns.str.strip()

    # obavezne kolone
    title_col = 'Product Title'
    category_col = 'Category Label'

    # izbaci redove bez naslova ili bez kategorije
    df = df.dropna(subset=[title_col, category_col])

    X = df[title_col]
    y = df[category_col]

    return X, y


def build_pipeline():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000
        )),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    return pipeline


def main(args):
    print(f"[INFO] Učitavanje podataka iz: {args.data_path}")
    X, y = load_data(args.data_path)

    print("[INFO] Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Treniranje modela...")
    model = build_pipeline()
    model.fit(X_train, y_train)

    print("[INFO] Evaluacija modela:")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # napravi folder models/ ako ne postoji
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    print(f"[INFO] Čuvam model u: {args.model_path}")
    joblib.dump(model, args.model_path)

    print("[INFO] Gotovo! Model je uspešno sačuvan.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train product category classifier")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/products.csv",
        help="Putanja do CSV fajla"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/product_category_model.pkl",
        help="Putanja gde će model biti sačuvan"
    )

    args = parser.parse_args()
    main(args)
