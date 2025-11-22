#!/usr/bin/env python3

"""
predict_category.py

Učitava trenirani model (product_category_model.pkl)
i omogućava interaktivno predviđanje kategorije na osnovu naslova proizvoda.
"""

import argparse
import joblib


def load_model(model_path: str):
    print(f"[INFO] Učitavam model iz: {model_path}")
    model = joblib.load(model_path)
    print("[INFO] Model učitan.")
    return model


def interactive_loop(model):
    print("=== Predikcija kategorije proizvoda ===")
    print("Unesi naziv proizvoda (ili 'exit' za izlaz):")

    while True:
        try:
            title = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Izlaz.")
            break

        if title.lower() in ("exit", "quit", ""):
            print("[INFO] Izlaz.")
            break

        # model očekuje listu stringova
        pred = model.predict([title])[0]
        print(f"→ Predikcija kategorije: {pred}\n")


def main(args):
    model = load_model(args.model_path)
    interactive_loop(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively predict product category from title")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/product_category_model.pkl",
        help="Putanja do .pkl modela"
    )

    args = parser.parse_args()
    main(args)
