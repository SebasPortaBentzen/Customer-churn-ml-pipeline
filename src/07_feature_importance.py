import os

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_PATH = "data/processed/customer_churn_ml.csv"
RESULTS_DIR = "data/results"
IMAGES_DIR = "images"


def main():
    print("=" * 80)
    print("CUSTOMER CHURN - FEATURE IMPORTANCE")
    print("=" * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train,
    )

    print("\nRandom Forest trained successfully.")

    feature_importance = (
        pd.DataFrame(
            {
                "Feature": X.columns,
                "Importance": model.feature_importances_,
            }
        )
        .sort_values(
            by="Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    csv_path = os.path.join(
        RESULTS_DIR,
        "feature_importance.csv",
    )

    feature_importance.to_csv(
        csv_path,
        index=False,
        sep=";",
        decimal=",",
    )

    top_features = feature_importance.head(10)

    plt.figure(figsize=(8, 5))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importance (Random Forest)")

    plt.grid(
        axis="x",
        linestyle="--",
        alpha=0.5,
    )

    for index, value in enumerate(top_features["Importance"]):
        plt.text(
            value,
            index,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    image_path = os.path.join(
        IMAGES_DIR,
        "feature_importance.png",
    )

    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print("\nTop 10 Features")
    print("-" * 40)
    print(top_features.to_string(index=False))

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {image_path}")
    print("\nFeature importance completed successfully.")


if __name__ == "__main__":
    main()