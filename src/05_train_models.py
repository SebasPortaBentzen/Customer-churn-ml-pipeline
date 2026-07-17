import os

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_PATH = "data/processed/customer_churn_ml.csv"
RESULTS_DIR = "data/results"
OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "training_metrics.csv",
)


def create_models():
    """Create the classification models used during training."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE
        ),
    }


def train_and_evaluate(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Train a model and return its classification metrics."""
    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(
        X_test
    )

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
    }

    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1 Score : {metrics['F1']:.4f}")

    return metrics


def main():
    print("=" * 80)
    print("CUSTOMER CHURN - MODEL TRAINING")
    print("=" * 80)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    df = pd.read_csv(
        DATA_PATH
    )

    print("\nDataset loaded successfully!")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    X = df.drop(
        columns=["Churn"]
    )
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain/Test Split completed!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    models = create_models()
    results = []

    for model_name, model in models.items():
        metrics = train_and_evaluate(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        results.append(
            metrics
        )

    results_df = pd.DataFrame(
        results
    ).sort_values(
        by="F1",
        ascending=False,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
        sep=";",
        decimal=",",
    )

    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nModel training completed successfully.")


if __name__ == "__main__":
    main()