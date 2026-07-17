import os

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_PATH = "data/processed/customer_churn_ml.csv"
RESULTS_DIR = "data/results"
IMAGES_DIR = "images"


def create_models():
    """Create the classification models used in the comparison."""
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


def main():
    print("=" * 80)
    print("CUSTOMER CHURN - MODEL EVALUATION")
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

    print("\nTrain/Test Split completed!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    models = create_models()
    results = []

    best_model_name = None
    best_auc = float("-inf")
    best_predictions = None

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print(model_name)
        print("=" * 80)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC-AUC": auc,
            }
        )

        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            ax=ax,
            name=model_name,
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")

        print(
            "\nClassification Report:\n",
            classification_report(
                y_test,
                y_pred,
                zero_division=0,
            ),
        )

        if auc > best_auc:
            best_auc = auc
            best_model_name = model_name
            best_predictions = y_pred

    results_df = pd.DataFrame(results).sort_values(
        by="ROC-AUC",
        ascending=False,
    )

    best_metrics = results_df.loc[
        results_df["Model"] == best_model_name
    ]

    model_comparison_path = os.path.join(
        RESULTS_DIR,
        "model_comparison.csv",
    )

    best_metrics_path = os.path.join(
        RESULTS_DIR,
        "best_model_metrics.csv",
    )

    results_df.to_csv(
        model_comparison_path,
        index=False,
        sep=";",
        decimal=",",
    )

    best_metrics.to_csv(
        best_metrics_path,
        index=False,
        sep=";",
        decimal=",",
    )

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print(f"\nBest model: {best_model_name}")
    print(f"Best ROC-AUC: {best_auc:.4f}")
    print(f"\nSaved: {model_comparison_path}")
    print(f"Saved: {best_metrics_path}")

    ax.set_title("ROC Curve Comparison")
    ax.grid(True)

    roc_path = os.path.join(
        IMAGES_DIR,
        "roc_curve_comparison.png",
    )

    fig.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    cm = confusion_matrix(
        y_test,
        best_predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot()
    plt.title(f"Confusion Matrix - {best_model_name}")

    confusion_matrix_path = os.path.join(
        IMAGES_DIR,
        "confusion_matrix.png",
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print(f"Saved: {roc_path}")
    print(f"Saved: {confusion_matrix_path}")
    print("\nModel evaluation completed successfully.")


if __name__ == "__main__":
    main()