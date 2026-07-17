import os
import sqlite3

import pandas as pd


DATABASE_PATH = "database/customer_churn.db"
OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "customer_churn_clean.csv",
)


def main():
    print("=" * 80)
    print("CUSTOMER CHURN - DATA PREPROCESSING")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        df = pd.read_sql_query(
            "SELECT * FROM customer_data",
            connection,
        )

    print("\nDataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    initial_rows = len(df)

    df["TotalCharges"] = (
        df["TotalCharges"]
        .astype(str)
        .str.strip()
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["TotalCharges"]
    ).copy()

    removed_rows = initial_rows - len(df)

    print("\nTotalCharges cleaned successfully.")
    print(f"Rows removed: {removed_rows}")

    df["Churn"] = df["Churn"].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    if df["Churn"].isna().any():
        raise ValueError(
            "Unexpected values found in the Churn column."
        )

    df["Churn"] = df["Churn"].astype(int)

    print("Target variable encoded successfully.")

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nDataset exported successfully.")
    print(f"Location: {OUTPUT_PATH}")
    print(f"Final rows: {df.shape[0]}")
    print(f"Final columns: {df.shape[1]}")
    print("\nData preprocessing completed successfully.")


if __name__ == "__main__":
    main()