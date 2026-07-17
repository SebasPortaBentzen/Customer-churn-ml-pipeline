import pandas as pd

print("=" * 80)
print("CUSTOMER CHURN - FEATURE ENGINEERING")
print("=" * 80)

# Load cleaned dataset
df = pd.read_csv("data/processed/customer_churn_clean.csv")

print("\nDataset loaded successfully!")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nRemoving customerID...")

df.drop(
    columns=["customerID"],
    inplace=True
)

print("customerID removed successfully.")

binary_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling"
]

for column in binary_columns:

    if column == "gender":
        df[column] = df[column].map({
            "Female": 0,
            "Male": 1
        })

    else:
        df[column] = df[column].map({
            "No": 0,
            "Yes": 1
        })

print("\nBinary encoding completed.")

print("\nPreview after Binary Encoding:")
print(df.head())

print("\nColumn types after Binary Encoding:")
print(df.dtypes)

print("\nApplying One-Hot Encoding...")

categorical_columns = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

print("✓ One-Hot Encoding completed")

print("\nDataset shape after One-Hot Encoding:")
print(df.shape)

print("\nColumn types after One-Hot Encoding:")
print(df.dtypes)

output_path = "data/processed/customer_churn_ml.csv"

df.to_csv(
    output_path,
    index=False
)

print("\nML dataset saved successfully!")
print(f"Location: {output_path}")