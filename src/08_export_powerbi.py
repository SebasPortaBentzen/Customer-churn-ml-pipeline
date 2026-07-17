import pandas as pd

print("=" * 80)
print("EXPORT DATASET FOR POWER BI")
print("=" * 80)

df = pd.read_csv("data/processed/customer_churn_clean.csv")

output_path = "powerbi/customer_churn_powerbi.csv"

df.to_csv(
    output_path,
    index=False,
    sep=";",
    decimal=","
)

print("Power BI dataset exported successfully.")
print(f"Location: {output_path}")