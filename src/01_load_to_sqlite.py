import sqlite3
import pandas as pd


# Load CSV dataset
df = pd.read_csv("data/raw/customer_churn.csv")


# Connect to SQLite database
conn = sqlite3.connect("database/customer_churn.db")


# Save dataframe into SQLite
df.to_sql(
    "customer_data",
    conn,
    if_exists="replace",
    index=False
)


# Close connection
conn.close()


print("Database created successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")