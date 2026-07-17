import sqlite3
import pandas as pd


# Connect to SQLite database
conn = sqlite3.connect("database/customer_churn.db")


def run_query(query, title):
    """
    Executes a SQL query and prints the result with a title.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    result = pd.read_sql_query(query, conn)
    print(result)

    return result


# 1. Total customers
run_query(
    """
    SELECT COUNT(*) AS total_customers
    FROM customer_data;
    """,
    "1. Total number of customers"
)


# 2. Churn distribution
run_query(
    """
    SELECT 
        Churn,
        COUNT(*) AS customers,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customer_data), 2) AS percentage
    FROM customer_data
    GROUP BY Churn;
    """,
    "2. Churn distribution"
)


# 3. Churn by contract type
run_query(
    """
    SELECT 
        Contract,
        Churn,
        COUNT(*) AS customers
    FROM customer_data
    GROUP BY Contract, Churn
    ORDER BY Contract, Churn;
    """,
    "3. Churn by contract type"
)


# 4. Average monthly charges by churn
run_query(
    """
    SELECT 
        Churn,
        ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
    FROM customer_data
    GROUP BY Churn;
    """,
    "4. Average monthly charges by churn"
)


# 5. Churn by internet service
run_query(
    """
    SELECT 
        InternetService,
        Churn,
        COUNT(*) AS customers
    FROM customer_data
    GROUP BY InternetService, Churn
    ORDER BY InternetService, Churn;
    """,
    "5. Churn by internet service"
)


# 6. Churn by payment method
run_query(
    """
    SELECT 
        PaymentMethod,
        Churn,
        COUNT(*) AS customers
    FROM customer_data
    GROUP BY PaymentMethod, Churn
    ORDER BY PaymentMethod, Churn;
    """,
    "6. Churn by payment method"
)


# 7. Average tenure by churn
run_query(
    """
    SELECT 
        Churn,
        ROUND(AVG(tenure), 2) AS avg_tenure_months
    FROM customer_data
    GROUP BY Churn;
    """,
    "7. Average tenure by churn"
)


# Close connection
conn.close()

print("\nSQL analysis completed successfully.")