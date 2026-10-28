"""Generate a synthetic manufacturing customer dataset for a portfolio prototype."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 1800

def main():
    rng = np.random.default_rng(SEED)
    n = N_CUSTOMERS

    tenure = rng.integers(2, 121, n)
    orders_90d = rng.poisson(lam=np.clip(5.8 - tenure / 55, 1.2, 5.5), size=n)
    orders_90d = np.clip(orders_90d, 0, 18)
    avg_order_value = np.round(rng.lognormal(mean=8.15, sigma=0.55, size=n), 2)
    days_since_last_order = np.clip(
        rng.gamma(shape=2.0, scale=18.0, size=n) + (orders_90d == 0) * rng.integers(25, 95, n),
        1, 180
    ).astype(int)
    late_delivery_rate = np.round(rng.beta(1.8, 7.5, n), 2)
    complaint_count = rng.poisson(lam=0.45 + late_delivery_rate * 2.0, size=n)
    payment_delay_days = np.clip(rng.gamma(1.6, 5.0, n), 0, 45).round().astype(int)
    discount_dependency = np.round(rng.beta(2.0, 4.0, n), 2)
    order_frequency_change = np.round(rng.normal(-0.05, 0.28, n), 2)
    region = rng.choice(["North", "South", "East", "West", "Central"], n)
    product_category = rng.choice(
        ["Components", "Packaging", "Industrial Equipment", "Raw Materials", "Finished Goods"], n
    )
    customer_type = rng.choice(["Retailer", "Wholesaler", "Distributor"], n, p=[0.40, 0.35, 0.25])

    # Simulated churn mechanism: the data is synthetic and these relationships are illustrative.
    logit = (
        -2.25
        + 0.026 * days_since_last_order
        - 0.17 * orders_90d
        + 2.2 * late_delivery_rate
        + 0.30 * complaint_count
        + 0.025 * payment_delay_days
        - 1.0 * order_frequency_change
        - 0.008 * tenure
        + 1.2 * discount_dependency
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    df = pd.DataFrame({
        "customer_id": [f"CUST{i:05d}" for i in range(1, n + 1)],
        "tenure_months": tenure,
        "orders_last_90_days": orders_90d,
        "avg_order_value": avg_order_value,
        "days_since_last_order": days_since_last_order,
        "late_delivery_rate": late_delivery_rate,
        "complaint_count": complaint_count,
        "payment_delay_days": payment_delay_days,
        "discount_dependency": discount_dependency,
        "order_frequency_change": order_frequency_change,
        "region": region,
        "product_category": product_category,
        "customer_type": customer_type,
        "churn": churn,
    })

    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    path = out / "manufacturing_customers.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} synthetic customer records to: {path}")
    print(f"Illustrative churn rate: {df['churn'].mean():.1%}")
    print("Reminder: synthetic data is for demonstrating the workflow, not real business decisions.")

if __name__ == "__main__":
    main()
