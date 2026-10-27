"""Generate the synthetic customer churn dataset used in this project.

Run from the project root:  python data/generate_data.py
The output is data/telecom_churn.csv. The seed is fixed, so everyone gets the same file.

The company is a fictional broadband + mobile provider ("SwiftNet"). Each row is one customer at the end of a month,
and `churn` says whether that customer cancelled during the NEXT month.

Churn follows realistic patterns (month-to-month contracts, new customers, fibre users without tech support, people
paying by electronic check and customers who complained a lot leave more often) plus randomness.
Some messiness is added on purpose so the notebook can show how to clean it:
  * total_charges is stored as TEXT, and is a blank " " for brand-new customers (tenure 0) - a classic real-world issue
  * a few missing avg_data_gb values (usage meter not reported)
  * a few duplicated rows (the same customer exported twice)
"""
import os

import numpy as np
import pandas as pd

SEED = 42
N = 5000


def make_dataset(n=N, seed=SEED):
    rng = np.random.default_rng(seed)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    # tenure depends on contract: long contracts belong to older customers
    tenure_mean = pd.Series(contract).map({"Month-to-month": 14, "One year": 32, "Two year": 46}).to_numpy()
    tenure = rng.gamma(2.0, tenure_mean / 2.0).round().clip(0, 72).astype(int)
    # about 2% brand-new customers (joined this month, no bill yet)
    tenure[rng.random(n) < 0.02] = 0

    internet = rng.choice(["Fiber optic", "DSL", "No internet"], size=n, p=[0.45, 0.35, 0.20])
    tech_support = np.where(internet == "No internet", "No internet service",
                            rng.choice(["Yes", "No"], size=n, p=[0.35, 0.65]))
    payment = rng.choice(["Electronic check", "UPI autopay", "Credit card (automatic)", "Bank transfer (automatic)"],
                         size=n, p=[0.32, 0.28, 0.20, 0.20])
    paperless = rng.choice(["Yes", "No"], size=n, p=[0.6, 0.4])
    senior = (rng.random(n) < 0.16).astype(int)
    city_tier = rng.choice(["Tier 1", "Tier 2", "Tier 3"], size=n, p=[0.45, 0.35, 0.20])

    # monthly charges in rupees: mobile plan + internet plan + extras
    base = pd.Series(internet).map({"Fiber optic": 999, "DSL": 649, "No internet": 299}).to_numpy()
    monthly = (base + rng.normal(0, 120, n) + np.where(tech_support == "Yes", 150, 0)).clip(199, 1999).round(0)

    # total charges = what the customer paid so far (roughly tenure x monthly, prices changed a bit over time)
    total = (monthly * tenure * rng.normal(1.0, 0.05, n)).round(2)

    avg_data = np.where(internet == "Fiber optic", rng.gamma(4, 30, n),
                        np.where(internet == "DSL", rng.gamma(3, 18, n), rng.gamma(2, 3, n))).round(1)
    complaints = rng.poisson(np.where(internet == "Fiber optic", 0.8, 0.5) + np.where(tech_support == "No", 0.3, 0))
    complaints = complaints.clip(0, 8)

    # ---- the "true" churn rule (unknown to the model) ----
    logit = (
        -3.15
        + pd.Series(contract).map({"Month-to-month": 1.25, "One year": -0.3, "Two year": -1.3}).to_numpy()
        - 0.035 * np.minimum(tenure, 48)                       # loyal customers stay
        + np.where(tenure <= 3, 0.6, 0.0)                      # first months are risky
        + np.where(internet == "Fiber optic", 0.45, 0.0)
        + np.where(tech_support == "No", 0.45, 0.0)
        + np.where(payment == "Electronic check", 0.55, 0.0)
        + np.where(paperless == "Yes", 0.2, 0.0)
        + 0.45 * complaints                                    # complaints are a strong signal
        + 0.0012 * (monthly - 700)                             # expensive plans churn more
        + 0.35 * senior
        + pd.Series(city_tier).map({"Tier 1": 0.15, "Tier 2": 0.0, "Tier 3": -0.15}).to_numpy()
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < prob, "Yes", "No")

    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
        "tenure_months": tenure,
        "contract_type": contract,
        "monthly_charges": monthly,
        # stored as TEXT on purpose; new customers have a blank " "
        "total_charges": np.where(tenure == 0, " ", total.astype(str)),
        "internet_service": internet,
        "tech_support": tech_support,
        "payment_method": payment,
        "paperless_billing": paperless,
        "num_complaints_6m": complaints,
        "avg_data_gb": avg_data,
        "senior_citizen": senior,
        "city_tier": city_tier,
        "churn": churn,
    })

    # realistic messiness: missing usage values and duplicated exports
    idx = rng.choice(n, size=40, replace=False)
    df.loc[idx, "avg_data_gb"] = np.nan
    dupes = df.sample(30, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telecom_churn.csv")
    data = make_dataset()
    data.to_csv(out, index=False)
    print(f"Wrote {len(data)} rows to {out}")
    print("Churn rate:", round((data["churn"] == "Yes").mean() * 100, 1), "%")
    print("Blank total_charges:", int((data["total_charges"] == " ").sum()))
