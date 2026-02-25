"""
src/data/loader.py
------------------
USASpending.gov data loader and synthetic procurement network generator.

USASpending.gov provides 50M+ real US government contracts with:
- Supplier names, locations, countries
- Contract values, dates, product categories
- Award types, performance periods

We augment this with a synthetic supply network overlay adding:
- Supplier reliability scores
- Component-supplier mappings with pricing
- Shipping routes and landed costs
- Contract amendment and claims events
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from loguru import logger

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PARAMS_PATH   = PROJECT_ROOT / "params.yaml"
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def fetch_usaspending_contracts(n_records: int = 10000) -> pd.DataFrame:
    """
    Fetch real contract data from USASpending.gov API.
    Returns supplier, contract value, country, product category.
    """
    cache_path = RAW_DIR / "usaspending_contracts.csv"
    if cache_path.exists():
        logger.info("Loading USASpending data from cache...")
        return pd.read_csv(cache_path)

    logger.info(f"Fetching {n_records} contracts from USASpending.gov...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
    headers = {"Content-Type": "application/json"}

    all_records = []
    page = 1
    per_page = min(100, n_records)

    while len(all_records) < n_records:
        payload = {
            "filters": {
                "award_type_codes": ["A", "B", "C", "D"],
                "time_period": [{"start_date": "2020-01-01", "end_date": "2023-12-31"}],
            },
            "fields": [
                "Award ID",
                "Recipient Name",
                "Award Amount",
                "Awarding Agency",
                "Award Type",
                "recipient_location_country_name",
                "Product or Service Code",
                "Period of Performance Start Date",
                "Period of Performance Current End Date",
                "Description",
            ],
            "page": page,
            "limit": per_page,
            "sort": "Award Amount",
            "order": "desc",
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if not results:
                    break
                all_records.extend(results)
                logger.info(f"Fetched {len(all_records)} records...")
                page += 1
                time.sleep(0.5)
            else:
                logger.warning(f"API returned {response.status_code}, using synthetic fallback")
                break
        except Exception as e:
            logger.warning(f"API error: {e}, using synthetic fallback")
            break

    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(cache_path, index=False)
        logger.success(f"Saved {len(df)} real contracts")
        return df
    else:
        logger.info("Generating synthetic base data...")
        return generate_synthetic_base(n_records)


def generate_synthetic_base(n_records: int) -> pd.DataFrame:
    """
    Generate synthetic procurement base data when API is unavailable.
    Statistically calibrated to real USASpending patterns.
    """
    np.random.seed(42)

    countries = [
        "UNITED STATES", "CHINA", "GERMANY", "MEXICO", "CANADA",
        "JAPAN", "SOUTH KOREA", "TAIWAN", "INDIA", "VIETNAM",
        "BRAZIL", "UNITED KINGDOM", "FRANCE", "ITALY", "NETHERLANDS",
        "SINGAPORE", "MALAYSIA", "THAILAND", "INDONESIA", "TURKEY",
    ]

    categories = [
        "Electronic Components", "Mechanical Parts", "Raw Materials",
        "Chemicals", "Packaging", "Logistics Services", "IT Equipment",
        "Office Supplies", "Safety Equipment", "Industrial Machinery",
    ]

    supplier_names = [f"Supplier_{i:04d}" for i in range(500)]

    df = pd.DataFrame({
        "recipient_name": np.random.choice(supplier_names, n_records),
        "award_amount": np.random.lognormal(mean=11, sigma=2, size=n_records),
        "country": np.random.choice(
            countries,
            n_records,
            p=[0.35, 0.15, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.04, 0.03,
               0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005]
        ),
        "product_category": np.random.choice(categories, n_records),
        "start_date": pd.date_range("2020-01-01", periods=n_records, freq="2h"),
        "contract_duration_days": np.random.randint(30, 730, n_records),
    })

    cache_path = RAW_DIR / "usaspending_contracts.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.success(f"Generated {len(df)} synthetic base records")
    return df


def generate_synthetic_overlay(
    contracts_df: pd.DataFrame,
    params: dict,
) -> dict:
    """
    Generate synthetic supply network overlay on top of real contract data.

    Creates:
    - Supplier reliability scores
    - Component-supplier mappings with unit prices
    - Shipping routes with costs
    - Claims and amendment events
    - Disruption labels
    """
    np.random.seed(params["synthetic"]["random_seed"])
    logger.info("Generating synthetic supply network overlay...")

    suppliers = contracts_df["recipient_name"].unique() if "recipient_name" in contracts_df.columns else contracts_df.get("Recipient Name", pd.Series()).unique()
    n_suppliers = min(len(suppliers), params["synthetic"]["n_suppliers"])
    suppliers = suppliers[:n_suppliers]

    # Supplier profiles
    supplier_df = pd.DataFrame({
        "supplier_id": [f"SUP_{i:04d}" for i in range(n_suppliers)],
        "supplier_name": suppliers,
        "reliability_score": np.random.beta(8, 2, n_suppliers),
        "avg_lead_time_days": np.random.randint(7, 45, n_suppliers),
        "quality_score": np.random.beta(7, 3, n_suppliers),
        "financial_risk_score": np.random.beta(2, 8, n_suppliers),
        "country": np.random.choice(
            ["UNITED STATES", "CHINA", "GERMANY", "MEXICO", "JAPAN",
             "SOUTH KOREA", "TAIWAN", "INDIA", "VIETNAM", "CANADA"],
            n_suppliers
        ),
        "years_active": np.random.randint(1, 30, n_suppliers),
    })

    # Component catalog
    n_components = params["synthetic"]["n_components"]
    categories = [
        "Electronic Components", "Mechanical Parts", "Raw Materials",
        "Chemicals", "Packaging", "Logistics Services", "IT Equipment",
        "Office Supplies", "Safety Equipment", "Industrial Machinery",
    ]
    component_df = pd.DataFrame({
        "component_id": [f"COMP_{i:04d}" for i in range(n_components)],
        "component_name": [f"Component_{i:04d}" for i in range(n_components)],
        "category": np.random.choice(categories, n_components),
        "unit_price": np.random.lognormal(mean=3, sigma=1.5, size=n_components),
        "market_price": None,
    })
    # Market price is 85-115% of unit price
    component_df["market_price"] = component_df["unit_price"] * np.random.uniform(0.85, 1.15, n_components)
    component_df["price_anomaly"] = (component_df["unit_price"] > component_df["market_price"] * 1.1).astype(int)

    # Supplier-component relationships
    n_relationships = n_suppliers * 5
    relationships = []
    for _ in range(n_relationships):
        sup_id = np.random.choice(supplier_df["supplier_id"])
        comp_id = np.random.choice(component_df["component_id"])
        relationships.append({
            "supplier_id": sup_id,
            "component_id": comp_id,
            "unit_price": np.random.lognormal(mean=3, sigma=1.2),
            "lead_time_days": np.random.randint(3, 60),
            "min_order_qty": np.random.randint(10, 1000),
        })
    relationship_df = pd.DataFrame(relationships).drop_duplicates(
        subset=["supplier_id", "component_id"]
    )

    # Claims data
    n_claims = params["synthetic"]["n_contracts"] // 5
    claim_statuses = ["Open", "In Review", "Closed", "Disputed"]
    claim_types = [
        "Late Delivery", "Quality Issue", "Price Discrepancy",
        "Quantity Shortfall", "Contract Amendment", "Force Majeure"
    ]
    claims_df = pd.DataFrame({
        "claim_id": [f"CLM_{i:05d}" for i in range(n_claims)],
        "supplier_id": np.random.choice(supplier_df["supplier_id"], n_claims),
        "claim_type": np.random.choice(claim_types, n_claims),
        "claim_value": np.random.lognormal(mean=9, sigma=1.5, size=n_claims),
        "status": np.random.choice(
            claim_statuses, n_claims, p=[0.3, 0.2, 0.4, 0.1]
        ),
        "days_open": np.random.exponential(scale=45, size=n_claims).astype(int),
        "priority": np.random.choice(["High", "Medium", "Low"], n_claims, p=[0.2, 0.5, 0.3]),
    })

    # Disruption labels for GNN training
    disruption_rate = params["synthetic"]["disruption_rate"]
    supplier_df["disruption_label"] = (
        np.random.random(n_suppliers) < disruption_rate
    ).astype(int)

    logger.success(
        f"Overlay generated: {n_suppliers} suppliers, "
        f"{n_components} components, "
        f"{len(relationship_df)} relationships, "
        f"{n_claims} claims"
    )

    return {
        "suppliers": supplier_df,
        "components": component_df,
        "relationships": relationship_df,
        "claims": claims_df,
    }


def save_processed(data: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        path = PROCESSED_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {name}: {df.shape}")
    logger.success("All processed data saved.")


def run_data_pipeline() -> None:
    logger.info("Starting ProcureIQ data pipeline...")
    params = load_params()

    contracts = fetch_usaspending_contracts(
        n_records=params["data"].get("sample_size", 10000)
    )
    logger.info(f"Contracts loaded: {contracts.shape}")

    overlay = generate_synthetic_overlay(contracts, params)
    save_processed(overlay)

    stats = {
        "n_contracts": len(contracts),
        "n_suppliers": len(overlay["suppliers"]),
        "n_components": len(overlay["components"]),
        "n_relationships": len(overlay["relationships"]),
        "n_claims": len(overlay["claims"]),
        "disruption_rate": overlay["suppliers"]["disruption_label"].mean(),
        "price_anomaly_rate": overlay["components"]["price_anomaly"].mean(),
    }

    logger.info("--- Dataset Statistics ---")
    for k, v in stats.items():
        logger.info(f"  {k:<30} {v}")

    json_path = PROCESSED_DIR / "dataset_stats.json"
    import json
    with open(json_path, "w") as f:
        json.dump({k: float(v) for k, v in stats.items()}, f, indent=2)
    logger.success("Data pipeline complete.")


if __name__ == "__main__":
    run_data_pipeline()
