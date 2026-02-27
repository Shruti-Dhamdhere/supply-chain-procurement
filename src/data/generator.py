"""
ProcureIQ — Synthetic Procurement Data Generator  (Day 2)
==========================================================
Generates a realistic procurement dataset with 5 entity types drawn from
the ProcureIQ architecture:
    Supplier · Component · Country · Contract · Route

Design decisions:
  - Spend and revenue follow log-normal (mirrors real procurement data)
  - Quality / delivery scores use Beta distributions (bounded 0-1, right-skewed)
  - Lead times use Gamma (positive, heavy-tailed)
  - Risk events and sole-source flags use Bernoulli with domain-calibrated p
  - Routes encode realistic logistics (cost per kg, transit days, modal type)

All counts, seeds, and output paths are driven by params.yaml → data.synthetic
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import synthetic_cfg, SYNTHETIC_DIR

# ─── Seeding ──────────────────────────────────────────────────────────────────
_CFG = synthetic_cfg
SEED = _CFG.get("random_seed", _CFG.get("seed", 42))
random.seed(SEED)
np.random.seed(SEED)

# ─── Curated Entity Name Pools ────────────────────────────────────────────────
# Real-sounding industrial supplier names grouped by category.
# 130 names → cycling covers n_suppliers=120 with no repeats.

_SUPPLIER_NAMES = [
    # Electronics & Connectivity
    "Kyocera Advanced Components", "Murata Electronics", "TDK Industrial Solutions",
    "Vishay Precision Group", "Molex Connectivity", "TE Connectivity",
    "Amphenol Industrial", "Bel Fuse Components", "CTS Electronic Components",
    "KEMET Electronics", "Bourns Electronics", "Panasonic Industrial Devices",
    "Alps Alpine Technologies", "Hirose Electric", "Sumitomo Electric Industries",
    "Shindengen Electric", "Rohm Semiconductor", "Sanken Electric",
    # Mechanical & Precision Parts
    "Precision Castparts Corp", "Parker Hannifin", "SKF Bearing Solutions",
    "NSK Technologies", "Timken Industrial", "Rexnord Components",
    "Gates Polymer & Rubber", "Trelleborg Sealing Solutions",
    "Freudenberg Sealing Technologies", "Dana Incorporated",
    "Schaeffler Group", "NTN Bearing Corp", "JTEKT Corporation",
    "Kaydon Specialty Bearings", "Roper Technologies",
    # Raw Materials & Advanced Materials
    "Materion Advanced Materials", "Carpenter Technology", "Allegheny Technologies",
    "Haynes International", "Special Metals Corp", "Plansee Group",
    "Heraeus Holding", "Umicore Advanced Materials", "Cabot Microelectronics",
    "Rogers Corporation", "Celanese Specialty Materials", "Solvay Advanced Polymers",
    "Evonik Industries", "Lanxess Specialty Chemicals", "Clariant International",
    # Packaging
    "Sealed Air Corporation", "Berry Global Group", "Sonoco Products",
    "Pregis Packaging Systems", "Nefab Group", "Lacerta Group",
    "UFP Technologies", "Hexacomb Protective Packaging", "Innopack Solutions",
    "Ranpak Holdings",
    # Logistics & 3PL
    "Geodis Supply Chain", "Kuehne + Nagel", "Bolloré Logistics",
    "DB Schenker Industrial", "CEVA Logistics", "Nippon Express",
    "Kintetsu World Express", "Hitachi Transport System", "Agility Logistics",
    "Rhenus Logistics",
    # MRO & Industrial Distribution
    "Grainger Industrial Supply", "MSC Industrial Direct", "Fastenal Company",
    "Applied Industrial Technologies", "Kaman Distribution", "Motion Industries",
    "DXP Enterprises", "Wesco International", "Hagemeyer Americas",
    "Anixter International",
    # Software & Engineering Tools
    "PTC Industrial Software", "Siemens Digital Industries", "Dassault Systèmes",
    "ANSYS Simulation", "Cadence Design Systems", "Synopsys Inc",
    "Mentor Graphics", "National Instruments", "MathWorks",
    "Zuken Engineering Tools",
    # Contract Manufacturing & Assemblies
    "Jabil Circuit", "Flex Ltd", "Celestica Inc",
    "Benchmark Electronics", "Sanmina Corporation", "Kimball Electronics",
    "API Technologies", "IEC Electronics", "Creation Technologies",
    "SMTEK International",
    # Semiconductors & Wafer Equipment
    "Entegris Inc", "Cohu Semiconductor", "Axcelis Technologies",
    "Brooks Automation", "Ultra Clean Holdings", "Photronics Inc",
    "Onto Innovation", "Ichor Holdings", "Plasma-Therm",
    "Advanced Energy Industries",
    # Chemical & Compounds
    "Momentive Performance Materials", "Henkel Industrial Adhesives",
    "Loctite Industrial", "3M Advanced Materials", "Dow Chemical Solutions",
    "BASF Performance Materials", "DuPont Advanced Materials",
    "Huntsman Advanced Materials", "Eastman Chemical", "Arkema Specialty Chemicals",
    # Electromechanical & Sensors
    "Honeywell Sensing", "Sensata Technologies", "Measurement Specialties",
    "Kistler Instrument", "PCB Piezotronics", "Omega Engineering",
    "Roper Industries", "IDEX Corporation", "Watts Water Technologies",
    "Circor International",
]

# Component description templates — realistic part/assembly descriptions
_COMPONENT_DESCRIPTIONS = [
    "32-bit ARM Cortex-M4 Microcontroller 168MHz",
    "Multilayer Ceramic Capacitor 100nF X7R 50V",
    "Thick Film Chip Resistor 10kΩ 1% 0402",
    "4-Layer FR4 PCB Assembly 150×100mm",
    "IP67 Circular Connector 12-Pin Bayonet",
    "MEMS Inertial Measurement Unit ±16g",
    "Brushless DC Motor 24V 80W 3000RPM",
    "Helical Gear Set Module 2 Steel Case-Hardened",
    "Deep Groove Ball Bearing 6205 25×52×15mm",
    "FKM O-Ring 50×3mm Fluoroelastomer",
    "NOR Flash Memory 128Mb SPI Interface",
    "48V DC-DC Buck Converter 20A Synchronous",
    "Single-Mode Fiber Optic Transceiver 10GbE",
    "SMA RF Connector 50Ω Panel Mount",
    "PTFE Shaft Seal 30×52×8mm",
    "Angular Contact Ball Bearing Pair 7206",
    "Injection-Moulded PA66 Housing Assembly",
    "Aluminium Extrusion Profile 40×80mm T-Slot",
    "Titanium Fastener Set M6 Grade 5 (50pcs)",
    "SMD Tantalum Capacitor 47µF 25V Case D",
    "16-Channel ADC 24-bit SPI 1MSPS",
    "Radar Level Sensor 80GHz 4–20mA Output",
    "Harmonic Drive Gearbox 100:1 Ratio",
    "Worm Gear Reducer 50:1 IEC B5 Flange",
    "Pneumatic Cylinder 63mm Bore 200mm Stroke",
    "3-Phase IGBT Power Module 1200V 50A",
    "Polyurethane Drive Belt Profile T10 1000mm",
    "Stainless Steel Solenoid Valve 1/4 NPT 24VDC",
    "Ethernet Switch 8-Port Managed DIN Rail",
    "Epoxy Structural Adhesive Film 0.13mm",
    "Silicon Carbide Schottky Diode 650V 20A",
    "Linear Guide Rail HGR20 Length 500mm",
    "CAN Bus Transceiver ISO 11898-2 Automotive",
    "Laser Distance Sensor Class 2 50m Range",
    "Servomotor 400W 3000RPM Encoder Feedback",
    "ESD-Safe Corrugated Tray 400×300×100mm",
    "Zinc Die-Cast Housing IP54 150×100×60mm",
    "Liquid Crystal Polymer Connector 0.5mm Pitch",
    "Temperature Probe PT100 Class A 4-Wire",
    "Custom ASIC 28nm CMOS 1.8V I/O",
]


def _supplier_name(idx: int) -> str:
    """Cycle through the curated supplier list — no repeats for n ≤ 130."""
    return _SUPPLIER_NAMES[idx % len(_SUPPLIER_NAMES)]


def _component_desc(idx: int) -> str:
    """Cycle through realistic component descriptions."""
    return _COMPONENT_DESCRIPTIONS[idx % len(_COMPONENT_DESCRIPTIONS)]


# ─── Domain Vocabularies ──────────────────────────────────────────────────────

CATEGORIES = [
    "Electronic Components", "Mechanical Parts", "Raw Materials",
    "Packaging", "Software Licenses", "MRO Supplies",
    "Logistics Services", "Chemical Compounds", "Semiconductors", "Assemblies",
]

COUNTRIES = [
    "USA", "Germany", "China", "Japan", "Mexico", "India",
    "South Korea", "Taiwan", "Netherlands", "Canada", "Brazil",
    "Vietnam", "Thailand", "Malaysia", "Poland", "Czech Republic",
    "Hungary", "Ireland", "Singapore", "Australia", "France",
    "UK", "Italy", "Spain", "Turkey",
]

TRANSPORT_MODES = ["Air", "Sea", "Rail", "Road", "Multimodal"]
PAYMENT_TERMS   = ["Net-30", "Net-60", "Net-90", "2/10 Net-30", "EOM"]
CONTRACT_STATUS = ["Active", "Expired", "Under Negotiation", "Terminated"]
RISK_TIERS      = ["Low", "Medium", "High", "Critical"]
CRITICALITY     = ["Low", "Medium", "High", "Critical"]

# Transit days by mode (μ, σ)
TRANSIT_PARAMS = {
    "Air":        (3,  1),
    "Sea":        (28, 7),
    "Rail":       (14, 4),
    "Road":       (5,  2),
    "Multimodal": (18, 5),
}

# Cost per kg (USD) by mode
COST_PER_KG = {
    "Air":        (5.0,  2.0),
    "Sea":        (0.3,  0.1),
    "Rail":       (0.8,  0.3),
    "Road":       (1.5,  0.5),
    "Multimodal": (1.2,  0.4),
}


# ─── Entity Generators ────────────────────────────────────────────────────────

def gen_suppliers(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "supplier_id":          f"SUP-{i+1:04d}",
            "name":                 _supplier_name(i),
            "category":             random.choice(CATEGORIES),
            "country":              random.choice(COUNTRIES),
            "annual_spend_usd":     round(np.random.lognormal(13.0, 1.5), 2),
            "on_time_delivery":     round(float(np.random.beta(8, 2)), 4),
            "quality_score":        round(float(np.random.beta(9, 2)), 4),
            "lead_time_days":       max(1, int(np.random.gamma(4, 5))),
            "defect_rate_ppm":      max(0, int(np.random.exponential(350))),
            "financial_risk_score": round(float(np.random.beta(2, 8)), 4),
            "sustainability_score": round(float(np.random.uniform(0.3, 1.0)), 4),
            "years_active":         random.randint(1, 25),
            "is_sole_source":       int(random.random() < 0.15),
            "is_preferred":         int(random.random() < 0.30),
            "headcount":            max(10, int(np.random.lognormal(6, 1.2))),
            "revenue_usd_M":        round(float(np.random.lognormal(4, 1.5)), 1),
            "risk_tier":            random.choice(RISK_TIERS),
        })
    return pd.DataFrame(rows)


def gen_components(n: int) -> pd.DataFrame:
    prefixes = ["MCU", "CAP", "RES", "PCB", "CONN", "SENS", "MOT",
                "GEAR", "CHIP", "MEM", "PWR", "OPT", "RF", "SEAL", "BEAR"]
    rows = []
    for i in range(n):
        rows.append({
            "component_id":    f"CMP-{i+1:04d}",
            "part_number":     f"{random.choice(prefixes)}-{random.randint(10000, 99999)}",
            "description":     _component_desc(i),
            "category":        random.choice(CATEGORIES),
            "unit_cost_usd":   round(float(np.random.lognormal(3, 1.8)), 2),
            "annual_volume":   max(1, int(np.random.lognormal(8, 1.5))),
            "criticality":     random.choice(CRITICALITY),
            "lead_time_weeks": max(1, int(np.random.gamma(3, 3))),
            "inventory_days":  max(0, int(np.random.gamma(5, 8))),
            "weight_kg":       round(float(np.random.lognormal(0.5, 1.2)), 3),
            "is_custom":       int(random.random() < 0.25),
            "commodity_code":  f"{random.randint(1000,9999)}.{random.randint(10,99)}.{random.randint(10,99)}",
        })
    return pd.DataFrame(rows)


def gen_countries(country_list: list) -> pd.DataFrame:
    # Realistic country-level supply chain attributes
    rows = []
    risk_by_country = {
        "China": 0.55, "Taiwan": 0.60, "USA": 0.15, "Germany": 0.12,
        "Japan": 0.18, "Mexico": 0.35, "India": 0.40, "South Korea": 0.22,
        "Vietnam": 0.38, "Thailand": 0.33, "Malaysia": 0.30,
    }
    tariff_by_country = {
        "China": 0.25, "USA": 0.04, "Germany": 0.04, "Mexico": 0.05,
        "Japan": 0.04, "India": 0.15, "Taiwan": 0.10, "Brazil": 0.18,
    }
    for c in country_list:
        rows.append({
            "country_id":          f"COU-{c[:3].upper()}",
            "name":                c,
            "geopolitical_risk":   round(risk_by_country.get(c, random.uniform(0.1, 0.6)), 3),
            "avg_tariff_rate":     round(tariff_by_country.get(c, random.uniform(0.03, 0.20)), 3),
            "logistics_index":     round(float(np.random.uniform(2.5, 4.2)), 2),  # LPI scale
            "currency_volatility": round(float(np.random.uniform(0.01, 0.15)), 4),
            "labor_cost_index":    round(float(np.random.uniform(0.1, 1.0)), 3),  # relative to USA=1
            "trade_agreements":    random.randint(5, 80),
        })
    return pd.DataFrame(rows)


def gen_contracts(n: int, supplier_ids: list, component_ids: list) -> pd.DataFrame:
    base_date = datetime(2018, 1, 1)
    rows = []
    for i in range(n):
        start    = base_date + timedelta(days=random.randint(0, 1800))
        duration = random.choice([12, 24, 36, 48])
        end      = start + timedelta(days=duration * 30)
        value    = round(float(np.random.lognormal(12, 1.8)), 2)
        rows.append({
            "contract_id":          f"CON-{i+1:04d}",
            "supplier_id":          random.choice(supplier_ids),
            "primary_component_id": random.choice(component_ids),
            "start_date":           start.strftime("%Y-%m-%d"),
            "end_date":             end.strftime("%Y-%m-%d"),
            "duration_days":        duration * 30,
            "value_usd":            value,
            "payment_terms":        random.choice(PAYMENT_TERMS),
            "status":               random.choice(CONTRACT_STATUS),
            "is_active":            int(end > datetime.now()),
            "auto_renew":           int(random.random() < 0.40),
            "savings_realized_usd": round(value * float(np.random.uniform(0, 0.15)), 2),
            "sla_penalty_usd":      round(float(np.random.exponential(5000)), 2),
            "negotiation_rounds":   random.randint(1, 6),
            "has_rebate":           int(random.random() < 0.25),
            "price_per_unit":       round(float(np.random.lognormal(3, 1.5)), 2),
        })
    return pd.DataFrame(rows)


def gen_routes(n: int, country_list: list, component_ids: list) -> pd.DataFrame:
    """
    Routes model the physical logistics path: origin country → destination country,
    carrying a specific component category. These are the 'route' nodes in the KG.
    """
    rows = []
    for i in range(n):
        mode    = random.choice(TRANSPORT_MODES)
        t_mu, t_sig = TRANSIT_PARAMS[mode]
        c_mu, c_sig = COST_PER_KG[mode]
        origin  = random.choice(country_list)
        dest    = random.choice([c for c in country_list if c != origin])
        rows.append({
            "route_id":            f"RTE-{i+1:04d}",
            "origin_country":      origin,
            "destination_country": dest,
            "component_id":        random.choice(component_ids),
            "transport_mode":      mode,
            "transit_days":        max(1, int(np.random.normal(t_mu, t_sig))),
            "cost_per_kg_usd":     max(0.05, round(float(np.random.normal(c_mu, c_sig)), 3)),
            "reliability_score":   round(float(np.random.beta(7, 2)), 4),
            "carbon_kg_per_ton":   round(float(np.random.lognormal(3, 0.8)), 1),
            "is_active":           int(random.random() < 0.85),
            "customs_delay_days":  max(0, int(np.random.exponential(2))),
        })
    return pd.DataFrame(rows)


def gen_sourcing_links(supplier_ids: list, component_ids: list) -> pd.DataFrame:
    """Many-to-many: supplier ↔ component sourcing relationships."""
    pairs: set[tuple] = set()
    for cid in component_ids:
        n_src = random.randint(1, 3)
        for sid in random.sample(supplier_ids, min(n_src, len(supplier_ids))):
            pairs.add((sid, cid))
    df = pd.DataFrame(list(pairs), columns=["supplier_id", "component_id"])
    df["is_primary"]     = (np.random.random(len(df)) < 0.5).astype(int)
    df["unit_price_usd"] = np.round(np.random.lognormal(3, 1.5, len(df)), 2)
    df["share_pct"]      = np.round(np.random.uniform(10, 100, len(df)), 1)
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_all(output_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """Generate all synthetic procurement tables and write CSVs."""
    out = Path(output_dir) if output_dir else SYNTHETIC_DIR
    out.mkdir(parents=True, exist_ok=True)

    n_sup  = _CFG["n_suppliers"]
    n_cmp  = _CFG["n_components"]
    n_cou  = _CFG["n_countries"]
    n_con  = _CFG["n_contracts"]
    n_rte  = _CFG["n_routes"]

    country_list = COUNTRIES[:n_cou]

    print("⚙  suppliers  …", end="  ")
    suppliers  = gen_suppliers(n_sup);                 print(f"{len(suppliers)} rows")

    print("⚙  components …", end="  ")
    components = gen_components(n_cmp);                print(f"{len(components)} rows")

    print("⚙  countries  …", end="  ")
    countries  = gen_countries(country_list);          print(f"{len(countries)} rows")

    print("⚙  contracts  …", end="  ")
    contracts  = gen_contracts(
        n_con,
        suppliers["supplier_id"].tolist(),
        components["component_id"].tolist(),
    );                                                 print(f"{len(contracts)} rows")

    print("⚙  routes     …", end="  ")
    routes = gen_routes(
        n_rte,
        country_list,
        components["component_id"].tolist(),
    );                                                 print(f"{len(routes)} rows")

    print("⚙  sourcing links …", end="  ")
    sourcing = gen_sourcing_links(
        suppliers["supplier_id"].tolist(),
        components["component_id"].tolist(),
    );                                                 print(f"{len(sourcing)} rows")

    datasets = {
        "suppliers":     suppliers,
        "components":    components,
        "countries":     countries,
        "contracts":     contracts,
        "routes":        routes,
        "sourcing_links": sourcing,
    }
    for name, df in datasets.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)

    meta = {
        "generated_at":   datetime.now().isoformat(),
        "seed":           SEED,
        "counts":         {k: len(v) for k, v in datasets.items()},
        "schema_version": "2.0",
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅  All synthetic data saved to: {out}")
    return datasets


if __name__ == "__main__":
    generate_all()
