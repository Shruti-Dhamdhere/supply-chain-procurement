"""
tests/test_schemas.py
Unit tests for ProcureIQ schemas and data loader utilities.
"""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_supplier_search_request():
    from src.api.schemas import SupplierSearchRequest
    req = SupplierSearchRequest(
        category="Electronic Components",
        max_lead_time_days=30,
        top_k=5,
    )
    assert req.category == "Electronic Components"
    assert req.top_k == 5


def test_health_response():
    from src.api.schemas import HealthResponse
    h = HealthResponse(
        status="healthy",
        version="0.1.0",
        graph_loaded=True,
        model_loaded=True,
        n_suppliers=100,
        n_components=500,
    )
    assert h.status == "healthy"


def test_synthetic_base_generation():
    from src.data.loader import generate_synthetic_base
    df = generate_synthetic_base(100)
    assert len(df) == 100
    assert "recipient_name" in df.columns
    assert "award_amount" in df.columns
    assert "country" in df.columns


def test_synthetic_overlay_generation():
    from src.data.loader import generate_synthetic_base, generate_synthetic_overlay
    import yaml
    params = yaml.safe_load(open("params.yaml"))
    params["synthetic"]["n_suppliers"] = 50
    params["synthetic"]["n_components"] = 100
    params["synthetic"]["n_contracts"] = 200
    df = generate_synthetic_base(100)
    overlay = generate_synthetic_overlay(df, params)
    assert "suppliers" in overlay
    assert "components" in overlay
    assert "relationships" in overlay
    assert "claims" in overlay
    assert len(overlay["suppliers"]) > 0
    assert "disruption_label" in overlay["suppliers"].columns
    assert "price_anomaly" in overlay["components"].columns


def test_disruption_rate():
    from src.data.loader import generate_synthetic_base, generate_synthetic_overlay
    import yaml
    params = yaml.safe_load(open("params.yaml"))
    params["synthetic"]["n_suppliers"] = 200
    params["synthetic"]["n_components"] = 100
    params["synthetic"]["n_contracts"] = 500
    df = generate_synthetic_base(200)
    overlay = generate_synthetic_overlay(df, params)
    rate = overlay["suppliers"]["disruption_label"].mean()
    assert 0.05 <= rate <= 0.25
