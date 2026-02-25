"""
src/api/schemas.py
Pydantic schemas for ProcureIQ API.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class SupplierSearchRequest(BaseModel):
    component_id: Optional[str] = None
    category: Optional[str] = None
    max_lead_time_days: Optional[int] = None
    max_unit_price: Optional[float] = None
    preferred_countries: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)

    class Config:
        json_schema_extra = {
            "example": {
                "category": "Electronic Components",
                "max_lead_time_days": 30,
                "max_unit_price": 50.0,
                "preferred_countries": ["MEXICO", "VIETNAM"],
                "top_k": 5,
            }
        }


class SupplierResult(BaseModel):
    supplier_id: str
    supplier_name: str
    country: str
    reliability_score: float
    avg_lead_time_days: int
    quality_score: float
    similarity_score: float


class SupplierSearchResponse(BaseModel):
    results: List[SupplierResult]
    n_results: int
    query_time_ms: float


class ClaimRequest(BaseModel):
    supplier_id: str
    claim_type: str
    claim_value: float
    description: str
    contract_id: Optional[str] = None


class ClaimResponse(BaseModel):
    claim_id: str
    draft_document: str
    recommended_actions: List[str]
    estimated_resolution_days: int
    priority: str


class CostSavingsRequest(BaseModel):
    supplier_id: Optional[str] = None
    category: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)


class CostSavingsOpportunity(BaseModel):
    component_id: str
    component_name: str
    current_price: float
    market_price: float
    savings_per_unit: float
    savings_pct: float
    recommended_action: str


class CostSavingsResponse(BaseModel):
    opportunities: List[CostSavingsOpportunity]
    total_potential_savings: float
    n_opportunities: int


class HealthResponse(BaseModel):
    status: str
    version: str
    graph_loaded: bool
    model_loaded: bool
    n_suppliers: int
    n_components: int
