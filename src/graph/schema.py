from dataclasses import dataclass, field
from typing import NamedTuple

class EdgeType(NamedTuple):
    src: str
    relation: str
    dst: str

EDGE_TYPES = [
    EdgeType("supplier",  "supplies",      "component"),
    EdgeType("supplier",  "located_in",    "country"),
    EdgeType("contract",  "covers",        "component"),
    EdgeType("contract",  "signed_with",   "supplier"),
    EdgeType("route",     "originates_in", "country"),
    EdgeType("route",     "delivers_to",   "country"),
    EdgeType("route",     "carries",       "component"),
    EdgeType("supplier",  "co_supplier",   "supplier"),
    EdgeType("country",   "trades_with",   "country"),
]

ALL_EDGE_TYPES = EDGE_TYPES + [
    EdgeType(e.dst, f"rev_{e.relation}", e.src)
    for e in EDGE_TYPES if e.src != e.dst
]

NODE_FEATURE_SPECS = {
    "supplier":  ["annual_spend_usd","on_time_delivery","quality_score","lead_time_days",
                  "defect_rate_ppm","financial_risk_score","sustainability_score","years_active",
                  "is_sole_source","is_preferred","headcount","revenue_usd_M"],
    "component": ["unit_cost_usd","annual_volume","lead_time_weeks","inventory_days",
                  "weight_kg","is_custom","criticality_encoded"],
    "country":   ["geopolitical_risk","avg_tariff_rate","logistics_index",
                  "currency_volatility","labor_cost_index","trade_agreements"],
    "contract":  ["value_usd","savings_realized_usd","sla_penalty_usd","negotiation_rounds",
                  "duration_days","is_active","auto_renew","has_rebate"],
    "route":     ["transit_days","cost_per_kg_usd","reliability_score",
                  "carbon_kg_per_ton","is_active","customs_delay_days"],
}

CRITICALITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

@dataclass
class GraphStats:
    n_nodes:     dict = field(default_factory=dict)
    n_edges:     dict = field(default_factory=dict)
    total_nodes: int  = 0
    total_edges: int  = 0
    density:     float = 0.0

    def summary(self):
        lines = [
            "── ProcureIQ Knowledge Graph ─────────────────────────",
            f"  Total nodes : {self.total_nodes:>5d}",
            f"  Total edges : {self.total_edges:>5d}",
            f"  Density     : {self.density:.6f}",
            "", "  Nodes by type:",
        ]
        for k, v in self.n_nodes.items():
            lines.append(f"    {k:12s}: {v:>4d}")
        lines += ["", "  Edges by relation:"]
        for k, v in self.n_edges.items():
            lines.append(f"    {k:30s}: {v:>4d}")
        return "\n".join(lines)
