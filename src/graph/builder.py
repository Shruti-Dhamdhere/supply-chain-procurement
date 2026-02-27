import json, logging
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import torch
from src.config import graph_cfg, SYNTHETIC_DIR, PROCESSED_DIR
from src.graph.schema import NODE_FEATURE_SPECS, CRITICALITY_MAP, GraphStats

log = logging.getLogger("procure-iq.graph")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

class KnowledgeGraphBuilder:
    def __init__(self, data_dir=None, output_dir=None):
        self.data_dir   = Path(data_dir)   if data_dir   else SYNTHETIC_DIR
        self.output_dir = Path(output_dir) if output_dir else PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.G = nx.MultiDiGraph()
        self.id_maps = {}
        self.node_features = {}
        self.hetero = None
        self._data = {}

    def build(self):
        self._load()
        self._nodes()
        self._edges()
        self._features()
        self.hetero = self._to_pyg()
        self._save()
        log.info("Build complete: %d nodes | %d edges",
                 self.G.number_of_nodes(), self.G.number_of_edges())
        return self

    def _load(self):
        for t in ["suppliers","components","countries","contracts","routes","sourcing_links"]:
            p = self.data_dir / f"{t}.csv"
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}. Run: python -m src.data.generator")
            self._data[t] = pd.read_csv(p)
            log.info("  Loaded %-18s %d rows", t, len(self._data[t]))

    def _safe(self, row, keys):
        out = {}
        for k in keys:
            v = row.get(k, 0.0)
            try:    out[k] = float(v) if (v is not None and v == v) else 0.0
            except: out[k] = 0.0
        return out

    def _nodes(self):
        for i, (_, row) in enumerate(self._data["suppliers"].iterrows()):
            nid = row["supplier_id"]
            self.id_maps.setdefault("supplier", {})[nid] = i
            self.G.add_node(nid, node_type="supplier", label=row["name"],
                country=row["country"], category=row["category"],
                risk_tier=row["risk_tier"],
                **self._safe(row, NODE_FEATURE_SPECS["supplier"]))
        log.info("  + %d supplier nodes", len(self._data["suppliers"]))

        for i, (_, row) in enumerate(self._data["components"].iterrows()):
            nid = row["component_id"]
            self.id_maps.setdefault("component", {})[nid] = i
            feats = self._safe(row, ["unit_cost_usd","annual_volume","lead_time_weeks",
                                     "inventory_days","weight_kg","is_custom"])
            feats["criticality_encoded"] = CRITICALITY_MAP.get(str(row.get("criticality","Low")), 0)
            self.G.add_node(nid, node_type="component", label=row["part_number"],
                category=row["category"], criticality=row["criticality"], **feats)
        log.info("  + %d component nodes", len(self._data["components"]))

        self.id_maps["_country_by_name"] = {}
        for i, (_, row) in enumerate(self._data["countries"].iterrows()):
            nid = row["country_id"]
            self.id_maps.setdefault("country", {})[nid] = i
            self.id_maps["_country_by_name"][row["name"]] = nid
            self.G.add_node(nid, node_type="country", label=row["name"],
                **self._safe(row, NODE_FEATURE_SPECS["country"]))
        log.info("  + %d country nodes", len(self._data["countries"]))

        for i, (_, row) in enumerate(self._data["contracts"].iterrows()):
            nid = row["contract_id"]
            self.id_maps.setdefault("contract", {})[nid] = i
            self.G.add_node(nid, node_type="contract", label=nid,
                status=row["status"], payment_terms=row["payment_terms"],
                **self._safe(row, NODE_FEATURE_SPECS["contract"]))
        log.info("  + %d contract nodes", len(self._data["contracts"]))

        for i, (_, row) in enumerate(self._data["routes"].iterrows()):
            nid = row["route_id"]
            self.id_maps.setdefault("route", {})[nid] = i
            self.G.add_node(nid, node_type="route",
                label=f"{row['origin_country']} -> {row['destination_country']}",
                transport_mode=row["transport_mode"],
                **self._safe(row, NODE_FEATURE_SPECS["route"]))
        log.info("  + %d route nodes", len(self._data["routes"]))

    def _edges(self):
        by_name = self.id_maps.get("_country_by_name", {})
        n = 0
        for _, row in self._data["sourcing_links"].iterrows():
            s, c = row["supplier_id"], row["component_id"]
            if s in self.G and c in self.G:
                self.G.add_edge(s, c, relation="supplies",
                    is_primary=int(row["is_primary"]),
                    unit_price_usd=float(row["unit_price_usd"]))
                n += 1
        log.info("  + %d supplies edges", n)

        n = 0
        for _, row in self._data["suppliers"].iterrows():
            sid  = row["supplier_id"]
            cnid = by_name.get(row["country"])
            if sid in self.G and cnid and cnid in self.G:
                self.G.add_edge(sid, cnid, relation="located_in"); n += 1
        log.info("  + %d located_in edges", n)

        cc = sc = 0
        for _, row in self._data["contracts"].iterrows():
            cid, sid, cmp = row["contract_id"], row["supplier_id"], row["primary_component_id"]
            if cid in self.G and cmp in self.G:
                self.G.add_edge(cid, cmp, relation="covers"); cc += 1
            if cid in self.G and sid in self.G:
                self.G.add_edge(cid, sid, relation="signed_with"); sc += 1
        log.info("  + %d covers | %d signed_with", cc, sc)

        oc = dc = car = 0
        for _, row in self._data["routes"].iterrows():
            rid  = row["route_id"]
            ocid = by_name.get(row["origin_country"])
            dcid = by_name.get(row["destination_country"])
            cmp  = row["component_id"]
            if rid in self.G:
                if ocid and ocid in self.G:
                    self.G.add_edge(rid, ocid, relation="originates_in"); oc += 1
                if dcid and dcid in self.G:
                    self.G.add_edge(rid, dcid, relation="delivers_to"); dc += 1
                if cmp in self.G:
                    self.G.add_edge(rid, cmp, relation="carries"); car += 1
        log.info("  + %d originates_in | %d delivers_to | %d carries", oc, dc, car)

        comp_to_sups = {}
        for _, row in self._data["sourcing_links"].iterrows():
            comp_to_sups.setdefault(row["component_id"], []).append(row["supplier_id"])
        co = set()
        for sups in comp_to_sups.values():
            for i in range(len(sups)):
                for j in range(i+1, len(sups)):
                    co.add(tuple(sorted([sups[i], sups[j]])))
        cn = 0
        for s1, s2 in co:
            if s1 in self.G and s2 in self.G:
                self.G.add_edge(s1, s2, relation="co_supplier")
                self.G.add_edge(s2, s1, relation="co_supplier")
                cn += 2

        tw = set()
        for _, row in self._data["routes"].iterrows():
            oc2 = by_name.get(row["origin_country"])
            dc2 = by_name.get(row["destination_country"])
            if oc2 and dc2: tw.add((oc2, dc2))
        tn = 0
        for c1, c2 in tw:
            if c1 in self.G and c2 in self.G:
                self.G.add_edge(c1, c2, relation="trades_with"); tn += 1
        log.info("  + %d co_supplier | %d trades_with (derived)", cn, tn)

    def _features(self):
        for ntype, feat_names in NODE_FEATURE_SPECS.items():
            nodes = [(n, d) for n, d in self.G.nodes(data=True)
                     if d.get("node_type") == ntype]
            if not nodes: continue
            matrix = np.array(
                [[float(d.get(f, 0.0) or 0.0) for f in feat_names] for _, d in nodes],
                dtype=np.float32)
            mins = matrix.min(0); maxs = matrix.max(0)
            rng  = np.where(maxs - mins == 0, 1.0, maxs - mins)
            self.node_features[ntype] = (matrix - mins) / rng
            log.info("  Feature [%-12s]: %s", ntype, self.node_features[ntype].shape)

    def _to_pyg(self):
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            log.warning("torch_geometric not installed — skipping PyG export")
            return None
        data = HeteroData()
        for ntype, feat in self.node_features.items():
            data[ntype].x = torch.tensor(feat, dtype=torch.float)
        edge_defs = [
            ("supplies",      "supplier", "component"),
            ("located_in",    "supplier", "country"),
            ("covers",        "contract", "component"),
            ("signed_with",   "contract", "supplier"),
            ("originates_in", "route",    "country"),
            ("delivers_to",   "route",    "country"),
            ("carries",       "route",    "component"),
            ("co_supplier",   "supplier", "supplier"),
            ("trades_with",   "country",  "country"),
        ]
        for rel, src_t, dst_t in edge_defs:
            sm = self.id_maps.get(src_t, {})
            dm = self.id_maps.get(dst_t, {})
            edges = [(sm[u], dm[v]) for u, v, d in self.G.edges(data=True)
                     if d.get("relation") == rel and u in sm and v in dm]
            if edges:
                ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[src_t, rel, dst_t].edge_index = ei
        return data

    def _save(self):
        G2 = self.G.copy()
        for _, d in G2.nodes(data=True):
            for k, v in list(d.items()):
                if isinstance(v, (bool, np.bool_)): d[k] = int(v)
        for _, _, d in G2.edges(data=True):
            for k, v in list(d.items()):
                if isinstance(v, (bool, np.bool_)): d[k] = int(v)
        gml = self.output_dir / "procurement_kg.graphml"
        nx.write_graphml(G2, str(gml))
        log.info("  -> %s", gml)
        if self.hetero is not None:
            torch.save(self.hetero, self.output_dir / "procurement_kg.pt")
            log.info("  -> procurement_kg.pt")
        maps = {k: {str(kk): int(vv) for kk, vv in v.items()}
                for k, v in self.id_maps.items() if not k.startswith("_")}
        with open(self.output_dir / "id_maps.json", "w") as f:
            json.dump(maps, f, indent=2)
        s = self.stats()
        with open(self.output_dir / "kg_stats.json", "w") as f:
            json.dump({"n_nodes": s.n_nodes, "n_edges": s.n_edges,
                       "total_nodes": s.total_nodes, "total_edges": s.total_edges,
                       "density": s.density}, f, indent=2)

    def stats(self):
        s = GraphStats()
        for nt in ["supplier","component","country","contract","route"]:
            s.n_nodes[nt] = sum(1 for _, d in self.G.nodes(data=True)
                                if d.get("node_type") == nt)
        for rel in ["supplies","located_in","covers","signed_with","originates_in",
                    "delivers_to","carries","co_supplier","trades_with"]:
            s.n_edges[rel] = sum(1 for _, _, d in self.G.edges(data=True)
                                 if d.get("relation") == rel)
        s.total_nodes = self.G.number_of_nodes()
        s.total_edges = self.G.number_of_edges()
        n = s.total_nodes
        s.density = s.total_edges / (n * (n - 1)) if n > 1 else 0.0
        return s
