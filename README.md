# ProcureIQ

[![CI Pipeline](https://github.com/Shruti-Dhamdhere/procure-iq/actions/workflows/ci.yml/badge.svg)](https://github.com/Shruti-Dhamdhere/procure-iq/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An Agentic AI + Graph Neural Network platform for procurement intelligence.
> Claims management, cost optimization, and sourcing lineage — powered by
> a supplier knowledge graph and multi-agent reasoning.

---

## The Problem

Procurement teams at mid-market manufacturers face four critical challenges:

- **Supplier switching** is slow and manual when suppliers fail or policies change
- **Claims management** is document-heavy, poorly tracked, and takes too long to close
- **Cost savings** opportunities are invisible without systematic price benchmarking
- **Sourcing lineage** — true landed cost including packaging and shipping — is unknown

Enterprise tools like SAP Ariba solve this for Fortune 500 companies at $500K+/year.
Nothing good exists for mid-market companies running on Excel and internal ERP tools.

---

## Solution

ProcureIQ combines two novel layers:

**GNN Layer** — A heterogeneous supplier knowledge graph where nodes are suppliers,
components, countries, contracts, and routes. GraphSAGE + GAT learn embeddings
that capture supply network relationships, risk propagation, and price anomalies.

**Agent Layer** — A multi-agent system built on LangGraph + Claude that uses GNN
embeddings as its knowledge base to reason, draft documents, and recommend actions.

---

## Novel Contributions

1. First system combining heterogeneous supplier knowledge graph with multi-agent
   procurement reasoning in a unified framework
2. GNN-based price anomaly detection across supplier networks
3. Causal disruption propagation simulation for what-if analysis
4. Active learning from procurement decisions improving graph embeddings over time

---

## Architecture
```
USASpending.gov + Synthetic Overlay
            |
    Supplier Knowledge Graph
    (Suppliers, Components, Countries, Contracts, Routes)
            |
    GNN Layer (GraphSAGE + GAT)
    - Supplier embeddings
    - Price anomaly scores
    - Risk propagation
            |
    Multi-Agent Layer (LangGraph + Claude)
    - Claims Agent
    - Cost Savings Agent
    - Sourcing Lineage Agent
    - Orchestrator
            |
        FastAPI
        /     \
  Chat UI   Dashboard
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Knowledge Graph | NetworkX + PyTorch Geometric |
| GNN | GraphSAGE + GAT |
| Agent Framework | LangGraph |
| LLM | Claude (Anthropic) |
| Backend | FastAPI |
| Frontend | React + Tailwind |
| Dashboard | Plotly Dash |
| Experiment Tracking | MLflow |
| CI/CD | GitHub Actions |

---

## Paper

**Target:** Expert Systems with Applications (Elsevier, IF 8.5)

**Title:** ProcureIQ: A Graph-Augmented Multi-Agent Framework for Procurement
Intelligence — Claims Management, Cost Optimization, and Sourcing Lineage

---

## License

MIT License
