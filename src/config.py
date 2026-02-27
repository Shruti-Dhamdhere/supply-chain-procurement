from pathlib import Path
import yaml

_ROOT = Path(__file__).parent.parent

with open(_ROOT / "params.yaml") as f:
    _params = yaml.safe_load(f)

cfg           = _params
data_cfg      = _params.get("data", {})
synthetic_cfg = _params.get("synthetic", {})
graph_cfg     = _params.get("graph", {})
gnn_cfg       = _params.get("gnn", {})
agent_cfg     = _params.get("agents", {})
api_cfg       = _params.get("api", {})

RAW_DIR       = _ROOT / "data/raw"
PROCESSED_DIR = _ROOT / "data/processed"
SYNTHETIC_DIR = _ROOT / "data/raw/synthetic"

for _d in [RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
