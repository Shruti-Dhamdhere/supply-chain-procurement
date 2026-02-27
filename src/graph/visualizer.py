import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

PALETTE = {"supplier":"#4C72B0","component":"#55A868","country":"#C44E52",
           "contract":"#8172B2","route":"#CCB974"}
SIZE    = {"supplier":120,"component":80,"country":160,"contract":60,"route":90}
plt.rcParams.update({"font.family":"DejaVu Sans","figure.dpi":150})

def _colors(G, nodes): return [PALETTE.get(G.nodes[n].get("node_type",""),"#BBBBBB") for n in nodes]
def _sizes(G, nodes):  return [SIZE.get(G.nodes[n].get("node_type",""),60) for n in nodes]
def _patches(): return [mpatches.Patch(color=c, label=t.title()) for t, c in PALETTE.items()]

def plot_full_graph(G, path):
    random.seed(42)
    quota = {"supplier":28,"component":20,"country":18,"contract":30,"route":24}
    sample = set()
    for nt, q in quota.items():
        pool = [n for n, d in G.nodes(data=True) if d.get("node_type") == nt]
        sample.update(random.sample(pool, min(q, len(pool))))
    H = G.subgraph(sample).copy()
    pos = nx.spring_layout(H, k=3.0, seed=42, iterations=70)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#F7F8FA"); fig.patch.set_facecolor("#F7F8FA")
    nl = list(H.nodes)
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.12, arrows=True,
        arrowstyle="-|>", arrowsize=7, edge_color="#AAAAAA", width=0.5)
    nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=nl,
        node_color=_colors(H, nl), node_size=_sizes(H, nl), alpha=0.88)
    ax.legend(handles=_patches(), loc="upper left", fontsize=9, framealpha=0.9, ncol=2)
    ax.set_title(f"ProcureIQ — Procurement Knowledge Graph\n"
                 f"({H.number_of_nodes()} nodes · {H.number_of_edges()} edges · sampled)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.axis("off"); plt.tight_layout()
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  saved {path}")

def plot_supplier_ego(G, path):
    sups = [(n, d) for n, d in G.nodes(data=True) if d.get("node_type") == "supplier"]
    if not sups: return
    center = max(sups, key=lambda x: float(x[1].get("financial_risk_score", 0)))[0]
    ego = nx.ego_graph(G.to_undirected(), center, radius=2)
    pos = nx.spring_layout(ego, k=3.5, seed=17, iterations=80)
    nl = list(ego.nodes)
    colors = ["#FF4500" if n == center else PALETTE.get(ego.nodes[n].get("node_type",""),"#BBBBBB") for n in nl]
    sizes  = [400 if n == center else SIZE.get(ego.nodes[n].get("node_type",""),60) for n in nl]
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_facecolor("#F7F8FA"); fig.patch.set_facecolor("#F7F8FA")
    nx.draw_networkx_edges(ego, pos, ax=ax, alpha=0.25, edge_color="#888888", width=0.8, arrows=False)
    nx.draw_networkx_nodes(ego, pos, ax=ax, nodelist=nl, node_color=colors, node_size=sizes, alpha=0.9)
    nx.draw_networkx_labels(ego, pos, ax=ax,
        labels={center: G.nodes[center].get("label", center)[:25]},
        font_size=8, font_weight="bold")
    patches = _patches() + [mpatches.Patch(color="#FF4500", label="High-Risk Supplier")]
    ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(f"2-Hop Ego Graph — Highest-Risk Supplier\n"
                 f"{ego.number_of_nodes()} nodes · {ego.number_of_edges()} edges",
                 fontsize=12, fontweight="bold", pad=12)
    ax.axis("off"); plt.tight_layout()
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  saved {path}")

def plot_degree_distributions(G, path):
    node_types = ["supplier","component","country","contract","route"]
    G_ud = G.to_undirected()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Degree Distribution by Node Type", fontsize=14, fontweight="bold")
    for ax, nt in zip(axes.flat, node_types):
        nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == nt]
        degs  = np.array([G_ud.degree(n) for n in nodes])
        if not len(degs): ax.set_visible(False); continue
        ax.hist(degs, bins=min(20, len(set(degs))),
                color=PALETTE.get(nt,"#888"), alpha=0.8, edgecolor="white")
        ax.axvline(degs.mean(), color="black", linestyle="--", linewidth=1.3,
                   label=f"mean={degs.mean():.1f}")
        ax.set_title(nt.title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Degree"); ax.set_ylabel("Count"); ax.legend(fontsize=8)
    axes.flat[-1].set_visible(False)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  saved {path}")

def run_all(graphml_path="data/processed/procurement_kg.graphml",
            output_dir="outputs/visualizations"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Loading {graphml_path} ...")
    G = nx.read_graphml(graphml_path)
    print(f"  {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")
    plot_full_graph(G,            f"{output_dir}/01_full_graph.png")
    plot_supplier_ego(G,          f"{output_dir}/02_supplier_ego.png")
    plot_degree_distributions(G,  f"{output_dir}/03_degree_distributions.png")
    print(f"Done. Visuals in {output_dir}/")

if __name__ == "__main__":
    run_all()
