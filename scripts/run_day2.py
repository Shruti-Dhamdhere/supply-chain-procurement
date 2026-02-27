import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    t0 = time.perf_counter()
    print("=" * 55)
    print(" ProcureIQ  |  Day 2: Knowledge Graph Construction")
    print("=" * 55)

    print("\n[1/3]  Generating synthetic data ...")
    from src.data.generator import generate_all
    datasets = generate_all()
    print({k: len(v) for k, v in datasets.items()})

    print("\n[2/3]  Building knowledge graph ...")
    from src.graph.builder import KnowledgeGraphBuilder
    builder = KnowledgeGraphBuilder()
    builder.build()
    print(builder.stats().summary())

    print("\n[3/3]  Generating visualizations ...")
    from src.graph.visualizer import run_all
    run_all()

    print(f"\nDone in {time.perf_counter()-t0:.1f}s")
    print("  data/raw/synthetic/          <- CSVs")
    print("  data/processed/              <- GraphML + PyG .pt")
    print("  outputs/visualizations/      <- PNG plots")
    print("\nNext:")
    print('  git add src/ params.yaml scripts/')
    print('  git commit -m "feat(graph): Day 2 — knowledge graph construction"')

if __name__ == "__main__":
    main()
