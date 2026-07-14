"""
Benchmark (single-model, SUPERSEDED): Pyomo vs gurobipy-pandas build time
==========================================================================

Kept for reference.  Use benchmark_comprehensive.py instead: it compares four
pipelines (Pyomo persistent, Pyomo+templatization, gurobipy-pandas, COO+MVar)
across three problem families under a GC-controlled measurement protocol.

Model: Capacitated multi-commodity flow (example_3_multiflow pattern)
  - |E| × |K| variables
  - |E| capacity constraints  (P1 groupby pattern)
  - |N| × |K| flow-balance constraints (P3 indexed-set subtraction pattern)

The translator converts the Pyomo function to gurobipy-pandas source once,
offline.  The timed sections measure only model-construction cost.

Tune N_NODES / N_EDGES / N_COMMODITIES to hit the target Pyomo build time.
Calibration on a typical laptop (M-series Mac, Python 3.12):
  ~11 000 constraints/second (rule eval + gurobi_persistent compilation)
  → 660 000 constraints ≈ 60 s
  → N_NODES=7000, N_EDGES=56000, N_COMMODITIES=50 gives ~420 000 constrs
"""

import random
import time
import sys

import gurobipy as gp
from pyomo.environ import SolverFactory

from translator import translate
import examples.example_3_multiflow as ex3

# ---------------------------------------------------------------------------
# Scale parameters — adjust to get Pyomo build time ~60 s on your machine.
# Constraints ≈ N_NODES × N_COMMODITIES  +  N_EDGES
# Variables   ≈ N_EDGES × N_COMMODITIES
# ---------------------------------------------------------------------------
N_NODES       = 7_000
N_EDGES       = 56_000   # ≈ 8 × N_NODES  (avg degree 16)
N_COMMODITIES = 50
SEED          = 42


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(n_nodes, n_edges, n_commodities, seed):
    rng = random.Random(seed)
    nodes       = [f"N{i:05d}" for i in range(n_nodes)]
    commodities = [f"K{k}"    for k  in range(n_commodities)]

    # Sample unique directed edges without the slow while-loop rejection method:
    # enumerate all possible (i, j) pairs and sample from them.
    all_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    # Directed edges: treat each undirected pair as two directed edges, then sample
    # half of n_edges undirected pairs → n_edges directed edges.
    if n_edges > 2 * len(all_pairs):
        raise ValueError("n_edges exceeds the number of possible directed edges")
    sampled = rng.sample(all_pairs, n_edges // 2)
    edges = [(nodes[i], nodes[j]) for i, j in sampled] + \
            [(nodes[j], nodes[i]) for i, j in sampled]

    capacity = {e: rng.randint(100, 1_000) for e in edges}

    # Keep only nodes that appear in at least one edge.
    # Isolated nodes would cause Pyomo to see `0 == 0` (trivial True) and crash.
    nodes = sorted({n for e in edges for n in e})

    # Sparse supply/demand: only ~5 % of (node, commodity) pairs are non-zero.
    # preprocess_data fills missing pairs with 0.
    demand: dict = {}
    n_sources = max(1, len(nodes) // 20)
    for k in commodities:
        sources = rng.sample(nodes, n_sources)
        sinks   = rng.sample(nodes, n_sources)
        total_supply = 0
        for n in sources:
            amt = rng.randint(10, 100)
            demand[(n, k)] = amt
            total_supply  += amt
        per_sink = total_supply // n_sources
        for n in sinks:
            demand[(n, k)] = -per_sink   # balanced enough for feasibility

    raw = {
        "Nodes":       nodes,
        "Commodities": commodities,
        "Edges":       edges,
        "Capacity":    capacity,
        "Demand":      demand,
    }
    return ex3.preprocess_data(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{'─' * 58}")
    print(f"  {title}")
    print(f"{'─' * 58}")


def model_stats(gurobi_m, label):
    gurobi_m.update()
    print(f"  {label:30s}  vars={gurobi_m.NumVars:>9,}  "
          f"constrs={gurobi_m.NumConstrs:>9,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- 0. Translate once (offline, not timed in the benchmark) -----------
    section("Translation  (one-time offline cost)")
    t0 = time.perf_counter()
    translated_src = translate(ex3.build_pyomo_model)
    _ns: dict = {}
    exec(compile(translated_src, "<translated>", "exec"), _ns)
    build_vectorized_model = _ns["build_vectorized_model"]
    translation_ms = (time.perf_counter() - t0) * 1_000
    print(f"  translate() completed in {translation_ms:.1f} ms")

    # --- 1. Generate data --------------------------------------------------
    section("Data generation")
    print(f"  N_NODES={N_NODES:,}  N_EDGES={N_EDGES:,}  N_COMMODITIES={N_COMMODITIES}")
    t0 = time.perf_counter()
    data = generate_data(N_NODES, N_EDGES, N_COMMODITIES, SEED)
    data_ms = (time.perf_counter() - t0) * 1_000
    n_vars    = len(data["Edges"]) * N_COMMODITIES
    n_constrs = len(data["Nodes"]) * N_COMMODITIES + len(data["Edges"])
    print(f"  Data ready in {data_ms:.0f} ms")
    print(f"  Expected model size: {n_vars:,} variables, ~{n_constrs:,} constraints")

    # --- 2. Warm up Gurobi (eat license-check overhead) --------------------
    section("Warming up Gurobi")
    _ = gp.Model()
    print("  Done.")

    # --- 3. Pyomo build ----------------------------------------------------
    section("PYOMO  —  Python-loop rule evaluation + Gurobi compilation")
    t0 = time.perf_counter()

    pyomo_model = ex3.build_pyomo_model(data)

    t_rule = time.perf_counter()
    opt = SolverFactory("gurobi_persistent")
    opt.set_instance(pyomo_model)
    t1 = time.perf_counter()

    pyomo_rule_s    = t_rule - t0
    pyomo_compile_s = t1 - t_rule
    pyomo_total_s   = t1 - t0

    gurobi_from_pyomo = opt._solver_model
    model_stats(gurobi_from_pyomo, "Pyomo model")
    print(f"  Rule evaluation:     {pyomo_rule_s:6.2f} s")
    print(f"  Gurobi compilation:  {pyomo_compile_s:6.2f} s")
    print(f"  Total build time:    {pyomo_total_s:6.2f} s")

    # --- 4. gurobipy-pandas build -----------------------------------------
    section("PANDAS  —  gurobipy-pandas vectorized build")
    t0 = time.perf_counter()
    gppd_model = build_vectorized_model(data)
    gppd_model.update()
    t1 = time.perf_counter()
    gppd_total_s = t1 - t0

    model_stats(gppd_model, "Translated model")
    print(f"  Total build time:    {gppd_total_s:6.2f} s")

    # --- 5. Results --------------------------------------------------------
    section("RESULTS")
    speedup = pyomo_total_s / gppd_total_s
    match = (gurobi_from_pyomo.NumVars    == gppd_model.NumVars and
             gurobi_from_pyomo.NumConstrs == gppd_model.NumConstrs)
    print(f"  Pyomo  build time:  {pyomo_total_s:6.2f} s")
    print(f"  Pandas build time:  {gppd_total_s:6.2f} s")
    print(f"  Speedup:            {speedup:6.1f}×")
    print(f"  Models identical:   {'YES ✓' if match else 'NO ✗  — check constraint counts above'}")
    print()
    if not match:
        sys.exit(1)
