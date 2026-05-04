"""
Comprehensive build-time benchmark
===================================
Compares three model-construction paths across three problem types.
Sizes are chosen so Pyomo starts at ~1 minute and grows to ~15 minutes.
Expected total wall-clock time: 1.5 – 2.5 hours.

  ① Pyomo               — ConcreteModel + rule evaluation + gurobi_persistent
  ② gurobipy-pandas     — translator-old.translate() → gppd.add_vars / add_constrs
  ③ COO / MVar          — translator.translate()     → addMVar + addMConstr

All three methods are derived from the same Pyomo build function via translate().

Output
------
  benchmark_results.png  — three-panel log-log chart (minutes on y-axis)
  benchmark_results.csv  — raw timing data
"""

from __future__ import annotations

import csv
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable

import gurobipy as gp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pyomo.environ import SolverFactory
import pyomo.environ as pyo

import translator     as _new_tr   # COO + MVar
import importlib.util

def _load_old_translator():
    spec = importlib.util.spec_from_file_location(
        "translator_old",
        "/Users/michaelkhalfin/Repositories/math-prog-build-time/translator-old.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["translator_old"] = mod
    spec.loader.exec_module(mod)
    return mod

_old_tr = _load_old_translator()

PYOMO_TIMEOUT_MIN = 25.0   # skip remaining Pyomo sizes once one exceeds this


# ===========================================================================
# Helper: compile a translated builder
# ===========================================================================

def _make_builder(pyomo_fn: Callable, translator_mod) -> Callable:
    code = translator_mod.translate(pyomo_fn)
    ns: dict = {}
    exec(compile(code, f"<translated:{pyomo_fn.__name__}>", "exec"), ns)
    fn = ns["build_vectorized_model"]
    def wrapper(data):
        m = fn(data)
        m.update()
        return m
    return wrapper


def _timed(fn: Callable, data: dict) -> tuple[object, float]:
    t0 = time.perf_counter()
    result = fn(data)
    return result, time.perf_counter() - t0


# ===========================================================================
# PROBLEM A — Supply-Demand  (P1)
# Variables:  x[I × J]         n_i * n_j
# Constrs:    sum_j x[i,j] <= Supply[i]  (n_i),  sum_i x[i,j] >= Demand[j]  (n_j)
# ===========================================================================

def gen_supply_demand(n_i: int, n_j: int, seed: int = 1) -> dict:
    rng = random.Random(seed)
    I = [f"I{k}" for k in range(n_i)]
    J = [f"J{k}" for k in range(n_j)]
    return {
        "I": I, "J": J,
        "Supply": {i: rng.uniform(50, 200) for i in I},
        "Demand": {j: rng.uniform(20, 80)  for j in J},
    }


def sd_pyomo_build(data: dict):
    m = pyo.ConcreteModel()
    m.I      = pyo.Set(initialize=data["I"])
    m.J      = pyo.Set(initialize=data["J"])
    m.Supply = pyo.Param(m.I, initialize=data["Supply"])
    m.Demand = pyo.Param(m.J, initialize=data["Demand"])
    m.x      = pyo.Var(m.I, m.J, domain=pyo.NonNegativeReals)

    def supply_rule(m, i):
        return sum(m.x[i, j] for j in m.J) <= m.Supply[i]
    m.supply_constr = pyo.Constraint(m.I, rule=supply_rule)

    def demand_rule(m, j):
        return sum(m.x[i, j] for i in m.I) >= m.Demand[j]
    m.demand_constr = pyo.Constraint(m.J, rule=demand_rule)

    opt = SolverFactory("gurobi_persistent")
    opt.set_instance(m)
    return opt._solver_model


# Calibrated so Pyomo ≈ 1 min at the first point, ~15 min at the last.
# Scaling is O(n²) in vars: time ∝ n_i × n_j.
SD_SIZES = [
    ("2300²",  dict(n_i=2300, n_j=2300, seed=1)),   #  ~5.3 M vars  → Pyomo ~1.0 min
    ("3300²",  dict(n_i=3300, n_j=3300, seed=1)),   # ~10.9 M vars  → Pyomo ~2.1 min
    ("4700²",  dict(n_i=4700, n_j=4700, seed=1)),   # ~22.1 M vars  → Pyomo ~4.3 min
    ("6500²",  dict(n_i=6500, n_j=6500, seed=1)),   # ~42.3 M vars  → Pyomo ~8.2 min
    ("9000²",  dict(n_i=9000, n_j=9000, seed=1)),   # ~81.0 M vars  → Pyomo ~15.8 min
]


# ===========================================================================
# PROBLEM B — Multi-commodity Network Flow  (P1 + P3)
# Variables:  x[Edges × K]      n_edges * n_k
# Constrs:    cap  (n_edges),  flow balance  (n_active_nodes × n_k)
# ===========================================================================

def gen_network_flow(n_nodes: int, avg_degree: int, n_k: int, seed: int = 2) -> dict:
    rng    = random.Random(seed)
    nodes  = [f"N{v:06d}" for v in range(n_nodes)]
    n_half = (n_nodes * avg_degree) // 2

    # O(n_half) random sampling — avoids O(n_nodes²) pair enumeration
    edges_set: set = set()
    while len(edges_set) < n_half:
        i = rng.randrange(n_nodes)
        j = rng.randrange(n_nodes)
        if i != j:
            edges_set.add((min(i, j), max(i, j)))

    sampled = list(edges_set)
    edges = [(nodes[i], nodes[j]) for i, j in sampled] + \
            [(nodes[j], nodes[i]) for i, j in sampled]
    K = [f"K{k}" for k in range(n_k)]

    out_arcs: dict = {n: [] for n in nodes}
    in_arcs:  dict = {n: [] for n in nodes}
    for ni, nj in edges:
        out_arcs[ni].append(nj)
        in_arcs[nj].append(ni)

    active = sorted({n for e in edges for n in e})
    cap    = {e: float(rng.randint(50, 500)) for e in edges}

    demand: dict = {(n, k): 0.0 for n in active for k in K}
    n_src = max(1, len(active) // 10)
    for k in K:
        for s in rng.sample(active, n_src): demand[(s, k)] += rng.randint(10, 50)
        for t in rng.sample(active, n_src): demand[(t, k)] -= rng.randint(10, 50)

    return {
        "Nodes": active, "Commodities": K, "Edges": edges,
        "Capacity": cap, "Demand": demand,
        "OutArcs": {n: out_arcs[n] for n in active},
        "InArcs":  {n: in_arcs[n]  for n in active},
    }


def nf_pyomo_build(data: dict):
    m = pyo.ConcreteModel()
    m.N       = pyo.Set(initialize=data["Nodes"])
    m.K       = pyo.Set(initialize=data["Commodities"])
    m.E       = pyo.Set(dimen=2, initialize=data["Edges"])
    m.OutArcs = pyo.Set(m.N, initialize=data["OutArcs"])
    m.InArcs  = pyo.Set(m.N, initialize=data["InArcs"])
    m.Cap     = pyo.Param(m.E, initialize=data["Capacity"])
    m.Dem     = pyo.Param(m.N, m.K, initialize=data["Demand"])
    m.x       = pyo.Var(m.E, m.K, domain=pyo.NonNegativeReals)

    def cap_rule(m, i, j):
        return sum(m.x[i, j, k] for k in m.K) <= m.Cap[i, j]
    m.cap_constr = pyo.Constraint(m.E, rule=cap_rule)

    def flow_rule(m, node, k):
        flow_out = sum(m.x[node, j, k] for j in m.OutArcs[node])
        flow_in  = sum(m.x[i, node, k] for i in m.InArcs[node])
        return flow_out - flow_in == m.Dem[node, k]
    m.flow_constr = pyo.Constraint(m.N, m.K, rule=flow_rule)

    opt = SolverFactory("gurobi_persistent")
    opt.set_instance(m)
    return opt._solver_model


# n_vars ≈ n_nodes * avg_degree * n_k  (directed edges × commodities)
NF_SIZES = [
    ("n5000,k175",   dict(n_nodes=5000,  avg_degree=6, n_k=175, seed=2)),  # ~5.3 M vars  → Pyomo ~1.5 min
    ("n7500,k225",   dict(n_nodes=7500,  avg_degree=6, n_k=225, seed=2)),  # ~10.1 M vars → Pyomo ~2.8 min
    ("n11000,k285",  dict(n_nodes=11000, avg_degree=6, n_k=285, seed=2)),  # ~18.8 M vars → Pyomo ~5.2 min
    ("n16000,k360",  dict(n_nodes=16000, avg_degree=6, n_k=360, seed=2)),  # ~34.6 M vars → Pyomo ~9.6 min
    ("n22000,k430",  dict(n_nodes=22000, avg_degree=6, n_k=430, seed=2)),  # ~56.8 M vars → Pyomo ~15.8 min
]


# ===========================================================================
# PROBLEM C — BOM / Cross-dimensional  (P4 + P6)
# Variables:  build[P × T],  buy[C × T]     (n_p + n_c) * n_t
# Constrs:    cap   build[p,t] <= Cap[p,t]         (P × T)
#             comp  buy[c,t] == sum_p BOM[p,c]*build[p,t]  (C × T)
# ===========================================================================

def gen_bom(n_p: int, n_c: int, n_t: int, avg_uses: int = 3, seed: int = 3) -> dict:
    rng = random.Random(seed)
    P = [f"P{k}" for k in range(n_p)]
    C = [f"C{k}" for k in range(n_c)]
    T = [f"T{t}" for t in range(n_t)]

    bom: dict = {}
    prods_using: dict = {c: [] for c in C}
    for p in P:
        for c in rng.sample(C, min(avg_uses, len(C))):
            bom[(p, c)] = rng.uniform(0.5, 3.0)
            prods_using[c].append(p)
    for c in C:
        if not prods_using[c]:
            p = rng.choice(P)
            bom[(p, c)] = 1.0
            prods_using[c].append(p)

    cap = {(p, t): rng.uniform(50, 500) for p in P for t in T}

    return {
        "Products": P, "Components": C, "Time": T,
        "BOM": bom, "Cap": cap, "ProdsUsingComp": prods_using,
    }


def bom_pyomo_build(data: dict):
    m = pyo.ConcreteModel()
    m.P              = pyo.Set(initialize=data["Products"])
    m.C              = pyo.Set(initialize=data["Components"])
    m.T              = pyo.Set(initialize=data["Time"])
    m.ProdsUsingComp = pyo.Set(m.C, initialize=data["ProdsUsingComp"])
    m.BOM            = pyo.Param(m.P, m.C, initialize=data["BOM"], default=0)
    m.Cap            = pyo.Param(m.P, m.T, initialize=data["Cap"])
    m.build          = pyo.Var(m.P, m.T, domain=pyo.NonNegativeReals)
    m.buy            = pyo.Var(m.C, m.T, domain=pyo.NonNegativeReals)

    def cap_rule(m, p, t):
        return m.build[p, t] <= m.Cap[p, t]
    m.cap_constr = pyo.Constraint(m.P, m.T, rule=cap_rule)

    def comp_rule(m, c, t):
        return m.buy[c, t] == sum(m.BOM[p, c] * m.build[p, t]
                                  for p in m.ProdsUsingComp[c])
    m.comp_constr = pyo.Constraint(m.C, m.T, rule=comp_rule)

    opt = SolverFactory("gurobi_persistent")
    opt.set_instance(m)
    return opt._solver_model


# n_vars = (n_p + n_c) * n_t
BOM_SIZES = [
    ("p10k,c20k,t80",  dict(n_p=10000, n_c=20000, n_t=80,  seed=3)),  #  2.4 M vars → Pyomo ~1.0 min
    ("p15k,c30k,t100", dict(n_p=15000, n_c=30000, n_t=100, seed=3)),  #  4.5 M vars → Pyomo ~1.9 min
    ("p22k,c44k,t120", dict(n_p=22000, n_c=44000, n_t=120, seed=3)),  #  7.9 M vars → Pyomo ~3.3 min
    ("p32k,c64k,t150", dict(n_p=32000, n_c=64000, n_t=150, seed=3)),  # 14.4 M vars → Pyomo ~6.0 min
    ("p45k,c90k,t180", dict(n_p=45000, n_c=90000, n_t=180, seed=3)),  # 24.3 M vars → Pyomo ~10.1 min
]


# ===========================================================================
# Problem registry
# ===========================================================================

@dataclass
class ProblemSpec:
    name:      str
    pyomo_fn:  Callable
    gen_fn:    Callable
    sizes:     list
    n_vars_fn: Callable


PROBLEMS = [
    ProblemSpec(
        name="Supply-Demand (P1)",
        pyomo_fn=sd_pyomo_build,
        gen_fn=gen_supply_demand,
        sizes=SD_SIZES,
        n_vars_fn=lambda n_i, n_j, seed: n_i * n_j,
    ),
    ProblemSpec(
        name="Network Flow (P1+P3)",
        pyomo_fn=nf_pyomo_build,
        gen_fn=gen_network_flow,
        sizes=NF_SIZES,
        n_vars_fn=lambda n_nodes, avg_degree, n_k, seed: n_nodes * avg_degree * n_k,
    ),
    ProblemSpec(
        name="BOM / Cross-dim (P4+P6)",
        pyomo_fn=bom_pyomo_build,
        gen_fn=gen_bom,
        sizes=BOM_SIZES,
        n_vars_fn=lambda n_p, n_c, n_t, seed: (n_p + n_c) * n_t,
    ),
]

METHODS = [
    ("Pyomo",           "#d62728"),
    ("gurobipy-pandas", "#1f77b4"),
    ("COO + MVar",      "#2ca02c"),
]


# ===========================================================================
# Benchmark runner
# ===========================================================================

def run_benchmark(spec: ProblemSpec) -> list[dict]:
    try:
        gppd_builder = _make_builder(spec.pyomo_fn, _old_tr)
    except Exception as e:
        print(f"  [WARN] gurobipy-pandas translation failed: {e}")
        gppd_builder = None

    try:
        mvar_builder = _make_builder(spec.pyomo_fn, _new_tr)
    except Exception as e:
        print(f"  [WARN] COO+MVar translation failed: {e}")
        mvar_builder = None

    builders = {
        "Pyomo":           spec.pyomo_fn,
        "gurobipy-pandas": gppd_builder,
        "COO + MVar":      mvar_builder,
    }

    rows = []
    pyomo_skipped = False

    for size_label, kwargs in spec.sizes:
        data   = spec.gen_fn(**kwargs)
        n_vars = spec.n_vars_fn(**kwargs)
        print(f"\n  size={size_label:<18s}  n_vars={n_vars:>12,}")

        for method_label, _ in METHODS:
            builder = builders[method_label]

            skip_reason = None
            if builder is None:
                skip_reason = "N/A"
            elif method_label == "Pyomo" and pyomo_skipped:
                skip_reason = "TIMEOUT"

            if skip_reason:
                print(f"    {method_label:<20s} {skip_reason}")
                rows.append({"problem": spec.name, "method": method_label,
                             "n_vars": n_vars, "build_s": float("nan"), "skipped": True})
                continue

            try:
                _, elapsed = _timed(builder, data)
                build_s = elapsed
            except Exception as e:
                print(f"    {method_label:<20s} ERROR: {e}")
                rows.append({"problem": spec.name, "method": method_label,
                             "n_vars": n_vars, "build_s": float("nan"), "skipped": True})
                continue

            if method_label == "Pyomo" and build_s / 60.0 > PYOMO_TIMEOUT_MIN:
                pyomo_skipped = True

            rows.append({"problem": spec.name, "method": method_label,
                         "n_vars": n_vars, "build_s": build_s, "skipped": False})
            print(f"    {method_label:<20s} {build_s / 60.0:>8.3f} min")

    return rows


# ===========================================================================
# Plotting  (y-axis in minutes, log-log)
# ===========================================================================

def plot_results(all_rows: list[dict], out_path: str = "benchmark_results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("Model Build Time vs. Number of Variables",
                 fontsize=14, fontweight="bold", y=1.01)

    colors  = {m: c for m, c in METHODS}
    markers = {"Pyomo": "o", "gurobipy-pandas": "s", "COO + MVar": "^"}
    ls_map  = {"Pyomo": "--", "gurobipy-pandas": "-.", "COO + MVar": "-"}

    for ax, spec in zip(axes, PROBLEMS):
        rows = [r for r in all_rows if r["problem"] == spec.name]
        for method_label, color in METHODS:
            pts = [
                (r["n_vars"], r["build_s"] / 60.0)
                for r in rows
                if r["method"] == method_label
                and not r.get("skipped", False)
                and not np.isnan(r["build_s"])
            ]
            if not pts:
                continue
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys,
                    color=color, linestyle=ls_map[method_label],
                    marker=markers[method_label], markersize=7,
                    linewidth=2, label=method_label)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of variables", fontsize=11)
        ax.set_ylabel("Build time  (minutes)", fontsize=11)
        ax.set_title(spec.name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{int(x):,}"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, _: f"{y:.3f}" if y < 0.01 else f"{y:.2f}" if y < 1 else f"{y:.1f}"))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    _ = gp.Model()   # eat licence-check overhead

    all_rows: list[dict] = []

    for spec in PROBLEMS:
        print(f"\n{'═' * 70}")
        print(f"  {spec.name}")
        print(f"{'═' * 70}")
        rows = run_benchmark(spec)
        all_rows.extend(rows)

    csv_path = "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "method", "n_vars", "build_s", "skipped"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n  CSV saved → {csv_path}")

    print(f"\n{'─' * 70}")
    print(f"  {'Problem':<30}  {'Method':<22}  {'n_vars':>12}  {'time (min)':>10}")
    print(f"{'─' * 70}")
    for r in all_rows:
        if r.get("skipped") or np.isnan(r["build_s"]):
            val = "SKIP"
        else:
            val = f"{r['build_s'] / 60.0:.3f}"
        print(f"  {r['problem']:<30}  {r['method']:<22}  {r['n_vars']:>12,}  {val:>10}")

    plot_results(all_rows)
