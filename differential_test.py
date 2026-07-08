"""
Data-parametric differential testing for the Pyomo -> Gurobi transpiler.
=======================================================================

For each formulation we build the Gurobi model two ways and check that the
two solver-level models are structurally identical:

    (a) Pyomo reference    :  build_pyomo_model(data) -> gurobi_persistent
    (b) Transpiler output  :  translate(build_pyomo_model) -> build_vectorized_model(data)

The two pipelines name variables and constraints differently (Pyomo emits
``x1, x2, ...`` / ``x7, x8``; the transpiler emits ``x[0]`` / ``supply_constr[1]``),
so a name-based comparison is impossible.  Instead we compare a
*permutation-invariant canonical fingerprint* of each model: a Weisfeiler-Leman
color refinement over the constraint matrix A, augmented with senses, RHS,
variable bounds / types, and the objective.  Two models receive the same
fingerprint iff they are identical up to a relabeling of rows and columns --
exactly the equivalence we want, and dramatically stronger than comparing
``NumVars`` / ``NumConstrs`` alone.

Two test families are run:

  1. DATA-PARAMETRIC (the paper's headline): the three canonical problems are
     each instantiated on many REDUCED datasets -- small sizes and several
     random seeds -- that preserve the full problem's indexing structure while
     keeping Pyomo's instantiation fast.  Agreement across every reduced
     instance makes coincidental correctness of the transpiler untenable.

  2. PATTERN COVERAGE: every module in examples/ (each exercising a distinct
     transpiler pattern, P1..P6 and the sum variants) is checked on its fixed
     instance.

Run:  python differential_test.py
Exit status is non-zero if any instance disagrees.
"""

from __future__ import annotations

import hashlib
import importlib
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from pyomo.environ import SolverFactory
from translator import translate

import benchmark_comprehensive as bench


# ===========================================================================
# Canonical fingerprint (Weisfeiler-Leman color refinement)
# ===========================================================================

def _hash(obj) -> int:
    """Stable 64-bit hash of a repr-able object (Python's hash() is salted)."""
    return int.from_bytes(
        hashlib.blake2b(repr(obj).encode(), digest_size=8).digest(), "big"
    )


@dataclass
class ModelData:
    A: object                # scipy csr matrix (m x n)
    lb: np.ndarray
    ub: np.ndarray
    vtype: list
    obj: np.ndarray
    sense: list
    rhs: np.ndarray
    model_sense: int         # 1 = minimize, -1 = maximize


def extract(model) -> ModelData:
    """Pull the full solver-level structure out of a gurobipy Model."""
    model.update()
    A = model.getA().tocsr()
    gvars = model.getVars()
    gcons = model.getConstrs()
    return ModelData(
        A=A,
        lb=np.array([v.LB for v in gvars], dtype=float),
        ub=np.array([v.UB for v in gvars], dtype=float),
        vtype=[v.VType for v in gvars],
        obj=np.array([v.Obj for v in gvars], dtype=float),
        sense=[c.Sense for c in gcons],
        rhs=np.array([c.RHS for c in gcons], dtype=float),
        model_sense=int(model.ModelSense),
    )


def fingerprint(model, rounds: int = 4, tol: int = 6) -> dict:
    """Permutation-invariant canonical fingerprint of a Gurobi model.

    Color refinement: seed each column with (bounds, type, objective) and each
    row with (sense, rhs), then iteratively re-hash every node against the
    sorted multiset of its neighbors' colors and the connecting coefficients.
    After a few rounds the multiset of stabilized colors is a strong isomorphism
    invariant.  We also carry cheap, independently permutation-invariant
    multisets (rhs, objective, sense histogram) so that a mismatch yields a
    readable diff rather than an opaque hash difference.

    Equality rows are compared up to global sign: ``a . x = b`` and
    ``-a . x = -b`` are the same constraint, and the two pipelines legitimately
    differ on which orientation they emit (e.g. ``buy - sum(...) = 0`` vs
    ``sum(...) - buy = 0``).  Rather than *pick* a sign (impossible to do
    consistently for a row whose coefficient multiset is symmetric under
    negation, such as ``buy - build = 0``), each equality row is encoded by the
    UNORDERED pair of its two sign-sides -- the variables with positive
    coefficient and the variables with negative coefficient, each described by
    their column colors and |coefficient|.  Global negation merely swaps the two
    sides, so the encoding is invariant by construction while remaining fully
    sensitive to the sign *pattern* (``a - b`` distinguishable from ``a + b``).
    Inequality rows keep their orientation, since negating one flips ``<=``/``>=``.
    Equality RHS enters as |rhs| (all equality RHS are 0 in these models).
    """
    md = extract(model)
    m, n = md.A.shape
    # `+ 0.0` normalizes -0.0 to +0.0: the transpiler's RHS arithmetic can yield
    # a negative zero where Pyomo yields a positive zero, and repr(-0.0) hashes
    # differently from repr(0.0).  Without this the two pipelines would appear to
    # disagree on any constraint whose canonical RHS is zero.
    rnd = lambda x: round(float(x), tol) + 0.0

    Acsr = md.A.tocsr()
    is_eq = [s == "=" for s in md.sense]

    # Per-row nonzero structure, split into sign-sides for equality rows.
    row_cols = []      # list of (col_index, coeff) for each row
    for i in range(m):
        s, e = Acsr.indptr[i], Acsr.indptr[i + 1]
        row_cols.append([(int(Acsr.indices[k]), rnd(Acsr.data[k])) for k in range(s, e)])

    col_color = [
        _hash(("C", rnd(md.lb[j]), rnd(md.ub[j]), md.vtype[j], rnd(md.obj[j])))
        for j in range(n)
    ]
    row_color = [
        _hash(("R", "=" if is_eq[i] else md.sense[i],
               abs(rnd(md.rhs[i])) if is_eq[i] else rnd(md.rhs[i])))
        for i in range(m)
    ]

    def sides(i):
        """(pos_side, neg_side) each a sorted list of (col_color, |coeff|)."""
        pos = sorted((col_color[j], abs(c)) for j, c in row_cols[i] if c > 0)
        neg = sorted((col_color[j], abs(c)) for j, c in row_cols[i] if c < 0)
        return pos, neg

    for _ in range(rounds):
        new_row = []
        for i in range(m):
            if is_eq[i]:
                pos, neg = sides(i)
                # unordered pair of sides -> invariant to global negation
                pair = tuple(sorted((tuple(pos), tuple(neg))))
                new_row.append(_hash((row_color[i], pair)))
            else:
                nbrs = sorted((col_color[j], c) for j, c in row_cols[i])
                new_row.append(_hash((row_color[i], tuple(nbrs))))

        col_nbrs = [[] for _ in range(n)]
        for i in range(m):
            if is_eq[i]:
                pos, neg = sides(i)
                pos_t, neg_t = tuple(pos), tuple(neg)
                for j, c in row_cols[i]:
                    # each column sees its own side and the opposite side, but not
                    # a global +/- label -> invariant to negation, structure kept
                    mine, other = (pos_t, neg_t) if c > 0 else (neg_t, pos_t)
                    col_nbrs[j].append((row_color[i], "=", abs(c), mine, other))
            else:
                for j, c in row_cols[i]:
                    col_nbrs[j].append((row_color[i], md.sense[i], c))
        new_col = [_hash((col_color[j], tuple(sorted(col_nbrs[j], key=repr)))) for j in range(n)]

        row_color, col_color = new_row, new_col

    # Readable multisets: |coeff| / |rhs| for equality rows (orientation-folded),
    # signed for inequality rows.
    coef_vals = []
    for i in range(m):
        for _j, c in row_cols[i]:
            coef_vals.append(abs(c) if is_eq[i] else c)
    rhs_vals = [abs(rnd(md.rhs[i])) if is_eq[i] else rnd(md.rhs[i]) for i in range(m)]

    return {
        "shape": (m, n),
        "nnz": int(md.A.nnz),
        "model_sense": md.model_sense,
        "coef_hist": tuple(sorted(Counter(coef_vals).items())),
        "rhs_hist": tuple(sorted(Counter(rhs_vals).items())),
        "obj_hist": tuple(sorted(Counter(round(float(v), tol) for v in md.obj).items())),
        "sense_hist": tuple(sorted(Counter(md.sense).items())),
        "bound_hist": tuple(sorted(Counter(
            (round(float(l), tol), round(float(u), tol), t)
            for l, u, t in zip(md.lb, md.ub, md.vtype)
        ).items())),
        "row_colors": tuple(sorted(Counter(row_color).items())),
        "col_colors": tuple(sorted(Counter(col_color).items())),
    }


def diff_fingerprints(a: dict, b: dict) -> list[str]:
    """Return a list of human-readable differences (empty if identical)."""
    labels = {
        "shape": "matrix shape (rows x cols)",
        "nnz": "number of nonzeros",
        "model_sense": "objective sense (1=min, -1=max)",
        "coef_hist": "constraint-coefficient multiset",
        "rhs_hist": "RHS multiset",
        "obj_hist": "objective-coefficient multiset",
        "sense_hist": "constraint-sense histogram",
        "bound_hist": "variable (lb, ub, type) multiset",
        "row_colors": "canonical row (constraint) structure",
        "col_colors": "canonical column (variable) structure",
    }
    diffs = []
    for key, label in labels.items():
        if a.get(key) != b.get(key):
            if key in ("shape", "nnz", "model_sense"):
                diffs.append(f"{label}: reference={a.get(key)} transpiled={b.get(key)}")
            else:
                diffs.append(label)
    return diffs


# ===========================================================================
# Build both pipelines
# ===========================================================================

def build_reference(model_fn: Callable, data: dict):
    """Pyomo -> gurobi_persistent (the trusted reference model)."""
    pm = model_fn(data)
    opt = SolverFactory("gurobi_persistent")
    opt.set_instance(pm)
    g = opt._solver_model
    g.update()
    return g


_builder_cache: dict[int, Callable] = {}


def build_transpiled(model_fn: Callable, data: dict):
    """translate(model_fn) -> build_vectorized_model(data)."""
    key = id(model_fn)
    if key not in _builder_cache:
        ns: dict = {}
        exec(compile(translate(model_fn), f"<transpiled:{model_fn.__name__}>", "exec"), ns)
        _builder_cache[key] = ns["build_vectorized_model"]
    g = _builder_cache[key](data)
    g.update()
    return g


def check_instance(model_fn: Callable, data: dict) -> tuple[bool, list[str]]:
    """Build both ways, compare fingerprints. Returns (ok, diffs)."""
    ref = fingerprint(build_reference(model_fn, data))
    tr = fingerprint(build_transpiled(model_fn, data))
    diffs = diff_fingerprints(ref, tr)
    return (not diffs), diffs


# ===========================================================================
# Test family 1 — data-parametric reduced instances of the canonical problems
# ===========================================================================

# Small sizes that preserve each problem's indexing structure but instantiate
# in milliseconds.  Multiple seeds per size guard against a transpiler that is
# only coincidentally correct on one data realization.
SD_GRID = [dict(n_i=n_i, n_j=n_j, seed=s)
           for (n_i, n_j) in [(3, 4), (6, 5), (10, 8), (15, 12)]
           for s in range(3)]

NF_GRID = [dict(n_nodes=nn, avg_degree=3, n_k=nk, seed=s)
           for (nn, nk) in [(6, 2), (10, 3), (16, 4)]
           for s in range(3)]

BOM_GRID = [dict(n_p=n_p, n_c=n_c, n_t=n_t, seed=s)
            for (n_p, n_c, n_t) in [(3, 5, 2), (6, 10, 3), (12, 20, 4)]
            for s in range(3)]

CANONICAL = [
    ("Supply-Demand (P1)",        bench.gen_supply_demand, bench.sd_pyomo_model,  SD_GRID),
    ("Network Flow (P1+P3)",      bench.gen_network_flow,  bench.nf_pyomo_model,  NF_GRID),
    ("Bill-of-Materials (P4+P6)", bench.gen_bom,           bench.bom_pyomo_model, BOM_GRID),
]


def _size_label(kwargs: dict) -> str:
    parts = [f"{k}={v}" for k, v in kwargs.items() if k != "seed"]
    return ", ".join(parts) + f", seed={kwargs['seed']}"


def run_data_parametric() -> list[str]:
    failures = []
    for name, gen_fn, model_fn, grid in CANONICAL:
        print(f"\n  {name}")
        for kwargs in grid:
            data = gen_fn(**kwargs)
            try:
                ok, diffs = check_instance(model_fn, data)
            except Exception as e:
                ok, diffs = False, [f"exception: {type(e).__name__}: {e}"]
            ref = build_reference(model_fn, data)
            tag = f"{_size_label(kwargs):<34s}"
            if ok:
                print(f"    [OK]   {tag}  ({ref.NumConstrs} constrs x {ref.NumVars} vars)")
            else:
                print(f"    [FAIL] {tag}")
                for d in diffs:
                    print(f"           - {d}")
                failures.append(f"{name} [{_size_label(kwargs)}]")
    return failures


# ===========================================================================
# Test family 2 — pattern coverage over examples/
# ===========================================================================

EXAMPLE_MODULES = [
    "example_1_supply", "example_2_network", "example_3_multiflow",
    "example_4_bom", "example_5_shifts", "example_6_set_cover",
    "example_7_tuple_relation", "example_8_index_alignment",
    "example_9_inline_p3", "example_10_weighted_groupby",
    "example_11_multi_term", "example_12_intra_sum", "example_13_inter_sum",
    "example_14_three_var_intra", "example_15_mixed_shape_intra",
    "example_16_indexed_subset", "example_17_subset_tuple",
    "example_18_lhs_equality", "example_19_jk_secretary",
    "example_20_p3_name_mismatch", "example_21_p4_name_mismatch",
]


def run_pattern_coverage() -> list[str]:
    failures = []
    for mod_name in EXAMPLE_MODULES:
        mod = importlib.import_module(f"examples.{mod_name}")
        try:
            ok, diffs = check_instance(mod.build_pyomo_model, mod.data)
        except Exception as e:
            ok, diffs = False, [f"exception: {type(e).__name__}: {e}"]
        if ok:
            ref = build_reference(mod.build_pyomo_model, mod.data)
            print(f"    [OK]   {mod_name:<28s}  ({ref.NumConstrs} constrs x {ref.NumVars} vars)")
        else:
            print(f"    [FAIL] {mod_name}")
            for d in diffs:
                print(f"           - {d}")
            failures.append(mod_name)
    return failures


# ===========================================================================
# Test family 0 — negative controls (the verifier must have teeth)
# ===========================================================================

def run_negative_controls() -> list[str]:
    """Confirm the fingerprint DETECTS deliberate corruptions of the reference
    model.  A differential test that never fails proves nothing; these controls
    show that dropping a constraint, tightening a bound, or altering an
    objective coefficient each changes the fingerprint."""
    data = bench.gen_supply_demand(6, 5, seed=0)
    base_fp = fingerprint(build_reference(bench.sd_pyomo_model, data))

    def corrupt(kind):
        m = bench.sd_pyomo_model(data)
        opt = SolverFactory("gurobi_persistent")
        opt.set_instance(m)
        g = opt._solver_model
        g.update()
        if kind == "drop a constraint":
            g.remove(g.getConstrs()[0])
        elif kind == "tighten a bound":
            g.getVars()[0].UB = 5.0
        elif kind == "perturb a coefficient":
            row = g.getRow(g.getConstrs()[0])
            g.chgCoeff(g.getConstrs()[0], row.getVar(0), row.getCoeff(0) + 1.0)
        elif kind == "change an objective coeff":
            g.getVars()[0].Obj = 3.0
        g.update()
        return g

    failures = []
    for kind in ["drop a constraint", "tighten a bound",
                 "perturb a coefficient", "change an objective coeff"]:
        diffs = diff_fingerprints(base_fp, fingerprint(corrupt(kind)))
        if diffs:
            print(f"    [OK]   {kind:<26s} detected ({len(diffs)} keys differ)")
        else:
            print(f"    [FAIL] {kind:<26s} NOT detected -- verifier is blind!")
            failures.append(f"negative control: {kind}")
    return failures


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    import sys

    _ = build_reference(bench.sd_pyomo_model, bench.gen_supply_demand(2, 2, seed=0))  # warm license

    print("=" * 72)
    print("  FAMILY 0: Negative controls (verifier sensitivity)")
    print("=" * 72)
    fails_0 = run_negative_controls()

    print()
    print("=" * 72)
    print("  FAMILY 1: Data-parametric differential testing (reduced instances)")
    print("=" * 72)
    fails_1 = run_data_parametric()

    print()
    print("=" * 72)
    print("  FAMILY 2: Pattern coverage over examples/")
    print("=" * 72)
    fails_2 = run_pattern_coverage()

    n_param = sum(len(g) for _, _, _, g in CANONICAL)
    n_examples = len(EXAMPLE_MODULES)
    print()
    print("=" * 72)
    all_fails = fails_0 + fails_1 + fails_2
    if all_fails:
        print(f"  {len(all_fails)} FAILED out of {n_param + n_examples} instances:")
        for f in all_fails:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print(f"  ALL {n_param + n_examples} instances structurally identical "
              f"({n_param} reduced canonical + {n_examples} pattern-coverage).")
    print("=" * 72)
