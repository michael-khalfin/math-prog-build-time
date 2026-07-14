# math-prog-build-time

A **Pyomo → Gurobi transpiler**: write your optimization model once, in
readable Pyomo, and build it at the speed of hand-written Gurobi matrix code.
The transpiler reads a restricted `build_pyomo_model(data)` function as source
text (never executing Pyomo) and emits an equivalent builder that assembles
sparse COO matrices and calls `addMVar` / `addMConstr` — typically **10–100×
faster construction** at millions of variables, plus an in-place update
function for cheap re-solves.

Every transpiled model is certified by **differential testing** before it is
trusted: the transpiled and Pyomo-built Gurobi models are compared through a
permutation-invariant structural signature on reduced data instances.

The transpiler itself was produced by an agentic LLM workflow (Claude Code)
gated by that same differential oracle; the accompanying paper
(`paper/writeup/paper.tex`) describes the methodology.

## Quickstart

```python
from translator import translate, solve
from differential_test import verify
import examples.example_1_supply as ex1

verify(ex1.build_pyomo_model, ex1.data)      # certify equivalence first
gp_model, values = solve(ex1.build_pyomo_model, ex1.data)   # then use it
```

See **[TRANSLATOR_GUIDE.md](TRANSLATOR_GUIDE.md)** for the full usage options
(inspecting generated code, in-place updates, dynamically `exec`'d models),
authoring conventions, and the catalog of supported constraint shapes.

## Repository map

| Path | Purpose |
|---|---|
| `translator.py` | The transpiler (COO+MVar). Pipeline: parse (AST) → classify shapes → reconcile indices → generate code. Public API: `translate`, `solve`, `make_model_fn`, `populate_pyomo`, `solution_proxy`. |
| `translator_old.py` | The first-generation transpiler (gurobipy-pandas). Superseded; kept as a benchmark comparator. |
| `differential_test.py` | **Primary oracle.** `verify(model_fn, data)` for one model; `python differential_test.py` runs the full suite (negative controls + reduced canonical instances + every example). |
| `test_translator.py` | Transpiler API tests: `solve()`, `populate_pyomo()`, `update_vectorized_model()`. |
| `run_tests.py` | Sanity checks for the hand-written vectorized models in `examples/`. |
| `examples/` | 23 models; each isolates one or more modeling features and doubles as a regression test. |
| `benchmark_comprehensive.py` | Four-pipeline build-time benchmark (Pyomo persistent / Pyomo+templatization / gurobipy-pandas / COO+MVar) with a GC-controlled protocol. `--no-timing` disables Pyomo's per-component timing output. |
| `benchmark.py` | Early single-model benchmark (superseded). |
| `visualize.py` | Interactive four-panel viewer of the translation pipeline (serves HTML locally). |
| `TRANSLATOR_GUIDE.md` | User guide: usage options A–E, authoring rules, supported shapes, verification. |
| `paper/` | LaTeX sources for the paper; `paper/writeup/paper.tex` is the main document. |

## Running the tests

```bash
python differential_test.py     # primary: structural equivalence, 53 instances
python test_translator.py      # solve / populate / update APIs
python run_tests.py            # hand-written example sanity
```

All three must pass. New models that `verify` cleanly should be added to
`EXAMPLE_MODULES` in `differential_test.py` so they are retained as
regressions.

## Requirements

Python ≥ 3.9 with `pyomo`, `gurobipy` (licensed), `gurobipy-pandas`, `pandas`,
`numpy`, `scipy`, and `matplotlib` (benchmarks only). Compiling the paper
additionally needs LaTeX with `minted` (Pygments on `PATH`, `-shell-escape`).
