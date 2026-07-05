# Pyomo → COO/MVar Translator Guide

`translate(func)` converts a `build_pyomo_model` function into the source of two
equivalent functions that use Gurobi's matrix API directly:

- **`build_vectorized_model(data)`** — builds the model using `addMVar` + `addMConstr`
  (scipy sparse COO matrices, O(1) C-API calls per constraint block)
- **`update_vectorized_model(m, new_data)`** — updates an already-built model with new
  parameter values and re-exposes it for re-optimization, without rebuilding variables

```python
from translator import translate, solve, populate_pyomo, solution_proxy
import examples.example_1_supply as ex1

# Option A — inspect generated code
code = translate(ex1.build_pyomo_model)
exec(code, ns := {})
model = ns['build_vectorized_model'](ex1.data)

# Option B — translate, solve, and get values back in one call
gp_model, values = solve(ex1.build_pyomo_model, ex1.data)
populate_pyomo(ex1.build_pyomo_model(ex1.data), values)

# Option C — build once, update parameters, re-solve (no variable rebuild)
exec(code, ns := {})
m = ns['build_vectorized_model'](ex1.data)
m.optimize()
new_data = {**ex1.data, 'Supply': {'Plant_A': 200, 'Plant_B': 300}}
update_vectorized_model(m, new_data)
m.reset()
m.optimize()
```

---

## Hard requirements

| Requirement | Notes |
|---|---|
| Model object named `m` | Parser matches `m.X = pyo.Y(...)` |
| Pyomo imported as `pyo` | Parser matches `pyo.Set`, `pyo.Var`, etc. |
| Single argument named `data` | Parser uses `data['key']` to locate data keys |

---

## Writing translatable rules

1. **No data-dependent branching.**
   ```python
   # BAD
   def supply_rule(m, i):
       if data['Supply'][i] > 0:
           return sum(m.x[i, j] for j in m.J) <= m.Supply[i]
       return pyo.Constraint.Skip
   ```
   Pre-filter indices in preprocessing; every constraint row must be valid.

2. **No arithmetic on index values.** Pre-compute derived mappings as Pyomo Sets.
   ```python
   # BAD
   sum(m.starts[s, t - tau] for tau in range(ShiftLength))

   # GOOD
   sum(m.starts[s, tp] for tp in m.ValidStarts[t])
   ```

3. **No generator conditions.**
   ```python
   # BAD
   sum(m.x[i, j] for j in m.J if data['active'][j])

   # GOOD — filter materialised as a Set
   sum(m.x[i, j] for j in m.ActiveJ)
   ```

4. **One comparison operator per return.** Chained comparisons (`lb <= expr <= ub`) raise `NotImplementedError`.

5. **No isolated nodes in flow-balance constraints.** A node with no edges produces a trivially-true `0 == 0` constraint. Filter in preprocessing.

---

## Supported declarations

```python
# Sets
m.S   = pyo.Set(initialize=data['S'])                  # 1-D
m.E   = pyo.Set(dimen=2, initialize=data['E'])          # 2-D (tuples)
m.Sub = pyo.Set(within=m.S, initialize=data['Sub'])     # subset
m.Rel = pyo.Set(m.S, initialize=data['Rel'])            # indexed (dict of lists)

# Params
m.Alpha = pyo.Param(initialize=data['Alpha'])           # scalar
m.Cost  = pyo.Param(m.I, initialize=data['Cost'])       # 1-D
m.BOM   = pyo.Param(m.P, m.C, initialize=data['BOM'])  # 2-D

# Vars — domains: NonNegativeReals (default), NonNegativeIntegers, Binary
m.x = pyo.Var(m.I, m.J, domain=pyo.NonNegativeReals)
m.y = pyo.Var(m.I,      domain=pyo.NonNegativeIntegers)
m.z = pyo.Var(m.F,      domain=pyo.Binary)

# Objective — rule= or expr=, multi-term, param × var or plain var
m.obj = pyo.Objective(rule=cost_obj, sense=pyo.minimize)

# Constraints — rule= (indexed) or expr= (scalar only)
m.c1 = pyo.Constraint(m.I, rule=some_rule)
m.c2 = pyo.Constraint(expr=sum(m.Cost[f] * m.build[f] for f in m.F) <= data['Budget'])
```

All parameters must be initialised from `data['key']`.

---

## Generated code structure

Every `build_vectorized_model` emits this skeleton:

```python
def build_vectorized_model(data):
    import gurobipy as gp
    import pandas as pd
    import numpy as np
    import scipy.sparse
    m = gp.Model()
    m._mconstr = {}   # name → MConstr, for update_vectorized_model
    m._rhs_ord  = {}  # name → index ordering used when RHS was set
    m._mvars    = {}  # var_name → MVar
    m._var_idx  = {}  # var_name → pd.MultiIndex

    # --- variable blocks ---
    _idx_x  = pd.MultiIndex.from_product([data['I'], data['J']], names=['i', 'j'])
    _var_x  = m.addMVar(len(_idx_x), lb=0.0, name='x')
    _flat_x = pd.DataFrame({'_col': np.arange(len(_idx_x))}, index=_idx_x).reset_index()
    m._mvars['x'] = _var_x
    m._var_idx['x'] = _idx_x

    # --- param blocks ---
    s_supply = pd.Series(data['Supply'], name='supply').rename_axis('i')

    # --- objective (if present) ---
    _c_obj = s_cost.reindex(_idx_x, level='i').values
    m.setObjective(_c_obj @ _var_x, gp.GRB.MINIMIZE)

    # --- constraint blocks (see patterns below) ---
    ...

    # --- solution accessor ---
    m._get_values = lambda: {'x': pd.Series(_var_x.X, index=_idx_x)}
    return m
```

The companion `update_vectorized_model` follows immediately:

```python
def update_vectorized_model(m, new_data):
    import gurobipy as gp
    import pandas as pd
    import numpy as np
    # Re-initialise params
    s_supply = pd.Series(new_data['Supply'], name='supply').rename_axis('i')
    # Hot-swap RHS vectors
    if 'supply_constr' in m._mconstr:
        m._mconstr['supply_constr'].setAttr('RHS',
            s_supply.reindex(m._rhs_ord['supply_constr']).values)
    # Re-set objective coefficients if model has an objective
    ...
    return m
```

Call `m.reset()` before re-optimising after an update (clears the incumbent
without discarding the variable objects).

**What update changes depends on the constraint type:**

| Constraint type | How update handles it | Warm-start preserved? |
|---|---|---|
| RHS-only (structural A matrix: 1s/−1s) | `setAttr('RHS', …)` on existing `MConstr` | Yes — LP basis intact |
| Matrix-coefficient (param values in A) | `m.remove` + full COO rebuild + re-add | No — A change invalidates basis |
| Objective | `m.setObjective(…)` with new coefficient vector | Yes |

**What update does NOT change:** variable objects, variable bounds, variable
types, or the index sets. If any of those need to change, call
`build_vectorized_model` again.

---

## Translation patterns

All patterns follow the same COO construction recipe:

1. Build `_flat_{var}` — a DataFrame with a `_col` ordinal for each variable slot.
2. Build `_constr_{name}` — a DataFrame with a `_row` ordinal for each constraint row.
3. `pd.merge` on shared index keys → (row, col) pairs.
4. `scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(m, n))` → constraint matrix A.
5. `m.addMConstr(A, mvar, sense, b)` or `m.addConstr((A1@v1 + A2@v2) op b)`.

---

### P1 — Groupby sum (plain set)

```python
def supply_rule(m, i):
    return sum(m.x[i, j] for j in m.J) <= m.Supply[i]
m.supply_constr = pyo.Constraint(m.I, rule=supply_rule)
```

Generated:
```python
_constr_supply_constr = pd.DataFrame({'_row': np.arange(len(s_supply))},
                                      index=s_supply.index).reset_index()
_coo_supply_constr = pd.merge(_flat_x, _constr_supply_constr, on=['i'])
_A_supply_constr = scipy.sparse.csr_matrix(
    (np.ones(len(_coo_supply_constr)),
     (_coo_supply_constr['_row'].values, _coo_supply_constr['_col'].values)),
    shape=(len(s_supply), len(_idx_x)))
m._mconstr['supply_constr'] = m.addMConstr(
    _A_supply_constr, _var_x, gp.GRB.LESS_EQUAL, s_supply.values,
    name='supply_constr')
m._rhs_ord['supply_constr'] = s_supply.index
```

Weighted variant (`param × var`) works identically; coefficient values replace
`np.ones(...)` in the sparse matrix.

Iteration set must be a plain (non-indexed) `pyo.Set`.

---

### P2 — Scalar global constraint

```python
m.budget_constr = pyo.Constraint(
    expr=sum(m.Cost[f] * m.build[f] for f in m.F) <= data['Budget']
)
```

Generated: a single `m.addLConstr(lhs <= rhs, name=...)` using a dense dot
product. No sparse matrix needed.

---

### P3 — Flow balance (subtraction of two sums)

```python
def flow_rule(m, node, k):
    flow_out = sum(m.x[node, j, k] for j in m.OutArcs[node])
    flow_in  = sum(m.x[i, node, k] for i in m.InArcs[node])
    return flow_out - flow_in == m.Dem[node, k]
m.flow_constr = pyo.Constraint(m.N, m.K, rule=flow_rule)
```

Generated: two merge passes (+1 / −1 coefficients), concatenated into one COO
and assembled into a single `addMConstr`.

```python
_constr_fwd_flow_constr = _constr_flow_constr.rename(columns={'node': 'i'})
_fwd_flow_constr = pd.merge(_flat_x, _constr_fwd_flow_constr, on=['i', 'k'])
_constr_bwd_flow_constr = _constr_flow_constr.rename(columns={'node': 'j'})
_bwd_flow_constr = pd.merge(_flat_x, _constr_bwd_flow_constr, on=['j', 'k'])
_coo_flow_constr = pd.concat([
    _fwd_flow_constr.assign(_val=1.0),
    _bwd_flow_constr.assign(_val=-1.0)], ignore_index=True)
_A_flow_constr = scipy.sparse.csr_matrix(
    (_coo_flow_constr['_val'].values, ...),
    shape=(len(s_demand), len(_idx_x)))
m._mconstr['flow_constr'] = m.addMConstr(
    _A_flow_constr, _var_x, gp.GRB.EQUAL, s_demand.values, name='flow_constr')
```

Both sums must iterate over **indexed sets** (`m.OutArcs[node]`, `m.InArcs[node]`).

---

### P4 — Cross-dimensional merge (indexed set + param × var)

```python
def component_rule(m, c, t):
    return m.buy_comp[c, t] == sum(m.BOM[p, c] * m.build[p, t]
                                    for p in m.ProdsUsingComp[c])
```

Generated: two A-matrix blocks (`_A_sum` for the RHS sum, `_A_neg` for the LHS
variable) assembled with `addConstr((A_sum @ build + A_neg @ buy) == 0)`.

```python
_fp_component_constr = s_bom.reset_index()
_m1_component_constr = pd.merge(_fp_component_constr, _flat_build, on='p')
_coo_component_constr = pd.merge(_m1_component_constr, _constr_component_constr, on=['c', 't'])
_A_sum_component_constr = scipy.sparse.csr_matrix(
    (_coo_component_constr['bom'].values, ...), shape=(...))
_dc_component_constr = pd.merge(_constr_component_constr, _flat_buy_comp, on=['c', 't'])
_A_neg_component_constr = scipy.sparse.csr_matrix(
    (-np.ones(len(_dc_component_constr)), ...), shape=(...))
m._mconstr['component_constr'] = m.addConstr(
    (_A_sum_component_constr @ _var_build + _A_neg_component_constr @ _var_buy_comp)
    == np.zeros(len(_idx_component_constr)), name='component_constr')
```

---

### P5 — Indexed relation / rolling window

```python
def cover_rule(m, s, t):
    return sum(m.starts[s, tp] for tp in m.ValidStarts[t]) >= m.Demand[s, t]
```

Generated: an explicit time-mapping DataFrame is built from the dict-of-lists,
merged with `_flat_starts` to get (row, col) pairs, then passed to `addMConstr`.

```python
_mapping_cover_constr = [(t, tp) for t, _inner in data['ValidStarts'].items()
                          for tp in _inner]
_map_cover_constr  = pd.DataFrame(_mapping_cover_constr, columns=['t', 'tp'])
_reset_cover_constr = _flat_starts.rename(columns={'t': 'tp'})
_lagged_cover_constr = pd.merge(_map_cover_constr, _reset_cover_constr, on='tp')
_coo_cover_constr   = pd.merge(_lagged_cover_constr, _constr_cover_constr, on=['s', 't'])
_A_cover_constr = scipy.sparse.csr_matrix(
    (np.ones(len(_coo_cover_constr)), ...), shape=(len(s_demand), len(_idx_starts)))
m._mconstr['cover_constr'] = m.addMConstr(
    _A_cover_constr, _var_starts, gp.GRB.GREATER_EQUAL, s_demand.values,
    name='cover_constr')
```

Tuple destructuring loop variables (`orig, dest`) are supported.

---

### P_inter_add — Sum of independent terms

```python
def cap_rule(m, p):
    return (sum(m.x_std[p, s] for s in m.S) +
            sum(m.x_exp[p, e] for e in m.E)) <= m.Cap[p]
```

Generated: one A matrix per sum term; constraint assembled with
`m.addConstr((A1 @ v1 + A2 @ v2) <= b)`.

---

### P_intra_add — Linear combination inside a single sum

```python
def net_rule(m, n):
    return sum(m.Cost[t] * m.prod[n, t] - m.salvage[n, t] for t in m.T) <= m.Budget[n]
```

Generated: one A matrix per variable term (signs baked into values);
assembled with `m.addConstr((A0 @ v0 + A1 @ v1) <= b)`.

---

### P6 — Direct variable access

```python
def demand_rule(m, p, t):
    return m.build[p, t] >= m.Demand[p, t]

# Subset filter
m.premium_constr = pyo.Constraint(m.PremiumProducts, m.T, rule=premium_rule)
```

Generated: identity-like COO (one nonzero per row) assembled from a merge of
the constraint index against `_flat_{var}`.

---

## Solve and populate API

```python
from translator import solve, solution_proxy, populate_pyomo

# Translate + build + optimize in one call
gp_model, values = solve(build_pyomo_model, data)
# gp_model — solved gp.Model; inspect gp_model.ObjVal, gp_model.SolCount, etc.
# values   — {var_name: pd.Series(index → float)}, empty if infeasible

# Access solution values — zero-cost, no Pyomo model built
sol = solution_proxy(values)
sol.x['i1', 'j1'].value   # same interface as pyo_model.x['i1', 'j1'].value

# Round-trip values back into a Pyomo model
pyo_m = build_pyomo_model(data)
populate_pyomo(pyo_m, values)
```

---

## Update API

```python
from translator import translate

code = translate(build_pyomo_model)
exec(code, ns := {})

# Build once
m = ns['build_vectorized_model'](data)
m.setParam('OutputFlag', 0)
m.optimize()

# Update parameters, re-solve without rebuilding variables
ns['update_vectorized_model'](m, new_data)
m.reset()       # clears incumbent; keeps variables, bounds, types
m.optimize()
```

`update_vectorized_model` handles two cases:

- **RHS-only constraints** (A matrix has structural 1s/−1s only): calls
  `MConstr.setAttr('RHS', …)` — the LP basis is preserved and warm-starting
  works. This covers P1 without a weighted param, P3, P5, P6.
- **Matrix-coefficient constraints** (param values appear inside the sum):
  calls `m.remove` on the old constraint, rebuilds the COO sparse matrix with
  new param values, and re-adds via `addMConstr`/`addConstr`. The LP basis is
  invalidated by the coefficient change regardless, so no warm-start
  opportunity is lost. This covers P1 with weighted param, P2, P4, P_inter_add,
  P_intra_add when those constraints carry a param inside the sum.
- **Objective**: always re-emitted via `m.setObjective(…)`.

It does **not** update: variable objects, variable bounds, variable types, or
index sets. Rebuild with `build_vectorized_model` if those change.

### Model attributes stored for update

| Attribute | Type | Purpose |
|---|---|---|
| `m._mconstr` | `dict[str, MConstr]` | Constraint handles, keyed by Pyomo name |
| `m._rhs_ord` | `dict[str, pd.Index]` | Row ordering for `setAttr('RHS', …)` |
| `m._constr_idx` | `dict[str, pd.Index]` | Row ordering for matrix-coefficient rebuilds |
| `m._mvars` | `dict[str, MVar]` | Variable MVar objects |
| `m._var_idx` | `dict[str, pd.MultiIndex]` | Variable index (for rebuilding flat frames) |
| `m._get_values` | `lambda` | `() → {var_name: pd.Series}` after solve |

---

## What is NOT supported

| Pattern | Alternative |
|---|---|
| `if`/`else` inside rules or generator conditions | Pre-filter into a Pyomo Set |
| Arithmetic on index values (`t - 1`, `i + 1`) | Pre-compute as an indexed Set |
| Chained comparisons (`lb <= expr <= ub`) | Split into two constraints |
| RHS as Python expression (`data['A'] - data['B']`) | Store result as a scalar Param |
| `pyo.Constraint.Skip` / `.Feasible` | Ensure all indices are valid in preprocessing |
| `pyo.Param(default=...)` | Pre-fill missing entries in preprocessing |
| P3 (`sum - sum`) in `expr=` form | Use `rule=` for flow-balance constraints |

---

## Preprocessing checklist

- Pre-compute all indexed set relations (adjacency dicts, rolling-window mappings).
- Fill sparse parameter dicts so no key is missing at solve time.
- Filter out isolated nodes (no edges) from flow-balance sets.

---

## Naming rule

Rule argument names become pandas index level names throughout the generated
code. Choose short, meaningful names (`i`, `t`, `node`) — they appear in every
`merge`, `rename`, and `reindex` call.

---

## Examples index

| File | Patterns covered |
|---|---|
| `example_1_supply.py` | P1 plain groupby |
| `example_2_network.py` | P2 scalar, `dimen=2` var, param × var |
| `example_3_multiflow.py` | P1, P3 flow balance (named intermediates) |
| `example_4_bom.py` | P4, P6, subset, objective |
| `example_5_shifts.py` | P5 rolling window, integer vars |
| `example_6_set_cover.py` | P5 relation, P2 `expr=`, binary vars |
| `example_7_tuple_relation.py` | P5 tuple loop var |
| `example_8_index_alignment.py` | P1 with 3-D var |
| `example_9_inline_p3.py` | P3 inline (no intermediates) |
| `example_10_weighted_groupby.py` | P1 weighted (param × var) |
| `example_11_multi_term.py` | P_inter_add, multi-term objective |
| `example_12_intra_sum.py` | P_intra_add (two vars, same shape) |
| `example_13_inter_sum.py` | P_inter_add, multi-term objective |
| `example_14_three_var_intra.py` | P_intra_add (three vars, same shape) |
| `example_15_mixed_shape_intra.py` | P_intra_add (2-D + 1-D broadcast) |
| `example_16_indexed_subset.py` | P5 indexed subset (`x[i,j]` over `SubSet[i]`) |
| `example_17_subset_tuple.py` | P5 tuple destructuring in indexed set |
| `example_18_lhs_equality.py` | P6 with equality constraint |
| `example_19_jk_secretary.py` | P_inter_add + P6, multi-var objective |
