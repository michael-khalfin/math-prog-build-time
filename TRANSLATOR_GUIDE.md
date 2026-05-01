# Pyomo → gurobipy-pandas Translator Guide

`translate(func)` converts a `build_pyomo_model` function into the source of an
equivalent `build_vectorized_model` function that uses gurobipy-pandas.

```python
from translator import translate, solve, populate_pyomo
import examples.example_1_supply as ex1

# Option A — inspect generated code
code = translate(ex1.build_pyomo_model)
exec(compile(code, "<translated>", "exec"))
model = build_vectorized_model(ex1.data)

# Option B — translate, solve, and get values back in one call
gp_model, values = solve(ex1.build_pyomo_model, ex1.data)
populate_pyomo(ex1.build_pyomo_model(ex1.data), values)
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

5. **No isolated nodes in flow-balance constraints.** A node with no edges produces a trivially-true `0 == 0` constraint that crashes the persistent solver. Filter in preprocessing.

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

## Translation patterns

### P1 — Groupby sum (plain set)

```python
def supply_rule(m, i):
    return sum(m.x[i, j] for j in m.J) <= m.Supply[i]
m.supply_constr = pyo.Constraint(m.I, rule=supply_rule)
```

Weighted (param × var):
```python
def machine_rule(m, mach):
    return sum(m.Efficiency[w] * m.assign[w, mach] for w in m.W) >= m.MinHours[mach]
```

Iteration set must be a plain (non-indexed) `pyo.Set`.

---

### P2 — Scalar global constraint

```python
m.budget_constr = pyo.Constraint(
    expr=sum(m.Cost[f] * m.build[f] for f in m.F) <= data['Budget']
)
```

No index sets on `pyo.Constraint`. Works equally with `rule=` (zero args after `m`).

---

### P3 — Flow balance (subtraction of two sums)

```python
# named-intermediate or inline — both work
def flow_rule(m, node, k):
    flow_out = sum(m.x[node, j, k] for j in m.OutArcs[node])
    flow_in  = sum(m.x[i, node, k] for i in m.InArcs[node])
    return flow_out - flow_in == m.Demand[node, k]
```

Both sums must iterate over **indexed sets** (`m.OutArcs[node]`, `m.InArcs[node]`).

---

### P4 — Cross-dimensional merge (indexed set + param × var)

```python
def component_rule(m, c, t):
    return m.buy_comp[c, t] == sum(m.BOM[p, c] * m.build[p, t]
                                    for p in m.ProdsUsingComp[c])
```

Iteration set must be indexed; element must be `param × var`.

---

### P5 — Indexed relation / subset (pure var sum)

```python
# Rolling window
def cover_rule(m, s, t):
    return sum(m.starts[s, tp] for tp in m.ValidStarts[t]) >= m.Demand[s, t]

# Tuple loop variable
def hub_rule(m, h):
    return sum(m.ship[orig, dest] for orig, dest in m.HubCoverage[h]) <= m.HubCap[h]

# Indexed subset — outer index also in variable subscript
def cap_rule(m, p):
    return sum(m.x[p, w] for w in m.SubSet[p]) <= m.Cap[p]
```

Declare the indexed set as `pyo.Set(m.I, initialize=data['SubSet'])`.
Tuple destructuring loop variables are supported.

---

### P_inter_add — Sum of independent terms

```python
def cap_rule(m, p):
    return (sum(m.x_std[p, s] for s in m.S) +
            sum(m.x_exp[p, e] for e in m.E)) <= m.Cap[p]
```

Any number of `sum(...)` calls joined by `+`; each may use a different variable.
Works in `expr=` form too.

---

### P_intra_add — Linear combination inside a single sum

```python
# Any number of variables, any mix of index shapes, +/- and optional param weight
def cap_rule(m, n):
    return sum(m.x[n, t] + m.y[n, t] + m.z[n, t] for t in m.T) <= m.Cap[n]

def demand_rule(m, p):
    return sum(m.assign[p, w] + m.flex[w] for w in m.W) >= m.Demand[p]

def net_rule(m, n):
    return sum(m.Cost[t] * m.prod[n, t] - m.salvage[n, t] for t in m.T) <= m.Budget[n]
```

---

### P6 — Direct variable access

```python
def demand_rule(m, p, t):
    return m.build[p, t] >= m.Demand[p, t]

# Subset filter
m.premium_constr = pyo.Constraint(m.PremiumProducts, m.T, rule=premium_rule)
```

LHS is a direct variable subscript, not a sum.

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
`groupby`, `merge`, and `reindex` call.

---

## Solve and populate API

```python
from translator import solve, populate_pyomo

# Translate + build + optimize in one call
gp_model, values = solve(build_pyomo_model, data)
# gp_model — solved gp.Model; check gp_model.ObjVal, gp_model.SolCount, etc.
# values   — {var_name: pd.Series(index → float)}, empty if infeasible

# Load solution back into a Pyomo model
pyo_model = build_pyomo_model(data)
populate_pyomo(pyo_model, values)
# pyo_model.x[i, j].value now holds the optimal value
```

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
