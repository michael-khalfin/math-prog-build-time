# Pyomo → gurobipy-pandas Translator Guide

`translate(func)` converts a `build_pyomo_model` function into the source of an
equivalent `build_vectorized_model` function that uses gurobipy-pandas.
The translated model is built orders of magnitude faster for large instances
because it replaces Python for-loops with vectorized pandas operations.

---

## Quick start

```python
from translator import translate
import examples.example_1_supply as ex1

code = translate(ex1.build_pyomo_model)   # returns a Python source string
exec(compile(code, "<translated>", "exec"))
model = build_vectorized_model(ex1.data)  # ready-to-solve gp.Model
```

---

## How to write a translatable model

### Hard requirements

| Requirement | Why |
|---|---|
| The Pyomo model object must be named `m` | The parser looks for `m.X = pyo.Y(...)` |
| Pyomo must be imported as `pyo` | The parser matches `pyo.Set`, `pyo.Var`, etc. |
| Function must be named `build_pyomo_model` | Required for `test_translator.py` test harness; not required by `translate()` itself |
| The function's single argument must be named `data` | The parser uses `data['key']` to locate data keys |

### AbstractModel templatization rules

These are the rules that make a Pyomo model translatable.  Violating them
produces models that cannot be vectorized correctly (or at all).

1. **No data-dependent branching inside rules.**
   Rules must be pure arithmetic expressions over sets and parameters.
   ```python
   # GOOD
   def supply_rule(m, i):
       return sum(m.x[i, j] for j in m.J) <= m.Supply[i]

   # BAD — if/else on data value
   def supply_rule(m, i):
       if data['Supply'][i] > 0:
           return sum(m.x[i, j] for j in m.J) <= m.Supply[i]
       return pyo.Constraint.Skip
   ```

2. **No arithmetic on index values inside rules.**
   Pre-compute any derived index mappings in `preprocess_data` and store them
   as Pyomo Sets.
   ```python
   # BAD — index arithmetic inside rule
   def cover_rule(m, s, t):
       return sum(m.starts[s, t - tau] for tau in range(ShiftLength)) >= m.Demand[s, t]

   # GOOD — mapping pre-computed; rule iterates over a set
   def cover_rule(m, s, t):
       return sum(m.starts[s, tp] for tp in m.ValidStarts[t]) >= m.Demand[s, t]
   ```

3. **No generator conditions (`if` inside comprehensions).**
   ```python
   # BAD
   sum(m.x[i, j] for j in m.J if data['active'][j])

   # GOOD — filter materialised as a Set
   m.ActiveJ = pyo.Set(initialize=[j for j in data['J'] if data['active'][j]])
   sum(m.x[i, j] for j in m.ActiveJ)
   ```

4. **Each constraint rule returns a single `Compare` expression.**
   One comparison operator per return statement.  Chained comparisons such as
   `0 <= expr <= UB` are not supported and will raise `NotImplementedError`.

5. **No isolated nodes in flow-balance constraints.**
   A node that has neither inbound nor outbound edges produces a trivially-True
   Pyomo constraint (`0 == 0`) that crashes the persistent solver.
   Filter such nodes in preprocessing.

---

## Supported Pyomo declarations

### `pyo.Set`

```python
m.S   = pyo.Set(initialize=data['S'])           # plain 1-D set
m.E   = pyo.Set(dimen=2, initialize=data['E'])  # 2-D set (tuples)
m.Sub = pyo.Set(within=m.S, initialize=data['Sub'])  # subset
m.Rel = pyo.Set(m.S, initialize=data['Rel'])    # indexed set (dict of lists)
```

`dimen=` can be any integer.  Indexed sets (`pyo.Set(m.X, ...)`) are used for
adjacency / rolling-window patterns (P5).

### `pyo.Param`

```python
m.Alpha = pyo.Param(initialize=data['Alpha'])        # scalar
m.Cost  = pyo.Param(m.I, initialize=data['Cost'])    # 1-D
m.BOM   = pyo.Param(m.P, m.C, initialize=data['BOM'])  # 2-D
```

All parameters must be initialised from `data['key']`.

### `pyo.Var`

```python
m.x = pyo.Var(m.I, m.J, domain=pyo.NonNegativeReals)   # continuous (default)
m.y = pyo.Var(m.I, domain=pyo.NonNegativeIntegers)      # integer
m.z = pyo.Var(m.F, domain=pyo.Binary)                   # binary
```

Supported domains: `NonNegativeReals`, `NonNegativeIntegers`, `Binary`.

### `pyo.Objective`

```python
def cost_obj(m):
    return sum(m.Cost[i] * m.x[i, t] for i in m.I for t in m.T)
m.obj = pyo.Objective(rule=cost_obj, sense=pyo.minimize)
```

Supports a single `param * var` term with one or two generator clauses.

### `pyo.Constraint`

Constraints are translated by pattern (see next section).  Both the
`rule=` form (indexed) and the `expr=` form (scalar, no index) are supported:

```python
m.c1 = pyo.Constraint(m.I, rule=some_rule)           # indexed
m.c2 = pyo.Constraint(expr=sum(...) <= data['UB'])    # scalar inline
```

---

## The six translation patterns

### P1 — Groupby sum (indexed constraint, plain set iteration)

```python
def supply_rule(m, i):
    return sum(m.x[i, j] for j in m.J) <= m.Supply[i]
m.supply_constr = pyo.Constraint(m.I, rule=supply_rule)
```

Weighted variant (param × var):

```python
def machine_rule(m, mach):
    return sum(m.Efficiency[w] * m.assign[w, mach] for w in m.W) >= m.MinHours[mach]
m.machine_constr = pyo.Constraint(m.M, rule=machine_rule)
```

**Requirement:** iteration set must be a plain `pyo.Set` (not indexed).

---

### P2 — Scalar global constraint (`expr=` or empty rule args)

```python
def budget_rule(m):
    return sum(m.Cost[f] * m.build[f] for f in m.F) <= data['Budget']
m.budget_constr = pyo.Constraint(rule=budget_rule)

# Equivalent inline form:
m.budget_constr = pyo.Constraint(
    expr=sum(m.Cost[f] * m.build[f] for f in m.F) <= data['Budget']
)
```

The constraint has **no index sets** (no `m.X` as positional args to
`pyo.Constraint`).

---

### P3 — Flow balance (indexed or inline subtraction of two sums)

```python
# Named-intermediate style
def flow_rule(m, node, k):
    flow_out = sum(m.x[node, j, k] for j in m.OutArcs[node])
    flow_in  = sum(m.x[i, node, k] for i in m.InArcs[node])
    return flow_out - flow_in == m.Demand[node, k]

# Inline style (equivalent)
def flow_rule(m, node, k):
    return (sum(m.x[node, j, k] for j in m.OutArcs[node]) -
            sum(m.x[i, node, k] for i in m.InArcs[node])) == m.Demand[node, k]
```

**Requirement:** both sums iterate over **indexed sets** (`m.OutArcs[node]`,
`m.InArcs[node]`).  The variable subscripts uniquely identify which index
position is the "free" dimension for each sum.

---

### P4 — Cross-dimensional merge (indexed set + param × var)

```python
def component_rule(m, c, t):
    return m.buy_comp[c, t] == sum(m.BOM[p, c] * m.build[p, t]
                                    for p in m.ProdsUsingComp[c])
```

The iteration set must be indexed (`m.ProdsUsingComp[c]`) and the element
must be `param × var` (in either order).

---

### P5 — Universal adjacency (indexed set, pure var sum)

```python
def cover_rule(m, s, t):
    return sum(m.starts[s, tp] for tp in m.ValidStarts[t]) >= m.Demand[s, t]

def hub_rule(m, h):
    return sum(m.ship[orig, dest] for orig, dest in m.HubCoverage[h]) <= m.HubCap[h]
```

The iteration set must be indexed.  Tuple destructuring loop variables
(`orig, dest`) are supported.

---

### P6 — Direct variable access

```python
# Simple
def demand_rule(m, p, t):
    return m.build[p, t] >= m.Demand[p, t]

# With subset filter
def premium_rule(m, p, t):
    return m.build[p, t] >= m.MinPremium
m.premium_constr = pyo.Constraint(m.PremiumProducts, m.T, rule=premium_rule)
```

The LHS is a direct subscript of a decision variable, not a sum.

---

## What is NOT supported

| Pattern | Alternative |
|---|---|
| `if`/`else` inside rules or generator conditions | Pre-filter into a Pyomo Set in preprocessing |
| Arithmetic on index values (`t - 1`, `i + 1`) | Pre-compute the mapping as an indexed Set |
| Chained comparisons (`lb <= expr <= ub`) | Split into two separate constraints |
| Sum of two independent terms on LHS (`sum_A + sum_B`) | Not yet supported |
| RHS computed from Python expressions (`data['A'] - data['B']`) | Store the result as a scalar Param |
| `pyo.Constraint.Skip` or `pyo.Constraint.Feasible` | Not supported; ensure all nodes/indices are valid |
| `pyo.Param` with `default=` keyword | Not supported; pre-fill missing entries in preprocessing |
| Multiple variables in a single sum (`m.x[i] + m.y[i]`) | Not supported; use two separate constraints |
| Objectives with multiple independent sum terms | Only one `param * var` term is supported per objective |

---

## Preprocessing conventions

These conventions make models robust and translator-compatible:

```python
def preprocess_data(raw):
    # 1. Pre-compute indexed set relations
    out_arcs = {n: [] for n in raw['Nodes']}
    for i, j in raw['Edges']:
        out_arcs[i].append(j)
    raw['OutArcs'] = out_arcs

    # 2. Fill sparse dicts to avoid missing-key errors
    full_demand = {(n, k): 0 for n in raw['Nodes'] for k in raw['Commodities']}
    full_demand.update(raw['Demand'])
    raw['Demand'] = full_demand

    # 3. Remove isolated nodes (nodes with no edges)
    raw['ActiveNodes'] = sorted({n for e in raw['Edges'] for n in e})

    return raw
```

---

## Name mapping rules

pandas requires exact MultiIndex level names for `groupby`, `merge`, and
`reindex`.  The translator derives level names from the Pyomo rule function
argument names — **the rule arg names become the index level names**.

```python
# Rule arg 'i' → index level 'i' throughout
def supply_rule(m, i):
    return sum(m.x[i, j] for j in m.J) <= m.Supply[i]
```

For variables indexed by a `dimen=2` set that never appears as a constraint
index (e.g., an `Arcs` set only referenced inside a P3 sum), the translator
infers dimension names from the loop variable names used in subscripts.

Loop variable names that differ from the set name (e.g., `fac` instead of `f`)
are automatically resolved to the index-level name registered from the rule args.

---

## Examples index

| File | Patterns covered |
|---|---|
| `example_1_supply.py` | P1 plain groupby |
| `example_2_network.py` | P2 scalar, `dimen=2` var set, param × var |
| `example_3_multiflow.py` | P1, P3 flow balance (named intermediates) |
| `example_4_bom.py` | P4 cross-dim merge, P6 direct var, subset, objective |
| `example_5_shifts.py` | P5 rolling window, integer vars |
| `example_6_set_cover.py` | P5 relation, P2 inline `expr=`, binary vars |
| `example_7_tuple_relation.py` | P5 with tuple loop var |
| `example_8_index_alignment.py` | P1 with 3-D var, groupby over middle dimension |
| `example_9_inline_p3.py` | P3 flow balance written inline (no intermediate vars) |
| `example_10_weighted_groupby.py` | P1 weighted (param × var, non-indexed set) |
