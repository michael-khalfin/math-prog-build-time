"""Mixed-dimensionality index products.

Locks in three structural cases surfaced by probing (2026-07):
  1. a variable over a product of a 1-D and a 2-D set, m.y = Var(m.W, m.Y2),
     grouped over either factor;
  2. a variable over a 3-D tuple set that appears ONLY in the objective
     (its index names must come from the objective scan / arity-safe fallback,
     previously "Length of names must match number of levels in MultiIndex");
  3. a direct bound alongside the above, so the model mixes shapes.
"""
import pyomo.environ as pyo

data = {
    "W": ["w1", "w2", "w3"],
    "Y2": [("a1", "b1"), ("a1", "b2"), ("a2", "b1")],
    "T3": [("a1", "b1", "c1"), ("a1", "b2", "c1"), ("a2", "b1", "c2")],
    "Cap": {"w1": 2, "w2": 2, "w3": 1},
    "Lim": {("a1", "b1"): 2, ("a1", "b2"): 1, ("a2", "b1"): 2},
    "UB": {"w1": 1.5, "w2": 2.5, "w3": 0.5},
}


def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    m.W = pyo.Set(initialize=data["W"])
    m.Y2 = pyo.Set(initialize=data["Y2"], dimen=2)
    m.T3 = pyo.Set(initialize=data["T3"], dimen=3)
    m.Cap = pyo.Param(m.W, initialize=data["Cap"])
    m.Lim = pyo.Param(m.Y2, initialize=data["Lim"])
    m.UB = pyo.Param(m.W, initialize=data["UB"])

    m.y = pyo.Var(m.W, m.Y2, domain=pyo.Binary)       # 1-D x 2-D product
    m.x = pyo.Var(m.T3, domain=pyo.Binary)            # 3-D, objective-only
    m.u = pyo.Var(m.W, domain=pyo.NonNegativeReals)

    def obj_rule(m):
        return (sum(m.x[a, b, c] for (a, b, c) in m.T3)
                + sum(m.u[w] for w in m.W))
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    def cap_rule(m, w):                               # grouped over the 1-D factor
        return sum(m.y[w, a, b] for (a, b) in m.Y2) <= m.Cap[w]
    m.cap_constr = pyo.Constraint(m.W, rule=cap_rule)

    def lim_rule(m, a, b):                            # grouped over the 2-D factor
        return sum(m.y[w, a, b] for w in m.W) <= m.Lim[a, b]
    m.lim_constr = pyo.Constraint(m.Y2, rule=lim_rule)

    def ub_rule(m, w):                                # direct bound
        return m.u[w] <= m.UB[w]
    m.ub_constr = pyo.Constraint(m.W, rule=ub_rule)

    return m
