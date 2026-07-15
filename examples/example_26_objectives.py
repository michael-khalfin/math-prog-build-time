"""Objective-parity regression (2026-07).

One model whose single multi-term objective combines every hard objective
form: a param-weighted sum over a mixed 1-D x 2-D product variable, a
subset-tuple-set sum (membership indicator, not dense), a NESTED RELATION sum
whose elements repeat across keys (coefficient = multiplicity), and a
scalar-coefficient term.  Also fixes the numeric objective sense (-1 = max).
"""
import pyomo.environ as pyo

data = {
    "W": ["w1", "w2", "w3"],
    "Y2": [("a1", "b1"), ("a1", "b2"), ("a2", "b1"), ("a2", "b2")],
    "S2": [("a1", "b2"), ("a2", "b1")],                 # subset of Y2
    "G": ["g1", "g2"],
    "R": {"g1": [("a1", "b1"), ("a1", "b2")],           # ("a1","b1") under both keys:
          "g2": [("a1", "b1"), ("a2", "b2")]},          # multiplicity 2
    "CW": {"w1": 2.0, "w2": 3.0, "w3": 5.0},
    "Cap": {"w1": 4, "w2": 4, "w3": 4},
}


def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    m.W = pyo.Set(initialize=data["W"])
    m.Y2 = pyo.Set(initialize=data["Y2"], dimen=2)
    m.S2 = pyo.Set(initialize=data["S2"], dimen=2)
    m.G = pyo.Set(initialize=data["G"])
    m.R = pyo.Set(m.G, dimen=2, initialize=data["R"])
    m.CW = pyo.Param(m.W, initialize=data["CW"])
    m.Cap = pyo.Param(m.W, initialize=data["Cap"])

    m.y = pyo.Var(m.W, m.Y2, domain=pyo.Binary)         # mixed 1-D x 2-D product
    m.u = pyo.Var(m.Y2, domain=pyo.NonNegativeReals)
    m.v = pyo.Var(m.W, domain=pyo.NonNegativeReals)

    def cap_rule(m, w):
        return sum(m.y[w, a, b] for (a, b) in m.Y2) <= m.Cap[w]
    m.cap = pyo.Constraint(m.W, rule=cap_rule)

    def link_rule(m, a, b):
        return m.u[a, b] <= sum(m.y[w, a, b] for w in m.W)
    m.link = pyo.Constraint(m.Y2, rule=link_rule)

    def obj_rule(m):
        return (
            sum(m.CW[w] * m.y[w, a, b] for w in m.W for (a, b) in m.Y2)
            - sum(m.u[a, b] for g in m.G for (a, b) in m.R[g])
            + sum(m.u[a, b] for (a, b) in m.S2)
            - sum(0.5 * m.v[w] for w in m.W)
        )
    m.obj = pyo.Objective(rule=obj_rule, sense=-1)      # numeric sense: maximize
    return m
