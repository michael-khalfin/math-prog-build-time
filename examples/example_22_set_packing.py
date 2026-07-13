import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# Small consistent instance. Membership relations that the model relies on:
#   C, E[d], K   subset of A          (u is indexed by A)
#   F, G         subset of B          (v is indexed by B)
#   H[p,q] tuples (r,z,w) with (r,w) in A
#   L[p,q] tuples (a,b,r,w) with (r,w) in B
data = {
    "A": [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)],
    "B": [(1, 1), (1, 2), (2, 1)],
    "C": [(1, 1), (1, 2), (2, 1)],
    "D": ["d1", "d2"],
    "E": {"d1": [(1, 1), (2, 1)], "d2": [(1, 2), (3, 1)]},
    "F": [(1, 1)],
    "G": [(1, 2), (2, 1)],
    "H": {(1, 2): [(1, 9, 1), (2, 9, 2)], (2, 1): [(3, 9, 1)]},
    "K": [(1, 1), (2, 2)],
    "L": {(1, 1): [(7, 8, 1, 1), (7, 8, 2, 1)], (2, 2): [(7, 8, 1, 2)]},
}

def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    m.A = pyo.Set(initialize=data["A"], dimen=2)
    m.B = pyo.Set(initialize=data["B"], dimen=2)
    m.C = pyo.Set(initialize=data["C"], dimen=2)
    m.D = pyo.Set(initialize=data["D"])
    m.E = pyo.Set(m.D, dimen=2, initialize=data["E"])
    m.F = pyo.Set(initialize=data["F"], dimen=2)
    m.G = pyo.Set(initialize=data["G"], dimen=2)
    m.H = pyo.Set(m.G, dimen=3, initialize=data["H"])
    m.K = pyo.Set(initialize=data["K"], dimen=2)
    m.L = pyo.Set(m.K, dimen=4, initialize=data["L"])

    m.u = pyo.Var(m.A, domain=pyo.Binary)
    m.v = pyo.Var(m.B, domain=pyo.Binary)

    def obj_rule(m):
        return sum(m.u[p,q] for (p,q) in m.A)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    def c0_rule(m):
        return sum(m.u[p, q] for (p,q) in m.C) == 3
    m.c0 = pyo.Constraint(rule=c0_rule)

    def c1_rule(m, d):
        return sum(m.u[p,q] for (p,q) in m.E[d]) <= 1
    m.c1 = pyo.Constraint(m.D, rule=c1_rule)
    
    def c2_rule(m, p, q):
        return m.v[p,q] == 0
    m.c2 = pyo.Constraint(m.F, rule=c2_rule)

    def c3_rule(m, p, q):
        return m.v[p,q] == sum(
            m.u[r,w]
            for (r,z,w) in m.H[p,q]
        )
    m.c3 = pyo.Constraint(m.G, rule=c3_rule)

    def c4_rule(m, p, q):
        return m.u[p,q] <= sum(
            m.v[r,w]
            for (a,b,r,w) in m.L[p,q]
        )
    m.c4 = pyo.Constraint(m.K, rule=c4_rule)

    return m