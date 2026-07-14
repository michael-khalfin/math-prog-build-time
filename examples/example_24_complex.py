# Small consistent instance.  Referential-integrity obligations:
#   D5, D7[g], D10, D19[a,t] subsets of D2      (z0's tail index)
#   D8 subset of D3; D9[a,b] tuples (p,q,r) with (p,r) in D2
#   D11[a,b], D12, D16[c,a] subsets of D4       (z2's tail index)
#   D13[a,b,h] subset of D3
#   every (a,b,h) in D4: (a,h) in D2 (c4) and a in D15 (c7)
#   every (a,b,h,q) in D14: (a,b,h) in D4, (a,h) in D2, (b,q) in D3
data = {
    "D0": ["c1"],
    "D2": [("a1", "b1"), ("a1", "b2"), ("a2", "b1")],
    "D3": [("a1", "q1"), ("a2", "q1"), ("b1", "q1")],
    "D4": [("a1", "b1", "b2"), ("a2", "b1", "b1")],
    "D5": [("a1", "b1"), ("a2", "b1")],
    "D6": ["g1"],
    "D7": {"g1": [("a1", "b1"), ("a1", "b2")]},
    "D8": [("a1", "q1")],
    "D9": {("a1", "q1"): [("a1", "x", "b1"), ("a2", "x", "b1")]},
    "D10": [("a1", "b1")],
    "D11": {("a1", "b1"): [("a1", "b1", "b2"), ("a2", "b1", "b1")]},
    "D12": [("a1", "b1", "b2")],
    "D13": {("a1", "b1", "b2"): [("a1", "q1")]},
    "D14": [("a1", "b1", "b2", "q1")],
    "D15": ["a1", "a2"],
    "D16": {("c1", "a1"): [("a1", "b1", "b2")],
            ("c1", "a2"): [("a2", "b1", "b1")]},
    "D17": ["t1"],
    "D19": {("a1", "t1"): [("a1", "b1")],
            ("a2", "t1"): [("a2", "b1")]},
    "N0": 1,
    "N1": 2,
    "N2": 10,
    "N3": 1,
}


def build_pyomo_model_generic_full(data):
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()

    m.D0 = pyo.Set(initialize=data["D0"])
    m.D1 = pyo.Set(initialize=["H", "L"])

    m.E0 = pyo.Set(
        initialize=[(e, c) for e in ["H", "L"] for c in data["D0"]],
        dimen=2,
    )

    m.D2 = pyo.Set(initialize=data["D2"], dimen=2)
    m.D3 = pyo.Set(initialize=data["D3"], dimen=2)
    m.D4 = pyo.Set(initialize=data["D4"], dimen=3)

    m.z0 = pyo.Var(m.E0, m.D2, domain=pyo.Binary)
    m.z1 = pyo.Var(m.E0, m.D3, domain=pyo.Binary)
    m.z2 = pyo.Var(m.E0, m.D4, domain=pyo.Binary)

    m.D5 = pyo.Set(initialize=data["D5"], dimen=2)

    def c0(m, e, c):
        return sum(m.z0[e, c, a, b] for (a, b) in m.D5) == data["N0"]

    m.c0 = pyo.Constraint(m.E0, rule=c0)

    m.D6 = pyo.Set(initialize=data["D6"])
    m.D7 = pyo.Set(m.D6, dimen=2, initialize=data["D7"])

    def c1(m, e, c, g):
        return sum(m.z0[e, c, a, b] for (a, b) in m.D7[g]) <= 1

    m.c1 = pyo.Constraint(m.E0, m.D6, rule=c1)

    m.D8 = pyo.Set(initialize=data["D8"], dimen=2)
    m.D9 = pyo.Set(m.D8, dimen=3, initialize=data["D9"])

    def c2(m, e, c, a, b):
        return m.z1[e, c, a, b] == sum(
            m.z0[e, c, p, r]
            for (p, q, r) in m.D9[a, b]
        )

    m.c2 = pyo.Constraint(m.E0, m.D8, rule=c2)

    m.D10 = pyo.Set(initialize=data["D10"], dimen=2)
    m.D11 = pyo.Set(m.D10, dimen=3, initialize=data["D11"])

    def c3(m, e, c, a, b):
        return m.z0[e, c, a, b] <= sum(
            m.z2[e, c, p, q, r]
            for (p, q, r) in m.D11[a, b]
        )

    m.c3 = pyo.Constraint(m.E0, m.D10, rule=c3)

    def c4(m, e, c, a, b, h):
        return m.z2[e, c, a, b, h] <= m.z0[e, c, a, h]

    m.c4 = pyo.Constraint(m.E0, m.D4, rule=c4)

    m.D12 = pyo.Set(initialize=data["D12"], dimen=3)
    m.D13 = pyo.Set(m.D12, dimen=2, initialize=data["D13"])

    def c5(m, e, c, a, b, h):
        return m.z2[e, c, a, b, h] <= sum(
            m.z1[e, c, p, q]
            for (p, q) in m.D13[a, b, h]
        )

    m.c5 = pyo.Constraint(m.E0, m.D12, rule=c5)

    m.D14 = pyo.Set(initialize=data["D14"], dimen=4)

    def c6(m, e, c, a, b, h, q):
        return m.z2[e, c, a, b, h] >= (
            m.z0[e, c, a, h] + m.z1[e, c, b, q] - 1
        )

    m.c6 = pyo.Constraint(m.E0, m.D14, rule=c6)

    m.D15 = pyo.Set(initialize=data["D15"])

    m.z3 = pyo.Var(m.D1, m.D0, m.D15, domain=pyo.Binary)
    m.z4 = pyo.Var(m.D1, m.D0, domain=pyo.NonNegativeIntegers)
    m.z5 = pyo.Var(m.D1, m.D0, domain=pyo.Binary)

    def c7(m, e, c, a, b, h):
        return m.z3[e, c, a] >= m.z2[e, c, a, b, h]

    m.c7 = pyo.Constraint(m.D1, m.D0, m.D4, rule=c7)

    m.D16 = pyo.Set(m.D0, m.D15, dimen=3, initialize=data["D16"])

    def c8(m, e, c, a):
        return m.z3[e, c, a] <= sum(
            m.z2[e, c, p, q, r]
            for (p, q, r) in m.D16[c, a]
        )

    m.c8 = pyo.Constraint(m.D1, m.D0, m.D15, rule=c8)

    def c9(m, e, c):
        return m.z4[e, c] == sum(m.z3[e, c, a] for a in m.D15)

    m.c9 = pyo.Constraint(m.D1, m.D0, rule=c9)

    def c10(m, e, c):
        return m.z4[e, c] >= data["N1"] * m.z5[e, c]

    m.c10 = pyo.Constraint(m.D1, m.D0, rule=c10)

    def c11(m, e, c):
        return m.z4[e, c] <= data["N1"] - 1 + data["N2"] * m.z5[e, c]

    m.c11 = pyo.Constraint(m.D1, m.D0, rule=c11)

    m.D17 = pyo.Set(initialize=data["D17"])
    m.D18 = pyo.Set(
        initialize=[(a, b) for a in data["D15"] for b in data["D17"]],
        dimen=2,
    )

    m.D19 = pyo.Set(m.D18, dimen=2, initialize=data["D19"])

    m.z6 = pyo.Var(m.D18, domain=pyo.Binary)
    m.z7 = pyo.Var(m.D18, domain=pyo.Binary)

    def c12(m):
        return sum(m.z6[a, b] for (a, b) in m.D18) <= data["N3"]

    m.c12 = pyo.Constraint(rule=c12)

    def c13(m, e, c, a, b):
        return (
            sum(m.z0[e, c, p, q] for (p, q) in m.D19[a, b])
            - m.z7[a, b]
            <= 1 - m.z6[a, b]
        )

    m.c13 = pyo.Constraint(m.E0, m.D18, rule=c13)

    def c14(m, e, c, a, b):
        return (
            m.z7[a, b]
            - sum(m.z0[e, c, p, q] for (p, q) in m.D19[a, b])
            <= 1 - m.z6[a, b]
        )

    m.c14 = pyo.Constraint(m.E0, m.D18, rule=c14)

    def o(m):
        return (
            sum(m.z5["H", c] for c in m.D0)
            - sum(m.z5["L", c] for c in m.D0)
        )

    m.o = pyo.Objective(rule=o, sense=pyo.maximize)

    return m

# Standard entry-point aliases used by the test suites.
build_pyomo_model = build_pyomo_model_generic_full
