# Small consistent instance.  Referential-integrity obligations:
#   D8, D10[g], D14 subsets of D3; D11, D12 subsets of D4
#   D1 subset of D0 x D3 (objective);  D15[a,b] subset of D5
#   D13[(a,b) in D12] tuples (p,q,r) with (p,r) in D3
#   every (a,b,c) in D5: (a,c) in D3, and D16 must key every D5 element,
#     with values q such that (b,q) in D4
#   every (a,b,c,q) in D6: (a,b,c) in D5, (a,c) in D3, (b,q) in D4
data = {
    "D0": ["e1"],
    "D1": [("e1", "a1", "b1"), ("e1", "a2", "b1")],
    "D2": ["x1"],
    "D3": [("a1", "b1"), ("a1", "b2"), ("a2", "b1")],
    "D4": [("g1", "q1"), ("g2", "q1")],
    "D5": [("a1", "g1", "b1"), ("a2", "g2", "b1")],
    "D6": [("a1", "g1", "b1", "q1")],
    "D7": [("y1", "y2")],
    "D8": [("a1", "b1"), ("a2", "b1")],
    "D9": ["g1"],
    "D10": {"g1": [("a1", "b1"), ("a1", "b2")]},
    "D11": [("g1", "q1")],
    "D12": [("g2", "q1")],
    "D13": {("g2", "q1"): [("a1", "x", "b1")]},
    "D14": [("a1", "b2")],
    "D15": {("a1", "b2"): [("a1", "g1", "b1")]},
    "D16": {("a1", "g1", "b1"): ["q1"], ("a2", "g2", "b1"): ["q1"]},
    "N0": 1,
}


def build_model(data):
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()

    m.D0 = pyo.Set(initialize=data["D0"])
    m.D1 = pyo.Set(initialize=data["D1"], dimen=3)

    m.D2 = pyo.Set(initialize=data["D2"])

    m.D3 = pyo.Set(initialize=data["D3"], dimen=2)
    m.D4 = pyo.Set(initialize=data["D4"], dimen=2)
    m.D5 = pyo.Set(initialize=data["D5"], dimen=3)
    m.D6 = pyo.Set(initialize=data["D6"], dimen=4)

    m.D7 = pyo.Set(initialize=data["D7"], dimen=2)

    m.D8 = pyo.Set(initialize=data["D8"], dimen=2)

    m.D9 = pyo.Set(initialize=data["D9"])
    m.D10 = pyo.Set(
        m.D9,
        dimen=2,
        initialize=data["D10"],
    )

    m.D11 = pyo.Set(initialize=data["D11"], dimen=2)
    m.D12 = pyo.Set(initialize=data["D12"], dimen=2)
    m.D13 = pyo.Set(
        m.D12,
        dimen=3,
        initialize=data["D13"],
    )

    m.D14 = pyo.Set(initialize=data["D14"], dimen=2)
    m.D15 = pyo.Set(
        m.D14,
        dimen=3,
        initialize=data["D15"],
    )

    m.D16 = pyo.Set(
        m.D5,
        dimen=1,
        initialize=data["D16"],
    )

    m.N0 = pyo.Param(initialize=data["N0"])

    m.z0 = pyo.Var(m.D0, m.D3, domain=pyo.Binary)
    m.z1 = pyo.Var(m.D0, m.D4, domain=pyo.Binary)
    m.z2 = pyo.Var(m.D0, m.D5, domain=pyo.Binary)

    def c0(m, e):
        return sum(
            m.z0[e, a, b]
            for (a, b) in m.D8
        ) == m.N0

    m.c0 = pyo.Constraint(m.D0, rule=c0)

    def c1(m, e, g):
        return sum(
            m.z0[e, a, b]
            for (a, b) in m.D10[g]
        ) <= 1

    m.c1 = pyo.Constraint(m.D0, m.D9, rule=c1)

    def c2(m, e, a, b):
        return m.z1[e, a, b] == 0

    m.c2 = pyo.Constraint(m.D0, m.D11, rule=c2)

    def c3(m, e, a, b):
        return m.z1[e, a, b] == sum(
            m.z0[e, p, r]
            for (p, q, r) in m.D13[a, b]
        )

    m.c3 = pyo.Constraint(m.D0, m.D12, rule=c3)

    def c4(m, e, a, b):
        return m.z0[e, a, b] <= sum(
            m.z2[e, p, q, r]
            for (p, q, r) in m.D15[a, b]
        )

    m.c4 = pyo.Constraint(m.D0, m.D14, rule=c4)

    def c5(m, e, a, b, c):
        return m.z2[e, a, b, c] <= m.z0[e, a, c]

    m.c5 = pyo.Constraint(m.D0, m.D5, rule=c5)

    def c6(m, e, a, b, c):
        return m.z2[e, a, b, c] <= sum(
            m.z1[e, b, q]
            for q in m.D16[a, b, c]
        )

    m.c6 = pyo.Constraint(m.D0, m.D5, rule=c6)

    def c7(m, e, a, b, c, q):
        return m.z2[e, a, b, c] >= (
            m.z0[e, a, c] + m.z1[e, b, q] - 1
        )

    m.c7 = pyo.Constraint(m.D0, m.D6, rule=c7)

    def o(m):
        return sum(
            m.z0[e, a, b]
            for (e, a, b) in m.D1
        )

    m.o = pyo.Objective(rule=o, sense=pyo.maximize)

    return m

# Standard entry-point alias used by the test suites.
build_pyomo_model = build_model
