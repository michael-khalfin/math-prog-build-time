"""
Regression test for P4 naming bug: demand_rule uses (prod, t) so the
registry records P -> ['prod'], but component_rule loops with 'p' in
sum(BOM[p,c]*build[p,t] for p in ProdsUsingComp[c]).  The old name-based
on_key check (loop_v in var_index_names) would fall back to ['p'] which
doesn't exist as a column in _flat_build (which has column 'prod').
"""
import pyomo.environ as pyo

raw_data = {
    'Products':   ['Car', 'Truck'],
    'Components': ['Wheel', 'Engine'],
    'Time':       [1, 2],
    'BOM': {('Car', 'Wheel'): 4, ('Car', 'Engine'): 1,
            ('Truck', 'Wheel'): 6, ('Truck', 'Engine'): 2},
    'Demand': {('Car', 1): 10, ('Car', 2): 15,
               ('Truck', 1): 5,  ('Truck', 2): 8},
    'CompCost': {'Wheel': 50, 'Engine': 2000},
}

def preprocess_data(data):
    prods_using_comp = {c: [] for c in data['Components']}
    for p, c in data['BOM'].keys():
        prods_using_comp[c].append(p)
    data['ProdsUsingComp'] = prods_using_comp
    return data

data = preprocess_data(raw_data)

def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    m.P = pyo.Set(initialize=data['Products'])
    m.C = pyo.Set(initialize=data['Components'])
    m.T = pyo.Set(initialize=data['Time'])
    m.ProdsUsingComp = pyo.Set(m.C, initialize=data['ProdsUsingComp'])

    m.BOM     = pyo.Param(m.P, m.C, initialize=data['BOM'])
    m.Demand  = pyo.Param(m.P, m.T, initialize=data['Demand'])
    m.CompCost = pyo.Param(m.C, initialize=data['CompCost'])

    m.build    = pyo.Var(m.P, m.T, domain=pyo.NonNegativeReals)
    m.buy_comp = pyo.Var(m.C, m.T, domain=pyo.NonNegativeReals)

    def cost_obj(m):
        return sum(m.CompCost[c] * m.buy_comp[c, t] for c in m.C for t in m.T)
    m.TotalCost = pyo.Objective(rule=cost_obj, sense=pyo.minimize)

    # demand_rule uses 'prod' — this registers P -> 'prod' in the registry
    def demand_rule(m, prod, t):
        return m.build[prod, t] >= m.Demand[prod, t]
    m.demand_constr = pyo.Constraint(m.P, m.T, rule=demand_rule)

    # component_rule loops with 'p' — different name from the registry's 'prod'
    def component_rule(m, c, t):
        return m.buy_comp[c, t] == sum(m.BOM[p, c] * m.build[p, t]
                                       for p in m.ProdsUsingComp[c])
    m.component_constr = pyo.Constraint(m.C, m.T, rule=component_rule)

    return m
