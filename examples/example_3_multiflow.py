import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- DATA & PREPROCESSING (Allowed outside the function) ---
raw_data = {
    'Nodes': ['N1', 'N2', 'N3'],
    'Commodities': ['Apples', 'Oranges'],
    'Edges': [('N1', 'N2'), ('N2', 'N3'), ('N1', 'N3')],
    'Capacity': {('N1', 'N2'): 10, ('N2', 'N3'): 15, ('N1', 'N3'): 5},
    # Positive is supply, Negative is demand, missing is 0 (transshipment)
    'Demand': {('N1', 'Apples'): 5, ('N3', 'Apples'): -5, 
               ('N1', 'Oranges'): 8, ('N3', 'Oranges'): -8} 
}

def preprocess_data(data):
    """
    Pre-calculate subsets so the Pyomo rules do not need 'if' statements.
    """
    out_arcs = {i: [] for i in data['Nodes']}
    in_arcs = {i: [] for i in data['Nodes']}
    for i, j in data['Edges']:
        out_arcs[i].append(j)
        in_arcs[j].append(i)
    
    data['OutArcs'] = out_arcs
    data['InArcs'] = in_arcs
    
    # Ensure all (Node, Commodity) pairs exist in demand dict to avoid key errors
    for n in data['Nodes']:
        for k in data['Commodities']:
            if (n, k) not in data['Demand']:
                data['Demand'][n, k] = 0.0
                
    return data

data = preprocess_data(raw_data)

# --- THE PYOMO TARGET ---
def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    m.N = pyo.Set(initialize=data['Nodes'])
    m.K = pyo.Set(initialize=data['Commodities'])
    m.E = pyo.Set(dimen=2, initialize=data['Edges'])
    
    # Preprocessed subsets
    m.OutArcs = pyo.Set(m.N, initialize=data['OutArcs'])
    m.InArcs = pyo.Set(m.N, initialize=data['InArcs'])
    
    m.x = pyo.Var(m.E, m.K, domain=pyo.NonNegativeReals)
    m.Cap = pyo.Param(m.E, initialize=data['Capacity'])
    m.Demand = pyo.Param(m.N, m.K, initialize=data['Demand'])

    # Constraint 1: Edge Capacity (sum across all commodities)
    def cap_rule(m, i, j):
        return sum(m.x[i, j, k] for k in m.K) <= m.Cap[i, j]
    m.capacity_constr = pyo.Constraint(m.E, rule=cap_rule)

    # Constraint 2: Node Flow Balance (Flow Out - Flow In == Supply/Demand)
    def flow_rule(m, node, k):
        flow_out = sum(m.x[node, j, k] for j in m.OutArcs[node])
        flow_in  = sum(m.x[i, node, k] for i in m.InArcs[node])
        return flow_out - flow_in == m.Demand[node, k]
    m.flow_constr = pyo.Constraint(m.N, m.K, rule=flow_rule)
    
    return m

# --- THE PANDAS EQUIVALENT ---
def build_vectorized_model(data):
    m = gp.Model()
    
    # Variables: Index is (i, j, k)
    idx_tuples = [(i, j, k) for (i, j) in data['Edges'] for k in data['Commodities']]
    idx_x = pd.MultiIndex.from_tuples(idx_tuples, names=['i', 'j', 'k'])
    df_x = pd.DataFrame(index=idx_x)
    df_x = df_x.gppd.add_vars(m, name="x")
    
    # Params
    s_cap = pd.Series(data['Capacity']).rename_axis(['i', 'j'])
    s_demand = pd.Series(data['Demand']).rename_axis(['node', 'k'])
    
    # Constraint 1: Capacity
    # Pyomo: sum(m.x[i,j,k] for k in m.K) -> Pandas: group by everything EXCEPT k
    lhs_cap = df_x.groupby(['i', 'j'])['x'].sum()
    gppd.add_constrs(m, lhs_cap, gp.GRB.LESS_EQUAL, s_cap, name="capacity")
    
    # Constraint 2: Flow Balance
    # Pyomo flow_out -> Pandas: group by source 'i' and commodity 'k'
    flow_out = df_x.groupby(['i', 'k'])['x'].sum()
    
    # Pyomo flow_in -> Pandas: group by target 'j' and commodity 'k'
    flow_in = df_x.groupby(['j', 'k'])['x'].sum()
    
    # To subtract them, we must rename 'i' and 'j' to match 'node'
    flow_out = flow_out.rename_axis(index={'i': 'node'})
    flow_in = flow_in.rename_axis(index={'j': 'node'})
    
    # Subtract (using fill_value=0 in case a node has only in-edges or only out-edges)
    lhs_flow = flow_out.sub(flow_in, fill_value=0)
    
    gppd.add_constrs(m, lhs_flow, gp.GRB.EQUAL, s_demand, name="flow")
    
    return m