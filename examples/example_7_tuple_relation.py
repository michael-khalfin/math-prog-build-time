import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- DATA & PREPROCESSING ---
raw_data = {
    'Hubs': ['H1', 'H2'],
    'Origins': ['O1', 'O2'],
    'Destinations': ['D1', 'D2'],
    
    # Sparse Relation: Which (Origin, Destination) pairs can a Hub process?
    'HubCoverage': {
        'H1': [('O1', 'D1'), ('O1', 'D2')],
        'H2': [('O2', 'D1'), ('O2', 'D2')]
    },
    
    'HubCap': {'H1': 1000, 'H2': 850}
}

data = raw_data

# ==========================================
# THE PYOMO TARGET (The Ingestion File)
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    
    m.H = pyo.Set(initialize=data['Hubs'])
    m.O = pyo.Set(initialize=data['Origins'])
    m.D = pyo.Set(initialize=data['Destinations'])
    
    # Indexed set where the elements are tuples
    m.HubCoverage = pyo.Set(m.H, initialize=data['HubCoverage'])
    m.HubCap = pyo.Param(m.H, initialize=data['HubCap'])
    
    m.ship = pyo.Var(m.O, m.D, domain=pyo.NonNegativeReals)

    # PATTERN: Tuple destructuring in the loop variable
    # STRESS TEST: 'orig' and 'dest' must be mapped back to 'o' and 'd'
    def hub_rule(m, h):
        return sum(m.ship[orig, dest] for orig, dest in m.HubCoverage[h]) <= m.HubCap[h]
    
    m.hub_constr = pyo.Constraint(m.H, rule=hub_rule)
    
    return m

# ==========================================
# THE PANDAS EQUIVALENT (The Output Goal)
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()
    
    idx_od = pd.MultiIndex.from_product([data['Origins'], data['Destinations']], names=['o', 'd'])
    df_vars = pd.DataFrame(index=idx_od)
    df_vars = df_vars.gppd.add_vars(m, name="ship")
    
    s_cap = pd.Series(data['HubCap']).rename_axis('h')
    
    # --- Constraint 1: Tuple Relation ---
    # 1. Flatten the dictionary into a 3-column relation table
    mapping_tuples = [(h, o, d) for h, od_list in data['HubCoverage'].items() for o, d in od_list]
    df_relation = pd.DataFrame(mapping_tuples, columns=['h', 'o', 'd'])
    
    # 2. Extract variable and merge on BOTH dimensions
    df_ship_flat = df_vars['ship'].reset_index()
    df_join = pd.merge(df_relation, df_ship_flat, on=['o', 'd'])
    
    # 3. Group by the outer rule index 'h'
    lhs_hub = df_join.groupby('h')['ship'].sum()
    
    # 4. Reindex to ensure no dropped hubs
    idx_h = pd.Index(data['Hubs'], name='h')
    lhs_hub = lhs_hub.reindex(idx_h, fill_value=0.0)
    
    gppd.add_constrs(m, lhs_hub, gp.GRB.LESS_EQUAL, s_cap, name="hub_constr")
    
    return m