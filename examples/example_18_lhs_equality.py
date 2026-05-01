import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- 1. DATA & PREPROCESSING ---
raw_data = {
    'A_Idx': ['A1', 'A2'],
    'B_Idx': ['B1', 'B2'],
    'U_Idx': ['U1', 'U2'],
    'V_Idx': ['V1', 'V2'],
    'W_Idx': ['W1', 'W2'],
    
    # R(a, b): To fulfill target (a, b), what (u, v, w) combinations are used?
    'Rel': {
        ('A1', 'B1'): [('U1', 'V1', 'W1'), ('U2', 'V2', 'W1')],
        ('A2', 'B2'): [('U1', 'V2', 'W2')]
    }
}

data = raw_data

# ==========================================
# 2. THE PYOMO TARGET (The Ingestion File)
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    
    m.A = pyo.Set(initialize=data['A_Idx'])
    m.B = pyo.Set(initialize=data['B_Idx'])
    m.U = pyo.Set(initialize=data['U_Idx'])
    m.V = pyo.Set(initialize=data['V_Idx'])
    m.W = pyo.Set(initialize=data['W_Idx'])
    
    # NEW: Define the sparse 2D set for valid combinations
    m.ValidAB = pyo.Set(dimen=2, initialize=list(data['Rel'].keys()))
    
    # Restrict the relation set to only valid outer keys
    m.Rel = pyo.Set(m.ValidAB, initialize=data['Rel'])
    
    # LHS Variable: Restrict to the exact same sparse outer constraint keys
    m.z = pyo.Var(m.ValidAB, domain=pyo.NonNegativeReals)
    
    # RHS Variable: Drops dimension 'v'
    m.y = pyo.Var(m.U, m.W, domain=pyo.NonNegativeReals)

    # PATTERN: z_{a,b} == sum_{(u,v,w) in R(a,b)} y_{u,w}
    # Iterate over ValidAB to avoid KeyError
    def balance_rule(m, a, b):
        return m.z[a, b] == sum(m.y[u, w] for u, v, w in m.Rel[a, b])
    
    m.balance_constr = pyo.Constraint(m.ValidAB, rule=balance_rule)
    
    return m

# ==========================================
# 3. THE PANDAS EQUIVALENT (The Output Goal)
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()
    
    # Initialize LHS Variable strictly over valid sparse keys
    valid_keys = list(data['Rel'].keys())
    idx_ab = pd.MultiIndex.from_tuples(valid_keys, names=['a', 'b'])
    df_z = pd.DataFrame(index=idx_ab).gppd.add_vars(m, name="z")
    
    # Initialize RHS Variable
    idx_uw = pd.MultiIndex.from_product([data['U_Idx'], data['W_Idx']], names=['u', 'w'])
    df_y = pd.DataFrame(index=idx_uw).gppd.add_vars(m, name="y")
    
    # --- Constraint: Equality Projection ---
    mapping_tuples = [
        (a, b, u, v, w) 
        for (a, b), uvw_list in data['Rel'].items() 
        for u, v, w in uvw_list
    ]
    df_rel = pd.DataFrame(mapping_tuples, columns=['a', 'b', 'u', 'v', 'w'])
    
    df_y_flat = df_y.reset_index()
    
    # Merge ONLY on the intersection ('u', 'w'), dropping 'v'
    df_join = pd.merge(df_rel, df_y_flat, on=['u', 'w'])
    
    # Group by the outer rule indices ('a', 'b') to form the RHS
    rhs_sum = df_join.groupby(['a', 'b'])['y'].sum()
    
    # Reindex the RHS to perfectly match the restricted LHS DataFrame index
    rhs_sum = rhs_sum.reindex(df_z.index, fill_value=0.0)
    
    # Form the equality constraint between the LHS Var Series and the RHS Sum Series
    gppd.add_constrs(m, df_z['z'], gp.GRB.EQUAL, rhs_sum, name="balance_constr")
    
    return m