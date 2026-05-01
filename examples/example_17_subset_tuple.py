import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- 1. DATA & PREPROCESSING ---
raw_data = {
    'Markets': ['M1', 'M2'],
    'Products': ['P1', 'P2'],
    'Plants': ['F1', 'F2'],
    'Whses': ['W1', 'W2'],
    'Suppliers': ['S1', 'S2'],
    
    # R(a, b): To supply (Market, Product), what are the valid (Plant, Whse, Supplier) paths?
    'ValidPaths': {
        ('M1', 'P1'): [('F1', 'W1', 'S1'), ('F2', 'W2', 'S1')],
        ('M2', 'P2'): [('F1', 'W1', 'S2')]
    },
    
    'Demand': {
        ('M1', 'P1'): 100, 
        ('M2', 'P2'): 150
        # Zero-demand / invalid paths removed from preprocessing to prevent empty-sum boolean crashes
    }
}

data = raw_data

# ==========================================
# 2. THE PYOMO TARGET (The Ingestion File)
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    
    m.M = pyo.Set(initialize=data['Markets'])
    m.P = pyo.Set(initialize=data['Products'])
    m.F = pyo.Set(initialize=data['Plants'])
    m.W = pyo.Set(initialize=data['Whses'])
    m.S = pyo.Set(initialize=data['Suppliers'])
    
    # NEW: Define a 2D set of valid target combinations based on our sparse dictionary
    m.ValidMP = pyo.Set(dimen=2, initialize=list(data['ValidPaths'].keys()))
    
    m.ValidPaths = pyo.Set(m.ValidMP, initialize=data['ValidPaths'])
    m.Demand = pyo.Param(m.ValidMP, initialize=data['Demand'])
    
    m.produce = pyo.Var(m.F, m.S, domain=pyo.NonNegativeReals)

    # PATTERN: z_{a,b} = sum_{(u,v,w) in R(a,b)} y_{u,w}
    # Notice we iterate over the restricted domain m.ValidMP
    def demand_rule(m, mkt, prod):
        return sum(m.produce[f, s] for f, w, s in m.ValidPaths[mkt, prod]) >= m.Demand[mkt, prod]
    
    m.demand_constr = pyo.Constraint(m.ValidMP, rule=demand_rule)
    
    return m

# ==========================================
# 3. THE PANDAS EQUIVALENT (The Output Goal)
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()
    
    idx_fs = pd.MultiIndex.from_product([data['Plants'], data['Suppliers']], names=['f', 's'])
    df_vars = pd.DataFrame(index=idx_fs).gppd.add_vars(m, name="produce")
    
    # Limit the Series to only the valid keys
    valid_keys = list(data['ValidPaths'].keys())
    idx_mp = pd.MultiIndex.from_tuples(valid_keys, names=['mkt', 'prod'])
    
    s_demand = pd.Series(data['Demand'], name='demand').reindex(idx_mp)
    
    # --- Constraint: Subset Tuple Relation ---
    mapping_tuples = [
        (mkt, prod, f, w, s) 
        for (mkt, prod), fws_list in data['ValidPaths'].items() 
        for f, w, s in fws_list
    ]
    df_relation = pd.DataFrame(mapping_tuples, columns=['mkt', 'prod', 'f', 'w', 's'])
    
    df_prod_flat = df_vars['produce'].reset_index()
    
    # Join ONLY on the intersection of loop vars and variable indices ('f', 's')
    df_join = pd.merge(df_relation, df_prod_flat, on=['f', 's'])
    
    lhs_demand = df_join.groupby(['mkt', 'prod'])['produce'].sum()
    
    # Reindex using the restricted valid combinations, not the full cartesian product
    lhs_demand = lhs_demand.reindex(idx_mp, fill_value=0.0)
    
    gppd.add_constrs(m, lhs_demand, gp.GRB.GREATER_EQUAL, s_demand, name="demand_constr")
    
    return m