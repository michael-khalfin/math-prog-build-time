import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- DATA & PREPROCESSING ---
raw_data = {
    'Plants': ['P1', 'P2'],
    'Warehouses': ['W1', 'W2'],
    'Customers': ['C1', 'C2'],
    
    # Limit on total flow from a specific Plant to a specific Customer (ignoring warehouse)
    'PathLimit': {
        ('P1', 'C1'): 100, ('P1', 'C2'): 150,
        ('P2', 'C1'): 200, ('P2', 'C2'): 250
    }
}

data = raw_data

# ==========================================
# THE PYOMO TARGET (The Ingestion File)
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    
    # Long set names
    m.Plants = pyo.Set(initialize=data['Plants'])
    m.Whse = pyo.Set(initialize=data['Warehouses'])
    m.Cust = pyo.Set(initialize=data['Customers'])
    
    m.PathLimit = pyo.Param(m.Plants, m.Cust, initialize=data['PathLimit'])
    
    # 3D Variable
    m.ship = pyo.Var(m.Plants, m.Whse, m.Cust, domain=pyo.NonNegativeReals)

    # PATTERN: 3D variable aggregated over the middle dimension
    # STRESS TEST: The names of the constraint indices ('p', 'c') and loop index ('w') 
    # must perfectly match the names used when building the pd.MultiIndex for 'm.ship'.
    def path_rule(m, p, c):
        return sum(m.ship[p, w, c] for w in m.Whse) <= m.PathLimit[p, c]
    
    m.path_constr = pyo.Constraint(m.Plants, m.Cust, rule=path_rule)
    
    return m

# ==========================================
# THE PANDAS EQUIVALENT (The Output Goal)
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()
    
    # The transpiler MUST use the registry names ['p', 'w', 'c'] derived from the rules,
    # NOT the set names ['Plants', 'Whse', 'Cust']
    idx_pwc = pd.MultiIndex.from_product(
        [data['Plants'], data['Warehouses'], data['Customers']], 
        names=['p', 'w', 'c']
    )
    df_vars = pd.DataFrame(index=idx_pwc)
    df_vars = df_vars.gppd.add_vars(m, name="ship")
    
    s_limit = pd.Series(data['PathLimit']).rename_axis(['p', 'c'])
    
    # --- Constraint 1: Multi-Dimensional Groupby ---
    # Because 'ship' is indexed natively by ['p', 'w', 'c'], grouping by the outer 
    # constraint indices ['p', 'c'] automatically sums over 'w'.
    # If the index names don't match exactly, this throws a KeyError.
    lhs_path = df_vars.groupby(['p', 'c'])['ship'].sum()
    
    gppd.add_constrs(m, lhs_path, gp.GRB.LESS_EQUAL, s_limit, name="path_constr")
    
    return m