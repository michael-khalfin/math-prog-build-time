import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- DATA & PREPROCESSING ---
raw_data = {
    'Zones': ['Z1', 'Z2', 'Z3'], # Z4 removed to satisfy the "no empty sums" rule
    'Facilities': ['F1', 'F2', 'F3'],
    
    # Sparse Relation: Which Facilities can cover which Zone?
    'Coverage': {
        'Z1': ['F1', 'F2'],
        'Z2': ['F2', 'F3'],
        'Z3': ['F1', 'F3']
    },
    
    'MinReq': {'Z1': 1, 'Z2': 1, 'Z3': 2},
    'Cost': {'F1': 500, 'F2': 400, 'F3': 600},
    'Budget': 1000
}

data = raw_data 

# ==========================================
# THE PYOMO TARGET (The Ingestion File)
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    
    m.Z = pyo.Set(initialize=data['Zones'])
    m.F = pyo.Set(initialize=data['Facilities'])
    
    m.Coverage = pyo.Set(m.Z, initialize=data['Coverage'])
    m.Req = pyo.Param(m.Z, initialize=data['MinReq'])
    m.Cost = pyo.Param(m.F, initialize=data['Cost'])
    
    m.build = pyo.Var(m.F, domain=pyo.Binary)

    # PATTERN: Sum over an indexed subset relation
    # STRESS TEST: The loop variable 'fac' differs from the set name 'F' and rule index
    def cover_rule(m, z):
        return sum(m.build[fac] for fac in m.Coverage[z]) >= m.Req[z]
    m.cover_constr = pyo.Constraint(m.Z, rule=cover_rule)
    
    # PATTERN: Scalar constraint using `expr=` instead of `rule=`
    # The team needs to know `expr=` is ONLY allowed for scalar (unindexed) constraints.
    m.budget_constr = pyo.Constraint(
        expr=sum(m.Cost[f] * m.build[f] for f in m.F) <= data['Budget']
    )
    
    return m

# ==========================================
# THE PANDAS EQUIVALENT (The Output Goal)
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()
    
    idx_f = pd.Index(data['Facilities'], name='f')
    df_vars = pd.DataFrame(index=idx_f)
    df_vars = df_vars.gppd.add_vars(m, vtype=gp.GRB.BINARY, name="build")
    
    s_req = pd.Series(data['MinReq']).rename_axis('z')
    s_cost = pd.Series(data['Cost']).rename_axis('f')
    
    # --- Constraint 1: Indexed Subset Relation ---
    mapping_tuples = [(z, f) for z, f_list in data['Coverage'].items() for f in f_list]
    # The transpiler MUST map 'fac' back to 'f' here
    df_relation = pd.DataFrame(mapping_tuples, columns=['z', 'f'])
    
    df_build_flat = df_vars['build'].reset_index()
    df_join = pd.merge(df_relation, df_build_flat, on='f')
    
    lhs_cover = df_join.groupby('z')['build'].sum()
    
    gppd.add_constrs(m, lhs_cover, gp.GRB.GREATER_EQUAL, s_req, name="cover_constr")
    
    # --- Constraint 2: Global expr= Constraint ---
    lhs_budget = (s_cost * df_vars['build']).sum()
    m.addLConstr(lhs_budget <= data['Budget'], name="budget_constr")
    
    return m