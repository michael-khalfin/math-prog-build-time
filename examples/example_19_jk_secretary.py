import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- 1. DATA & PREPROCESSING ---
raw_data = {
    'Applicants': ['A1', 'A2', 'A3', 'A4'],
    'Positions': ['P1', 'P2'],
    'Scores': {
        ('A1', 'P1'): 0.8, ('A1', 'P2'): 0.4,
        ('A2', 'P1'): 0.9, ('A2', 'P2'): 0.5,
        ('A3', 'P1'): 0.7, ('A3', 'P2'): 0.9,
        ('A4', 'P1'): 0.6, ('A4', 'P2'): 0.8
    },
    # Online Thresholding: To evaluate applicant 'a', we sum over all previous assignments
    'PastAssign': {
        'A1': [],
        'A2': [('A1', 'P1'), ('A1', 'P2')],
        'A3': [('A1', 'P1'), ('A1', 'P2'), ('A2', 'P1'), ('A2', 'P2')],
        'A4': [('A1', 'P1'), ('A1', 'P2'), ('A2', 'P1'), ('A2', 'P2'), ('A3', 'P1'), ('A3', 'P2')]
    }
}

data = raw_data

# ==========================================
# 2. THE PYOMO TARGET
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    
    m.A = pyo.Set(initialize=data['Applicants'])
    m.P = pyo.Set(initialize=data['Positions'])
    m.Scores = pyo.Param(m.A, m.P, initialize=data['Scores'])
    
    # Sparse 2D Set for past relation mapping
    m.ValidA = pyo.Set(initialize=list(data['PastAssign'].keys()))
    m.PastAssign = pyo.Set(m.ValidA, initialize=data['PastAssign'])
    
    m.x = pyo.Var(m.A, m.P, domain=pyo.NonNegativeReals, bounds=(0, 1))
    m.thresh = pyo.Var(m.ValidA, domain=pyo.NonNegativeReals)

    # Standard assignments
    def one_per_pos_rule(m, p):
        return sum(m.x[a, p] for a in m.A) <= 1
    m.one_per_pos = pyo.Constraint(m.P, rule=one_per_pos_rule)

    def one_per_app_rule(m, a):
        return sum(m.x[a, p] for p in m.P) <= 1
    m.one_per_app = pyo.Constraint(m.A, rule=one_per_app_rule)

    # PATTERN 1: LHS Equality + Tuple Unpacking
    def threshold_rule(m, a):
        return m.thresh[a] == sum(m.x[past_a, p] for past_a, p in m.PastAssign[a])
    m.threshold_constr = pyo.Constraint(m.ValidA, rule=threshold_rule)

    # PATTERN 2: Multi-term Inter-Sum Objective
    m.obj = pyo.Objective(
        expr=sum(m.Scores[a, p] * m.x[a, p] for a in m.A for p in m.P) - sum(0.1 * m.thresh[a] for a in m.ValidA),
        sense=pyo.maximize
    )
    
    return m

# ==========================================
# 3. THE PANDAS EQUIVALENT
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()
    
    idx_ap = pd.MultiIndex.from_product([data['Applicants'], data['Positions']], names=['a', 'p'])
    df_x = pd.DataFrame(index=idx_ap).gppd.add_vars(m, lb=0, ub=1, name="x")
    
    idx_a = pd.Index(list(data['PastAssign'].keys()), name='a')
    df_thresh = pd.DataFrame(index=idx_a).gppd.add_vars(m, lb=0, name="thresh")
    
    s_scores = pd.Series(data['Scores'], name='scores').rename_axis(['a', 'p'])
    
    # --- Standard Assignment Constraints ---
    gppd.add_constrs(m, df_x.groupby('p')['x'].sum(), gp.GRB.LESS_EQUAL, 1.0, name="one_per_pos")
    gppd.add_constrs(m, df_x.groupby('a')['x'].sum(), gp.GRB.LESS_EQUAL, 1.0, name="one_per_app")
    
    # --- LHS Equality + Tuple Unpacking ---
    mapping_tuples = [
        (a_curr, a_past, p) 
        for a_curr, past_list in data['PastAssign'].items() 
        for a_past, p in past_list
    ]
    df_rel = pd.DataFrame(mapping_tuples, columns=['a', 'past_a', 'p'])
    
    # Rename 'a' to 'past_a' in the flattened variable df so the merge aligns correctly
    df_x_flat = df_x.reset_index().rename(columns={'a': 'past_a'})
    
    df_join = pd.merge(df_rel, df_x_flat, on=['past_a', 'p'])
    rhs_sum = df_join.groupby('a')['x'].sum()
    rhs_sum = rhs_sum.reindex(df_thresh.index, fill_value=0.0)
    
    gppd.add_constrs(m, df_thresh['thresh'], gp.GRB.EQUAL, rhs_sum, name="threshold_constr")
    
    # --- Multi-term Objective ---
    df_x_cost = df_x.join(s_scores)
    term1 = (df_x_cost['x'] * df_x_cost['scores']).sum()
    term2 = (df_thresh['thresh'] * 0.1).sum()
    
    m.setObjective(term1 - term2, gp.GRB.MAXIMIZE)
    
    return m