import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- DATA & PREPROCESSING ---
raw_data = {
    'Stores': ['Store_A', 'Store_B'],
    'Hours': [8, 9, 10, 11, 12, 13], # 8 AM to 1 PM
    'ShiftLength': 3,                # A shift covers the start hour + 2 subsequent hours
    
    # Minimum workers needed at each store, each hour
    'Demand': {
        ('Store_A', 8): 2, ('Store_A', 9): 3, ('Store_A', 10): 5, 
        ('Store_A', 11): 5, ('Store_A', 12): 4, ('Store_A', 13): 2,
        ('Store_B', 8): 1, ('Store_B', 9): 2, ('Store_B', 10): 3, 
        ('Store_B', 11): 4, ('Store_B', 12): 4, ('Store_B', 13): 3
    }
}

def preprocess_data(data):
    """
    To avoid `t - tau` algebra inside Pyomo rules, we pre-calculate a mapping 
    of every hour 't' to the valid past start times 'tp' that would cover it.
    """
    length = data['ShiftLength']
    
    valid_starts = {}
    for t in data['Hours']:
        # A shift starting at `tp` covers `t` if: tp <= t < tp + length
        valid_starts[t] = [tp for tp in data['Hours'] if tp <= t < tp + length]
        
    data['ValidStarts'] = valid_starts
    return data

data = preprocess_data(raw_data)

# ==========================================
# THE PYOMO TARGET (The Ingestion File)
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    
    m.S = pyo.Set(initialize=data['Stores'])
    m.T = pyo.Set(initialize=data['Hours'])
    
    # The preprocessed rolling window map
    m.ValidStarts = pyo.Set(m.T, initialize=data['ValidStarts'])
    
    m.Demand = pyo.Param(m.S, m.T, initialize=data['Demand'])
    
    # Variable: Number of shifts *starting* at store `s` at hour `t`
    m.starts = pyo.Var(m.S, m.T, domain=pyo.NonNegativeIntegers)

    # Constraint: Coverage at hour `t` must meet demand
    def cover_rule(m, s, t):
        # Coverage is the sum of all valid shifts that started at or before `t`
        coverage = sum(m.starts[s, tp] for tp in m.ValidStarts[t])
        return coverage >= m.Demand[s, t]
        
    m.cover_constr = pyo.Constraint(m.S, m.T, rule=cover_rule)
    
    return m

# ==========================================
# THE PANDAS EQUIVALENT (The Output Goal)
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()
    
    idx_st = pd.MultiIndex.from_product([data['Stores'], data['Hours']], names=['s', 't'])
    
    df_vars = pd.DataFrame(index=idx_st)
    df_vars = df_vars.gppd.add_vars(m, vtype=gp.GRB.INTEGER, name="starts")
    
    s_demand = pd.Series(data['Demand']).rename_axis(['s', 't'])
    
    # --- Rolling Window Vectorization ---
    # 1. Convert the dictionary map into a flat DataFrame mapping (t, tp)
    mapping_tuples = [(t, tp) for t, start_list in data['ValidStarts'].items() for tp in start_list]
    df_time_map = pd.DataFrame(mapping_tuples, columns=['t', 'tp'])
    
    # 2. Extract variables and rename time index to 'tp' (time_past / time_start)
    df_starts = df_vars['starts'].reset_index()
    df_starts = df_starts.rename(columns={'t': 'tp'})
    
    # 3. Join variables with the time map on 'tp'
    df_lagged = pd.merge(df_starts, df_time_map, on='tp')
    
    # 4. Group by store 's' and current target hour 't', then sum
    lhs_coverage = df_lagged.groupby(['s', 't'])['starts'].sum()
    
    # THE REINDEX TRAP: 
    # If an hour 't' had zero valid starts in the preprocessed data, 
    # Pandas drops it from the sum. We reindex to match the original MultiIndex, 
    # filling missing sums with 0 to prevent alignment KeyError.
    lhs_coverage = lhs_coverage.reindex(df_vars.index, fill_value=0.0)
    
    gppd.add_constrs(m, lhs_coverage, gp.GRB.GREATER_EQUAL, s_demand, name="coverage")
    
    return m