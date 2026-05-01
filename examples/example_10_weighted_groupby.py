import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- DATA ---
# Worker-machine scheduling: assign hours from workers to machines.
#
# Stresses P1 with a param * var weighted sum over a *non-indexed* set:
#
#   sum(m.Efficiency[w] * m.assign[w, m_] for w in m.W) >= m.MinHours[m_]
#
# The Efficiency param is 1-D (indexed by W) while assign is 2-D (W x M).
# The naive P1 codegen tried `df_assign['_eff'] = s_efficiency` which NaN-fills
# on a MultiIndex.  The correct approach is `.join(s_param, on=shared_dim)`.

data = {
    'Workers':    ['W1', 'W2', 'W3'],
    'Machines':   ['M1', 'M2'],
    'Efficiency': {'W1': 0.8, 'W2': 1.0, 'W3': 0.9},
    'MinHours':   {'M1': 8.0, 'M2': 6.0},
    'MaxHours':   {'W1': 10.0, 'W2': 10.0, 'W3': 8.0},
}

# ==========================================
# THE PYOMO TARGET (The Ingestion File)
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()

    m.W = pyo.Set(initialize=data['Workers'])
    m.M = pyo.Set(initialize=data['Machines'])

    m.Efficiency = pyo.Param(m.W, initialize=data['Efficiency'])
    m.MinHours   = pyo.Param(m.M, initialize=data['MinHours'])
    m.MaxHours   = pyo.Param(m.W, initialize=data['MaxHours'])

    m.assign = pyo.Var(m.W, m.M, domain=pyo.NonNegativeReals)

    # PATTERN: P1 with param * var  (non-indexed set, weighted aggregation)
    # STRESS TEST: Efficiency is 1-D but assign is 2-D — must join on shared dim.
    def machine_rule(m, mach):
        return sum(m.Efficiency[w] * m.assign[w, mach] for w in m.W) >= m.MinHours[mach]
    m.machine_constr = pyo.Constraint(m.M, rule=machine_rule)

    # PATTERN: Plain P1 (no param) — contrast case
    def worker_rule(m, w):
        return sum(m.assign[w, mach] for mach in m.M) <= m.MaxHours[w]
    m.worker_constr = pyo.Constraint(m.W, rule=worker_rule)

    return m

# ==========================================
# THE PANDAS EQUIVALENT (The Output Goal)
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()

    idx_wm = pd.MultiIndex.from_product([data['Workers'], data['Machines']], names=['w', 'm'])
    df_assign = pd.DataFrame(index=idx_wm)
    df_assign = df_assign.gppd.add_vars(m, name='assign')

    s_efficiency = pd.Series(data['Efficiency'], name='efficiency').rename_axis('w')
    s_minhours   = pd.Series(data['MinHours'],   name='minhours').rename_axis('m')
    s_maxhours   = pd.Series(data['MaxHours'],   name='maxhours').rename_axis('w')

    # --- Constraint 1: machine_constr (P1 weighted) ---
    # Join the 1-D efficiency param onto the 2-D (w, m) frame on the shared dim 'w',
    # then do a weighted groupby over the constraint index 'm'.
    _df_machine_constr_w = df_assign.join(s_efficiency, on='w')
    lhs_machine_constr = (_df_machine_constr_w['efficiency'] * _df_machine_constr_w['assign']).groupby('m').sum()
    gppd.add_constrs(m, lhs_machine_constr, gp.GRB.GREATER_EQUAL, s_minhours, name='machine_constr')

    # --- Constraint 2: worker_constr (P1 plain) ---
    lhs_worker_constr = df_assign.groupby('w')['assign'].sum()
    gppd.add_constrs(m, lhs_worker_constr, gp.GRB.LESS_EQUAL, s_maxhours, name='worker_constr')

    return m
