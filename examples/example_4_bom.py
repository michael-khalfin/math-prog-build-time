import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- DATA & PREPROCESSING ---
raw_data = {
    'Products': ['Car', 'Truck'],
    'Components': ['Wheel', 'Engine', 'Chassis'],
    'Machines': ['Assembly_Line', 'Paint_Shop'],
    'Time': [1, 2],  # The New Time Dimension
    
    # (Product, Component) -> Quantity required
    'BOM': {('Car', 'Wheel'): 4, ('Car', 'Engine'): 1, ('Car', 'Chassis'): 1,
            ('Truck', 'Wheel'): 6, ('Truck', 'Engine'): 2, ('Truck', 'Chassis'): 1},
            
    # (Product, Machine) -> Hours required
    'Routing': {('Car', 'Assembly_Line'): 15, ('Car', 'Paint_Shop'): 5,
                ('Truck', 'Assembly_Line'): 25, ('Truck', 'Paint_Shop'): 8},
                
    'MachineCap': {'Assembly_Line': 500, 'Paint_Shop': 150},
    
    # Demand is now (Product, Time)
    'Demand': {('Car', 1): 10, ('Car', 2): 15, 
               ('Truck', 1): 5, ('Truck', 2): 8},
               
    # New Objective Data: Component Cost
    'CompCost': {'Wheel': 50, 'Engine': 2000, 'Chassis': 1500},
    
    # New Subset Data: Premium Products require a minimum build per period
    'PremiumProducts': ['Car'],
    'MinPremiumBuild': 12
}

def preprocess_data(data):
    """Pre-calculate subsets so Pyomo avoids 'if' statements."""
    prods_using_comp = {c: [] for c in data['Components']}
    for p, c in data['BOM'].keys():
        prods_using_comp[c].append(p)
        
    prods_on_mach = {m: [] for m in data['Machines']}
    for p, m in data['Routing'].keys():
        prods_on_mach[m].append(p)
        
    data['ProdsUsingComp'] = prods_using_comp
    data['ProdsOnMach'] = prods_on_mach
    return data

data = preprocess_data(raw_data)

# ==========================================
# THE PYOMO TARGET (The Ingestion File)
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    
    # Sets
    m.P = pyo.Set(initialize=data['Products'])
    m.C = pyo.Set(initialize=data['Components'])
    m.M = pyo.Set(initialize=data['Machines'])
    m.T = pyo.Set(initialize=data['Time'])
    m.PremiumProducts = pyo.Set(within=m.P, initialize=data['PremiumProducts'])
    
    m.ProdsUsingComp = pyo.Set(m.C, initialize=data['ProdsUsingComp'])
    m.ProdsOnMach = pyo.Set(m.M, initialize=data['ProdsOnMach'])
    
    # Parameters
    m.BOM = pyo.Param(m.P, m.C, initialize=data['BOM'])
    m.Routing = pyo.Param(m.P, m.M, initialize=data['Routing'])
    m.MachineCap = pyo.Param(m.M, initialize=data['MachineCap'])
    m.Demand = pyo.Param(m.P, m.T, initialize=data['Demand'])
    m.CompCost = pyo.Param(m.C, initialize=data['CompCost'])
    m.MinPremium = pyo.Param(initialize=data['MinPremiumBuild'])
    
    # Variables (Now Multi-Period)
    m.build = pyo.Var(m.P, m.T, domain=pyo.NonNegativeReals)
    m.buy_comp = pyo.Var(m.C, m.T, domain=pyo.NonNegativeReals)

    # OBJECTIVE: Minimize component purchasing costs across all time
    def cost_obj(m):
        return sum(m.CompCost[c] * m.buy_comp[c, t] for c in m.C for t in m.T)
    m.TotalCost = pyo.Objective(rule=cost_obj, sense=pyo.minimize)

    # Constraint 1: Meet Demand
    def demand_rule(m, p, t):
        return m.build[p, t] >= m.Demand[p, t]
    m.demand_constr = pyo.Constraint(m.P, m.T, rule=demand_rule)

    # Constraint 2: Component Balance 
    def component_rule(m, c, t):
        return m.buy_comp[c, t] == sum(m.BOM[p, c] * m.build[p, t] for p in m.ProdsUsingComp[c])
    m.component_constr = pyo.Constraint(m.C, m.T, rule=component_rule)

    # Constraint 3: Machine Capacity
    def capacity_rule(m, mach, t):
        return sum(m.Routing[p, mach] * m.build[p, t] for p in m.ProdsOnMach[mach]) <= m.MachineCap[mach]
    m.capacity_constr = pyo.Constraint(m.M, m.T, rule=capacity_rule)
    
    # Constraint 4: Subset Constraint (Premium products minimum)
    def premium_rule(m, p, t):
        return m.build[p, t] >= m.MinPremium
    m.premium_constr = pyo.Constraint(m.PremiumProducts, m.T, rule=premium_rule)
    
    return m

# ==========================================
# THE PANDAS EQUIVALENT (The Output Goal)
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()
    
    # --- Variable Instantiation (Using the fixed .gppd accessor) ---
    idx_pt = pd.MultiIndex.from_product([data['Products'], data['Time']], names=['p', 't'])
    df_build = pd.DataFrame(index=idx_pt)
    df_build = df_build.gppd.add_vars(m, name="build")
    
    idx_ct = pd.MultiIndex.from_product([data['Components'], data['Time']], names=['c', 't'])
    df_buy = pd.DataFrame(index=idx_ct)
    df_buy = df_buy.gppd.add_vars(m, name="buy_comp")
    
    # --- Parameter Instantiation ---
    s_bom = pd.Series(data['BOM'], name='qty').rename_axis(['p', 'c'])
    s_routing = pd.Series(data['Routing'], name='hrs').rename_axis(['p', 'mach'])
    s_mach_cap = pd.Series(data['MachineCap']).rename_axis('mach')
    s_demand = pd.Series(data['Demand']).rename_axis(['p', 't'])
    s_cost = pd.Series(data['CompCost'], name='cost').rename_axis('c')
    
    # --- Objective Function ---
    # Join the 1D cost parameter to the 2D (c, t) variable dataframe on 'c'
    df_buy_cost = df_buy.join(s_cost, on='c')
    obj_expr = (df_buy_cost['cost'] * df_buy_cost['buy_comp']).sum()
    m.setObjective(obj_expr, gp.GRB.MINIMIZE)
    
    # --- Constraint 1: Demand ---
    gppd.add_constrs(m, df_build['build'], gp.GRB.GREATER_EQUAL, s_demand, name="demand")
    
    # --- Constraint 2: Component Balance (The Multi-Dimensional Join) ---
    # We must merge BOM (p, c) with Build (p, t) to create a flat table of (p, c, t)
    df_bom_flat = s_bom.reset_index()
    df_bld_flat = df_build['build'].reset_index()
    df_comp_join = pd.merge(df_bom_flat, df_bld_flat, on='p')
    
    # Multiply and collapse 'p', leaving 'c' and 't'
    lhs_components = (df_comp_join['qty'] * df_comp_join['build']).groupby([df_comp_join['c'], df_comp_join['t']]).sum()
    
    gppd.add_constrs(m, df_buy['buy_comp'], gp.GRB.EQUAL, lhs_components, name="comp_balance")
    
    # --- Constraint 3: Machine Capacity ---
    # Identical dimensional logic to Constraint 2
    df_rout_flat = s_routing.reset_index()
    df_mach_join = pd.merge(df_rout_flat, df_bld_flat, on='p')
    
    lhs_hours = (df_mach_join['hrs'] * df_mach_join['build']).groupby([df_mach_join['mach'], df_mach_join['t']]).sum()
    s_mach_cap_aligned = s_mach_cap.reindex(lhs_hours.index, level='mach')
    
    gppd.add_constrs(m, lhs_hours, gp.GRB.LESS_EQUAL, s_mach_cap_aligned, name="mach_capacity")
    
    # --- Constraint 4: Premium Subset ---
    # Use pd.IndexSlice to safely filter a MultiIndex dataframe keeping all 't'
    premium_idx = pd.IndexSlice[data['PremiumProducts'], :]
    gppd.add_constrs(
        m, 
        df_build.loc[premium_idx, 'build'], 
        gp.GRB.GREATER_EQUAL, 
        data['MinPremiumBuild'], 
        name="premium"
    )
    
    return m