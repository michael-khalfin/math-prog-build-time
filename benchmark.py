import time
import random
import pandas as pd
import gurobipy as gp
from pyomo.environ import SolverFactory
from examples import example_3_multiflow

def generate_massive_data(num_nodes=10**6, num_edges=2*10**6, num_commodities=10):
    print(f"Generating synthetic data: {num_nodes} nodes, {num_edges} edges...")
    
    nodes = [f"N_{i}" for i in range(num_nodes)]
    commodities = [f"C_{k}" for k in range(num_commodities)]
    
    edges = set()
    while len(edges) < num_edges:
        i = random.choice(nodes)
        j = random.choice(nodes)
        if i != j:
            edges.add((i, j))
    edges = list(edges)
    
    capacity = {e: random.randint(10, 100) for e in edges}
    
    # Isolate only connected nodes to prevent Pyomo constraint evaluation errors
    connected_nodes = list(set([i for i, j in edges] + [j for i, j in edges]))
    
    demand = {}
    for _ in range(len(connected_nodes) * 2): 
        n = random.choice(connected_nodes)
        c = random.choice(commodities)
        demand[(n, c)] = random.randint(-50, 50)
        
    raw_data = {
        'Nodes': connected_nodes,
        'Commodities': commodities,
        'Edges': edges,
        'Capacity': capacity,
        'Demand': demand
    }
    
    return example_3_multiflow.preprocess_data(raw_data)

if __name__ == "__main__":
    # Crank these numbers up further if you want to test the absolute memory limits
    huge_data = generate_massive_data(10**6, 2*10**6, 10)
    
    # 0. Eat the Gurobi license check overhead
    print("\n[Initializing Gurobi License...]")
    _ = gp.Model() 
    
    print("\n--- 1. Testing Pyomo (AST Build + Matrix Compilation) ---")
    t0 = time.time()
    pyomo_model = example_3_multiflow.build_pyomo_model(huge_data)
    
    # Force Pyomo to build the C-backend matrix
    opt = SolverFactory('gurobi_persistent')
    opt.set_instance(pyomo_model) 
    
    t1 = time.time()
    pyomo_time = t1 - t0
    print(f"Pyomo True Build Time: {pyomo_time:.2f} seconds")
    
    print("\n--- 2. Testing Gurobipy-Pandas (Vectorized C-API) ---")
    t0 = time.time()
    gppd_model = example_3_multiflow.build_vectorized_model(huge_data)
    t1 = time.time()
    gppd_time = t1 - t0
    print(f"Pandas True Build Time: {gppd_time:.2f} seconds")
    
    print(f"\nSpeedup: {pyomo_time / gppd_time:.1f}x faster")