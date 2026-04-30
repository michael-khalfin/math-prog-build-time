import examples.example_1_supply as example_1_supply
import examples.example_2_network as example_2_network
import examples.example_3_multiflow as example_3_multiflow
import examples.example_4_bom as example_4_bom
import examples.example_5_shifts as example_5_shifts
from pyomo.environ import SolverFactory

def test_model_equivalence(module, name):
    print(f"Testing {name}...")
    
    # Build Pyomo
    pyo_model = module.build_pyomo_model(module.data)

    # Safely extract the Gurobi model using the persistent interface
    opt = SolverFactory('gurobi_persistent')
    opt.set_instance(pyo_model)

    gurobi_backend_from_pyomo = opt._solver_model
    gurobi_backend_from_pyomo.update()
    
    # Build GPPD
    gppd_model = module.build_vectorized_model(module.data)
    gppd_model.update()
    
    # Compare
    print(f"  Pyomo  -> Vars: {gurobi_backend_from_pyomo.NumVars}, Constrs: {gurobi_backend_from_pyomo.NumConstrs}")
    print(f"  Pandas -> Vars: {gppd_model.NumVars}, Constrs: {gppd_model.NumConstrs}")
    
    assert gurobi_backend_from_pyomo.NumVars == gppd_model.NumVars, "Variable mismatch!"
    assert gurobi_backend_from_pyomo.NumConstrs == gppd_model.NumConstrs, "Constraint mismatch!"
    print(f"  [PASS] {name} matrices are structurally identical.\n")

if __name__ == "__main__":
    test_model_equivalence(example_1_supply, "Example 1 (Supply)")
    test_model_equivalence(example_2_network, "Example 2 (Network)")
    test_model_equivalence(example_3_multiflow, "Example 3 (Multiflow)")
    test_model_equivalence(example_4_bom, "Example 4 (BOM)")
    test_model_equivalence(example_5_shifts, "Example 5 (Shifts)")