"""Sanity suite for the HAND-WRITTEN vectorized models in examples/.

Compares variable/constraint counts between each example's Pyomo model and its
hand-written build_vectorized_model (the early, pre-transpiler baselines).
It does NOT exercise the transpiler.  Test hierarchy:

    differential_test.py  — primary oracle: structural-signature equivalence
                            of transpiled vs Pyomo models (run this first)
    test_translator.py    — transpiler API tests: solve(), populate_pyomo(),
                            update_vectorized_model()
    run_tests.py          — this file: hand-written example sanity only
"""
import examples.example_1_supply as example_1_supply
import examples.example_2_network as example_2_network
import examples.example_3_multiflow as example_3_multiflow
import examples.example_4_bom as example_4_bom
import examples.example_5_shifts as example_5_shifts
import examples.example_6_set_cover as example_6_set_cover
import examples.example_7_tuple_relation as example_7_tuple_relation
import examples.example_8_index_alignment as example_8_index_alignment
import examples.example_11_multi_term as example_11_multi_term
import examples.example_12_intra_sum as example_12_intra_sum
import examples.example_13_inter_sum as example_13_inter_sum
import examples.example_17_subset_tuple as example_17_subset_tuple
import examples.example_18_lhs_equality as example_18_lhs_equality
import examples.example_19_jk_secretary as example_19_jk_secretary
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
    test_model_equivalence(example_6_set_cover, "Example 6 (Set Cover)")
    test_model_equivalence(example_7_tuple_relation, "Example 7 (Tuple Relation)")
    test_model_equivalence(example_8_index_alignment, "Example 8 (Index Alignment)")
    test_model_equivalence(example_11_multi_term, "Example 11 (Multi-Term)")
    test_model_equivalence(example_12_intra_sum, "Example 12 (Intra-Sum)")
    test_model_equivalence(example_13_inter_sum, "Example 13 (Inter-Sum)")
    test_model_equivalence(example_17_subset_tuple, "Example 17 (Subset Tuple)")
    test_model_equivalence(example_18_lhs_equality, "Example 18 (LHS Equality)")
    test_model_equivalence(example_19_jk_secretary, "Example 19 (JK Secretary)")