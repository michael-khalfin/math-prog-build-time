"""
Test that the translator generates correct gurobipy-pandas code for each example.
Runs alongside run_tests.py (which tests the hand-written vectorized models).
"""
import sys
import examples.example_1_supply as ex1
import examples.example_2_network as ex2
import examples.example_3_multiflow as ex3
import examples.example_4_bom as ex4
import examples.example_5_shifts as ex5
import examples.example_6_set_cover as ex6
import examples.example_7_tuple_relation as ex7
import examples.example_8_index_alignment as ex8
import examples.example_9_inline_p3 as ex9
import examples.example_10_weighted_groupby as ex10
import examples.example_11_multi_term as ex11
import examples.example_12_intra_sum as ex12
import examples.example_13_inter_sum as ex13
import examples.example_14_three_var_intra as ex14
import examples.example_15_mixed_shape_intra as ex15
import examples.example_16_indexed_subset as ex16
import examples.example_17_subset_tuple as ex17
import examples.example_18_lhs_equality as ex18
import examples.example_19_jk_secretary as ex19

from translator import translate, solve, populate_pyomo
from pyomo.environ import SolverFactory
import gurobipy as gp


def build_from_translated(module):
    code = translate(module.build_pyomo_model)
    print("--- Generated code ---")
    print(code)
    print("----------------------")
    ns = {}
    exec(compile(code, "<translated>", "exec"), ns)
    return ns['build_vectorized_model'](module.data)


def test_equivalence(module, name):
    print(f"\nTesting translator output for {name}...")

    pyo_model = module.build_pyomo_model(module.data)
    opt = SolverFactory('gurobi_persistent')
    opt.set_instance(pyo_model)
    gurobi_from_pyomo = opt._solver_model
    gurobi_from_pyomo.update()

    gppd_model = build_from_translated(module)
    gppd_model.update()

    print(f"  Pyomo      -> Vars: {gurobi_from_pyomo.NumVars}, Constrs: {gurobi_from_pyomo.NumConstrs}")
    print(f"  Translated -> Vars: {gppd_model.NumVars}, Constrs: {gppd_model.NumConstrs}")

    assert gurobi_from_pyomo.NumVars == gppd_model.NumVars, \
        f"Variable mismatch! Pyomo={gurobi_from_pyomo.NumVars} Translated={gppd_model.NumVars}"
    assert gurobi_from_pyomo.NumConstrs == gppd_model.NumConstrs, \
        f"Constraint mismatch! Pyomo={gurobi_from_pyomo.NumConstrs} Translated={gppd_model.NumConstrs}"
    print(f"  [PASS] {name}")


def test_solve_api(module, name):
    """Test solve() and populate_pyomo():
    1. Objective values match between Pyomo+gurobi_persistent and solve().
    2. populate_pyomo() correctly round-trips all variable values.
    """
    print(f"\nTesting solve/populate API for {name}...")

    # --- Pyomo path ---
    pyo_model = module.build_pyomo_model(module.data)
    opt = SolverFactory('gurobi_persistent')
    opt.set_instance(pyo_model)
    opt.solve()
    pyo_gurobi = opt._solver_model
    pyo_gurobi.update()

    # --- Translated solve ---
    gp_model, values = solve(module.build_pyomo_model, module.data)

    if gp_model.SolCount == 0:
        # Model is infeasible / unbounded — verify Pyomo agrees and skip value checks
        assert pyo_gurobi.SolCount == 0, (
            "solve() returned no solution but Pyomo did — models disagree on feasibility"
        )
        print(f"  (infeasible — skipping value checks)")
        print(f"  [PASS] {name}")
        return

    # Compare objective values (both models should be optimally solved)
    if pyo_gurobi.SolCount > 0:
        assert abs(pyo_gurobi.ObjVal - gp_model.ObjVal) < 1e-4, (
            f"Objective mismatch: Pyomo={pyo_gurobi.ObjVal:.6f} "
            f"Translated={gp_model.ObjVal:.6f}"
        )

    # --- populate_pyomo round-trip ---
    pyo_model2 = module.build_pyomo_model(module.data)
    populate_pyomo(pyo_model2, values)
    for var_name, series in values.items():
        pyo_var = getattr(pyo_model2, var_name)
        for idx, val in series.items():
            pyomo_val = pyo_var[idx].value
            assert abs(pyomo_val - val) < 1e-9, (
                f"{var_name}[{idx}]: populate_pyomo set {pyomo_val}, expected {val}"
            )

    print(f"  [PASS] {name}")


if __name__ == "__main__":
    examples = [
        (ex1,  "Example 1  (Supply)"),
        (ex2,  "Example 2  (Network)"),
        (ex3,  "Example 3  (Multiflow)"),
        (ex4,  "Example 4  (BOM)"),
        (ex5,  "Example 5  (Shifts)"),
        (ex6,  "Example 6  (Set Cover)"),
        (ex7,  "Example 7  (Tuple Relation)"),
        (ex8,  "Example 8  (Index Alignment)"),
        (ex9,  "Example 9  (Inline P3)"),
        (ex10, "Example 10 (Weighted Groupby)"),
        (ex11, "Example 11 (Multi-Term)"),
        (ex12, "Example 12 (Intra-Sum)"),
        (ex13, "Example 13 (Inter-Sum)"),
        (ex14, "Example 14 (Three-Var Intra)"),
        (ex15, "Example 15 (Mixed-Shape Intra)"),
        (ex16, "Example 16 (Indexed Subset)"),
        (ex17, "Example 17 (Subset Tuple)"),
        (ex18, "Example 18 (LHS Equality)"),
        (ex19, "Example 19 (JK Secretary)"),
    ]

    failed = []

    print("=" * 60)
    print("PHASE 1: Structural equivalence (NumVars / NumConstrs)")
    print("=" * 60)
    for module, name in examples:
        try:
            test_equivalence(module, name)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append(f"{name} [equivalence]")

    print()
    print("=" * 60)
    print("PHASE 2: solve() + populate_pyomo() API")
    print("=" * 60)
    for module, name in examples:
        try:
            test_solve_api(module, name)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append(f"{name} [solve API]")

    print(f"\n{'=' * 60}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All translator tests PASSED.")
