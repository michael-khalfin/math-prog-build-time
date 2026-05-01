"""
Test that the translator generates correct gurobipy-pandas code for each example.
Runs alongside run_tests.py (which tests the hand-written vectorized models).
"""
import sys
import types
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

from translator import translate
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


if __name__ == "__main__":
    examples = [
        (ex1, "Example 1 (Supply)"),
        (ex2, "Example 2 (Network)"),
        (ex3, "Example 3 (Multiflow)"),
        (ex4, "Example 4 (BOM)"),
        (ex5, "Example 5 (Shifts)"),
        (ex6, "Example 6 (Set Cover)"),
        (ex7, "Example 7 (Tuple Relation)"),
        (ex8, "Example 8 (Index Alignment)"),
        (ex9, "Example 9 (Inline P3)"),
        (ex10, "Example 10 (Weighted Groupby)"),
    ]
    failed = []
    for module, name in examples:
        try:
            test_equivalence(module, name)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append(name)

    print(f"\n{'='*50}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All translator tests PASSED.")
