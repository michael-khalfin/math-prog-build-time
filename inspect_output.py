from translator import translate
from examples import example_3_multiflow

# Feed the Pyomo function into the translator
translated_string = translate(example_3_multiflow.build_pyomo_model)

print("--- GENERATED PANDAS CODE ---")
print(translated_string)