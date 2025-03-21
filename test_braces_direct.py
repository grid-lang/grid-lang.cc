from grid_lang import run_grid

test_code = """' Test different uses of curly braces
[A1:C1] := {5, 6, 7}  ' Range-based array assignment
[@A1] : list = {5, 6, 7}  ' Named array assignment
: a = 51
: b = 15
: c = sum{a, b}  ' Function call with arguments"""

run_grid(test_code) 