from grid_lang import run_grid

test_code = """
[A1] := 5
: a = 10
[@B2] := {12, (15 + a) / 1.2e3, 1+[A1], 8}  # Array with expressions
"""

run_grid(test_code) 