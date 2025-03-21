from grid_lang import run_grid

test_code = """[A1] := 5
[A2] := 7
: x = sum{A1, A2}"""

run_grid(test_code) 