from grid_lang import run_grid

test_code = """
[A1] := 5
[A2] := 7
: x = sum{[A1], [A2]}  # Sum of individual cells
: y = sum[A1:A2]       # Sum of cell range
: z = sum([A1:A2])     # Sum of cell range with parentheses
"""

run_grid(test_code) 