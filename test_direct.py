from grid_lang import run_grid

test_code = """[A1] : a = 51
[A2] : b = (a + 15) / 1.2e3
[A3] : c = sum[A1:A2]"""

run_grid(test_code) 