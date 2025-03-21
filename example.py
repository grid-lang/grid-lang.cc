from grid_lang import run_grid

# Read the .grid file
with open('example.grid', 'r') as file:
    code = file.read()

# Run with debug mode enabled to get CSV output
run_grid(code, debug=True) 