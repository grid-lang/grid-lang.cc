from grid_lang.interpreter import Interpreter
from grid_lang.parser import Parser
from grid_lang.lexer import Lexer

def run_test(code, expected_cells=None, expected_vars=None, name="Test"):
    """
    Run a test with the given code and verify expected results.
    
    Args:
        code (str): The GridLang code to execute
        expected_cells (dict): Dictionary of expected cell values
        expected_vars (dict): Dictionary of expected variable values
        name (str): Name of the test for reporting
    """
    print(f"\n=== Running {name} ===")
    print(f"Code:\n{code}")
    
    # Tokenize, parse, and interpret
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    
    interpreter = Interpreter()
    interpreter.interpret(ast)
    
    # Print the raw grid for debugging
    print("\nRaw grid data:")
    for cell, value in sorted(interpreter.grid.items()):
        print(f"  {cell} = {value}")
    
    # Check cells
    if expected_cells:
        errors = 0
        print("\nChecking cell values:")
        for cell, expected_value in expected_cells.items():
            actual_value = interpreter.grid.get(cell, None)
            
            # If the actual value is a list and we expect a single value,
            # we might have an array stored in a cell. Check if this is part
            # of an array assignment that wasn't spread across cells
            if isinstance(actual_value, list) and not isinstance(expected_value, list):
                index = 0
                # Check if the cell name matches what we're looking for
                if cell in interpreter.grid:
                    # Get the position within the array based on column difference
                    cell_col_letter = ''.join(c for c in cell if c.isalpha())
                    cell_row = ''.join(c for c in cell if c.isdigit())
                    expected_col_letter = ''.join(c for c in cell if c.isalpha())
                    expected_row = ''.join(c for c in cell if c.isdigit())
                    
                    # We need to determine which element in the array to check
                    col_diff = ord(expected_col_letter[0]) - ord(cell_col_letter[0])
                    if col_diff >= 0 and col_diff < len(actual_value):
                        # Get the value at the appropriate index in the array
                        actual_element = actual_value[col_diff]
                        match = (isinstance(expected_value, (int, float)) and 
                                isinstance(actual_element, (int, float)) and 
                                abs(actual_element - expected_value) < 0.0001) or actual_element == expected_value
                        
                        if match:
                            print(f"  ✓ {cell}[{col_diff}] = {actual_element}")
                            continue
                        else:
                            print(f"  ✗ {cell}[{col_diff}] = {actual_element}, expected {expected_value}")
                            errors += 1
                            continue
            
            # Normal case - direct value comparison
            if isinstance(expected_value, list) and isinstance(actual_value, list):
                match = all(abs(a - b) < 0.0001 for a, b in zip(actual_value, expected_value)) if len(actual_value) == len(expected_value) else False
            elif isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                match = abs(actual_value - expected_value) < 0.0001
            else:
                match = actual_value == expected_value
                
            if match:
                print(f"  ✓ {cell} = {actual_value}")
            else:
                print(f"  ✗ {cell} = {actual_value}, expected {expected_value}")
                errors += 1
        
        if errors == 0:
            print("  All cell values match expected values!")
        else:
            print(f"  Found {errors} cell value mismatches.")
    
    # Check variables
    if expected_vars:
        errors = 0
        print("\nChecking variable values:")
        for var, expected_value in expected_vars.items():
            actual_value = interpreter.variables.get(var, None)
            if isinstance(expected_value, list) and isinstance(actual_value, list):
                match = all(abs(a - b) < 0.0001 for a, b in zip(actual_value, expected_value)) if len(actual_value) == len(expected_value) else False
            elif isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                match = abs(actual_value - expected_value) < 0.0001
            else:
                match = actual_value == expected_value
                
            if match:
                print(f"  ✓ {var} = {actual_value}")
            else:
                print(f"  ✗ {var} = {actual_value}, expected {expected_value}")
                errors += 1
        
        if errors == 0:
            print("  All variable values match expected values!")
        else:
            print(f"  Found {errors} variable value mismatches.")
    
    print(f"=== {name} completed ===\n")
    return interpreter

# Test 1: Regular cell assignment
test1_code = """
[A1] := 42
"""
test1_expected_cells = {
    "A1": 42.0
}

# Test 2: Variable declaration and simple assignment
test2_code = """
[A1] : x = 100
"""
test2_expected_cells = {
    "A1": 100.0
}
test2_expected_vars = {
    "x": 100.0
}

# Test 3: Direct array assignment
test3_code = """
[A1] := 51
[@B2] := {12, (15 + [A1]) / 1.2e3, 1+[A1], 8}
"""
test3_expected_cells = {
    "A1": 51.0,
    "B2": 12.0,
    "C2": 0.055,
    "D2": 52.0,
    "E2": 8.0
}

# Test 4: Variable array assignment
test4_code = """
: arr = {100, 200, 300}
[@C3] := arr
"""
test4_expected_cells = {
    "C3": 100.0,
    "D3": 200.0,
    "E3": 300.0
}
test4_expected_vars = {
    "arr": [100.0, 200.0, 300.0]
}

# Test 5: Combined test with all methods
test5_code = """
# Regular cell assignments
[A1] := 42
[B1] := 12.5

# Variable declaration and assignment
[C1] : x = 100

# Array with expressions
[@A2] := {10, 20, 30 + [C1], x * 2}

# Variable array
: arr = {5, 10, 15, 20, 25}
[@B3] := arr
"""
test5_expected_cells = {
    "A1": 42.0,
    "B1": 12.5,
    "C1": 100.0,
    "A2": 10.0,
    "B2": 20.0,
    "C2": 130.0,
    "D2": 200.0,
    "B3": 5.0,
    "C3": 10.0,
    "D3": 15.0,
    "E3": 20.0,
    "F3": 25.0
}
test5_expected_vars = {
    "x": 100.0,
    "arr": [5.0, 10.0, 15.0, 20.0, 25.0]
}

# Run all tests
run_test(test1_code, test1_expected_cells, name="Regular Cell Assignment")
run_test(test2_code, test2_expected_cells, test2_expected_vars, name="Variable Declaration")
run_test(test3_code, test3_expected_cells, name="Direct Array Assignment")
run_test(test4_code, test4_expected_cells, test4_expected_vars, name="Variable Array Assignment")
run_test(test5_code, test5_expected_cells, test5_expected_vars, name="Combined Test") 