import csv
import io
import sys
from .lexer import Lexer
from .parser import Parser
from .interpreter import Interpreter

def run_grid(code, debug=False):
    """Run the Grid language interpreter.
    
    Args:
        code (str): The source code to execute
        debug (bool): If True, outputs the final grid state in CSV format
        
    Returns:
        str: The CSV output if debug is True, otherwise None
    """
    print("Input code:")
    print(code)
    print("\nTokenizing...")
    
    try:
        # Convert source code to tokens
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        print("Tokens:", [str(t) for t in tokens])
        
        # Convert tokens to AST
        print("\nParsing...")
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Execute the AST
        print("\nInterpreting...")
        interpreter = Interpreter()
        interpreter.interpret(ast)
        
        # In debug mode, output the grid state as CSV
        if debug:
            csv_output = interpreter.get_grid_csv()
            print("\nGrid State (CSV format):")
            print(csv_output)
            return csv_output
            
    except Exception as e:
        print(f"Error: {str(e)}")
        if debug:
            # In debug mode, write an empty CSV to indicate failure
            print("\nGrid State (CSV format):")
            print("")  # Empty CSV output on error
            return ""
    
    return None

if __name__ == "__main__":
    test_code = """
    # Grid-based (spatial) data
    [A1] := -5
    [A2] := 10
    [A3] := 15
    
    # Array assignments
    [@B1] : list = {1, 2, 4}  # Will assign to B1, C1, D1 and name it "list"
    [@A4] : numbers = {7, 8, 19}  # Will assign to A4, B4, C4 and name it "numbers"
    
    # Range-based array assignments
    [D1:F1] := {15, 6, 7}  # Will assign to A1, E1, F1
    [A5:D5] := {1, 2, 3, 4}  # Will assign to A2, B2, A3, B3
    
    # Non-spatial data (variables)
    :PI = 3.14159
    :MAX_ROWS = 100
    :TEMP = 25.5
    :N = 9  # Used for dynamic cell references
    
    # Using both spatial and non-spatial data
    [B2] := SUM[A1:A3]  # Should be 30
    [B3] := [A1] + :TEMP  # Should be 30.5
    
    # Dynamic cell references
    [A&{N}] := 42  # Will assign to A5
    [B&{N}] := [A&{N}] + 11  # Will assign to B5
    """
    
    run_grid(test_code) 