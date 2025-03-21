from grid_lang.interpreter import Interpreter
from grid_lang.parser import Parser
from grid_lang.lexer import Lexer

# Test all three cases
code = """
# Regular cell assignment
[AB2] := 33

# Direct array assignment
[A1] : a = 51
[@B2] := {12, (15 + a) / 1.2e3, 1+[A1], 8}

# Variable array assignment
: b = {100, 200, 300}
[@D2] := b
"""

print("Running comprehensive test...")

# Tokenize
lexer = Lexer(code)
tokens = lexer.tokenize()

print("\nTokens from lexer:")
for i, token in enumerate(tokens):
    print(f"{i}: {token}")
    
    # Print tokens inside array values
    if token.type == 'ARRAY_VALUES':
        print("  Array values:")
        for j, val_token in enumerate(token.value):
            print(f"    {j}: {val_token}")
            if val_token.type == 'EXPRESSION':
                print("      Expression tokens:")
                for k, expr_token in enumerate(val_token.value):
                    print(f"        {k}: {expr_token}")

# Parse
parser = Parser(tokens)
ast = parser.parse()

# Interpret
interpreter = Interpreter()
interpreter.interpret(ast)

print("\nGrid cells:")
for cell, value in sorted(interpreter.grid.items()):
    print(f"{cell} = {value}")

print("\nDetailed variable info:")
for var_name, var_value in interpreter.variables.items():
    print(f"{var_name} = {var_value}")
    print(f"Type: {type(var_value)}")
    if isinstance(var_value, list):
        print(f"List contents: {var_value}")
        for i, item in enumerate(var_value):
            print(f"  Item {i}: {item} (type: {type(item)})")

print("\nTest completed.") 