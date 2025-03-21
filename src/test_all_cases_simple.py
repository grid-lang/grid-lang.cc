from grid_lang.interpreter import Interpreter
from grid_lang.parser import Parser
from grid_lang.lexer import Lexer

# Simple test focusing on variable array assignments
code = """
: b = {100, 200, 300}
[@D2] := b
"""

print("Running array assignment test...")

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

# Parse and build AST
parser = Parser(tokens)
ast = parser.parse()

print("\nAST structure:")
for i, node in enumerate(ast):
    print(f"Statement {i}:")
    print(f"  Type: {type(node).__name__}")
    if hasattr(node, 'target'):
        print(f"  Target: {node.target.__class__.__name__}")
        if hasattr(node.target, 'name'):
            print(f"    Name: {node.target.name}")
    if hasattr(node, 'value'):
        print(f"  Value: {node.value.__class__.__name__}")
        if hasattr(node.value, 'values'):
            print(f"    Values count: {len(node.value.values)}")
            print("    Values types:")
            for j, val in enumerate(node.value.values):
                print(f"      {j}: {type(val).__name__}")

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