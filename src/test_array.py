from grid_lang.interpreter import Interpreter
from grid_lang.parser import Parser
from grid_lang.lexer import Lexer
import sys

# Example code
code = """
[A1] : a = 51
: b = {12, (15 + a) / 1.2e3, 1+[A1], 8}
[@B2] := b
"""

print("Running test...")

# Tokenize, parse, and interpret
print("Creating lexer...")
lexer = Lexer(code)

print("Tokenizing...")
tokens = lexer.tokenize()
print(f"Found {len(tokens)} tokens")
for i, t in enumerate(tokens[:10]):  # Show first 10 tokens
    print(f"{i}: {t}")

print("\nParsing...")
parser = Parser(tokens)
ast = parser.parse()
print(f"Found {len(ast)} AST nodes")

print("\nAST nodes:")
for i, node in enumerate(ast):
    print(f"{i}: {node}")

print("\nCreating interpreter...")
interpreter = Interpreter()

print("\nInterpreting...")
try:
    interpreter.interpret(ast)
    print("Interpretation successful")
except Exception as e:
    print(f"Interpretation failed: {e}")
    import traceback
    traceback.print_exc()

print("\nVariable 'b' contents:")
if 'b' in interpreter.variables:
    print(f"b = {interpreter.variables['b']}")
else:
    print("Variable 'b' not found")

print("\nGrid cells:")
for cell, value in sorted(interpreter.grid.items()):
    print(f"{cell} = {value}")

print("\nTest completed.") 