from grid_lang.interpreter import Interpreter
from grid_lang.parser import Parser
from grid_lang.lexer import Lexer

# Example code with direct array assignment
code = """
[A1] : a = 51
[@B2] := {12, (15 + a) / 1.2e3, 1+[A1], 8}
"""

print("Running test...")

# Tokenize, parse, and interpret
print("Creating lexer...")
lexer = Lexer(code)

print("Tokenizing...")
tokens = lexer.tokenize()
print(f"Found {len(tokens)} tokens")
for i, t in enumerate(tokens):
    print(f"{i}: {t}")

print("\nParsing...")
parser = Parser(tokens)
ast = parser.parse()
print(f"Found {len(ast)} AST nodes")

print("\nAST nodes (detailed):")
for i, node in enumerate(ast):
    print(f"{i}: {node}")
    if hasattr(node, '__dict__'):
        for k, v in node.__dict__.items():
            print(f"  {k}: {v}")
            if hasattr(v, '__dict__'):
                for k2, v2 in v.__dict__.items():
                    print(f"    {k2}: {v2}")

print("\nInterpreting...")
interpreter = Interpreter()
interpreter.interpret(ast)

print("\nGrid cells:")
for cell, value in sorted(interpreter.grid.items()):
    print(f"{cell} = {value}")

print("\nTest completed.") 