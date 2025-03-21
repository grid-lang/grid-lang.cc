from grid_lang.interpreter import Interpreter
from grid_lang.parser import Parser
from grid_lang.lexer import Lexer

# Test both direct array assignment and variable array assignment
code1 = """
[A1] : a = 51
[@B2] := {12, (15 + a) / 1.2e3, 1+[A1], 8}
"""

code2 = """
[A1] : a = 51
: b = {12, (15 + a) / 1.2e3, 1+[A1], 8}
[@B2] := b
"""

def run_test(code, title):
    print(f"\n===== Testing {title} =====")
    print(f"Code:\n{code}")
    
    # Tokenize, parse, and interpret
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    
    interpreter = Interpreter()
    interpreter.interpret(ast)
    
    print("\nGrid cells:")
    for cell, value in sorted(interpreter.grid.items()):
        print(f"{cell} = {value}")
    
    print(f"\n===== {title} test completed =====")

# Run both tests
run_test(code1, "Direct Array Assignment")
run_test(code2, "Variable Array Assignment") 