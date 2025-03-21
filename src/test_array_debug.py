from grid_lang.interpreter import Interpreter
from grid_lang.parser import Parser
from grid_lang.lexer import Lexer

def debug_array_assignment():
    code = """
    [@B2] := {12, 34, 56}
    """
    
    print("Testing array assignment:")
    print(code)
    
    # Tokenize
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    
    print("\nTokens:")
    for token in tokens:
        print(token)
    
    # Parse
    parser = Parser(tokens)
    ast = parser.parse()
    
    print("\nAST:")
    for node in ast:
        print(node)
    
    # Interpret
    interpreter = Interpreter()
    interpreter.interpret(ast)
    
    print("\nRaw grid data:")
    for cell, value in sorted(interpreter.grid.items()):
        print(f"  {cell} = {value}")
    
    # Dump the entire grid object for inspection
    print("\nEntire grid dictionary:")
    print(interpreter.grid)

if __name__ == "__main__":
    debug_array_assignment() 