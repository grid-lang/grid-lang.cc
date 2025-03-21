import re

# Token class represents a single token in the source code with position tracking
# Used by the lexer to break down source code into meaningful units
class Token:
    def __init__(self, type, value, line=1, column=1):
        self.type = type      # Token type (e.g., NUMBER, OPERATOR, CELL_ADDRESS)
        self.value = value    # Actual value of the token
        self.line = line      # Line number in source code
        self.column = column  # Column number in source code

    def __str__(self):
        return f"Token({self.type}, {self.value}, line={self.line}, col={self.column})"

# Abstract Syntax Tree (AST) Node classes
# These classes represent the structure of the program after parsing
class Node:
    pass

# Represents an assignment statement (e.g., [A1] := 5 or :X = 10)
class Assignment(Node):
    def __init__(self, target, value, array_name=None):
        self.target = target  # Either CellReference or Variable
        self.value = value    # Expression to be assigned
        self.array_name = array_name  # Name for array assignments

# Represents a cell reference (e.g., [A1])
class CellReference(Node):
    def __init__(self, address):
        self.address = address  # Cell address (e.g., "A1")

# Represents a dynamic cell reference (e.g., A&{N})
class DynamicCellReference(Node):
    def __init__(self, prefix, dynamic_part):
        self.prefix = prefix      # Static part of the address (e.g., "A")
        self.dynamic_part = dynamic_part  # Variable or expression for the dynamic part

# Represents a variable reference (e.g., :X)
# Used for non-spatial data that is not placed on the grid
class Variable(Node):
    def __init__(self, name):
        self.name = name  # Variable name (without the ':' prefix)

# Represents a numeric literal (e.g., 5, 10, 15)
class Number(Node):
    def __init__(self, value):
        self.value = value  # Numeric value

# Represents a binary operation (e.g., A1 + B1)
class BinaryOp(Node):
    def __init__(self, left, op, right):
        self.left = left   # Left operand
        self.op = op       # Operator (+, -, *, /)
        self.right = right # Right operand

# Represents a function call (e.g., SUM[A1:A3])
class Function(Node):
    def __init__(self, name, range):
        self.name = name   # Function name (e.g., "SUM")
        self.range = range # CellRange object

# Represents a range of cells (e.g., A1:A3)
class CellRange(Node):
    def __init__(self, start, end):
        self.start = start  # Starting cell address
        self.end = end      # Ending cell address

# Represents an array of values
class ArrayValues(Node):
    def __init__(self, values):
        self.values = values  # List of numeric values

# Lexer class breaks down source code into tokens
class Lexer:
    def __init__(self, text):
        self.text = text.strip()  # Source code to tokenize
        self.pos = 0              # Current position in text
        self.line = 1             # Current line number
        self.column = 1           # Current column number
        self.tokens = []          # List of tokens
        self.keywords = {'SUM'}   # Reserved keywords

    # Main tokenization method that processes the entire source code
    def tokenize(self):
        while self.pos < len(self.text):
            char = self.text[self.pos]
            
            # Skip comments (lines starting with #)
            if char == '#':
                self._skip_comment()
                continue
                
            # Handle whitespace and newlines
            if char.isspace():
                if char == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.pos += 1
                continue
            # Handle identifiers and keywords
            elif char.isalpha():
                word = self._read_word()
                if word in self.keywords:
                    self.tokens.append(Token('FUNCTION', word, self.line, self.column - len(word)))
                    # Special handling for function calls with cell ranges
                    if self.pos < len(self.text) and self.text[self.pos] == '[':
                        self.pos += 1  # Skip '['
                        self.column += 1
                        self.tokens.append(self._function_cell_address())
                        if self.pos < len(self.text) and self.text[self.pos] == ':':
                            self.tokens.append(Token('RANGE_SEP', ':', self.line, self.column))
                            self.pos += 1
                            self.column += 1
                            self.tokens.append(self._function_cell_address())
                        if self.pos < len(self.text) and self.text[self.pos] == ']':
                            self.pos += 1  # Skip ']'
                            self.column += 1
                else:
                    self.tokens.append(Token('IDENTIFIER', word.upper(), self.line, self.column - len(word)))
            # Handle numbers (including negative numbers)
            elif char.isdigit() or char == '-':
                self.tokens.append(self._number())
            # Handle cell addresses (e.g., [A1])
            elif char == '[':
                self.pos += 1  # Skip '['
                self.column += 1
                # Check for @ symbol for array assignments
                if self.pos < len(self.text) and self.text[self.pos] == '@':
                    self.pos += 1  # Skip '@'
                    self.column += 1
                    self.tokens.append(Token('ARRAY_START', '@', self.line, self.column - 1))
                # Handle the cell address
                cell_token = self._cell_address()
                if cell_token not in self.tokens:  # Only append if not already added
                    self.tokens.append(cell_token)
                # Check for range separator
                if self.pos < len(self.text) and self.text[self.pos] == ':':
                    self.pos += 1  # Skip ':'
                    self.column += 1
                    self.tokens.append(Token('RANGE_SEP', ':', self.line, self.column - 1))
                    # Handle the end cell address
                    end_cell_token = self._cell_address()
                    if end_cell_token not in self.tokens:
                        self.tokens.append(end_cell_token)
                # Skip the closing bracket
                if self.pos < len(self.text) and self.text[self.pos] == ']':
                    self.pos += 1
                    self.column += 1
                # Check for named array assignment
                if self.pos < len(self.text) and self.text[self.pos] == ':':
                    self.pos += 1  # Skip ':'
                    self.column += 1
                    # Read the array name
                    name = self._read_word()
                    self.tokens.append(Token('ARRAY_NAME', name.upper(), self.line, self.column - len(name)))
            # Handle array or variable references (e.g., {X})
            elif char == '{':
                self.tokens.append(self._array_values())
            # Handle operators and range separators
            elif char == ':':
                if self.pos + 1 < len(self.text) and self.text[self.pos + 1] == '=':
                    self.tokens.append(Token('OPERATOR', ':=', self.line, self.column))
                    self.pos += 2
                    self.column += 2
                else:
                    # Check if this is a variable declaration
                    next_pos = self.pos + 1
                    while next_pos < len(self.text) and self.text[next_pos].isspace():
                        next_pos += 1
                    if next_pos < len(self.text) and self.text[next_pos].isalpha():
                        # Only treat as VAR_DECL if it's not after a cell address
                        if not (len(self.tokens) > 0 and self.tokens[-1].type == 'CELL_ADDRESS'):
                            self.tokens.append(Token('VAR_DECL', ':', self.line, self.column))
                            self.pos += 1
                            self.column += 1
                        else:
                            self.tokens.append(Token('ARRAY_NAME_SEP', ':', self.line, self.column))
                            self.pos += 1
                            self.column += 1
                    else:
                        self.tokens.append(Token('RANGE_SEP', ':', self.line, self.column))
                        self.pos += 1
                        self.column += 1
            # Handle arithmetic operators
            elif char in '=+-*/':
                self.tokens.append(Token('OPERATOR', char, self.line, self.column))
                self.column += 1
                self.pos += 1
            else:
                raise SyntaxError(f"Unexpected character: {char} at line {self.line}, column {self.column}")
        return self.tokens

    # Helper methods for tokenization
    def _skip_comment(self):
        while self.pos < len(self.text) and self.text[self.pos] != '\n':
            self.pos += 1

    def _read_word(self):
        result = ''
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            result += self.text[self.pos]
            self.pos += 1
            self.column += 1
        return result

    def _number(self):
        result = ''
        start_col = self.column
        while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] in '-.'):
            result += self.text[self.pos]
            self.pos += 1
            self.column += 1
        return Token('NUMBER', float(result) if '.' in result else int(result), self.line, start_col)

    def _cell_address(self):
        result = ''
        start_col = self.column
        while self.pos < len(self.text) and self.text[self.pos] != ']' and self.text[self.pos] != ':':
            if self.text[self.pos] == '&' and self.pos + 1 < len(self.text) and self.text[self.pos + 1] == '{':
                # Found a dynamic reference
                self.pos += 2  # Skip '&{'
                self.column += 2
                # Create tokens for the dynamic reference
                cell_token = Token('CELL_ADDRESS', result.upper(), self.line, start_col)
                self.tokens.append(cell_token)
                self.tokens.append(Token('DYNAMIC_REF', '&', self.line, self.column - 1))
                self.tokens.append(self._dynamic_value())
                return cell_token
            result += self.text[self.pos]
            self.pos += 1
            self.column += 1
        return Token('CELL_ADDRESS', result.upper(), self.line, start_col)

    def _dynamic_value(self):
        result = ''
        start_col = self.column
        while self.pos < len(self.text) and self.text[self.pos] != '}':
            result += self.text[self.pos]
            self.pos += 1
            self.column += 1
        if self.pos < len(self.text) and self.text[self.pos] == '}':
            self.pos += 1  # Skip '}'
            self.column += 1
        return Token('DYNAMIC_VALUE', result.upper(), self.line, start_col)

    def _array_values(self):
        values = []
        start_col = self.column
        self.pos += 1  # Skip '{'
        self.column += 1
        
        while self.pos < len(self.text) and self.text[self.pos] != '}':
            # Skip whitespace
            while self.pos < len(self.text) and self.text[self.pos].isspace():
                if self.text[self.pos] == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.pos += 1
            
            if self.pos >= len(self.text) or self.text[self.pos] == '}':
                break
                
            # Read a number
            if self.text[self.pos].isdigit() or self.text[self.pos] == '-':
                num_token = self._number()
                values.append(num_token.value)
            else:
                raise SyntaxError(f"Unexpected character in array: {self.text[self.pos]} at line {self.line}, column {self.column}")
            
            # Skip comma and whitespace
            while self.pos < len(self.text) and self.text[self.pos].isspace():
                if self.text[self.pos] == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.pos += 1
                
            if self.pos < len(self.text) and self.text[self.pos] == ',':
                self.pos += 1
                self.column += 1
                
        if self.pos < len(self.text) and self.text[self.pos] == '}':
            self.pos += 1
            self.column += 1
            
        return Token('ARRAY_VALUES', values, self.line, start_col)

    def _function_cell_address(self):
        result = ''
        start_col = self.column
        while self.pos < len(self.text) and self.text[self.pos] != ']' and self.text[self.pos] != ':':
            result += self.text[self.pos]
            self.pos += 1
            self.column += 1
        return Token('CELL_ADDRESS', result.upper(), self.line, start_col)

# Parser class converts tokens into an Abstract Syntax Tree (AST)
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    # Main parsing method that processes all tokens
    def parse(self):
        statements = []
        while self.pos < len(self.tokens):
            stmt = self._statement()
            if stmt:
                statements.append(stmt)
        return statements

    # Parses a single statement (assignment or function call)
    def _statement(self):
        if self.pos >= len(self.tokens):
            return None
        token = self.tokens[self.pos]
        
        # Handle array assignments (e.g., [@A1] : list = {5, 6, 7} or [A1:C1] := {5, 6, 7})
        if token.type == 'CELL_ADDRESS':
            self.pos += 1
            # Check if this is a range assignment
            if self._match('RANGE_SEP'):
                self.pos += 1  # Skip ':'
                if not self._match('CELL_ADDRESS'):
                    raise SyntaxError(self._error("Expected end cell address in range"))
                end_cell = self.tokens[self.pos].value
                self.pos += 1
                
                if self._match('OPERATOR', ':='):
                    self.pos += 1
                    if not self._match('ARRAY_VALUES'):
                        raise SyntaxError(self._error("Expected array values"))
                    values = self.tokens[self.pos].value
                    self.pos += 1
                    return Assignment(CellRange(token.value, end_cell), ArrayValues(values))
            
            # Check if this is a dynamic cell reference
            if self.pos < len(self.tokens) and self.tokens[self.pos].type == 'DYNAMIC_REF':
                self.pos += 1  # Skip '&'
                if not self._match('DYNAMIC_VALUE'):
                    raise SyntaxError(self._error("Expected dynamic value"))
                dynamic_part = self.tokens[self.pos].value
                self.pos += 1
                target = DynamicCellReference(token.value, Variable(dynamic_part))
            else:
                target = CellReference(token.value)
            
            if self._match('OPERATOR', ':='):
                self.pos += 1
                value = self._expression()
                return Assignment(target, value)
        # Handle array assignments with @ symbol (e.g., [@A1] : list = {5, 6, 7})
        elif token.type == 'ARRAY_START':
            self.pos += 1  # Skip '@'
            if not self._match('CELL_ADDRESS'):
                raise SyntaxError(self._error("Expected cell address after @"))
            start_cell = self.tokens[self.pos].value
            self.pos += 1
            
            # Check for named array assignment
            array_name = None
            if self._match('ARRAY_NAME_SEP'):
                self.pos += 1  # Skip ':'
                if not self._match('IDENTIFIER'):
                    raise SyntaxError(self._error("Expected array name after ':'"))
                array_name = self.tokens[self.pos].value
                self.pos += 1
            
            # Check for either := or = operator
            if not (self._match('OPERATOR', ':=') or self._match('OPERATOR', '=')):
                raise SyntaxError(self._error("Expected ':=' or '=' after cell address"))
            self.pos += 1
            
            if not self._match('ARRAY_VALUES'):
                raise SyntaxError(self._error("Expected array values"))
            values = self.tokens[self.pos].value
            self.pos += 1
            return Assignment(CellReference(start_cell), ArrayValues(values), array_name)
        # Handle non-spatial variable assignments (e.g., :X = 10)
        elif token.type == 'VAR_DECL':
            self.pos += 1
            if self._match('IDENTIFIER'):
                var_name = self.tokens[self.pos].value
                self.pos += 1
                if self._match('OPERATOR', '='):
                    self.pos += 1
                    value = self._expression()
                    return Assignment(Variable(var_name), value)
        
        raise SyntaxError(self._error("Invalid syntax"))

    # Parses expressions (including function calls and binary operations)
    def _expression(self):
        if self.pos < len(self.tokens) and self.tokens[self.pos].type == 'FUNCTION':
            return self._function_call()
        return self._binary_expression()

    # Parses function calls (e.g., SUM[A1:A3])
    def _function_call(self):
        func_token = self.tokens[self.pos]
        self.pos += 1
        
        # Expect a cell range [A1:A2]
        if not self._match('CELL_ADDRESS'):
            raise SyntaxError(self._error("Expected cell address after function"))
        
        start_cell = self.tokens[self.pos].value
        self.pos += 1
        
        if not self._match('RANGE_SEP'):
            raise SyntaxError(self._error("Expected ':' in range"))
        self.pos += 1
        
        if not self._match('CELL_ADDRESS'):
            raise SyntaxError(self._error("Expected end cell address in range"))
        
        end_cell = self.tokens[self.pos].value
        self.pos += 1
        
        return Function(func_token.value, CellRange(start_cell, end_cell))

    # Parses binary expressions (e.g., A1 + B1)
    def _expression(self):
        left = self._term()
        while (self.pos < len(self.tokens) and 
               self.tokens[self.pos].type == 'OPERATOR' and 
               self.tokens[self.pos].value in ['+', '-', '*', '/']):
            op = self.tokens[self.pos].value
            self.pos += 1
            right = self._term()
            left = BinaryOp(left, op, right)
        return left

    # Parses terms (numbers, variables, cell references, function calls)
    def _term(self):
        token = self.tokens[self.pos]
        if token.type == 'NUMBER':
            self.pos += 1
            return Number(token.value)
        elif token.type == 'IDENTIFIER':
            self.pos += 1
            return Variable(token.value)
        elif token.type == 'ARRAY_OR_VAR':
            self.pos += 1
            return Variable(token.value)
        elif token.type == 'FUNCTION':
            return self._function_call()
        elif token.type == 'CELL_ADDRESS':
            self.pos += 1
            # Check if this is a dynamic cell reference
            if self.pos < len(self.tokens) and self.tokens[self.pos].type == 'DYNAMIC_REF':
                self.pos += 1  # Skip '&'
                if not self._match('DYNAMIC_VALUE'):
                    raise SyntaxError(self._error("Expected dynamic value"))
                dynamic_part = self.tokens[self.pos].value
                self.pos += 1
                return DynamicCellReference(token.value, Variable(dynamic_part))
            return CellReference(token.value)
        elif token.type == 'VAR_DECL':
            self.pos += 1  # Skip the ':'
            if self._match('IDENTIFIER'):
                var_name = self.tokens[self.pos].value
                self.pos += 1
                return Variable(var_name)
        raise SyntaxError(self._error("Expected term"))

    # Helper methods for parsing
    def _match(self, type, value=None):
        if self.pos < len(self.tokens) and self.tokens[self.pos].type == type:
            if value is None or self.tokens[self.pos].value == value:
                return True
        return False

    def _error(self, message):
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            return f"{message} at line {token.line}, column {token.column}"
        return f"{message} at end of input"

# Interpreter class executes the AST and maintains program state
class Interpreter:
    def __init__(self):
        self.grid = {}      # Dictionary to store cell values
        self.variables = {} # Dictionary to store variable values
        self.arrays = {}    # Dictionary to store named arrays

    def _display_grid(self):
        if not self.grid:
            print("\nGrid is empty")
            return

        # Find the grid dimensions
        max_row = 0
        max_col = 0
        for cell in self.grid.keys():
            match = re.match(r"([A-Z]+)(\d+)", cell)
            if match:
                col, row = match.groups()
                max_row = max(max_row, int(row))
                # Convert column letters to numbers
                col_num = sum((ord(c) - ord('A') + 1) * (26 ** i) 
                            for i, c in enumerate(reversed(col)))
                max_col = max(max_col, col_num)

        # Print column headers (A, B, C, etc.)
        print("\n    " + "   ".join(chr(ord('A') + i) for i in range(max_col)))
        
        # Print grid rows
        for row in range(1, max_row + 1):
            # Print row number
            print(f"{row:2d} |", end="")
            # Print cell values
            for col in range(1, max_col + 1):
                col_letter = chr(ord('A') + col - 1)
                cell = f"{col_letter}{row}"
                value = self.grid.get(cell, "")
                print(f"{value:4}", end="")
            print()  # New line after each row

    # Main interpretation method that executes all statements
    def interpret(self, statements):
        for stmt in statements:
            if isinstance(stmt, Assignment):
                if isinstance(stmt.target, CellRange):
                    # Handle range-based array assignment
                    value = self._eval_expression(stmt.value)
                    if isinstance(value, list):
                        start_row, start_col = self._parse_cell(stmt.target.start)
                        end_row, end_col = self._parse_cell(stmt.target.end)
                        
                        # Ensure proper range order
                        start_row, end_row = min(start_row, end_row), max(start_row, end_row)
                        start_col, end_col = min(start_col, end_col), max(start_col, end_col)
                        
                        # Calculate range dimensions
                        range_width = end_col - start_col + 1
                        range_height = end_row - start_row + 1
                        
                        # Validate array dimensions
                        if len(value) != range_width * range_height:
                            raise ValueError(f"Array dimensions ({len(value)}) do not match cell range dimensions ({range_width * range_height})")
                        
                        # Assign values to cells in the range
                        for i, val in enumerate(value):
                            # Calculate row and column for this value
                            col_offset = i % range_width
                            row_offset = i // range_width
                            col_letter = chr(ord('A') + start_col - 1 + col_offset)
                            row_num = start_row + row_offset
                            cell_address = f"{col_letter}{row_num}"
                            self.grid[cell_address] = val
                    else:
                        raise ValueError("Range assignment requires array values")
                elif isinstance(stmt.target, CellReference):
                    value = self._eval_expression(stmt.value)
                    if isinstance(value, list):  # Array assignment
                        # Get the starting cell's row and column
                        start_row, start_col = self._parse_cell(stmt.target.address)
                        # Assign each value to consecutive cells in the row
                        for i, val in enumerate(value):
                            col_letter = chr(ord('A') + start_col - 1 + i)
                            cell_address = f"{col_letter}{start_row}"
                            self.grid[cell_address] = val
                        # If this is a named array, store the reference
                        if stmt.array_name:
                            self.arrays[stmt.array_name] = {
                                'start_cell': stmt.target.address,
                                'values': value
                            }
                    else:
                        self.grid[stmt.target.address] = value
                elif isinstance(stmt.target, DynamicCellReference):
                    # Evaluate the dynamic part (e.g., N in A&{N})
                    dynamic_value = self._eval_expression(stmt.target.dynamic_part)
                    # Construct the full cell address
                    cell_address = f"{stmt.target.prefix}{dynamic_value}"
                    # Evaluate and assign the value
                    value = self._eval_expression(stmt.value)
                    self.grid[cell_address] = value
                elif isinstance(stmt.target, Variable):
                    self.variables[stmt.target.name] = self._eval_expression(stmt.value)
        
        # Print final state
        print("\nGrid State:")
        self._display_grid()
        
        print("\nVariables:")
        for var, value in sorted(self.variables.items()):
            print(f"{var}: {value}")
            
        print("\nNamed Arrays:")
        for name, array in sorted(self.arrays.items()):
            print(f"{name}: starts at {array['start_cell']}, values = {array['values']}")

    # Evaluates expressions and returns their values
    def _eval_expression(self, node):
        if isinstance(node, Function):
            if node.name == 'SUM':
                return self._eval_sum(node.range)
            raise ValueError(f"Unknown function: {node.name}")
        elif isinstance(node, Number):
            return node.value
        elif isinstance(node, Variable):
            if node.name not in self.variables:
                raise ValueError(f"Undefined variable: {node.name}")
            return self.variables[node.name]
        elif isinstance(node, CellReference):
            if node.address not in self.grid:
                return 0  # Default value for empty cells
            return self.grid[node.address]
        elif isinstance(node, DynamicCellReference):
            # Evaluate the dynamic part (e.g., N in A&{N})
            dynamic_value = self._eval_expression(node.dynamic_part)
            # Construct the full cell address
            cell_address = f"{node.prefix}{dynamic_value}"
            if cell_address not in self.grid:
                return 0  # Default value for empty cells
            return self.grid[cell_address]
        elif isinstance(node, ArrayValues):
            return node.values  # Return the array values for assignment
        elif isinstance(node, BinaryOp):
            left = self._eval_expression(node.left)
            right = self._eval_expression(node.right)
            if node.op == '+':
                return left + right
            elif node.op == '-':
                return left - right
            elif node.op == '*':
                return left * right
            elif node.op == '/':
                if right == 0:
                    raise ValueError("Division by zero")
                return left / right
            else:
                raise ValueError(f"Unknown operator: {node.op}")
        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    # Evaluates the SUM function for a range of cells
    def _eval_sum(self, range_node):
        start_row, start_col = self._parse_cell(range_node.start)
        end_row, end_col = self._parse_cell(range_node.end)
        
        # Ensure proper range order
        start_row, end_row = min(start_row, end_row), max(start_row, end_row)
        start_col, end_col = min(start_col, end_col), max(start_col, end_col)
        
        total = 0
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                # Convert row/col back to cell address
                col_letter = chr(ord('A') + col - 1)
                cell_address = f"{col_letter}{row}"
                value = self.grid.get(cell_address, 0)
                total += value
        
        return total

    # Converts cell address (e.g., "A1") to row and column numbers
    def _parse_cell(self, address):
        match = re.match(r"([A-Z]+)(\d+)", address)
        if not match:
            raise ValueError(f"Invalid cell address: {address}")
        col, row = match.groups()
        # Convert column letters to numbers (A=1, B=2, etc.)
        col_num = sum((ord(c) - ord('A') + 1) * (26 ** i) 
                     for i, c in enumerate(reversed(col)))
        return int(row), col_num

# Main function to run the Grid language interpreter
def run_grid(code):
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
        
    except Exception as e:
        print(f"Error: {str(e)}")

# Test the implementation with a sample program
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