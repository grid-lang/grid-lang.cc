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
                if word.upper() in self.keywords:
                    self.tokens.append(Token('FUNCTION', word.upper(), self.line, self.column - len(word)))
                    # Skip whitespace
                    while self.pos < len(self.text) and self.text[self.pos].isspace():
                        if self.text[self.pos] == '\n':
                            self.line += 1
                            self.column = 1
                        else:
                            self.column += 1
                        self.pos += 1
                    # Handle function calls with cell ranges
                    if self.pos < len(self.text) and self.text[self.pos] == '[':
                        self.pos += 1  # Skip '['
                        self.column += 1
                        # Check if this is a range function call
                        cell_token = self._cell_address()
                        self.tokens.append(cell_token)
                        if self.pos < len(self.text) and self.text[self.pos] == ':':
                            self.tokens.append(Token('RANGE_SEP', ':', self.line, self.column))
                            self.pos += 1
                            self.column += 1
                            self.tokens.append(self._cell_address())
                        if self.pos < len(self.text) and self.text[self.pos] == ']':
                            self.pos += 1  # Skip ']'
                            self.column += 1
                    # Handle function calls with arguments in curly braces
                    elif self.pos < len(self.text) and self.text[self.pos] == '{':
                        values = self._function_args()
                        self.tokens.append(Token('FUNCTION_ARGS', values, self.line, self.column - 1))
                    # Handle function calls with parentheses
                    elif self.pos < len(self.text) and self.text[self.pos] == '(':
                        self.pos += 1  # Skip '('
                        self.column += 1
                        # Skip whitespace
                        while self.pos < len(self.text) and self.text[self.pos].isspace():
                            if self.text[self.pos] == '\n':
                                self.line += 1
                                self.column = 1
                            else:
                                self.column += 1
                            self.pos += 1
                        # Handle cell range
                        if self.pos < len(self.text) and self.text[self.pos] == '[':
                            self.pos += 1  # Skip '['
                            self.column += 1
                            cell_token = self._cell_address()
                            self.tokens.append(cell_token)
                            if self.pos < len(self.text) and self.text[self.pos] == ':':
                                self.tokens.append(Token('RANGE_SEP', ':', self.line, self.column))
                                self.pos += 1
                                self.column += 1
                                self.tokens.append(self._cell_address())
                            if self.pos < len(self.text) and self.text[self.pos] == ']':
                                self.pos += 1  # Skip ']'
                                self.column += 1
                        # Skip closing parenthesis
                        if self.pos < len(self.text) and self.text[self.pos] == ')':
                            self.pos += 1  # Skip ')'
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
                # Check for variable declaration or array name
                if self.pos < len(self.text) and self.text[self.pos] == ':':
                    self.pos += 1  # Skip ':'
                    self.column += 1
                    # Skip whitespace
                    while self.pos < len(self.text) and self.text[self.pos].isspace():
                        if self.text[self.pos] == '\n':
                            self.line += 1
                            self.column = 1
                        else:
                            self.column += 1
                        self.pos += 1
                    # Read the name
                    name = self._read_word()
                    # Check if this is an array assignment
                    if self.tokens[-2].type == 'ARRAY_START':
                        self.tokens.append(Token('ARRAY_NAME', name.upper(), self.line, self.column - len(name)))
                    else:
                        self.tokens.append(Token('VAR_DECL', ':', self.line, self.column - len(name) - 1))
                        self.tokens.append(Token('IDENTIFIER', name.upper(), self.line, self.column - len(name)))
            # Handle array or variable references (e.g., {X})
            elif char == '{':
                print(f"Found curly brace at position {self.pos}")
                print(f"Previous token: {self.tokens[-1] if self.tokens else 'None'}")
                # Check if this is a function argument list
                if self.pos > 0 and self.tokens and self.tokens[-1].type == 'FUNCTION':
                    print("Handling function arguments")
                    values = self._function_args()
                    self.tokens.append(Token('FUNCTION_ARGS', values, self.line, self.column - 1))
                else:
                    print("Handling array values")
                    self.tokens.append(self._handle_array_values())
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
                        self.tokens.append(Token('VAR_DECL', ':', self.line, self.column))
                        self.pos += 1
                        self.column += 1
                    else:
                        self.tokens.append(Token('RANGE_SEP', ':', self.line, self.column))
                        self.pos += 1
                        self.column += 1
            # Handle arithmetic operators
            elif char in '=+-*/^':
                self.tokens.append(Token('OPERATOR', char, self.line, self.column))
                self.column += 1
                self.pos += 1
            # Handle parentheses
            elif char in '()':
                self.tokens.append(Token('PAREN', char, self.line, self.column))
                self.column += 1
                self.pos += 1
            else:
                raise SyntaxError(f"Unexpected character: {char} at line {self.line}, column {self.column}")
        return self.tokens

    # Helper methods for tokenization
    def _skip_comment(self):
        while self.pos < len(self.text) and self.text[self.pos] != '\n':
            self.pos += 1
            self.column += 1
        if self.pos < len(self.text) and self.text[self.pos] == '\n':
            self.line += 1
            self.column = 1
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
        # Handle negative numbers
        if self.pos < len(self.text) and self.text[self.pos] == '-':
            result += self.text[self.pos]
            self.pos += 1
            self.column += 1
        
        # Read integer part
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            result += self.text[self.pos]
            self.pos += 1
            self.column += 1
            
        # Handle decimal point
        if self.pos < len(self.text) and self.text[self.pos] == '.':
            result += self.text[self.pos]
            self.pos += 1
            self.column += 1
            # Read decimal part
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                result += self.text[self.pos]
                self.pos += 1
                self.column += 1
                
        # Handle scientific notation (e.g., 1.2e3, 1.2E-3)
        if self.pos < len(self.text) and self.text[self.pos].lower() == 'e':
            result += self.text[self.pos]
            self.pos += 1
            self.column += 1
            # Handle exponent sign
            if self.pos < len(self.text) and self.text[self.pos] in '+-':
                result += self.text[self.pos]
                self.pos += 1
                self.column += 1
            # Read exponent digits
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                result += self.text[self.pos]
                self.pos += 1
                self.column += 1
                
        return Token('NUMBER', float(result), self.line, start_col)

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

    def _handle_array_values(self):
        """Handle array values between curly braces."""
        values = []
        self.pos += 1  # Skip '{'
        self.column += 1
        
        while self.pos < len(self.text) and self.text[self.pos] != '}':
            if self.text[self.pos].isspace():
                self.pos += 1
                self.column += 1
                continue
                
            # Skip commas
            if self.text[self.pos] == ',':
                values.append(Token('COMMA', ',', self.line, self.column))
                self.pos += 1
                self.column += 1
                continue
                
            # Start of a new value
            start_pos = self.pos
            start_col = self.column
            expr_tokens = []
            
            # Read until we hit a comma or closing brace
            while self.pos < len(self.text) and self.text[self.pos] != ',' and self.text[self.pos] != '}':
                char = self.text[self.pos]
                
                # Skip whitespace
                if char.isspace():
                    self.pos += 1
                    self.column += 1
                    continue
                
                # Handle expressions in parentheses
                if char == '(':
                    expr_tokens.append(Token('PAREN', '(', self.line, self.column))
                    self.pos += 1
                    self.column += 1
                    continue
                    
                if char == ')':
                    expr_tokens.append(Token('PAREN', ')', self.line, self.column))
                    self.pos += 1
                    self.column += 1
                    continue
                    
                # Handle operators
                if char in '+-*/^':
                    expr_tokens.append(Token('OPERATOR', char, self.line, self.column))
                    self.pos += 1
                    self.column += 1
                    continue
                    
                # Handle numbers
                if char.isdigit() or char == '-':
                    num_str = ''
                    if char == '-':
                        num_str += char
                        self.pos += 1
                        self.column += 1
                    
                    while (self.pos < len(self.text) and 
                           (self.text[self.pos].isdigit() or 
                            self.text[self.pos] == '.' or 
                            self.text[self.pos] == 'e' or 
                            self.text[self.pos] == 'E')):
                        num_str += self.text[self.pos]
                        self.pos += 1
                        self.column += 1
                    
                    # Handle exponent sign
                    if (self.pos < len(self.text) and 
                        self.text[self.pos - 1].lower() == 'e' and 
                        self.text[self.pos] in '+-'):
                        num_str += self.text[self.pos]
                        self.pos += 1
                        self.column += 1
                        # Read exponent digits
                        while (self.pos < len(self.text) and 
                               self.text[self.pos].isdigit()):
                            num_str += self.text[self.pos]
                            self.pos += 1
                            self.column += 1
                    
                    try:
                        expr_tokens.append(Token('NUMBER', float(num_str), self.line, self.column - len(num_str)))
                    except ValueError:
                        raise ValueError(f"Invalid number format: {num_str}")
                    continue
                    
                # Handle identifiers
                if char.isalpha():
                    id_str = ''
                    while (self.pos < len(self.text) and 
                           (self.text[self.pos].isalnum() or 
                            self.text[self.pos] == '_')):
                        id_str += self.text[self.pos]
                        self.pos += 1
                        self.column += 1
                    expr_tokens.append(Token('IDENTIFIER', id_str.upper(), self.line, self.column - len(id_str)))
                    continue
                    
                # Handle cell references
                if char == '[':
                    self.pos += 1
                    self.column += 1
                    cell_ref = ''
                    while (self.pos < len(self.text) and 
                           self.text[self.pos] != ']' and 
                           not self.text[self.pos].isspace()):
                        cell_ref += self.text[self.pos]
                        self.pos += 1
                        self.column += 1
                    if self.text[self.pos] == ']':
                        self.pos += 1
                        self.column += 1
                    expr_tokens.append(Token('CELL_ADDRESS', cell_ref.upper(), self.line, self.column - len(cell_ref) - 2))
                    continue
                
                # Skip any other characters
                self.pos += 1
                self.column += 1
            
            # Add the expression tokens to the values list
            if expr_tokens:
                values.append(Token('EXPRESSION', expr_tokens, self.line, start_col))
        
        if self.text[self.pos] == '}':
            self.pos += 1
            self.column += 1
            
        return Token('ARRAY_VALUES', values, self.line, self.column - 1)

    def _function_cell_address(self):
        result = ''
        start_col = self.column
        while self.pos < len(self.text) and self.text[self.pos] != ']' and self.text[self.pos] != ':':
            result += self.text[self.pos]
            self.pos += 1
            self.column += 1
        return Token('CELL_ADDRESS', result.upper(), self.line, start_col)

    def _function_args(self):
        """Reads function arguments enclosed in curly braces."""
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
                
            # Read a value
            if self.text[self.pos].isdigit() or self.text[self.pos] == '-':
                values.append(self._number())
            elif self.text[self.pos].isalpha():
                values.append(Token('IDENTIFIER', self._read_word().upper(), self.line, self.column))
            elif self.text[self.pos] == '[':
                # Handle cell reference
                self.pos += 1  # Skip '['
                self.column += 1
                cell_token = self._cell_address()
                values.append(cell_token)
                # Skip closing bracket
                if self.pos < len(self.text) and self.text[self.pos] == ']':
                    self.pos += 1  # Skip ']'
                    self.column += 1
            else:
                raise SyntaxError(f"Unexpected character in function arguments: {self.text[self.pos]} at line {self.line}, column {self.column}")
            
            # Skip whitespace
            while self.pos < len(self.text) and self.text[self.pos].isspace():
                if self.text[self.pos] == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.pos += 1
                
            # Skip comma
            if self.pos < len(self.text) and self.text[self.pos] == ',':
                self.pos += 1
                self.column += 1
                
        if self.pos < len(self.text) and self.text[self.pos] == '}':
            self.pos += 1  # Skip '}'
            self.column += 1
            
        return values