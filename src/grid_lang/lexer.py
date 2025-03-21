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