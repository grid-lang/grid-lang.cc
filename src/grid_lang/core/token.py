class Token:
    def __init__(self, type, value, line=1, column=1):
        self.type = type      # Token type (e.g., NUMBER, OPERATOR, CELL_ADDRESS)
        self.value = value    # Actual value of the token
        self.line = line      # Line number in source code
        self.column = column  # Column number in source code

    def __str__(self):
        return f"Token({self.type}, {self.value}, line={self.line}, col={self.column})" 