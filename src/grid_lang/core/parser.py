from ..nodes.ast_nodes import (
    Assignment, CellReference, DynamicCellReference, Variable,
    Number, BinaryOp, Function, CellRange, ArrayValues
)

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def parse(self):
        statements = []
        while self.pos < len(self.tokens):
            stmt = self._statement()
            if stmt:
                statements.append(stmt)
        return statements

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

    def _expression(self):
        if self.pos < len(self.tokens) and self.tokens[self.pos].type == 'FUNCTION':
            return self._function_call()
        return self._binary_expression()

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

    def _binary_expression(self):
        left = self._term()
        while (self.pos < len(self.tokens) and 
               self.tokens[self.pos].type == 'OPERATOR' and 
               self.tokens[self.pos].value in ['+', '-', '*', '/']):
            op = self.tokens[self.pos].value
            self.pos += 1
            right = self._term()
            left = BinaryOp(left, op, right)
        return left

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