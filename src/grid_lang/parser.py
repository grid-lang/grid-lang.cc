from .lexer import Token
from .ast import (
    Assignment, CellReference, DynamicCellReference, Variable,
    Number, BinaryOp, Function, CellRange, ArrayValues
)

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
        
        # Handle cell address with variable assignment (e.g., [A1] : a = 51)
        if token.type == 'CELL_ADDRESS':
            self.pos += 1
            cell_ref = CellReference(token.value)
            assignments = []
            
            # Check for variable declaration
            if self._match('VAR_DECL'):
                self.pos += 1  # Skip ':'
                if not self._match('IDENTIFIER'):
                    raise SyntaxError(self._error("Expected variable name after ':'"))
                var_name = self.tokens[self.pos].value
                self.pos += 1
                
                if self._match('OPERATOR', '='):
                    self.pos += 1
                    value = self._expression()
                    assignments.extend([
                        Assignment(cell_ref, value),
                        Assignment(Variable(var_name), value)
                    ])
                    
                    # Check for more variable declarations
                    while self._match('VAR_DECL'):
                        self.pos += 1  # Skip ':'
                        if not self._match('IDENTIFIER'):
                            raise SyntaxError(self._error("Expected variable name after ':'"))
                        var_name = self.tokens[self.pos].value
                        self.pos += 1
                        
                        if self._match('OPERATOR', '='):
                            self.pos += 1
                            value = self._expression()
                            assignments.append(Assignment(Variable(var_name), value))
                        else:
                            break
                    
                    return assignments
            
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
                
                # Check for variable declaration after dynamic cell reference
                if self._match('VAR_DECL'):
                    self.pos += 1  # Skip ':'
                    if not self._match('IDENTIFIER'):
                        raise SyntaxError(self._error("Expected variable name after ':'"))
                    var_name = self.tokens[self.pos].value
                    self.pos += 1
                    
                    if self._match('OPERATOR', '='):
                        self.pos += 1
                        value = self._expression()
                        return [
                            Assignment(target, value),
                            Assignment(Variable(var_name), value)
                        ]
            else:
                target = cell_ref
            
            if self._match('OPERATOR', ':='):
                self.pos += 1
                value = self._expression()
                return Assignment(target, value)
        # Handle array assignments with @ symbol (e.g., [@A1] : list = {5, 6, 7} or [@B2] := b)
        elif token.type == 'ARRAY_START':
            self.pos += 1  # Skip '@'
            if not self._match('CELL_ADDRESS'):
                raise SyntaxError(self._error("Expected cell address after @"))
            start_cell = self.tokens[self.pos].value
            self.pos += 1
            
            # Check for named array assignment
            array_name = True  # Default to True for @ array assignments
            if self._match('VAR_DECL'):
                self.pos += 1  # Skip ':'
                if not self._match('IDENTIFIER'):
                    raise SyntaxError(self._error("Expected array name after ':'"))
                array_name = self.tokens[self.pos].value
                self.pos += 1
            
            # Check for either := or = operator
            if not (self._match('OPERATOR', ':=') or self._match('OPERATOR', '=')):
                raise SyntaxError(self._error("Expected ':=' or '=' after cell address"))
            self.pos += 1
            
            # Check if this is a variable reference (e.g., [@B2] := b)
            if self._match('IDENTIFIER'):
                var_name = self.tokens[self.pos].value
                self.pos += 1
                # Create an assignment with the Variable as the value and array_name set
                return Assignment(CellReference(start_cell), Variable(var_name), array_name)
            
            # Otherwise, expect array values
            if not self._match('ARRAY_VALUES'):
                raise SyntaxError(self._error("Expected array values or variable name"))
            values = self.tokens[self.pos].value
            self.pos += 1
            
            # Evaluate expressions in array values
            evaluated_values = []
            
            for value in values:
                if value.type == 'EXPRESSION':
                    # Create a temporary parser for the expression tokens
                    temp_parser = Parser(value.value)
                    temp_parser.pos = 0
                    expr = temp_parser._expression()
                    evaluated_values.append(expr)
                elif value.type == 'COMMA':
                    continue
                else:
                    # Handle single values (numbers, variables, cell references)
                    if value.type == 'NUMBER':
                        evaluated_values.append(Number(value.value))
                    elif value.type == 'IDENTIFIER':
                        evaluated_values.append(Variable(value.value))
                    elif value.type == 'CELL_ADDRESS':
                        evaluated_values.append(CellReference(value.value))
            
            # Create an assignment with array_name set and ArrayValues as the value
            return Assignment(CellReference(start_cell), ArrayValues(evaluated_values), array_name)
        # Handle non-spatial variable assignments (e.g., :X = 10)
        elif token.type == 'VAR_DECL':
            self.pos += 1
            if self._match('IDENTIFIER'):
                var_name = self.tokens[self.pos].value
                self.pos += 1
                if self._match('OPERATOR', '='):
                    self.pos += 1
                    # Check if this is an array assignment
                    if self._match('ARRAY_VALUES'):
                        array_token = self.tokens[self.pos]
                        self.pos += 1
                        
                        # Process the array values into a list of objects
                        processed_values = []
                        
                        # First check if we only have simple expressions (just numbers)
                        has_simple_values = True
                        for val in array_token.value:
                            if val.type == 'EXPRESSION':
                                exp_tokens = val.value
                                if len(exp_tokens) == 1 and exp_tokens[0].type == 'NUMBER':
                                    continue
                                else:
                                    has_simple_values = False
                                    break
                            elif val.type != 'COMMA':
                                has_simple_values = False
                                break
                        
                        # For simple values, convert to Number objects
                        if has_simple_values:
                            for val in array_token.value:
                                if val.type == 'EXPRESSION':
                                    exp_tokens = val.value
                                    if len(exp_tokens) == 1 and exp_tokens[0].type == 'NUMBER':
                                        # Simple number
                                        processed_values.append(Number(exp_tokens[0].value))
                                elif val.type != 'COMMA':
                                    # Add other token types directly
                                    processed_values.append(val)
                            
                            return Assignment(Variable(var_name), ArrayValues(processed_values))
                        
                        # For more complex expressions
                        for val in array_token.value:
                            if val.type == 'EXPRESSION':
                                # Create a temporary parser for the expression tokens
                                temp_parser = Parser(val.value)
                                temp_parser.pos = 0
                                expr = temp_parser._expression()
                                processed_values.append(expr)
                            elif val.type == 'COMMA':
                                # Skip commas
                                continue
                            else:
                                # Handle other token types
                                processed_values.append(val)
                        
                        return Assignment(Variable(var_name), ArrayValues(processed_values))
                    else:
                        value = self._expression()
                        return Assignment(Variable(var_name), value)
        
        raise SyntaxError(self._error("Invalid syntax"))

    # Parses expressions (including function calls and binary operations)
    def _expression(self):
        if self.pos < len(self.tokens) and self.tokens[self.pos].type == 'IDENTIFIER' and self.tokens[self.pos].value == 'SUM':
            return self._function_call()
        return self._add_sub()

    def _add_sub(self):
        left = self._mul_div()
        while (self.pos < len(self.tokens) and 
               self.tokens[self.pos].type == 'OPERATOR' and 
               self.tokens[self.pos].value in ['+', '-']):
            op = self.tokens[self.pos].value
            self.pos += 1
            right = self._mul_div()
            left = BinaryOp(left, op, right)
        return left

    def _mul_div(self):
        left = self._power()
        while (self.pos < len(self.tokens) and 
               self.tokens[self.pos].type == 'OPERATOR' and 
               self.tokens[self.pos].value in ['*', '/']):
            op = self.tokens[self.pos].value
            self.pos += 1
            right = self._power()
            left = BinaryOp(left, op, right)
        return left

    def _power(self):
        left = self._term()
        while (self.pos < len(self.tokens) and 
               self.tokens[self.pos].type == 'OPERATOR' and 
               self.tokens[self.pos].value == '^'):
            op = self.tokens[self.pos].value
            self.pos += 1
            right = self._term()  # Right-associative
            left = BinaryOp(left, op, right)
        return left

    # Parses function calls (e.g., SUM[A1:A3] or SUM([A1:A2]) or SUM{a, b})
    def _function_call(self):
        func_token = self.tokens[self.pos]
        self.pos += 1
        
        # Check if this is a cell range function call
        if self._match('CELL_ADDRESS'):
            # Skip opening parenthesis if present
            if self._match('PAREN', '('):
                self.pos += 1
            
            # Expect a cell range [A1:A2]
            start_cell = self.tokens[self.pos].value
            self.pos += 1
            
            if not self._match('RANGE_SEP'):
                raise SyntaxError(self._error("Expected ':' in range"))
            self.pos += 1
            
            if not self._match('CELL_ADDRESS'):
                raise SyntaxError(self._error("Expected end cell address in range"))
            
            end_cell = self.tokens[self.pos].value
            self.pos += 1
            
            # Skip closing parenthesis if present
            if self._match('PAREN', ')'):
                self.pos += 1
            
            return Function(func_token.value, CellRange(start_cell, end_cell))
        # Check if this is a function call with arguments
        elif self._match('FUNCTION_ARGS'):
            args = self.tokens[self.pos].value
            self.pos += 1
            # Evaluate each argument
            evaluated_args = []
            for arg in args:
                if arg.type == 'NUMBER':
                    evaluated_args.append(Number(arg.value))
                elif arg.type == 'IDENTIFIER':
                    evaluated_args.append(Variable(arg.value))
                elif arg.type == 'CELL_ADDRESS':
                    evaluated_args.append(CellReference(arg.value))
                else:
                    raise SyntaxError(self._error(f"Unexpected argument type: {arg.type}"))
            return Function(func_token.value, evaluated_args)
        else:
            raise SyntaxError(self._error("Expected cell range or function arguments"))

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
        elif token.type == 'PAREN' and token.value == '(':
            self.pos += 1  # Skip '('
            expr = self._expression()
            if not self._match('PAREN', ')'):
                raise SyntaxError(self._error("Expected closing parenthesis"))
            self.pos += 1  # Skip ')'
            return expr
        elif token.type == 'ARRAY_VALUES':
            self.pos += 1
            return ArrayValues(token.value)
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