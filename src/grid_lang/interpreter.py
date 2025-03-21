import re
import io
import csv
from .lexer import Token
from .ast import (
    Assignment, CellReference, DynamicCellReference, Variable,
    Number, BinaryOp, Function, CellRange, ArrayValues
)

class Interpreter:
    def __init__(self):
        self.grid = {}      # Dictionary to store cell values
        self.variables = {} # Dictionary to store variable values
        self.arrays = {}    # Dictionary to store named arrays

    def _get_col_number(self, col_letters):
        """Converts column letters (e.g., 'A', 'AB') to column number (1-based)."""
        return sum((ord(c) - ord('A') + 1) * (26 ** i) 
                  for i, c in enumerate(reversed(col_letters)))

    def _get_col_letters(self, col_num):
        """Converts column number (1-based) to column letters (e.g., 1->'A', 28->'AB')."""
        letters = []
        while col_num > 0:
            col_num -= 1
            letters.append(chr(ord('A') + (col_num % 26)))
            col_num //= 26
        return ''.join(reversed(letters))

    def get_grid_csv(self):
        """Returns the grid state in CSV format.
        Only includes spatial data (grid cells), not variables or named arrays.
        Does not include row or column headers.
        
        Returns:
            str: CSV representation of the grid
        """
        if not self.grid:
            return ""  # Return empty string for empty grid
            
        # Find the grid dimensions
        max_row = 0
        max_col = 0
        for cell in self.grid.keys():
            match = re.match(r"([A-Z]+)(\d+)", cell)
            if match:
                col_letters, row = match.groups()
                max_row = max(max_row, int(row))
                col_num = self._get_col_number(col_letters)
                max_col = max(max_col, col_num)
        
        # Create CSV output
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write data rows without headers
        for row in range(1, max_row + 1):
            row_data = []
            for col in range(1, max_col + 1):
                col_letter = self._get_col_letters(col)
                cell = f"{col_letter}{row}"
                value = self.grid.get(cell, "")  # Empty string for undefined cells
                row_data.append(str(value))
            writer.writerow(row_data)
        
        return output.getvalue()

    def interpret(self, statements):
        """Interprets and executes the AST statements.
        
        Args:
            statements (list): List of AST nodes representing the program
        """
        for stmt in statements:
            # Handle lists of assignments (e.g., from cell references with multiple variable declarations)
            if isinstance(stmt, list):
                for sub_stmt in stmt:
                    self._execute_assignment(sub_stmt)
            else:
                self._execute_assignment(stmt)
        
        # Print final state
        print("\nGrid State:")
        self._display_grid()
        
        print("\nVariables:")
        for var, value in sorted(self.variables.items()):
            print(f"{var}: {value}")
            
        print("\nNamed Arrays:")
        for name, array in sorted(self.arrays.items()):
            print(f"{name}: starts at {array['start_cell']}, values = {array['values']}")

    def _execute_assignment(self, stmt):
        """Executes a single assignment statement."""
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
                # Check if this is an array assignment (has @ symbol) or regular cell assignment
                is_array_assignment = hasattr(stmt, 'array_name') and stmt.array_name is not None
                
                if is_array_assignment:
                    # Array assignment - evaluate the value
                    value = self._eval_expression(stmt.value)
                    
                    # Make sure it's a list
                    if not isinstance(value, list):
                        if isinstance(stmt.value, ArrayValues):
                            # Direct array values - evaluate each value in the array
                            values = []
                            for val in stmt.value.values:
                                evaluated_val = self._eval_expression(val)
                                values.append(evaluated_val)
                            value = values
                        else:
                            # Handle non-list values
                            value = [value]
                    
                    # Get the starting cell's row and column
                    start_row, start_col = self._parse_cell(stmt.target.address)
                    start_cell = stmt.target.address
                    
                    # Store array reference for named arrays
                    if hasattr(stmt, 'array_name') and stmt.array_name and stmt.array_name is not True:
                        self.arrays[stmt.array_name] = {
                            'start_cell': start_cell,
                            'values': value
                        }
                    
                    # Distribute array values to individual cells
                    for i, val in enumerate(value):
                        col_letter = chr(ord('A') + start_col - 1 + i)
                        cell_address = f"{col_letter}{start_row}"
                        # Store individual values, not the array itself
                        self.grid[cell_address] = val
                else:
                    # Regular cell assignment
                    value = self._eval_expression(stmt.value)
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
                # For variable assignments, evaluate the value first
                if isinstance(stmt.value, ArrayValues):
                    # Directly evaluate array values using _evaluate_array
                    value = self._evaluate_array(stmt.value.values)
                else:
                    value = self._eval_expression(stmt.value)
                
                # Store the value in the variables dictionary
                # Keep variable names as they are (preserve case)
                var_name = stmt.target.name
                self.variables[var_name] = value
                
                # Also store a lowercase version for case-insensitive lookups
                lowercase_name = var_name.lower()
                if lowercase_name != var_name:
                    self.variables[lowercase_name] = value

    def _display_grid(self):
        """Displays the grid state in a human-readable format."""
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
                
                # Format the value
                formatted_value = ""
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:4}"
                elif isinstance(value, Variable):
                    var_value = self.variables.get(value.name, "")
                    formatted_value = f"{var_value:4}"
                else:
                    formatted_value = f"{str(value):4}"
                
                print(formatted_value, end="")
            print()  # New line after each row

    def _evaluate_array(self, tokens):
        """Evaluates an array of tokens into a list of values."""
        if not tokens:
            return []
            
        # First, check if we have AST Number nodes (already parsed numbers)
        has_ast_number_nodes = False
        for token in tokens:
            if isinstance(token, Number):
                has_ast_number_nodes = True
                break
                
        if has_ast_number_nodes:
            results = []
            for token in tokens:
                if isinstance(token, Number):
                    results.append(token.value)
                elif isinstance(token, Token) and token.type == 'COMMA':
                    continue
                else:
                    evaluated = self._eval_expression(token)
                    results.append(evaluated)
            return results
        
        # First check if we have a list of EXPRESSION and COMMA tokens
        has_expression_tokens = False
        for tok in tokens:
            if isinstance(tok, Token) and tok.type == 'EXPRESSION':
                has_expression_tokens = True
                break
                
        if has_expression_tokens:
            results = []
            for i, tok in enumerate(tokens):
                if isinstance(tok, Token):
                    if tok.type == 'EXPRESSION':
                        # Evaluate the expression token
                        inner_tokens = tok.value
                        if len(inner_tokens) == 1 and inner_tokens[0].type == 'NUMBER':
                            value = float(inner_tokens[0].value)
                            results.append(value)
                        else:
                            expr_value = self._eval_token_expression(inner_tokens)
                            results.append(expr_value)
                    elif tok.type == 'COMMA':
                        continue
                    elif tok.type == 'NUMBER':
                        results.append(float(tok.value))
                    else:
                        results.append(self._eval_token_expression([tok]))
                else:
                    results.append(tok)
            return results
            
        # Check if we have direct array values (e.g., {100, 200, 300})
        # These might be processed into individual number tokens and commas
        if all(isinstance(tok, Token) and (tok.type == 'NUMBER' or tok.type == 'COMMA') for tok in tokens):
            results = []
            for tok in tokens:
                if isinstance(tok, Token):
                    if tok.type == 'NUMBER':
                        results.append(float(tok.value))
                    # Skip commas
            return results
            
        # Handle other cases with mixed token types
        chunks = []
        current_chunk = []
        
        for tok in tokens:
            if isinstance(tok, Token) and tok.type == 'COMMA':
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
            else:
                current_chunk.append(tok)
                
        # Add the last chunk if there is one
        if current_chunk:
            chunks.append(current_chunk)
            
        results = []
        for i, chunk in enumerate(chunks):
            # If we have a single token in the chunk
            if len(chunk) == 1:
                item = chunk[0]
                if isinstance(item, Token):
                    if item.type == 'NUMBER':
                        results.append(float(item.value))
                        continue
                    elif item.type == 'EXPRESSION':
                        expr_value = self._eval_token_expression(item.value)
                        results.append(expr_value)
                        continue
                    else:
                        results.append(self._eval_token_expression([item]))
                        continue
                elif hasattr(item, 'value'):
                    results.append(item.value)
                    continue
                else:
                    results.append(item)
                    continue
            
            # For multiple tokens in a chunk, evaluate them individually
            if all(hasattr(t, 'value') for t in chunk):
                for item in chunk:
                    if hasattr(item, 'value'):
                        results.append(item.value)
                continue
            
            # If we couldn't handle the chunk in special ways, try as an expression
            result = self._eval_token_expression(chunk)
            results.append(result)
            
        return results

    def _build_expression_string(self, tokens):
        """Builds an expression string from tokens for evaluation."""
        expr = ''
        for tok in tokens:
            if isinstance(tok, Token):
                if tok.type == 'NUMBER':
                    expr += str(tok.value)
                elif tok.type == 'IDENTIFIER':
                    # Get variable value case-insensitive
                    var_value = 0
                    for var_name, value in self.variables.items():
                        if var_name.upper() == tok.value.upper():
                            var_value = value
                            break
                    expr += str(var_value)
                elif tok.type == 'CELL_ADDRESS':
                    cell_value = self.grid.get(tok.value, 0)
                    expr += str(cell_value)
                elif tok.type == 'OPERATOR':
                    expr += tok.value
                elif tok.type == 'PAREN':
                    expr += tok.value
                else:
                    expr += str(tok.value)
            elif hasattr(tok, 'value'):
                # Handle AST nodes like Number
                expr += str(tok.value)
            else:
                expr += str(tok)
        return expr
        
    def _eval_token_expression(self, tokens):
        """Evaluates a list of tokens as an expression."""
        if not tokens:
            return 0
            
        # Special case for single token
        if len(tokens) == 1:
            tok = tokens[0]
            if isinstance(tok, Token):
                if tok.type == 'NUMBER':
                    return float(tok.value)
                elif tok.type == 'IDENTIFIER':
                    # Get variable value case-insensitive
                    for var_name, var_value in self.variables.items():
                        if var_name.upper() == tok.value.upper():
                            return var_value
                    return 0
                elif tok.type == 'CELL_ADDRESS':
                    # Get cell value
                    return self.grid.get(tok.value, 0)
                elif tok.type == 'EXPRESSION':
                    # Evaluate nested expression
                    inner_tokens = tok.value
                    if isinstance(inner_tokens, list):
                        expr = self._build_expression_string(inner_tokens)
                        try:
                            result = eval(expr)
                            return result
                        except Exception as e:
                            return 0
                    return inner_tokens
                else:
                    return tok.value
            elif hasattr(tok, 'value'):
                return tok.value
            else:
                return tok
                
        # For multiple tokens, build an expression string
        if all(hasattr(t, 'value') for t in tokens):
            pass
            
        expr = self._build_expression_string(tokens)
        
        try:
            result = eval(expr)
            return result
        except Exception as e:
            # Quietly handle errors by returning 0 for failed expressions
            return 0

    def _eval_expression(self, tokens):
        """Evaluates an expression node and returns its value."""
        if isinstance(tokens, list):
            # If it's a list of tokens, use _eval_token_expression
            return self._eval_token_expression(tokens)
        elif isinstance(tokens, Number):
            return tokens.value
        elif isinstance(tokens, Variable):
            # Handle variable reference
            var_name = tokens.name
            
            if var_name not in self.variables:
                # Try case-insensitive lookup
                for name in self.variables:
                    if name.upper() == var_name.upper():
                        var_name = name
                        break
                
            if var_name not in self.variables:
                raise NameError(f"Variable '{var_name}' is not defined")
                
            var_value = self.variables[var_name]
            
            # Check if the variable contains array values that need to be evaluated
            if isinstance(var_value, list):
                # For simple arrays of numbers, return as is
                if all(isinstance(item, (int, float)) for item in var_value):
                    return var_value
                    
                # For arrays with tokens or expressions, evaluate each item
                evaluated_values = []
                for i, item in enumerate(var_value):
                    if isinstance(item, Token):
                        if item.type == 'EXPRESSION':
                            # Handle expression token
                            evaluated_value = self._eval_token_expression(item.value)
                            evaluated_values.append(evaluated_value)
                        elif item.type == 'COMMA':
                            # Skip comma tokens
                            continue
                        elif item.type == 'NUMBER':
                            # Handle number token
                            evaluated_values.append(float(item.value))
                        else:
                            # Handle other token types
                            evaluated_values.append(self._eval_token_expression([item]))
                    else:
                        # Handle non-token values
                        evaluated_values.append(item)
                return evaluated_values
            
            return var_value
        elif isinstance(tokens, CellReference):
            if tokens.address not in self.grid:
                return 0  # Default value for empty cells
            return self.grid[tokens.address]
        elif isinstance(tokens, DynamicCellReference):
            # Evaluate the dynamic part (e.g., N in A&{N})
            dynamic_value = self._eval_expression(tokens.dynamic_part)
            # Construct the full cell address
            cell_address = f"{tokens.prefix}{dynamic_value}"
            if cell_address not in self.grid:
                return 0  # Default value for empty cells
            return self.grid[cell_address]
        elif isinstance(tokens, ArrayValues):
            # Evaluate array values
            return self._evaluate_array(tokens.values)
        elif isinstance(tokens, BinaryOp):
            left = self._eval_expression(tokens.left)
            right = self._eval_expression(tokens.right)
            if tokens.op == '+':
                return left + right
            elif tokens.op == '-':
                return left - right
            elif tokens.op == '*':
                return left * right
            elif tokens.op == '/':
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                return left / right
            elif tokens.op == '^':
                return left ** right
            else:
                raise ValueError(f"Unknown operator: {tokens.op}")
        elif isinstance(tokens, Function):
            if tokens.name == 'SUM':
                if isinstance(tokens.range, CellRange):
                    return self._eval_sum(tokens.range)
                elif isinstance(tokens.range, list):
                    # Handle sum of arguments
                    total = 0
                    for arg in tokens.range:
                        # Evaluate each argument
                        total += self._eval_expression(arg)
                    return total
            raise ValueError(f"Unknown function: {tokens.name}")
        else:
            return tokens  # Return as-is if not a known node type

    def _eval_sum(self, range_node):
        """Evaluates the SUM function for a range of cells."""
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

    def _eval_sum_range(self, start, end):
        """Evaluates the SUM function for a range of cells."""
        start_col, start_row = re.match(r'([A-Z]+)(\d+)', start).groups()
        end_col, end_row = re.match(r'([A-Z]+)(\d+)', end).groups()
        
        # Convert column letters to numbers
        start_col_num = self._get_col_number(start_col)
        end_col_num = self._get_col_number(end_col)
        
        # Ensure proper range order
        start_row, end_row = min(int(start_row), int(end_row)), max(int(start_row), int(end_row))
        start_col_num, end_col_num = min(start_col_num, end_col_num), max(start_col_num, end_col_num)
        
        total = 0
        for row in range(start_row, end_row + 1):
            for col in range(start_col_num, end_col_num + 1):
                col_letter = self._get_col_letters(col)
                cell_address = f"{col_letter}{row}"
                value = self.grid.get(cell_address, 0)
                total += value
        
        return total

    def _parse_cell(self, address):
        """Converts cell address (e.g., "A1") to row and column numbers."""
        match = re.match(r"([A-Z]+)(\d+)", address)
        if not match:
            raise ValueError(f"Invalid cell address: {address}")
        col, row = match.groups()
        # Convert column letters to numbers (A=1, B=2, etc.)
        col_num = sum((ord(c) - ord('A') + 1) * (26 ** i) 
                     for i, c in enumerate(reversed(col)))
        return int(row), col_num 