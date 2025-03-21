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
                # Handle different types of values
                if isinstance(value, (int, float)):
                    print(f"{value:4}", end="")
                elif isinstance(value, Variable):
                    var_value = self.variables.get(value.name, "")
                    print(f"{var_value:4}", end="")
                else:
                    print(f"{str(value):4}", end="")
            print()  # New line after each row

    def _eval_expression(self, node):
        """Evaluates an expression node and returns its value."""
        if isinstance(node, Function):
            if node.name == 'SUM':
                if isinstance(node.range, CellRange):
                    return self._eval_sum(node.range)
                elif isinstance(node.range, list):
                    # Handle sum of arguments
                    total = 0
                    for arg in node.range:
                        # Evaluate each argument
                        if isinstance(arg, Variable):
                            if arg.name not in self.variables:
                                raise ValueError(f"Undefined variable: {arg.name}")
                            total += self.variables[arg.name]
                        elif isinstance(arg, CellReference):
                            if arg.address not in self.grid:
                                total += 0  # Default value for empty cells
                            else:
                                total += self.grid[arg.address]
                        elif isinstance(arg, Number):
                            total += arg.value
                        else:
                            raise ValueError(f"Unexpected argument type: {type(arg)}")
                    return total
            raise ValueError(f"Unknown function: {node.name}")
        elif isinstance(node, Number):
            return node.value
        elif isinstance(node, Variable):
            if node.name not in self.variables:
                raise NameError(f"Variable '{node.name}' is not defined")
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
            evaluated_values = []
            for value in node.values:
                if isinstance(value, (Number, Variable, CellReference)):
                    evaluated_values.append(self._eval_expression(value))
                elif isinstance(value, BinaryOp):
                    evaluated_values.append(self._eval_expression(value))
                elif isinstance(value, list):  # Handle nested expressions
                    # Create a temporary parser for the expression tokens
                    temp_parser = Parser(value)
                    temp_parser.pos = 0
                    expr = temp_parser._expression()
                    evaluated_values.append(self._eval_expression(expr))
                else:
                    evaluated_values.append(value)
            return evaluated_values
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
                    raise ZeroDivisionError("Division by zero")
                return left / right
            elif node.op == '^':
                return left ** right
            else:
                raise ValueError(f"Unknown operator: {node.op}")
        else:
            raise ValueError(f"Unknown node type: {type(node)}")

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