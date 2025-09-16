"""
Type processing functionality for GridLang compiler.
Handles type definitions, type code execution, and type-related operations.
"""

import re


class GridLangTypeProcessor:
    """Handles type definitions and type-related processing."""

    def __init__(self, compiler=None):
        self.compiler = compiler

    def _parse_type_def(self, lines, line_number=None):
        """Parse type definition lines and extract fields and executable code."""
        fields = {}
        executable_code = []

        for line in lines:
            if line.startswith(':'):
                # Standard field definition
                parts = line[1:].strip().split(' as ')
                if len(parts) != 2:
                    raise SyntaxError(
                        f"Invalid field definition: '{line}' at line {line_number}")
                field_name, field_type = parts
                fields[field_name.strip()] = field_type.strip()
            else:
                # Executable code inside type definition
                executable_code.append(line.strip())

        # Add executable code as a special field
        if executable_code:
            fields['_executable_code'] = executable_code

        return fields

    def _execute_type_code(self, code_lines, var_name, value_dict, line_number):
        """Execute code that was defined inside a type definition"""

        # Create a temporary scope for execution
        self.compiler.push_scope()

        # Add the type instance to the scope so code can reference it
        self.compiler.current_scope().define(var_name, value_dict, 'object')

        # Add a 'grid' field if it doesn't exist (for the binomial case)
        if 'grid' not in value_dict:
            value_dict['grid'] = {}

        try:
            for code_line in code_lines:
                if not code_line.strip():
                    continue

                # Handle different types of code lines
                if code_line.startswith('[') and ':=' in code_line:
                    # Assignment to grid: [B1] := 1
                    self._process_grid_assignment(
                        code_line, var_name, value_dict, line_number)
                elif code_line.lower().startswith('for '):
                    # For loop: For a in 2 to 10 AND b in 2 to a+1 do
                    self._process_type_for_loop(
                        code_line, code_lines, var_name, value_dict, line_number)
                    break  # Exit after processing the loop
                elif code_line.lower().startswith('let '):
                    # Let statement: Let grid{a, b} = grid{a-1, b-1} + grid{a-1, b}
                    self._process_type_let_statement(
                        code_line, var_name, value_dict, line_number)
                elif code_line.lower().startswith('end'):
                    # End of loop
                    break

        except Exception as e:
            raise
        finally:
            self.compiler.pop_scope()

    def _process_grid_assignment(self, line, var_name, value_dict, line_number):
        """Process grid assignment like [B1] := 1"""
        # Extract cell reference and value
        match = re.match(r'\[([A-Z]+\d+)\]\s*:=\s*(.+)$', line)
        if match:
            cell_ref, value_expr = match.groups()
            # Convert cell reference to grid coordinates
            col = re.match(r'([A-Z]+)', cell_ref).group(1)
            row = int(re.match(r'[A-Z]+(\d+)', cell_ref).group(1))

            # Convert column letters to numbers (A=1, B=2, etc.)
            col_num = 0
            for char in col:
                col_num = col_num * 26 + (ord(char.upper()) - ord('A') + 1)

            # Evaluate the value
            value = self.compiler.expr_evaluator.eval_expr(
                value_expr, self.compiler.current_scope().get_full_scope(), line_number)

            # Store in grid
            if 'grid' not in value_dict:
                value_dict['grid'] = {}
            value_dict['grid'][(row, col_num)] = value

    def _process_type_for_loop(self, loop_line, all_lines, var_name, value_dict, line_number):
        """Process for loop inside type definition"""
        # Extract loop variables and ranges
        # For a in 2 to 10 AND b in 2 to a+1 do
        match = re.match(
            r'For\s+(\w+)\s+in\s+(\d+)\s+to\s+(\d+)\s+AND\s+(\w+)\s+in\s+(\d+)\s+to\s+(.+?)\s+do', loop_line, re.I)
        if match:
            var1, start1, end1, var2, start2, end2_expr = match.groups()

            start1, end1, start2 = int(start1), int(end1), int(start2)

            # Find the loop body
            loop_body = []
            in_loop = False
            for line in all_lines:
                if line.lower().startswith('for '):
                    in_loop = True
                    continue
                elif line.lower().startswith('end'):
                    break
                elif in_loop:
                    loop_body.append(line)

            # Execute the loop
            for a in range(start1, end1 + 1):
                # Create a new scope for each iteration of the outer loop
                self.compiler.push_scope(is_private=False)

                # Set first loop variable in scope
                self.compiler.current_scope().define(var1, a, 'number')

                # Calculate end2 for this iteration of a
                try:
                    # Create a temporary scope with 'a' defined to evaluate the expression
                    temp_scope = {var1: a}
                    end2 = self.compiler.expr_evaluator.eval_expr(
                        end2_expr, temp_scope, line_number)
                    end2 = int(end2)
                except Exception as e:
                    end2 = start2

                for b in range(start2, end2 + 1):
                    # Create a new scope for each iteration of the inner loop
                    self.compiler.push_scope(is_private=False)

                    # Set second loop variable in scope
                    self.compiler.current_scope().define(var2, b, 'number')

                    # Execute loop body
                    for body_line in loop_body:
                        if body_line.lower().startswith('let '):
                            self._process_type_let_statement(
                                body_line, var_name, value_dict, line_number)

                    # Pop the scope for this inner loop iteration
                    self.compiler.pop_scope()

                # Pop the scope for this iteration
                self.compiler.pop_scope()
        else:
            raise SyntaxError(f"Unsupported loop syntax: {loop_line}")

    def _process_type_let_statement(self, line, var_name, value_dict, line_number):
        """Process let statement inside type definition"""
        # Let grid{a, b} = grid{a-1, b-1} + grid{a-1, b}
        match = re.match(r'Let\s+(\w+)\{([^}]+)\}\s*=\s*(.+)$', line, re.I)
        if match:
            field_name, indices_str, value_expr = match.groups()

            if field_name.lower() == 'grid':
                # Parse indices
                indices = [idx.strip() for idx in indices_str.split(',')]

                # Evaluate indices (they should be variables in scope)
                scope = self.compiler.current_scope().get_full_scope()
                try:
                    idx1 = scope.get(indices[0])
                    idx2 = scope.get(indices[1])

                    if idx1 is not None and idx2 is not None:
                        # Create a special scope for evaluating the value expression
                        # that includes the grid field and current loop variables
                        eval_scope = scope.copy()
                        eval_scope['grid'] = value_dict.get('grid', {})

                        # Handle grid access in the expression (e.g., grid{a-1, b-1})
                        # Replace grid{var1, var2} with actual values
                        processed_expr = value_expr
                        for idx_var in indices:
                            if idx_var in scope:
                                processed_expr = processed_expr.replace(
                                    f'grid{{{idx_var}}}', f'grid[{scope[idx_var]}]')

                        # Also handle grid{expr1, expr2} patterns
                        grid_pattern = r'grid\{([^}]+)\}'

                        def replace_grid_access(match):
                            grid_indices = match.group(1).split(',')
                            if len(grid_indices) == 2:
                                idx1_expr, idx2_expr = grid_indices
                                # Evaluate the index expressions
                                try:
                                    val1 = self.compiler.expr_evaluator.eval_expr(
                                        idx1_expr.strip(), scope, line_number)
                                    val2 = self.compiler.expr_evaluator.eval_expr(
                                        idx2_expr.strip(), scope, line_number)
                                    # Get the actual value from the grid
                                    grid_value = value_dict.get(
                                        'grid', {}).get((val1, val2))
                                    if grid_value is None:
                                        # If grid value doesn't exist, return 0 as default
                                        # This allows the loop to continue with valid expressions
                                        return "0"
                                    return str(grid_value)
                                except Exception as e:
                                    return "0"
                            return "0"

                        # Process grid access patterns
                        processed_expr = re.sub(
                            grid_pattern, replace_grid_access, processed_expr)

                        # Evaluate the processed expression
                        value = self.compiler.expr_evaluator.eval_expr(
                            processed_expr, eval_scope, line_number)

                        # Store in grid
                        if 'grid' not in value_dict:
                            value_dict['grid'] = {}
                        value_dict['grid'][(idx1, idx2)] = value

                except Exception as e:
                    raise
