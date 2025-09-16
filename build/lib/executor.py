
import re
import math
import pyarrow as pa
from expression import ExpressionEvaluator
from array_handler import ArrayHandler
from control_flow import GridLangControlFlow
from parser import GridLangParser
from utils import col_to_num, num_to_col, split_cell, offset_cell, validate_cell_ref


class GridLangExecutor:
    def __init__(self):
        self.control_flow = GridLangControlFlow(self)
        self.exit_loop = False  # Simple boolean flag for breaking out of loops
        self.output_variables = []  # List of output variables
        self.output_values = {}  # Dictionary to store output values

    def get_global_scope(self):
        """Get the global scope from the compiler"""
        if hasattr(self, 'compiler') and self.compiler:
            return self.compiler.scopes[0]  # First scope is always global
        else:
            # Fallback if no compiler reference
            return self.current_scope()

    def collect_input_output_variables(self):
        """Collect all input and output variables from the current scope"""
        global_scope = self.get_global_scope()
        self.input_variables = list(global_scope.input_variables)
        self.output_variables = list(global_scope.output_variables)
        # Add 'output' as a default output variable for push() calls
        if 'output' not in self.output_variables:
            self.output_variables.append('output')

    def run(self, code, args=None):
        self._reset_state()
        lines, label_lines, dim_lines = self._preprocess_code(code)
        self._process_declarations_and_labels(lines, label_lines, dim_lines)
        # Store the root scope after declarations
        self.root_scope = self.current_scope()

        # Collect input/output variables and set input values
        self.collect_input_output_variables()
        self.set_input_values(args)

        # Pre-scan blocks to map structure before processing
        self.control_flow.pre_scan_blocks(lines)

        i = 0
        while i < len(lines):
            line, line_number = lines[i]
            if not line.strip():
                i += 1
                continue
            # Skip global declarations as they're already processed in _process_declarations_and_labels
            if line.startswith(':'):
                i += 1
                continue

            if line.lower().startswith("for ") and 'grid dim' in line.lower():
                m = re.match(r'^\s*FOR\s+(.+?)(?:\s+DO\s*$|\s*$)', line, re.I)
                if not m:
                    raise SyntaxError(
                        f"Invalid FOR syntax at line {line_number}")
                var_def = m.group(1).strip()
                var, type_name, constraints, expr = self._parse_variable_def(
                    var_def, line_number)

                if type_name and 'with' in constraints and type_name in self.types_defined:
                    current_tensor = var
                    with_constraints = constraints.get('with', {})
                    dim_constraints = constraints.get('dim', {})
                    if dim_constraints:
                        dims = dim_constraints.get('dims', [])
                        shape = [size for label, size in dims]
                        pa_type = pa.float64()
                        matrix_data = dim_constraints.get('matrix_data')
                        data_var = dim_constraints.get('data_var')
                        if data_var:
                            scope = self.current_scope().get_full_scope()
                            if data_var not in scope:
                                raise NameError(
                                    f"Variable '{data_var}' not defined at line {line_number}")
                            matrix_data = scope[data_var]
                        default_value = dim_constraints.get('value', float(
                            expr) if expr and not isinstance(expr, list) else None)
                        grid_data = self.array_handler.create_array(
                            shape, default_value, pa_type, line_number, matrix_data=matrix_data, is_grid_dim=True)

                        tensor_struct = {f: with_constraints.get(
                            f) for f in self.types_defined[type_name]}
                        tensor_struct['grid'] = grid_data['array']
                        tensor_struct['original_shape'] = grid_data['original_shape']
                        tensor_struct['constraints'] = constraints

                        self.current_scope().define(var, tensor_struct, type_name,
                                                    constraints, is_uninitialized=False)
                        for field in self.types_defined[type_name]:
                            if field in with_constraints:
                                self.current_scope().define(
                                    f"{var}.{field}", with_constraints[field], 'text')
                i += 1
            elif line.lower().startswith("push "):
                # Handle PUSH statements: Result.Push($"Hello, {FirstName}!")
                m = re.match(
                    r'^\s*([\w_]+)\.Push\s*\(\s*(.+?)\s*\)\s*$', line, re.I)
                if m:
                    output_var, value_expr = m.groups()
                    output_var = output_var.strip()
                    value_expr = value_expr.strip()

                    # Check if the variable is an output - use global scope
                    if not self.get_global_scope().is_output(output_var):
                        raise ValueError(
                            f"'{output_var}' is not an output variable at line {line_number}")

                    # Evaluate the expression to get the value to push
                    try:
                        value = self.expr_evaluator.eval_expr(
                            value_expr, self.get_global_scope().get_evaluation_scope(), line_number)
                        # Push the value through the output - use global scope
                        self.get_global_scope().push_value(output_var, value, line_number)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to evaluate PUSH expression at line {line_number}: {e}")
                else:
                    # Try to handle push() function calls (e.g., push(mid))
                    m_func = re.match(
                        r'^\s*push\s*\(\s*([^)]+)\s*\)\s*$', line, re.I)
                    if m_func:
                        value_expr = m_func.group(1).strip()
                        # For now, just evaluate the expression and print it
                        # This can be expanded later to handle actual pushing logic
                        try:
                            value = self.expr_evaluator.eval_expr(
                                value_expr, self.current_scope().get_evaluation_scope(), line_number)
                        except Exception as e:
                            pass
                    else:
                        raise SyntaxError(
                            f"Invalid PUSH syntax at line {line_number}")
                i += 1
            # Check for one-liner FOR loops with .push() first (before .push() detection)
            # Check for FOR loops with .push() (both one-liner and multi-line)
            elif line.lower().strip().startswith("for ") and ' do ' in line.lower() and '.push(' in line.lower():

                # Check if this is a multi-line FOR loop by looking for incomplete string literals
                if line.count('"') % 2 == 1:  # Odd number of quotes means incomplete string

                    # Collect all lines until we find the complete .push() call
                    complete_lines = [line]
                    current_i = i
                    current_line = line

                    # Look for the closing parenthesis and quote
                    while not (current_line.strip().endswith(')') and current_line.count('"') % 2 == 0):
                        current_i += 1
                        if current_i >= len(lines):
                            raise SyntaxError(
                                f"Unterminated string literal in FOR loop starting at line {line_number}")

                        # Get the line content from the tuple
                        current_line = lines[current_i][0]
                        complete_lines.append(current_line)

                    # Combine all lines into a single executable part
                    executable_part = ' '.join(complete_lines)

                    # Extract the FOR loop part from the first line
                    first_line = complete_lines[0]
                    if ' do ' in first_line.lower():
                        parts = first_line.split(' do ', 1)
                        if len(parts) == 2:
                            for_part = parts[0].strip()
                            # Remove the partial executable part from the first line
                            executable_part = executable_part.replace(
                                first_line, '').strip()
                            if executable_part.startswith('do '):
                                executable_part = executable_part[3:].strip()

                            # Parse the FOR loop part
                            for_match = re.match(
                                r'^\s*for\s+(.+?)$', for_part, re.I)
                            if for_match:
                                var_defs = for_match.group(1).strip()

                                # Parse the range expression
                                range_match = re.match(
                                    r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', var_defs, re.I)
                                if range_match:
                                    var_name, range_expr, step_str, index_var = range_match.groups()
                                    step = int(step_str) if step_str else 1

                                    # Parse the range expression
                                    if ' to ' in range_expr:
                                        range_parts = range_expr.split(' to ')
                                        start = int(range_parts[0].strip())
                                        end = int(range_parts[1].strip())

                                        # Handle negative steps correctly
                                        if step < 0:
                                            values = list(
                                                range(start, end - 1, step))
                                        else:
                                            values = list(
                                                range(start, end + 1, step))

                                        # Process the executable part for each value
                                        if '.push(' in executable_part.lower():
                                            for idx, value in enumerate(values):
                                                self.push_scope(
                                                    is_private=True, is_loop_scope=True)
                                                self.current_scope().define(var_name, value, 'number')
                                                if index_var:
                                                    self.current_scope().define(index_var, idx + 1, 'number')

                                                # Process the .push() call
                                                self._process_push_call(
                                                    executable_part, line_number)
                                                self.pop_scope()

                                        # Skip all the lines we processed in the main loop
                                        i = current_i + 1
                                        continue
                                    else:
                                        i = current_i + 1
                                        continue
                                else:
                                    i = current_i + 1
                                    continue
                            else:
                                i = current_i + 1
                                continue
                        else:
                            i = current_i + 1
                            continue
                    else:
                        i = current_i + 1
                        continue

                else:

                    # Split the line at " do " to separate loop definition and executable code
                    parts = line.split(' do ', 1)
                    if len(parts) == 2:
                        for_part = parts[0].strip()
                        executable_part = parts[1].strip()

                        # Parse the FOR loop part
                        for_match = re.match(
                            r'^\s*for\s+(.+?)$', for_part, re.I)
                        if for_match:
                            var_defs = for_match.group(1).strip()

                            # Parse the range expression
                            range_match = re.match(
                                r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', var_defs, re.I)
                            if range_match:
                                var_name, range_expr, step_str, index_var = range_match.groups()
                                step = int(step_str) if step_str else 1

                                # Parse the range expression
                                if ' to ' in range_expr:
                                    range_parts = range_expr.split(' to ')
                                    start = int(range_parts[0].strip())
                                    end = int(range_parts[1].strip())

                                    # Handle negative steps correctly
                                    if step < 0:
                                        values = list(
                                            range(start, end - 1, step))
                                    else:
                                        values = list(
                                            range(start, end + 1, step))

                                    # Process the executable part for each value
                                    if '.push(' in executable_part.lower():
                                        for idx, value in enumerate(values):
                                            self.push_scope(
                                                is_private=True, is_loop_scope=True)
                                            self.current_scope().define(var_name, value, 'number')
                                            if index_var:
                                                self.current_scope().define(index_var, idx + 1, 'number')

                                            # Process the .push() call
                                            self._process_push_call(
                                                executable_part, line_number)
                                            self.pop_scope()

                                    # Skip this line in the main loop - it's been fully processed
                                    i += 1
                                    continue
                                else:
                                    i += 1
                                    continue
                            else:
                                i += 1
                                continue
                        else:
                            i += 1
                            continue
                    else:
                        i += 1
                        continue

            elif re.search(r'\.push\(', line, re.I):
                # Use the push processor to handle this properly
                self._process_push_call(line, line_number)
                i += 1
            elif re.search(r'\bpush\s*\(', line, re.I):
                # Handle push() function calls (e.g., push(mid))
                m = re.match(
                    r'^\s*push\s*\(\s*([^)]+)\s*\)\s*$', line, re.I)
                if m:
                    value_expr = m.group(1).strip()
                    # Evaluate the expression and add it to output values
                    try:
                        value = self.expr_evaluator.eval_expr(
                            value_expr, self.current_scope().get_evaluation_scope(), line_number)

                        # Add to output values for the 'output' variable
                        if 'output' not in self.output_values:
                            self.output_values['output'] = []
                        self.output_values['output'].append(value)
                        # Also add to output_variables so it gets displayed
                        if 'output' not in self.output_variables:
                            self.output_variables.append('output')
                    except Exception as e:
                        raise ValueError(
                            f"Failed to evaluate push() expression at line {line_number}: {e}")
                else:
                    raise SyntaxError(
                        f"Invalid push() syntax at line {line_number}")
                i += 1
            # Array access like names(0) and names[3] will be handled by the expression evaluator
            # No need for special handling here
            elif re.search(r'([\w_]+)!(\w+)\.Label\s*\{([^}]+)\}', line):
                # Handle named dimension label assignment like Results!Quarter.Label{"Q1", "Q2", "Q3", "Q4"}
                m = re.match(
                    r'^\s*([\w_]+)!(\w+)\.Label\s*\{([^}]+)\}\s*$', line)
                if m:
                    var_name, dim_name, labels_str = m.groups()

                    # Parse the labels
                    labels = [label.strip().strip('"')
                              for label in labels_str.split(',')]

                    # Store the labels in the compiler for later use
                    if not hasattr(self, 'dim_labels'):
                        self.dim_labels = {}
                    if var_name not in self.dim_labels:
                        self.dim_labels[var_name] = {}
                    self.dim_labels[var_name][dim_name] = {
                        label: i for i, label in enumerate(labels)}

                    i += 1
                    continue
                else:
                    # Not a valid label pattern, skip this line
                    i += 1
                    continue
            # Named dimension access like Results!Quarter("Q2") will be handled by the expression evaluator
            # No need for special handling here
            elif line.lower().strip().startswith("for "):
                # Continue with normal FOR loop processing for multi-line FOR loops
                pass

                # Handle special case: for element = array(index) (array element assignment)
                m = re.match(
                    r'^\s*for\s+([\w_]+)\s*=\s*([\w_]+)\(([^)]+)\)\s*$', line, re.I)
                if m:
                    var_name, array_name, index_expr = m.groups()

                    try:
                        # Evaluate the index expression
                        index_value = self.expr_evaluator.eval_expr(
                            index_expr, self.current_scope().get_evaluation_scope(), line_number)

                        # Get the array
                        array = self.current_scope().get(array_name)
                        if array is None:
                            raise ValueError(
                                f"Array '{array_name}' is not defined at line {line_number}")

                        # Convert to list if it's a set
                        if isinstance(array, set):
                            array = list(array)

                        # Access the array element (1-based indexing)
                        if isinstance(index_value, (int, float)):
                            idx = int(index_value) - 1  # Convert to 0-based
                            if 0 <= idx < len(array):
                                element_value = array[idx]

                                # Define the variable with the element value
                                self.current_scope().define(var_name, element_value,
                                                            'number', {}, is_uninitialized=False)
                            else:
                                raise IndexError(
                                    f"Index {index_value} out of range for array '{array_name}' at line {line_number}")
                        else:
                            raise ValueError(
                                f"Invalid index type {type(index_value)} for array access at line {line_number}")

                    except Exception as e:
                        raise ValueError(
                            f"Failed to process array element assignment at line {line_number}: {e}")

                    i += 1
                    continue

                # Handle For var as type dim {dimensions} syntax
                m = re.match(
                    r'^\s*for\s+([\w_]+)\s+as\s+(\w+)\s+dim\s*(\{[^}]*\})', line, re.I)
                if m:
                    var_name, type_name, dim_str = m.groups()

                    # Parse the dimension string
                    dim_str = dim_str.strip()
                    if dim_str.startswith('{') and dim_str.endswith('}'):
                        dim_content = dim_str[1:-1].strip()
                        # Parse dimensions like {30, 2}
                        dim_parts = [part.strip()
                                     for part in dim_content.split(',')]
                        shape = []
                        for part in dim_parts:
                            try:
                                shape.append(int(part))
                            except ValueError:
                                # Handle named dimensions or other formats
                                shape.append(part)

                        # Create the array with the specified shape
                        default_value = 0 if type_name.lower() == 'number' else None
                        array_data = self.array_handler.create_array(
                            shape, default_value, pa.float64(), line_number)

                        # Define the variable with the created array
                        self.current_scope().define(var_name, array_data,
                                                    type_name.lower(), {'dim': dim_str}, False)
                    else:
                        constraints = {}
                        self.current_scope().define(var_name, None, type_name.lower(), constraints, True)
                    i += 1
                    continue

                # Handle For var as type dim number syntax
                m = re.match(
                    r'^\s*for\s+([\w_]+)\s+as\s+(\w+)\s+dim\s+(\d+)', line, re.I)
                if m:
                    var_name, type_name, dim_size = m.groups()

                    # Create an array with the specified size
                    dim_size = int(dim_size)
                    # Initialize array with zeros
                    initial_array = pa.array(
                        [0.0] * dim_size, type=pa.float64())

                    # Define the variable with the initialized array
                    self.current_scope().define(var_name, initial_array, type_name.lower(),
                                                {'dim': [(None, dim_size)]}, False)
                    i += 1
                    continue

                # Check for AND syntax in single FOR line first
                m = re.match(r'^\s*for\s+(.+?)(?:\s+do\s*$|\s*$)', line, re.I)
                if m:
                    var_defs = m.group(1).strip()
                    # Check if this is actually a single line FOR loop (contains .push() or :=)
                    # even if it ends with 'do'
                    has_executable = '.push(' in line.lower() or ':=' in line
                    is_block = line.strip().lower().endswith('do') and not has_executable

                    # If this is a single line FOR loop with executable code, handle it specially
                    if has_executable:
                        # Extract the FOR loop part and the executable part
                        # The line format is: "For var in range do executable_code"
                        # We need to split at " do " and process both parts
                        if ' do ' in line.lower():
                            parts = line.split(' do ', 1)
                            if len(parts) == 2:
                                for_part = parts[0].strip()
                                executable_part = parts[1].strip()

                                # Parse the FOR loop part
                                for_match = re.match(
                                    r'^\s*for\s+(.+?)(?:\s+do\s*$|\s*$)', for_part, re.I)
                                if for_match:
                                    var_defs = for_match.group(1).strip()

                                    # Parse the range expression
                                    range_match = re.match(
                                        r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', var_defs, re.I)
                                    if range_match:
                                        var_name, range_expr, step_str, index_var = range_match.groups()
                                        step = int(step_str) if step_str else 1

                                        # Parse the range expression
                                        if ' to ' in range_expr:
                                            range_parts = range_expr.split(
                                                ' to ')
                                            start = int(range_parts[0].strip())
                                            end = int(range_parts[1].strip())
                                            # Handle negative steps correctly
                                            if step < 0:
                                                values = list(
                                                    range(start, end - 1, step))
                                            else:
                                                values = list(
                                                    range(start, end + 1, step))

                                            # Process the executable part for each value
                                            if '.push(' in executable_part.lower():
                                                for idx, value in enumerate(values):
                                                    self.push_scope(
                                                        is_private=True, is_loop_scope=True)
                                                    self.current_scope().define(var_name, value, 'number')
                                                    if index_var:
                                                        self.current_scope().define(index_var, idx + 1, 'number')

                                                    # Process the .push() call
                                                    self._process_push_call(
                                                        executable_part, line_number)
                                                    self.pop_scope()

                                                # Skip this line in the main loop
                                                i += 1
                                                continue
                                            else:
                                                i += 1
                                                continue
                                        else:
                                            i += 1
                                            continue
                                    else:
                                        i += 1
                                        continue
                                else:
                                    i += 1
                                    continue
                            else:
                                i += 1
                                continue
                        else:
                            i += 1
                            continue

                    if ' AND ' in var_defs:
                        and_parts = var_defs.split(' AND ')
                        and_loops = []

                        for part in and_parts:
                            part = part.strip()
                            # Parse each part: "var in range" or "var in range index idx"
                            # First, find where the range expression ends
                            range_end = part.find(' step ')
                            if range_end == -1:
                                range_end = part.find(' index ')
                            if range_end == -1:
                                range_end = part.find(' Index ')
                            if range_end == -1:
                                range_end = len(part)

                            range_expr = part[part.find(
                                ' in ') + 4:range_end].strip()
                            var_name = part[:part.find(' in ')].strip()

                            # Extract step and index if present
                            step_str = None
                            index_var = None
                            remaining = part[range_end:].strip()

                            # Clean up range_expr if it contains index keyword
                            if ' index ' in range_expr:
                                range_expr = range_expr[:range_expr.find(
                                    ' index ')].strip()
                            elif ' Index ' in range_expr:
                                range_expr = range_expr[:range_expr.find(
                                    ' Index ')].strip()

                            # Check for step
                            step_match = re.search(
                                r'step\s+(\d+)', remaining, re.I)
                            if step_match:
                                step_str = step_match.group(1)
                                remaining = remaining[:step_match.start(
                                )] + remaining[step_match.end():]

                            # Check for index
                            index_match = re.search(
                                r'index\s+([\w_]+)', remaining, re.I)
                            if index_match:
                                index_var = index_match.group(1)
                            else:
                                pass

                            step = int(step_str) if step_str else 1

                            # Parse the range expression
                            if ' to ' in range_expr:
                                range_parts = range_expr.split(' to ')
                                start_expr = range_parts[0].strip()
                                end_expr = range_parts[1].strip()

                                # Check if this is a dynamic range first
                                is_dynamic = False
                                # Check if expressions contain variables (not just numbers)
                                if not start_expr.replace('.', '').replace('-', '').isdigit() or not end_expr.replace('.', '').replace('-', '').replace('+', '').replace('*', '').replace('/', '').replace('(', '').replace(')', '').isdigit():
                                    is_dynamic = True
                                    values = None  # Will be computed during execution
                                else:
                                    # Static range, compute now
                                    start = int(start_expr)
                                    end = int(end_expr)
                                    values = list(range(start, end + 1, step))
                            elif range_expr.startswith('{') and range_expr.endswith('}'):
                                values_str = range_expr[1:-1].split(',')
                                values = []
                                for v in values_str:
                                    v_clean = v.strip()
                                    if v_clean.startswith('"') and v_clean.endswith('"'):
                                        values.append(v_clean[1:-1])
                                    elif v_clean.startswith("'") and v_clean.endswith("'"):
                                        values.append(v_clean[1:-1])
                                    else:
                                        try:
                                            values.append(float(v_clean))
                                        except ValueError:
                                            values.append(v_clean)
                            else:
                                raise SyntaxError(
                                    f"Invalid range expression: {range_expr} at line {line_number}")

                            # For sets, use the already computed values
                            if ' to ' not in range_expr:
                                is_dynamic = False

                            and_loops.append({
                                'var_name': var_name,
                                'values': values,
                                'index_var': index_var,
                                'step': step,
                                'is_dynamic': is_dynamic,
                                'start_expr': start_expr if is_dynamic else None,
                                'end_expr': end_expr if is_dynamic else None
                            })

                        if len(and_loops) >= 2:

                            if is_block:
                                # Process AND loops with block
                                # For dynamic ranges, we need to compute values during execution
                                # Start with the first loop's values
                                current_values = []
                                for loop in and_loops:
                                    if loop['is_dynamic']:
                                        # For dynamic loops, we'll compute values during execution
                                        current_values.append([])
                                    else:
                                        current_values.append(loop['values'])

                                # Generate combinations dynamically
                                def generate_combinations(loop_idx=0, current_combo=[]):
                                    if loop_idx >= len(and_loops):
                                        yield current_combo
                                        return

                                    loop = and_loops[loop_idx]
                                    if loop['is_dynamic']:
                                        # Compute dynamic range for this iteration
                                        # We need to evaluate the expressions with current variable values
                                        self.push_scope(
                                            is_private=True, is_loop_scope=True)
                                        # Set up variables from previous loops
                                        for i, prev_loop in enumerate(and_loops[:loop_idx]):
                                            self.current_scope().define(
                                                prev_loop['var_name'], current_combo[i], 'number')

                                        # Evaluate the dynamic range
                                        start = int(self.expr_evaluator.eval_expr(
                                            loop['start_expr'], self.current_scope().get_evaluation_scope(), line_number))
                                        end = int(self.expr_evaluator.eval_expr(
                                            loop['end_expr'], self.current_scope().get_evaluation_scope(), line_number))
                                        dynamic_values = list(
                                            range(start, end + 1, loop['step']))
                                        self.pop_scope()

                                        for val in dynamic_values:
                                            yield from generate_combinations(loop_idx + 1, current_combo + [val])
                                    else:
                                        # Static range
                                        for val in loop['values']:
                                            yield from generate_combinations(loop_idx + 1, current_combo + [val])

                                value_combinations = list(
                                    generate_combinations())

                                # Collect block lines
                                for_block_lines = []
                                i += 1
                                depth = 1
                                while i < len(lines) and depth > 0:
                                    next_line, next_line_number = lines[i]
                                    next_line_clean = next_line.strip().lower()
                                    if self._is_keyword(next_line_clean, "end"):
                                        depth -= 1
                                        if depth == 0:
                                            break
                                    elif next_line_clean.endswith("do"):
                                        depth += 1
                                    for_block_lines.append(
                                        (next_line, next_line_number))
                                    i += 1

                                # Execute for each combination
                                # Execute for each combination
                                for combo_idx, combo in enumerate(value_combinations):
                                    self.push_scope(
                                        is_private=True, is_loop_scope=True)
                                    for loop_idx, loop in enumerate(and_loops):
                                        self.current_scope().define(
                                            loop['var_name'], combo[loop_idx], 'number')
                                        if loop['index_var']:
                                            # For AND loops, we need to track the iteration count for each loop separately
                                            # This is more complex than regular loops because we're doing a product
                                            # We need to calculate which iteration this is for each individual loop
                                            loop_values = loop['values']
                                            current_value = combo[loop_idx]
                                            # Find the position of this value in the loop's values (1-based)
                                            value_index = loop_values.index(
                                                current_value) + 1
                                            # If index_var already exists in the current scope, update it instead of redefining
                                            if loop['index_var'] in self.current_scope().variables:
                                                self.current_scope().update(
                                                    loop['index_var'], value_index)
                                            else:
                                                self.current_scope().define(
                                                    loop['index_var'], value_index, 'number')
                                    self.control_flow._process_block(
                                        for_block_lines)

                                    # Check if exit loop was requested immediately after processing block
                                    if self.exit_loop:
                                        self.exit_loop = False
                                        self.pop_scope()
                                        break

                                    self.pop_scope()

                                continue
                            else:
                                # Single line AND loops - find assignment
                                if i + 1 < len(lines):
                                    next_line, next_line_number = lines[i + 1]
                                    if ':=' in next_line:
                                        # For dynamic ranges, we need to compute values during execution
                                        # Generate combinations dynamically
                                        def generate_combinations(loop_idx=0, current_combo=[]):
                                            if loop_idx >= len(and_loops):
                                                yield current_combo
                                                return

                                            loop = and_loops[loop_idx]
                                            if loop['is_dynamic']:
                                                # Compute dynamic range for this iteration
                                                # We need to evaluate the expressions with current variable values
                                                self.push_scope(
                                                    is_private=True, is_loop_scope=True)
                                                # Set up variables from previous loops
                                                for i, prev_loop in enumerate(and_loops[:loop_idx]):
                                                    self.current_scope().define(
                                                        prev_loop['var_name'], current_combo[i], 'number')

                                                # Evaluate the dynamic range
                                                start = int(self.expr_evaluator.eval_expr(
                                                    loop['start_expr'], self.current_scope().get_evaluation_scope(), line_number))
                                                end = int(self.expr_evaluator.eval_expr(
                                                    loop['end_expr'], self.current_scope().get_evaluation_scope(), line_number))
                                                dynamic_values = list(
                                                    range(start, end + 1, loop['step']))
                                                self.pop_scope()

                                                for val in dynamic_values:
                                                    yield from generate_combinations(loop_idx + 1, current_combo + [val])
                                            else:
                                                # Static range
                                                for val in loop['values']:
                                                    yield from generate_combinations(loop_idx + 1, current_combo + [val])

                                        value_combinations = list(
                                            generate_combinations())

                                        for combo in value_combinations:
                                            self.push_scope(
                                                is_private=True, is_loop_scope=True)
                                            for loop_idx, loop in enumerate(and_loops):
                                                self.current_scope().define(
                                                    loop['var_name'], combo[loop_idx], 'number')
                                                if loop['index_var']:
                                                    # If index_var already exists in the current scope, update it instead of redefining
                                                    if loop['index_var'] in self.current_scope().variables:
                                                        self.current_scope().update(
                                                            loop['index_var'], loop_idx + 1)
                                                    else:
                                                        self.current_scope().define(
                                                            loop['index_var'], loop_idx + 1, 'number')
                                            self.array_handler.evaluate_line_with_assignment(
                                                next_line, next_line_number, self.current_scope().get_evaluation_scope())
                                            self.pop_scope()

                                        i += 2
                                        continue

                # Check for nested FOR loops
                nested_for_loops = []
                current_i = i
                while current_i < len(lines) and lines[current_i][0].strip().lower().startswith("for "):
                    current_line, current_line_number = lines[current_i]
                    m = re.match(
                        r'^\s*FOR\s+(.+?)(?:\s+do\s*$|\s*$)', current_line, re.I)
                    if not m:
                        break
                    var_defs = m.group(1).strip()
                    is_block = current_line.strip().lower().endswith('do')

                    # Check for FOR loops with ranges or sets
                    range_match = re.match(
                        r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', var_defs, re.I)
                    if range_match:
                        var_name, range_expr, step_str, index_var = range_match.groups()
                        step = int(step_str) if step_str else 1

                        # Parse the range expression
                        if ' to ' in range_expr:
                            # Clean up the range expression by removing any step part
                            clean_range_expr = range_expr.split(
                                ' step ')[0] if ' step ' in range_expr else range_expr
                            range_parts = clean_range_expr.split(' to ')

                            # Evaluate start and end as expressions instead of literal integers
                            try:
                                start_expr = range_parts[0].strip()
                                end_expr = range_parts[1].strip()

                                # Evaluate start expression
                                start_value = self.expr_evaluator.eval_expr(
                                    start_expr, self.current_scope().get_evaluation_scope(), current_line_number)
                                start = int(start_value)

                                # Evaluate end expression
                                end_value = self.expr_evaluator.eval_expr(
                                    end_expr, self.current_scope().get_evaluation_scope(), current_line_number)
                                end = int(end_value)

                                values = list(range(start, end + 1, step))
                            except Exception as e:
                                # Fallback to original behavior for simple literals
                                start = int(range_parts[0].strip())
                                end = int(range_parts[1].strip())
                                values = list(range(start, end + 1, step))
                        elif range_expr.startswith('{') and range_expr.endswith('}'):
                            values_str = range_expr[1:-1].split(',')
                            values = []
                            for v in values_str:
                                v_clean = v.strip()
                                if v_clean.startswith('"') and v_clean.endswith('"'):
                                    values.append(v_clean[1:-1])
                                elif v_clean.startswith("'") and v_clean.endswith("'"):
                                    values.append(v_clean[1:-1])
                                else:
                                    try:
                                        values.append(float(v_clean))
                                    except ValueError:
                                        values.append(v_clean)
                        else:
                            break

                        nested_for_loops.append({
                            'var_name': var_name,
                            'values': values,
                            'index_var': index_var,
                            'is_block': is_block,
                            'line_number': current_line_number
                        })
                        current_i += 1
                    else:
                        break

                if len(nested_for_loops) == 2 and not any(loop['is_block'] for loop in nested_for_loops):
                    # Use zip logic for exactly two consecutive FORs
                    assignment_line = None
                    assignment_line_number = None
                    if current_i < len(lines):
                        next_line, next_line_number = lines[current_i]
                        if ':=' in next_line:
                            assignment_line = next_line
                            assignment_line_number = next_line_number
                            current_i += 1
                    if assignment_line:
                        vals1 = nested_for_loops[0]['values']
                        vals2 = nested_for_loops[1]['values']
                        var1 = nested_for_loops[0]['var_name']
                        var2 = nested_for_loops[1]['var_name']
                        for a, b in zip(vals1, vals2):
                            self.push_scope(is_private=True,
                                            is_loop_scope=True)
                            self.current_scope().define(var1, a, 'number')
                            self.current_scope().define(var2, b, 'number')
                            self.array_handler.evaluate_line_with_assignment(
                                assignment_line, assignment_line_number, self.current_scope().get_evaluation_scope())
                            self.pop_scope()
                        i = current_i
                        continue

                if len(nested_for_loops) > 1:
                    # Handle nested FOR loops

                    # Find the assignment line after the nested loops
                    assignment_line = None
                    assignment_line_number = None
                    if current_i < len(lines):
                        next_line, next_line_number = lines[current_i]
                        if ':=' in next_line:
                            assignment_line = next_line
                            assignment_line_number = next_line_number
                            current_i += 1

                    if assignment_line:
                        # Generate all combinations of nested loop values
                        from itertools import product
                        value_combinations = list(
                            product(*[loop['values'] for loop in nested_for_loops]))

                        # Execute for each combination
                        for combo in value_combinations:
                            self.push_scope(is_private=True,
                                            is_loop_scope=True)
                            for loop_idx, loop in enumerate(nested_for_loops):
                                self.current_scope().define(
                                    loop['var_name'], combo[loop_idx], 'number')
                                if loop['index_var']:
                                    # If index_var already exists in the current scope, update it instead of redefining
                                    if loop['index_var'] in self.current_scope().variables:
                                        self.current_scope().update(
                                            loop['index_var'], loop_idx + 1)
                                    else:
                                        self.current_scope().define(
                                            loop['index_var'], loop_idx + 1, 'number')
                            self.array_handler.evaluate_line_with_assignment(
                                assignment_line, assignment_line_number, self.current_scope().get_evaluation_scope())
                            self.pop_scope()

                        # Skip all the FOR lines and the assignment line
                        i = current_i
                        continue
                    else:
                        # No assignment line found, process as regular FOR loops
                        pass
                    # Use zip logic for exactly two consecutive FORs
                    assignment_line = None
                    assignment_line_number = None
                    if current_i < len(lines):
                        next_line, next_line_number = lines[current_i]
                        if ':=' in next_line:
                            assignment_line = next_line
                            assignment_line_number = next_line_number
                            current_i += 1
                    if assignment_line:
                        vals1 = nested_for_loops[0]['values']
                        vals2 = nested_for_loops[1]['values']
                        var1 = nested_for_loops[0]['var_name']
                        var2 = nested_for_loops[1]['var_name']
                        for a, b in zip(vals1, vals2):
                            self.push_scope(is_private=True,
                                            is_loop_scope=True)
                            self.current_scope().define(var1, a, 'number')
                            self.current_scope().define(var2, b, 'number')
                            self.array_handler.evaluate_line_with_assignment(
                                assignment_line, assignment_line_number, self.current_scope().get_evaluation_scope())
                            self.pop_scope()
                        i = current_i
                        continue

                # Process as single FOR loop (original logic)
                m = re.match(r'^\s*for\s+(.+?)(?:\s+do\s*$|\s*$)', line, re.I)
                if not m:
                    raise SyntaxError(
                        f"Invalid FOR syntax at line {line_number}")
                var_defs = m.group(1).strip()
                is_block = line.strip().lower().endswith('do')

                # Check for FOR loops with ranges or sets
                range_match = re.match(
                    r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', var_defs, re.I)
                if range_match:
                    var_name, range_expr, step_str, index_var = range_match.groups()
                    step = int(step_str) if step_str else 1

                    # Parse the range expression
                    if ' to ' in range_expr:
                        # Range like "1 to 3"
                        range_parts = range_expr.split(' to ')

                        # Evaluate start and end as expressions instead of literal integers
                        try:
                            start_expr = range_parts[0].strip()
                            end_expr = range_parts[1].strip()

                            # Evaluate start expression
                            start_value = self.expr_evaluator.eval_expr(
                                start_expr, self.current_scope().get_evaluation_scope(), current_line_number)
                            start = int(start_value)

                            # Evaluate end expression
                            end_value = self.expr_evaluator.eval_expr(
                                end_expr, self.current_scope().get_evaluation_scope(), current_line_number)
                            end = int(end_value)
                        except Exception as e:
                            # Fallback to original behavior for simple literals
                            start = int(range_parts[0].strip())
                            end = int(range_parts[1].strip())
                        # Handle negative steps correctly
                        if step < 0:
                            # For negative steps, range goes from start down to end (inclusive)
                            values = list(range(start, end - 1, step))
                        else:
                            # For positive steps, range goes from start up to end (inclusive)
                            values = list(range(start, end + 1, step))
                    elif range_expr.startswith('{') and range_expr.endswith('}'):
                        # Set like "{1, 2, 3}"
                        values_str = range_expr[1:-1].split(',')
                        values = []
                        for v in values_str:
                            v_clean = v.strip()
                            if v_clean.startswith('"') and v_clean.endswith('"'):
                                values.append(v_clean[1:-1])
                            elif v_clean.startswith("'") and v_clean.endswith("'"):
                                values.append(v_clean[1:-1])
                            else:
                                try:
                                    values.append(float(v_clean))
                                except ValueError:
                                    values.append(v_clean)
                    else:
                        raise SyntaxError(
                            f"Invalid range expression: {range_expr} at line {line_number}")

                    # Handle consecutive FORs as zipped pairs
                    if i + 1 < len(lines):
                        next_line, next_line_number = lines[i + 1]
                        if next_line.strip().lower().startswith('for '):
                            m2 = re.match(
                                r'^\s*FOR\s+([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', next_line, re.I)
                            if m2:
                                var2, range2, step2, _ = m2.groups()
                                step2 = int(step2) if step2 else 1
                                # Parse range for second FOR
                                if ' to ' in range2:
                                    s2, e2 = range2.split(' to ')
                                    vals2 = list(
                                        range(int(s2.strip()), int(e2.strip()) + 1, step2))
                                elif range2.startswith('{') and range2.endswith('}'):
                                    vals2 = [float(v.strip())
                                             for v in range2[1:-1].split(',')]
                                else:
                                    vals2 = []

                                # Look for the next executable line (assignment or .push() call)
                                next_executable_line = None
                                next_executable_line_number = None
                                for j in range(i + 2, len(lines)):
                                    candidate_line, candidate_line_number = lines[j]
                                    if ':=' in candidate_line or '.push(' in candidate_line.lower():
                                        next_executable_line = candidate_line
                                        next_executable_line_number = candidate_line_number
                                        break

                                if next_executable_line:
                                    for a, b in zip(values, vals2):
                                        self.push_scope(
                                            is_private=True, is_loop_scope=True)
                                        self.current_scope().define(var_name, a, 'number')
                                        self.current_scope().define(var2, b, 'number')
                                        if ':=' in next_executable_line:
                                            self.array_handler.evaluate_line_with_assignment(
                                                next_executable_line, next_executable_line_number, self.current_scope().get_evaluation_scope())
                                        elif '.push(' in next_executable_line.lower():
                                            # Process the .push() call directly using the existing logic
                                            self._process_push_call(
                                                next_executable_line, next_executable_line_number)
                                        self.pop_scope()
                                    # Skip both FOR lines and the executable line
                                    i = j + 1
                                    continue

                    if is_block:
                        # Process FOR block
                        for_block_lines = []
                        i += 1
                        depth = 1
                        while i < len(lines) and depth > 0:
                            next_line, next_line_number = lines[i]
                            next_line_clean = next_line.strip().lower()
                            if next_line_clean.lower() == "end":
                                depth -= 1
                                if depth == 0:
                                    i += 1
                                    break
                            elif next_line_clean.startswith("for ") and " do" in next_line_clean.lower():
                                depth += 1
                            elif next_line_clean.startswith("let ") and " then " in next_line_clean:
                                depth += 1
                            elif next_line_clean.startswith("if ") and " then" in next_line_clean.lower():
                                depth += 1
                            for_block_lines.append(
                                (next_line, next_line_number))
                            i += 1

                        if depth > 0:
                            raise SyntaxError(
                                f"Unclosed FOR block starting at line {line_number}")

                        # Execute the block for each value
                        for idx, value in enumerate(values):
                            self.push_scope(is_private=True,
                                            is_loop_scope=True)
                            self.current_scope().define(var_name, value, 'number')
                            if index_var:
                                # If index_var already exists in the current scope, update it instead of redefining
                                if index_var in self.current_scope().variables:
                                    self.current_scope().update(index_var, idx + 1)
                                else:
                                    self.current_scope().define(index_var, idx + 1, 'number')
                            self.control_flow._process_block(for_block_lines)

                            # Check if exit loop was requested immediately after processing block
                            if self.exit_loop:
                                self.exit_loop = False
                                self.pop_scope()
                                break

                            for_block_scope = self.current_scope()
                            self.pop_scope()

                            # Transfer updated variables from the FOR block scope to the parent scope, but skip the loop and index variables
                            for var, val in for_block_scope.variables.items():
                                if var not in (var_name, index_var):
                                    self.current_scope().variables[var] = val
                        # Skip the FOR statement in the main loop
                        continue
                    else:
                        # Single line FOR
                        # Look for the next executable line (assignment or .push() call)
                        next_executable_line = None
                        next_executable_line_number = None
                        skip_lines = 0
                        for j in range(i + 1, len(lines)):
                            candidate_line, candidate_line_number = lines[j]
                            if ':=' in candidate_line or '.push(' in candidate_line.lower():
                                next_executable_line = candidate_line
                                next_executable_line_number = candidate_line_number
                                skip_lines = j - i
                                break

                        if next_executable_line:
                            if ':=' in next_executable_line:
                                # Execute for each value
                                for idx, value in enumerate(values):
                                    self.push_scope(
                                        is_private=True, is_loop_scope=True)
                                    self.current_scope().define(var_name, value, 'number')
                                    if index_var:
                                        # If index_var already exists in the current scope, update it instead of redefining
                                        if index_var in self.current_scope().variables:
                                            self.current_scope().update(index_var, idx + 1)
                                        else:
                                            self.current_scope().define(index_var, idx + 1, 'number')
                                    self.array_handler.evaluate_line_with_assignment(
                                        next_executable_line, next_executable_line_number, self.current_scope().get_evaluation_scope())
                                    self.pop_scope()
                                # Skip the FOR statement and all lines up to the executable line
                                i += skip_lines + 1
                                continue
                            elif '.push(' in next_executable_line.lower():
                                # Execute for each value
                                for idx, value in enumerate(values):
                                    self.push_scope(
                                        is_private=True, is_loop_scope=True)
                                    self.current_scope().define(var_name, value, 'number')
                                    if index_var:
                                        # If index_var already exists in the current scope, update it instead of redefining
                                        if index_var in self.current_scope().variables:
                                            self.current_scope().update(index_var, idx + 1)
                                        else:
                                            self.current_scope().define(index_var, idx + 1, 'number')
                                    # Process the .push() call directly using the existing logic
                                    self._process_push_call(
                                        next_executable_line, next_executable_line_number)
                                    self.pop_scope()
                                # Skip the FOR statement and all lines up to the executable line
                                i += skip_lines + 1
                                continue
                        i += 1
                        continue

                # Handle nested FOR loops with AND
                elif ' and ' in var_defs.lower() and ' in ' in var_defs.lower():
                    # Parse multiple FOR loops like "a in {1, 2} AND b in 9 to 15 step 3"
                    var_parts = re.split(r'\s+and\s+', var_defs, flags=re.I)
                    loop_configs = []

                    for var_part in var_parts:
                        range_match = re.match(
                            r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(\d+))?(?:\s+index\s+([\w_]+))?$', var_part.strip(), re.I)
                        if range_match:
                            var_name, range_expr, step_str, index_var = range_match.groups()
                            step = int(step_str) if step_str else 1

                            # Parse the range expression
                            if ' to ' in range_expr:
                                range_parts = range_expr.split(' to ')
                                start_expr = range_parts[0].strip()
                                end_expr = range_parts[1].strip()

                                # Check if this is a dynamic range first
                                is_dynamic = False
                                # Check if expressions contain variables (not just numbers)
                                if not start_expr.replace('.', '').replace('-', '').isdigit() or not end_expr.replace('.', '').replace('-', '').replace('+', '').replace('*', '').replace('/', '').replace('(', '').replace(')', '').isdigit():
                                    is_dynamic = True
                                    values = None  # Will be computed during execution
                                else:
                                    # Static range, compute now
                                    start = int(start_expr)
                                    end = int(end_expr)
                                    values = list(range(start, end + 1, step))
                            elif range_expr.startswith('{') and range_expr.endswith('}'):
                                values_str = range_expr[1:-1].split(',')
                                values = []
                                for v in values_str:
                                    v_clean = v.strip()
                                    if v_clean.startswith('"') and v_clean.endswith('"'):
                                        values.append(v_clean[1:-1])
                                    elif v_clean.startswith("'") and v_clean.endswith("'"):
                                        values.append(v_clean[1:-1])
                                    else:
                                        try:
                                            values.append(float(v_clean))
                                        except ValueError:
                                            values.append(v_clean)
                            else:
                                raise SyntaxError(
                                    f"Invalid range expression: {range_expr} at line {line_number}")

                            loop_configs.append((var_name, values, index_var, is_dynamic,
                                                start_expr if is_dynamic else None, end_expr if is_dynamic else None, step))
                        else:
                            raise SyntaxError(
                                f"Invalid FOR loop syntax: {var_part} at line {line_number}")

                    if is_block:
                        # Process FOR block
                        for_block_lines = []
                        i += 1
                        depth = 1
                        while i < len(lines) and depth > 0:
                            next_line, next_line_number = lines[i]
                            next_line_clean = next_line.strip().lower()
                            if self._is_keyword(next_line_clean, "end"):
                                depth -= 1
                                # Include 'end' if it closes a nested block, not the FOR itself
                                if depth > 0:
                                    for_block_lines.append(
                                        (next_line, next_line_number))
                                    i += 1
                                    continue
                                else:
                                    i += 1
                                    break
                            elif next_line_clean.startswith("for ") and " do" in next_line_clean.lower():
                                depth += 1
                            # Correctly account for nested IF blocks inside FOR blocks
                            elif next_line_clean.startswith("if ") and next_line_clean.endswith("then"):
                                depth += 1
                            elif next_line_clean.startswith("for ") and " do" in next_line_clean.lower():
                                depth += 1

                            for_block_lines.append(
                                (next_line, next_line_number))
                            i += 1

                        if depth > 0:
                            raise SyntaxError(
                                f"Unclosed FOR block starting at line {line_number}")

                        # Generate all combinations
                        from itertools import product

                        # For dynamic ranges, we need to compute values during execution
                        def generate_combinations(loop_idx=0, current_combo=[]):
                            if loop_idx >= len(loop_configs):
                                yield current_combo
                                return

                            var_name, values, index_var, is_dynamic, start_expr, end_expr, step = loop_configs[
                                loop_idx]

                            if is_dynamic:
                                # Compute dynamic range for this iteration
                                # We need to evaluate the expressions with current variable values
                                self.push_scope(
                                    is_private=True, is_loop_scope=True)
                                # Set up variables from previous loops
                                for i, prev_config in enumerate(loop_configs[:loop_idx]):
                                    prev_var_name = prev_config[0]
                                    self.current_scope().define(
                                        prev_var_name, current_combo[i], 'number')

                                # Evaluate the dynamic range
                                start = int(self.expr_evaluator.eval_expr(
                                    start_expr, self.current_scope().get_evaluation_scope(), line_number))
                                end = int(self.expr_evaluator.eval_expr(
                                    end_expr, self.current_scope().get_evaluation_scope(), line_number))
                                dynamic_values = list(
                                    range(start, end + 1, step))
                                self.pop_scope()

                                for val in dynamic_values:
                                    yield from generate_combinations(loop_idx + 1, current_combo + [val])
                            else:
                                # Static range
                                for val in values:
                                    yield from generate_combinations(loop_idx + 1, current_combo + [val])

                        value_combinations = list(generate_combinations())

                        # Execute the block for each combination
                        for combo_idx, combo in enumerate(value_combinations):
                            # Check if exit loop was requested
                            if self.exit_loop:
                                self.exit_loop = False
                                break

                            self.push_scope(is_private=True,
                                            is_loop_scope=True)
                            for var_idx, (var_name, _, index_var, _, _, _, _) in enumerate(loop_configs):
                                self.current_scope().define(
                                    var_name, combo[var_idx], 'number')
                                if index_var:
                                    # If index_var already exists in the current scope, update it instead of redefining
                                    if index_var in self.current_scope().variables:
                                        self.current_scope().update(index_var, var_idx + 1)
                                    else:
                                        self.current_scope().define(index_var, var_idx + 1, 'number')
                            self.control_flow._process_block(for_block_lines)

                           # Check if exit loop was requested immediately after processing block
                            if self.exit_loop:
                                self.exit_loop = False
                                self.pop_scope()
                                break

                            self.pop_scope()
                    else:
                        # Single line nested FOR
                        if i + 1 < len(lines):
                            next_line, next_line_number = lines[i + 1]
                            if ':=' in next_line:
                                # Generate all combinations
                                from itertools import product

                                # For dynamic ranges, we need to compute values during execution
                                def generate_combinations(loop_idx=0, current_combo=[]):
                                    if loop_idx >= len(loop_configs):
                                        yield current_combo
                                        return

                                    var_name, values, index_var, is_dynamic, start_expr, end_expr, step = loop_configs[
                                        loop_idx]

                                    if is_dynamic:
                                        # Compute dynamic range for this iteration
                                        # We need to evaluate the expressions with current variable values
                                        self.push_scope(
                                            is_private=True, is_loop_scope=True)
                                        # Set up variables from previous loops
                                        for i, prev_config in enumerate(loop_configs[:loop_idx]):
                                            prev_var_name = prev_config[0]
                                            self.current_scope().define(
                                                prev_var_name, current_combo[i], 'number')

                                        # Evaluate the dynamic range
                                        start = int(self.expr_evaluator.eval_expr(
                                            start_expr, self.current_scope().get_evaluation_scope(), line_number))
                                        end = int(self.expr_evaluator.eval_expr(
                                            end_expr, self.current_scope().get_evaluation_scope(), line_number))
                                        dynamic_values = list(
                                            range(start, end + 1, step))
                                        self.pop_scope()

                                        for val in dynamic_values:
                                            yield from generate_combinations(loop_idx + 1, current_combo + [val])
                                    else:
                                        # Static range
                                        for val in values:
                                            yield from generate_combinations(loop_idx + 1, current_combo + [val])

                                value_combinations = list(
                                    generate_combinations())

                                # Execute for each combination
                                for combo_idx, combo in enumerate(value_combinations):
                                    self.push_scope(
                                        is_private=True, is_loop_scope=True)
                                    for var_idx, (var_name, _, index_var, _, _, _, _) in enumerate(loop_configs):
                                        self.current_scope().define(
                                            var_name, combo[var_idx], 'number')
                                        if index_var:
                                            # If index_var already exists in the current scope, update it instead of redefining
                                            if index_var in self.current_scope().variables:
                                                self.current_scope().update(index_var, var_idx + 1)
                                            else:
                                                self.current_scope().define(index_var, var_idx + 1, 'number')
                                    self.array_handler.evaluate_line_with_assignment(
                                        next_line, next_line_number, self.current_scope().get_evaluation_scope())
                                    self.pop_scope()
                        i += 1
                        continue

                var_list = []
                try:
                    if ' and ' in var_defs.lower():
                        var_parts = re.split(
                            r'\s+and\s+', var_defs, flags=re.I)
                        for var_part in var_parts:
                            var, type_name, constraints, value = self._parse_variable_def(
                                var_part, line_number)
                            var_list.append(
                                (var, type_name, constraints, value))
                    else:
                        var, type_name, constraints, value = self._parse_variable_def(
                            var_defs, line_number)
                        var_list.append((var, type_name, constraints, value))

                        # Handle comma-separated variable lists (e.g., "low, high as number")
                        if 'var_list' in constraints:
                            var_names = constraints['var_list']
                            # Clear the original entry and add each variable separately
                            var_list.clear()
                            for var_name in var_names:
                                var_list.append(
                                    (var_name, type_name, constraints.copy(), value))

                except Exception as e:
                    raise SyntaxError(
                        f"Invalid FOR variable definition at line {line_number}")

                # Handle dimension-based FOR (e.g., For names dim 0 to 4)
                if 'dim' in var_defs.lower():
                    m_dim = re.match(
                        r'^([\w_]+)\s+dim\s+(\d+)\s+to\s+(\d+)$', var_defs, re.I)
                    if m_dim:
                        var_name, start, end = m_dim.groups()
                        start, end = int(start), int(end)
                        constraints = {'dim': [(None, (start, end))]}
                        self.current_scope().define(var_name, None, 'array',
                                                    constraints, is_uninitialized=True)
                        self.dimensions[var_name] = [(None, (start, end))]
                        i += 1
                        continue

                # Handle custom type with 'with' clause (e.g., For V as tensor with (...))
                if var_list[0][1] and 'with' in var_list[0][2]:
                    var, type_name, constraints, value = var_list[0]
                    if type_name.lower() in self.types_defined:
                        with_constraints = constraints.get('with', {})
                        dims = constraints.get('dim', [])
                        if dims:
                            shape = [size_spec for _, size_spec in dims]
                            struct_fields = [(f.lower(), pa.string() if self.types_defined[type_name.lower()].get(f.lower()) == 'text' else pa.float64())
                                             for f in self.types_defined[type_name.lower()]]
                            struct_fields.append(('value', pa.float64()))
                            pa_type = pa.struct(struct_fields)
                            default_struct = {f.lower(): with_constraints.get(
                                f) for f in self.types_defined[type_name.lower()]}
                            default_struct['value'] = float(
                                value) if value else 1.0
                            array = self.array_handler.create_array(
                                shape, default_struct, pa_type, line_number)
                            self.current_scope().define(var, array, 'array', constraints, is_uninitialized=False)
                            for field in self.types_defined[type_name.lower()]:
                                if field in with_constraints:
                                    self.current_scope().define(
                                        f"{var}.{field}", with_constraints[field], 'text')
                        i += 1
                        continue  # Skip further FOR processing to avoid overwriting V

                # Handle simple FOR assignments (e.g., For x = 34)
                if not is_block and len(var_list) == 1 and var_list[0][3] is not None:
                    var, type_name, constraints, value = var_list[0]
                    evaluated_value = self.expr_evaluator.eval_expr(
                        str(value), self.current_scope().get_evaluation_scope(), line_number)
                    inferred_type = type_name or self.array_handler.infer_type(
                        evaluated_value, line_number)
                    if inferred_type == 'int':
                        inferred_type = 'number'
                    defining_scope = self.current_scope().get_defining_scope(var)
                    if defining_scope:
                        if defining_scope.types.get(var) == 'number' and not isinstance(evaluated_value, (int, float)):
                            raise TypeError(
                                f"Cannot assign non-numeric value {evaluated_value} to '{var}' at line {line_number}")
                        defining_scope.update(var, float(evaluated_value) if isinstance(
                            evaluated_value, (int, float)) else evaluated_value, line_number)
                    else:
                        # Store constraint expression if it exists
                        if value and str(value) != str(evaluated_value):
                            constraints['constant'] = str(value)
                        self.get_global_scope().define(var, evaluated_value,
                                                       inferred_type, constraints, False)
                    pending = list(
                        self.current_scope().pending_assignments.items())
                    for key, (expr, ln, deps) in pending:
                        unresolved = any(self.current_scope().is_uninitialized(
                            dep) or dep in self.pending_assignments for dep in deps)
                        if not unresolved:
                            try:
                                self.array_handler.evaluate_line_with_assignment(
                                    expr, ln, self.current_scope().get_evaluation_scope())
                                del self.current_scope(
                                ).pending_assignments[key]
                            except Exception as e:
                                pass
                    i += 1
                    continue

                # Handle multiple FOR assignments (e.g., For a = 20 and b = 4)
                if not is_block and len(var_list) > 1:
                    # Check if all variables have assignments
                    all_have_assignments = all(
                        v[3] is not None for v in var_list)

                    if all_have_assignments:
                        # Handle assignments
                        for var, type_name, constraints, value in var_list:
                            evaluated_value = self.expr_evaluator.eval_expr(
                                str(value), self.current_scope().get_evaluation_scope(), line_number)
                            inferred_type = type_name or self.array_handler.infer_type(
                                evaluated_value, line_number)
                            if inferred_type == 'int':
                                inferred_type = 'number'
                            defining_scope = self.current_scope().get_defining_scope(var)
                            if defining_scope:
                                if defining_scope.types.get(var) == 'number' and not isinstance(evaluated_value, (int, float)):
                                    raise TypeError(
                                        f"Cannot assign non-numeric value {evaluated_value} to '{var}' at line {line_number}")
                                defining_scope.update(var, float(evaluated_value) if isinstance(
                                    evaluated_value, (int, float)) else evaluated_value, line_number)
                            else:
                                self.get_global_scope().define(var, evaluated_value,
                                                               inferred_type, constraints, False)
                        pending = list(
                            self.current_scope().pending_assignments.items())
                        for key, (expr, ln, deps) in pending:
                            unresolved = any(self.current_scope().is_uninitialized(
                                dep) or dep in self.pending_assignments for dep in deps)
                            if not unresolved:
                                try:
                                    self.array_handler.evaluate_line_with_assignment(
                                        expr, ln, self.current_scope().get_evaluation_scope())
                                    del self.current_scope(
                                    ).pending_assignments[key]
                                except Exception as e:
                                    pass
                        i += 1
                        continue
                    else:
                        # Handle declarations without assignments (e.g., "for low, high as number")
                        for var, type_name, constraints, _ in var_list:
                            # Initialize with reasonable defaults based on variable type
                            if type_name and type_name.lower() == 'number':
                                try:
                                    array_var = None
                                    for var_name in self.current_scope().variables:
                                        var_value = self.current_scope().get(var_name)
                                        if var_value and hasattr(var_value, '__len__') and not isinstance(var_value, str):
                                            array_var = var_value
                                            break
                                    if array_var:
                                        default_value = len(array_var)
                                    else:
                                        default_value = 1
                                except:
                                    default_value = 1
                            elif type_name and type_name.lower() == 'text':
                                default_value = ""
                            else:
                                default_value = 0
                            self.current_scope().define(var, default_value, type_name,
                                                        constraints, is_uninitialized=False)
                        i += 1
                        continue

                # Handle FOR blocks or LET followers
                for var, _, _, _ in var_list:
                    defining_scope = self.current_scope().get_defining_scope(var)
                    if defining_scope and (is_block or not (i + 1 < len(lines) and lines[i + 1][0].lower().startswith('let '))):
                        raise ValueError(
                            f"Variable '{var}' already defined in scope at line {line_number}")

                # For comma-separated variables without assignments, define them in current scope
                if not is_block and len(var_list) > 1 and all(v[3] is None for v in var_list):
                    for var, type_name, constraints, _ in var_list:
                        self.current_scope().define(var, None, type_name, constraints, is_uninitialized=True)

                self.push_scope(is_private=True, is_loop_scope=True)
                for var, type_name, constraints, value in var_list:
                    if value is not None:
                        try:
                            evaluated_value = self.expr_evaluator.eval_expr(
                                str(value), self.current_scope().get_evaluation_scope(), line_number)
                            inferred_type = type_name or self.array_handler.infer_type(
                                evaluated_value, line_number)
                            if inferred_type == 'int':
                                inferred_type = 'number'
                            self.current_scope().define(var, evaluated_value, inferred_type, constraints)
                            self.current_scope().update(var, evaluated_value, line_number)
                        except NameError as e:
                            self.get_global_scope().define(var, None, type_name,
                                                           constraints, is_uninitialized=True)
                    else:
                        self.get_global_scope().define(var, None, type_name,
                                                       constraints, is_uninitialized=True)

                if is_block:
                    for_block_lines = []
                    i += 1
                    depth = 1
                    while i < len(lines) and depth > 0:
                        next_line, next_line_number = lines[i]
                        next_line_clean = next_line.strip().lower()
                        if next_line_clean == "end":
                            depth -= 1
                            if depth == 0:
                                i += 1
                                break
                        elif next_line_clean.startswith("for ") and next_line_clean.endswith("do"):
                            depth += 1
                        elif next_line_clean.startswith("let ") and " then " in next_line_clean:
                            depth += 1
                        for_block_lines.append((next_line, next_line_number))
                        i += 1
                    if depth > 0:
                        raise SyntaxError(
                            f"Unclosed FOR block starting at line {line_number}")
                    for var, _, constraints, _ in var_list:
                        if constraints:
                            try:
                                var_value = self.current_scope().get(var)
                                if var_value is not None:
                                    self.current_scope()._check_constraints(var, var_value, line_number)
                            except ValueError as e:
                                for_block_lines = []
                                break
                    self.control_flow._process_block(for_block_lines)
                    self.pop_scope()
                else:
                    if len(var_list) == 1 and i + 1 < len(lines) and lines[i + 1][0].lower().startswith('let '):
                        for_block_lines = []
                        i += 1
                        for_block_lines.append(lines[i])
                        if i + 1 < len(lines) and not lines[i + 1][0].strip().lower() == "end":
                            for_block_lines.append(lines[i + 1])
                            i += 1
                        for var, _, constraints, _ in var_list:
                            if constraints:
                                try:
                                    var_value = self.current_scope().get(var)
                                    if var_value is not None:
                                        self.current_scope()._check_constraints(var, var_value, line_number)
                                except ValueError as e:
                                    for_block_lines = []
                                    break
                        self.control_flow._process_block(for_block_lines)
                        self.pop_scope()
                    else:
                        self.pop_scope()
                i += 1
            elif line.lower().startswith("if "):
                # Check if this is a complex IF statement (has ELSEIF/ELSE clauses)
                if_block = None
                for block in self.control_flow.if_blocks:
                    if block['start_line'] == line_number:
                        if_block = block
                        break

                if if_block and len(if_block.get('clauses', [])) > 1:
                    # Complex IF statement with ELSEIF/ELSE clauses
                    skip_lines = self.control_flow._process_if_statement_rich(
                        line, line_number, lines, i)
                else:
                    # Simple IF statement
                    skip_lines = self.control_flow._process_if_statement(
                        line, line_number, lines, i)
                i += skip_lines
                continue
            elif line.lower().startswith("let "):
                m = re.match(
                    r'^\s*let\s+(.+?)(?:\s+then\s*$|\s*$)', line, re.I)
                if not m:
                    raise SyntaxError(
                        f"Invalid LET syntax at line {line_number}")
                var_def = m.group(1).strip()
                has_block = line.lower().strip().endswith('then')
                var_list = []
                if ' and ' in var_def.lower():
                    var_parts = re.split(r'\s+and\s+', var_def, flags=re.I)
                    for var_part in var_parts:
                        var, type_name, constraints, expr = self._parse_variable_def(
                            var_part, line_number)
                        var_list.append((var, type_name, constraints, expr))
                else:
                    var, type_name, constraints, expr = self._parse_variable_def(
                        var_def, line_number)
                    var_list.append((var, type_name, constraints, expr))

                # First pass: Evaluate LET assignments, fail on NameError
                scope_dict = self.current_scope().get_evaluation_scope()
                unresolved_vars = []
                for var, type_name, constraints, expr in var_list:
                    # Handle array indexing assignment (e.g., D{i+1, 1}, D[A1], or D(k+1))
                    array_index_match = re.match(r'^([\w_]+)\{([^}]+)\}$', var)
                    cell_index_match = re.match(r'^([\w_]+)\[([^\]]+)\]$', var)
                    paren_index_match = re.match(r'^([\w_]+)\(([^)]+)\)$', var)
                    if array_index_match or cell_index_match or paren_index_match:
                        if array_index_match:
                            var_name, indices_str = array_index_match.groups()
                            indices = []
                            for index_expr in indices_str.split(','):
                                index_expr = index_expr.strip()
                                index_value = self.expr_evaluator.eval_expr(
                                    index_expr, scope_dict, line_number)
                                if isinstance(index_value, float) and index_value.is_integer():
                                    index_value = int(index_value)
                                indices.append(
                                    index_value - 1 if isinstance(index_value, int) else index_value)
                        elif cell_index_match:
                            var_name, index_expr = cell_index_match.groups()
                            try:
                                index_value = self.expr_evaluator.eval_expr(
                                    index_expr, scope_dict, line_number)
                                if isinstance(index_value, float) and index_value.is_integer():
                                    index_value = int(index_value)
                                indices = [index_value - 1]
                            except:
                                indices = self.array_handler.cell_ref_to_indices(
                                    index_expr, line_number)
                        else:  # paren_index_match
                            var_name, index_expr = paren_index_match.groups()
                            index_value = self.expr_evaluator.eval_expr(
                                index_expr, scope_dict, line_number)
                            if isinstance(index_value, float) and index_value.is_integer():
                                index_value = int(index_value)
                            indices = [index_value - 1]
                        value = self.expr_evaluator.eval_or_eval_array(
                            expr, scope_dict, line_number)
                        defining_scope = self.current_scope().get_defining_scope(var_name)
                        if defining_scope:
                            actual_key = defining_scope._get_case_insensitive_key(
                                var_name, defining_scope.variables)
                            if actual_key:
                                arr = defining_scope.variables[actual_key]
                                if isinstance(arr, dict) and 'array' in arr:
                                    updated_array = self.array_handler.set_array_element(
                                        arr['array'], indices, value, line_number)
                                    arr['array'] = updated_array
                                    defining_scope.variables[actual_key] = arr
                                    scope_dict[actual_key] = arr
                                else:
                                    updated_array = self.array_handler.set_array_element(
                                        arr, indices, value, line_number)
                                    defining_scope.variables[actual_key] = updated_array
                                    scope_dict[actual_key] = updated_array
                            else:
                                raise NameError(
                                    f"Array variable '{var_name}' not defined at line {line_number}")
                        else:
                            raise NameError(
                                f"Array variable '{var_name}' not defined at line {line_number}")
                        continue
                    defining_scope = self.current_scope().get_defining_scope(var)
                    if defining_scope:
                        if var in defining_scope.constraints and not constraints:
                            constraints = defining_scope.constraints[var]
                        if constraints:
                            defining_scope.constraints[var] = constraints
                        if expr is None and var in defining_scope.variables and defining_scope.variables[var] is not None:
                            continue
                    else:
                        if self.current_scope().is_shadowed(var):
                            print(
                                f"Warning: LET defines '{var}' which shadows a variable in an outer scope at line {line_number}")
                        self.current_scope().define(var, None, type_name, constraints, is_uninitialized=True)
                    if expr is not None:
                        try:
                            evaluated_value = self.expr_evaluator.eval_or_eval_array(
                                expr, scope_dict, line_number)
                            self.current_scope().update(var, evaluated_value, line_number)
                            scope_dict[var] = evaluated_value
                        except NameError as e:
                            unresolved_vars.append(var)

                # Fail if any variables had unresolved dependencies
                if unresolved_vars:
                    raise NameError(
                        f"Undefined variables in LET statement: {unresolved_vars} at line {line_number}")

                # Second pass: Only for constraints or uninitialized vars without expr
                for var, type_name, constraints, expr in var_list:
                    if expr is None and self.current_scope().is_uninitialized(var):
                        try:
                            var_value = self.current_scope().get(var)
                            if var_value is not None:
                                self.current_scope()._check_constraints(var, var_value, line_number)
                        except Exception as e:
                            pass

                if has_block:
                    self.push_scope(is_private=True)
                    block_lines = []
                    i += 1
                    depth = 1
                    while i < len(lines) and depth > 0:
                        next_line, next_line_number = lines[i]
                        next_line_clean = next_line.strip().lower()
                        if next_line_clean == "end":
                            depth -= 1
                            if depth == 0:
                                i += 1
                                break
                        elif next_line_clean.startswith("for ") and next_line_clean.endswith("do"):
                            depth += 1
                        elif next_line_clean.startswith("let ") and " then " in next_line_clean:
                            depth += 1
                        block_lines.append((next_line, next_line_number))
                        i += 1
                    if depth > 0:
                        raise SyntaxError(
                            f"Unclosed LET block starting at line {line_number}")
                    for var, _, constraints, _ in var_list:
                        if constraints:
                            defining_scope = self.current_scope().get_defining_scope(var) or self.current_scope()
                            if var in self.pending_assignments and defining_scope.get(var) is None:
                                expr, ln, deps = self.pending_assignments[var]
                                unresolved = any(self.current_scope().is_uninitialized(dep) or (
                                    dep in self.pending_assignments and dep != var) for dep in deps)
                                if not unresolved:
                                    try:
                                        value = self.expr_evaluator.eval_or_eval_array(
                                            expr, defining_scope.get_full_scope(), ln)
                                        defining_scope.update(var, value, ln)
                                        del self.pending_assignments[var]
                                    except Exception as e:
                                        pass
                            var_value = defining_scope.get(var)
                            if var_value is None:
                                self.grid.clear()
                                self.pop_scope()
                                return self.grid
                            for op, threshold in constraints.items() if constraints else []:
                                try:
                                    threshold_value = float(threshold)
                                    if isinstance(var_value, (int, float)):
                                        if op == '<' and var_value >= threshold_value:
                                            self.grid.clear()
                                            self.pop_scope()
                                            return self.grid
                                        elif op == '>' and var_value <= threshold_value:
                                            self.grid.clear()
                                            self.pop_scope()
                                            return self.grid
                                    else:
                                        pass
                                except ValueError:
                                    pass
                            if defining_scope.constraints.get(var, {}):
                                try:
                                    defining_scope._check_constraints(
                                        var, var_value, line_number)
                                except ValueError as e:
                                    self.grid.clear()
                                    self.pop_scope()
                                    return self.grid
                    block_pending = self.control_flow._process_block(
                        block_lines)
                    if block_pending:
                        self.scopes[0].pending_assignments.update(
                            block_pending)
                    self.pop_scope()
                else:
                    for var, _, constraints, _ in var_list:
                        if constraints:
                            defining_scope = self.current_scope().get_defining_scope(var) or self.current_scope()
                            if var in self.pending_assignments and defining_scope.get(var) is None:
                                expr, ln, deps = self.pending_assignments[var]
                                unresolved = any(self.current_scope().is_uninitialized(dep) or (
                                    dep in self.pending_assignments and dep != var) for dep in deps)
                                if not unresolved:
                                    try:
                                        value = self.expr_evaluator.eval_or_eval_array(
                                            expr, defining_scope.get_full_scope(), ln)
                                        defining_scope.update(var, value, ln)
                                        del self.pending_assignments[var]
                                    except Exception as e:
                                        pass
                            var_value = defining_scope.get(var)
                            if var_value is None:
                                j = i + 1
                                while j < len(lines):
                                    next_line, next_line_number = lines[j]
                                    if ':=' in next_line:
                                        target, rhs = next_line.split(':=')
                                        rhs_vars = set(re.findall(
                                            r'\b[\w_]+\b(?=\s*(?:[\[\{]|$))', rhs.strip()))
                                        if var in rhs_vars:
                                            lines[j] = ("", next_line_number)
                                    j += 1
                                i += 1
                                continue
                            for op, threshold in constraints.items() if constraints else []:
                                try:
                                    threshold_value = float(threshold)
                                    if isinstance(var_value, (int, float)):
                                        if op == '<' and var_value >= threshold_value:
                                            self.pending_assignments.clear()
                                            i += 1
                                            continue
                                        elif op == '>' and var_value <= threshold_value:
                                            self.pending_assignments.clear()
                                            i += 1
                                            continue
                                    else:
                                        pass
                                except ValueError:
                                    pass
                            if defining_scope.constraints.get(var, {}):
                                try:
                                    defining_scope._check_constraints(
                                        var, var_value, line_number)
                                except ValueError as e:
                                    self.pending_assignments.clear()
                                    i += 1
                                    continue
                    i += 1
            elif line.lower().startswith('for '):
                m = re.match(r'^\s*FOR\s+(.+?)(?:\s+DO\s*$|\s*$)', line, re.I)
                if not m:
                    raise SyntaxError(
                        f"Invalid FOR syntax at line {line_number}")
                var_def = m.group(1).strip()
                has_block = line.lower().strip().endswith('do')
                var_list = []
                if ' and ' in var_def.lower():
                    var_parts = re.split(r'\s+and\s+', var_def, flags=re.I)
                    for var_part in var_parts:
                        var, type_name, constraints, expr = self._parse_variable_def(
                            var_part, line_number)
                        var_list.append((var, type_name, constraints, expr))
                else:
                    var, type_name, constraints, expr = self._parse_variable_def(
                        var_def, line_number)
                    var_list.append((var, type_name, constraints, expr))

                # Process variables
                for var, type_name, constraints, expr in var_list:
                    defining_scope = self.current_scope().get_defining_scope(var)
                    if defining_scope:
                        if constraints:
                            defining_scope.constraints[var] = constraints
                        if expr is None and var in defining_scope.variables and defining_scope.variables[var] is not None:
                            continue
                    else:
                        if self.current_scope().is_shadowed(var):
                            print(
                                f"Warning: FOR defines '{var}' which shadows a variable in an outer scope at line {line_number}")
                        self.current_scope().define(var, None, type_name, constraints, is_uninitialized=True)

                # Handle WITH clause and array initialization
                if var_list[0][1] and 'with' in var_list[0][2]:
                    var, type_name, constraints, value = var_list[0]
                    if type_name.lower() in self.types_defined:
                        with_constraints = constraints.get('with', {})
                        dims = constraints.get('dim', [])
                        if dims:
                            shape = []
                            for _, size_spec in dims:
                                if isinstance(size_spec, tuple):
                                    start, end = size_spec
                                    size = end - start + 1
                                else:
                                    size = size_spec
                                shape.append(size)
                            pa_type = pa.struct([(f, pa.string() if f == 'name' else pa.float64(
                            )) for f in self.types_defined[type_name.lower()]])
                            array = self.array_handler.create_array(
                                shape, None, pa_type, line_number)
                            for i in range(shape[0]):
                                for j in range(shape[1]):
                                    for k in range(shape[2]):
                                        obj = {f: with_constraints.get(
                                            f, None) for f in self.types_defined[type_name.lower()]}
                                        obj['value'] = float(
                                            value) if value else 1.0
                                        array = self.array_handler.set_array_element(
                                            array, [i, j, k], obj, line_number)
                            self.current_scope().define(var, array, 'array', constraints, is_uninitialized=False)
                            i += 1
                            continue

            elif line.startswith(":") and "=" in line and not line.lower().startswith(("for ", "let ")):
                var_def, expr = map(str.strip, line[1:].split("=", 1))
                var, type_name, constraints, value = self._parse_variable_def(
                    var_def, line_number)
                deps = set(re.findall(r'\b[\w_]+\b', expr))
                if not deps:
                    try:
                        evaluated_value = self.expr_evaluator.eval_expr(
                            expr, self.current_scope().get_evaluation_scope(), line_number)
                        inferred_type = type_name or self.array_handler.infer_type(
                            evaluated_value, line_number)
                        if inferred_type == 'int':
                            inferred_type = 'number'
                        self.current_scope().define(var, evaluated_value, inferred_type,
                                                    constraints, is_uninitialized=False)
                        if var in self.pending_assignments:
                            del self.pending_assignments[var]
                    except Exception as e:
                        self.pending_assignments[var] = (
                            expr, line_number, deps)
                else:
                    self.pending_assignments[var] = (expr, line_number, deps)
                i += 1
            else:
                if ':=' in line and '.grid' in line:
                    target, value = line.split(':=')
                    target = target.strip()
                    value = value.strip()
                    if re.match(r'^\[@?[A-Za-z]+\d+\]$', target):
                        cell_ref = target[1:-1].strip()
                        try:
                            if cell_ref.startswith('@'):
                                # Handle array reference - remove @ and validate the cell reference
                                array_cell_ref = cell_ref[1:].strip()
                                validate_cell_ref(array_cell_ref)
                                scope_value = self.current_scope().get_evaluation_scope()
                                evaluated_value = self.expr_evaluator.eval_or_eval_array(
                                    value, scope_value, line_number, is_grid_dim=True)

                                # Check if this is a grid dictionary that should spill values
                                if isinstance(evaluated_value, dict):
                                    # Check if it's a grid dictionary with coordinate tuples as keys
                                    if all(isinstance(k, tuple) and len(k) == 2 for k in evaluated_value.keys()):
                                        # This is a grid dictionary, spill the values onto the actual grid
                                        for (row, col), val in evaluated_value.items():
                                            if isinstance(row, (int, float)) and isinstance(col, (int, float)):
                                                # Convert to cell reference format
                                                grid_row = int(row)
                                                grid_col = int(col)
                                                col_letter = num_to_col(
                                                    grid_col)
                                                cell_ref = f"{col_letter}{grid_row}"
                                                self.grid[cell_ref] = val
                                        # For [@A1] syntax, don't spill sequentially
                                        # Just set the target to 0 since grid{1,1} is undefined
                                        self.grid[array_cell_ref] = 0
                                    elif 'grid' in evaluated_value:
                                        # This is a type instance with a grid field
                                        grid_dict = evaluated_value['grid']
                                        if isinstance(grid_dict, dict):
                                            # Extract grid values and spill them onto the actual grid
                                            for (row, col), val in grid_dict.items():
                                                if isinstance(row, int) and isinstance(col, int):
                                                    # Convert to cell reference format
                                                    col_letter = num_to_col(
                                                        col)
                                                    cell_ref = f"{col_letter}{row}"
                                                    self.grid[cell_ref] = val
                                            # For [@A1] syntax, don't spill sequentially
                                            # Just set the target to 0 since grid{1,1} is undefined
                                            self.grid[array_cell_ref] = 0
                                        else:
                                            # For [@A1] syntax, don't assign the entire value
                                            # Just set the target to 0 since it's not a grid
                                            self.grid[array_cell_ref] = 0
                                    else:
                                        # For [@A1] syntax, don't assign the entire value
                                        # Just set the target to 0 since it's not a grid
                                        self.grid[array_cell_ref] = 0
                                else:
                                    # For [@A1] syntax, don't assign the entire value
                                    # Just set the target to 0 since it's not a grid
                                    self.grid[array_cell_ref] = 0
                            else:
                                # Handle regular cell reference
                                validate_cell_ref(cell_ref)
                                scope_value = self.current_scope().get_evaluation_scope()
                                evaluated_value = self.expr_evaluator.eval_or_eval_array(
                                    value, scope_value, line_number, is_grid_dim=True)
                                self.grid[cell_ref] = evaluated_value
                        except Exception as e:
                            raise RuntimeError(
                                f"Error evaluating '{value}': {e} at line {line_number}")
                    else:
                        raise ValueError(
                            f"Invalid assignment target '{target}' at line {line_number}")
                    i += 1
                elif ':=' in line:
                    target, rhs = line.split(':=')
                    target, rhs = target.strip(), rhs.strip()
                    rhs_vars = set(re.findall(
                        r'\b[\w_]+\b(?=\s*(?:[\[\{]|!\w+\s*\(|(?:\.\w+)?\s*$))', rhs))
                    if '$"' in rhs:
                        placeholders = re.findall(r'\{\s*([^}]*?)\s*\}', rhs)
                        for ph in placeholders:
                            rhs_vars.update(re.findall(r'\b[\w_]+\b', ph))
                    field_vars = set(re.findall(
                        r'\b[\w_]+\b(?=\.\w+\s*$)', rhs))
                    rhs_vars.update(field_vars)
                    target_vars = set()
                    if '{' in target:
                        for match in re.finditer(r'\{([^}]+)\}', target):
                            expr = match.group(1).strip()
                            target_vars.update(re.findall(r'\b[\w_]+\b', expr))
                    unresolved = any(self.current_scope().is_uninitialized(
                        var) or var in self.pending_assignments for var in rhs_vars | target_vars)
                    if unresolved:
                        self.pending_assignments[f"__line_{line_number}"] = (
                            line, line_number, rhs_vars | target_vars)
                    else:
                        violations = []
                        for var in rhs_vars:
                            defining_scope = self.current_scope().get_defining_scope(var)
                            if defining_scope and var in defining_scope.constraints:
                                try:
                                    var_value = defining_scope.get(var)
                                    if var_value is not None:
                                        defining_scope._check_constraints(
                                            var, var_value, line_number)
                                    else:
                                        violations.append(var)
                                except ValueError as e:
                                    violations.append(var)
                        if not violations:
                            self.array_handler.evaluate_line_with_assignment(
                                line, line_number, self.current_scope().get_evaluation_scope())
                        else:
                            self.pending_assignments[f"__line_{line_number}"] = (
                                line, line_number, rhs_vars | target_vars)
                    i += 1
                elif re.match(r'^\[[A-Za-z]+\d+\]\s*:\s*\w+\s*=', line):
                    # Handle cell variable definition (e.g., [A1] : a = 51)
                    # Extract cell reference and variable definition
                    cell_match = re.match(
                        r'^\[([A-Za-z]+\d+)\]\s*:\s*(\w+)\s*=\s*(.+)$', line)
                    if cell_match:
                        cell_ref, var_name, expr = cell_match.groups()
                        cell_ref = cell_ref.strip()
                        var_name = var_name.strip()
                        expr = expr.strip()

                        # Validate cell reference
                        validate_cell_ref(cell_ref)

                        # Evaluate the expression
                        try:
                            scope_value = self.current_scope().get_evaluation_scope()
                            evaluated_value = self.expr_evaluator.eval_or_eval_array(
                                expr, scope_value, line_number)

                            # Assign to the cell
                            self.grid[cell_ref] = evaluated_value

                            # Define the variable in the current scope if it doesn't already exist
                            try:
                                self.current_scope().get(var_name)
                                self.current_scope().update(var_name, evaluated_value, line_number)
                            except NameError:
                                self.current_scope().define(var_name, evaluated_value,
                                                            'number', {}, is_uninitialized=False)

                        except Exception as e:
                            raise RuntimeError(
                                f"Error in cell variable definition: {e} at line {line_number}")
                    i += 1
                else:
                    i += 1

        self._resolve_pending_assignments()

        # Process any remaining deferred assignments
        self._process_deferred_assignments()

        # Print all output variables
        self._print_outputs()

        return self.grid

    def _process_push_call(self, line, line_number):
        """Process a .push() method call"""

        # Handle multi-line expressions by removing newlines and extra whitespace
        clean_line = line.replace('\n', ' ').replace('  ', ' ').strip()

        # Handle .push() method calls (e.g., var.push(value))
        m = re.match(
            r'^\s*([\w_]+)\.push\s*\(\s*(.+?)\s*\)\s*$', clean_line, re.I)
        if m:
            var_name, value_expr = m.groups()
            var_name = var_name.strip()
            value_expr = value_expr.strip()

            # Evaluate the expression to get the value
            try:
                # Use the global scope to ensure we can access updated variable values
                global_scope = self.get_global_scope()
                eval_scope = global_scope.get_evaluation_scope()

                # For simple variable expressions, get the value directly from global scope
                if value_expr.strip() in ['result', 'RESULT', 'Result']:
                    value = global_scope.variables.get('result')
                else:
                    value = self.expr_evaluator.eval_expr(
                        value_expr, eval_scope, line_number)

                # Update the variable with the new value
                try:
                    # Try to get the variable to see if it exists
                    self.current_scope().get(var_name)
                    # Variable exists, update it
                    self.current_scope().update(var_name, value, line_number)
                except NameError:
                    # Variable doesn't exist, define it
                    self.current_scope().define(
                        var_name, value, 'number', {}, is_uninitialized=False)

                # If this is an output variable, collect the value for final printing
                global_scope = self.get_global_scope()
                if var_name.lower() in global_scope.output_variables:
                    if var_name.lower() not in self.output_values:
                        self.output_values[var_name.lower()] = []
                    self.output_values[var_name.lower()].append(value)

            except Exception as e:
                raise ValueError(
                    f"Failed to evaluate .push() expression at line {line_number}: {e}")
        else:
            raise SyntaxError(f"Invalid .push() syntax at line {line_number}")

    def _print_outputs(self):
        """Print all output variables as required by Grid language"""
        # Check if we have a compiler reference with output variables
        if hasattr(self, 'compiler') and hasattr(self.compiler, 'output_variables'):
            # Only use compiler's output variables if we don't have any of our own
            if not self.output_variables:
                self.output_variables = self.compiler.output_variables

        # Get output values from self.output_values (the executor's output_values)
        output_values = self.output_values
        for output_var in self.output_variables:
            if output_var in output_values and output_values[output_var]:
                for value in output_values[output_var]:
                    print(f"{output_var}: {value}")

    def _process_deferred_assignments(self):
        """Process any deferred assignments stored with __line_ keys."""

        deferred_assignments = []
        for key, assignment in self.pending_assignments.items():
            if key.startswith('__line_'):
                line_content, line_number, deps = assignment[:3]
                deferred_assignments.append((line_number, line_content, deps))

        # Sort by line number to process in order
        deferred_assignments.sort(key=lambda x: x[0])

        for line_number, line_content, deps in deferred_assignments:
            try:
                # Check if all dependencies are now resolved
                unresolved = any(self.current_scope().is_uninitialized(dep)
                                 for dep in deps if dep != '__line_')
                if not unresolved:
                    # Process the assignment
                    self.array_handler.evaluate_line_with_assignment(
                        line_content, line_number, self.current_scope().get_evaluation_scope())
                    # Remove from pending assignments
                    if f"__line_{line_number}" in self.pending_assignments:
                        del self.pending_assignments[f"__line_{line_number}"]
                else:
                    pass
            except Exception as e:
                # Remove from pending assignments even on error to avoid infinite loops
                if f"__line_{line_number}" in self.pending_assignments:
                    del self.pending_assignments[f"__line_{line_number}"]
