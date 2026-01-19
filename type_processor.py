"""
Type processing functionality for GridLang compiler.
Handles type definitions, type code execution, and type-related operations.
"""

import re
import numbers


class GridLangTypeProcessor:
    """Handles type definitions and type-related processing."""

    def __init__(self, compiler=None):
        self.compiler = compiler

    def _parse_type_def(self, lines, line_number=None):
        """Parse type definition lines and extract fields and executable code."""
        fields = {}
        executable_code = []
        inputs = []
        hidden_fields = set()
        field_constraints = {}

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if lowered.startswith('input '):
                input_body = stripped[5:].strip()
                input_type = None
                input_default = None
                names_part = input_body
                m_as = re.search(r'\s+as\s+', input_body, re.I)
                if m_as:
                    names_part = input_body[:m_as.start()].strip()
                    remainder = input_body[m_as.end():].strip()
                    m_default = re.search(r'\bor\s*=\s*(.+)$', remainder, re.I)
                    if m_default:
                        input_type = remainder[:m_default.start()].strip() or None
                        input_default = m_default.group(1).strip()
                    else:
                        input_type = remainder.strip() or None
                else:
                    m_default = re.search(r'\bor\s*=\s*(.+)$', input_body, re.I)
                    if m_default:
                        names_part = input_body[:m_default.start()].strip()
                        input_default = m_default.group(1).strip()
                name_list = [n.strip() for n in names_part.split(',') if n.strip()]
                if not name_list:
                    raise SyntaxError(
                        f"Invalid input definition: '{line}' at line {line_number}")
                for in_name in name_list:
                    inputs.append({
                        'name': in_name.strip(),
                        'type': input_type.lower() if input_type else None,
                        'default': input_default
                    })
                continue

            field_line = None
            if stripped.startswith(':'):
                field_line = stripped[1:].strip()
            elif lowered.startswith('for ') and ' do' not in lowered:
                field_line = stripped[4:].strip()
            elif lowered.startswith('let ') and '=' not in stripped:
                field_line = stripped[4:].strip()

            if field_line:
                init_expr = None
                default_match = re.search(r'\bor\s*=\s*(.+)$', field_line, re.I)
                if default_match:
                    init_expr = default_match.group(1).strip()
                    field_line = field_line[:default_match.start()].strip()
                init_match = re.search(r'\binit\b', field_line, re.I)
                if init_match:
                    init_expr = field_line[init_match.end():].strip()
                    field_line = field_line[:init_match.start()].strip()

                m = re.match(
                    r'^(\$?[A-Za-z_][\w_]*(?:\s*,\s*\$?[A-Za-z_][\w_]*)*)', field_line)
                if not m:
                    raise SyntaxError(
                        f"Invalid field definition: '{line}' at line {line_number}")
                var_names = [v.strip() for v in m.group(1).split(',') if v.strip()]
                type_candidates = re.findall(
                    r'\bas\s+([A-Za-z_][\w_]*)', field_line, re.I)
                type_name = type_candidates[-1].lower(
                ) if type_candidates else None
                has_dim = re.search(r'\bdim\b', field_line, re.I)
                effective_type = 'array' if has_dim and not type_name else (type_name or 'unknown')
                parsed_constraints = {}
                parsed_type = type_name
                if self.compiler and hasattr(self.compiler, '_parse_variable_def'):
                    try:
                        parsed_var, parsed_type, parsed_constraints, _ = self.compiler._parse_variable_def(
                            field_line, line_number)
                        parsed_constraints = parsed_constraints or {}
                    except Exception:
                        parsed_constraints = {}
                if parsed_type and 'type' not in parsed_constraints and 'type_union' not in parsed_constraints:
                    parsed_constraints['type'] = parsed_type.lower()
                parsed_constraints.pop('var_list', None)
                for name in var_names:
                    clean_name = name[1:] if name.startswith('$') else name
                    if name.startswith('$'):
                        hidden_fields.add(clean_name.lower())
                    fields[clean_name] = effective_type
                    if parsed_constraints:
                        field_constraints[clean_name] = dict(parsed_constraints)
                    if init_expr:
                        executable_code.append(f"{clean_name} = {init_expr}")
                # Allow constructor-style assignments (e.g., ": x = in_x") to execute.
                if re.search(r'^\$?[A-Za-z_][\w_]*\s*=', field_line) and 'or =' not in lowered:
                    executable_code.append(field_line)
                continue

            # Executable code inside type definition (strip leading colon if present)
            executable_code.append(line.lstrip(':').strip())

        if executable_code:
            fields['_executable_code'] = executable_code
        if inputs:
            fields['_inputs'] = inputs
        if field_constraints:
            fields['_field_constraints'] = field_constraints
        if hidden_fields:
            fields['_hidden_fields'] = hidden_fields

        return fields

    def _execute_type_code(self, code_lines, var_name, value_dict, line_number, input_values=None):
        """Execute code that was defined inside a type definition"""

        # Create a temporary scope for execution
        self.compiler.push_scope()
        prev_hidden_access = getattr(
            self.compiler, '_allow_hidden_field_access', False)
        prev_hidden_member_calls = getattr(
            self.compiler, '_allow_hidden_member_calls', False)
        self.compiler._allow_hidden_field_access = True
        self.compiler._allow_hidden_member_calls = True

        # Add the type instance to the scope so code can reference it
        inferred_type = value_dict.get('_type_name') if isinstance(
            value_dict, dict) else None
        self.compiler.current_scope().define(
            var_name, value_dict, inferred_type or 'object')
        # Use lowercase "this"; case-insensitive lookup covers "This".
        if str(var_name).lower() != 'this':
            self.compiler.current_scope().define(
                'this', value_dict, inferred_type or 'object')
        input_values = input_values or {}
        for in_name, in_val in input_values.items():
            self.compiler.current_scope().define(
                in_name, in_val, None, {'input': True}, is_uninitialized=False)

        # Add a 'grid' field only if constructor code references it
        if 'grid' not in value_dict and any('grid' in line.lower() for line in code_lines):
            value_dict['grid'] = {}

        try:
            self._execute_type_block(
                code_lines, value_dict, input_values, line_number)
        except Exception as e:
            raise
        finally:
            self.compiler._allow_hidden_field_access = prev_hidden_access
            self.compiler._allow_hidden_member_calls = prev_hidden_member_calls
            self.compiler.pop_scope()

    def _execute_type_block(self, code_lines, value_dict, input_values, line_number):
        """Execute a list of type code lines within the current scope."""
        i = 0
        while i < len(code_lines):
            code_line = code_lines[i]
            stripped_line = code_line.strip()
            if not stripped_line:
                i += 1
                continue


            if (re.match(r'^(for|let)\b', stripped_line, re.I) and
                    '=' not in stripped_line and not re.search(r'\bdo\b', stripped_line, re.I)):
                # Skip field declarations that slipped into executable code.
                i += 1
                continue

            if stripped_line.lower().startswith('push '):
                assign_line = stripped_line[5:].strip()
                self._process_type_assignment(
                    assign_line, value_dict, input_values, line_number)
                i += 1
                continue
            if (re.match(r'^for\b', stripped_line, re.I) and
                    '=' in stripped_line and not re.search(r'\bdo\b', stripped_line, re.I)):
                for_body = stripped_line[4:].strip()
                var, type_name, constraints, expr = self.compiler._parse_variable_def(
                    for_body, line_number)
                init_expr = (constraints or {}).get('init')
                if expr is None and init_expr is not None:
                    expr = init_expr
                if expr is not None:
                    eval_scope = self._build_type_eval_scope(
                        value_dict, input_values)
                    value = self.compiler.expr_evaluator.eval_or_eval_array(
                        str(expr), eval_scope, line_number)
                    scope = self.compiler.current_scope()
                    defining_scope = scope.get_defining_scope(var)
                    inferred = type_name or self.compiler.array_handler.infer_type(
                        value, line_number)
                    if inferred == 'int':
                        inferred = 'number'
                    if defining_scope:
                        defining_scope.update(var, value, line_number)
                    else:
                        scope.define(var, value, inferred,
                                     constraints or {}, is_uninitialized=False)
                i += 1
                continue
            if re.match(r'^[A-Za-z_][\w_]*\s*\(.*\)\s*$', stripped_line):
                helper_name = stripped_line.split('(', 1)[0].strip()
                type_name = value_dict.get('_type_name') if isinstance(
                    value_dict, dict) else None
                helper_defs = {}
                if type_name:
                    type_def = self.compiler.types_defined.get(type_name.lower(), {})
                    helper_defs = type_def.get('_private_helpers', {}) if isinstance(
                        type_def, dict) else {}
                if helper_name.lower() == 'super' or helper_name.lower() in helper_defs:
                    args_text = stripped_line[stripped_line.find('(') + 1: stripped_line.rfind(')')]
                    args = []
                    if args_text.strip():
                        args = [a.strip()
                                for a in re.split(r',(?![^{]*})', args_text) if a.strip()]
                    eval_scope = self._build_type_eval_scope(
                        value_dict, input_values)
                    arg_values = [self.compiler.expr_evaluator.eval_or_eval_array(
                        a, eval_scope, line_number) for a in args]
                    self._execute_private_helper(
                        type_name, helper_name, value_dict, line_number, arg_values)
                    i += 1
                    continue
            if stripped_line.startswith('[') and ':=' in stripped_line:
                # Assignment to grid: [B1] := 1
                self._process_grid_assignment(
                    stripped_line, 'this', value_dict, line_number)
                i += 1
                continue
            if re.match(r'^for\b', stripped_line, re.I) and re.search(r'\bdo\b', stripped_line, re.I):
                i = self._process_type_for_loop(
                    code_lines, i, value_dict, input_values, line_number)
                continue
            if re.match(r'^let\b', stripped_line, re.I):
                # Let statement: Let grid{a, b} = grid{a-1, b-1} + grid{a-1, b}
                self._process_type_let_statement(
                    stripped_line, 'this', value_dict, line_number)
                i += 1
                continue
            if '=' in stripped_line:
                # Simple assignment inside constructor (e.g., x = in_x)
                self._process_type_assignment(
                    stripped_line, value_dict, input_values, line_number)
                i += 1
                continue
            if stripped_line.lower().startswith('end'):
                i += 1
                continue

            i += 1

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


    def _process_type_for_loop(self, all_lines, start_index, value_dict, input_values, line_number):
        """Process for loop inside type definition."""
        loop_line = all_lines[start_index].strip()
        lower_line = loop_line.lower()
        if not re.search(r'\bdo\b', lower_line):
            # This is a field declaration, not a loop.
            return start_index + 1

        match = re.match(r'^\s*for\s+(.+?)\s+do\s*$', loop_line, re.I)
        if not match:
            raise SyntaxError(f"Unsupported loop syntax: {loop_line}")

        var_defs = match.group(1).strip()
        var_parts = re.split(r'\s+and\s+', var_defs, flags=re.I)
        loop_defs = []
        for var_part in var_parts:
            part = var_part.strip()
            part_match = re.match(
                r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$',
                part, re.I)
            if not part_match:
                raise SyntaxError(f"Unsupported loop syntax: {loop_line}")
            var_name, range_expr, step_str, index_var = part_match.groups()
            step = int(step_str) if step_str else None
            loop_defs.append({
                'var_name': var_name,
                'range_expr': range_expr.strip(),
                'step': step,
                'index_var': index_var
            })

        # Extract loop body
        depth = 1
        end_index = None
        i = start_index + 1
        while i < len(all_lines):
            line = all_lines[i].strip()
            if re.match(r'^\s*for\b', line, re.I) and re.search(r'\bdo\b', line, re.I):
                depth += 1
            elif re.match(r'^\s*end\b', line, re.I):
                depth -= 1
                if depth == 0:
                    end_index = i
                    break
            i += 1
        if end_index is None:
            raise SyntaxError(
                f"Unclosed FOR block starting at line {line_number}")

        loop_body = all_lines[start_index + 1:end_index]

        def _evaluate_loop_values(range_expr, step_value):
            eval_scope = self._build_type_eval_scope(value_dict, input_values)
            if ' to ' in range_expr:
                start_expr, end_expr = range_expr.split(' to ', 1)
                step_expr = None
                if ' step ' in end_expr:
                    end_expr, step_expr = [
                        part.strip() for part in end_expr.split(' step ', 1)]
                start_val = self.compiler.expr_evaluator.eval_expr(
                    start_expr.strip(), eval_scope, line_number)
                end_val = self.compiler.expr_evaluator.eval_expr(
                    end_expr.strip(), eval_scope, line_number)
                start_val = int(start_val)
                end_val = int(end_val)
                step_final = step_value
                if step_expr and step_final is None:
                    step_final = int(self.compiler.expr_evaluator.eval_expr(
                        step_expr, eval_scope, line_number))
                if step_final is None:
                    step_final = 1
                if step_final < 0:
                    return list(range(start_val, end_val - 1, step_final))
                return list(range(start_val, end_val + 1, step_final))
            if range_expr.startswith('{') and range_expr.endswith('}'):
                values = []
                inner = range_expr[1:-1].strip()
                if not inner:
                    return []
                parts = [p.strip() for p in inner.split(',') if p.strip()]
                for part in parts:
                    if (part.startswith('"') and part.endswith('"')) or (
                            part.startswith("'") and part.endswith("'")):
                        values.append(part[1:-1])
                        continue
                    try:
                        values.append(self.compiler.expr_evaluator.eval_expr(
                            part, eval_scope, line_number))
                    except Exception:
                        try:
                            values.append(float(part))
                        except ValueError:
                            values.append(part)
                return values

            result = self.compiler.expr_evaluator.eval_or_eval_array(
                range_expr, eval_scope, line_number)
            if hasattr(result, 'to_pylist'):
                return result.to_pylist()
            if isinstance(result, (list, tuple)):
                return list(result)
            if isinstance(result, numbers.Real):
                return [result]
            if result is None:
                return []
            return [result]

        def _execute_loop(level=0):
            if level >= len(loop_defs):
                self._execute_type_block(
                    loop_body, value_dict, input_values, line_number)
                return
            loop_def = loop_defs[level]
            values = _evaluate_loop_values(
                loop_def['range_expr'], loop_def['step'])
            for idx, val in enumerate(values, start=1):
                self.compiler.push_scope(is_private=False)
                self.compiler.current_scope().define(
                    loop_def['var_name'], val, 'number')
                if loop_def['index_var']:
                    self.compiler.current_scope().define(
                        loop_def['index_var'], idx, 'number')
                _execute_loop(level + 1)
                self.compiler.pop_scope()

        _execute_loop()
        return end_index + 1

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
                    if isinstance(idx1, numbers.Real):
                        idx1 = int(round(idx1))
                    if isinstance(idx2, numbers.Real):
                        idx2 = int(round(idx2))

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
                                    if isinstance(val1, numbers.Real):
                                        val1 = int(round(val1))
                                    if isinstance(val2, numbers.Real):
                                        val2 = int(round(val2))
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
        else:
            # Let field = expr
            match = re.match(r'Let\s+(\$?[\w_]+)\s*=\s*(.+)$', line, re.I)
            if not match:
                return
            field_name, value_expr = match.groups()
            if field_name.startswith('$'):
                field_name = field_name[1:]
            scope = self._build_type_eval_scope(value_dict, {})
            value = self.compiler.expr_evaluator.eval_expr(
                value_expr.strip(), scope, line_number)
            value_dict[field_name] = value

    def _process_type_assignment(self, line, value_dict, input_values, line_number):
        """Handle assignments inside type definitions (e.g., x = in_x)."""
        def _coerce_field_value(field_name, raw_value):
            if not isinstance(value_dict, dict):
                return raw_value
            type_name = value_dict.get('_type_name')
            if not type_name:
                return raw_value
            type_def = self.compiler.types_defined.get(type_name.lower(), {})
            field_map = {
                str(k).lower(): k for k in type_def.keys() if not str(k).startswith('_')
            }
            field_key = field_map.get(str(field_name).lower())
            if not field_key:
                return raw_value
            field_type = type_def.get(field_key)
            if not isinstance(field_type, str):
                return raw_value
            if field_type.lower() not in self.compiler.types_defined:
                return raw_value
            if isinstance(raw_value, dict):
                return raw_value
            if isinstance(raw_value, (list, tuple)):
                return self.compiler._instantiate_type(
                    field_type, list(raw_value), line_number, allow_default_if_empty=True)
            return raw_value

        paren_match = re.match(r'^\s*(\$?[\w_]+)\s*\(([^)]+)\)\s*=\s*(.+)$', line)
        if paren_match:
            field_name, index_expr, value_expr = paren_match.groups()
            if field_name.startswith('$'):
                field_name = field_name[1:]
            scope = self._build_type_eval_scope(value_dict, input_values)
            index_val = self.compiler.expr_evaluator.eval_expr(
                index_expr.strip(), scope, line_number)
            if isinstance(index_val, numbers.Real):
                index_val = int(round(index_val))
            value = self.compiler.expr_evaluator.eval_or_eval_array(
                value_expr.strip(), scope, line_number)
            value = _coerce_field_value(field_name, value)
            arr = value_dict.get(field_name)
            if arr is None or not isinstance(arr, list):
                arr = []
            if index_val is None or index_val < 1:
                raise ValueError(
                    f"Invalid index {index_val} for '{field_name}' at line {line_number}")
            while len(arr) < index_val:
                arr.append(None)
            arr[index_val - 1] = value
            value_dict[field_name] = arr
            return

        brace_match = re.match(r'^\s*(\$?[\w_]+)\s*\{([^}]+)\}\s*=\s*(.+)$', line)
        if brace_match:
            field_name, indices_str, value_expr = brace_match.groups()
            if field_name.startswith('$'):
                field_name = field_name[1:]
            scope = self._build_type_eval_scope(value_dict, input_values)
            indices = [idx.strip() for idx in indices_str.split(',') if idx.strip()]
            if len(indices) != 1:
                raise ValueError(
                    f"Expected 1 index for '{field_name}', got {len(indices)} at line {line_number}")
            index_val = self.compiler.expr_evaluator.eval_expr(
                indices[0], scope, line_number)
            if isinstance(index_val, numbers.Real):
                index_val = int(round(index_val))
            value = self.compiler.expr_evaluator.eval_or_eval_array(
                value_expr.strip(), scope, line_number)
            value = _coerce_field_value(field_name, value)
            arr = value_dict.get(field_name)
            if arr is None or not isinstance(arr, list):
                arr = []
            if index_val is None or index_val < 1:
                raise ValueError(
                    f"Invalid index {index_val} for '{field_name}' at line {line_number}")
            while len(arr) < index_val:
                arr.append(None)
            arr[index_val - 1] = value
            value_dict[field_name] = arr
            return

        match = re.match(r'^\s*(\$?[\w_]+)\s*=\s*(.+)$', line)
        if not match:
            return
        field_name, value_expr = match.groups()
        if field_name.startswith('$'):
            field_name = field_name[1:]
        scope = self._build_type_eval_scope(value_dict, input_values)
        value = self.compiler.expr_evaluator.eval_expr(
            value_expr.strip(), scope, line_number)
        value_dict[field_name] = value
        if input_values:
            tokens = re.findall(r'\b[\w_]+\b', value_expr)
            input_names = {name.lower() for name in input_values.keys()}
            if any(tok.lower() in input_names for tok in tokens):
                immutable_fields = value_dict.setdefault(
                    '_immutable_fields', set())
                immutable_fields.add(field_name.lower())

    def _build_type_eval_scope(self, value_dict, input_values):
        scope = self.compiler.current_scope().get_full_scope()
        scope = dict(scope) if isinstance(scope, dict) else {}
        if input_values:
            scope.update(input_values)
        if isinstance(value_dict, dict):
            for key, val in value_dict.items():
                if not str(key).startswith('_'):
                    scope[key] = val
            if 'grid' in value_dict:
                scope['grid'] = value_dict.get('grid', {})
        return scope

    def _execute_private_helper(self, type_name, helper_name, value_dict, line_number, arg_values):
        if not type_name:
            raise NameError(
                f"Private helper '{helper_name}' has no type context at line {line_number}")
        type_def = self.compiler.types_defined.get(type_name.lower(), {})
        if helper_name.lower() == 'super':
            parent = type_def.get('_parent')
            if not parent:
                raise NameError(
                    f"Super() called but type '{type_name}' has no parent at line {line_number}")
            parent_obj = self.compiler._instantiate_type(
                parent, arg_values, line_number, allow_default_if_empty=True)
            for key, val in parent_obj.items():
                if key == '_type_name':
                    continue
                if key == '_hidden_fields':
                    hidden = set(value_dict.get('_hidden_fields', set()))
                    hidden.update(val or [])
                    value_dict['_hidden_fields'] = hidden
                    continue
                if key == '_immutable_fields':
                    imm = set(value_dict.get('_immutable_fields', set()))
                    imm.update(val or [])
                    value_dict['_immutable_fields'] = imm
                    continue
                if key == 'grid' and isinstance(val, dict):
                    value_dict.setdefault('grid', {}).update(val)
                    continue
                value_dict[key] = val
            return

        helpers = type_def.get('_private_helpers', {}) or {}
        helper_def = helpers.get(helper_name.lower())
        if not helper_def:
            raise NameError(
                f"Private helper '{helper_name}' not defined for type '{type_name}' at line {line_number}")
        input_defs = helper_def.get('input_defs', [])
        input_values = {}
        if input_defs:
            if len(arg_values) > len(input_defs):
                raise ValueError(
                    f"Too many arguments for helper '{helper_name}' at line {line_number}")
            for idx, entry in enumerate(input_defs):
                name = entry.get('name')
                if idx < len(arg_values):
                    input_values[name] = arg_values[idx]
                else:
                    default_expr = entry.get('constraints', {}).get('default') or entry.get('default')
                    if default_expr is None:
                        raise ValueError(
                            f"Missing argument '{name}' for helper '{helper_name}' at line {line_number}")
                    eval_scope = self._build_type_eval_scope(
                        value_dict, input_values)
                    input_values[name] = self.compiler.expr_evaluator.eval_or_eval_array(
                        str(default_expr), eval_scope, line_number)
        elif arg_values:
            raise ValueError(
                f"Helper '{helper_name}' does not take arguments at line {line_number}")

        code_lines = helper_def.get('code_lines') or helper_def.get('code', '').splitlines()
        self._execute_type_code(
            code_lines, 'this', value_dict, line_number, input_values)
