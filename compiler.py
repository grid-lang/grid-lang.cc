import re
import math
import sys
import copy
import pyarrow as pa
from expression import ExpressionEvaluator
from array_handler import ArrayHandler
from utils import col_to_num, num_to_col, split_cell, offset_cell, validate_cell_ref, object_public_keys
from scope import Scope
from control_flow import GridLangControlFlow
from type_processor import GridLangTypeProcessor
from parser import GridLangParser


class SubprocessResult:
    """Container for subprocess execution results."""

    def __init__(self, grid=None, variables=None, outputs=None):
        self.grid = grid
        self._variables = variables or {}
        self.outputs = outputs or {}

    def __getattr__(self, item):
        if item in self._variables:
            return self._variables[item]
        raise AttributeError(
            f"Attribute '{item}' not found in subprocess result")


class GridLangCompiler:
    def __init__(self):
        self.grid = {}
        self.scopes = [Scope(self)]
        self.variables = self.current_scope().variables
        self.types = self.scopes[0].types
        self.dimensions = {}
        self.dim_names = {}
        self.dim_labels = {}
        self.pending_assignments = {}
        self.deferred_lines = []
        self._cell_var_map = {}
        self._cell_array_map = {}
        self.types_defined = {}
        self.functions = {}
        self.subprocesses = {}
        self.expr_evaluator = ExpressionEvaluator(self)
        self.array_handler = ArrayHandler(self)
        self.control_flow = GridLangControlFlow(self)
        self.type_processor = GridLangTypeProcessor(self)
        self.parser = GridLangParser(self)
        self.handled_assignments = set()
        # Grid language features
        self.input_variables = []  # Ordered list of input variables
        self.output_variables = []  # List of output variables
        self.output_values = {}
        self._allow_hidden_field_access = False
        self._allow_hidden_member_calls = False
        self.undefined_dependencies = set()
        self.dependency_graph = {'nodes': [], 'by_variable': {}, 'by_line': {}}
        self.global_guard_entries = []
        self.global_for_line_numbers = set()
        self.executed_global_for_lines = set()
        self.global_for_entries = []
        # Control whether missing inputs should prompt the user (CLI mode only)
        self.prompt_missing_inputs = False

    def _get_public_type_fields(self, type_name_or_def):
        """Return the declared fields for a type, excluding internal metadata keys."""
        from utils import public_type_fields
        if isinstance(type_name_or_def, dict):
            return public_type_fields(type_name_or_def)
        return public_type_fields(self.types_defined.get(type_name_or_def.lower(), {}))

    def _get_all_type_fields(self, type_name_or_def):
        """Return all declared fields for a type, including hidden ones."""
        if isinstance(type_name_or_def, dict):
            return {k: v for k, v in type_name_or_def.items() if not str(k).startswith('_')}
        type_def = self.types_defined.get(type_name_or_def.lower(), {})
        return {k: v for k, v in type_def.items() if not str(k).startswith('_')}

    def _is_hidden_field(self, obj, field_name):
        if not isinstance(obj, dict):
            return False
        hidden = obj.get('_hidden_fields', set())
        if not isinstance(hidden, (set, list, tuple)):
            hidden = set()
        return str(field_name).lower() in {str(h).lower() for h in hidden}

    def _parse_type_header(self, line, line_number=None):
        """Parse a type definition header, returning name, parent, and constraints."""
        m = re.match(
            r'^\s*define\s+([\w_]+)\s+as\s+type(?:\s*\(\s*([\w_]+)\s*\))?\s*(.*)$', line, re.I)
        if not m:
            return None, None, None
        type_name = m.group(1).strip()
        parent = m.group(2).strip() if m.group(2) else None
        remainder = m.group(3).strip()
        constraints = {}
        if remainder:
            try:
                _, _, constraints, _ = self.parser._parse_variable_def(
                    f"_type {remainder}", line_number)
            except Exception as exc:
                raise SyntaxError(
                    f"Invalid type constraints in '{line}' at line {line_number}: {exc}")
        return type_name, parent, constraints

    def _resolve_type_inheritance(self):
        """Merge inherited fields and constraints into child type definitions."""
        primitives = {'number', 'text', 'array', 'bool'}
        progress = True
        while progress:
            progress = False
            for type_name, type_def in list(self.types_defined.items()):
                parent = type_def.get('_parent')
                if not parent or type_def.get('_inheritance_applied'):
                    continue
                parent_lower = parent.lower()
                parent_def = self.types_defined.get(parent_lower)
                if parent_lower in primitives:
                    base_type = parent_lower
                elif parent_def:
                    base_type = parent_def.get('_base_type')
                else:
                    continue

                if parent_def:
                    parent_fields = {k: v for k, v in parent_def.items()
                                     if not str(k).startswith('_')}
                    # Rebuild public field order: parent fields first, then child fields.
                    merged_fields = {}
                    for field, f_type in parent_fields.items():
                        merged_fields[field] = f_type
                    for field, f_type in type_def.items():
                        if str(field).startswith('_'):
                            continue
                        merged_fields[field] = f_type
                    internal_items = {k: v for k, v in type_def.items()
                                      if str(k).startswith('_')}
                    type_def.clear()
                    type_def.update(merged_fields)
                    type_def.update(internal_items)

                    hidden = set(type_def.get('_hidden_fields', set()))
                    parent_hidden = parent_def.get('_hidden_fields', set())
                    if isinstance(parent_hidden, (set, list, tuple)):
                        hidden.update(parent_hidden)
                    if hidden:
                        type_def['_hidden_fields'] = hidden

                    parent_constraints = parent_def.get(
                        '_constraints', {}) or {}
                    child_constraints = type_def.get('_constraints', {}) or {}
                    if parent_constraints or child_constraints:
                        merged = dict(parent_constraints)
                        merged.update(child_constraints)
                        type_def['_constraints'] = merged

                    parent_field_constraints = parent_def.get(
                        '_field_constraints', {}) or {}
                    child_field_constraints = type_def.get(
                        '_field_constraints', {}) or {}
                    if parent_field_constraints or child_field_constraints:
                        merged_fields = dict(parent_field_constraints)
                        merged_fields.update(child_field_constraints)
                        type_def['_field_constraints'] = merged_fields

                if base_type:
                    type_def['_base_type'] = base_type
                type_def['_inheritance_applied'] = True
                progress = True

    def _is_type_compatible(self, actual_type, expected_type):
        if not actual_type or not expected_type:
            return False
        actual = actual_type.lower()
        expected = expected_type.lower()
        if actual == expected:
            return True
        visited = set()
        while actual and actual not in visited:
            visited.add(actual)
            t_def = self.types_defined.get(actual)
            parent = t_def.get('_parent') if isinstance(t_def, dict) else None
            if not parent:
                break
            actual = parent.lower()
            if actual == expected:
                return True
        return False

    def _resolve_member_function(self, obj_type, method_name):
        if not obj_type:
            return None
        current = obj_type.lower()
        visited = set()
        while current and current not in visited:
            visited.add(current)
            func_key = f"{current}.{method_name}".lower()
            if func_key in getattr(self, 'functions', {}):
                return func_key
            t_def = self.types_defined.get(current)
            parent = t_def.get('_parent') if isinstance(t_def, dict) else None
            if not parent:
                break
            current = parent.lower()
        return None

    def _convert_array_to_object(self, type_name, value, line_number=None):
        type_def = self.types_defined.get(type_name.lower())
        if not isinstance(type_def, dict):
            return value
        inputs_list = type_def.get('_inputs', [])
        if inputs_list:
            raise ValueError(
                f"Cannot convert array to type '{type_name}' with constructor inputs at line {line_number}")
        if isinstance(value, pa.Array):
            value = value.to_pylist()
        if not isinstance(value, list):
            return value
        public_fields = list(self._get_public_type_fields(type_def).keys())
        if len(value) != len(public_fields):
            raise ValueError(
                f"Expected {len(public_fields)} values for type '{type_name}', got {len(value)} at line {line_number}")
        obj = {}
        for field_name, field_val in zip(public_fields, value):
            obj[field_name] = field_val
        obj['_type_name'] = type_name.lower()
        hidden_fields = type_def.get('_hidden_fields', set())
        if hidden_fields:
            obj['_hidden_fields'] = set(hidden_fields)
        obj.setdefault('grid', {})
        return obj

    def current_scope(self):
        return self.scopes[-1]

    def push_scope(self, is_private=False, is_loop_scope=False):
        scope = Scope(self, parent=self.current_scope(), is_private=is_private)
        if is_loop_scope:
            scope.is_loop_scope = True
        self.scopes.append(scope)

    def pop_scope(self):
        if len(self.scopes) > 1:
            self.scopes.pop()
        else:
            raise RuntimeError("Cannot pop global scope")

    def run(self, code, args=None, suppress_output=False, return_output=False):
        """Delegates to the extracted run function."""
        from executor import GridLangExecutor

        extracted = GridLangExecutor()
        # Store reference to compiler for output values access
        extracted.compiler = self
        # Copy all necessary attributes from self to extracted
        for attr in ['grid', 'scopes', 'expr_evaluator', 'array_handler', 'types_defined',
                     'dimensions', 'pending_assignments', 'dim_labels', 'undefined_dependencies',
                     'dependency_graph', 'global_guard_entries', 'global_for_line_numbers',
                     'executed_global_for_lines', 'output_values', 'functions', 'subprocesses',
                     'prompt_missing_inputs', '_allow_hidden_field_access', '_allow_hidden_member_calls']:
            if hasattr(self, attr):
                setattr(extracted, attr, getattr(self, attr))

        # Copy ALL methods from self to extracted (except run to avoid recursion)
        for method_name in dir(self):
            if callable(getattr(self, method_name)) and method_name != 'run' and not method_name.startswith('__'):
                setattr(extracted, method_name, getattr(self, method_name))

        # Call the extracted run function
        result = extracted.run(code, args, suppress_output=suppress_output,
                               return_output=return_output)

        # Capture the final root-scope variables for callers that need them
        try:
            self._last_scope_vars = extracted.current_scope().variables.copy()
        except Exception:
            self._last_scope_vars = {}

        # Copy back any changes to attributes
        for attr in ['grid', 'scopes', 'expr_evaluator', 'array_handler', 'types_defined',
                     'dimensions', 'pending_assignments', 'dim_labels', 'undefined_dependencies',
                     'dependency_graph', 'global_guard_entries', 'global_for_line_numbers',
                     'executed_global_for_lines', 'output_values', 'functions', 'subprocesses',
                     'prompt_missing_inputs', '_allow_hidden_field_access', '_allow_hidden_member_calls']:
            if hasattr(extracted, attr):
                setattr(self, attr, getattr(extracted, attr))

        # Keep variables reference aligned with the active root scope
        if hasattr(self, 'scopes') and self.scopes:
            self.variables = self.scopes[0].variables

        return result

    def _extract_functions(self, lines, label_lines, dim_lines):
        """Extract user-defined functions and remove them from main code."""
        functions = getattr(self, 'functions', {}) or {}
        subprocesses = getattr(self, 'subprocesses', {}) or {}
        new_lines = []
        i = 0
        while i < len(lines):
            line, line_number = lines[i]
            m = re.match(
                r'^\s*define\s+(\$?[\w\.]+)\s+as\s+(function|subprocess|privatehelper)', line, re.I)
            if m:
                raw_name = m.group(1).strip()
                def_kind = m.group(2).lower()
                hidden = raw_name.startswith('$')
                func_name = raw_name[1:] if hidden else raw_name
                body_lines = []
                block_depth = 0
                i += 1
                while i < len(lines):
                    body_line, body_ln = lines[i]
                    stripped = body_line.strip().lower()
                    # Track nested control blocks so generic END inside the body doesn't terminate the function
                    if stripped.startswith(('for ', 'while ', 'if ')):
                        block_depth += 1
                    if stripped.startswith('end'):
                        # Named end takes precedence
                        if re.match(r'^\s*end\s+\$?%s\s*$' % re.escape(func_name), body_line, re.I):
                            break
                        # Allow bare "end" (or "end function") to close the definition when not nested
                        if block_depth == 0 and re.match(r'^\s*end(?:\s+(function|subprocess))?\s*$', stripped, re.I):
                            break
                        block_depth = max(0, block_depth - 1)
                        body_lines.append(body_line)
                        i += 1
                        continue
                    body_lines.append(body_line)
                    i += 1
                func_code = "\n".join(body_lines)
                code_lines = [ln for ln in body_lines
                              if not ln.strip().lower().startswith(('input ', 'output '))]
                inputs, input_defs, outputs = [], [], []
                for b in body_lines:
                    m_in = re.match(r'^\s*input\s+(.+)$', b, re.I)
                    if m_in:
                        try:
                            parsed_var, parsed_type, parsed_constraints, _ = self.parser._parse_variable_def(
                                b.strip(), body_ln)
                        except Exception:
                            parsed_var, parsed_type, parsed_constraints = None, None, {}
                        if parsed_var:
                            var_list = (parsed_constraints or {}).get(
                                'var_list') if parsed_constraints else None
                            names = var_list if var_list else [parsed_var]
                            inputs.extend(names)
                            for name in names:
                                input_defs.append({
                                    'name': name,
                                    'type': parsed_type,
                                    'constraints': parsed_constraints or {}
                                })
                    m_out = re.match(r'^\s*output\s+([\w_]+)', b, re.I)
                    if m_out:
                        outputs.append(m_out.group(1).strip())
                entry = {
                    'name': func_name,
                    'code': func_code,
                    'outputs': outputs,
                    'inputs': inputs,
                    'input_defs': input_defs,
                    'member_of': func_name.split('.')[0] if '.' in func_name else None,
                    # Keep the original casing so we can expose multiple aliases
                    'original': func_name,
                    'hidden': hidden,
                    'code_lines': code_lines
                }
                if def_kind == 'privatehelper':
                    type_name = func_name.split(
                        '.')[0] if '.' in func_name else None
                    if not type_name:
                        raise SyntaxError(
                            f"PrivateHelper '{func_name}' missing type prefix at line {line_number}")
                    type_def = self.types_defined.get(type_name.lower())
                    if not type_def:
                        raise SyntaxError(
                            f"Type '{type_name}' not defined for private helper '{func_name}' at line {line_number}")
                    helpers = type_def.setdefault('_private_helpers', {})
                    helpers[func_name.split('.', 1)[1].lower()] = entry
                elif def_kind == 'function':
                    functions[func_name.lower()] = entry
                else:
                    subprocesses[func_name.lower()] = entry
                i += 1
                continue
            new_lines.append((line, line_number))
            i += 1
        self.functions = functions
        self.subprocesses = subprocesses
        return new_lines, label_lines, dim_lines

    def call_function(self, name, args, instance_type=None, collect_all=False, vectorize=True):
        """Invoke a user-defined function by name with the given arguments.

        When collect_all is True, return all pushed output values (as lists)
        instead of collapsing to the last pushed value.
        """
        func_def = getattr(self, 'functions', {}).get(name.lower())
        if not func_def:
            raise NameError(f"Function '{name}' not defined")
        if func_def.get('hidden') and not getattr(self, '_allow_hidden_member_calls', False):
            raise PermissionError(
                f"Hidden member function '{name}' cannot be called here")
        member_of = func_def.get('member_of')
        if member_of:
            if not args:
                raise ValueError(
                    f"Member function '{name}' requires an instance of '{member_of}' as first argument")
            instance = args[0]
            inferred_type = instance_type
            # Fallback: infer from object keys
            if inferred_type is None and isinstance(instance, dict):
                inferred_type = instance.get('_type_name')
            if inferred_type is None and isinstance(instance, dict):
                for t_name, t_def in self.types_defined.items():
                    public_fields = self._get_public_type_fields(t_def)
                    if public_fields and object_public_keys(instance) == set(public_fields.keys()):
                        inferred_type = t_name
                        break
            if inferred_type and not self._is_type_compatible(inferred_type, member_of):
                raise TypeError(
                    f"Member function '{name}' expects instance of '{member_of}', got '{inferred_type}'")

        input_defs = func_def.get('input_defs') or []
        if vectorize and not collect_all:
            array_args = []
            for idx, arg in enumerate(args):
                if isinstance(arg, pa.Array):
                    array_args.append((idx, arg.to_pylist()))
                elif isinstance(arg, (list, tuple)):
                    array_args.append((idx, list(arg)))
            if array_args:
                should_vectorize = False
                for idx, _vals in array_args:
                    dim_spec = None
                    if idx < len(input_defs):
                        dim_spec = input_defs[idx].get(
                            'constraints', {}).get('dim')
                    if isinstance(dim_spec, str) and dim_spec.replace(' ', '') == '{}':
                        continue
                    should_vectorize = True
                    break
                if should_vectorize:
                    lengths = {len(vals) for _, vals in array_args}
                    if len(lengths) > 1:
                        should_vectorize = False
                        array_args = []
                        lengths = set()
                    if not should_vectorize:
                        pass
                    else:
                        count = lengths.pop() if lengths else 0
                        results = []
                        for i in range(count):
                            elem_args = []
                            for arg in args:
                                if isinstance(arg, pa.Array):
                                    elem_args.append(arg.to_pylist()[i])
                                elif isinstance(arg, (list, tuple)):
                                    elem_args.append(list(arg)[i])
                                else:
                                    elem_args.append(arg)
                            results.append(self.call_function(
                                name, elem_args, instance_type=instance_type,
                                collect_all=False, vectorize=False))
                        return results
        for idx, input_def in enumerate(input_defs):
            if idx >= len(args):
                break
            expected_type = input_def.get('type')
            if not expected_type:
                continue
            actual_val = args[idx]
            expected_lower = expected_type.lower()
            if expected_lower in self.types_defined:
                if not isinstance(actual_val, dict):
                    raise TypeError(
                        f"Input '{input_def.get('name')}' expects {expected_type}, got {type(actual_val).__name__}")
                actual_type = actual_val.get('_type_name')
                if actual_type is None:
                    actual_type = None
                    for t_name, t_def in self.types_defined.items():
                        public_fields = self._get_public_type_fields(t_def)
                        if public_fields and object_public_keys(actual_val) == set(public_fields.keys()):
                            actual_type = t_name
                            break
                if actual_type and not self._is_type_compatible(actual_type, expected_lower):
                    raise TypeError(
                        f"Input '{input_def.get('name')}' expects {expected_type}, got {actual_type}")
        sub_compiler = GridLangCompiler()
        sub_compiler.types_defined = getattr(self, 'types_defined', {})
        sub_compiler.functions = getattr(self, 'functions', {})
        sub_compiler.preserve_types_defined = True
        sub_compiler.preserve_functions = True
        sub_compiler._allow_hidden_field_access = True
        sub_compiler._allow_hidden_member_calls = True
        # Seed function scope with global variables so functions can read globals.
        try:
            if getattr(self, 'scopes', None):
                import copy
                global_scope = self.scopes[0]
                input_names = {d.get('name', '').lower()
                               for d in input_defs if d.get('name')}
                seed_vars = {}
                seed_constraints = {}
                seed_types = {}
                for key, val in global_scope.variables.items():
                    if key.lower() in input_names:
                        continue
                    seed_vars[key] = copy.deepcopy(val)
                    if key in global_scope.constraints:
                        seed_constraints[key] = copy.deepcopy(
                            global_scope.constraints[key])
                    if key in global_scope.types:
                        seed_types[key] = copy.deepcopy(
                            global_scope.types[key])
                sub_compiler._seed_globals = {
                    'variables': seed_vars,
                    'constraints': seed_constraints,
                    'types': seed_types,
                    'dimensions': copy.deepcopy(getattr(self, 'dimensions', {})),
                    'dim_names': copy.deepcopy(getattr(self, 'dim_names', {})),
                    'dim_labels': copy.deepcopy(getattr(self, 'dim_labels', {})),
                }
        except Exception:
            pass
        func_result = sub_compiler.run(
            func_def['code'], list(args),
            suppress_output=True, return_output=True)
        outputs = func_result or {}

        # Normalize outputs to lists to preserve all pushed values
        normalized_outputs = {}
        for k, v in outputs.items():
            if isinstance(v, list):
                normalized_outputs[k] = v
            elif v is None:
                normalized_outputs[k] = []
            else:
                normalized_outputs[k] = [v]

        def _pick_default():
            target_names = func_def['outputs'] if func_def['outputs'] else [
                'output']
            for out_name in target_names:
                if out_name in normalized_outputs:
                    return normalized_outputs[out_name]
            if normalized_outputs:
                return next(iter(normalized_outputs.values()))
            return []

        if collect_all:
            return normalized_outputs

        default_values = _pick_default()
        if default_values:
            return default_values[-1]
        return None

    def _grid_to_matrix(self, grid_dict):
        """Convert a grid dict {'A1': val, ...} to a dense 2D list."""
        if not grid_dict:
            return []
        max_row = 0
        max_col = 0
        cells = []
        for cell in grid_dict:
            if isinstance(cell, str):
                try:
                    col, row = split_cell(cell)
                    row_i = int(row)
                    col_i = col_to_num(col)
                except ValueError:
                    continue
            elif isinstance(cell, tuple) and len(cell) == 2:
                try:
                    row_i = int(cell[0])
                    col_i = int(cell[1])
                except (TypeError, ValueError):
                    continue
            else:
                continue
            max_row = max(max_row, row_i)
            max_col = max(max_col, col_i)
            cells.append((row_i, col_i, grid_dict[cell]))

        if max_row == 0 or max_col == 0:
            return []

        matrix = [[0 for _ in range(max_col)] for _ in range(max_row)]
        for r, c, val in cells:
            matrix[r - 1][c - 1] = val
        return matrix

    def _instantiate_type(self, type_name, args, line_number, allow_default_if_empty=False, var_name=None, execute_code=True):
        """Create an instance dict for a user-defined type, honoring inputs and constructor code."""
        type_def = self.types_defined[type_name.lower()]
        public_fields = self._get_public_type_fields(type_def)
        all_fields = self._get_all_type_fields(type_def)
        inputs_list = type_def.get('_inputs', [])
        exec_lines = type_def.get('_executable_code', [])
        base_type = type_def.get('_base_type')
        expected_args = len(inputs_list) if inputs_list else len(public_fields)

        def _normalize_inputs(raw_inputs):
            normalized = []
            for entry in raw_inputs or []:
                if isinstance(entry, dict):
                    normalized.append(entry)
                else:
                    normalized.append(
                        {'name': entry, 'default': None, 'type': None})
            return normalized

        inputs_list = _normalize_inputs(inputs_list)

        args = args or []
        if base_type and not all_fields:
            if not args and allow_default_if_empty:
                if base_type == 'number':
                    return 0
                if base_type == 'text':
                    return ""
                if base_type == 'array':
                    return []
                return None
            if len(args) != 1:
                raise ValueError(
                    f"Expected 1 value for type '{type_name}', got {len(args)} at line {line_number}")
            return args[0]

        if not args and allow_default_if_empty:
            value_dict = {}
            input_values = {}
            if inputs_list:
                for entry in inputs_list:
                    default_expr = entry.get('default')
                    if default_expr is None:
                        raise ValueError(
                            f"Expected {expected_args} values for type '{type_name}', got 0 at line {line_number}")
                    eval_scope = self.current_scope().get_full_scope()
                    input_values[entry.get('name')] = self.expr_evaluator.eval_or_eval_array(
                        str(default_expr), eval_scope, line_number)
            else:
                for field_name, field_type in public_fields.items():
                    if field_type.lower() == 'number':
                        value_dict[field_name] = 0
                    elif field_type.lower() == 'text':
                        value_dict[field_name] = ""
                    elif field_type.lower() == 'array':
                        value_dict[field_name] = []
                    else:
                        value_dict[field_name] = None
            value_dict['_type_name'] = type_name.lower()
            hidden_fields = type_def.get('_hidden_fields', set())
            if hidden_fields:
                value_dict['_hidden_fields'] = set(hidden_fields)
            value_dict.setdefault('grid', {})
            if exec_lines and execute_code:
                self._execute_type_code(
                    exec_lines, var_name or type_name, value_dict, line_number, input_values)
            elif inputs_list:
                value_dict.update(
                    {name: val for name, val in input_values.items() if name in public_fields})
                if value_dict:
                    immutable = value_dict.setdefault(
                        '_immutable_fields', set())
                    immutable.update(n.lower() for n in value_dict.keys())
            return value_dict

        if inputs_list and len(args) > expected_args:
            raise ValueError(
                f"Expected {expected_args} values for type '{type_name}', got {len(args)} at line {line_number}")
        if not inputs_list and expected_args != len(args):
            raise ValueError(
                f"Expected {expected_args} values for type '{type_name}', got {len(args)} at line {line_number}")

        value_dict = {}
        input_values = {}
        target_names = [entry['name'] for entry in inputs_list] if inputs_list else list(
            public_fields.keys())
        for field_name, val in zip(target_names, args):
            if inputs_list:
                input_values[field_name] = val
            else:
                value_dict[field_name] = val

        if inputs_list:
            for entry in inputs_list[len(args):]:
                default_expr = entry.get('default')
                if default_expr is None:
                    raise ValueError(
                        f"Missing value for input '{entry.get('name')}' in type '{type_name}' at line {line_number}")
                eval_scope = self.current_scope().get_full_scope()
                input_values[entry.get('name')] = self.expr_evaluator.eval_or_eval_array(
                    str(default_expr), eval_scope, line_number)

        value_dict['_type_name'] = type_name.lower()
        hidden_fields = type_def.get('_hidden_fields', set())
        if hidden_fields:
            value_dict['_hidden_fields'] = set(hidden_fields)
        value_dict.setdefault('grid', {})
        if exec_lines and execute_code:
            # Track immutability for fields derived from inputs
            if inputs_list:
                value_dict.setdefault('_immutable_fields', set())
            self._execute_type_code(
                exec_lines, var_name or type_name, value_dict, line_number, input_values)
        elif inputs_list:
            # Map inputs directly to matching fields when no executable code is provided
            value_dict.update(
                {name: val for name, val in input_values.items() if name in public_fields})
            if value_dict:
                immutable = value_dict.setdefault('_immutable_fields', set())
                immutable.update(n.lower() for n in value_dict.keys())

        return value_dict

    def _evaluate_with_value(self, raw_value, scope, line_number=None):
        """Evaluate a WITH clause value when it looks like an expression."""
        if isinstance(raw_value, str):
            if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', raw_value):
                return scope.get(raw_value, raw_value)
            try:
                return self.expr_evaluator.eval_or_eval_array(
                    raw_value, scope, line_number)
            except Exception:
                # Fallback: treat brace values as literal lists when identifiers are unresolved
                text = raw_value.strip()
                if text.startswith('{') and text.endswith('}'):
                    inner = text[1:-1].strip()
                    if not inner:
                        return []
                    items = [item.strip() for item in inner.split(',')]
                    resolved = []
                    for item in items:
                        if not item:
                            continue
                        if (item.startswith('"') and item.endswith('"')) or (
                                item.startswith("'") and item.endswith("'")):
                            resolved.append(item[1:-1])
                        elif re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', item) and item in scope:
                            resolved.append(scope.get(item))
                        else:
                            resolved.append(item)
                    return resolved
                return raw_value
        return raw_value

    def _apply_with_constraints(self, value, with_constraints, scope, line_number=None, type_name=None):
        """Apply WITH clause values to a newly created object."""
        if not with_constraints or not isinstance(value, dict):
            return value
        field_map = {}
        if type_name and type_name.lower() in self.types_defined:
            public_fields = self._get_public_type_fields(type_name)
            field_map = {k.lower(): k for k in public_fields}
        else:
            field_map = {k.lower(): k for k in object_public_keys(value)}
        for key, raw_value in with_constraints.items():
            key_name = field_map.get(str(key).lower(), key)
            value[key_name] = self._evaluate_with_value(
                raw_value, scope, line_number)
            if type_name:
                self._check_type_field_constraints(
                    type_name, key_name, value[key_name], value, line_number)
        if type_name and type_name.lower() in self.types_defined:
            self._recompute_type_fields_after_with(
                type_name, value, scope, line_number)
        return value

    def _check_type_field_constraints(self, type_name, field_name, value, value_dict, line_number=None):
        type_def = self.types_defined.get(type_name.lower())
        if not type_def or not isinstance(type_def, dict):
            return
        constraints_map = type_def.get('_field_constraints', {}) or {}
        actual_key = None
        for key in constraints_map.keys():
            if str(key).lower() == str(field_name).lower():
                actual_key = key
                break
        if actual_key is None:
            return
        constraints = constraints_map.get(actual_key, {})
        if not constraints:
            return
        tmp_scope = Scope(self)
        if isinstance(value_dict, dict):
            for k, v in value_dict.items():
                if not str(k).startswith('_'):
                    tmp_scope.variables[k] = v
        tmp_scope.constraints[actual_key] = constraints
        tmp_scope._check_constraints(actual_key, value, line_number)

    def _recompute_type_fields_after_with(self, type_name, value, scope, line_number=None):
        type_def = self.types_defined.get(type_name.lower())
        if not type_def or not isinstance(value, dict):
            return
        exec_lines = type_def.get('_executable_code', [])
        if not exec_lines:
            return
        input_names = {
            entry.get('name', '').lower()
            for entry in (type_def.get('_inputs') or [])
            if isinstance(entry, dict)
        }
        field_names = {k.lower()
                       for k in self._get_public_type_fields(type_def)}
        eval_scope = {}
        if hasattr(self.current_scope(), 'get_full_scope'):
            eval_scope = self.current_scope().get_full_scope()
        elif isinstance(scope, dict):
            eval_scope = scope
        eval_scope = dict(eval_scope) if isinstance(eval_scope, dict) else {}
        for key, val in value.items():
            if not str(key).startswith('_'):
                eval_scope[key] = val

        string_pattern = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
        ident_pattern = re.compile(r'[A-Za-z_][A-Za-z0-9_.]*')

        for raw_line in exec_lines:
            stripped = raw_line.strip()
            if not stripped or stripped.lower().startswith('super'):
                continue
            m = re.match(r'^\s*(\$?[\w_]+)\s*=\s*(.+)$', stripped)
            if not m:
                continue
            field_name, expr = m.groups()
            if field_name.startswith('$'):
                field_name = field_name[1:]
            cleaned = string_pattern.sub(' ', expr)
            tokens = ident_pattern.findall(cleaned)
            deps = {tok.split('.')[0].lower() for tok in tokens if tok}
            deps.discard(field_name.lower())
            if not deps:
                continue
            if deps & input_names:
                continue
            if not deps & field_names:
                continue
            value[field_name] = self.expr_evaluator.eval_expr(
                expr, eval_scope, line_number)
            eval_scope[field_name] = value[field_name]

    def _parse_with_clause(self, with_text, line_number=None):
        """Parse a WITH clause body into key/value expression pairs."""
        if not with_text:
            return {}
        text = with_text.strip()
        if text.lower().startswith('with'):
            text = text[4:].strip()
        if text.startswith('(') and text.endswith(')'):
            text = text[1:-1].strip()
        if not text:
            return {}

        parts = []
        current = ""
        in_quotes = False
        paren_level = 0
        brace_level = 0
        for char in text + ',':
            if char == '"' and (len(current) == 0 or current[-1] != '\\'):
                in_quotes = not in_quotes
            elif not in_quotes:
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level = max(paren_level - 1, 0)
                elif char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level = max(brace_level - 1, 0)
            if char == ',' and not in_quotes and paren_level == 0 and brace_level == 0:
                if current.strip():
                    parts.append(current.strip())
                current = ""
            else:
                current += char

        assignments = {}
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                assignments[key.strip()] = value.strip()
            else:
                name = part.strip()
                if name:
                    assignments[name] = name
        return assignments

    def _split_new_with_expr(self, expr):
        """Split a 'new Type ... with (...)' expression into base and with clause."""
        if not expr or not expr.lower().startswith('new '):
            return None, None
        in_quote = None
        paren_level = 0
        brace_level = 0
        lower_expr = expr.lower()
        for idx, ch in enumerate(expr):
            if in_quote:
                if ch == in_quote and (idx == 0 or expr[idx - 1] != '\\'):
                    in_quote = None
                continue
            if ch in ('"', "'"):
                in_quote = ch
                continue
            if ch == '(':
                paren_level += 1
            elif ch == ')':
                paren_level = max(paren_level - 1, 0)
            elif ch == '{':
                brace_level += 1
            elif ch == '}':
                brace_level = max(brace_level - 1, 0)
            if paren_level == 0 and brace_level == 0 and lower_expr.startswith('with', idx):
                before = expr[idx - 1] if idx > 0 else ' '
                after = expr[idx + 4] if idx + 4 < len(expr) else ''
                if (before.isspace() or before in '})') and (after.isspace() or after == '('):
                    base_expr = expr[:idx].strip()
                    with_text = expr[idx:].strip()
                    return base_expr, with_text
        return None, None

    def _apply_single_binding(self, binding, values, scope, line_number=None):
        """Apply subprocess output binding to a target."""
        if values is None:
            return
        target = binding.strip()
        if not target:
            return
        if isinstance(values, pa.Array):
            values = values.to_pylist()
        simple_value = values[0] if isinstance(
            values, list) and len(values) == 1 else values

        if target.startswith('[') and target.endswith(']'):
            inner = target[1:-1].strip()
            is_horizontal = inner.startswith('^')
            inner = inner[1:].strip() if is_horizontal else inner
            cell_ref = self.expr_evaluator._resolve_column_interpolated_cell(
                inner, scope.get_evaluation_scope(), line_number) or inner
            validate_cell_ref(cell_ref)
            if isinstance(values, list):
                all_scalars = all(not isinstance(v, (list, dict, pa.Array))
                                  for v in values)
                if all_scalars:
                    # For scalar pushes, keep the last value only
                    simple_value = values[-1]
                    self.grid[cell_ref] = simple_value
                    return
            assign_value = values if isinstance(values, list) else [values]
            expr_hint = '{' + ','.join(['x'] * len(assign_value)) + '}'
            if is_horizontal or len(assign_value) > 1:
                self.array_handler._assign_horizontal_array(
                    cell_ref, assign_value, expr_hint, line_number=line_number)
            else:
                self.grid[cell_ref] = simple_value
            return

        try:
            scope.update(target, simple_value, line_number)
        except NameError:
            scope.define(target, simple_value)

    def _apply_subprocess_outputs(self, sp_def, output_values, output_bindings, caller_scope=None, line_number=None):
        """Map subprocess output variables to caller-provided bindings."""
        if not output_bindings or not sp_def.get('outputs'):
            return
        scope = caller_scope or self.current_scope()
        outputs = output_values or {}
        for idx, out_name in enumerate(sp_def.get('outputs', [])):
            if idx >= len(output_bindings):
                break
            binding = output_bindings[idx]
            if binding is None:
                continue
            out_key = out_name.lower()
            if out_key not in outputs:
                continue
            self._apply_single_binding(
                binding, outputs.get(out_key), scope, line_number=line_number)

    def call_subprocess(self, name, args, output_bindings=None, line_number=None, collect_all=False):
        """Invoke a user-defined subprocess by name with the given arguments."""
        sp_def = getattr(self, 'subprocesses', {}).get(name.lower())
        if not sp_def:
            raise NameError(f"Subprocess '{name}' not defined")
        sub_compiler = GridLangCompiler()
        sub_compiler.types_defined = getattr(self, 'types_defined', {})
        sub_compiler.functions = getattr(self, 'functions', {})
        sub_compiler.subprocesses = getattr(self, 'subprocesses', {})
        sub_compiler.preserve_types_defined = True
        sub_compiler.preserve_functions = True
        sub_compiler.preserve_subprocesses = True
        # Seed subprocess scope with global variables so subprocesses can read globals.
        try:
            if getattr(self, 'scopes', None):
                import copy
                global_scope = self.scopes[0]
                input_names = {name.lower()
                               for name in (sp_def.get('inputs') or []) if name}
                seed_vars = {}
                seed_constraints = {}
                seed_types = {}
                for key, val in global_scope.variables.items():
                    if key.lower() in input_names:
                        continue
                    seed_vars[key] = copy.deepcopy(val)
                    if key in global_scope.constraints:
                        seed_constraints[key] = copy.deepcopy(
                            global_scope.constraints[key])
                    if key in global_scope.types:
                        seed_types[key] = copy.deepcopy(
                            global_scope.types[key])
                sub_compiler._seed_globals = {
                    'variables': seed_vars,
                    'constraints': seed_constraints,
                    'types': seed_types,
                    'dimensions': copy.deepcopy(getattr(self, 'dimensions', {})),
                    'dim_names': copy.deepcopy(getattr(self, 'dim_names', {})),
                    'dim_labels': copy.deepcopy(getattr(self, 'dim_labels', {})),
                }
        except Exception:
            pass
        sub_output = sub_compiler.run(
            sp_def['code'], list(args),
            suppress_output=True, return_output=True)

        # Merge declared outputs from subprocess scope when not pushed explicitly.
        merged_outputs = dict(sub_output or {})
        try:
            for out_name in sp_def.get('outputs', []) or []:
                out_key = out_name.lower()
                if out_key in merged_outputs:
                    continue
                try:
                    merged_val = sub_compiler.current_scope().get(out_name)
                except Exception:
                    merged_val = None
                if merged_val is not None:
                    merged_outputs[out_key] = merged_val
        except Exception:
            pass

        if output_bindings:
            self._apply_subprocess_outputs(
                sp_def, merged_outputs, output_bindings, caller_scope=self.current_scope(), line_number=line_number)

        normalized_outputs = {}
        if merged_outputs:
            for k, v in merged_outputs.items():
                if isinstance(v, list):
                    normalized_outputs[k] = v
                elif v is None:
                    normalized_outputs[k] = []
                else:
                    normalized_outputs[k] = [v]

        grid_source = sub_compiler.grid or sub_compiler.current_scope().variables.get(
            'grid')

        def _normalize_grid(value):
            if value is None:
                return []
            if isinstance(value, pa.Array):
                value = value.to_pylist()
            if isinstance(value, dict):
                return self._grid_to_matrix(value)
            if isinstance(value, list):
                if value and all(isinstance(r, list) for r in value):
                    return value
                return [value]
            return []

        result_grid = _normalize_grid(grid_source)

        # If caller requested raw outputs map, include grid as a flattened sequence for generator use
        if collect_all and result_grid:
            flat = []
            for row in result_grid:
                if isinstance(row, list):
                    flat.extend(row)
                else:
                    flat.append(row)
            # Preserve any existing outputs, but make the grid-derived sequence available
            normalized_outputs.setdefault('grid', flat)

        # Propagate shared variables back to caller scope when names overlap
        if hasattr(self, 'current_scope'):
            caller_scope = self.current_scope()
            sub_vars = getattr(sub_compiler, '_last_scope_vars', {}) or {}
            for var_name, val in sub_vars.items():
                try:
                    defining = caller_scope.get_defining_scope(var_name)
                except Exception:
                    defining = None
                if defining:
                    try:
                        defining.update(
                            var_name, copy.deepcopy(val), line_number)
                    except Exception:
                        try:
                            defining.update(var_name, val, line_number)
                        except Exception:
                            pass

        if collect_all:
            return normalized_outputs
        return SubprocessResult(
            grid=result_grid,
            variables=sub_compiler.current_scope().variables.copy(),
            outputs=normalized_outputs or {})

    def _resolve_global_dependency(self, var, line_number, target_scope=None):
        if var not in self.pending_assignments:
            return False
        assignment = self.pending_assignments[var]
        expr, assign_line, deps = assignment[:3]
        constraints = assignment[3] if len(assignment) > 3 else {}
        scope = target_scope if target_scope is not None else self.current_scope()
        unresolved = any(
            dep != var and self.has_unresolved_dependency(dep, scope=scope)
            for dep in deps)
        if unresolved:
            return False
        try:
            eval_scope = scope.get_full_scope()
            if hasattr(scope, 'get_evaluation_scope'):
                eval_scope = scope.get_evaluation_scope()
            value = self.expr_evaluator.eval_or_eval_array(
                expr, eval_scope, assign_line)
            value = self.array_handler.check_dimension_constraints(
                var, value, assign_line)
            if constraints.get('with'):
                defining_scope = scope.get_defining_scope(var)
                type_name = None
                if defining_scope:
                    actual_key = defining_scope._get_case_insensitive_key(
                        var, defining_scope.types)
                    if actual_key:
                        type_name = defining_scope.types.get(actual_key)
                value = self._apply_with_constraints(
                    value, constraints.get('with', {}),
                    scope.get_full_scope(), assign_line,
                    type_name=type_name)
            defining_scope = scope.get_defining_scope(var)
            if not defining_scope:
                defining_scope = self.current_scope()
            defining_scope.update(var, value, assign_line)
            del self.pending_assignments[var]
            return True
        except ValueError as e:
            del self.pending_assignments[var]
            self.grid.clear()
            return False
        except NameError as e:
            return False
        except Exception as e:
            raise RuntimeError(
                f"Error resolving global dependency '{var}': {e} at line {assign_line}")

    def _resolve_pending_assignments(self):
        max_attempts = len(self.pending_assignments) + 10
        attempt = 0
        while self.pending_assignments and attempt < max_attempts:
            unresolved_before = set(self.pending_assignments.keys())
            for var, assignment in sorted(self.pending_assignments.items(), key=lambda x: (x[0].startswith('__line_'), int(x[0].replace('__line_', '') if x[0].startswith('__line_') else '0'))):
                expr, line_number, deps = assignment[:3]
                if var in deps and not var.startswith('__line_'):
                    raise ValueError(
                        f"Self-referential assignment '{var} = {expr}' at line {line_number}")
                if var.startswith("__line_"):
                    target, rhs = expr.split(':=')
                    target, rhs = target.strip(), rhs.strip()
                    unresolved = any(
                        self.has_unresolved_dependency(
                            dep, scope=self.current_scope())
                        for dep in deps)
                    if unresolved:
                        continue
                    is_array_indexing = bool(re.match(
                        r'^[\w_]+\s*(?:\[\w+\]|\{\s*\d+\s*(?:,\s*\d+\s*)*\}|!\w+\s*\(\s*"\w+"\s*\))$', rhs.strip()))
                    cell_refs = set()
                    if not is_array_indexing:
                        cell_refs = self._extract_cell_refs(rhs)
                    if cell_refs and not all(ref in self.grid for ref in cell_refs):
                        continue
                    try:
                        violations = []
                        constraints = assignment[3] if len(
                            assignment) > 3 else {}
                        for dep in deps:
                            defining_scope = self.current_scope().get_defining_scope(dep)
                            dep_value = defining_scope.get(
                                dep) if defining_scope else None
                            if dep in constraints:
                                for constraint_type, constraint_val in constraints[dep].items():
                                    try:
                                        constraint_val = float(self.expr_evaluator.eval_expr(
                                            constraint_val, self.current_scope().get_full_scope(), line_number))
                                        if dep_value is not None:
                                            if constraint_type == '>' and dep_value <= constraint_val:
                                                raise ValueError(
                                                    f"'{dep}' is not greater than {constraint_val} at line {line_number}")
                                            if constraint_type == '<' and dep_value >= constraint_val:
                                                raise ValueError(
                                                    f"'{dep}' is not less than {constraint_val} at line {line_number}")
                                            if constraint_type == '=' and dep_value != constraint_val:
                                                raise ValueError(
                                                    f"'{dep}' is not equal to {constraint_val} at line {line_number}")
                                        else:
                                            violations.append(dep)
                                    except ValueError as e:
                                        violations.append(dep)
                            if defining_scope and dep in defining_scope.constraints:
                                try:
                                    if dep_value is not None:
                                        defining_scope._check_constraints(
                                            dep, dep_value, line_number)
                                    else:
                                        violations.append(dep)
                                except ValueError as e:
                                    violations.append(dep)
                        if not violations:
                            value = self.expr_evaluator.eval_or_eval_array(
                                expr, self.current_scope().get_full_scope(), line_number)
                            value = self.array_handler.check_dimension_constraints(
                                var, value, line_number)
                            if constraints.get('with'):
                                scope = self.current_scope()
                                defining_scope = scope.get_defining_scope(var)
                                type_name = None
                                if defining_scope:
                                    actual_key = defining_scope._get_case_insensitive_key(
                                        var, defining_scope.types)
                                    if actual_key:
                                        type_name = defining_scope.types.get(
                                            actual_key)
                                value = self._apply_with_constraints(
                                    value, constraints.get('with', {}),
                                    scope.get_full_scope(), line_number,
                                    type_name=type_name)
                            self.current_scope().update(var, value, line_number)
                            del self.pending_assignments[var]
                        else:
                            self.grid.clear()
                            del self.pending_assignments[var]
                    except NameError as e:
                        missing = self.extract_missing_dependencies(e)
                        if missing:
                            for dep in missing:
                                self.mark_dependency_missing(dep)
                            updated_deps = set(deps) | set(missing)
                            self.pending_assignments[var] = (
                                expr, line_number, updated_deps, constraints)
                        continue
                    except Exception as e:
                        raise RuntimeError(
                            f"Error resolving '{var}' from '{expr}': {e} at line {line_number}")
                else:
                    constraints = assignment[3] if len(assignment) > 3 else {}
                    cell_refs = self._extract_cell_refs(expr)
                    if cell_refs and not all(ref in self.grid for ref in cell_refs):
                        continue
                    unresolved = any(
                        self.has_unresolved_dependency(
                            dep, scope=self.current_scope())
                        for dep in deps)
                    if unresolved:
                        continue
                    try:
                        value = self.expr_evaluator.eval_or_eval_array(
                            expr, self.current_scope().get_full_scope(), line_number)
                        value = self.array_handler.check_dimension_constraints(
                            var, value, line_number)
                        if constraints.get('with') and not target.startswith('['):
                            scope = self.current_scope()
                            defining_scope = scope.get_defining_scope(target)
                            type_name = None
                            if defining_scope:
                                actual_key = defining_scope._get_case_insensitive_key(
                                    target, defining_scope.types)
                                if actual_key:
                                    type_name = defining_scope.types.get(
                                        actual_key)
                            value = self._apply_with_constraints(
                                value, constraints.get('with', {}),
                                scope.get_full_scope(), line_number,
                                type_name=type_name)
                        self.current_scope().update(var, value, line_number)
                        violations = []
                        defining_scope = self.current_scope().get_defining_scope(var)
                        if defining_scope and var in defining_scope.constraints:
                            try:
                                defining_scope._check_constraints(
                                    var, value, line_number)
                            except ValueError as e:
                                violations.append(var)
                                self.grid.clear()
                        if not violations:
                            del self.pending_assignments[var]
                        else:
                            del self.pending_assignments[var]
                    except ValueError as e:
                        del self.pending_assignments[var]
                        self.grid.clear()
                    except NameError as e:
                        missing = self.extract_missing_dependencies(e)
                        if missing:
                            for dep in missing:
                                self.mark_dependency_missing(dep)
                            updated_deps = set(deps) | set(missing)
                            self.pending_assignments[var] = (
                                expr, line_number, updated_deps, constraints)
                        continue
                    except Exception as e:
                        raise RuntimeError(
                            f"Error resolving '{var}' from '{expr}': {e} at line {line_number}")
            if set(self.pending_assignments.keys()) == unresolved_before:
                break
            attempt += 1

        block_pending = {}
        for scope in self.scopes:
            if hasattr(scope, 'pending_assignments'):
                block_pending.update(scope.pending_assignments)
        for var, assignment in sorted(block_pending.items(), key=lambda x: int(x[0].replace('__block_line_', '')) if x[0].startswith('__block_line_') else '0'):
            expr, line_number, deps = assignment[:3]
            constraints = assignment[3] if len(assignment) > 3 else {}
            scope = self.current_scope()
            unresolved = any(
                self.has_unresolved_dependency(dep, scope=scope) for dep in deps)
            if unresolved:
                continue
            try:
                violations = []
                for dep in deps:
                    defining_scope = self.current_scope().get_defining_scope(dep)
                    dep_value = defining_scope.get(
                        dep) if defining_scope else None
                    if dep in constraints:
                        for constraint_type, constraint_val in constraints[dep].items():
                            try:
                                constraint_val = float(self.expr_evaluator.eval_expr(
                                    constraint_val, self.current_scope().get_full_scope(), line_number))
                                if dep_value is not None:
                                    if constraint_type == '>' and dep_value <= constraint_val:
                                        raise ValueError(
                                            f"'{dep}' is not greater than {constraint_val} at line {line_number}")
                                    if constraint_type == '<' and dep_value >= constraint_val:
                                        raise ValueError(
                                            f"'{dep}' is not less than {constraint_val} at line {line_number}")
                                    if constraint_type == '=' and dep_value != constraint_val:
                                        raise ValueError(
                                            f"'{dep}' is not equal to {constraint_val} at line {line_number}")
                                else:
                                    violations.append(dep)
                            except ValueError as e:
                                violations.append(dep)
                    if defining_scope and dep in defining_scope.constraints:
                        try:
                            if dep_value is not None:
                                defining_scope._check_constraints(
                                    dep, dep_value, line_number)
                            else:
                                violations.append(dep)
                        except ValueError as e:
                            violations.append(dep)
                if not violations:
                    value = self.expr_evaluator.eval_or_eval_array(
                        expr, self.current_scope().get_full_scope(), line_number)
                    value = self.array_handler.check_dimension_constraints(
                        var, value, line_number)
                    if constraints.get('with'):
                        scope = self.current_scope()
                        defining_scope = scope.get_defining_scope(var)
                        type_name = None
                        if defining_scope:
                            actual_key = defining_scope._get_case_insensitive_key(
                                var, defining_scope.types)
                            if actual_key:
                                type_name = defining_scope.types.get(
                                    actual_key)
                        value = self._apply_with_constraints(
                            value, constraints.get('with', {}),
                            scope.get_full_scope(), line_number,
                            type_name=type_name)
                    self.current_scope().update(var, value, line_number)
                    del block_pending[var]
                    for scope in self.scopes:
                        if hasattr(scope, 'pending_assignments') and var in scope.pending_assignments:
                            del scope.pending_assignments[var]
                else:
                    self.grid.clear()
                    del block_pending[var]
                    for scope in self.scopes:
                        if hasattr(scope, 'pending_assignments') and var in scope.pending_assignments:
                            del scope.pending_assignments[var]
            except Exception as e:
                raise RuntimeError(
                    f"Error resolving block assignment '{var}': {e} at line {line_number}")
        if self.pending_assignments or block_pending:
            unresolved = list(self.pending_assignments.keys()
                              ) + list(block_pending.keys())
            raise RuntimeError(f"Unresolved assignments: {unresolved}")

    def _parse_variable_def(self, def_str, line_number):
        """Delegate to parser."""
        return self.parser._parse_variable_def(def_str, line_number)

    def _extract_identifier_tokens(self, expr):
        """Extract identifier-like tokens ignoring string literals and numeric literals."""
        if not expr:
            return set()
        cleaned = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', ' ', expr)
        # Remove member accesses like "obj.field" or "obj.method" to avoid
        # treating field/method names as standalone dependencies.
        cleaned = re.sub(r'\.\s*[A-Za-z_][A-Za-z0-9_]*', ' ', cleaned)
        tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', cleaned)
        filtered = set()
        keyword_exclusions = {
            'to', 'and', 'or', 'not', 'then', 'do', 'step', 'by', 'in'
        }
        for tok in tokens:
            if re.match(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$', tok, re.I):
                continue
            if re.match(r'^e[+-]?\d*$', tok, re.I):
                continue
            lower_tok = tok.lower()
            if lower_tok in keyword_exclusions:
                continue
            if lower_tok in getattr(self, 'types_defined', {}):
                continue
            if lower_tok in getattr(self, 'functions', {}):
                continue
            if lower_tok in getattr(self, 'subprocesses', {}):
                continue
            filtered.add(tok)
        return filtered

    def _extract_cell_refs(self, expr):
        cell_refs = set()
        single_matches = re.finditer(r'\[([A-Za-z]+\d+)\]', expr)
        for match in single_matches:
            cell_refs.add(match.group(1))
        range_matches = re.finditer(r'\[([A-Za-z]+\d+):([A-Za-z]+\d+)\]', expr)
        for match in range_matches:
            start, end = match.group(1), match.group(2)
            start_col, start_row = split_cell(start)
            end_col, end_row = split_cell(end)
            start_col_num = col_to_num(start_col)
            end_col_num = col_to_num(end_col)
            for col_num in range(start_col_num, end_col_num + 1):
                for row in range(int(start_row), int(end_row) + 1):
                    cell_refs.add(f"{num_to_col(col_num)}{row}")
        if '$"' in expr:
            placeholders = re.findall(r'\{\s*([^}]*?)\s*\}', expr)
            for ph in placeholders:
                cell_refs.update(re.findall(r'\b[A-Za-z]+\d+\b', ph))
        return cell_refs

    def _split_assignment_expr(self, text):
        """Split on the first standalone '=' not part of comparison operators."""
        in_quote = False
        quote_char = None
        paren_level = 0
        brace_level = 0
        bracket_level = 0
        for i, ch in enumerate(text):
            if ch in ('"', "'") and (i == 0 or text[i - 1] != '\\'):
                if not in_quote:
                    in_quote = True
                    quote_char = ch
                elif quote_char == ch:
                    in_quote = False
                    quote_char = None
            if not in_quote:
                if ch == '(':
                    paren_level += 1
                elif ch == ')':
                    paren_level = max(0, paren_level - 1)
                elif ch == '{':
                    brace_level += 1
                elif ch == '}':
                    brace_level = max(0, brace_level - 1)
                elif ch == '[':
                    bracket_level += 1
                elif ch == ']':
                    bracket_level = max(0, bracket_level - 1)
            if ch == '=' and not in_quote and paren_level == 0 and brace_level == 0 and bracket_level == 0:
                prev = text[i - 1] if i > 0 else ''
                if prev in ('<', '>', '!'):
                    continue
                return text[:i], text[i + 1:]
        return text, None

    # Dependency tracking helpers
    def mark_dependency_missing(self, name):
        if not name:
            return
        self.undefined_dependencies.add(name.lower())

    def mark_dependency_resolved(self, name):
        if not name:
            return
        self.undefined_dependencies.discard(name.lower())

    def extract_missing_dependencies(self, error):
        """Return identifiers mentioned in NameError messages for deferral."""
        pattern = re.compile(r"name '([^']+)' is not defined", re.I)
        stack = [error]
        seen = set()
        missing = set()
        while stack:
            err = stack.pop()
            if err in seen:
                continue
            seen.add(err)
            message = str(err)
            for match in pattern.finditer(message):
                missing.add(match.group(1))
            cause = getattr(err, '__cause__', None)
            context = getattr(err, '__context__', None)
            if cause:
                stack.append(cause)
            if context:
                stack.append(context)
        return missing

    def has_unresolved_dependency(self, name, scope=None, include_global_pending=True, scope_pending=None):
        if not name:
            return False
        normalized = name.lower()
        if normalized.startswith("__line_"):
            return False
        # User-defined functions are not variable dependencies
        if hasattr(self, 'functions') and normalized in (self.functions or {}):
            return False
        # Treat direct cell references (e.g., A1, B2) as resolved if the grid
        # already contains the cell value, even though they are not tracked in
        # the variable scope dictionary.
        if re.match(r'^[A-Za-z]+\d+$', name):
            if name in self.grid:
                return False
        scope = scope or self.current_scope()
        # Treat entirely undefined names as unresolved so dependent lines are deferred
        if not scope.get_defining_scope(name):
            return True
        if normalized in self.undefined_dependencies:
            return True
        if scope.is_uninitialized(name):
            return True
        if include_global_pending and name in self.pending_assignments:
            return True
        if scope_pending and name in scope_pending:
            return True
        return False

    def _reset_state(self):
        self.grid.clear()
        self.scopes = [Scope(self)]
        self.variables = self.scopes[0].variables
        self.types = self.scopes[0].types
        self.pending_assignments = {}
        self.dimensions.clear()
        self.dim_names.clear()
        self.dim_labels.clear()
        self._cell_var_map.clear()
        self._cell_array_map.clear()
        if not getattr(self, 'preserve_types_defined', False):
            self.types_defined.clear()
        self.handled_assignments.clear()
        self.root_scope = self.current_scope()  # Always set root scope here
        if hasattr(self, 'output_values'):
            self.output_values.clear()
        # Seed globals for function/subprocess calls when provided.
        seed = getattr(self, '_seed_globals', None)
        if seed:
            scope = self.current_scope()
            seed_vars = seed.get('variables', {}) or {}
            seed_constraints = seed.get('constraints', {}) or {}
            seed_types = seed.get('types', {}) or {}
            for name, value in seed_vars.items():
                try:
                    scope.define(
                        name,
                        value,
                        seed_types.get(name),
                        seed_constraints.get(name, {}),
                        is_uninitialized=False)
                except Exception:
                    try:
                        scope.variables[name] = value
                        scope.types[name] = seed_types.get(name)
                        scope.constraints[name] = seed_constraints.get(
                            name, {})
                        scope.uninitialized.discard(name)
                    except Exception:
                        pass
            try:
                self.dimensions = seed.get('dimensions', {}) or {}
                self.dim_names = seed.get('dim_names', {}) or {}
                self.dim_labels = seed.get('dim_labels', {}) or {}
            except Exception:
                pass
        # Execution scheduling metadata
        if hasattr(self, 'global_guard_line_numbers'):
            self.global_guard_line_numbers.clear()
        else:
            self.global_guard_line_numbers = set()
        self.global_guard_allows_execution = True
        if hasattr(self, 'undefined_dependencies'):
            self.undefined_dependencies.clear()
        else:
            self.undefined_dependencies = set()
        if hasattr(self, 'dependency_graph'):
            self.dependency_graph['nodes'].clear()
            self.dependency_graph['by_variable'].clear()
            self.dependency_graph['by_line'].clear()
        else:
            self.dependency_graph = {'nodes': [],
                                     'by_variable': {}, 'by_line': {}}
        if hasattr(self, 'global_guard_entries'):
            self.global_guard_entries.clear()
        else:
            self.global_guard_entries = []
        if hasattr(self, 'global_for_line_numbers'):
            self.global_for_line_numbers.clear()
        else:
            self.global_for_line_numbers = set()
        if hasattr(self, 'executed_global_for_lines'):
            self.executed_global_for_lines.clear()
        else:
            self.executed_global_for_lines = set()
        if hasattr(self, 'global_for_entries'):
            self.global_for_entries.clear()
        else:
            self.global_for_entries = []

    def _preprocess_code(self, code):

        lines = []
        label_lines = []
        dim_lines = []
        type_def_lines = []

        in_type_def = False
        type_name = None
        type_parent = None
        type_constraints = {}
        line_number = 0
        current_line = ""
        in_multiline = False
        continuation_line = ""
        in_continuation = False

        for line in code.strip().splitlines():
            line_number += 1
            s = line.rstrip()

            # Skip empty lines or full-line comments
            if not s or s.startswith("'"):
                continue

            # Remove inline comments, respecting quoted strings
            in_quotes = False
            comment_start = -1
            i = 0
            while i < len(s):
                if s[i] == '"' and (i == 0 or s[i - 1] != '\\'):
                    in_quotes = not in_quotes
                elif s[i] == "'" and not in_quotes:
                    comment_start = i
                    break
                i += 1
            if comment_start != -1:
                s = s[:comment_start].rstrip()
            if not s:
                continue

            # Normalize INIT statements into LET assignments.
            init_match = re.match(r'^(\s*)init\b(.*)$', s, re.I)
            if init_match:
                leading_ws, rest = init_match.groups()
                s = f"{leading_ws}Let{rest}"

            # Normalize brackets like [ A 12 ] → [A12]
            s = re.sub(r'\[\s*([A-Z]+)\s+[A-Z]*(\d+)\s*\]', r'[\1\2]', s)

            if in_multiline:
                # Multiline string/push handling below.
                pass
            else:
                # Handle underscore line continuation (e.g., array literals)
                if in_continuation:
                    s = f"{continuation_line} {s.lstrip()}".rstrip()
                    if s.endswith('_'):
                        continuation_line = s[:-1].rstrip()
                        continue
                    continuation_line = ""
                    in_continuation = False
                elif s.endswith('_'):
                    continuation_line = s[:-1].rstrip()
                    in_continuation = True
                    continue

            # Handle start of type definition
            if s.lower().startswith("define "):
                parsed_name, parsed_parent, parsed_constraints = self._parse_type_header(
                    s, line_number)
                if parsed_name:
                    in_type_def = True
                    type_name = parsed_name
                    type_parent = parsed_parent
                    type_constraints = parsed_constraints or {}
                    type_def_lines = []
                    continue
                # Other definitions (functions/subprocesses) are handled later
                in_type_def = False

            # Handle lines inside a type definition block
            if in_type_def:
                end_pattern = rf'^\s*end(\s+type|\s+{re.escape(type_name)})\s*$'
                if re.match(end_pattern, s, re.I):
                    in_type_def = False
                    type_def = self._parse_type_def(
                        type_def_lines, line_number)
                    if type_parent:
                        type_def['_parent'] = type_parent
                    if type_constraints:
                        type_def['_constraints'] = type_constraints
                    self.types_defined[type_name.lower()] = type_def
                    continue
                type_def_lines.append(s.lstrip())
                continue

            # Handle multiline assignments (e.g., [^A1] := $" ... multiline ... ")
            if ':=' in s and s.startswith('[') and '$"' in s and not s.endswith('"'):
                current_line = s
                in_multiline = True
                continue
            elif in_multiline and not '.push(' in current_line.lower():
                current_line += "\n" + line.lstrip()
                if line.rstrip().endswith('"'):
                    lines.append((current_line, line_number))
                    current_line = ""
                    in_multiline = False
                continue

            # Handle multiline .push() method calls with string interpolation
            # But skip if this is a FOR loop with .push() on the same line
            if ('.push(' in s.lower() and '$"' in s and not s.endswith('"') and
                    not (s.lower().startswith('for ') and ' do ' in s.lower())):
                current_line = s
                in_multiline = True
                continue
            elif in_multiline:
                current_line += "\n" + line.lstrip()
                if '"' in line:
                    # Find the closing parenthesis and add it back
                    if not current_line.endswith(')'):
                        current_line += ')'
                    lines.append((current_line, line_number))
                    current_line = ""
                    in_multiline = False
                else:
                    pass
                continue

            # Collect dim declarations separately
            if s.startswith(':') and 'dim' in s.lower():
                dim_lines.append((s, line_number))

            # Collect label lines separately
            elif '!' in s and '.Label' in s:
                label_lines.append((s, line_number))

            # All other lines go into main lines
            else:
                lines.append((s, line_number))

        self._resolve_type_inheritance()
        return lines, label_lines, dim_lines

    def _parse_type_def(self, lines, line_number=None):
        """Delegate to type processor."""
        return self.type_processor._parse_type_def(lines, line_number)

    def _execute_type_code(self, code_lines, var_name, value_dict, line_number, input_values=None):
        """Delegate to type processor."""
        return self.type_processor._execute_type_code(code_lines, var_name, value_dict, line_number, input_values)

    def _process_grid_assignment(self, line, var_name, value_dict, line_number):
        """Delegate to type processor."""
        return self.type_processor._process_grid_assignment(line, var_name, value_dict, line_number)

    def _process_type_for_loop(self, loop_line, all_lines, var_name, value_dict, line_number):
        """Delegate to type processor."""
        return self.type_processor._process_type_for_loop(loop_line, all_lines, var_name, value_dict, line_number)

    def _process_type_let_statement(self, line, var_name, value_dict, line_number):
        """Delegate to type processor."""
        return self.type_processor._process_type_let_statement(line, var_name, value_dict, line_number)

    def _process_declarations_and_labels(self, lines, label_lines, dim_lines):
        for line, line_number in dim_lines:
            self._collect_global_declarations(line, line_number)

        # Track block depth to avoid treating nested ':' declarations as global
        depth = 0
        type_depth = 0
        for line, line_number in lines:
            lstrip_line = line.lstrip()
            stripped = lstrip_line.strip().lower()
            is_block_start = (
                (stripped.startswith('if ') and stripped.endswith('then')) or
                (stripped.startswith('for ') and stripped.endswith('do')) or
                (stripped.startswith('while ') and stripped.endswith('do'))
            )
            is_end = stripped == 'end'

            # Track entering/exiting type definitions to avoid treating inner lines as globals
            if stripped.startswith("define ") and " as type" in stripped:
                type_depth += 1
            elif stripped.startswith("end") and type_depth > 0:
                type_depth -= 1
                continue

            if depth == 0 and type_depth == 0:
                if (lstrip_line.startswith(':') and not lstrip_line.lower().startswith(("for ", "let "))) or stripped.startswith(("input ", "output ")):
                    self._collect_global_declarations(lstrip_line, line_number)

            if is_block_start:
                depth += 1
            elif is_end and depth > 0:
                depth -= 1

        for line, line_number in label_lines:
            self._process_label_assignment(line, line_number)
        for line, line_number in lines:
            self._process_cell_binding_declaration(line, line_number)
        for line, line_number in lines:
            if not line.startswith(':') and ':=' not in line and '!' not in line:
                self._evaluate_cell_var_definition(line, line_number)

    def _collect_global_declarations(self, line, line_number=None):

        # Handle INPUT and OUTPUT declarations that don't start with ':'
        line_stripped = line.lstrip()
        if line_stripped.lower().startswith(("input ", "output ")):
            a = line_stripped.strip()
        else:
            a = line_stripped[1:].strip()

        parsed = None

        # Immediately fail on simple self-referential assignments like "x = x"
        m_self = re.match(r'^([\w_]+)\s*=\s*\1\s*$', a, re.I)
        if m_self:
            raise ValueError(
                f"Self-referential assignment '{m_self.group(1)} = {m_self.group(1)}' at line {line_number}")

        # Handle INPUT declarations
        if a.lower().startswith('input '):
            var, type_name, constraints, expr = self.parser._parse_variable_def(
                a, line_number)
            constraints = constraints or {}
            default_value = expr
            var_names = constraints.pop('var_list', [var])
            for var_name in var_names:
                effective_type = 'array' if constraints.get(
                    'dim') else (type_name or None)
                self.current_scope().define_input(
                    var_name, effective_type, default_value, line_number, constraints)
            return

        # Handle OUTPUT declarations
        m_output = re.match(
            r'^OUTPUT\s+(.+)$', a, re.I | re.S)
        if m_output:
            def_str = m_output.group(1).strip()
            var, type_name, constraints, expr = self.parser._parse_variable_def(
                def_str, line_number)
            constraints = constraints or {}
            constraints['output'] = True
            if expr is not None and 'init' not in constraints:
                constraints['init'] = expr
                expr = None
            parsed = (var, type_name or 'text', constraints, expr)

        if parsed:
            var, type_name, constraints, expr = parsed
            self.current_scope().define_output(
                var, type_name, line_number, constraints)
            if constraints.get('init') is not None:
                init_expr = constraints.pop('init')
                deps = set(self._extract_identifier_tokens(init_expr))
                func_names = set(getattr(self, 'functions', {}).keys())
                deps = {d for d in deps if d.lower() not in func_names}
                unresolved = False
                for dep in deps:
                    dep_scope = self.current_scope().get_defining_scope(dep)
                    if dep_scope and dep_scope.is_uninitialized(dep):
                        unresolved = True
                        break
                    try:
                        self.current_scope().get(dep)
                    except Exception:
                        unresolved = True
                        break
                if unresolved:
                    self.pending_assignments[var] = (
                        init_expr, line_number, deps)
                else:
                    eval_scope = self.current_scope().get_full_scope()
                    import copy
                    init_val = self.expr_evaluator.eval_expr(
                        init_expr, eval_scope, line_number)
                    init_val = copy.deepcopy(init_val)
                    self.current_scope().update(var, init_val, line_number)
            return

        var_def, expr = self._split_assignment_expr(a)
        if not parsed and expr is None:
            try:
                var, type_name, constraints, expr = self._parse_variable_def(
                    a, line_number)
            except Exception:
                var = None
            if var and (type_name or constraints):
                constraints = constraints or {}
                effective_type = 'array' if constraints.get(
                    'dim') else (type_name or 'unknown')
                self.current_scope().types.setdefault(var, effective_type)
                if constraints.get('dim'):
                    dims = constraints['dim']
                    if isinstance(dims, dict) and 'dims' in dims:
                        dims = dims['dims']
                    if isinstance(dims, list):
                        self.dimensions[var] = dims
                        self.dim_names[var] = {
                            name: idx for idx, (name, _) in enumerate(dims) if name}
                        self.dim_labels[var] = {}
                if constraints.get('dim') and isinstance(constraints.get('dim'), list):
                    if any(isinstance(size_spec, str) for _, size_spec in constraints['dim']):
                        self.current_scope().define(
                            var, None, effective_type, constraints, is_uninitialized=True, line_number=line_number)
                        return
                    shape = []
                    for _, size_spec in constraints['dim']:
                        if isinstance(size_spec, tuple):
                            start, end = size_spec
                            size = end - start + 1
                        elif size_spec is None:
                            size = 1
                        else:
                            size = size_spec
                        shape.append(size)
                    pa_type = pa.float64() if effective_type in (
                        'number', 'array') else pa.string()
                    value = self.array_handler.create_array(
                        shape, 0 if effective_type in ('number', 'array') else '', pa_type, line_number)
                    self.current_scope().define(
                        var, value, effective_type, constraints, is_uninitialized=False, line_number=line_number)
                else:
                    self.current_scope().define(
                        var, None, effective_type, constraints, is_uninitialized=True, line_number=line_number)
                return
        m_new = re.match(
            r'^([\w_]+)\s*=\s*new\s+(\w+)\s*(\{|\()(.*)$', a, re.I)
        if m_new:
            var, type_name, opener, remainder = m_new.groups()
            if type_name.lower() not in self.types_defined:
                raise SyntaxError(
                    f"Type '{type_name}' not defined at line {line_number}")
            pairs = {'{': '}', '(': ')'}
            closer = pairs[opener]

            # Locate the argument segment, honoring nested delimiters
            start_pos = a.find(opener, m_new.start(3))
            stack = [closer]
            values_str = None
            trailing = ""
            for i, ch in enumerate(a[start_pos + 1:], start_pos + 1):
                if ch in pairs:
                    stack.append(pairs[ch])
                elif stack and ch == stack[-1]:
                    stack.pop()
                    if not stack:
                        values_str = a[start_pos + 1:i]
                        trailing = a[i + 1:].strip()
                        break
                elif ch in pairs.values():
                    raise SyntaxError(
                        f"Mismatched delimiter in constructor at line {line_number}: {a}")

            if values_str is None:
                raise SyntaxError(
                    f"Unclosed constructor for '{type_name}' at line {line_number}: {a}")
            with_assignments = {}
            if trailing:
                if trailing.lower().startswith('with'):
                    with_assignments = self._parse_with_clause(
                        trailing, line_number=line_number)
                else:
                    raise SyntaxError(
                        f"Unexpected characters after constructor at line {line_number}: {trailing}")

            def _split_args(arg_text):
                args = []
                current = ""
                nest_stack = []
                for ch in arg_text + ',':
                    if ch == ',' and not nest_stack:
                        if current.strip():
                            args.append(current.strip())
                        current = ""
                        continue
                    current += ch
                    if ch in pairs:
                        nest_stack.append(pairs[ch])
                    elif nest_stack and ch == nest_stack[-1]:
                        nest_stack.pop()
                    elif ch in pairs.values():
                        raise SyntaxError(
                            f"Mismatched delimiter in constructor arguments at line {line_number}: {a}")
                if nest_stack:
                    raise SyntaxError(
                        f"Unbalanced constructor arguments at line {line_number}: {a}")
                return [arg for arg in args if arg.strip()]

            values = _split_args(values_str)
            type_fields = self.types_defined[type_name.lower()]
            actual_fields = self._get_public_type_fields(type_fields)
            inputs_list = type_fields.get('_inputs', [])

            expected_args = len(
                inputs_list) if inputs_list else len(actual_fields)

            if not values and values_str.strip() == '':
                value_dict = self._instantiate_type(
                    type_name, [], line_number, allow_default_if_empty=True, var_name=var)
                if with_assignments:
                    value_dict = self._apply_with_constraints(
                        value_dict,
                        with_assignments,
                        self.current_scope().get_full_scope(),
                        line_number,
                        type_name=type_name)
                self.current_scope().define(var, value_dict, type_name)
            else:
                all_literals = all(re.match(r'^-?\d*\.?\d+$|^\".*\"$', v)
                                   for v in values)
                evaluated_args = [self.expr_evaluator.eval_expr(
                    value, self.current_scope().get_full_scope(), line_number) for value in values]
                value_dict = self._instantiate_type(
                    type_name, evaluated_args, line_number, allow_default_if_empty=False, var_name=var)
                if with_assignments:
                    value_dict = self._apply_with_constraints(
                        value_dict,
                        with_assignments,
                        self.current_scope().get_full_scope(),
                        line_number,
                        type_name=type_name)
                self.current_scope().define(var, value_dict, type_name)
                if not all_literals:
                    deps = self._extract_identifier_tokens(values_str)
                    if var in deps:
                        raise ValueError(
                            f"Self-referential assignment '{var} = new {type_name}{{{values_str}}}' at line {line_number}")
                    self.pending_assignments[var] = (
                        f"new {type_name}{{{values_str}}}", line_number, deps)
                # End constructor handling
            # After processing a constructor-style declaration, we're done
            return

        # Fallback: handle INIT-only declarations (no '=')
        try:
            var, type_name, constraints, value = self._parse_variable_def(
                a, line_number)
        except Exception:
            var = None
        if var and constraints.get('init') is not None:
            # Leave INIT for runtime evaluation to preserve execution order.
            self.current_scope().types.setdefault(var, type_name or 'unknown')
            existing_scope = self.current_scope().get_defining_scope(var)
            if (not existing_scope) or (var not in existing_scope.variables):
                self.current_scope().define(var, None, type_name or 'unknown',
                                            constraints, is_uninitialized=True)
            else:
                existing_scope.constraints[existing_scope._get_case_insensitive_key(
                    var, existing_scope.constraints) or var] = constraints
            return

        if expr is not None:
            var_def, expr = map(str.strip, (var_def, expr))
            var, type_name, constraints, value = self._parse_variable_def(
                var_def, line_number)
            dim_spec = constraints.get('dim')
            if dim_spec:
                dims = dim_spec
                if isinstance(dims, dict) and 'dims' in dims:
                    dims = dims['dims']
                if isinstance(dims, list):
                    self.dimensions[var] = dims
                    self.dim_names[var] = {
                        name: idx for idx, (name, _) in enumerate(dims) if name}
                    self.dim_labels[var] = {}
            if value is not None:
                constraints['constant'] = expr
            elif expr:  # Store constraint expression even when value is None
                constraints['constant'] = expr
            self.current_scope().types.setdefault(var, type_name or 'unknown')

            # Try to evaluate simple literals immediately
            evaluated_value = None
            is_uninitialized = True
            if expr:
                base_expr, with_text = self._split_new_with_expr(expr)
                if with_text:
                    with_assignments = self._parse_with_clause(
                        with_text, line_number)
                    deps = set()
                    for value_expr in with_assignments.values():
                        deps |= self._extract_identifier_tokens(value_expr)
                else:
                    # Extract dependencies, but exclude quoted strings and numeric literals
                    deps = set()
                    interpolation_only = False
                    if expr.strip().startswith('$"') and expr.strip().endswith('"'):
                        for match in re.finditer(r'\{([^{}]*)\}', expr):
                            deps |= self._extract_identifier_tokens(
                                match.group(1))
                        interpolation_only = True
                    if not interpolation_only:
                        # Skip quoted strings when looking for dependencies
                        expr_no_quotes = re.sub(r'"[^"]*"', '', expr)
                        expr_no_quotes = re.sub(r"'[^']*'", '', expr_no_quotes)

                        # Strip numeric literals (including scientific notation) before tokenizing
                        expr_no_numbers = re.sub(
                            r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?',
                            ' ', expr_no_quotes, flags=re.I)
                        # Strip cell references/ranges like [A1] or [A1:D1]
                        expr_no_numbers = re.sub(
                            r'\[[^\]]*\]', ' ', expr_no_numbers)
                        # Find potential dependencies, but filter out numeric literals and built-in functions
                        potential_deps = re.findall(
                            r'\b[\w_]+\b', expr_no_numbers)
                        built_in_functions = {
                            'sum', 'rows', 'sqrt', 'min', 'max', 'abs', 'int', 'float', 'str', 'len',
                            'to', 'step', 'by', 'mod', 'div', 'and', 'or', 'not', 'new'}
                        known_funcs = set(
                            getattr(self, 'functions', {}).keys())
                        known_subs = set(
                            getattr(self, 'subprocesses', {}).keys())
                        known_types = set(
                            getattr(self, 'types_defined', {}).keys())
                        member_suffixes = {name.split(
                            '.', 1)[1] for name in known_funcs if '.' in name}
                        deps = set()
                        for dep in potential_deps:
                            # Skip if it's a numeric literal (including negative numbers)
                            if re.match(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$', dep, re.I) or re.match(r'^e[+-]?\d+$', dep, re.I):
                                continue
                            dep_lower = dep.lower()
                            if dep_lower in built_in_functions:
                                continue
                            if dep_lower in known_funcs or dep_lower in known_subs or dep_lower in known_types:
                                continue
                            if dep_lower in member_suffixes:
                                continue
                            deps.add(dep)

                if var in deps:
                    raise ValueError(
                        f"Self-referential assignment '{var} = {expr}' at line {line_number}")

                # If no dependencies and it's a simple literal, evaluate immediately
                is_simple_literal = (expr.startswith('"') and expr.endswith('"') or
                                     expr.startswith("'") and expr.endswith("'") or
                                     expr.replace('.', '').replace('-', '').isdigit() or
                                     (expr.startswith('{') and expr.endswith('}') and
                                     all(item.strip().replace('-', '').replace('.', '').isdigit()
                                         for item in expr[1:-1].split(','))))
                if not deps and is_simple_literal:
                    try:
                        evaluated_value = self.expr_evaluator.eval_expr(
                            expr, self.current_scope().get_evaluation_scope(), line_number)
                        if constraints.get('with'):
                            evaluated_value = self._apply_with_constraints(
                                evaluated_value,
                                constraints.get('with', {}),
                                self.current_scope().get_full_scope(),
                                line_number,
                                type_name=type_name)
                        is_uninitialized = False
                    except Exception as e:
                        self.pending_assignments[var] = (
                            expr, line_number, deps, constraints)
                else:
                    self.pending_assignments[var] = (
                        expr, line_number, deps, constraints)

            self.current_scope().define(var, evaluated_value, type_name or 'unknown',
                                        constraints, is_uninitialized=is_uninitialized)
            return
        raise SyntaxError(
            f"Invalid global definition syntax: {line} at line {line_number}")

    def _parse_dim_size(self, size_str, line_number=None):
        """Delegate to parser."""
        return self.parser._parse_dim_size(size_str, line_number)

    def _process_label_assignment(self, line, line_number=None):
        m = re.match(
            r'^([\w_]+)!(\w+)\.Label\s*\{\s*([^}]*)\s*\}$', line, re.I)
        if m:
            var_name, dim_name, labels_str = m.groups()
            if var_name not in self.dim_names:
                raise SyntaxError(
                    f"Variable '{var_name}' has no named dimensions at line {line_number}")
            if dim_name not in self.dim_names[var_name]:
                raise SyntaxError(
                    f"Dimension '{dim_name}' not found in variable '{var_name}' at line {line_number}")
            dim_idx = self.dim_names[var_name][dim_name]
            array = self.current_scope().get(var_name)
            expected_size = None
            if array is not None:
                shape = self.array_handler.get_array_shape(array, line_number)
                if dim_idx < len(shape):
                    expected_size = shape[dim_idx]
            if expected_size is None:
                dims = self.dimensions.get(var_name, [])
                if dim_idx < len(dims):
                    _, size_spec = dims[dim_idx]
                    if isinstance(size_spec, tuple):
                        start, end = size_spec
                        expected_size = end - start + 1
                    elif isinstance(size_spec, int):
                        expected_size = size_spec
            labels = [lbl.strip().strip('"')
                      for lbl in labels_str.split(',') if lbl.strip()]
            if expected_size is not None and len(labels) != expected_size:
                raise ValueError(
                    f"Number of labels ({len(labels)}) does not match dimension size ({expected_size}) at line {line_number}")
            self.array_handler.set_labels(
                var_name, dim_name, labels, line_number)
        else:
            raise SyntaxError(
                f"Invalid label assignment syntax: {line} at line {line_number}")

    def _evaluate_cell_var_definition(self, line, line_number=None):
        m = re.match(r'^\[\s*\^?([A-Z]+\d+)\s*\]\s*:\s*(.+)$', line, re.S)
        if not m:
            return
        cell, rhs = map(str.strip, m.groups())
        if not re.match(r'^[A-Za-z]+\d+$', cell):
            raise ValueError(
                f"Invalid cell reference '{cell}' at line {line_number}")
        var_def, expr = self._split_assignment_expr(rhs)
        if expr is None:
            return
        var, type_name, constraints, _ = self._parse_variable_def(
            var_def, line_number)
        if not re.match(r'^[\w_]+$', var):
            raise SyntaxError(
                f"Invalid variable name: '{var}' at line {line_number}")
        if cell in self._cell_var_map and self._cell_var_map[cell] != var:
            raise SyntaxError(
                f"Cell '{cell}' already mapped to '{self._cell_var_map[cell]}' at line {line_number}")
        for c, v in self._cell_var_map.items():
            if v == var and c != cell:
                raise SyntaxError(
                    f"Variable '{var}' already mapped to cell '{c}' at line {line_number}")

        if expr is not None and 'constant' not in constraints:
            constraints['constant'] = expr
        value = self.expr_evaluator.eval_or_eval_array(
            expr, self.current_scope().get_full_scope(), line_number)
        value = self.array_handler.check_dimension_constraints(
            var, value, line_number)
        if constraints.get('with'):
            value = self._apply_with_constraints(
                value, constraints.get('with', {}),
                self.current_scope().get_full_scope(), line_number,
                type_name=type_name)
        inferred_type = type_name or self.array_handler.infer_type(
            value, line_number)
        if inferred_type == 'int':
            inferred_type = 'number'
        defining_scope = self.current_scope().get_defining_scope(var)
        if defining_scope:
            if constraints:
                defining_scope.constraints[var] = constraints
            defining_scope.update(var, value, line_number)
        else:
            self.current_scope().define(
                var, value, inferred_type, constraints, is_uninitialized=False, line_number=line_number)
        self.grid[cell] = value.to_pylist() if isinstance(
            value, pa.Array) else value
        self._cell_var_map[cell] = var

    def _process_cell_binding_declaration(self, line, line_number=None):
        """Process lines like [A1]: width as number to bind variables to cells."""
        m = re.match(r'^\[\s*\^?([A-Za-z]+\d+)\s*\]\s*:\s*(.+)$', line, re.I)
        if not m:
            return False
        cell_ref, rhs = m.groups()
        cell_ref = cell_ref.upper()
        rhs = rhs.strip()
        validate_cell_ref(cell_ref)

        # If this is an assignment, let the assignment handler manage it.
        var_def, expr = self._split_assignment_expr(rhs)
        if expr is not None:
            return False

        var, type_name, constraints, _ = self.parser._parse_variable_def(
            var_def.strip(), line_number)
        constraints = constraints or {}
        if not re.match(r'^[\w_]+$', var):
            raise SyntaxError(
                f"Invalid variable name: '{var}' at line {line_number}")

        # Ensure variable exists (case-insensitive) with the right type metadata.
        defining_scope = self.current_scope().get_defining_scope(var)
        if not defining_scope:
            self.current_scope().define(
                var, None, type_name, constraints, is_uninitialized=True)
        else:
            actual_key = defining_scope._get_case_insensitive_key(
                var, defining_scope.constraints) or var
            defining_scope.constraints[actual_key] = constraints
            if type_name:
                type_key = defining_scope._get_case_insensitive_key(
                    var, defining_scope.types) or var
                defining_scope.types[type_key] = type_name

        # Prevent conflicting mappings.
        existing = self._cell_var_map.get(cell_ref)
        if existing and existing.lower() != var.lower():
            raise SyntaxError(
                f"Cell '{cell_ref}' already mapped to '{existing}' at line {line_number}")
        for mapped_cell, mapped_var in self._cell_var_map.items():
            if mapped_var.lower() == var.lower() and mapped_cell != cell_ref:
                raise SyntaxError(
                    f"Variable '{var}' already mapped to cell '{mapped_cell}' at line {line_number}")

        self._cell_var_map[cell_ref] = var

        # If the variable already has a value, reflect it in the grid immediately.
        try:
            current_value = self.current_scope().get(var)
            if current_value is not None:
                converted = current_value.to_pylist() if isinstance(
                    current_value, pa.Array) else current_value
                self.grid[cell_ref] = converted
                self._record_output_value(var, converted)
        except NameError:
            pass
        return True

    def _sync_cell_bindings(self, var_name, value):
        """Propagate variable updates to any bound cells."""
        if value is None:
            return
        var_lower = var_name.lower()
        for cell, mapped_var in self._cell_array_map.items():
            if mapped_var.lower() == var_lower:
                try:
                    self.array_handler._assign_horizontal_array(
                        cell, value, "{}", line_number=None)
                except Exception:
                    self.grid[cell] = value
        for cell, mapped_var in self._cell_var_map.items():
            if mapped_var.lower() == var_lower:
                converted = value.to_pylist() if isinstance(
                    value, pa.Array) else value
                self.grid[cell] = converted

    def _record_output_value(self, var_name, value):
        """Record values for declared output variables."""
        if value is None:
            return
        global_scope = self.get_global_scope()
        if not global_scope.is_output(var_name):
            return
        var_key = var_name.lower()
        if isinstance(value, pa.Array):
            value = value.to_pylist()
        # Keep all pushed values in order
        existing = self.output_values.get(var_key, [])
        if not isinstance(existing, list):
            existing = [] if existing is None else [existing]
        existing.append(value)
        self.output_values[var_key] = existing

    def truncate_output(self, output_dict, max_length=100):
        """Truncate long test output and add '...' if needed"""
        output_str = str(output_dict)
        if len(output_str) <= max_length:
            return output_str
        return output_str[:max_length-3] + "..."

    def _is_keyword(self, line, keyword):
        """Case-insensitive keyword check"""
        return line.strip().lower() == keyword.lower()

    def _starts_with_keyword(self, line, keyword):
        """Case-insensitive keyword start check"""
        return line.strip().lower().startswith(keyword.lower())

    def _ends_with_keyword(self, line, keyword):
        """Case-insensitive keyword end check"""
        return line.strip().lower().endswith(keyword.lower())

    def get_global_scope(self):
        return self.scopes[0]

    def set_input_values(self, args, prompt_missing=False):
        """Set input values from args/defaults. Optionally prompt for missing ones."""
        global_scope = self.get_global_scope()
        if args is None:
            args = []
        can_prompt = prompt_missing and sys.stdin and sys.stdin.isatty()

        for i, input_var in enumerate(self.input_variables):
            value_assigned = False
            provided_arg = i < len(args)
            if provided_arg:
                value = args[i]
                actual_key = global_scope._get_case_insensitive_key(
                    input_var, global_scope.types) or input_var
                var_type = global_scope.types.get(actual_key, 'text')
                constraints = global_scope.constraints.get(actual_key, {})
                comparison_keys = ('<', '<=', '>', '>=', '<>')
                type_union = constraints.get('type_union')
                union_allows_text = isinstance(
                    type_union, (list, tuple, set)) and 'text' in type_union
                union_allows_number = isinstance(
                    type_union, (list, tuple, set)) and 'number' in type_union
                union_number_only = union_allows_number and not union_allows_text
                needs_number = union_number_only or (var_type == 'number' and not union_allows_text) or any(
                    key in constraints for key in comparison_keys + ('range',)) or any(
                    key.startswith('not_') and key[4:] in comparison_keys for key in constraints)
                not_type = constraints.get('not_type')
                if needs_number:
                    try:
                        value = float(value)
                    except ValueError:
                        print(
                            f"Warning: Could not convert '{value}' to number for input '{input_var}', skipping provided arg")
                        value = None
                elif union_allows_text and union_allows_number:
                    try:
                        value = float(value)
                    except ValueError:
                        value = value
                elif not_type == 'text':
                    try:
                        value = float(value)
                    except ValueError:
                        value = value
                elif isinstance(value, str) and not type_union and constraints.get('type') is None and not any(
                        key in constraints for key in comparison_keys + ('range',)):
                    try:
                        value = float(value)
                    except ValueError:
                        value = value
                if value is not None:
                    global_scope.update(input_var, value)
                    value_assigned = True

            default_expr = global_scope.constraints.get(
                input_var, {}).get('default')
            if (not value_assigned) and default_expr is not None:
                try:
                    value = self.expr_evaluator.eval_expr(
                        str(default_expr), global_scope.get_evaluation_scope())
                    global_scope.update(input_var, value)
                    value_assigned = True
                except Exception as exc:
                    print(
                        f"Warning: Failed to evaluate default for input '{input_var}': {exc}")

            # Prompt if allowed and still unset
            if not value_assigned and prompt_missing:
                if can_prompt:
                    try:
                        self._prompt_for_input(input_var, global_scope)
                    except RuntimeError as exc:
                        print(f"Warning: {exc}")
                        break
                else:
                    pass

            actual_key = global_scope._get_case_insensitive_key(
                input_var, global_scope.variables) or input_var
            current_value = global_scope.variables.get(actual_key)
            if isinstance(current_value, str):
                constraints = global_scope.constraints.get(actual_key, {})
                comparison_keys = ('<', '<=', '>', '>=', '<>')
                if constraints.get('type') is None and not any(
                        key in constraints for key in comparison_keys + ('range',)):
                    try:
                        coerced = float(current_value)
                        global_scope.update(input_var, coerced)
                    except ValueError:
                        pass

    def _prompt_for_input(self, input_var, global_scope):
        """Prompt user for input value and set it in the global scope"""
        actual_key = global_scope._get_case_insensitive_key(
            input_var, global_scope.types) or input_var
        var_type = global_scope.types.get(actual_key, 'text')
        constraints = global_scope.constraints.get(actual_key, {})
        comparison_keys = ('<', '<=', '>', '>=', '<>')
        type_union = constraints.get('type_union')
        union_allows_text = isinstance(
            type_union, (list, tuple, set)) and 'text' in type_union
        union_allows_number = isinstance(
            type_union, (list, tuple, set)) and 'number' in type_union
        union_number_only = union_allows_number and not union_allows_text
        needs_number = union_number_only or (var_type == 'number' and not union_allows_text) or any(
            key in constraints for key in comparison_keys + ('range',)) or any(
            key.startswith('not_') and key[4:] in comparison_keys for key in constraints)
        if not (sys.stdin and sys.stdin.isatty()):
            raise RuntimeError(
                f"Cannot prompt for input '{input_var}' (no interactive input available). "
                "Please supply arguments when running the program."
            )

        while True:
            try:
                not_type = constraints.get('not_type')
                if needs_number:
                    user_input = input(f"{input_var}: ")
                    value = float(user_input)
                elif union_allows_text and union_allows_number:
                    user_input = input(f"{input_var}: ")
                    try:
                        value = float(user_input)
                    except ValueError:
                        value = user_input
                elif not_type == 'text':
                    user_input = input(f"{input_var}: ")
                    try:
                        value = float(user_input)
                    except ValueError:
                        value = user_input
                else:
                    user_input = input(f"{input_var}: ")
                    if constraints.get('type') is None and not any(
                            key in constraints for key in comparison_keys + ('range',)):
                        try:
                            value = float(user_input)
                        except ValueError:
                            value = user_input
                    else:
                        value = user_input

                global_scope.update(input_var, value)
                break

            except EOFError:
                raise RuntimeError(
                    f"Cannot prompt for input '{input_var}' (no interactive input available). "
                    "Please supply arguments when running the program."
                )
            except ValueError:
                if union_allows_text and union_allows_number:
                    display_type = 'text or number'
                else:
                    display_type = 'number' if needs_number else (
                        var_type or 'value')
                print(f"Invalid input. Please enter a valid {display_type}.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(1)

    def collect_input_output_variables(self):
        """Collect all input and output variables from the current scope"""
        global_scope = self.get_global_scope()
        self.input_variables = list(global_scope.input_variables)
        self.output_variables = list(global_scope.output_variables)
