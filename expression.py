# expression.py
# This module defines the ExpressionEvaluator class, responsible for evaluating expressions,
# arrays, ranges, and special functions in the GridLang compiler. It handles various syntax
# including binary operations, dimension constraints, sums, interpolations, and more.

import re
import math
import random
import pyarrow as pa
# Utility for validating cell references
from utils import validate_cell_ref, col_to_num, num_to_col, object_public_keys


class CaseInsensitiveDict(dict):
    """Dictionary with case-insensitive string key lookup, preserving original keys."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._keymap = {}
        if args or kwargs:
            self.update(*args, **kwargs)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self._keymap[key.lower()] = key
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, str):
            key = self._keymap.get(key.lower(), key)
        return super().__getitem__(key)

    def __contains__(self, key):
        if isinstance(key, str):
            return key.lower() in self._keymap
        return super().__contains__(key)

    def get(self, key, default=None):
        if isinstance(key, str):
            key = self._keymap.get(key.lower(), key)
        return super().get(key, default)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value


class ExpressionEvaluator:
    """
    Evaluator for expressions, arrays, and special constructs in GridLang.
    Handles scalar expressions, array operations, range evaluations, interpolations,
    and dimension constraints.
    """

    def __init__(self, compiler):
        """
        Initialize the evaluator with a reference to the compiler.
        :param compiler: The GridLangCompiler instance.
        """
        self.compiler = compiler
        self.dim_ranges = {}  # Stores dimension ranges for variables

    def eval_or_eval_array(self, expr, scope, line_number=None, is_grid_dim=False):
        """
        Evaluate an expression that could be a scalar, array, range, sum, or grid dimension.
        Handles dimension constraints, reshaping, and grid indexing if specified.
        :param expr: The expression string to evaluate.
        :param scope: The current scope dictionary.
        :param line_number: Optional line number for error reporting.
        :param is_grid_dim: Use grid dimension logic if True.
        :return: Evaluated value (scalar, list, or pyarrow Array).
        """
        if isinstance(expr, list):  # Skip if already a dimension constraint list
            return expr
        expr = expr.strip()
        if is_grid_dim and '--' in expr:
            expr = expr.split('--')[0].strip()
        if is_grid_dim and expr.startswith('"') and expr.endswith('"'):
            return expr[1:-1]

        # Check for binary array operations
        binary_op_match = re.match(
            r'^\s*(\{[^}]*\})\s*([+\-*/^\\]|mod)\s*(\{[^}]*\})\s*$', expr, re.I)
        if binary_op_match and not is_grid_dim:
            return self.eval_expr(expr, scope, line_number)

        # Handle grid indexing (e.g., var.grid{1,2})
        if is_grid_dim and re.match(r'^([\w_]+)\.grid\{([\d,\s]+)\}$', expr, re.I):
            var_name, indices_str = re.match(
                r'^([\w_]+)\.grid\{([\d,\s]+)\}$', expr, re.I).groups()
            indices = [int(i.strip()) for i in indices_str.split(',')]
            if isinstance(scope, dict):
                scope_lookup = scope
            elif hasattr(scope, 'get_evaluation_scope'):
                scope_lookup = scope.get_evaluation_scope()
            elif hasattr(scope, 'get_full_scope'):
                scope_lookup = scope.get_full_scope()
            else:
                scope_lookup = {}
            tensor = scope_lookup.get(var_name)
            if tensor is None:
                tensor = scope_lookup.get(
                    var_name.lower(), scope_lookup.get(var_name.upper()))
            if tensor is None and hasattr(self.compiler, 'current_scope'):
                try:
                    tensor = self.compiler.current_scope().get(var_name)
                except Exception:
                    tensor = None
            if tensor is not None:
                if isinstance(tensor, dict) and 'grid' in tensor and 'original_shape' in tensor:
                    array = tensor['grid']
                    original_shape = tensor['original_shape']

                    # Convert 1-based indices to 0-based indices
                    adjusted_indices = [idx - 1 for idx in indices]

                    try:
                        result = self.compiler.array_handler.get_array_element(
                            array, adjusted_indices, line_number, original_shape=original_shape)
                        return result
                    except (IndexError, ValueError) as e:
                        raise IndexError(
                            f"Invalid indices {indices_str} for '{var_name}.grid': {e} at line {line_number}")
                raise TypeError(
                    f"Variable '{var_name}' does not have a 'grid' or 'original_shape' field at line {line_number}")

        # Handle field access (e.g., var.field)
        if is_grid_dim and re.match(r'^[\w_]+\.\w+$', expr):
            var, field = expr.split('.')
            if isinstance(scope, dict):
                scope_lookup = scope
            elif hasattr(scope, 'get_evaluation_scope'):
                scope_lookup = scope.get_evaluation_scope()
            elif hasattr(scope, 'get_full_scope'):
                scope_lookup = scope.get_full_scope()
            else:
                scope_lookup = {}
            if f"{var}.{field}" in scope_lookup:
                return scope_lookup[f"{var}.{field}"]
            tensor = scope_lookup.get(var)
            if tensor is None:
                tensor = scope_lookup.get(
                    var.lower(), scope_lookup.get(var.upper()))
            if tensor is None and hasattr(self.compiler, 'current_scope'):
                try:
                    tensor = self.compiler.current_scope().get(var)
                except Exception:
                    tensor = None
            if tensor is not None:
                if isinstance(tensor, dict) and field in tensor:
                    return tensor[field]
                if isinstance(tensor, pa.ListArray):
                    shape = self.compiler.array_handler.get_array_shape(
                        tensor, line_number)
                    zero_indices = [1] * len(shape)
                    first_element = self.compiler.array_handler.get_array_element(
                        tensor, zero_indices, line_number, return_struct=True)
                    if isinstance(first_element, dict) and field in first_element:
                        return first_element[field]

                raise NameError(
                    f"Cannot access field '{field}' of '{var}' at line {line_number}")

        # Handle dimension constraints (e.g., expr dim {dims})
        if 'dim' in expr.lower() and not is_grid_dim:
            m = re.match(r'^(.*)\s+dim\s*\{([^}]*)\}$', expr, re.I)
            if m:
                value_expr, dim_spec = m.groups()
                value = self.eval_or_eval_array(
                    value_expr.strip(), scope, line_number)
                new_dims = [self._parse_dim_size(
                    d.strip(), line_number) for d in dim_spec.split(',') if d.strip()]
                var_name = value_expr.strip() if value_expr.strip(
                ) in self.compiler.variables else None
                if var_name:
                    self.dim_ranges[var_name] = new_dims
                return self.compiler.array_handler.reshape_array(value, new_dims, line_number)

        # Evaluate inline arrays
        if expr.startswith('{') and expr.endswith('}') and not expr.startswith('{$"') and not is_grid_dim:
            result = self._evaluate_array(expr, scope, line_number)
            # Ensure array literals are returned as lists, not sets
            if isinstance(result, set):
                result = sorted(list(result))
            return result

        # Handle cell ranges (e.g., [A1:B2])
        range_match = re.match(
            r'^\s*\[\s*([A-Z]+\d+)\s*:\s*([A-Z]+\d+)\s*\]\s*$', expr)
        if range_match and not is_grid_dim:
            s_ref, e_ref = range_match.groups()
            try:
                validate_cell_ref(s_ref)
                validate_cell_ref(e_ref)
                values = self.compiler.array_handler.get_range_values(
                    s_ref, e_ref, line_number)
                flat_values = [v for row in values for v in (
                    row if isinstance(row, list) else [row])]
                if hasattr(self.compiler, 'last_dim_var') and self.compiler.last_dim_var:
                    var_name = self.compiler.last_dim_var
                    self.compiler.current_scope().update(var_name, flat_values)
                    self.compiler.last_dim_var = None
                return pa.array(flat_values, type=pa.float64())
            except Exception as e:
                raise RuntimeError(
                    f"Error evaluating range '{s_ref}:{e_ref}': {e} at line {line_number}")

        # Handle sum over range (e.g., sum[A1:B2])
        if expr.lower().startswith('sum[') and expr.endswith(']') and not is_grid_dim:
            return self._evaluate_sum_range(expr, scope, line_number)

        # Handle sum with parentheses (e.g., sum([A1:B2]))
        if expr.lower().startswith('sum([') and expr.endswith('])') and not is_grid_dim:
            inner_expr = expr[4:-2].strip()
            m = re.match(
                r'^\[?([A-Z]+\d+)\s*:\s*([A-Z]+\d+)\]?$', inner_expr, re.I)
            if m:
                start_ref, end_ref = m.groups()
                return self._evaluate_sum_range(f"sum[{start_ref}:{end_ref}]", scope, line_number)
            raise SyntaxError(
                f"Invalid sum range syntax in parentheses: {expr} at line {line_number}")

        # Handle sum over variables (e.g., sum{a, b})
        if expr.startswith('sum{') and expr.endswith('}') and not is_grid_dim:
            return self._evaluate_sum_vars(expr, scope, line_number)

        # Try to parse as number for grid_dim
        if is_grid_dim:
            try:
                return float(expr) if '.' in expr else int(expr)
            except ValueError:
                pass

        # Lookup in scope
        if expr in scope:
            return scope[expr]

        # Handle member calls on arrays (e.g., hand1.Name())
        if not is_grid_dim:
            m_member_array = re.match(r'^([\w_]+)\.([\w_]+)\((.*)\)$', expr)
            if m_member_array:
                obj_name, method_name, args_part = m_member_array.groups()
                obj_value = None
                try:
                    obj_value = scope.get(obj_name)
                except Exception:
                    try:
                        obj_value = self.compiler.current_scope().get(obj_name)
                    except Exception:
                        obj_value = None
                if isinstance(obj_value, (list, tuple, pa.Array)):
                    if isinstance(obj_value, pa.Array):
                        obj_list = obj_value.to_pylist()
                    else:
                        obj_list = list(obj_value)
                    args_list = []
                    if args_part.strip():
                        args_list = [a.strip()
                                     for a in re.split(r',(?![^{]*})', args_part) if a.strip()]
                    evaluated_args = [self.eval_or_eval_array(
                        a, scope, line_number) for a in args_list]
                    results = []
                    for elem in obj_list:
                        if not isinstance(elem, dict):
                            continue
                        element_type = elem.get('_type_name')
                        if not element_type:
                            for t_name, t_def in getattr(self.compiler, 'types_defined', {}).items():
                                public_fields = self.compiler._get_public_type_fields(
                                    t_def)
                                if public_fields and object_public_keys(elem) == set(public_fields.keys()):
                                    element_type = t_name
                                    break
                        if not element_type:
                            continue
                        func_key = self.compiler._resolve_member_function(
                            element_type, method_name)
                        if func_key and func_key in getattr(self.compiler, 'functions', {}):
                            results.append(self.compiler.call_function(
                                func_key, [elem] + evaluated_args, instance_type=element_type))
                    if results:
                        return results

            return self.eval_expr(expr, scope, line_number)

        raise NameError(f"Name '{expr}' not defined at line {line_number}")

    def _parse_dim_size(self, size_str, line_number=None):
        """
        Parse dimension size specification (e.g., '*', '1 to 10', or integer).
        :param size_str: Dimension string.
        :param line_number: Line number for errors.
        :return: Parsed size (None for '*', tuple for range, int for fixed).
        """
        size_str = size_str.strip()
        if size_str == '*':
            return None
        m = re.match(r'^(\d+)\s+to\s+(\d+)$', size_str)
        if m:
            start, end = map(int, m.groups())
            return (start, end)
        try:
            size = int(size_str)
            return (size, size)
        except ValueError:
            raise SyntaxError(
                f"Invalid dimension size: '{size_str}' at line {line_number}")

    def _evaluate_sum_range(self, expr, scope, line_number=None):
        """
        Evaluate sum over a cell range (e.g., sum[A1:B2]).
        :param expr: Sum expression.
        :param scope: Scope.
        :param line_number: Line number.
        :return: Sum of numeric values in range.
        """
        m = re.match(r'^sum\[([A-Z]+\d+)\s*:\s*([A-Z]+\d+)\]$', expr, re.I)
        if not m:
            raise SyntaxError(
                f"Invalid sum range syntax: {expr} at line {line_number}")
        start_ref, end_ref = m.groups()
        try:
            validate_cell_ref(start_ref)
            validate_cell_ref(end_ref)
            values = self.compiler.array_handler.get_range_values(
                start_ref, end_ref, line_number)
            flat_values = [v for row in values for v in (
                row if isinstance(row, list) else [row])]
            numeric_values = [
                float(v) for v in flat_values if isinstance(v, (int, float))]
            return sum(numeric_values)
        except Exception as e:
            raise RuntimeError(
                f"Error evaluating sum range '{expr}': {e} at line {line_number}")

    def _evaluate_sum_vars(self, expr, scope, line_number=None):
        """
        Evaluate sum over variables (e.g., sum{a, b}).
        :param expr: Sum expression.
        :param scope: Scope.
        :param line_number: Line number.
        :return: Sum of variable values (flattening arrays if needed).
        """
        inner = expr[4:-1].strip()
        if not inner:
            return 0
        var_names = [v.strip() for v in inner.split(',') if v.strip()]
        total = 0
        for var in var_names:
            if var not in scope:
                raise NameError(
                    f"Variable '{var}' not defined at line {line_number}")
            value = scope[var]
            if isinstance(value, (pa.Array, list)):
                value = sum(value.to_pylist() if isinstance(
                    value, pa.Array) else value)
            elif not isinstance(value, (int, float)):
                raise TypeError(
                    f"Cannot sum non-numeric value for '{var}' at line {line_number}")
            total += value
        return total

    def _evaluate_array(self, expr, scope, line_number=None):
        """
        Evaluate an inline array expression (e.g., {1,2;3,4}).
        Handles nested objects, ranges, and flattening.
        :param expr: Array string.
        :param scope: Scope.
        :param line_number: Line number.
        :return: List of lists representing the array.
        """
        # Use case-insensitive scope access
        cell_vars = {}
        for c, v in self.compiler._cell_var_map.items():
            try:
                cell_vars[v] = self.compiler.current_scope().get(v)
            except NameError:
                # Variable not found, skip it
                pass
        full_scope = {**scope, **cell_vars}

        # Final safety net for member calls on arrays before eval()
        m_member_array_fallback = re.match(
            r'^([\w_]+)\.([\w_]+)\((.*)\)$', expr)
        if m_member_array_fallback:
            obj_name, method_name, args_part = m_member_array_fallback.groups()
            obj_value = full_scope.get(obj_name)
            if isinstance(obj_value, (list, tuple, pa.Array)):
                if isinstance(obj_value, pa.Array):
                    obj_list = obj_value.to_pylist()
                else:
                    obj_list = list(obj_value)
                element_type = None
                for elem in obj_list:
                    if isinstance(elem, dict):
                        element_type = elem.get('_type_name')
                        if element_type:
                            break
                if element_type:
                    func_key = self.compiler._resolve_member_function(
                        element_type, method_name)
                    if func_key and func_key in getattr(self.compiler, 'functions', {}):
                        args_list = []
                        if args_part.strip():
                            args_list = [a.strip()
                                         for a in re.split(r',(?![^{]*})', args_part) if a.strip()]
                        evaluated_args = [self.eval_or_eval_array(
                            a, scope, line_number) for a in args_list]
                        return [self.compiler.call_function(
                            func_key, [elem] + evaluated_args, instance_type=element_type)
                            for elem in obj_list if isinstance(elem, dict)]
        if '|' in expr:
            return self._evaluate_pipe_array(expr, full_scope, line_number)
        inner = expr.strip()[1:-1].strip()
        if not inner:
            return []
        rows = inner.split(';')
        elements = []
        for row in rows:
            row_elements = []
            row_inner = row.strip()
            if row_inner:
                sub_elements = []
                current_element = ""
                brace_level = 0
                paren_level = 0
                in_quotes = False
                pipe_level = 0
                for char in row_inner + ',':
                    if char == '|' and brace_level == 0 and paren_level == 0 and not in_quotes:
                        pipe_level += 1
                    elif char == ',' and brace_level == 0 and paren_level == 0 and not in_quotes and pipe_level == 0:
                        if current_element.strip():
                            sub_elements.append(current_element.strip())
                        current_element = ""
                    else:
                        current_element += char
                        if char == '"':
                            in_quotes = not in_quotes
                        elif char == '{':
                            brace_level += 1
                        elif char == '}':
                            brace_level -= 1
                        elif char == '(':
                            paren_level += 1
                        elif char == ')':
                            paren_level -= 1
                sub_elements = [e for e in sub_elements if e.strip()]
                for elem_str in sub_elements:
                    try:
                        evaluated = self.eval_or_eval_array(
                            elem_str, full_scope, line_number)
                        if isinstance(evaluated, pa.Array):
                            row_elements.extend([float(v) if isinstance(
                                v, (int, float)) else v for v in evaluated.to_pylist()])
                        elif isinstance(evaluated, dict):
                            row_elements.append(evaluated)
                        elif isinstance(evaluated, list):
                            row_elements.extend([float(v) if isinstance(
                                v, (int, float)) else v for v in evaluated])
                        else:
                            row_elements.append(float(evaluated) if isinstance(
                                evaluated, (int, float)) else evaluated)
                    except Exception as e:
                        raise RuntimeError(
                            f"Error evaluating array element '{elem_str}': {e} at line {line_number}")
            elements.append(row_elements)
        if len(elements) > 1 or (elements and len(elements[0]) > 1 and ';' in inner):
            return elements
        return elements[0] if elements else []

    def _evaluate_pipe_array(self, expr, scope, line_number=None):
        """
        Evaluate piped arrays (e.g., {1,2} | {3,4}).
        Concatenates arrays along the outer dimension.
        :param expr: Piped array string.
        :param scope: Scope.
        :param line_number: Line number.
        :return: Pyarrow ListArray of concatenated values.
        """
        if not (expr.startswith('{') and expr.endswith('}')):
            raise SyntaxError(
                f"Invalid pipe array syntax: {expr} at line {line_number}")
        arrays = []
        current = ""
        brace_level = 0
        for char in expr:
            if char == '|' and brace_level == 0:
                if current.strip():
                    arrays.append('{' + current.strip() + '}')
                current = ""
            else:
                current += char
                if char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
        if current.strip():
            arrays.append('{' + current.strip() + '}')
        if brace_level != 0:
            raise SyntaxError(
                f"Unmatched braces in pipe array: {expr} at line {line_number}")
        if not arrays:
            raise ValueError(f"Empty pipe expression at line {line_number}")
        evaluated_arrays = []
        shapes = []
        for array_expr in arrays:
            if not (array_expr.startswith('{') and array_expr.endswith('}')):
                raise SyntaxError(
                    f"Invalid pipe array segment: {array_expr} at line {line_number}")
            array = self._evaluate_array(array_expr, scope, line_number)
            shape = self.compiler.array_handler.get_array_shape(
                array, line_number)
            shapes.append(shape)
            evaluated_arrays.append(array)
        dimensions = [len(shape) for shape in shapes]
        if len(set(dimensions)) != 1:
            raise ValueError(
                f"All arrays in pipe operation must have the same number of dimensions: {dimensions} at line {line_number}")
        num_dims = dimensions[0]
        if num_dims == 1:
            shapes = [(1, s[0]) if len(s) == 1 else s for s in shapes]
            inner_dim = shapes[0][1]
            if not all(s[1] == inner_dim for s in shapes):
                raise ValueError(
                    f"All arrays in pipe operation must have the same inner dimension: {shapes} at line {line_number}")
        elif num_dims == 2:
            inner_dim = shapes[0][1]
            if not all(s[1] == inner_dim for s in shapes):
                raise ValueError(
                    f"All arrays in pipe operation must have the same inner dimension: {shapes} at line {line_number}")
        else:
            raise ValueError(
                f"Pipe operator supports up to 2D arrays, got {num_dims}D at line {line_number}")
        flat_values = []
        row_lengths = []
        for arr in evaluated_arrays:
            arr_vals = arr.to_pylist() if isinstance(arr, pa.ListArray) else arr
            if isinstance(arr_vals, list) and arr_vals and isinstance(arr_vals[0], list):
                for row in arr_vals:
                    flat_values.extend([float(v) for v in row])
                    row_lengths.append(len(row))
            else:
                flat_values.extend([float(v) for v in arr_vals])
                row_lengths.append(len(arr_vals))
        offsets = [0]
        current_offset = 0
        for length in row_lengths:
            current_offset += length
            offsets.append(current_offset)
        offsets = pa.array(offsets, type=pa.int32())
        values = pa.array(flat_values, type=pa.float64())
        return pa.ListArray.from_arrays(offsets, values)

    def _resolve_column_interpolated_cell(self, inside, scope, line_number=None):
        m = re.match(
            r'^\{\s*([^}:]+?)\s*:\s*([A-Za-z]+)\s*\}\s*(\d+|\{[^}]+\})\s*$',
            inside)
        if not m:
            return None
        index_expr, base_col, row_part = m.groups()
        index_value = self.eval_expr(index_expr, scope, line_number)
        if isinstance(index_value, float) and index_value.is_integer():
            index_value = int(index_value)
        if not isinstance(index_value, int) or index_value < 1:
            raise ValueError(
                f"Invalid column index '{index_value}' must be a positive integer at line {line_number}")

        if row_part.startswith('{') and row_part.endswith('}'):
            row_expr = row_part[1:-1].strip()
            row_value = self.eval_expr(row_expr, scope, line_number)
        else:
            row_value = int(row_part)
        if isinstance(row_value, float) and row_value.is_integer():
            row_value = int(row_value)
        if not isinstance(row_value, int) or row_value < 1:
            raise ValueError(
                f"Invalid row index '{row_value}' must be a positive integer at line {line_number}")

        col_num = col_to_num(base_col) + index_value - 1
        cell_ref = f"{num_to_col(col_num)}{row_value}"
        validate_cell_ref(cell_ref)
        return cell_ref

    def _parse_indexed_member_call(self, expr):
        expr = expr.strip()
        base_match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\(', expr)
        if not base_match:
            return None
        base_name = base_match.group(1)
        start = base_match.end() - 1
        depth = 0
        in_quote = None
        end = None
        for pos in range(start, len(expr)):
            ch = expr[pos]
            if in_quote:
                if ch == in_quote and expr[pos - 1] != '\\':
                    in_quote = None
                continue
            if ch in ('"', "'"):
                in_quote = ch
                continue
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end = pos
                    break
        if end is None:
            return None
        index_expr = expr[start + 1:end]
        rest = expr[end + 1:].lstrip()
        if not rest.startswith('.'):
            return None
        rest = rest[1:]
        method_match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)(.*)$', rest)
        if not method_match:
            return None
        method_name = method_match.group(1)
        after = method_match.group(2).lstrip()
        if not after:
            return base_name, index_expr, method_name, ""
        if not after.startswith('('):
            return None
        depth = 0
        in_quote = None
        end = None
        for pos in range(0, len(after)):
            ch = after[pos]
            if in_quote:
                if ch == in_quote and after[pos - 1] != '\\':
                    in_quote = None
                continue
            if ch in ('"', "'"):
                in_quote = ch
                continue
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end = pos
                    break
        if end is None:
            return None
        args_part = after[1:end]
        if after[end + 1:].strip():
            return None
        return base_name, index_expr, method_name, args_part

    def eval_expr(self, expr, scope, line_number=None, depth=0):
        """
        Evaluate a general expression, handling ranges, sums, grid indexing, interpolations,
        sequences, and mathematical operations.
        :param expr: Expression string.
        :param scope: Scope.
        :param line_number: Line number.
        :param depth: Recursion depth to prevent infinite loops.
        :return: Evaluated result.
        """
        if depth > 10:  # Prevent infinite recursion
            raise RuntimeError(
                f"Expression evaluation depth limit exceeded for '{expr}' at line {line_number}")

        expr = expr.strip()
        if expr.endswith('?'):
            expr = expr[:-1].strip()
            if not expr:
                raise SyntaxError(
                    f"Invalid boolean expression '?' at line {line_number}")
            expr = f"(1 if ({expr}) else 0)"

        # Handle simple variable lookup first
        if expr in scope:
            value = scope[expr]
            if value is None and hasattr(self.compiler, 'pending_assignments'):
                pending = getattr(self.compiler, 'pending_assignments', {})
                if expr in pending and hasattr(self.compiler, '_resolve_global_dependency'):
                    target_scope = None
                    if hasattr(scope, 'get_full_scope'):
                        target_scope = scope
                    elif hasattr(self.compiler, 'current_scope'):
                        target_scope = self.compiler.current_scope()
                    if target_scope is not None:
                        try:
                            self.compiler._resolve_global_dependency(
                                expr, line_number, target_scope=target_scope)
                            if hasattr(target_scope, 'get_full_scope'):
                                value = target_scope.get_full_scope().get(expr, value)
                        except Exception:
                            pass
            return value

        # Handle grid indexing without variable prefix (e.g., grid{a, b}) first
        # This must happen before array access processing to avoid confusion
        if 'grid{' in expr:
            grid_pattern = re.compile(r'grid\{([^}]+)\}')
            grid_matches = grid_pattern.findall(expr)
            for indices_str in grid_matches:
                try:
                    # Parse indices (e.g., "a-1, b-1" -> [a-1, b-1])
                    indices = []
                    for index_expr in indices_str.split(','):
                        index_expr = index_expr.strip()
                        try:
                            # Evaluate the index expression (e.g., "a-1")
                            index_value = self.eval_expr(
                                index_expr, scope, line_number)
                            indices.append(index_value)
                        except Exception as e:
                            raise SyntaxError(
                                f"Invalid index expression '{index_expr}' in grid{{...}}: {e} at line {line_number}")

                    # Look for grid in the scope
                    if 'grid' in scope:
                        grid = scope['grid']
                        if isinstance(grid, dict):
                            # Convert tuple indices to the format used in the grid
                            if len(indices) == 2:
                                key = (int(indices[0]), int(indices[1]))
                                if key in grid:
                                    value = grid[key]
                                else:
                                    # Return 0 for undefined grid positions
                                    value = 0
                            else:
                                raise SyntaxError(
                                    f"Grid indexing expects 2 indices, got {len(indices)} at line {line_number}")
                        else:
                            raise ValueError(
                                f"Expected grid to be a dictionary, got {type(grid)} at line {line_number}")
                    else:
                        # Return 0 if grid is not defined
                        value = 0

                    # Replace the grid expression with the value
                    expr = expr.replace(f'grid{{{indices_str}}}', str(value))

                except Exception as e:
                    raise RuntimeError(
                        f"Invalid grid reference 'grid{{{indices_str}}}': {e} at line {line_number}")

        # Handle interpolated cell references (e.g., [B{I}], [{i :A}5]) first
        # This must happen before array access processing to avoid confusion
        col_interp_pattern = re.compile(
            r'\[\{\s*([^}:]+?)\s*:\s*([A-Za-z]+)\s*\}\s*(\d+|\{[^}]+\})\s*\]')

        def col_interp_replacer(match):
            inside = match.group(0)[1:-1].strip()
            cell_ref = self._resolve_column_interpolated_cell(
                inside, scope, line_number)
            value = self.compiler.array_handler.lookup_cell(
                cell_ref, line_number)
            if isinstance(value, (int, float)):
                return str(value)
            return f'"{value}"'

        expr = col_interp_pattern.sub(col_interp_replacer, expr)

        interpolated_cell_pattern = re.compile(r'\[([A-Za-z]+)\{([^}]+)\}\]')
        interpolated_matches = interpolated_cell_pattern.findall(expr)
        for col, index_expr in interpolated_matches:
            try:
                # Evaluate the index expression
                index_value = self.eval_expr(
                    index_expr, scope, line_number, depth + 1)
                if isinstance(index_value, float) and index_value.is_integer():
                    index_value = int(index_value)
                if not isinstance(index_value, int) or index_value < 1:
                    raise ValueError(
                        f"Invalid cell reference index '{index_value}' must be a positive integer at line {line_number}")

                # Construct the cell reference
                cell_ref = f"{col}{index_value}"
                validate_cell_ref(cell_ref)
                value = self.compiler.array_handler.lookup_cell(
                    cell_ref, line_number)
                if isinstance(value, (int, float)):
                    expr = expr.replace(f'[{col}{{{index_expr}}}]', str(value))
                else:
                    expr = expr.replace(
                        f'[{col}{{{index_expr}}}]', f'"{value}"')
            except Exception as e:
                raise RuntimeError(
                    f"Invalid interpolated cell reference '[{col}{{{index_expr}}}]': {e} at line {line_number}")

        # Handle member calls on array elements (e.g., hand1(i).Name())
        parsed_member = self._parse_indexed_member_call(expr)
        if parsed_member:
            array_name, index_expr, method_name, args_part = parsed_member
            array_value = None
            try:
                array_value = scope.get(array_name)
            except Exception:
                try:
                    array_value = self.compiler.current_scope().get(array_name)
                except Exception:
                    array_value = None
            if array_value is None:
                parsed_member = None
        if parsed_member:

            splitter = getattr(self.compiler, '_split_call_arguments', None)
            if splitter:
                index_parts = splitter(index_expr)
            else:
                index_parts = [p.strip()
                               for p in re.split(r',(?![^{]*})', index_expr) if p.strip()]
            indices = []
            for part in index_parts:
                idx_value = self.eval_expr(part, scope, line_number, depth + 1)
                if isinstance(idx_value, float) and idx_value.is_integer():
                    idx_value = int(idx_value)
                if isinstance(idx_value, int):
                    idx_value -= 1
                indices.append(idx_value)

            element = self.compiler.array_handler.get_array_element(
                array_value, indices, line_number)
            element_type = None
            if isinstance(element, dict):
                element_type = element.get('_type_name')
                if not element_type:
                    for t_name, t_def in getattr(self.compiler, 'types_defined', {}).items():
                        public_fields = self.compiler._get_public_type_fields(
                            t_def)
                        if public_fields and object_public_keys(element) == set(public_fields.keys()):
                            element_type = t_name
                            break
            func_key = self.compiler._resolve_member_function(
                element_type, method_name) if element_type else None
            if func_key and func_key in getattr(self.compiler, 'functions', {}):
                args_list = []
                if args_part and args_part.strip():
                    if splitter:
                        args_list = splitter(args_part)
                    else:
                        args_list = [a.strip()
                                     for a in re.split(r',(?![^{]*})', args_part) if a.strip()]
                evaluated_args = [self.eval_or_eval_array(
                    a, scope, line_number) for a in args_list]
                return self.compiler.call_function(
                    func_key, [element] + evaluated_args, instance_type=element_type)

        # Preprocess: evaluate all array accesses D{...} in the expression
        # But skip if this looks like object creation
        if not expr.strip().startswith('new '):
            def array_access_replacer(match):
                array_expr = match.group(0)
                # Parse the array access directly to avoid recursion
                var_name, indices_str = array_expr.split('{', 1)
                var_name = var_name.strip()
                indices_str = indices_str[:-1]  # Remove closing brace

                # Parse indices
                indices = []
                for index_expr in indices_str.split(','):
                    index_expr = index_expr.strip()
                    try:
                        # Try to evaluate as a number first
                        indices.append(int(index_expr))
                    except ValueError:
                        # If not a number, evaluate as an expression
                        index_value = self.eval_expr(
                            index_expr, scope, line_number)
                        indices.append(int(index_value))

                # Get the array and evaluate the access
                if hasattr(scope, 'get_evaluation_scope'):
                    full_scope = scope.get_evaluation_scope()
                else:
                    full_scope = scope

                if var_name in full_scope:
                    arr = full_scope[var_name]
                else:
                    try:
                        arr = self.compiler.current_scope().get(var_name)
                    except NameError:
                        # Not an array access; leave the original expression intact
                        return match.group(0)

                # Adjust indices based on dimension metadata
                dims = self.compiler.dimensions.get(var_name, [])
                # Also check constraints in scope for dimension-constrained arrays
                scope_constraints = None
                current_scope = self.compiler.current_scope()
                if hasattr(current_scope, 'constraints') and var_name in current_scope.constraints:
                    scope_constraints = current_scope.constraints[var_name].get(
                        'dim', [])
                else:
                    pass

                adjusted_indices = []
                for i, idx in enumerate(indices):
                    # Prefer scope constraints (dimension-constrained arrays)
                    if scope_constraints and i < len(scope_constraints):
                        constraint = scope_constraints[i]
                        if isinstance(constraint, tuple) and len(constraint) == 2 and isinstance(constraint[1], tuple):
                            base = constraint[1][0]
                            adjusted_idx = idx - base
                            adjusted_indices.append(adjusted_idx)
                        else:
                            adjusted_idx = idx - 1
                            adjusted_indices.append(adjusted_idx)
                    # range, e.g., (0, 10)
                    elif i < len(dims) and isinstance(dims[i][1], tuple):
                        base = dims[i][1][0]
                        adjusted_idx = idx - base
                        adjusted_indices.append(adjusted_idx)
                    else:  # just a size, treat as 1-based
                        adjusted_idx = idx - 1
                        adjusted_indices.append(adjusted_idx)

                # Get the array element
                try:
                    result = self.compiler.array_handler.get_array_element(
                        arr, adjusted_indices, line_number)
                    return str(result)
                except (IndexError, ValueError) as e:
                    raise IndexError(
                        f"Invalid indices {indices_str} for '{var_name}': {e} at line {line_number}")

            expr = re.sub(r'[\w_]+\{[^}]+\}', array_access_replacer, expr)

            # Handle member field indexing (e.g., obj.field(1 to n))
            def member_paren_access_replacer(match):
                obj_name, field_name, index_expr = match.groups()
                try:
                    obj_value = self.compiler.current_scope().get(obj_name)
                except Exception:
                    obj_value = None
                if not isinstance(obj_value, dict):
                    return match.group(0)
                obj_type = obj_value.get('_type_name')
                if obj_type:
                    func_key = self.compiler._resolve_member_function(
                        obj_type, field_name)
                    if func_key and func_key in getattr(self.compiler, 'functions', {}):
                        return match.group(0)
                    type_def = getattr(self.compiler, 'types_defined', {}).get(
                        obj_type.lower(), {})
                    helper_defs = type_def.get(
                        '_private_helpers', {}) if isinstance(type_def, dict) else {}
                    if field_name.lower() in helper_defs:
                        return match.group(0)
                field_val = obj_value.get(field_name)
                import pyarrow as pa
                if not isinstance(field_val, (list, tuple, pa.Array)):
                    return match.group(0)
                range_match = re.match(
                    r'^\s*(.+)\s+to\s+(.+?)\s*$', index_expr, re.I)
                if range_match:
                    start_expr, end_expr = range_match.groups()
                    try:
                        start_value = self.eval_expr(
                            start_expr, scope, line_number)
                        end_value = self.eval_expr(
                            end_expr, scope, line_number)
                        if isinstance(start_value, float) and start_value.is_integer():
                            start_value = int(start_value)
                        if isinstance(end_value, float) and end_value.is_integer():
                            end_value = int(end_value)
                    except Exception as e:
                        raise RuntimeError(
                            f"Error evaluating index expression '{index_expr}': {e} at line {line_number}")
                    start_idx = start_value - 1
                    end_idx = end_value - 1
                    if isinstance(field_val, pa.Array):
                        return repr(field_val.to_pylist()[start_idx:end_idx + 1])
                    if isinstance(field_val, list):
                        return repr(field_val[start_idx:end_idx + 1])
                    if isinstance(field_val, tuple):
                        return repr(list(field_val[start_idx:end_idx + 1]))
                try:
                    index_value = self.eval_expr(
                        index_expr, scope, line_number)
                    if isinstance(index_value, float) and index_value.is_integer():
                        index_value = int(index_value)
                except Exception:
                    return match.group(0)
                idx = index_value - 1
                try:
                    if isinstance(field_val, pa.Array):
                        result = field_val.to_pylist()[idx]
                    else:
                        result = field_val[idx]
                except Exception:
                    return match.group(0)
                if isinstance(result, str):
                    return f'"{result}"'
                if isinstance(result, (list, dict)):
                    return repr(result)
                return str(result)

            expr = re.sub(r'([A-Za-z_][\w_]*)\.(\w+)\(([^)]+)\)',
                          member_paren_access_replacer, expr)

            # Handle parentheses array indexing (e.g., D(k))
            def paren_access_replacer(match):
                var_name, index_expr = match.groups()
                # Skip user-defined functions/subprocesses
                known_funcs = {}
                if hasattr(self, 'functions'):
                    known_funcs.update(getattr(self, 'functions') or {})
                if hasattr(self.compiler, 'functions'):
                    known_funcs.update(
                        getattr(self.compiler, 'functions', {}) or {})
                known_subprocesses = {}
                if hasattr(self, 'subprocesses'):
                    known_subprocesses.update(
                        getattr(self, 'subprocesses') or {})
                if hasattr(self.compiler, 'subprocesses'):
                    known_subprocesses.update(
                        getattr(self.compiler, 'subprocesses', {}) or {})
                builtin_callables = {
                    'len', 'mid', 'textsplit', 'counta', 'randarray', 'sortby',
                    'sqrt', 'rows'
                }
                if var_name.lower() in known_funcs:
                    return match.group(0)
                if var_name.lower() in known_subprocesses:
                    return match.group(0)
                if var_name.lower() in builtin_callables:
                    return match.group(0)
                # Get the array and evaluate the access
                if hasattr(scope, 'get_evaluation_scope'):
                    full_scope = scope.get_evaluation_scope()
                else:
                    full_scope = scope

                if var_name in full_scope:
                    arr = full_scope[var_name]
                else:
                    try:
                        arr = self.compiler.current_scope().get(var_name)
                    except NameError:
                        return match.group(0)
                # If the target is not an indexable array/list/object, leave the call unchanged
                import pyarrow as pa
                if not isinstance(arr, (list, tuple, dict, pa.Array)):
                    return match.group(0)

                # Check if this array is 0-based or 1-based based on dimension constraints
                is_zero_based = False

                # Try to find constraints in the actual scope object
                actual_scope = None
                if hasattr(scope, 'constraints'):
                    actual_scope = scope
                elif hasattr(scope, 'get_defining_scope'):
                    # This is a Scope object, use it directly
                    actual_scope = scope
                else:
                    # This is an evaluation scope dictionary, try to find the actual scope
                    # Look for the variable in the compiler's scopes
                    for scope_obj in self.compiler.scopes:
                        if var_name in scope_obj.constraints:
                            actual_scope = scope_obj
                            break

                if actual_scope and hasattr(actual_scope, 'constraints') and var_name in actual_scope.constraints:
                    dim_constraints = actual_scope.constraints[var_name].get(
                        'dim', [])
                    for _, size_spec in dim_constraints:
                        if isinstance(size_spec, tuple):
                            start, end = size_spec
                            if start == 0:
                                is_zero_based = True
                                break

                # Handle range access like var(1 to n)
                range_match = re.match(
                    r'^\s*(.+)\s+to\s+(.+?)\s*$', index_expr, re.I)
                if range_match:
                    start_expr, end_expr = range_match.groups()
                    try:
                        start_value = self.eval_expr(
                            start_expr, scope, line_number)
                        end_value = self.eval_expr(
                            end_expr, scope, line_number)
                        if isinstance(start_value, float) and start_value.is_integer():
                            start_value = int(start_value)
                        if isinstance(end_value, float) and end_value.is_integer():
                            end_value = int(end_value)
                    except Exception as e:
                        raise RuntimeError(
                            f"Error evaluating index expression '{index_expr}': {e} at line {line_number}")
                    if is_zero_based:
                        start_idx = start_value
                        end_idx = end_value
                    else:
                        start_idx = start_value - 1
                        end_idx = end_value - 1
                    if isinstance(arr, pa.Array):
                        arr_list = arr.to_pylist()
                        return repr(arr_list[start_idx:end_idx + 1])
                    if isinstance(arr, list):
                        return repr(arr[start_idx:end_idx + 1])
                    if isinstance(arr, tuple):
                        return repr(list(arr[start_idx:end_idx + 1]))
                    return match.group(0)

                # Evaluate the index expression
                try:
                    index_value = self.eval_expr(
                        index_expr, scope, line_number)
                    if isinstance(index_value, float) and index_value.is_integer():
                        index_value = int(index_value)
                except Exception as e:
                    raise RuntimeError(
                        f"Error evaluating index expression '{index_expr}': {e} at line {line_number}")

                # Convert to 0-based indexing
                if is_zero_based:
                    adjusted_index = index_value  # Already 0-based
                else:
                    adjusted_index = index_value - 1  # Convert from 1-based to 0-based

                try:
                    result = self.compiler.array_handler.get_array_element(
                        arr, [adjusted_index], line_number)
                    # If the result is a string, quote it to prevent it from being treated as a variable name
                    if isinstance(result, str):
                        return f'"{result}"'
                    # If the result is a number, convert it to string to prevent regex substitution issues
                    return str(result)
                except (IndexError, ValueError) as e:
                    raise IndexError(
                        f"Invalid index {index_expr} for '{var_name}': {e} at line {line_number}")

            # Don't match function calls like SQRT(100), only array access like D(k)
            def paren_access_replacer_with_check(match):
                var_name, index_expr = match.groups()
                # Check if this is a function call by looking for common function names
                callable_names = {'SQRT', 'ABS', 'SIN', 'COS',
                                  'TAN', 'LOG', 'EXP', 'LEN', 'MID', 'TEXTSPLIT'}
                if hasattr(self.compiler, 'functions'):
                    callable_names.update(
                        {n.upper() for n in self.compiler.functions.keys()})
                if hasattr(self.compiler, 'subprocesses'):
                    callable_names.update(
                        {n.upper() for n in self.compiler.subprocesses.keys()})
                if var_name.upper() in callable_names:
                    return match.group(0)  # Don't process function calls
                # Check if this contains a dimension selector (e.g., Results!Quarter)
                if '!' in var_name:
                    return match.group(0)  # Don't process dimension selectors
                return paren_access_replacer(match)

            # Use a more specific pattern that doesn't match function calls within dimension selectors
            # First, let's handle dimension selectors before processing parentheses
            if '!' in expr and '(' in expr and ')' in expr:
                m = re.match(r'^([\w_]+)!(\w+)\(([^)]+)\)$', expr)
                if m:
                    var_name, dim_name, index_str = m.groups()
                    # Check if variable exists in scope or compiler variables
                    if hasattr(scope, 'get_evaluation_scope'):
                        full_scope = scope.get_evaluation_scope()
                    else:
                        full_scope = scope

                    if var_name not in full_scope and var_name not in self.compiler.variables:
                        raise NameError(
                            f"Variable '{var_name}' not defined at line {line_number}")
                    if var_name not in self.compiler.dim_names:
                        raise SyntaxError(
                            f"Variable '{var_name}' has no named dimensions at line {line_number}")
                    dim_idx = self.compiler.dim_names[var_name].get(dim_name)
                    if dim_idx is None:
                        raise SyntaxError(
                            f"Dimension '{dim_name}' not found for '{var_name}' at line {line_number}")
                    labels = self.compiler.dim_labels.get(
                        var_name, {}).get(dim_name, {})
                    array = full_scope.get(
                        var_name, self.compiler.variables.get(var_name))
                    shape = self.compiler.array_handler.get_array_shape(
                        array, line_number)
                    index_str_clean = index_str.strip('"')
                    dim_index = None
                    if index_str_clean in labels:
                        dim_index = labels[index_str_clean]
                    else:
                        for k, v in labels.items():
                            if str(v) == index_str_clean:
                                dim_index = k
                                break
                    if dim_index is None:
                        try:
                            dim_index = int(index_str_clean) - 1
                        except ValueError:
                            raise ValueError(
                                f"Invalid index '{index_str_clean}' for dimension '{dim_name}' at line {line_number}")
                    try:
                        dim_index = int(dim_index)
                    except Exception:
                        raise ValueError(
                            f"Invalid resolved index '{dim_index}' for dimension '{dim_name}' at line {line_number}")
                    if dim_index < 0 or dim_index >= shape[dim_idx]:
                        raise ValueError(
                            f"Index {dim_index + 1} out of bounds for dimension '{dim_name}' at line {line_number}")
                    flat_arr = self.compiler.array_handler.flatten_array(
                        array, line_number)
                    if len(shape) == 1:
                        result = flat_arr[dim_index]
                    else:
                        inner_size = shape[1] if len(shape) > 1 else 1
                        outer_size = shape[0]
                        result = []
                        if dim_idx == 0:
                            start_idx = dim_index * inner_size
                            result = flat_arr[start_idx:start_idx + inner_size]
                        else:
                            for i in range(outer_size):
                                result.append(
                                    flat_arr[i * inner_size + dim_index])
                    if len(result) == 1 and isinstance(result, list):
                        result = result[0]
                    return result

            # Now handle regular parentheses access (but not dimension selectors)
            # Use a balanced parser to support nested parentheses.
            def _replace_paren_access_balanced(text):
                out = []
                i = 0
                length = len(text)

                class _FakeMatch:
                    def __init__(self, name, idx_expr):
                        self._name = name
                        self._idx_expr = idx_expr
                        self._full = f"{name}({idx_expr})"

                    def groups(self):
                        return (self._name, self._idx_expr)

                    def group(self, idx):
                        if idx == 0:
                            return self._full
                        raise IndexError("Only group(0) supported")

                while i < length:
                    ch = text[i]
                    if ch in ('"', "'"):
                        quote = ch
                        out.append(ch)
                        i += 1
                        while i < length:
                            out.append(text[i])
                            if text[i] == quote and text[i - 1] != '\\':
                                i += 1
                                break
                            i += 1
                        continue

                    if ch.isalpha() or ch == '_':
                        start = i
                        i += 1
                        while i < length and (text[i].isalnum() or text[i] == '_'):
                            i += 1
                        name = text[start:i]
                        # Skip member calls like obj.method(...)
                        prev = start - 1
                        while prev >= 0 and text[prev].isspace():
                            prev -= 1
                        if prev >= 0 and text[prev] == '.':
                            out.append(name)
                            continue

                        j = i
                        while j < length and text[j].isspace():
                            j += 1
                        if j < length and text[j] == '(':
                            depth = 1
                            k = j + 1
                            while k < length and depth > 0:
                                if text[k] in ('"', "'"):
                                    quote = text[k]
                                    k += 1
                                    while k < length:
                                        if text[k] == quote and text[k - 1] != '\\':
                                            k += 1
                                            break
                                        k += 1
                                    continue
                                if text[k] == '(':
                                    depth += 1
                                elif text[k] == ')':
                                    depth -= 1
                                k += 1
                            if depth == 0:
                                index_expr = text[j + 1:k - 1]
                                fake = _FakeMatch(name, index_expr)
                                out.append(
                                    paren_access_replacer_with_check(fake))
                                i = k
                                continue
                        out.append(name)
                        continue

                    out.append(ch)
                    i += 1
                return ''.join(out)

            expr = _replace_paren_access_balanced(expr)

        # Handle ranges in expressions
        range_match = re.match(
            r'^\s*\[\s*([A-Z]+\d+)\s*:\s*([A-Z]+\d+)\s*\]\s*$', expr)
        if range_match:
            s_ref, e_ref = range_match.groups()
            try:
                validate_cell_ref(s_ref)
                validate_cell_ref(e_ref)
                values = self.compiler.array_handler.get_range_values(
                    s_ref, e_ref, line_number)
                flat_values = [v for row in values for v in (
                    row if isinstance(row, list) else [row])]
                if all(v == 0.0 for v in flat_values) and line_number is not None:
                    raise RuntimeError(
                        f"Range '{s_ref}:{e_ref}' may depend on unassigned grid cells at line {line_number}; defer initialization")
                result = []
                for v in flat_values:
                    if isinstance(v, pa.Array):
                        result.extend([float(x) for x in v.to_pylist()])
                    else:
                        result.append(float(v) if isinstance(
                            v, (int, float)) else v)
                return result
            except Exception as e:
                raise RuntimeError(
                    f"Error evaluating range '{s_ref}:{e_ref}': {e} at line {line_number}")

        # Handle sum ranges and vars (delegated to helper methods)
        if expr.lower().startswith('sum[') and expr.endswith(']'):
            return self._evaluate_sum_range(expr, scope, line_number)
        if expr.lower().startswith('sum([') and expr.endswith('])'):
            inner_expr = expr[4:-2].strip()
            m = re.match(
                r'^\[?([A-Z]+\d+)\s*:\s*([A-Z]+\d+)\]?$', inner_expr, re.I)
            if m:
                start_ref, end_ref = m.groups()
                return self._evaluate_sum_range(f"sum[{start_ref}:{end_ref}]", scope, line_number)
            raise SyntaxError(
                f"Invalid sum range syntax in parentheses: {expr} at line {line_number}")
        if expr.startswith('sum{') and expr.endswith('}'):
            return self._evaluate_sum_vars(expr, scope, line_number)

        # Handle grid indexing (e.g., var.grid{1,2})
        grid_index_match = re.match(
            r'^([\w_]+)\.grid\{([\d,\s]+)\}$', expr, re.I)
        if grid_index_match:
            var_name, indices_str = grid_index_match.groups()
            indices = [int(i.strip()) - 1 for i in indices_str.split(',')]
            full_scope = self.compiler.current_scope().get_full_scope()
            if var_name in full_scope:
                value = full_scope.get(var_name)
                if value is None:
                    raise ValueError(
                        f"Variable '{var_name}' is uninitialized at line {line_number}")
                try:
                    if isinstance(value, dict) and 'grid' in value:
                        result = self.compiler.array_handler.get_array_element(
                            value['grid'], indices, line_number, original_shape=value.get('original_shape'))
                    else:
                        result = self.compiler.array_handler.get_array_element(
                            value, indices, line_number)
                    return result
                except (IndexError, ValueError) as e:
                    raise IndexError(
                        f"Invalid indices {indices_str} for '{var_name}': {e} at line {line_number}")

        # Handle numeric indexing (e.g., var[1] or var(0))
        index_match = re.match(r'^([\w_]+)(?:\[(\d+)\]|\((\d+)\))$', expr)
        if index_match:
            var_name, index_bracket, index_paren = index_match.groups()
            index = index_bracket or index_paren
            if var_name in scope or var_name in self.compiler.variables:
                array = scope.get(
                    var_name, self.compiler.variables.get(var_name))
                try:
                    idx = int(index)
                    # Adjust for 1-based [n] or 0-based (n)
                    idx = idx - 1 if index_bracket else idx
                    result = self.compiler.array_handler.get_array_element(
                        array, [idx], line_number)
                    return result
                except (IndexError, ValueError) as e:
                    raise IndexError(
                        f"Invalid index {index} for '{var_name}': {e} at line {line_number}")

        # Handle parentheses indexing with expressions (e.g., D(k+1))
        paren_expr_match = re.match(r'^([\w_]+)\(([^)]+)\)$', expr)
        if paren_expr_match:
            var_name, index_expr = paren_expr_match.groups()
            if var_name in scope or var_name in self.compiler.variables:
                array = scope.get(
                    var_name, self.compiler.variables.get(var_name))
                try:
                    # Evaluate the index expression (e.g., "k+1")
                    index_value = self.eval_expr(
                        index_expr, scope, line_number)
                    if isinstance(index_value, float) and index_value.is_integer():
                        index_value = int(index_value)
                    # Parentheses indexing is 1-based, so adjust to 0-based
                    idx = index_value - 1
                    result = self.compiler.array_handler.get_array_element(
                        array, [idx], line_number)
                    return result
                except (IndexError, ValueError) as e:
                    raise IndexError(
                        f"Invalid index {index_expr} for '{var_name}': {e} at line {line_number}")

        # Handle cell-based indexing (e.g., var[A1])
        m = re.match(r'^([\w_]+)\[([A-Z]+\d+)\]$', expr)
        if m:
            var_name, cell_ref = m.groups()
            if var_name in scope or var_name in self.compiler.variables:
                return self.compiler.array_handler.resolve_cell_index(var_name, cell_ref, line_number)

        # Handle curly brace indexing (e.g., D{i, 2})
        curly_brace_match = re.match(r'^([\w_]+)\{([^}]+)\}$', expr)
        if curly_brace_match:
            var_name, indices_str = curly_brace_match.groups()
            if var_name in scope or var_name in self.compiler.variables:
                # Parse indices (e.g., "i+1, 1" -> [i+1, 1])
                indices = []
                for index_expr in indices_str.split(','):
                    index_expr = index_expr.strip()
                    try:
                        # Evaluate the index expression (e.g., "i+1")
                        index_value = self.eval_expr(
                            index_expr, scope, line_number)
                        indices.append(index_value)
                    except Exception as e:
                        raise SyntaxError(
                            f"Invalid index expression '{index_expr}' in '{expr}': {e} at line {line_number}")

                # Get the array
                array = scope.get(
                    var_name, self.compiler.variables.get(var_name))
                if array is None:
                    raise ValueError(
                        f"Variable '{var_name}' is uninitialized at line {line_number}")

                # Adjust indices based on dimension metadata
                dims = self.compiler.dimensions.get(var_name, [])
                # Also check constraints in scope for dimension-constrained arrays
                scope_constraints = None
                current_scope = self.compiler.current_scope()
                if hasattr(current_scope, 'constraints') and var_name in current_scope.constraints:
                    scope_constraints = current_scope.constraints[var_name].get(
                        'dim', [])
                else:
                    pass

                adjusted_indices = []
                for i, idx in enumerate(indices):
                    # Prefer scope constraints (dimension-constrained arrays)
                    if scope_constraints and i < len(scope_constraints):
                        constraint = scope_constraints[i]
                        if isinstance(constraint, tuple) and len(constraint) == 2 and isinstance(constraint[1], tuple):
                            base = constraint[1][0]
                            adjusted_idx = idx - base
                            adjusted_indices.append(adjusted_idx)
                        else:
                            adjusted_idx = idx - 1
                            adjusted_indices.append(adjusted_idx)
                    # range, e.g., (0, 10)
                    elif i < len(dims) and isinstance(dims[i][1], tuple):
                        base = dims[i][1][0]
                        adjusted_idx = idx - base
                        adjusted_indices.append(adjusted_idx)
                    else:  # just a size, treat as 1-based
                        adjusted_idx = idx - 1
                        adjusted_indices.append(adjusted_idx)

                # Special case: if all indices are 0 and we have dimension constraints, use indices as-is
                current_scope = self.compiler.current_scope()
                if all(idx == 0 for idx in indices) and var_name in current_scope.constraints:
                    scope_constraints = current_scope.constraints[var_name].get(
                        'dim', [])
                    if scope_constraints:
                        adjusted_indices = indices

                try:
                    result = self.compiler.array_handler.get_array_element(
                        array, adjusted_indices, line_number)
                    return result
                except (IndexError, ValueError) as e:
                    raise IndexError(
                        f"Invalid indices {indices_str} for '{var_name}': {e} at line {line_number}")

        # Handle dimension selector (e.g., var!dim("label"))
        if '!' in expr and '(' in expr and ')' in expr:
            m = re.match(r'^([\w_]+)!(\w+)\(([^)]+)\)$', expr)
            if m:
                var_name, dim_name, index_str = m.groups()
                # Check if variable exists in scope or compiler variables
                if hasattr(scope, 'get_evaluation_scope'):
                    full_scope = scope.get_evaluation_scope()
                else:
                    full_scope = scope

                if var_name not in full_scope and var_name not in self.compiler.variables:
                    raise NameError(
                        f"Variable '{var_name}' not defined at line {line_number}")
                if var_name not in self.compiler.dim_names:
                    raise SyntaxError(
                        f"Variable '{var_name}' has no named dimensions at line {line_number}")
                dim_idx = self.compiler.dim_names[var_name].get(dim_name)
                if dim_idx is None:
                    raise SyntaxError(
                        f"Dimension '{dim_name}' not found for '{var_name}' at line {line_number}")
                labels = self.compiler.dim_labels.get(
                    var_name, {}).get(dim_name, {})
                array = full_scope.get(
                    var_name, self.compiler.variables.get(var_name))
                shape = self.compiler.array_handler.get_array_shape(
                    array, line_number)
                index_str_clean = index_str.strip('"')
                dim_index = None
                if index_str_clean in labels:
                    dim_index = labels[index_str_clean]
                else:
                    for k, v in labels.items():
                        if str(v) == index_str_clean:
                            dim_index = k
                            break
                if dim_index is None:
                    try:
                        dim_index = int(index_str_clean) - 1
                    except ValueError:
                        raise ValueError(
                            f"Invalid index '{index_str_clean}' for dimension '{dim_name}' at line {line_number}")
                try:
                    dim_index = int(dim_index)
                except Exception:
                    raise ValueError(
                        f"Invalid resolved index '{dim_index}' for dimension '{dim_name}' at line {line_number}")
                if dim_index < 0 or dim_index >= shape[dim_idx]:
                    raise ValueError(
                        f"Index {dim_index + 1} out of bounds for dimension '{dim_name}' at line {line_number}")
                flat_arr = self.compiler.array_handler.flatten_array(
                    array, line_number)
                if len(shape) == 1:
                    result = flat_arr[dim_index]
                else:
                    inner_size = shape[1] if len(shape) > 1 else 1
                    outer_size = shape[0]
                    result = []
                    if dim_idx == 0:
                        start_idx = dim_index * inner_size
                        result = flat_arr[start_idx:start_idx + inner_size]
                    else:
                        for i in range(outer_size):
                            result.append(flat_arr[i * inner_size + dim_index])
                if len(result) == 1 and isinstance(result, list):
                    result = result[0]
                return result

        # Handle array indexing (e.g., var{1,2} or var{i,2})
        match = re.match(r'^[\w_]+\{[^}]+\}$', expr)
        if match:
            var_name, indices_str = expr.split('{', 1)
            var_name = var_name.strip()
            indices_str = indices_str[:-1]  # Remove closing brace
            # Parse indices - can be numbers or variables
            indices = []
            for index_expr in indices_str.split(','):
                index_expr = index_expr.strip()
                try:
                    # Try to evaluate as a number first
                    indices.append(int(index_expr))
                except ValueError:
                    # If not a number, evaluate as an expression
                    try:
                        index_value = self.eval_expr(
                            index_expr, scope, line_number)
                        indices.append(int(index_value))
                    except Exception as e:
                        raise ValueError(
                            f"Invalid index expression '{index_expr}' in '{expr}': {e} at line {line_number}")

            # Get the full scope to find variables in parent scopes
            if hasattr(scope, 'get_evaluation_scope'):
                full_scope = scope.get_evaluation_scope()
            else:
                full_scope = scope

            if var_name in full_scope:
                arr = full_scope[var_name]
            else:
                try:
                    arr = self.compiler.current_scope().get(var_name)
                except NameError:
                    raise NameError(
                        f"Variable '{var_name}' not found at line {line_number}")

            dims = self.compiler.dimensions.get(var_name, [])
            shape = self.compiler.array_handler.get_array_shape(
                arr, line_number)
            if len(indices) != len(shape):
                raise ValueError(
                    f"Expected {len(shape)} indices for '{var_name}', got {len(indices)} at line {line_number}")
            adjusted_indices = []
            for i, idx in enumerate(indices):
                if i < len(dims) and isinstance(dims[i][1], tuple):
                    start, end = dims[i][1]
                    dim_size = end - start + 1
                    adjusted_idx = idx - start
                    if adjusted_idx < 0 or adjusted_idx >= dim_size:
                        raise ValueError(
                            f"Index {idx} out of bounds for dimension {i} of '{var_name}' (range {start} to {end}) at line {line_number}")
                else:
                    adjusted_idx = idx - 1
                    dim_size = shape[i]
                    if adjusted_idx < 0 or adjusted_idx >= dim_size:
                        raise ValueError(
                            f"Index {idx} out of bounds for dimension {i} of '{var_name}' (size {dim_size}) at line {line_number}")
                adjusted_indices.append(adjusted_idx)
            if isinstance(arr, pa.ListArray):
                arr = arr.to_pylist()
            elif not isinstance(arr, list):
                raise ValueError(
                    f"Cannot index into non-array variable '{var_name}' at line {line_number}")
            result = arr
            for dim, idx in enumerate(adjusted_indices):
                if not isinstance(result, list):
                    raise ValueError(
                        f"Cannot index into non-list at dimension {dim} of '{var_name}' at line {line_number}")
                result = result[idx]
            return result

        # Handle text concatenation with & operator
        if '&' in expr:
            parts = expr.split('&')
            result = ''
            for part in parts:
                part = part.strip()
                if part.startswith('$"') and part.endswith('"'):
                    result += self._process_interpolation(
                        part, scope, line_number)
                elif part.startswith('"') and part.endswith('"'):
                    result += part[1:-1].replace('""', '"')
                else:
                    val = self.eval_expr(part, scope, line_number)
                    result += str(val)
            return result

        # Handle interpolated strings ($"...")
        if expr.startswith('$"') and expr.endswith('"'):
            return self._process_interpolation(expr, scope, line_number)

        # Handle literal strings
        if expr.startswith('"') and expr.endswith('"'):
            return expr[1:-1].replace('""', '"')

        # Handle special values (#INF, -#INF, #N/A)
        uexpr = expr.upper()
        if uexpr == '#INF':
            return float('inf')
        if uexpr == '-#INF':
            return float('-inf')
        if uexpr == '#N/A':
            return float('nan')

        # Handle numeric literals (integers, floats, scientific notation)
        try:
            nm = re.match(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$', expr)
            if nm:
                return float(expr) if '.' in expr or 'e' in expr.lower() else int(expr)
        except ValueError:
            pass

        # Handle number sequences (e.g., 1 to 10 step 2)
        sm = re.match(
            r'^(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)(?:\s+step\s+(-?\d+(?:\.\d+)?))?$', expr, re.I)
        if sm:
            return self._build_sequence(expr, line_number)

        # Handle object creation (e.g., new Type{val1, val2} or new Type(val1, val2))
        if expr.lower().startswith('new ') and 'with' in expr.lower():
            in_quote = None
            paren_level = 0
            brace_level = 0
            split_pos = None
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
                        split_pos = idx
                        break
            if split_pos is not None:
                base_expr = expr[:split_pos].strip()
                with_text = expr[split_pos:].strip()
                bare_new_match = re.match(r'^new\s+(\w+)\s*$', base_expr, re.I)
                if bare_new_match:
                    type_name = bare_new_match.group(1)
                    base_value = self.compiler._instantiate_type(
                        type_name, [], line_number,
                        allow_default_if_empty=True, execute_code=False)
                else:
                    base_value = self.eval_expr(base_expr, scope, line_number)
                if not isinstance(base_value, dict):
                    raise TypeError(
                        f"WITH can only be applied to object instances at line {line_number}")
                type_match = re.match(r'^new\s+(\w+)', base_expr, re.I)
                type_name = type_match.group(1) if type_match else None
                with_assignments = self.compiler._parse_with_clause(
                    with_text, line_number)
                return self.compiler._apply_with_constraints(
                    base_value, with_assignments, scope, line_number,
                    type_name=type_name)
        bare_new_match = re.match(r'^new\s+(\w+)\s*$', expr)
        if bare_new_match:
            type_name = bare_new_match.group(1)
            if type_name.lower() in self.compiler.types_defined:
                return self.compiler._instantiate_type(
                    type_name, [], line_number, allow_default_if_empty=True)
            raise ValueError(
                f"Type '{type_name}' not defined at line {line_number}")
        obj_match = re.match(r'^new\s+(\w+)\s*(\{|\()',
                             expr)
        if obj_match:
            type_name, opener = obj_match.groups()
            pairs = {'{': '}', '(': ')'}
            closer = pairs[opener]

            # Locate the full argument segment, supporting nested delimiters
            start_pos = expr.find(opener, obj_match.start(2))
            stack = [closer]
            args_str = None
            for i, char in enumerate(expr[start_pos + 1:], start_pos + 1):
                if char in pairs:
                    stack.append(pairs[char])
                elif stack and char == stack[-1]:
                    stack.pop()
                    if not stack:
                        args_str = expr[start_pos + 1:i]
                        break
                elif char in pairs.values():
                    raise SyntaxError(
                        f"Mismatched delimiter in object creation: {expr} at line {line_number}")

            if args_str is None:
                raise SyntaxError(
                    f"Unclosed delimiters in object creation: {expr} at line {line_number}")

            # Parse arguments, handling nested objects with braces or parentheses
            args_list = []
            current_arg = ""
            nest_stack = []
            for char in args_str + ',':
                if char == ',' and not nest_stack:
                    if current_arg.strip():
                        args_list.append(current_arg.strip())
                    current_arg = ""
                    continue

                current_arg += char
                if char in pairs:
                    nest_stack.append(pairs[char])
                elif nest_stack and char == nest_stack[-1]:
                    nest_stack.pop()
                elif char in pairs.values():
                    raise SyntaxError(
                        f"Mismatched delimiter in arguments: {expr} at line {line_number}")

            if nest_stack:
                raise SyntaxError(
                    f"Unbalanced delimiters in arguments: {expr} at line {line_number}")

            args_list = [a for a in args_list if a.strip()]

            evaluated_args = [self.eval_or_eval_array(
                a, scope, line_number) for a in args_list]

            if type_name.lower() in self.compiler.types_defined:
                allow_defaults = len(args_list) == 0 and args_str.strip() == ''
                return self.compiler._instantiate_type(
                    type_name, evaluated_args, line_number, allow_default_if_empty=allow_defaults)
            else:
                raise ValueError(
                    f"Type '{type_name}' not defined at line {line_number}")

        # Handle member calls on arrays before falling back to eval
        m_member_array = re.match(r'^([\w_]+)\.([\w_]+)\((.*)\)$', expr)
        if m_member_array:
            obj_name, method_name, args_part = m_member_array.groups()
            obj_value = None
            obj_type = None
            try:
                obj_value = scope.get(obj_name)
            except Exception:
                try:
                    obj_value = self.compiler.current_scope().get(obj_name)
                except Exception:
                    obj_value = None
            try:
                defining_scope = self.compiler.current_scope().get_defining_scope(obj_name)
                if defining_scope:
                    actual_key = defining_scope._get_case_insensitive_key(
                        obj_name, defining_scope.types)
                    if actual_key:
                        obj_type = defining_scope.types.get(actual_key)
            except Exception:
                pass
            if isinstance(obj_value, (list, tuple, pa.Array)):
                if isinstance(obj_value, pa.Array):
                    obj_list = obj_value.to_pylist()
                else:
                    obj_list = list(obj_value)
                element_type = obj_type
                func_key = None
                if element_type:
                    func_key = self.compiler._resolve_member_function(
                        element_type, method_name)
                if not func_key:
                    for elem in obj_list:
                        if not isinstance(elem, dict):
                            continue
                        candidate = elem.get('_type_name')
                        if not candidate:
                            for t_name, t_def in getattr(self.compiler, 'types_defined', {}).items():
                                public_fields = self.compiler._get_public_type_fields(
                                    t_def)
                                if public_fields and object_public_keys(elem) == set(public_fields.keys()):
                                    candidate = t_name
                                    break
                        if not candidate:
                            continue
                        func_key = self.compiler._resolve_member_function(
                            candidate, method_name)
                        if func_key:
                            element_type = candidate
                            break
                if func_key and func_key in getattr(self.compiler, 'functions', {}):
                    args_list = []
                    if args_part.strip():
                        args_list = [a.strip()
                                     for a in re.split(r',(?![^{]*})', args_part) if a.strip()]
                    evaluated_args = [self.eval_or_eval_array(
                        a, scope, line_number) for a in args_list]
                    results = []
                    for elem in obj_list:
                        if isinstance(elem, dict):
                            results.append(self.compiler.call_function(
                                func_key, [elem] + evaluated_args, instance_type=element_type))
                    return results

        # Handle plain user-defined function calls (Func(arg1, arg2))
        m_plain_func = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$', expr)
        if m_plain_func:
            func_name, arg_text = m_plain_func.groups()
            func_defs = getattr(self.compiler, 'functions', {}) or {}
            if func_name.lower() in func_defs:
                splitter = getattr(
                    self.compiler, '_split_call_arguments', None)
                if splitter:
                    args_list = splitter(arg_text)
                else:
                    args_list = [a.strip()
                                 for a in re.split(r',(?![^{]*})', arg_text) if a.strip()] if arg_text.strip() else []
                evaluated_args = [self.eval_or_eval_array(
                    a, scope, line_number) for a in args_list] if args_list else []
                return self.compiler.call_function(func_name, evaluated_args)

        # Handle function names that include dots (e.g., Point1.DistanceToOrigin(...))
        m_dotted_func = re.match(
            r'^([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)$', expr)
        if m_dotted_func:
            func_name, arg_text = m_dotted_func.groups()
            func_defs = getattr(self.compiler, 'functions', {}) or {}
            # Skip friendly member syntax (handled below) where the prefix is an object variable
            base_name = func_name.split('.')[0]
            is_scope_var = False
            try:
                if hasattr(self.compiler, 'current_scope'):
                    self.compiler.current_scope().get(base_name)
                    is_scope_var = True
            except Exception:
                is_scope_var = False
            if func_name.lower() in func_defs and not is_scope_var:
                args_list = []
                if arg_text.strip():
                    args_list = [a.strip()
                                 for a in re.split(r',(?![^{]*})', arg_text) if a.strip()]
                evaluated_args = [self.eval_or_eval_array(
                    a, scope, line_number) for a in args_list]
                return self.compiler.call_function(func_name, evaluated_args)

        # Handle member function calls (object.method(...) or object.method)
        m_member = re.match(r'^([\w_]+)\.([\w_]+)(?:\((.*)\))?$', expr)
        if m_member:
            obj_name, method_name, args_part = m_member.groups()
            # Retrieve the object value and type (if known)
            obj_value = None
            obj_type = None
            try:
                obj_value = scope.get(obj_name)
            except Exception:
                try:
                    obj_value = self.compiler.current_scope().get(obj_name)
                except Exception:
                    obj_value = None
            if obj_value is None and hasattr(self.compiler, '_materialize_inits'):
                try:
                    self.compiler._materialize_inits()
                    try:
                        obj_value = scope.get(obj_name)
                    except Exception:
                        obj_value = self.compiler.current_scope().get(obj_name)
                except Exception:
                    pass
            try:
                defining_scope = self.compiler.current_scope(
                ).get_defining_scope(obj_name)
                if defining_scope:
                    actual_key = defining_scope._get_case_insensitive_key(
                        obj_name, defining_scope.types)
                    if actual_key:
                        obj_type = defining_scope.types.get(actual_key)
            except Exception:
                pass
            if obj_type is None and isinstance(obj_value, dict):
                obj_type = obj_value.get('_type_name')
            if obj_type is None and isinstance(obj_value, dict):
                for t_name, t_def in getattr(self.compiler, 'types_defined', {}).items():
                    public_fields = self.compiler._get_public_type_fields(
                        t_def)
                    if public_fields and object_public_keys(obj_value) == set(public_fields.keys()):
                        obj_type = t_name
                        break

            # Handle member calls on arrays of objects (e.g., hand1.Name())
            if isinstance(obj_value, (list, tuple, pa.Array)):
                if isinstance(obj_value, pa.Array):
                    obj_list = obj_value.to_pylist()
                else:
                    obj_list = list(obj_value)
                element_type = obj_type
                func_key = None
                if element_type:
                    func_key = self.compiler._resolve_member_function(
                        element_type, method_name)
                if not func_key:
                    for elem in obj_list:
                        if not isinstance(elem, dict):
                            continue
                        candidate = elem.get('_type_name')
                        if not candidate:
                            for t_name, t_def in getattr(self.compiler, 'types_defined', {}).items():
                                public_fields = self.compiler._get_public_type_fields(
                                    t_def)
                                if public_fields and object_public_keys(elem) == set(public_fields.keys()):
                                    candidate = t_name
                                    break
                        if not candidate:
                            continue
                        func_key = self.compiler._resolve_member_function(
                            candidate, method_name)
                        if func_key:
                            element_type = candidate
                            break
                if func_key and func_key in getattr(self.compiler, 'functions', {}):
                    arg_text = args_part if args_part is not None else ""
                    args_list = []
                    if arg_text.strip():
                        args_list = [a.strip()
                                     for a in re.split(r',(?![^{]*})', arg_text) if a.strip()]
                    evaluated_args = [self.eval_or_eval_array(
                        a, scope, line_number) for a in args_list]
                    results = []
                    for elem in obj_list:
                        if isinstance(elem, dict):
                            results.append(self.compiler.call_function(
                                func_key, [elem] + evaluated_args, instance_type=element_type))
                    return results

            func_key = self.compiler._resolve_member_function(
                obj_type, method_name) if obj_type else None
            if func_key and func_key in getattr(self.compiler, 'functions', {}):
                arg_text = args_part if args_part is not None else ""
                args_list = []
                if arg_text.strip():
                    args_list = [a.strip()
                                 for a in re.split(r',(?![^{]*})', arg_text) if a.strip()]
                evaluated_args = [self.eval_or_eval_array(
                    a, scope, line_number) for a in args_list]
                return self.compiler.call_function(func_key, [obj_value] + evaluated_args, instance_type=obj_type)

            if obj_type:
                type_def = getattr(self.compiler, 'types_defined', {}).get(
                    obj_type.lower(), {})
                helper_defs = type_def.get(
                    '_private_helpers', {}) if isinstance(type_def, dict) else {}
                if method_name.lower() in helper_defs:
                    if not getattr(self.compiler, '_allow_hidden_member_calls', False):
                        raise PermissionError(
                            f"Private helper '{method_name}' cannot be called here at line {line_number}")
                    arg_text = args_part if args_part is not None else ""
                    args_list = []
                    if arg_text.strip():
                        args_list = [a.strip()
                                     for a in re.split(r',(?![^{]*})', arg_text) if a.strip()]
                    evaluated_args = [self.eval_or_eval_array(
                        a, scope, line_number) for a in args_list]
                    self.compiler.type_processor._execute_private_helper(
                        obj_type, method_name, obj_value, line_number, evaluated_args)
                    return obj_value

        # Handle field access (e.g., var.field)
        if re.match(r'^[\w_]+\.\w+$', expr):
            var, field = expr.split('.')
            if var in scope and isinstance(scope[var], dict):
                if self.compiler._is_hidden_field(scope[var], field) and not getattr(self.compiler, '_allow_hidden_field_access', False):
                    raise PermissionError(
                        f"Hidden field '{field}' is not accessible at line {line_number}")
                return scope[var].get(field)
            try:
                var_value = self.compiler.current_scope().get(var)
                if isinstance(var_value, dict):
                    if self.compiler._is_hidden_field(var_value, field) and not getattr(self.compiler, '_allow_hidden_field_access', False):
                        raise PermissionError(
                            f"Hidden field '{field}' is not accessible at line {line_number}")
                    return var_value.get(field)
            except NameError:
                pass

        # Handle single cell reference (e.g., [A1])
        if expr.startswith('[') and expr.endswith(']') and not ':' in expr:
            cell_ref = expr[1:-1].strip()
            if cell_ref.startswith('^'):
                cell_ref = cell_ref[1:].strip()
            if re.match(r'^[A-Za-z]+\d+$', cell_ref):
                try:
                    validate_cell_ref(cell_ref)
                    value = self.compiler.array_handler.lookup_cell(
                        cell_ref, line_number)
                    return value
                except ValueError as e:
                    raise RuntimeError(
                        f"Invalid cell reference '{cell_ref}': {e} at line {line_number}")

        # Prepare expression for eval, replacing cell refs
        eval_expr = expr
        # Use case-insensitive scope access for cell variables
        cell_vars = {}
        for c, v in self.compiler._cell_var_map.items():
            try:
                cell_vars[v] = self.compiler.current_scope().get(v)
            except NameError:
                # Variable not found, skip it
                pass
        full_scope = {**scope, **cell_vars}

        # Handle regular cell references (e.g., [A1], [B2])
        cell_ref_pattern = re.compile(r'\[\^?([A-Za-z]+\d+)\]')
        cell_refs = cell_ref_pattern.findall(eval_expr)
        for cell_ref in cell_refs:
            try:
                validate_cell_ref(cell_ref)
                value = self.compiler.array_handler.lookup_cell(
                    cell_ref, line_number)
                if isinstance(value, (int, float)):
                    eval_expr = eval_expr.replace(f'[{cell_ref}]', str(value))
                    eval_expr = eval_expr.replace(f'[^{cell_ref}]', str(value))
                else:
                    eval_expr = eval_expr.replace(
                        f'[{cell_ref}]', f'"{value}"')
                    eval_expr = eval_expr.replace(
                        f'[^{cell_ref}]', f'"{value}"')
            except ValueError as e:
                raise RuntimeError(
                    f"Invalid cell reference '{cell_ref}': {e} at line {line_number}")

        # Handle array operations (e.g., {1,2} + {3,4})
        array_op_match = re.match(
            r'^\s*(\{[^}]*\})\s*([+\-*/^\\]|mod)\s*(\{[^}]*\})\s*$', eval_expr, re.I)
        if array_op_match:
            left_array, op, right_array = array_op_match.groups()
            arr1 = self._evaluate_array(left_array, full_scope, line_number)
            arr2 = self._evaluate_array(right_array, full_scope, line_number)
            shapes = [self.compiler.array_handler.get_array_shape(
                arr, line_number) for arr in [arr1, arr2]]
            if shapes[0] != shapes[1]:
                raise ValueError(
                    f"Arrays must have the same shape for operation {op}: {shapes} at line {line_number}")
            rows, cols = shapes[0]
            arr1_vals = arr1 if isinstance(arr1, list) else arr1.to_pylist()
            arr2_vals = arr2 if isinstance(arr2, list) else arr2.to_pylist()
            result = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    val1 = arr1_vals[i][j]
                    val2 = arr2_vals[i][j]
                    if not (isinstance(val1, (int, float)) and isinstance(val2, (int, float))):
                        raise TypeError(
                            f"Non-numeric values at position [{i}][{j}]: {val1}, {val2} at line {line_number}")
                    try:
                        if op == '+':
                            result_val = float(val1 + val2)
                        elif op == '-':
                            result_val = float(val1 - val2)
                        elif op == '*':
                            result_val = float(val1 * val2)
                        elif op == '/':
                            result_val = float(
                                val1 / val2) if val2 != 0 else float('nan')
                        elif op == '^':
                            result_val = float(val1 ** val2)
                        elif op.lower() == 'mod':
                            result_val = float(
                                val1 % val2) if val2 != 0 else float('nan')
                        elif op == '\\':
                            result_val = float(
                                val1 // val2) if val2 != 0 else float('nan')
                        else:
                            raise SyntaxError(
                                f"Unsupported array operator: {op} at line {line_number}")
                        row.append(result_val)
                    except ZeroDivisionError:
                        row.append(float('nan'))
                    except Exception as e:
                        raise RuntimeError(
                            f"Error computing {val1} {op} {val2} at position [{i}][{j}]: {e} at line {line_number}")
                result.append(row)
            return result

        # Pre-process curly brace expressions (e.g., D{i, 1} → evaluated_value)
        curly_brace_pattern = re.compile(r'([\w_]+)\{([^}]+)\}')
        curly_matches = curly_brace_pattern.findall(eval_expr)
        for var_name, indices_str in curly_matches:
            if var_name in scope or var_name in self.compiler.variables:
                # Parse indices (e.g., "i+1, 1" -> [i+1, 1])
                indices = []
                for index_expr in indices_str.split(','):
                    index_expr = index_expr.strip()
                    try:
                        # Evaluate the index expression (e.g., "i+1")
                        index_value = self.eval_expr(
                            index_expr, scope, line_number)
                        indices.append(index_value)
                    except Exception as e:
                        raise SyntaxError(
                            f"Invalid index expression '{index_expr}' in '{var_name}{{{indices_str}}}': {e} at line {line_number}")

                # Get the array
                array = scope.get(
                    var_name, self.compiler.variables.get(var_name))
                if array is None:
                    raise ValueError(
                        f"Variable '{var_name}' is uninitialized at line {line_number}")

                # Adjust indices based on dimension metadata
                dims = self.compiler.dimensions.get(var_name, [])
                # Also check constraints in scope for dimension-constrained arrays
                scope_constraints = None
                current_scope = self.compiler.current_scope()
                if hasattr(current_scope, 'constraints') and var_name in current_scope.constraints:
                    scope_constraints = current_scope.constraints[var_name].get(
                        'dim', [])
                else:
                    pass

                adjusted_indices = []
                for i, idx in enumerate(indices):
                    # Prefer scope constraints (dimension-constrained arrays)
                    if scope_constraints and i < len(scope_constraints):
                        constraint = scope_constraints[i]
                        if isinstance(constraint, tuple) and len(constraint) == 2 and isinstance(constraint[1], tuple):
                            base = constraint[1][0]
                            adjusted_idx = idx - base
                            adjusted_indices.append(adjusted_idx)
                        else:
                            adjusted_idx = idx - 1
                            adjusted_indices.append(adjusted_idx)
                    # range, e.g., (0, 10)
                    elif i < len(dims) and isinstance(dims[i][1], tuple):
                        base = dims[i][1][0]
                        adjusted_idx = idx - base
                        adjusted_indices.append(adjusted_idx)
                    else:  # just a size, treat as 1-based
                        adjusted_idx = idx - 1
                        adjusted_indices.append(adjusted_idx)

                try:
                    result = self.compiler.array_handler.get_array_element(
                        array, adjusted_indices, line_number)
                    # Replace the curly brace expression with the evaluated result
                    curly_expr = f"{var_name}{{{indices_str}}}"
                    eval_expr = eval_expr.replace(curly_expr, str(result))
                except (IndexError, ValueError) as e:
                    raise IndexError(
                        f"Invalid indices {indices_str} for '{var_name}': {e} at line {line_number}")

        # Replace operators for Python eval (mod → %, ^ → **, \ → //)
        eval_expr = self._replace_operators(eval_expr, line_number)

        # Get the full scope for evaluation
        if hasattr(scope, 'get_evaluation_scope'):
            full_scope = scope.get_evaluation_scope()
        else:
            full_scope = scope

        # Allow attribute-style access on dictionaries (e.g., p.x)
        def _wrap_value(val):
            if isinstance(val, dict):
                evaluator = self

                class DotDict(dict):
                    def __getattr__(self, item):
                        hidden = self.get('_hidden_fields', set())
                        if isinstance(hidden, (set, list, tuple)):
                            hidden_set = {str(h).lower() for h in hidden}
                        else:
                            hidden_set = set()
                        if str(item).lower() in hidden_set and not getattr(evaluator.compiler, '_allow_hidden_field_access', False):
                            raise PermissionError(
                                f"Hidden field '{item}' is not accessible at line {line_number}")
                        return self.get(item)
                wrapped = DotDict()
                for k, v in val.items():
                    wrapped[k] = _wrap_value(v)
                return wrapped
            if isinstance(val, list):
                return [_wrap_value(v) for v in val]
            return val

        full_scope = {k: _wrap_value(v) for k, v in full_scope.items()} if isinstance(
            full_scope, dict) else full_scope
        if isinstance(full_scope, dict):
            full_scope = CaseInsensitiveDict(full_scope)

        # Add built-in functions to the scope
        def rows(arr):
            """Return the length of an array or list."""
            if hasattr(arr, '__len__'):
                return len(arr)
            elif isinstance(arr, (list, tuple)):
                return len(arr)
            else:
                return 0

        def sum_func(*args):
            """Sum function that handles sum{a, b} syntax."""
            if len(args) == 1 and isinstance(args[0], str):
                # Handle sum{a, b} syntax
                if args[0].startswith('{') and args[0].endswith('}'):
                    return self._evaluate_sum_vars(f"sum{args[0]}", scope, line_number)
                # Handle sum[A1:B2] syntax
                elif args[0].startswith('[') and args[0].endswith(']'):
                    return self._evaluate_sum_range(f"sum{args[0]}", scope, line_number)
            # Handle regular sum of arguments
            return sum(args)

        full_scope['rows'] = rows
        full_scope['sum'] = sum_func

        # Evaluate the prepared expression using Python's eval
        try:
            globals_dict = self._get_eval_globals()
            result = eval(eval_expr, globals_dict, full_scope)
            if isinstance(result, (int, float)) and math.isnan(result):
                return float('nan')
            # Ensure array literals are returned as lists, not sets
            if isinstance(result, set):
                result = sorted(list(result))
            return float(result) if isinstance(result, (int, float)) else result
        except ZeroDivisionError:
            return float('nan')
        except NameError as e:
            raise NameError(f"{e} at line {line_number}")
        except Exception as e:
            raise RuntimeError(
                f"Error evaluating '{expr}': {e} at line {line_number}")

    def get_full_scope(self):
        """Get the full evaluation scope (alias to get_evaluation_scope)."""
        return self.get_evaluation_scope()

    def get_evaluation_scope(self):
        """Build and return the full scope by traversing parent scopes."""
        scope = {}
        current = self
        while current and not current.is_private:
            scope.update(
                {k: v for k, v in current.variables.items() if k not in scope})
            current = current.parent
        return scope

    def _process_interpolation(self, expr, scope, line_number=None):
        """
        Process string interpolation expressions like $"Hello {name}!"
        """
        if not expr.startswith('$"') or not expr.endswith('"'):
            raise ValueError(f"Invalid interpolation expression: {expr}")
        content = expr[2:-
                       1] if expr.startswith('$"') and expr.endswith('"') else expr
        res = []
        i = 0
        is_cell_ref = re.fullmatch(r'[A-Z]+\{[^}]+\}', content) is not None
        while i < len(content):
            if i + 3 < len(content) and content[i:i+4] == '{{{*':
                res.append('{{{')
                i += 4
                continue
            elif i + 2 < len(content) and content[i:i+3] == '{{*':
                res.append('{{')
                i += 3
                continue
            elif i + 1 < len(content) and content[i:i+2] == '{*':
                res.append('{')
                i += 2
                continue
            elif i + 1 < len(content) and content[i:i+2] == '{{':
                res.append('{')
                i += 2
                continue
            if content[i] == '{' and i + 1 < len(content):
                if content[i + 1] == '}':
                    res.append('{}')
                    i += 2
                    continue
                j = i + 1
                brace_level = 0
                in_quotes = False
                while j < len(content):
                    if content[j] == '"' and (j == 0 or content[j-1] != '"'):
                        in_quotes = not in_quotes
                    if not in_quotes:
                        if content[j] == '{':
                            brace_level += 1
                        elif content[j] == '}':
                            if brace_level == 0:
                                break
                            brace_level -= 1
                    j += 1
                if j < len(content):
                    inner_expr_full = content[i + 1:j].strip()
                    inner_expr_eval = inner_expr_full
                    pad = 0
                    format_spec = None
                    if ',' in inner_expr_full:
                        parts = inner_expr_full.rsplit(',', 1)
                        potential_expr, modifier = parts[0].strip(
                        ), parts[1].strip()
                        try:
                            pad = int(modifier)
                            inner_expr_eval = potential_expr
                        except ValueError:
                            if re.match(r'^\w+$', modifier):
                                format_spec = modifier
                                inner_expr_eval = potential_expr
                            else:
                                inner_expr_eval = inner_expr_full
                    val = self.eval_expr(
                        inner_expr_eval, scope, line_number) if inner_expr_eval else ""
                    if isinstance(val, (int, float)):
                        if isinstance(val, float) and val.is_integer():
                            val = int(val)
                        if is_cell_ref:
                            sval = str(val)
                            if not sval.isdigit():
                                raise ValueError(
                                    f"Invalid cell reference index '{sval}' must be an integer at line {line_number}")
                        else:
                            sval = str(val) if not isinstance(
                                val, int) else str(val)
                    else:
                        sval = str(val if val is not None else "")
                    if format_spec:
                        if format_spec == 'upper':
                            sval = sval.upper()
                        elif format_spec == 'lower':
                            sval = sval.lower()
                    if pad != 0:
                        sval = sval.rjust(pad) if pad > 0 else sval.ljust(-pad)
                    res.append(str(sval))
                    i = j + 1
                else:
                    res.append(str('{'))
                    i += 1
            else:
                if content[i] == '\n':
                    res.append('\n')
                elif content[i] == '\t':
                    res.append('\t')
                elif content[i] == '"' and i > 0 and content[i - 1] == '"':
                    i += 1
                    continue
                else:
                    res.append(str(content[i]))
                i += 1
        result = ''.join(res).replace('""', '"')
        return result

    def _build_sequence(self, expr, line_number=None):
        """
        Build a number sequence (e.g., 1 to 10 step 2).
        :param expr: Sequence expression.
        :param line_number: Line number.
        :return: Pyarrow array of sequence values.
        """
        m = re.match(
            r'^(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)(?:\s+step\s+(-?\d+(?:\.\d+)?))?$', expr, re.I)
        if m:
            start, end, step = m.groups()
            start = float(start)
            end = float(end)
            step = float(step) if step else 1.0
            count = int((end - start) / step) + 1
            return pa.array([start + i * step for i in range(count)], type=pa.float64())
        raise ValueError(f"Invalid sequence: '{expr}' at line {line_number}")

    def _replace_operators(self, expr, line_number=None):
        """
        Replace GridLang operators with Python equivalents for eval.
        :param expr: Expression.
        :param line_number: Line number.
        :return: Modified expression.
        """
        expr = re.sub(r'\bmod\b', '%', expr, flags=re.I)
        expr = expr.replace('\\', '//')
        expr = expr.replace('^', '**')
        return expr

    def _get_eval_globals(self):
        """
        Provide safe globals for eval, including math functions and builtins.
        :return: Globals dictionary.
        """
        def rows(arr):
            """Return the length of an array or list."""
            if hasattr(arr, '__len__'):
                return len(arr)
            elif isinstance(arr, (list, tuple)):
                return len(arr)
            else:
                return 0

        def Len(val):
            if isinstance(val, str):
                return len(val)
            if isinstance(val, pa.Array):
                items = val.to_pylist()
            elif isinstance(val, (list, tuple)):
                items = list(val)
            else:
                raise TypeError("Len expects text or an array of text values")
            lengths = []
            for item in items:
                if item is None:
                    lengths.append(0)
                elif isinstance(item, str):
                    lengths.append(len(item))
                else:
                    raise TypeError(
                        "Len expects text or an array of text values")
            return lengths

        def Mid(text, start, length=1):
            s = str(text)
            start_idx = max(int(start) - 1, 0)
            length = int(length)
            return s[start_idx:start_idx + length]

        def TextSplit(text, delimiter):
            return str(text).split(str(delimiter))

        def _to_list(val):
            if isinstance(val, pa.Array):
                return val.to_pylist()
            if isinstance(val, (list, tuple, set)):
                return list(val)
            return [val]

        def CountA(val):
            items = _to_list(val)
            count = 0
            for item in items:
                if isinstance(item, pa.Array):
                    count += CountA(item)
                elif item is None:
                    continue
                elif isinstance(item, str) and item == "":
                    continue
                else:
                    count += 1
            return count

        def RandArray(n):
            length = int(n)
            return [random.random() for _ in range(length)]

        def SortBy(arr, ord_vals):
            arr_list = _to_list(arr)
            ord_list = _to_list(ord_vals)
            if len(arr_list) != len(ord_list):
                raise ValueError("SortBy expects arrays of the same length")
            pairs = list(zip(ord_list, arr_list))
            pairs.sort(key=lambda p: p[0])
            return [v for _, v in pairs]

        globals_dict = {
            '__builtins__': {
                'float': float,
                'int': int,
                'str': str,
                'len': len,
                'sum': sum
            },
            'math': math,
            'pa': pa,
            'Len': Len,
            'Mid': Mid,
            'TextSplit': TextSplit,
            'CountA': CountA,
            'RandArray': RandArray,
            'SortBy': SortBy,
            'SQRT': math.sqrt,
            '_lookup_cell': self.compiler.array_handler.lookup_cell,
            'rows': rows
        }
        if hasattr(self.compiler, 'functions'):
            for fname, fdef in self.compiler.functions.items():
                wrapper = (lambda *a, _fname=fname:
                           self.compiler.call_function(_fname, a))
                globals_dict[fname] = wrapper
                original = fdef.get('original')
                if original:
                    globals_dict.setdefault(original, wrapper)
                    globals_dict.setdefault(original.lower(), wrapper)
                    globals_dict.setdefault(original.upper(), wrapper)
        if hasattr(self.compiler, 'subprocesses'):
            for sname, sdef in self.compiler.subprocesses.items():
                wrapper = (lambda *a, _sname=sname:
                           self.compiler.call_subprocess(_sname, a))
                globals_dict[sname] = wrapper
                original = sdef.get('original')
                if original:
                    globals_dict.setdefault(original, wrapper)
                    globals_dict.setdefault(original.lower(), wrapper)
                    globals_dict.setdefault(original.upper(), wrapper)
        return CaseInsensitiveDict(globals_dict)
