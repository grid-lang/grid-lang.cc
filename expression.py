# expression.py
# This module defines the ExpressionEvaluator class, responsible for evaluating expressions,
# arrays, ranges, and special functions in the GridLang compiler. It handles various syntax
# including binary operations, dimension constraints, sums, interpolations, and more.

import re
import math
import pyarrow as pa
# Utility for validating cell references
from utils import validate_cell_ref


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
            full_scope = self.compiler.current_scope().get_full_scope()
            if var_name in full_scope:
                tensor = full_scope.get(var_name)
                if tensor is None:
                    raise ValueError(
                        f"Variable '{var_name}' is uninitialized at line {line_number}")
                if isinstance(tensor, dict) and 'grid' in tensor and 'original_shape' in tensor:
                    array = tensor['grid']
                    original_shape = tensor['original_shape']
                    try:
                        result = self.compiler.array_handler.get_array_element(
                            array, indices, line_number, is_grid_dim=True, original_shape=original_shape)
                        return result
                    except (IndexError, ValueError) as e:
                        raise IndexError(
                            f"Invalid indices {indices_str} for '{var_name}.grid': {e} at line {line_number}")
                else:
                    raise TypeError(
                        f"Variable '{var_name}' does not have a 'grid' or 'original_shape' field at line {line_number}")

        # Handle field access (e.g., var.field)
        if is_grid_dim and re.match(r'^[\w_]+\.\w+$', expr):
            var, field = expr.split('.')
            full_scope = self.compiler.current_scope().get_full_scope()
            if f"{var}.{field}" in full_scope:
                return full_scope[f"{var}.{field}"]
            if var in full_scope:
                tensor = full_scope.get(var)
                if isinstance(tensor, dict) and field in tensor:
                    return tensor[field]
                if isinstance(tensor, pa.ListArray):
                    shape = self.compiler.array_handler.get_array_shape(
                        tensor, line_number)
                    zero_indices = [1] * len(shape)
                    first_element = self.compiler.array_handler.get_array_element(
                        tensor, zero_indices, line_number, return_struct=True, is_grid_dim=True)
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
            return self._evaluate_array(expr, scope, line_number)

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
                    self.compiler.variables[var_name] = flat_values
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
        full_scope = self.compiler.current_scope().get_full_scope()
        if expr in full_scope:
            return full_scope[expr]

        # Fallback to general expression evaluation if not grid_dim
        if not is_grid_dim:
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
        full_scope = {**scope, **{v: self.compiler.variables[v]
                                  for c, v in self.compiler._cell_var_map.items() if v in self.compiler.variables}}
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
                            flattened = self.compiler.array_handler.flatten_object_fields(
                                evaluated, line_number)
                            row_elements.append(flattened)
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

    def eval_expr(self, expr, scope, line_number=None):
        """
        Evaluate a general expression, handling ranges, sums, grid indexing, interpolations,
        sequences, and mathematical operations.
        :param expr: Expression string.
        :param scope: Scope.
        :param line_number: Line number.
        :return: Evaluated result.
        """
        expr = expr.strip()
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
                array = full_scope.get(var_name)
                if array is None:
                    raise ValueError(
                        f"Variable '{var_name}' is uninitialized at line {line_number}")
                try:
                    result = self.compiler.array_handler.get_array_element(
                        array, indices, line_number)
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

        # Handle cell-based indexing (e.g., var[A1])
        m = re.match(r'^([\w_]+)\[([A-Z]+\d+)\]$', expr)
        if m:
            var_name, cell_ref = m.groups()
            if var_name in scope or var_name in self.compiler.variables:
                return self.compiler.array_handler.resolve_cell_index(var_name, cell_ref, line_number)

        # Handle dimension selector (e.g., var!dim("label"))
        if '!' in expr and '(' in expr and ')' in expr:
            m = re.match(r'^([\w_]+)!(\w+)\(([^)]+)\)$', expr)
            if m:
                var_name, dim_name, index_str = m.groups()
                if var_name not in self.compiler.variables:
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
                array = self.compiler.variables[var_name]
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

        # Handle array indexing (e.g., var{1,2})
        match = re.match(r'^[\w_]+\{\d+(\s*,\s*\d+)*\}$', expr)
        if match:
            var_name, indices_str = expr.split('{', 1)
            var_name = var_name.strip()
            indices = [int(i.strip()) for i in indices_str[:-1].split(',')]
            if var_name in scope:
                arr = scope[var_name]
            elif var_name in self.compiler.variables:
                arr = self.compiler.variables[var_name]
            else:
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

        # Handle object creation (e.g., new Type{val1, val2})
        if re.match(r'^new\s+(\w+)\s*\{', expr):
            match = re.match(r'^new\s+(\w+)\s*\{(.+)\}$', expr)
            if match:
                type_name, args = match.groups()
                args_list = []
                current_arg = ""
                brace_level = 0
                for char in args + ',':
                    if char == ',' and brace_level == 0:
                        if current_arg.strip():
                            args_list.append(current_arg.strip())
                        current_arg = ""
                    else:
                        current_arg += char
                        if char == '{':
                            brace_level += 1
                        elif char == '}':
                            brace_level -= 1
                args_list = [a for a in args_list if a.strip()]
                evaluated_args = [self.eval_or_eval_array(
                    a, scope, line_number) for a in args_list]
                if type_name.lower() in self.compiler.types_defined:
                    fields = self.compiler.types_defined[type_name.lower()]
                    expected_args = len(fields)
                    if len(evaluated_args) != expected_args:
                        raise ValueError(
                            f"Expected {expected_args} args for {type_name}, got {len(evaluated_args)} at line {line_number}")
                    return dict(zip(fields.keys(), evaluated_args))
                else:
                    raise ValueError(
                        f"Type '{type_name}' not defined at line {line_number}")

        # Handle field access (e.g., var.field)
        if re.match(r'^[\w_]+\.\w+$', expr):
            var, field = expr.split('.')
            if var in scope and isinstance(scope[var], dict):
                return scope[var].get(field)
            if var in self.compiler.variables and isinstance(self.compiler.variables[var], dict):
                return self.compiler.variables[var].get(field)

        # Handle single cell reference (e.g., [A1])
        if expr.startswith('[') and expr.endswith(']') and not ':' in expr:
            cell_ref = expr[1:-1].strip()
            if re.match(r'^[A-Z]+\d+$', cell_ref):
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
        full_scope = {**scope, **{v: self.compiler.variables[v]
                                  for c, v in self.compiler._cell_var_map.items() if v in self.compiler.variables}}
        cell_refs = re.findall(r'\[([A-Z]+\d+)\]', eval_expr)
        for cell_ref in cell_refs:
            try:
                validate_cell_ref(cell_ref)
                value = self.compiler.array_handler.lookup_cell(
                    cell_ref, line_number)
                if isinstance(value, (int, float)):
                    eval_expr = eval_expr.replace(f'[{cell_ref}]', str(value))
                else:
                    eval_expr = eval_expr.replace(
                        f'[{cell_ref}]', f'"{value}"')
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

        # Replace operators for Python eval (mod → %, ^ → **, \ → //)
        eval_expr = self._replace_operators(eval_expr, line_number)

        # Evaluate the prepared expression using Python's eval
        try:
            result = eval(eval_expr, self._get_eval_globals(), full_scope)
            if isinstance(result, (int, float)) and math.isnan(result):
                return float('nan')
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
        Process interpolated string ($"...{expr}..."), handling placeholders,
        padding, formatting, and escaping.
        :param expr: Interpolated string.
        :param scope: Scope.
        :param line_number: Line number.
        :return: Processed string.
        """
        content = expr[2:-
                       1] if expr.startswith('$"') and expr.endswith('"') else expr
        res = []
        i = 0
        is_cell_ref = re.match(r'^[A-Z]+', content) is not None
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
                        if val.is_integer():
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
                    res.append(sval)
                    i = j + 1
                else:
                    res.append('{')
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
                    res.append(content[i])
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
        return {
            '__builtins__': {
                'float': float,
                'int': int,
                'str': str,
                'len': len,
                'sum': sum
            },
            'math': math,
            'SQRT': math.sqrt,
            '_lookup_cell': self.compiler.array_handler.lookup_cell
        }
