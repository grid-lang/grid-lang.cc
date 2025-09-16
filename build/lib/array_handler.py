# array_handler.py
# This module defines the ArrayHandler class, responsible for managing array operations,
# assignments to grid cells, range handling, flattening, and shape inference in the GridLang compiler.
# It supports pyarrow arrays, lists, and custom object flattening for grid spilling.

import re
import pyarrow as pa
# Utilities for cell and column operations
from utils import col_to_num, num_to_col, split_cell, offset_cell, validate_cell_ref, prod
from functools import reduce
import operator


class ArrayHandler:
    """
    Handler for array-related operations in GridLang, including assignments, indexing,
    range operations, flattening, and shape management.
    """

    def __init__(self, compiler):
        """
        Initialize the array handler with a reference to the compiler.
        :param compiler: The GridLangCompiler instance.
        """
        self.compiler = compiler

    def resolve_cell_index(self, var_name, cell_ref, line_number=None):
        """
        Resolve an array index using a cell reference (e.g., Results[B1]).
        Maps column letters to indices for dimensioned arrays.
        :param var_name: Variable name of the array.
        :param cell_ref: Cell reference (e.g., 'B1').
        :param line_number: Optional line number for error reporting.
        :return: Value at the resolved index.
        """
        try:
            arr = self.compiler.current_scope().get(var_name)
        except NameError:
            raise NameError(
                f"Variable '{var_name}' not defined at line {line_number}")
        # Convert to Python list for indexing
        if isinstance(arr, pa.Array):
            if not isinstance(arr, pa.ListArray):
                raise TypeError(
                    f"Variable '{var_name}' is not a multi-dimensional array at line {line_number}")
            arr_pylist = arr.to_pylist()
        elif isinstance(arr, list):
            arr_pylist = arr
        else:
            raise TypeError(
                f"Variable '{var_name}' is not an array at line {line_number}")

        # Validate and parse cell reference
        validate_cell_ref(cell_ref)
        col, _ = split_cell(cell_ref)
        col_num = col_to_num(col)  # 1-based column number (A=1, B=2, etc.)

        # Assume 0-based index for the second dimension (e.g., Quarter)
        quarter_index = col_num - 1

        # Get array dimensions
        dims = self.compiler.dimensions.get(var_name, [])
        shape = self.get_array_shape(arr, line_number)

        # Handle general 2D arrays
        if len(shape) == 2:
            rows, cols = shape
            # Convert cell reference to row and column indices
            col, row = split_cell(cell_ref)
            col_idx = col_to_num(col) - 1  # 0-based column index
            row_idx = int(row) - 1  # 0-based row index

            if row_idx < 0 or row_idx >= rows:
                raise IndexError(
                    f"Row index {row_idx + 1} out of bounds for dimension size {rows} at line {line_number}")
            if col_idx < 0 or col_idx >= cols:
                raise IndexError(
                    f"Column index {col_idx + 1} out of bounds for dimension size {cols} at line {line_number}")

            # Access the element
            try:
                if isinstance(arr_pylist, list) and len(arr_pylist) > row_idx:
                    if isinstance(arr_pylist[row_idx], list) and len(arr_pylist[row_idx]) > col_idx:
                        return arr_pylist[row_idx][col_idx]
                    else:
                        raise IndexError(
                            f"Column index {col_idx} out of bounds for row {row_idx} at line {line_number}")
                else:
                    raise IndexError(
                        f"Row index {row_idx} out of bounds at line {line_number}")
            except Exception as e:
                raise IndexError(
                    f"Error accessing '{var_name}' at index derived from '{cell_ref}': {e} at line {line_number}")
        else:
            raise ValueError(
                f"Expected 2D array for cell-based indexing, got shape {shape} at line {line_number}")

    def cell_ref_to_indices(self, cell_ref, line_number=None):
        """
        Convert a cell reference (e.g., 'A1', 'B1') to array indices for setting values.
        :param cell_ref: Cell reference (e.g., 'A1').
        :param line_number: Optional line number for error reporting.
        :return: List of indices [row_index, col_index] (0-based).
        """
        # Validate and parse cell reference
        validate_cell_ref(cell_ref)
        col, row = split_cell(cell_ref)
        col_idx = col_to_num(col) - 1  # 0-based column index (A=0, B=1, etc.)
        row_idx = int(row) - 1  # 0-based row index (1=0, 2=1, etc.)

        return [row_idx, col_idx]

    def evaluate_line_with_assignment(self, line, line_number=None, scope=None):
        """
        Evaluate an assignment line (e.g., [A1] := expr), handling various targets like cells,
        ranges, arrays, dimension selectors, and index selectors.
        :param line: Assignment line string.
        :param line_number: Line number.
        :param scope: Optional scope; defaults to global variables.
        """
        # Handle both := and = assignments
        if ':=' in line:
            target_part, expr_part = map(str.strip, line.split(':=', 1))
        elif '=' in line:
            target_part, expr_part = map(str.strip, line.split('=', 1))
        else:
            return
        if scope is None:
            scope = self.compiler.variables
        if not expr_part:
            assignment_op = ':=' if ':=' in line else '='
            raise SyntaxError(
                f"Missing expression after '{assignment_op}' at line {line_number}")

        # Check for pipe connections: output := input (creates a pipe)
        if self.compiler.current_scope().is_output(target_part) and not target_part.startswith('['):
            # This is an output variable being assigned to something
            # Check if the right side is an input variable or cell reference
            if self.compiler.current_scope().is_input(expr_part) or expr_part.startswith('['):
                # Create a pipe connection from output to input
                self.compiler.current_scope().connect_pipe(target_part, expr_part, line_number)
                return
        target, is_range, is_harr, is_dim_selector, is_index_selector = None, False, False, False, False
        sr, er = None, None
        var_name, dim_name, dim_index = None, None, None
        indices = None
        is_simple_var_assignment = False

        # Parse target type
        if '[' in target_part and ']' in target_part and not target_part.startswith('[') and '!' not in target_part:
            m = re.match(r'^([\w_]+)\[([^\]]*)\]$', target_part)
            if m:
                var_name, indices_str = m.groups()
                indices = [i.strip() for i in indices_str.split('][')]
                if var_name not in self.compiler.variables:
                    raise SyntaxError(
                        f"Variable '{var_name}' not defined at line {line_number}")
                is_index_selector = True
        elif '!' in target_part and '(' in target_part and ')' in target_part:
            m = re.match(r'^([\w_]+)!(\w+)\(([^)]+)\)$', target_part)
            if m:
                var_name, dim_name, index_str = m.groups()
                if var_name not in self.compiler.variables or var_name not in self.compiler.dimensions:
                    raise SyntaxError(
                        f"Variable '{var_name}' not defined or not dimensioned at line {line_number}")
                dim_names = self.compiler.dim_names.get(var_name, {})
                dim_idx = dim_names.get(dim_name)
                if dim_idx is None:
                    raise SyntaxError(
                        f"Dimension '{dim_name}' not defined for '{var_name}' at line {line_number}")
                labels = self.compiler.dim_labels.get(
                    var_name, {}).get(dim_name, {})
                index_str_clean = index_str.strip('"')
                dim_index = labels.get(index_str_clean, int(
                    index_str) - 1 if index_str.isdigit() else None)
                if dim_index is None:
                    raise ValueError(
                        f"Invalid index '{index_str}' for dimension '{dim_name}' at line {line_number}")
                is_dim_selector = True
        elif target_part.startswith('[') and target_part.endswith(']'):
            inside = target_part[1:-1].strip()
            if not inside:
                raise SyntaxError(f"Empty target '[]' at line {line_number}")
            if re.match(r'^[A-Za-z]+\d+$', inside):
                try:
                    validate_cell_ref(inside)
                    target = inside
                except ValueError as e:
                    raise SyntaxError(
                        f"Invalid cell reference '{inside}': {e} at line {line_number}")
            elif inside.startswith('@'):
                target = inside[1:].strip()
                if not target:
                    raise SyntaxError(
                        f"Invalid array target '[@]' at line {line_number}")
                try:
                    validate_cell_ref(target)
                    is_harr = True
                except ValueError as e:
                    raise SyntaxError(
                        f"Invalid array reference '{inside}': {e} at line {line_number}")
            elif ':' in inside:
                rm = re.match(r'^([A-Za-z]+\d+)\s*:\s*([A-Za-z]+\d+)$', inside)
                if rm:
                    sr, er = rm.groups()
                    try:
                        validate_cell_ref(sr)
                        validate_cell_ref(er)
                        is_range = True
                    except ValueError as e:
                        raise SyntaxError(
                            f"Invalid range references '{inside}': {e} at line {line_number}")
                else:
                    try:
                        final_inside = self.compiler.expr_evaluator._process_interpolation(
                            f'$"{inside}"', scope, line_number)
                        rm = re.match(
                            r'^([A-Za-z]+\d+)\s*:\s*([A-Za-z]+\d+)$', final_inside)
                        if rm:
                            sr, er = rm.groups()
                            validate_cell_ref(sr)
                            validate_cell_ref(er)
                            is_range = True
                        else:
                            raise SyntaxError(
                                f"Interpolated range '{final_inside}' is invalid at line {line_number}")
                    except Exception as e:
                        raise SyntaxError(
                            f"Error interpolating range '{inside}': {e} at line {line_number}")
            else:
                try:
                    final_inside = self.compiler.expr_evaluator._process_interpolation(
                        f'$"{inside}"', scope, line_number)
                    if re.match(r'^[A-Za-z]+\d+$', final_inside):
                        validate_cell_ref(final_inside)
                        target = final_inside
                    elif final_inside.startswith('@'):
                        target = final_inside[1:].strip()
                        validate_cell_ref(target)
                        is_harr = True
                    else:
                        raise SyntaxError(
                            f"Interpolated target '{final_inside}' is invalid at line {line_number}")
                except Exception as e:
                    raise SyntaxError(
                        f"Error interpolating '{inside}': {e} at line {line_number}")
        else:
            # Handle simple variable assignments (e.g., result := 10)
            if re.match(r'^[\w_]+$', target_part):
                var_name = target_part
                # Check if variable exists in any scope
                defining_scope = self.compiler.current_scope().get_defining_scope(var_name)
                if not defining_scope:
                    raise NameError(
                        f"Variable '{var_name}' not defined at line {line_number}")
                # Set flag to indicate this is a simple variable assignment
                is_simple_var_assignment = True
            else:
                raise SyntaxError(
                    f"Invalid assignment target: '{target_part}' at line {line_number}")

        # Evaluate RHS
        try:
            # Check if this is an array operation by looking for array literals
            # Array literals start with { and contain comma-separated values
            array_literal_pattern = r'\{[^}]*,[^}]*\}'
            array_literals = re.findall(array_literal_pattern, expr_part)

            if '+' in expr_part and len(array_literals) >= 2:
                value = self.evaluate_array_operation(expr_part, line_number)
            else:
                value = self.compiler.expr_evaluator.eval_or_eval_array(
                    expr_part, scope, line_number)
        except Exception as e:
            raise

        # Handle simple variable assignments (e.g., result := 10)
        if is_simple_var_assignment:
            # Update the variable in its defining scope
            defining_scope.update(var_name, value, line_number)
            return

        # Handle special reshaping for array of two-field objects in horizontal assignment
        is_array_of_two_field_objects = False
        object_type_name = None
        values_per_object = 0
        reshaped_value = value
        if is_harr and expr_part.startswith('{') and expr_part.endswith('}'):
            inner = expr_part[1:-1].strip()
            elements = [elem.strip()
                        for elem in inner.split(',') if elem.strip()]
            # Check if all elements exist in scope
            all_elements_exist = True
            element_values = {}
            for elem in elements:
                try:
                    element_values[elem] = self.compiler.current_scope().get(
                        elem)
                except NameError:
                    all_elements_exist = False
                    break

            if all_elements_exist:
                first_elem = element_values[elements[0]]
                if isinstance(first_elem, dict):
                    for type_name, fields in self.compiler.types_defined.items():
                        if set(first_elem.keys()) == set(fields.keys()) and len(fields) == 2:
                            object_type_name = type_name
                            values_per_object = 2
                            break
                if object_type_name:
                    is_array_of_two_field_objects = all(
                        isinstance(element_values[elem], dict) and
                        set(element_values[elem].keys()) == set(
                            self.compiler.types_defined[object_type_name].keys())
                        for elem in elements
                    )
            if is_array_of_two_field_objects:
                if isinstance(value, list) and value and isinstance(value[0], list):
                    flat_inner = value[0]
                    num_objects = len(elements)
                    if len(flat_inner) == num_objects * values_per_object:
                        reshaped_value = [
                            flat_inner[i * values_per_object:(i + 1) * values_per_object] for i in range(num_objects)]

        # Perform assignment based on target type
        if is_index_selector:
            result = self._assign_index_selector(
                var_name, indices, value, line_number)
            if value is None:  # Read operation
                return result
        elif is_dim_selector:
            result = self._assign_dim_selector(
                var_name, dim_name, dim_index, value, line_number)
            if value is None:  # Read operation
                return result
        elif is_harr:
            self._assign_horizontal_array(
                target, reshaped_value, expr_part, is_array_of_two_field_objects, line_number)
        elif is_range:
            self.assign_range(sr, er, value, line_number)
        else:
            if isinstance(value, pa.Array):
                value = value.to_pylist()
            elif isinstance(value, pa.Scalar):
                py_value = value.as_py()
                value = int(py_value) if isinstance(
                    py_value, float) and py_value.is_integer() else py_value
            self.compiler.grid[target] = value

    def evaluate_array_operation(self, expr, line_number=None):
        """
        Evaluate array operations like {1,2} + {3,4}.
        Supports only addition for now.
        :param expr: Operation string.
        :param line_number: Line number.
        :return: Result array as list of lists.
        """
        parts = []
        current = ""
        brace_level = 0
        for char in expr:
            if char == '+' and brace_level == 0:
                if current.strip():
                    parts.append(current.strip())
                current = ""
            else:
                current += char
                if char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
        if current.strip():
            parts.append(current.strip())
        if len(parts) != 2:
            raise SyntaxError(
                f"Expected exactly two arrays for operation, got {len(parts)} at line {line_number}")
        left_expr, right_expr = parts
        if not (left_expr.startswith('{') and right_expr.startswith('{')):
            raise SyntaxError(
                f"Invalid array operation: {expr} at line {line_number}")
        left_array = self.evaluate_array(left_expr, line_number)
        right_array = self.evaluate_array(right_expr, line_number)
        left_shape = self.get_array_shape(left_array, line_number)
        right_shape = self.get_array_shape(right_array, line_number)
        if left_shape != right_shape:
            raise ValueError(
                f"Array shape mismatch: {left_shape} vs {right_shape} at line {line_number}")
        result = []
        for i in range(left_shape[0]):
            row = []
            for j in range(left_shape[1]):
                row.append(left_array[i][j] + right_array[i][j])
            result.append(row)
        return result

    def evaluate_array(self, expr, line_number=None):
        """
        Evaluate an inline array (e.g., {1,2;3,4}).
        :param expr: Array string.
        :param line_number: Line number.
        :return: List of lists representing the array.
        """
        expr = expr.strip()
        if not (expr.startswith('{') and expr.endswith('}')):
            raise SyntaxError(
                f"Invalid array syntax: {expr} at line {line_number}")
        inner = expr[1:-1].strip()
        if not inner:
            return []
        rows = [row.strip() for row in inner.split(';')]
        if not rows:
            return []
        values = []
        for row in rows:
            row_values = []
            row_items = []
            current = ""
            brace_level = 0
            for char in row + ',':
                if char == ',' and brace_level == 0:
                    if current.strip():
                        row_items.append(current.strip())
                    current = ""
                else:
                    current += char
                    if char == '{':
                        brace_level += 1
                    elif char == '}':
                        brace_level -= 1
            row_items = [item for item in row_items if item]
            for item in row_items:
                try:
                    value = self.compiler.expr_evaluator.eval_or_eval_array(
                        item, self.compiler.variables, line_number)
                    if isinstance(value, pa.Scalar):
                        value = value.as_py()
                        value = int(value) if isinstance(
                            value, float) and value.is_integer() else value
                    row_values.append(value)
                except Exception as e:
                    raise RuntimeError(
                        f"Error evaluating array element '{item}': {e} at line {line_number}")
            values.append(row_values)
        row_lengths = [len(row) for row in values]
        if len(set(row_lengths)) > 1:
            raise ValueError(
                f"Inconsistent row lengths in array: {row_lengths} at line {line_number}")
        return values

    def _assign_index_selector(self, var_name, indices, value, line_number=None):
        """
        Assign or read from an array using index selector (e.g., var[1]).
        Handles cell-based and numeric indices.
        :param var_name: Array variable.
        :param indices: List of indices.
        :param value: Value to assign (None for read).
        :param line_number: Line number.
        :return: Read value if value is None.
        """
        try:
            arr = self.compiler.current_scope().get(var_name)
        except NameError:
            raise NameError(
                f"Variable '{var_name}' not defined at line {line_number}")
        shape = self.get_array_shape(arr, line_number)

        # Handle cell reference index
        if len(indices) == 1 and re.match(r'^[A-Za-z]+\d+$', indices[0]):
            cell_ref = indices[0]
            col_str, _ = split_cell(cell_ref)
            col_idx = col_to_num(col_str) - 1

            if len(shape) != 2:
                raise ValueError(
                    f"Expected 2-dimensional array for cell-based indexing, got shape {shape} at line {line_number}")

            if col_idx < 0 or col_idx >= shape[1]:
                raise ValueError(
                    f"Column index {col_idx} out of bounds for dimension size {shape[1]} at line {line_number}")

            # Assume first dimension size 1
            if value is None:  # Read
                result = arr[0][col_idx]
                return result
            else:  # Write
                flat_value = self.flatten_array(value, line_number)
                flat_arr = arr.flatten().to_pylist()
                inner_size = shape[1]
                outer_size = shape[0]
                flat_arr = flat_arr + [0] * \
                    (inner_size * outer_size - len(flat_arr))
                idx = col_idx
                flat_arr[idx] = float(flat_value[0]) if flat_value else 0
                new_values = pa.array(flat_arr, type=pa.float64())
                self.compiler.current_scope().update(var_name, pa.ListArray.from_arrays(
                    offsets=pa.array(
                        [i * inner_size for i in range(outer_size + 1)], type=pa.int32()),
                    values=new_values
                ))

        else:
            # Numeric indices
            if len(indices) != len(shape):
                raise ValueError(
                    f"Expected {len(shape)} indices, got {len(indices)} at line {line_number}")
            flat_value = self.flatten_array(value, line_number)
            inner_size = shape[1] if len(shape) > 1 else 1
            outer_size = shape[0] if shape else 10
            flat_arr = arr.flatten().to_pylist() if isinstance(
                arr, pa.ListArray) else arr.to_pylist() + [0] * (inner_size * outer_size - len(arr))
            idx = 0
            for i, index in enumerate(indices):
                try:
                    idx_val = int(index) - 1
                except ValueError:
                    raise ValueError(
                        f"Invalid index '{index}' for dimension {i} at line {line_number}")
                if idx_val < 0 or idx_val >= shape[i]:
                    raise ValueError(
                        f"Index {idx_val + 1} out of bounds at line {line_number}")
                idx += idx_val * (inner_size if i == 0 else 1)
            if value is None:  # Read
                result = flat_arr[idx]
                return result
            flat_arr[idx] = float(flat_value[0]) if flat_value else 0
            new_values = pa.array(flat_arr, type=pa.float64())
            if len(shape) > 1:
                self.compiler.current_scope().update(var_name, pa.ListArray.from_arrays(
                    offsets=pa.array(
                        [i * inner_size for i in range(outer_size + 1)], type=pa.int32()),
                    values=new_values
                ))
            else:
                self.compiler.current_scope().update(var_name, new_values)

    def _assign_dim_selector(self, var_name, dim_name, dim_index, value, line_number=None):
        """
        Assign or read from a dimension selector (e.g., var!dim(1)).
        :param var_name: Array variable.
        :param dim_name: Dimension name.
        :param dim_index: Index in dimension.
        :param value: Value to assign (None for read).
        :param line_number: Line number.
        :return: Read value if value is None.
        """
        try:
            arr = self.compiler.current_scope().get(var_name)
        except NameError:
            raise NameError(
                f"Variable '{var_name}' not defined at line {line_number}")
        shape = self.get_array_shape(arr, line_number)
        flat_value = self.flatten_array(
            value, line_number) if value is not None else []
        dim_idx = self.compiler.dim_names[var_name].get(dim_name)
        if dim_idx is None:
            raise SyntaxError(
                f"Dimension '{dim_name}' not found at line {line_number}")
        if value is not None and len(flat_value) != shape[1 - dim_idx]:
            raise ValueError(
                f"Value size {len(flat_value)} does not match dimension size {shape[1 - dim_idx]} at line {line_number}")

        flat_arr = arr.flatten().to_pylist()
        inner_size = shape[1] if len(shape) > 1 else 1
        outer_size = shape[0] if shape else max(10, dim_index + 1)
        flat_arr = flat_arr + [0] * (inner_size * outer_size - len(flat_arr))

        if value is None:  # Read
            if dim_idx == 0:
                start_idx = dim_index * inner_size
                result = flat_arr[start_idx:start_idx + inner_size]
                return result
            else:
                result = [flat_arr[i * inner_size + dim_index]
                          for i in range(shape[0])]
                return result[0] if len(result) == 1 else result

        # Assignment
        if dim_idx == 0:
            start_idx = dim_index * inner_size
            for i, val in enumerate(flat_value):
                flat_arr[start_idx +
                         i] = float(val) if isinstance(val, (int, float)) else val
        else:
            for i in range(shape[0]):
                flat_arr[i * inner_size + dim_index] = float(flat_value[i]) if isinstance(
                    flat_value[i], (int, float)) else flat_value[i]
        new_values = pa.array(flat_arr, type=pa.float64())
        self.compiler.current_scope().update(var_name, pa.ListArray.from_arrays(
            offsets=pa.array(
                [i * inner_size for i in range(outer_size + 1)], type=pa.int32()),
            values=new_values
        ))

    def _assign_horizontal_array(self, target, value, expr_part, is_array_of_two_field_objects=False, line_number=None):
        """
        Assign an array horizontally (or vertically) to the grid starting at target cell.
        Handles objects, lists, pyarrow arrays, and reshaping for two-field objects.
        :param target: Starting cell (e.g., 'A1').
        :param value: Array or list to assign.
        :param expr_part: Original expression for orientation check.
        :param is_array_of_two_field_objects: Flag for special reshaping.
        :param line_number: Line number.
        """
        if isinstance(value, dict):
            flattened_values = self.flatten_object_fields(value, line_number)
            for i, val in enumerate(flattened_values):
                cell_to_assign = offset_cell(target, i, 0)
                self.compiler.grid[cell_to_assign] = val
        elif isinstance(value, list):
            is_object_array = False
            type_name = None
            if all(isinstance(item, dict) for item in value):
                for t_name, fields in self.compiler.types_defined.items():
                    if all(set(fields.keys()) == set(item.keys()) for item in value):
                        is_object_array = True
                        type_name = t_name
                        break
            if is_object_array:
                for row_idx, item in enumerate(value):
                    flattened_values = self.flatten_object_fields(
                        item, line_number)
                    for col_idx, val in enumerate(flattened_values):
                        cell_to_assign = offset_cell(target, col_idx, row_idx)
                        self.compiler.grid[cell_to_assign] = val
            elif is_array_of_two_field_objects:
                num_objects = len(value)
                start_col = col_to_num(
                    target[0:re.search(r'\d', target).start()])
                start_row = int(target[re.search(r'\d', target).start():])
                for row_idx in range(num_objects):
                    object_values = value[row_idx]
                    for col_idx in range(len(object_values)):
                        new_cell = f"{num_to_col(start_col + col_idx)}{start_row + row_idx}"
                        self.compiler.grid[new_cell] = float(
                            object_values[col_idx])
            else:
                is_vertical = ';' in expr_part.strip(
                )[1:-1] and ',' not in expr_part.strip()[1:-1]
                flattened_values = self.flatten_array(value, line_number)
                for i, val in enumerate(flattened_values):
                    cell_to_assign = offset_cell(
                        target, 0, i) if is_vertical else offset_cell(target, i, 0)
                    self.compiler.grid[cell_to_assign] = val
        elif isinstance(value, pa.Array):
            is_vertical = ';' in expr_part.strip(
            )[1:-1] and ',' not in expr_part.strip()[1:-1]
            shape = self.get_array_shape(value, line_number)
            flattened_values = value.flatten().to_pylist() if isinstance(
                value, pa.ListArray) else value.to_pylist()
            if len(shape) > 1:
                for row_idx in range(shape[0]):
                    for col_idx in range(shape[1]):
                        cell_to_assign = offset_cell(target, col_idx, row_idx)
                        self.compiler.grid[cell_to_assign] = flattened_values[row_idx *
                                                                              shape[1] + col_idx]
            else:
                for i, val in enumerate(flattened_values):
                    cell_to_assign = offset_cell(
                        target, 0, i) if is_vertical else offset_cell(target, i, 0)
                    self.compiler.grid[cell_to_assign] = val
        else:
            self.compiler.grid[target] = value

    def assign_range(self, sr_ref, er_ref, vals, line_number=None):
        """
        Assign values to a range of cells (e.g., A1:B2 := {1,2;3,4}).
        Handles scalars, 1D/2D arrays, repeating, and cycling.
        :param sr_ref: Start cell (e.g., 'A1').
        :param er_ref: End cell (e.g., 'B2').
        :param vals: Values to assign (scalar, list, array).
        :param line_number: Line number.
        """
        scs, sro = split_cell(sr_ref)
        ecs, ero = split_cell(er_ref)
        sc, ec = col_to_num(scs), col_to_num(ecs)
        sr, er = int(sro), int(ero)
        num_cols = ec - sc + 1
        num_rows = er - sr + 1
        if num_cols < 1 or num_rows < 1:
            raise ValueError(
                f"Invalid range: {num_cols}x{num_rows} at line {line_number}")

        shape = self.get_array_shape(vals, line_number) if isinstance(
            vals, (pa.Array, list)) else [1]
        flat_vals = self.flatten_array(vals, line_number) if isinstance(
            vals, (pa.Array, list)) else [vals]

        if isinstance(vals, (list, pa.Array, pa.ListArray)):
            effective_shape = shape
            if len(shape) == 2 and shape[0] == 1:
                effective_shape = [shape[1]]

            if len(effective_shape) == 1:
                array_length = effective_shape[0]
                if num_rows > 1 and num_cols == 1:  # Vertical assignment
                    for i, r in enumerate(range(sr, er + 1)):
                        cell = num_to_col(sc) + str(r)
                        value = flat_vals[i % len(flat_vals)]
                        self.compiler.grid[cell] = value
                elif num_cols > 1 and num_rows == 1:  # Horizontal assignment
                    for i, c in enumerate(range(sc, ec + 1)):
                        cell = num_to_col(c) + str(sr)
                        value = flat_vals[i % len(flat_vals)]
                        self.compiler.grid[cell] = value
                elif array_length == num_cols:  # Repeat across rows
                    for r in range(sr, er + 1):
                        for c in range(sc, ec + 1):
                            col_idx = c - sc
                            cell = num_to_col(c) + str(r)
                            value = flat_vals[col_idx]
                            self.compiler.grid[cell] = value
                else:  # Cycle over the range
                    idx = 0
                    for r in range(sr, er + 1):
                        for c in range(sc, ec + 1):
                            cell = num_to_col(c) + str(r)
                            value = flat_vals[idx % len(flat_vals)]
                            self.compiler.grid[cell] = value
                            idx += 1
            elif len(effective_shape) > 1:
                if effective_shape[0] == num_rows and effective_shape[1] == num_cols:
                    idx = 0
                    for r in range(sr, sr + effective_shape[0]):
                        for c in range(sc, sc + effective_shape[1]):
                            cell = num_to_col(c) + str(r)
                            value = flat_vals[idx]
                            self.compiler.grid[cell] = value
                            idx += 1
                elif effective_shape[0] == num_cols and effective_shape[1] == num_rows and num_cols == 1:
                    reshaped = [[flat_vals[i]] for i in range(len(flat_vals))]
                    idx = 0
                    for r in range(sr, sr + num_rows):
                        for c in range(sc, sc + num_cols):
                            cell = num_to_col(c) + str(r)
                            value = reshaped[idx][0]
                            self.compiler.grid[cell] = value
                            idx += 1
                elif effective_shape[0] == num_rows and effective_shape[1] == num_cols == 1:
                    idx = 0
                    for r in range(sr, sr + effective_shape[0]):
                        cell = num_to_col(sc) + str(r)
                        value = flat_vals[idx]
                        self.compiler.grid[cell] = value
                        idx += 1
                else:
                    raise ValueError(
                        f"Array shape {effective_shape} exceeds range ({num_rows}x{num_cols}) at line {line_number}")
        else:  # Scalar assignment to range
            for r in range(sr, er + 1):
                for c in range(sc, ec + 1):
                    cell = num_to_col(c) + str(r)
                    value = flat_vals[0]
                    self.compiler.grid[cell] = value

    def get_range_values(self, s_cell, e_cell, line_number=None):
        """
        Retrieve values from a grid range (e.g., A1:B2).
        Returns 1D list for single row/column, 2D list otherwise.
        :param s_cell: Start cell.
        :param e_cell: End cell.
        :param line_number: Line number.
        :return: List of values.
        """
        scs, sro = split_cell(s_cell)
        ecs, ero = split_cell(e_cell)
        sc, ec = col_to_num(scs), col_to_num(ecs)
        sr, er = int(sro), int(ero)
        values = []
        if sc == ec:  # Single column
            for r in range(min(sr, er), max(sr, er) + 1):
                cell_ref = num_to_col(sc) + str(r)
                values.append([self.lookup_cell(cell_ref, line_number)])
        elif sr == er:  # Single row
            row_values = []
            for c in range(min(sc, ec), max(sc, ec) + 1):
                cell_ref = num_to_col(c) + str(sr)
                row_values.append(self.lookup_cell(cell_ref, line_number))
            values = row_values
        else:  # 2D range
            for r in range(min(sr, er), max(sr, er) + 1):
                row_values = []
                for c in range(min(sc, ec), max(sc, ec) + 1):
                    cell_ref = num_to_col(c) + str(r)
                    row_values.append(self.lookup_cell(cell_ref, line_number))
                values.append(row_values)
        return values

    def lookup_cell(self, cell_ref, line_number=None):
        """
        Lookup value in a grid cell, default to 0 if unset.
        :param cell_ref: Cell reference.
        :param line_number: Line number.
        :return: Cell value.
        """
        # Case-insensitive lookup
        cell_ref_upper = cell_ref.upper()
        for key, value in self.compiler.grid.items():
            if key.upper() == cell_ref_upper:
                return value
        return 0

    def flatten_object_fields(self, obj, line_number=None):
        """
        Flatten fields of a custom object into a list, handling nested objects.
        :param obj: Object (dict).
        :param line_number: Line number.
        :return: Flattened list of field values.
        """
        result = []
        if isinstance(obj, dict):
            type_name = None
            for name, fields in self.compiler.types_defined.items():
                if set(fields.keys()) == set(obj.keys()):
                    type_name = name
                    break
            if type_name:
                fields = self.compiler.types_defined[type_name.lower()]
                for field in fields.keys():
                    value = obj[field]
                    if isinstance(value, dict):
                        nested_type = None
                        for n_name, n_fields in self.compiler.types_defined.items():
                            if set(n_fields.keys()) == set(value.keys()):
                                nested_type = n_name
                                break
                        if nested_type:
                            nested_fields = self.compiler.types_defined[nested_type]
                            for n_field in nested_fields.keys():
                                result.append(value[n_field])
                        else:
                            result.extend([value[k]
                                          for k in sorted(value.keys())])
                    else:
                        result.append(value)
            else:
                result.extend([obj[k] for k in sorted(obj.keys())])
        else:
            result.append(obj)
        return result

    def flatten_array(self, arr, line_number=None):
        """
        Flatten an array (pyarrow or list) into a 1D list.
        :param arr: Array or list.
        :param line_number: Line number.
        :return: Flattened list.
        """
        if isinstance(arr, pa.ListArray):
            return arr.flatten().to_pylist()
        if isinstance(arr, pa.Array):
            return arr.to_pylist()
        if isinstance(arr, list):
            result = []
            for item in arr:
                if isinstance(item, list):
                    result.extend(self.flatten_array(item, line_number))
                else:
                    result.append(item)
            return result
        return [arr]

    def infer_type(self, value, line_number=None):
        """
        Infer type of a value (array, number, text, object, etc.).
        :param value: Value to infer.
        :param line_number: Line number.
        :return: Inferred type string.
        """
        if isinstance(value, pa.ListArray) or (isinstance(value, list) and value and isinstance(value[0], list)):
            return 'array'
        if isinstance(value, (list, pa.Array)):
            return 'array'
        if isinstance(value, (int, float)):
            return 'number' if isinstance(value, float) else 'int'
        if isinstance(value, bool):
            return 'bool'
        if isinstance(value, str):
            return 'text'
        if isinstance(value, dict):
            return 'object'
        return 'unknown'

    def get_array_shape(self, arr, line_number=None):
        """
        Get the shape of an array (supports up to 3D pyarrow ListArrays or lists).
        :param arr: Array.
        :param line_number: Line number.
        :return: List of dimension sizes.
        """
        if isinstance(arr, pa.ListArray):
            if not pa.types.is_list(arr.type):
                return []
            shape = []
            d1 = len(arr)
            shape.append(d1)
            if d1 == 0:
                return shape
            l2_array = arr.values
            if not pa.types.is_list(l2_array.type):
                offsets = arr.offsets.to_pylist()
                if len(offsets) <= 1:
                    return [0]
                first_len = offsets[1] - offsets[0]
                return [len(offsets) - 1, first_len]
            d2 = len(l2_array) // d1
            shape.append(d2)
            if d2 == 0:
                return shape
            l3_array = l2_array.values
            if not pa.types.is_list(l3_array.type):
                d3 = len(l3_array) // len(l2_array)
                shape.append(d3)
                return shape
            raise ValueError(
                f"Unsupported array dimensions: more than 3D at line {line_number}")
        elif isinstance(arr, pa.Array):
            return [len(arr)]
        elif isinstance(arr, list):
            if not arr:
                return [0]
            if isinstance(arr[0], list):
                return [len(arr), len(arr[0])]
            return [len(arr)]
        else:
            return [1]

    def create_array(self, shape, default_value, pa_type, line_number=None, matrix_data=None, is_grid_dim=False):

        # Handle 0D array (scalar)
        if not shape:
            return pa.array([default_value], type=pa_type)

        # Calculate flat size
        flat_size = 1
        for dim in shape:
            flat_size *= dim

        # Initialize values
        if is_grid_dim and matrix_data:
            if len(matrix_data) != shape[-1]:
                raise ValueError(
                    f"Expected {shape[-1]} matrices for last dimension, got {len(matrix_data)} at line {line_number}")
            values = []
            # For grid DIM arrays, we need to interleave the depths correctly
            # Each row in the final array should contain all depths for that row
            for row_idx in range(shape[0]):
                for depth_idx in range(shape[-1]):
                    matrix = matrix_data[depth_idx]
                    if len(matrix) != shape[0]:
                        raise ValueError(
                            f"Expected {shape[0]} rows per matrix, got {len(matrix)} at line {line_number}")
                    row = matrix[row_idx]
                    if len(row) != shape[1]:
                        raise ValueError(
                            f"Expected {shape[1]} columns per row, got {len(row)} at line {line_number}")
                    for val in row:
                        if default_value is not None and float(val) != float(default_value):
                            raise ValueError(
                                f"Value {val} violates constraint {default_value} at line {line_number}")
                        values.append(val)
        else:
            if is_grid_dim and default_value is None:
                raise ValueError(
                    f"No matrix data provided for grid at line {line_number}")
            if isinstance(pa_type, pa.StructType):
                values = [
                    default_value if default_value is not None else None] * flat_size
            elif pa_type == pa.float64():
                values = [
                    float(default_value) if default_value is not None else 0.0] * flat_size
            elif pa_type == pa.string():
                values = [str(default_value)
                          if default_value is not None else ""] * flat_size
            else:
                values = [
                    default_value if default_value is not None else None] * flat_size

        # Create array based on dimensions
        if len(shape) == 1:
            return pa.array(values, type=pa_type)
        elif len(shape) >= 2:
            rows = shape[0]
            cols = shape[1] * \
                shape[2] if len(shape) == 3 and is_grid_dim else shape[1]
            offsets = pa.array(
                [i * cols for i in range(rows + 1)], type=pa.int32())
            flat_array = pa.array(values, type=pa_type)
            result = pa.ListArray.from_arrays(offsets, flat_array)
            return {'array': result, 'original_shape': shape} if is_grid_dim else result
        else:
            raise ValueError(
                f"Unsupported array dimensions: {shape} at line {line_number}")

    def set_labels(self, var_name, dim_name, labels, line_number=None):
        """
        Set labels for a named dimension in an array.
        :param var_name: Variable name.
        :param dim_name: Dimension name.
        :param labels: List of labels.
        :param line_number: Line number.
        """
        if var_name not in self.compiler.dim_names:
            raise SyntaxError(
                f"Variable '{var_name}' has no named dimensions at line {line_number}")
        dim_idx = self.compiler.dim_names[var_name].get(dim_name)
        if dim_idx is None:
            raise SyntaxError(
                f"Dimension '{dim_name}' not found in '{var_name}' at line {line_number}")
        self.compiler.dim_labels.setdefault(var_name, {})
        self.compiler.dim_labels[var_name][dim_name] = {
            lbl: i for i, lbl in enumerate(labels)}

    def check_dimension_constraints(self, var, value, line_number=None):
        """
        Check and reshape value to match dimension constraints of the variable.
        Infers '*' sizes and broadcasts scalars.
        :param var: Variable name.
        :param value: Value to check/reshape.
        :param line_number: Line number.
        :return: Reshaped value.
        """
        if var not in self.compiler.dimensions:
            return value
        dims = self.compiler.dimensions[var]

        # Handle scalar broadcasting
        if not isinstance(value, (list, pa.Array, pa.ListArray)) or isinstance(value, (int, float, str)):
            shape = []
            num_stars = sum(1 for _, size_spec in dims if size_spec is None)
            if num_stars > 0:
                for _, size_spec in dims:
                    if isinstance(size_spec, tuple):
                        start, end = size_spec
                        dim_size = end - start + 1
                    elif size_spec is None:
                        dim_size = 1
                    else:
                        dim_size = size_spec
                    shape.append(dim_size)
            else:
                for _, size_spec in dims:
                    if isinstance(size_spec, tuple):
                        start, end = size_spec
                        dim_size = end - start + 1
                    else:
                        dim_size = size_spec
                    shape.append(dim_size)
            pa_type = pa.float64() if self.compiler.types.get(
                var) in ('number', 'array') else pa.string()
            return self.create_array(shape, value, pa_type, line_number)

        # Compute expected shape
        expected_shape = []
        star_indices = [i for i, (_, size_spec) in enumerate(
            dims) if size_spec is None]
        known_product = 1
        for i, (_, size_spec) in enumerate(dims):
            if isinstance(size_spec, tuple):
                start, end = size_spec
                dim_size = end - start + 1
            elif size_spec is None:
                dim_size = None
            else:
                dim_size = size_spec
            expected_shape.append(dim_size)
            if dim_size is not None:
                known_product *= dim_size

        # Get actual shape and flat values
        shape = self.get_array_shape(value, line_number)
        flat_vals = self.flatten_array(value, line_number)
        total_elements = len(flat_vals)

        # Infer '*' sizes
        if star_indices:
            if len(star_indices) == 1:
                star_idx = star_indices[0]
                if known_product == 0:
                    expected_shape[star_idx] = 1
                else:
                    inferred_size = total_elements // known_product
                    if total_elements % known_product != 0:
                        raise ValueError(
                            f"Cannot infer '*' dimension size for '{var}': total elements {total_elements} not divisible by known dimensions product {known_product} at line {line_number}")
                    expected_shape[star_idx] = inferred_size
            else:
                raise ValueError(
                    f"Cannot infer sizes for multiple '*' dimensions in '{var}' at line {line_number}")

        # Check total elements
        expected_total = 1
        for dim in expected_shape:
            expected_total *= dim
        if total_elements != expected_total:
            raise ValueError(
                f"Element count mismatch for '{var}': expected {expected_total} elements, got {total_elements} at line {line_number}")

        # Reshape to expected shape
        if len(expected_shape) == 1:
            return flat_vals
        reshaped = []
        stride = expected_shape[1] if len(expected_shape) > 1 else 1
        for i in range(expected_shape[0]):
            start = i * stride
            row = flat_vals[start:start + stride]
            reshaped.append(row)
        return reshaped

    def set_array_element(self, array, indices, value, line_number=None):
        """
        Set an element in a pyarrow array of any dimension.
        :param array: Pyarrow array (ListArray or simple array).
        :param indices: List of indices corresponding to array dimensions.
        :param value: Value to set.
        :param line_number: Line number for error reporting.
        :return: Updated array.
        """

        # Handle simple arrays (like DoubleArray)
        if not isinstance(array, pa.ListArray):
            if len(indices) == 1:
                # For 1D arrays, convert to list, update, and convert back
                arr_list = array.to_pylist()
                arr_list[indices[0]] = float(value) if isinstance(
                    value, (int, float)) else value
                return pa.array(arr_list, type=array.type)
            else:
                raise TypeError(
                    f"Expected ListArray for multi-dimensional arrays, got {type(array)} at line {line_number}")

        # Handle ListArray (existing logic)
        shape = self.get_array_shape(array, line_number)
        if len(indices) != len(shape):
            raise ValueError(
                f"Expected {len(shape)} indices, got {len(indices)} at line {line_number}")
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= shape[i]:
                raise IndexError(
                    f"Index {idx} out of bounds for dimension {i} with size {shape[i]} at line {line_number}")

        # For 2D arrays, use row-major indexing
        if len(shape) == 2:
            rows, cols = shape
            flat_idx = indices[0] * cols + indices[1]
        else:
            # Calculate flat index for other dimensions
            flat_idx = 0
            stride = 1
            for i in range(len(shape) - 1, -1, -1):
                flat_idx += indices[i] * stride
                stride *= shape[i]

        # Get flat array and update value
        flat_arr = array.flatten().to_pylist()
        flat_arr[flat_idx] = float(value) if isinstance(
            value, (int, float)) else value

        # For 2D arrays, reconstruct using the original shape
        if len(shape) == 2:
            rows, cols = shape
            offsets = [i * cols for i in range(rows + 1)]
            result = pa.ListArray.from_arrays(
                pa.array(offsets, type=pa.int32()),
                pa.array(flat_arr, type=pa.float64())
            )
            return result
        else:
            # For other dimensions, return the flat array
            return pa.array(flat_arr, type=pa.float64())

    def reshape_array(self, arr, new_dims, line_number=None):
        """
        Reshape an array to new dimensions, padding with zeros if needed.
        :param arr: Array to reshape.
        :param new_dims: New dimension specs.
        :param line_number: Line number.
        :return: Reshaped pyarrow array.
        """
        flat = self.flatten_array(arr, line_number)
        shape = []
        for d in new_dims:
            if d is None:
                shape.append(
                    len(flat) if not shape else len(flat) // shape[-1])
            else:
                size = d[1] if isinstance(d, tuple) else d
                shape.append(size)
        total_size = 1
        for s in shape:
            total_size *= s
        flat.extend([0] * (total_size - len(flat)))
        if len(shape) > 1:
            return pa.ListArray.from_arrays(
                offsets=pa.array([i * shape[1]
                                 for i in range(shape[0] + 1)], type=pa.int32()),
                values=pa.array(flat, type=pa.float64())
            )
        return pa.array(flat[:shape[0]], type=pa.float64())

    def get_array_element(self, arr, indices, line_number=None, return_struct=False, original_shape=None):
        """
        Get an element from an array using indices.
        :param arr: Array or dict with 'array' key.
        :param indices: List of indices.
        :param line_number: Line number.
        :param return_struct: Return struct if True.
        :param original_shape: Original shape if provided.
        :return: Element value.
        """
        # Handle dictionary with array (e.g., from grid DIM)
        if isinstance(arr, dict) and 'array' in arr and isinstance(arr['array'], pa.ListArray):
            original_shape = arr.get(
                'original_shape') if original_shape is None else original_shape
            arr = arr['array']

        # Get the shape to use for indexing - prefer original_shape if available
        shape = self.get_array_shape(arr, line_number)
        indexing_shape = original_shape if original_shape is not None else getattr(
            arr, 'original_shape', shape)

        # Validate indices against the actual array shape (not the original shape)
        if len(indices) != len(indexing_shape):
            raise ValueError(
                f"Expected {len(indexing_shape)} indices for array with shape {indexing_shape}, got {len(indices)} at line {line_number}")

        # For grid DIM arrays, validate against the original shape since that's what the user expects
        validation_shape = indexing_shape
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= validation_shape[i]:
                raise IndexError(
                    f"Index {idx} out of bounds for dimension {i} with size {validation_shape[i]} at line {line_number}")

        # Handle array indexing
        if isinstance(arr, pa.ListArray):
            flat_arr = arr.flatten().to_pylist()

            # For grid DIM arrays, the array is stored as 2D but should be accessed as 3D
            # Check if this is a grid DIM array by looking at the original shape
            if original_shape and len(original_shape) == 3 and len(shape) == 2:
                # This is a 3D array stored as 2D (grid DIM case)
                # Calculate the flat index using the original 3D shape
                row = indices[0]
                # For 3D arrays flattened to 2D, each row contains all depths for each column
                # The layout is: [depth1_col0, depth1_col1, ..., depth1_colN, depth2_col0, depth2_col1, ..., depth2_colN]
                # So for indices [row, col, depth], the position is: row * (cols * depths) + col + depth * cols
                col = indices[1] + indices[2] * original_shape[1]
                flat_idx = row * shape[1] + col
            elif len(indexing_shape) == 2:
                # 2D array indexing
                row, col = indices
                flat_idx = row * indexing_shape[1] + col
            elif len(indexing_shape) == 3:
                # 3D array indexing
                flat_idx = indices[0] * indexing_shape[1] * \
                    indexing_shape[2] + indices[1] * \
                    indexing_shape[2] + indices[2]
            else:
                # N-dimensional array indexing
                flat_idx = 0
                for i, idx in enumerate(indices):
                    flat_idx = flat_idx * indexing_shape[i] + idx

            if flat_idx < 0 or flat_idx >= len(flat_arr):
                raise IndexError(
                    f"Calculated index {flat_idx} out of bounds for array length {len(flat_arr)} at line {line_number}")

            result = flat_arr[flat_idx]
            if isinstance(result, dict) and 'value' in result:
                return result['value']
            return result
        elif isinstance(arr, pa.Array):
            return arr[indices[0]].as_py()
        elif isinstance(arr, list):
            result = arr
            for idx in indices:
                result = result[idx]
            return result
        else:
            raise TypeError(
                f"Cannot index non-array type {type(arr)} at line {line_number}")

    def fill_array(self, array, value, line_number=None):
        """
        Fill an existing array with a value (scalar or list).
        :param array: Array to fill.
        :param value: Fill value.
        :param line_number: Line number.
        :return: Filled pyarrow array.
        """
        if not isinstance(array, (pa.Array, pa.ListArray)):
            raise ValueError(
                f"Expected pyarrow Array or ListArray, got {type(array)} at line {line_number}")

        shape = self.get_array_shape(array, line_number)
        flat_size = 1
        for dim in shape:
            flat_size *= dim

        def flatten(lst):
            for el in lst:
                if isinstance(el, (list, tuple, set)):
                    yield from flatten(el)
                else:
                    yield el

        if isinstance(value, (list, tuple, set)):
            values = [float(v) for v in flatten(value)]
        else:
            values = [float(value)] * flat_size

        if len(shape) == 1:
            return pa.array(values, type=pa.float64())

        inner_size = shape[1]
        outer_size = shape[0]
        offsets = [i * inner_size for i in range(outer_size + 1)]
        return pa.ListArray.from_arrays(
            offsets=pa.array(offsets, type=pa.int32()),
            values=pa.array(values, type=pa.float64())
        )

    def get_array_dimensions(self, arr):
        """
        Get the dimensions of an array.
        """
        if not isinstance(arr, (list, tuple)):
            return []

        def get_dims_recursive(item):
            if not isinstance(item, (list, tuple)) or len(item) == 0:
                return []

            dims = [len(item)]
            if len(item) > 0 and isinstance(item[0], (list, tuple)):
                # Recursively get dimensions of the first element
                sub_dims = get_dims_recursive(item[0])
                dims.extend(sub_dims)

            return dims

        return get_dims_recursive(arr)
