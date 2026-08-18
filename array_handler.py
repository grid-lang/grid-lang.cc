# array_handler.py
# This module defines the ArrayHandler class, responsible for managing array operations,
# assignments to grid cells, range handling, flattening, and shape inference in the GridLang compiler.
# It supports flat Python lists (with a shape dict) for bounded arrays, plain dicts
# keyed by index tuples for unbounded arrays, and custom object flattening for grid spilling.

import re
import copy
# Utilities for cell and column operations
from utils import col_to_num, num_to_col, split_cell, offset_cell, validate_cell_ref, prod, public_type_fields, object_public_keys, public_object_view, is_address, parse_address, indices_to_address, _ADDRESS_FRAGMENT, is_sparse_array
from functools import reduce
import operator
import itertools
from units import DIM_ERROR, NA_ERROR, REF_ERROR, TYPE_ERROR, UNIVERSAL_ZERO, ConstraintError, error_value, is_error_value


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
        :param cell_ref: Cell reference as a 0-based index tuple (e.g. (0, 0));
            a (None, col) tuple selects a whole grid column.
        :param line_number: Optional line number for error reporting.
        :return: Value at the resolved index.
        """
        try:
            arr = self.compiler.current_scope().get(var_name)
        except NameError:
            raise NameError(
                f"Variable '{var_name}' not defined at line {line_number}")
        if var_name.lower() == 'grid':
            # grid![cell] reads a cell directly from the grid array;
            # grid![A] reads a whole column (a (None, col) index tuple
            # where None selects every row of the column).
            if (isinstance(cell_ref, (tuple, list))
                    and len(cell_ref) == 2 and cell_ref[0] is None):
                return self.get_grid_column(
                    cell_ref[1], grid_source=arr, line_number=line_number)
            if isinstance(cell_ref, (tuple, list)):
                key = tuple(int(i) for i in cell_ref)
                if len(key) > 2:
                    # Extended addresses read the grid as N-D.
                    return self.lookup_cell(key, line_number)
                row, col = key
                return self.get_array_element(
                    arr, [row, col], line_number, var_name=var_name)
            if '.' in cell_ref:
                # Extended addresses read the grid as N-D.
                return self.lookup_cell(cell_ref, line_number)
            row, col = parse_address(cell_ref)[:2]
            return self.get_array_element(
                arr, [row - 1, col - 1], line_number, var_name=var_name)
        # Convert to Python list for indexing
        nd_shape = None
        sparse_arr = False
        if isinstance(arr, dict) and 'array' in arr:
            nd_shape = list(arr.get('shape') or arr.get('original_shape') or [])
            arr = arr['array']
        else:
            sparse_arr = is_sparse_array(arr) or (
                isinstance(arr, dict) and not arr)
        if isinstance(arr, list) or sparse_arr:
            arr_pylist = arr
        else:
            raise TypeError(
                f"Variable '{var_name}' is not an array at line {line_number}")

        # Extended (N-D) addresses resolve directly to array indices.
        if isinstance(cell_ref, (tuple, list)) and len(cell_ref) > 2:
            indices = [i + 1 for i in cell_ref]
            return self.get_array_element(
                arr, [i - 1 for i in indices], line_number,
                original_shape=nd_shape, var_name=var_name)

        # Sparse arrays are unbound: any cell is a valid address and an
        # unset one reads as #N/A (or the declared default).
        if isinstance(cell_ref, (tuple, list)):
            row_idx, col_idx = cell_ref
            if sparse_arr:
                value = arr.get((row_idx, col_idx))
                if value is None:
                    return self._array_unset_value(var_name, line_number)
                return value
        else:
            # Validate and parse cell reference
            parse_address(cell_ref)
            col, _ = split_cell(cell_ref)
            col_to_num(col)  # 1-based column number (A=1, B=2, etc.)

        if isinstance(cell_ref, str) and sparse_arr:
            col, row = split_cell(cell_ref)
            col_idx = col_to_num(col) - 1  # 0-based column index
            row_idx = int(row) - 1  # 0-based row index
            value = arr.get((row_idx, col_idx))
            if value is None:
                return self._array_unset_value(var_name, line_number)
            return value

        # Get array dimensions
        dims = self.compiler.dimensions.get(var_name, [])
        shape = self.get_array_shape(arr, line_number)
        if nd_shape is not None:
            shape = nd_shape

        # Handle general 2D arrays
        if len(shape) == 2:
            rows, cols = shape
            # Convert cell reference to row and column indices
            if isinstance(cell_ref, (tuple, list)):
                row_idx, col_idx = cell_ref
            else:
                col, row = split_cell(cell_ref)
                col_idx = col_to_num(col) - 1  # 0-based column index
                row_idx = int(row) - 1  # 0-based row index

            if row_idx < 0 or row_idx >= rows:
                raise IndexError(
                    f"Row index {row_idx + 1} out of bounds for dimension size {rows} at line {line_number}")
            if col_idx < 0 or col_idx >= cols:
                raise IndexError(
                    f"Column index {col_idx + 1} out of bounds for dimension size {cols} at line {line_number}")

            # Access the element.  The grid maps the fastest (first) declared
            # dim to grid rows and the slowest (last) dim to grid columns, so a
            # cell reference (row, col) maps to indices [row_idx, col_idx] and
            # the flat buffer is column-major: flat = row + rows*col.
            if nd_shape is not None:
                flat_idx = row_idx + rows * col_idx
                if flat_idx < 0 or flat_idx >= len(arr_pylist):
                    raise IndexError(
                        f"Calculated index {flat_idx} out of bounds for array length {len(arr_pylist)} at line {line_number}")
                return arr_pylist[flat_idx]
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

    def read_array_range(self, arr, s_addr, e_addr, line_number=None):
        """Read a hyper-rectangle sub-block from an N-D array variable.

        ``b![A1.1:A2.2]`` returns the sub-block in dict-form flat storage
        ({'array': flat values, 'shape': sub-block shape}).  Works for arrays
        of any rank stored as flat column-major buffers
        ({'array'/'grid' + shape}) or as nested lists.
        ``s_addr``/``e_addr`` are 0-based index tuples.
        """
        s_idx = [i + 1 for i in s_addr]
        e_idx = [i + 1 for i in e_addr]
        if len(s_idx) != len(e_idx):
            raise ValueError(
                f"Range addresses '{s_addr}' and '{e_addr}' must have the same rank at line {line_number}")
        flat = None
        nd_shape = None
        if isinstance(arr, dict) and ('array' in arr or 'grid' in arr):
            inner = arr.get('array', arr.get('grid'))
            if isinstance(inner, list):
                flat = inner
            nd_shape = list(arr.get('shape') or arr.get('original_shape') or [])
        shape = nd_shape if nd_shape is not None else self.get_array_shape(
            arr, line_number)
        if len(shape) != len(s_idx):
            raise ValueError(
                f"Address rank {len(s_idx)} does not match array rank {len(shape)} at line {line_number}")
        starts = [min(a, b) for a, b in zip(s_idx, e_idx)]
        rshape = [abs(a - b) + 1 for a, b in zip(s_idx, e_idx)]
        if flat is not None:
            strides = [1]
            for s in shape[:-1]:
                strides.append(strides[-1] * s)
            values = []
            for flat_i in range(prod(rshape)):
                rem = flat_i
                idxs = []
                for dim_size in rshape:
                    idxs.append(rem % dim_size)
                    rem //= dim_size
                pos = sum((starts[k] - 1 + idxs[k]) * strides[k]
                          for k in range(len(rshape)))
                values.append(flat[pos])
            return {'array': values, 'shape': rshape, 'original_shape': rshape}
        def rec(node, k):
            if k == len(rshape) - 1:
                return [node[starts[k] - 1 + d] for d in range(rshape[k])]
            return [rec(node[starts[k] - 1 + d], k + 1)
                    for d in range(rshape[k])]
        return {'array': self.flatten_array(rec(arr, 0)), 'shape': rshape, 'original_shape': rshape}

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
        assignment_op = None
        # Handle both := and = assignments
        if ':=' in line:
            assignment_op = ':='
            target_part, expr_part = map(str.strip, line.split(':=', 1))
        elif '=' in line:
            assignment_op = '='
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
        target_info = self._parse_assignment_target_details(
            target_part,
            scope,
            line_number,
        )
        target = target_info['target']
        is_range = target_info['is_range']
        is_harr = target_info['is_harr']
        is_dim_selector = target_info['is_dim_selector']
        is_index_selector = target_info['is_index_selector']
        sr = target_info['sr']
        er = target_info['er']
        var_name = target_info['var_name']
        dim_name = target_info['dim_name']
        dim_index = target_info['dim_index']
        indices = target_info['indices']

        implicit_expr = None
        if expr_part.lstrip().startswith('@'):
            implicit_expr = expr_part.lstrip()[1:].strip()
            if not implicit_expr:
                raise SyntaxError(
                    f"Missing expression after '@' at line {line_number}")

        value = None
        # Evaluate RHS
        if implicit_expr is None:
            try:
                # Check if this is an array operation by looking for array literals
                # Array literals start with { and contain comma-separated values.
                # A { after an identifier is element indexing (e.g., arr{1, 2} or
                # grid{row, col}), not an array literal.
                array_literal_pattern = r'(?<![\w_])\{[^}]*,[^}]*\}'
                array_literals = re.findall(
                    array_literal_pattern, expr_part)

                if '+' in expr_part and len(array_literals) >= 2:
                    value = self.evaluate_array_operation(
                        expr_part, line_number)
                else:
                    value = self.compiler.expr_evaluator.eval_or_eval_array(
                        expr_part, scope, line_number)
            except Exception as e:
                raise

        is_array_of_two_field_objects, reshaped_value = self._prepare_horizontal_reshape_value(
            implicit_expr,
            value,
            expr_part,
            is_harr,
        )

        return self._perform_assignment_write(
            assignment_op=assignment_op,
            target=target,
            is_range=is_range,
            is_harr=is_harr,
            is_dim_selector=is_dim_selector,
            is_index_selector=is_index_selector,
            sr=sr,
            er=er,
            var_name=var_name,
            dim_name=dim_name,
            dim_index=dim_index,
            indices=indices,
            value=value,
            expr_part=expr_part,
            implicit_expr=implicit_expr,
            scope=scope,
            line_number=line_number,
            reshaped_value=reshaped_value,
        )

    def _prepare_horizontal_reshape_value(self, implicit_expr, value, expr_part, is_harr):
        if implicit_expr is not None:
            return False, None
        reshaped_value = value
        is_array_of_two_field_objects, reshaped_value = self._reshape_horizontal_two_field_object_array(
            value,
            expr_part,
            is_harr,
        )
        return is_array_of_two_field_objects, reshaped_value

    def _parse_assignment_target_details(self, target_part, scope, line_number):
        target, is_range, is_harr, is_dim_selector, is_index_selector = None, False, False, False, False
        sr, er = None, None
        var_name, dim_name, dim_index = None, None, None
        indices = None

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
                    try:
                        eval_scope = self.compiler.current_scope().get_evaluation_scope()
                        evaluated_index = self.compiler.expr_evaluator.eval_expr(
                            index_str, eval_scope, line_number)
                    except Exception as exc:
                        raise ValueError(
                            f"Invalid index '{index_str}' for dimension '{dim_name}' at line {line_number}") from exc
                    if isinstance(evaluated_index, str):
                        if evaluated_index in labels:
                            dim_index = labels[evaluated_index]
                        elif evaluated_index.isdigit():
                            dim_index = int(evaluated_index) - 1
                        else:
                            raise ValueError(
                                f"Invalid index '{evaluated_index}' for dimension '{dim_name}' at line {line_number}")
                    elif isinstance(evaluated_index, (int, float)):
                        dim_index = int(evaluated_index) - 1
                    else:
                        raise ValueError(
                            f"Invalid index '{evaluated_index}' for dimension '{dim_name}' at line {line_number}")
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
                    parse_address(inside)
                    target = inside
                except ValueError as e:
                    raise SyntaxError(
                        f"Invalid cell reference '{inside}': {e} at line {line_number}")
            elif inside.startswith('^'):
                target = inside[1:].strip()
                if not target:
                    raise SyntaxError(
                        f"Invalid array target '[^]' at line {line_number}")
                if is_address(target):
                    is_harr = True
                else:
                    try:
                        validate_cell_ref(target)
                        is_harr = True
                    except ValueError as e:
                        raise SyntaxError(
                            f"Invalid array reference '{inside}': {e} at line {line_number}")
            elif is_address(inside):
                # Extended (dotted) single-cell target, e.g. [A1.B1].
                target = inside
            else:
                resolved = self.compiler.expr_evaluator._resolve_column_interpolated_cell(
                    inside, scope, line_number)
                if resolved:
                    target = resolved
                elif ':' in inside:
                    rm = re.match(
                        rf'^({_ADDRESS_FRAGMENT})\s*:\s*({_ADDRESS_FRAGMENT})$', inside)
                    if rm:
                        sr, er = rm.groups()
                        try:
                            parse_address(sr)
                            parse_address(er)
                            is_range = True
                        except ValueError as e:
                            raise SyntaxError(
                                f"Invalid range references '{inside}': {e} at line {line_number}")
                    else:
                        try:
                            final_inside = self.compiler.expr_evaluator._process_interpolation(
                                f'$"{inside}"', scope, line_number)
                            rm = re.match(
                                rf'^({_ADDRESS_FRAGMENT})\s*:\s*({_ADDRESS_FRAGMENT})$', final_inside)
                            if rm:
                                sr, er = rm.groups()
                                parse_address(sr)
                                parse_address(er)
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
                            parse_address(final_inside)
                            target = final_inside
                        elif is_address(final_inside):
                            target = final_inside
                        elif final_inside.startswith('^'):
                            target = final_inside[1:].strip()
                            if is_address(target):
                                is_harr = True
                            else:
                                validate_cell_ref(target)
                                is_harr = True
                        else:
                            raise SyntaxError(
                                f"Interpolated target '{final_inside}' is invalid at line {line_number}")
                    except Exception as e:
                        raise SyntaxError(
                            f"Error interpolating '{inside}': {e} at line {line_number}")
        else:
            raise SyntaxError(
                f"Invalid assignment target: '{target_part}' at line {line_number}. "
                "Use [address] := value.")

        if target is not None and isinstance(target, str):
            target = self.compiler._to_index(target)
        if sr is not None and isinstance(sr, str):
            sr = self.compiler._to_index(sr)
        if er is not None and isinstance(er, str):
            er = self.compiler._to_index(er)

        return {
            'target': target,
            'is_range': is_range,
            'is_harr': is_harr,
            'is_dim_selector': is_dim_selector,
            'is_index_selector': is_index_selector,
            'sr': sr,
            'er': er,
            'var_name': var_name,
            'dim_name': dim_name,
            'dim_index': dim_index,
            'indices': indices,
        }

    def _perform_assignment_write(
            self,
            assignment_op,
            target,
            is_range,
            is_harr,
            is_dim_selector,
            is_index_selector,
            sr,
            er,
            var_name,
            dim_name,
            dim_index,
            indices,
            value,
            expr_part,
            implicit_expr,
            scope,
            line_number,
            reshaped_value):
        target_disp = indices_to_address(
            [i + 1 for i in target]) if isinstance(target, tuple) else str(target)
        if is_index_selector:
            result = self._assign_index_selector(
                var_name, indices, value, line_number)
            if value is None:
                return result
            return None

        if is_dim_selector:
            result = self._assign_dim_selector(
                var_name, dim_name, dim_index, value, line_number)
            if value is None:
                return result
            return None

        if is_harr:
            if implicit_expr is not None:
                raise SyntaxError(
                    f"Implicit intersection '@' is not supported for horizontal array targets at line {line_number}")
            self._assign_horizontal_array(
                target, reshaped_value, expr_part, line_number)
            if (assignment_op == ':=' and
                    re.match(r'^[A-Za-z_][\w_]*$', expr_part) and
                    not getattr(self.compiler.current_scope(), 'is_private', False)):
                source_var = expr_part
                try:
                    self.compiler.current_scope().get(source_var)
                    existing = self.compiler._cell_array_map.get(target)
                    if existing and existing.lower() != source_var.lower():
                        raise SyntaxError(
                            f"Cell '{target_disp}' already mapped to '{existing}' at line {line_number}")
                    conflict = self.compiler._cell_var_map.get(target)
                    if conflict and conflict.lower() != source_var.lower():
                        raise SyntaxError(
                            f"Cell '{target_disp}' already mapped to '{conflict}' at line {line_number}")
                    self.compiler._cell_array_map[target] = source_var
                except Exception:
                    pass
            return None

        if is_range:
            if implicit_expr is not None:
                self.assign_implicit_intersection_range(
                    sr, er, implicit_expr, scope, line_number)
            else:
                self.assign_range(sr, er, value, line_number)
            return None

        if implicit_expr is not None:
            row, col = target
            value = self._evaluate_implicit_intersection(
                implicit_expr, row + 1, scope, line_number)
        if isinstance(value, dict) and 'array' in value:
            if not is_harr and isinstance(target, tuple):
                self._assign_horizontal_array(
                    target, value, expr_part, line_number)
                return None
            value = self.to_display_value(value)
        elif isinstance(value, list) and isinstance(target, tuple):
            if is_harr:
                self._assign_horizontal_array(
                    target, value, expr_part, line_number)
                return None
            is_obj_array = (
                value
                and all(isinstance(item, dict) for item in value)
                and self._find_object_array_type(value) is not None
            )
            if is_obj_array:
                type_name = self._find_object_array_type(value)
                for i, item in enumerate(value):
                    self.compiler._set_grid_cell(
                        offset_cell(target, i, 0),
                        public_object_view(item))
                return None
            self._assign_horizontal_array(
                target, value, expr_part, line_number)
            return None
        elif isinstance(value, dict) and value and all(isinstance(k, tuple) for k in value.keys()):
            if isinstance(target, tuple):
                self._assign_horizontal_array(
                    target, value, expr_part, line_number)
                return None
        # ``[A1] := x`` is sugar for ``Let grid![A1] = x``: the grid store is
        # the single backing store, so the cell write below IS the grid write.
        if self._update_bound_array_cell(target, value, line_number):
            self.compiler._set_grid_cell(target, value)
            return None
        bound_var = self.compiler._cell_var_map.get(target)
        if bound_var:
            defining_scope = self.compiler.current_scope().get_defining_scope(
                bound_var)
            if defining_scope:
                defining_scope.update(bound_var, value, line_number)
            else:
                inferred_type = self.compiler.array_handler.infer_type(
                    value, line_number)
                if inferred_type == 'int':
                    inferred_type = 'number'
                self.compiler.current_scope().define(
                    bound_var, value, inferred_type, {}, is_uninitialized=False)
        if (assignment_op == ':=' and
                re.match(r'^[A-Za-z_][\w_]*$', expr_part) and
                not getattr(self.compiler.current_scope(), 'is_private', False)):
            source_var = expr_part
            try:
                self.compiler.current_scope().get(source_var)
                existing = self.compiler._cell_var_map.get(target)
                if existing and existing.lower() != source_var.lower():
                    raise SyntaxError(
                        f"Cell '{target_disp}' already mapped to '{existing}' at line {line_number}")
                self.compiler._cell_var_map[target] = source_var
            except Exception:
                pass
        if isinstance(value, dict) and ('_type_name' in value or '_hidden_fields' in value):
            value = public_object_view(value)
        self.compiler._set_grid_cell(target, self.to_display_value(value))
        return None

    def _reshape_horizontal_two_field_object_array(self, value, expr_part, is_harr):
        if not (is_harr and expr_part.startswith('{') and expr_part.endswith('}')):
            return False, value

        elements = self._parse_inline_object_elements(expr_part)
        element_values = self._resolve_inline_object_values(elements)
        if element_values is None:
            return False, value

        object_type_name = self._find_two_field_object_type(
            element_values[elements[0]])
        if not object_type_name:
            return False, value

        expected_fields = set(
            public_type_fields(self.compiler.types_defined[object_type_name]).keys())
        is_array_of_two_field_objects = all(
            isinstance(element_values[elem], dict)
            and object_public_keys(element_values[elem]) == expected_fields
            for elem in elements
        )
        if not is_array_of_two_field_objects:
            return False, value

        reshaped_value = self._reshape_flat_object_values(
            value, len(elements), 2)
        return True, reshaped_value

    def _parse_inline_object_elements(self, expr_part):
        inner = expr_part[1:-1].strip()
        return [elem.strip() for elem in inner.split(',') if elem.strip()]

    def _resolve_inline_object_values(self, elements):
        element_values = {}
        for elem in elements:
            try:
                element_values[elem] = self.compiler.current_scope().get(elem)
            except NameError:
                return None
        return element_values

    def _find_two_field_object_type(self, value):
        if not isinstance(value, dict):
            return None
        value_keys = object_public_keys(value)
        for type_name, fields in self.compiler.types_defined.items():
            field_defs = public_type_fields(fields)
            if len(field_defs) == 2 and value_keys == set(field_defs.keys()):
                return type_name
        return None

    def _find_object_array_type(self, value):
        if not isinstance(value, list) or not value:
            return None
        if not all(isinstance(item, dict) for item in value):
            return None
        for type_name, fields in self.compiler.types_defined.items():
            public_fields = set(public_type_fields(fields).keys())
            if public_fields and all(
                    object_public_keys(item) == public_fields for item in value):
                return type_name
        return None

    def _reshape_flat_object_values(self, value, num_objects, values_per_object):
        if not (isinstance(value, list) and value and isinstance(value[0], list)):
            return value
        flat_inner = value[0]
        expected_length = num_objects * values_per_object
        if len(flat_inner) != expected_length:
            return value
        return [
            flat_inner[i * values_per_object:(i + 1) * values_per_object]
            for i in range(num_objects)
        ]

    def _rewrite_implicit_intersection(self, expr, row_number):
        pattern = re.compile(r'\[\s*\^?([A-Za-z]+)\s*\]')
        return pattern.sub(lambda m: f"[{m.group(1)}{{{row_number}}}]", expr)

    def _evaluate_implicit_intersection(self, expr, row_number, scope, line_number=None):
        rewritten = self._rewrite_implicit_intersection(expr, row_number)
        return self.compiler.expr_evaluator.eval_or_eval_array(
            rewritten, scope, line_number)

    def assign_implicit_intersection_range(self, sr_ref, er_ref, expr, scope, line_number=None):
        sr, sc = sr_ref
        er, ec = er_ref
        num_cols = ec - sc + 1
        num_rows = er - sr + 1
        if num_cols < 1 or num_rows < 1:
            raise ValueError(
                f"Invalid range: {num_cols}x{num_rows} at line {line_number}")

        for r in range(sr, er + 1):
            for c in range(sc, ec + 1):
                cell = (r, c)
                value = self._evaluate_implicit_intersection(
                    expr, r + 1, scope, line_number)
                self.compiler._set_grid_cell(cell, value)

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
            if value is None:
                raise NameError(
                    f"Variable '{var_name}' not defined at line {line_number}")
            # Auto-initialize array to fit requested indices
            max_idx = [int(i) if isinstance(i, (int, float))
                       else 0 for i in indices]
            if len(max_idx) == 1:
                arr = [None] * (max_idx[0] + 1)
            elif len(max_idx) == 2:
                arr = [[None] * (max_idx[1] + 1)
                       for _ in range(max_idx[0] + 1)]
            else:
                arr = [[[None for _ in range(max_idx[2] + 1)]
                        for _ in range(max_idx[1] + 1)]
                       for _ in range(max_idx[0] + 1)]
            self.compiler.current_scope().define(var_name, arr)
        shape = self.get_array_shape(arr, line_number)

        # Handle cell reference index
        if len(indices) == 1 and re.match(r'^[A-Za-z]+\d+$', indices[0]):
            cell_ref = indices[0]
            col_str, row_str = split_cell(cell_ref)
            col_idx = col_to_num(col_str) - 1
            row_idx = int(row_str) - 1

            if len(shape) != 2:
                raise ValueError(
                    f"Expected 2-dimensional array for cell-based indexing, got shape {shape} at line {line_number}")

            if col_idx < 0 or col_idx >= shape[1]:
                raise ValueError(
                    f"Column index {col_idx} out of bounds for dimension size {shape[1]} at line {line_number}")
            if row_idx < 0 or row_idx >= shape[0]:
                raise ValueError(
                    f"Row index {row_idx} out of bounds for dimension size {shape[0]} at line {line_number}")

            if value is None:  # Read
                return self.get_array_element(
                    arr, [row_idx, col_idx], line_number, var_name=var_name)
            else:  # Write
                flat_value = self.flatten_array(value, line_number)
                if isinstance(arr, list):
                    if not arr:
                        arr.append([])
                    while len(arr) <= row_idx:
                        arr.append([])
                    while len(arr[row_idx]) <= col_idx:
                        arr[row_idx].append(None)
                    arr[row_idx][col_idx] = flat_value[0] if flat_value else None
                    self.compiler.current_scope().update(var_name, arr, line_number)
                else:
                    updated = self.set_array_element(
                        arr, [row_idx, col_idx],
                        float(flat_value[0]) if flat_value else 0, line_number)
                    self.compiler.current_scope().update(
                        var_name, updated, line_number)

        else:
            # Numeric indices
            if len(indices) != len(shape):
                raise ValueError(
                    f"Expected {len(shape)} indices, got {len(indices)} at line {line_number}")
            flat_value = self.flatten_array(value, line_number)
            if isinstance(arr, list):
                def _ensure(lst, idxs, val):
                    if len(idxs) == 1:
                        while len(lst) <= idxs[0]:
                            lst.append(None)
                        lst[idxs[0]] = val
                    else:
                        while len(lst) <= idxs[0]:
                            lst.append([])
                        if not isinstance(lst[idxs[0]], list):
                            lst[idxs[0]] = []
                        _ensure(lst[idxs[0]], idxs[1:], val)
                idx_list = []
                for index in indices:
                    try:
                        idx_val = int(index) - 1
                    except ValueError:
                        raise ValueError(
                            f"Invalid index '{index}' at line {line_number}")
                    idx_list.append(max(idx_val, 0))
                if value is None:
                    # Read path
                    ref = arr
                    for j, idx_val in enumerate(idx_list):
                        if not isinstance(ref, list) or idx_val >= len(ref):
                            return None
                        ref = ref[idx_val]
                    return ref
                _ensure(arr, idx_list, flat_value[0] if flat_value else None)
                self.compiler.current_scope().update(var_name, arr, line_number)
            else:
                # flat lists or N-D dict-form arrays
                idx_list = []
                for index in indices:
                    try:
                        idx_val = int(index) - 1
                    except ValueError:
                        raise ValueError(
                            f"Invalid index '{index}' at line {line_number}")
                    if idx_val < 0 or idx_val >= shape[len(idx_list)]:
                        raise ValueError(
                            f"Index {idx_val + 1} out of bounds at line {line_number}")
                    idx_list.append(idx_val)
                if value is None:  # Read
                    return self.get_array_element(
                        arr, idx_list, line_number, var_name=var_name)
                updated = self.set_array_element(
                    arr, idx_list, flat_value[0] if flat_value else 0, line_number)
                self.compiler.current_scope().update(var_name, updated, line_number)

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

        nd_shape = None
        if isinstance(arr, dict) and 'array' in arr:
            nd_shape = arr.get('shape') or arr.get('original_shape')
            if nd_shape is not None:
                nd_shape = list(nd_shape)
            arr = arr['array']
        if isinstance(arr, list):
            flat_arr = list(arr)
        else:
            raise TypeError(
                f"Variable '{var_name}' is not an array at line {line_number}")
        inner_size = shape[1] if len(shape) > 1 else 1
        outer_size = shape[0] if shape else max(10, dim_index + 1)
        flat_arr = flat_arr + [0] * (inner_size * outer_size - len(flat_arr))

        # Column-major: first dim is fastest, flat = i0 + s0*i1
        if value is None:  # Read
            if dim_idx == 0:
                result = [flat_arr[dim_index + shape[0] * i]
                          for i in range(shape[1])]
                return result
            else:
                result = [flat_arr[i + shape[0] * dim_index]
                          for i in range(shape[0])]
                return result[0] if len(result) == 1 else result

        # Assignment
        if dim_idx == 0:
            for i in range(shape[1]):
                flat_arr[dim_index + shape[0] * i] = float(flat_value[i]) if isinstance(
                    flat_value[i], (int, float)) else flat_value[i]
        else:
            for i in range(shape[0]):
                flat_arr[i + shape[0] * dim_index] = float(flat_value[i]) if isinstance(
                    flat_value[i], (int, float)) else flat_value[i]
        new_values = flat_arr
        if nd_shape is not None:
            self.compiler.current_scope().update(var_name, {
                'array': new_values, 'shape': nd_shape, 'original_shape': list(nd_shape)}, line_number)
        else:
            self.compiler.current_scope().update(
                var_name, flat_arr, line_number)

    def _update_bound_array_cell(self, target, value, line_number=None):
        """Update a bound array element/field when a mapped grid cell is assigned."""
        bindings = self.compiler._cell_array_map
        if not bindings:
            return False
        try:
            if not (isinstance(target, (tuple, list)) and len(target) == 2):
                return False
            target_row, target_col = target
        except Exception:
            return False
        for start_cell, var_name in bindings.items():
            try:
                if not (isinstance(start_cell, (tuple, list)) and len(start_cell) == 2):
                    continue
                start_row, start_col = start_cell
            except Exception:
                continue
            if target_row < start_row or target_col < start_col:
                continue
            row_offset = target_row - start_row
            col_offset = target_col - start_col
            defining_scope = self.compiler.current_scope().get_defining_scope(
                var_name)
            if not defining_scope:
                continue
            actual_key = defining_scope._get_case_insensitive_key(
                var_name, defining_scope.variables) or var_name
            constraints = defining_scope.constraints.get(actual_key, {})
            dims = constraints.get('dim')
            if isinstance(dims, dict) and 'dims' in dims:
                dims = dims['dims']
            if not isinstance(dims, list) or not dims:
                continue
            var_type = defining_scope.types.get(actual_key)
            if not var_type or var_type.lower() not in self.compiler.types_defined:
                continue
            if len(dims) != 1:
                continue
            size_spec = dims[0][1]
            if isinstance(size_spec, tuple) and size_spec[1] is not None:
                size = size_spec[1] - size_spec[0] + 1
            elif self._is_unbounded_size_spec(size_spec):
                size = row_offset + 1
            else:
                size = int(size_spec)
            if row_offset < 0 or row_offset >= size:
                continue
            type_def = self.compiler.types_defined.get(var_type.lower(), {})
            field_names = list(
                self.compiler._get_public_type_fields(type_def).keys())
            if col_offset < 0 or col_offset >= len(field_names):
                continue
            arr = defining_scope.variables.get(actual_key)
            if not isinstance(arr, list):
                arr = self.create_object_array([size], None, line_number)
            if row_offset >= len(arr):
                arr.extend([None] * (row_offset + 1 - len(arr)))
            element = arr[row_offset]
            if not isinstance(element, dict):
                element = {name: None for name in field_names}
                element['_type_name'] = var_type.lower()
                hidden_fields = type_def.get('_hidden_fields', set())
                if hidden_fields:
                    element['_hidden_fields'] = set(hidden_fields)
                element.setdefault('grid', {})
            field_name = field_names[col_offset]
            element[field_name] = value
            try:
                self.compiler._recompute_computed_fields(
                    element, line_number=line_number, changed_field=field_name)
            except Exception:
                pass
            arr[row_offset] = element
            defining_scope.update(actual_key, arr, line_number)
            return True
        return False

    def _assign_horizontal_array(self, target, value, expr_part, line_number=None):
        """
        Assign an array horizontally (or vertically) to the grid starting at target cell.
        Handles objects, lists, arrays, and reshaping for two-field objects.
        :param target: Starting cell as a 0-based index tuple (e.g. (0, 0)).
        :param value: Array or list to assign.
        :param expr_part: Original expression for orientation check.
        :param line_number: Line number.
        """
        if isinstance(target, (tuple, list)) and len(target) > 2:
            try:
                indices = [i + 1 for i in target]
            except ValueError:
                indices = None
            if indices is not None:
                self._assign_extended_address(
                    target, value, line_number, expr_part)
                return
        if isinstance(value, dict) and value and all(
                isinstance(k, tuple) for k in value.keys()):
            # Sparse (no-dim) array: dict keyed by 0-based index tuples with
            # the first declared dim (grid row) first.  Each populated cell is
            # spilled at its (row, col) offset from the target cell.
            rank = len(next(iter(value.keys())))
            if rank == 1:
                for (i0,), val in value.items():
                    cell_to_assign = offset_cell(target, i0, 0)
                    self.compiler._set_grid_cell(cell_to_assign, val)
            elif rank == 2:
                for (row, col), val in value.items():
                    cell_to_assign = offset_cell(target, col, row)
                    self.compiler._set_grid_cell(cell_to_assign, val)
            else:
                raise ValueError(
                    f"Dim not supported: rank {rank} at line {line_number}")
            return
        if isinstance(value, dict) and 'array' in value:
            # N-D array (column-major flat buffer + declared shape).
            shape = value.get('shape') or value.get('original_shape')
            shape = list(shape)
            flat_vals = self.flatten_array(value, line_number)
            flat_vals = self._resolve_spill_unset(flat_vals, expr_part, line_number)
            if len(shape) >= 2:
                # Grid rows = first dim (fastest), grid cols = last dim
                for col_idx in range(shape[1]):
                    for row_idx in range(shape[0]):
                        cell_to_assign = offset_cell(target, col_idx, row_idx)
                        self.compiler._set_grid_cell(cell_to_assign, flat_vals[row_idx +
                                                                       shape[0] * col_idx])
            else:
                for i, val in enumerate(flat_vals):
                    cell_to_assign = offset_cell(target, i, 0)
                    self.compiler._set_grid_cell(cell_to_assign, val)
            return
        if isinstance(value, dict):
            flattened_values = self.flatten_object_fields(value, line_number)
            for i, val in enumerate(flattened_values):
                cell_to_assign = offset_cell(target, i, 0)
                self.compiler._set_grid_cell(cell_to_assign, val)
            return
        if isinstance(value, list):
            is_object_array = False
            type_name = None
            if all(isinstance(item, dict) for item in value):
                for t_name, fields in self.compiler.types_defined.items():
                    public_fields = set(public_type_fields(fields).keys())
                    if public_fields and all(object_public_keys(item) == public_fields for item in value):
                        is_object_array = True
                        type_name = t_name
                        break
                if is_object_array:
                    for row_idx, item in enumerate(value):
                        flattened_values = self.flatten_object_fields(
                            item, line_number)
                        for col_idx, val in enumerate(flattened_values):
                            cell_to_assign = offset_cell(
                                target, col_idx, row_idx)
                            self.compiler._set_grid_cell(cell_to_assign, val)
                    return
            if value and all(isinstance(row, (list, tuple)) for row in value):
                for row_idx, row in enumerate(value):
                    for col_idx, val in enumerate(row):
                        cell_to_assign = offset_cell(
                            target, col_idx, row_idx)
                        self.compiler._set_grid_cell(cell_to_assign, val)
                return
            is_vertical = ';' in expr_part.strip(
            )[1:-1] and ',' not in expr_part.strip()[1:-1]
            flattened_values = self.flatten_array(value, line_number)
            flattened_values = self._resolve_spill_unset(flattened_values, expr_part, line_number)
            for i, val in enumerate(flattened_values):
                cell_to_assign = offset_cell(
                    target, 0, i) if is_vertical else offset_cell(target, i, 0)
                self.compiler._set_grid_cell(cell_to_assign, val)
        else:
            self.compiler._set_grid_cell(target, value)

    def _assign_extended_address(self, target, value, line_number=None, expr_part=None):
        """Write to an extended (N-D) address.

        A matching grid DIM tensor receives the value at the addressed flat
        position; otherwise the value is stored under the dotted key in the
        current grid. Arrays spill along the last dimension of the address.
        ``target`` is a 0-based index tuple.
        """
        indices = [i + 1 for i in target]
        if self._write_extended_tensor(indices, value, line_number):
            return
        if (isinstance(value, list)
                or (isinstance(value, dict) and 'array' in value)):
            flat = self.flatten_array(value, line_number)
            flat = self._resolve_spill_unset(flat, expr_part, line_number)
        else:
            flat = [value]
        base = list(target)
        for i, val in enumerate(flat):
            addr_indices = list(base)
            addr_indices[-1] = base[-1] + i
            self.compiler._set_grid_cell(
                tuple(addr_indices), self.to_display_value(val))

    def assign_range(self, sr_ref, er_ref, vals, line_number=None):
        """
        Assign values to a range of cells (e.g., A1:B2 := {1,2;3,4}).
        Handles scalars, 1D/2D arrays, repeating, and cycling.
        :param sr_ref: Start cell as a 0-based index tuple (e.g. (0, 0)).
        :param er_ref: End cell as a 0-based index tuple (e.g. (1, 1)).
        :param vals: Values to assign (scalar, list, array).
        :param line_number: Line number.
        """
        if len(sr_ref) > 2 or len(er_ref) > 2:
            self._assign_extended_range(sr_ref, er_ref, vals, line_number)
            return
        sr, sc = sr_ref
        er, ec = er_ref
        num_cols = ec - sc + 1
        num_rows = er - sr + 1
        if num_cols < 1 or num_rows < 1:
            raise ValueError(
                f"Invalid range: {num_cols}x{num_rows} at line {line_number}")

        is_array = isinstance(
            vals, list) or (
            isinstance(vals, dict) and 'array' in vals)
        shape = self.get_array_shape(vals, line_number) if is_array else [1]
        flat_vals = self.flatten_array(vals, line_number) if is_array else [vals]

        if is_array:
            effective_shape = shape
            if len(shape) == 2 and shape[0] == 1:
                effective_shape = [shape[1]]

            if len(effective_shape) == 1:
                array_length = effective_shape[0]
                if num_rows > 1 and num_cols == 1:  # Vertical assignment
                    for i, r in enumerate(range(sr, er + 1)):
                        cell = (r, sc)
                        value = flat_vals[i % len(flat_vals)]
                        self.compiler._set_grid_cell(cell, value)
                elif num_cols > 1 and num_rows == 1:  # Horizontal assignment
                    for i, c in enumerate(range(sc, ec + 1)):
                        cell = (sr, c)
                        value = flat_vals[i % len(flat_vals)]
                        self.compiler._set_grid_cell(cell, value)
                elif array_length == num_cols:  # Repeat across rows (broadcast over first dim)
                    for r in range(sr, er + 1):
                        for c in range(sc, ec + 1):
                            col_idx = c - sc
                            cell = (r, c)
                            value = flat_vals[col_idx]
                            self.compiler._set_grid_cell(cell, value)
                else:  # Cycle over the range
                    idx = 0
                    for r in range(sr, er + 1):
                        for c in range(sc, ec + 1):
                            cell = (r, c)
                            value = flat_vals[idx % len(flat_vals)]
                            self.compiler._set_grid_cell(cell, value)
                            idx += 1
            elif len(effective_shape) > 1:
                if effective_shape[0] == num_rows and effective_shape[1] == num_cols:
                    # Column-major: first dim is the fastest (grid row)
                    for c in range(num_cols):
                        for r in range(num_rows):
                            cell = (sr + r, sc + c)
                            value = flat_vals[r + effective_shape[0] * c]
                            self.compiler._set_grid_cell(cell, value)
                elif effective_shape[0] == num_cols and effective_shape[1] == num_rows and num_cols == 1:
                    reshaped = [[flat_vals[i]] for i in range(len(flat_vals))]
                    idx = 0
                    for r in range(sr, sr + num_rows):
                        for c in range(sc, sc + num_cols):
                            cell = (r, c)
                            value = reshaped[idx][0]
                            self.compiler._set_grid_cell(cell, value)
                            idx += 1
                elif effective_shape[0] == num_rows and effective_shape[1] == num_cols == 1:
                    idx = 0
                    for r in range(sr, sr + effective_shape[0]):
                        cell = (r, sc)
                        value = flat_vals[idx]
                        self.compiler._set_grid_cell(cell, value)
                        idx += 1
                else:
                    raise ValueError(
                        f"Array shape {effective_shape} exceeds range ({num_rows}x{num_cols}) at line {line_number}")
        else:  # Scalar assignment to range
            for r in range(sr, er + 1):
                for c in range(sc, ec + 1):
                    cell = (r, c)
                    value = flat_vals[0]
                    self.compiler._set_grid_cell(cell, value)

    def _assign_extended_range(self, sr_ref, er_ref, vals, line_number=None):
        """Assign values to an extended (N-D) range.

        The hyper-rectangle between the two addresses is filled in column-major
        order (first index fastest). Scalars fill every cell; arrays repeat and
        cycle over the flat buffer. ``sr_ref``/``er_ref`` are 0-based tuples.
        """
        s_idx = [i + 1 for i in sr_ref]
        e_idx = [i + 1 for i in er_ref]
        if len(s_idx) != len(e_idx):
            raise ValueError(
                f"Range addresses '{indices_to_address([i for i in s_idx])}' and '{indices_to_address([i for i in e_idx])}' must have the same rank at line {line_number}")
        starts = [min(a, b) for a, b in zip(s_idx, e_idx)]
        shape = [abs(a - b) + 1 for a, b in zip(s_idx, e_idx)]
        is_array = isinstance(
            vals, list) or (
            isinstance(vals, dict) and 'array' in vals)
        flat_vals = self.flatten_array(
            vals, line_number) if is_array else [vals]
        if not flat_vals:
            return
        for flat_i in range(prod(shape)):
            rem = flat_i
            idxs = []
            for dim_size in shape:
                idxs.append(rem % dim_size)
                rem //= dim_size
            addr = tuple(starts[i] + idxs[i] - 1 for i in range(len(shape)))
            self.compiler._set_grid_cell(addr, self.to_display_value(
                flat_vals[flat_i % len(flat_vals)]))

    def get_range_values(self, s_cell, e_cell, line_number=None):
        """
        Retrieve values from a grid range (e.g., A1:B2).
        Returns 1D list for single row/column, 2D list otherwise.
        :param s_cell: Start cell as a 0-based index tuple (e.g. (0, 0)).
        :param e_cell: End cell as a 0-based index tuple (e.g. (1, 1)).
        :param line_number: Line number.
        :return: List of values.
        """
        sr, sc = s_cell
        er, ec = e_cell
        values = []
        for r in range(min(sr, er), max(sr, er) + 1):
            row_values = []
            for c in range(min(sc, ec), max(sc, ec) + 1):
                row_values.append(self.lookup_cell((r, c), line_number))
            values.append(row_values)
        return values

    def get_range_values_address(self, s_cell, e_cell, line_number=None):
        """
        Retrieve values from an extended (N-D) cell range.

        The full hyper-rectangle between the two addresses is enumerated in
        column-major order (first index fastest) and returned as nested lists
        in declared-dim order. Both endpoints must have the same rank.
        """
        s_idx = [i + 1 for i in s_cell]
        e_idx = [i + 1 for i in e_cell]
        if len(s_idx) != len(e_idx):
            raise ValueError(
                f"Range addresses '{indices_to_address([i for i in s_idx])}' and '{indices_to_address([i for i in e_idx])}' must have the same rank at line {line_number}")
        starts = [min(a, b) for a, b in zip(s_idx, e_idx)]
        shape = [abs(a - b) + 1 for a, b in zip(s_idx, e_idx)]
        flat_values = []
        for flat_i in range(prod(shape)):
            rem = flat_i
            idxs = []
            for dim_size in shape:
                idxs.append(rem % dim_size)
                rem //= dim_size
            addr = tuple(starts[i] + idxs[i] - 1 for i in range(len(shape)))
            flat_values.append(self.lookup_cell(addr, line_number))
        return self._nested_from_flat(flat_values, shape)

    def _array_unset_value(self, var_name, line_number=None):
        """Value read back for an unset item of an array variable.

        Without a 'default' constraint, unset items read as #N/A. With a
        constraint like ``Let grid not null or = None`` (or ``Let x as
        number dim * not null or = None``), unset (#N/A) items read as the
        evaluated default instead. The grid is just an array whose variable
        is 'grid'.
        """
        constraints = {}
        scope = None
        try:
            scope = self.compiler.scopes[0]
        except (AttributeError, IndexError):
            pass
        try:
            defining_scope = self.compiler.current_scope().get_defining_scope(
                var_name)
            if defining_scope is not None:
                key = defining_scope._get_case_insensitive_key(
                    var_name, defining_scope.constraints)
                if key:
                    constraints = defining_scope.constraints[key]
        except Exception:
            pass
        if not constraints and scope is not None:
            key = scope._get_case_insensitive_key(var_name, scope.constraints)
            if key:
                constraints = scope.constraints[key]
        default_expr = constraints.get('default')
        if default_expr is not None:
            try:
                return self.compiler.expr_evaluator.eval_expr(
                    str(default_expr), scope.get_evaluation_scope())
            except Exception:
                pass
        return error_value(NA_ERROR)

    def _resolve_spill_unset(self, flat_values, expr_part, line_number=None):
        """Replace None sentinel values in a flat list before spilling to grid.

        Uses the RHS expression's variable default if available, otherwise
        returns the global #N/A error value.
        """
        has_none = any(v is None for v in flat_values)
        if not has_none:
            return flat_values
        rhs_var = None
        if expr_part and re.match(r'^[A-Za-z_]\w*$', expr_part.strip()):
            rhs_var = expr_part.strip()
        if rhs_var:
            unset_default = self._array_unset_value(rhs_var, line_number)
        else:
            unset_default = error_value(NA_ERROR)
        return [unset_default if v is None else v for v in flat_values]

    def lookup_cell(self, cell_ref, line_number=None):
        """
        Lookup value in a grid cell, default to #N/A (or the grid's null
        default) if unset.
        :param cell_ref: Cell reference as a 0-based index tuple (or, for
            compatibility with python-fallback scope, an address string).
        :param line_number: Line number.
        :return: Cell value.
        """
        # The grid store is keyed by 0-based numeric index tuples; convert the
        # cell reference (case-insensitive) before looking it up.
        store = self.compiler.grid
        if isinstance(cell_ref, (tuple, list)):
            key = tuple(int(i) for i in cell_ref)
        else:
            try:
                key = tuple(i - 1 for i in parse_address(cell_ref))
            except ValueError:
                key = None
        if key is not None and key in store:
            return store[key]
        if key is not None and len(key) > 2:
            # Extended addresses fall back to any matching grid DIM tensor.
            value = self._lookup_extended_address(key, line_number)
            if value is not None:
                return value
        return self._array_unset_value('grid', line_number)

    def _iter_scopes(self):
        scope = self.compiler.current_scope()
        seen = set()
        while scope is not None:
            if id(scope) in seen:
                break
            seen.add(id(scope))
            yield scope
            scope = getattr(scope, 'parent', None)

    def _iter_tensor_grid_vars(self):
        """Yield grid DIM tensor stores (dicts with a 'grid' + shape)."""
        seen = set()
        for scope in self._iter_scopes():
            for val in list(getattr(scope, 'variables', {}).values()):
                if id(val) in seen:
                    continue
                seen.add(id(val))
                if (isinstance(val, dict) and 'grid' in val
                        and ('original_shape' in val or 'shape' in val)):
                    yield val

    def _tensor_index_for_indices(self, tensor, indices):
        shape = list(tensor.get('original_shape') or tensor.get('shape') or [])
        if len(shape) != len(indices):
            return None, None
        if not all(1 <= indices[i] <= shape[i] for i in range(len(shape))):
            return None, None
        flat_idx = 0
        stride = 1
        for i in range(len(indices)):
            flat_idx += (indices[i] - 1) * stride
            stride *= shape[i]
        return flat_idx, shape

    def _lookup_extended_address(self, cell_ref, line_number=None):
        """Resolve a dotted address against grid DIM tensor stores."""
        try:
            if isinstance(cell_ref, (tuple, list)):
                indices = [i + 1 for i in cell_ref]
            else:
                indices = parse_address(cell_ref)
        except ValueError:
            return None
        for tensor in self._iter_tensor_grid_vars():
            flat_idx, shape = self._tensor_index_for_indices(tensor, indices)
            if flat_idx is None:
                continue
            arr = tensor.get('grid')
            if isinstance(arr, list):
                flat = arr
                if 0 <= flat_idx < len(flat):
                    return flat[flat_idx]
        return None

    def _write_extended_tensor(self, indices, value, line_number=None):
        """Write value into a matching grid DIM tensor's flat buffer."""
        for tensor in self._iter_tensor_grid_vars():
            flat_idx, shape = self._tensor_index_for_indices(tensor, indices)
            if flat_idx is None:
                continue
            arr = tensor.get('grid')
            if not isinstance(arr, list):
                continue
            flat = arr
            if not (0 <= flat_idx < len(flat)):
                continue
            flat[flat_idx] = value
            tensor['grid'] = flat
            return True
        return False

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
                if object_public_keys(obj) == set(public_type_fields(fields).keys()):
                    type_name = name
                    break
            if type_name:
                fields = public_type_fields(
                    self.compiler.types_defined[type_name.lower()])
                for field in fields.keys():
                    value = obj.get(field)
                    if isinstance(value, dict):
                        nested_type = None
                        for n_name, n_fields in self.compiler.types_defined.items():
                            if object_public_keys(value) == set(public_type_fields(n_fields).keys()):
                                nested_type = n_name
                                break
                        if nested_type:
                            nested_fields = public_type_fields(
                                self.compiler.types_defined[nested_type])
                            for n_field in nested_fields.keys():
                                result.append(value.get(n_field))
                        else:
                            result.extend([value[k]
                                           for k in sorted(object_public_keys(value))])
                    else:
                        result.append(value)
            else:
                result.extend([obj[k]
                              for k in sorted(object_public_keys(obj))])
        else:
            result.append(obj)
        return result

    def flatten_array(self, arr, line_number=None):
        """
        Flatten an array (flat list + shape dict, or list) into a 1D list.
        :param arr: Array or list.
        :param line_number: Line number.
        :return: Flattened list.
        """
        if isinstance(arr, dict) and 'array' in arr:
            inner = arr['array']
            if isinstance(inner, list):
                return self.flatten_array(inner, line_number)
            return [inner]
        if isinstance(arr, dict):
            # Sparse unbounded (star-dim) array: dict keyed by 0-based index
            # tuples. Flatten ordered by index (lexicographic tuple order).
            if arr and all(isinstance(k, tuple) for k in arr.keys()):
                return [arr[k] for k in sorted(arr.keys())]
            return [arr]
        if isinstance(arr, list):
            if not arr or not all(isinstance(v, list) for v in arr):
                return list(arr)
            # Nested lists are the display form: the outermost index is the
            # first declared dim (fastest). Column-major flatten:
            # flat[i0 + s0*i1 + s0*s1*i2 + ...] = arr[i0][i1][i2]...
            shape = []
            node = arr
            while node and isinstance(node, list):
                shape.append(len(node))
                node = node[0] if node else None
            if not shape:
                return []
            total = 1
            for s in shape:
                total *= s
            result = []
            for flat_idx in range(total):
                node = arr
                rem = flat_idx
                for s in shape:
                    node = node[rem % s]
                    rem //= s
                result.append(node)
            return result
        return [arr]

    def to_display_value(self, value, line_number=None):
        """Convert an internal array representation to its display form.

        N-D arrays are stored as {'array', 'shape', 'original_shape'}; their
        display form is a nested Python list laid out in the declared
        dimension order (outermost index = first declared dim).
        """
        if isinstance(value, dict) and 'array' in value:
            shape = list(value.get('shape') or value.get('original_shape') or [])
            inner = value['array']
            if isinstance(inner, list):
                flat = self.flatten_array(inner, line_number)
            else:
                return value
            if len(shape) <= 1:
                return flat
            return self._nested_from_flat(flat, shape)
        if isinstance(value, dict):
            # Sparse unbounded (star-dim) array
            if value and all(isinstance(k, tuple) for k in value.keys()):
                rank = len(next(iter(value.keys())))
                if rank <= 1:
                    return [value[k] for k in sorted(value.keys())]
                if rank == 2:
                    # Materialize as rows (first dim = row, second = column);
                    # unset cells stay None so gaps render like sparse reads.
                    rows_map = {}
                    for k, v in value.items():
                        rows_map.setdefault(k[0], {})[k[1]] = v
                    result = []
                    for r in range(max(rows_map) + 1):
                        if r not in rows_map:
                            result.append([])
                            continue
                        cols = rows_map[r]
                        row_vals = [cols.get(c) for c in range(max(cols) + 1)]
                        while row_vals and row_vals[-1] is None:
                            row_vals.pop()
                        result.append(row_vals)
                    return result
            return value
        return value

    def _nested_from_flat(self, flat, shape):
        if len(shape) == 1:
            return list(flat)
        s0 = shape[0]
        inner_shape = shape[1:]
        inner_size = 1
        for s in inner_shape:
            inner_size *= s
        result = []
        for i0 in range(s0):
            inner_flat = flat[i0::s0][:inner_size]
            result.append(self._nested_from_flat(inner_flat, inner_shape))
        return result

    def materialize_list_array(self, value, line_number=None):
        """Convert a plain (possibly ragged) nested or flat list into sparse
        index-keyed dict storage: {tuple(0-based cell indices): value}.

        Each populated cell is stored under its 0-based indices with the
        first declared dim (grid row) first: {1, 2; 3, 4} ->
        {(0, 0): 1, (1, 0): 2, (0, 1): 3, (1, 1): 4}.  Empty cells have no
        entry (sparse).  Already dict-form values and non-list values are
        returned unchanged.
        """
        if isinstance(value, dict) or not isinstance(value, list):
            return value
        result = {}

        def walk(node, indices):
            for i, child in enumerate(node):
                if isinstance(child, list):
                    walk(child, indices + [i])
                else:
                    result[tuple(indices + [i])] = child

        walk(value, [])
        return result

    def infer_type(self, value, line_number=None):
        """
        Infer type of a value (array, number, text, object, etc.).
        :param value: Value to infer.
        :param line_number: Line number.
        :return: Inferred type string.
        """
        if isinstance(value, dict) and 'array' in value:
            return 'array'
        if isinstance(value, dict) and value and all(
                isinstance(k, tuple) for k in value.keys()):
            return 'array'
        if isinstance(value, list) and value and isinstance(value[0], list):
            return 'array'
        if isinstance(value, list):
            return 'array'
        if isinstance(value, (int, float)):
            return 'number' if isinstance(value, float) else 'int'
        if isinstance(value, bool):
            return 'logical'
        if isinstance(value, str):
            return 'text'
        if isinstance(value, dict):
            return 'object'
        from units import UnitValue
        if isinstance(value, UnitValue):
            return 'number'
        return 'unknown'

    def get_array_shape(self, arr, line_number=None):
        """
        Get the shape of an array (flat list + shape dict, nested lists).
        :param arr: Array.
        :param line_number: Line number.
        :return: List of dimension sizes.
        """
        if isinstance(arr, dict):
            for key in ('shape', 'original_shape'):
                if key in arr:
                    return list(arr[key])
            if arr and all(isinstance(k, tuple) for k in arr.keys()):
                shape = [0] * len(next(iter(arr.keys())))
                for k in arr.keys():
                    for i, idx in enumerate(k):
                        if idx + 1 > shape[i]:
                            shape[i] = idx + 1
                return shape
            return [1]
        elif isinstance(arr, list):
            if not arr:
                return [0]
            shape = []
            node = arr
            while isinstance(node, list) and node:
                shape.append(len(node))
                node = node[0]
            if isinstance(node, list) and not node and shape:
                shape.append(0)
            return shape
        else:
            return [1]

    def create_array(self, shape, default_value, pa_type, line_number=None, matrix_data=None, is_grid_dim=False, template=False):

        # Handle 0D array (scalar)
        if not shape:
            return [None if template else default_value]

        # Calculate flat size
        flat_size = 1
        for dim in shape:
            flat_size *= dim

        # Initialize values
        if template:
            # Template arrays (declared without a value) store None for every
            # element so that reads yield #N/A (or the variable's declared
            # default) until each cell is explicitly written.
            values = [None] * flat_size
        elif is_grid_dim and matrix_data:
            # Grid DIM matrices stack along the LAST declared dim (depth).
            # Each matrix is a 2D display: rows span the first dim, columns
            # span the remaining dims flattened column-major.  Storage is
            # column-major overall (first dim fastest, last dim slowest).
            if len(matrix_data) != shape[-1]:
                raise ValueError(
                    f"Expected {shape[-1]} matrices for the last dimension, got {len(matrix_data)} at line {line_number}")
            inner_size = 1
            for dim in shape[:-1]:
                inner_size *= dim
            col_size = inner_size // shape[0]
            values = []
            for matrix in matrix_data:
                if len(matrix) != shape[0]:
                    raise ValueError(
                        f"Expected {shape[0]} rows per matrix, got {len(matrix)} at line {line_number}")
                for sub in range(inner_size):
                    i0 = sub % shape[0]
                    j = sub // shape[0]
                    row = matrix[i0]
                    if len(row) != col_size:
                        raise ValueError(
                            f"Expected {col_size} columns per row, got {len(row)} at line {line_number}")
                    val = row[j]
                    if default_value is not None and float(val) != float(default_value):
                        raise ValueError(
                            f"Value {val} violates constraint {default_value} at line {line_number}")
                    values.append(val)
        else:
            if is_grid_dim and default_value is None:
                raise ValueError(
                    f"No matrix data provided for grid at line {line_number}")
            if pa_type == 'text':
                values = [str(default_value)
                          if default_value is not None else ""] * flat_size
            elif pa_type == 'number':
                values = [
                    float(default_value) if default_value is not None else 0.0] * flat_size
            else:
                values = [
                    default_value if default_value is not None else None] * flat_size

        # Create array based on dimensions
        if len(shape) == 1:
            return values
        elif len(shape) >= 2:
            return {'array': values, 'shape': list(shape), 'original_shape': list(shape)}
        else:
            raise ValueError(
                f"Unsupported array dimensions: {shape} at line {line_number}")

    def create_object_array(self, shape, default_value=None, line_number=None):
        """
        Create a nested Python list for object arrays (custom types).
        :param shape: List of dimension sizes.
        :param default_value: Default value to assign in leaf nodes.
        :param line_number: Line number for error reporting.
        """
        if not shape:
            return copy.deepcopy(default_value)
        size = shape[0]
        if not isinstance(size, int):
            raise ValueError(
                f"Invalid object array dimension '{size}' at line {line_number}")
        return [self.create_object_array(shape[1:], default_value, line_number) for _ in range(size)]

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

    def _is_unbounded_size_spec(self, size_spec):
        """True for '*' (None) and 'n to *' ((start, None)) size specs."""
        if size_spec is None:
            return True
        if isinstance(size_spec, tuple) and len(size_spec) == 2:
            return size_spec[1] is None
        return False

    def _dim_size(self, size_spec, star_size=1):
        if isinstance(size_spec, tuple):
            start, end = size_spec
            if end is None:
                return star_size
            return end - start + 1
        if size_spec is None:
            return star_size
        return size_spec

    def validate_array_element_types(self, var, value, var_type, line_number=None):
        """Validate that every scalar element of a dim array matches the declared base type.

        None (uninitialized), error values, and nested objects are left
        untouched.  A mismatched scalar raises a type error so the whole
        array assignment is rejected (mirrors scalar base-type checking).
        :param var: Variable name.
        :param value: Array value (any supported storage form).
        :param var_type: Declared base type ('number', 'text').
        :param line_number: Line number for error reporting.
        """
        flat = self.flatten_array(value, line_number)
        for element in flat:
            if element is None or element is UNIVERSAL_ZERO or is_error_value(element):
                continue
            if isinstance(element, (list, dict)):
                continue
            actual_type = self.infer_type(element, line_number)
            if var_type == 'number' and actual_type not in ('number', 'float64', 'int', 'int64'):
                raise ConstraintError(
                    TYPE_ERROR,
                    f"'{var}' must be a number array, got a {actual_type} element at line {line_number}")
            if var_type == 'text' and actual_type not in ('string', 'text'):
                raise ConstraintError(
                    TYPE_ERROR,
                    f"'{var}' must be a text array, got a {actual_type} element at line {line_number}")

    def check_dimension_constraints(self, var, value, line_number=None):
        """
        Check and reshape value to match dimension constraints of the variable.
        Infers '*' sizes and broadcasts scalars.
        :param var: Variable name.
        :param value: Value to check/reshape.
        :param line_number: Line number.
        :return: Reshaped value.
        """
        dims = None
        if var in self.compiler.dimensions:
            dims = self.compiler.dimensions[var]
        else:
            scope = self.compiler.current_scope().get_defining_scope(var)
            if scope:
                actual_key = scope._get_case_insensitive_key(
                    var, scope.constraints) or var
                dim_spec = scope.constraints.get(actual_key, {}).get('dim')
                if isinstance(dim_spec, dict) and 'dims' in dim_spec:
                    dims = dim_spec['dims']
                elif isinstance(dim_spec, list):
                    dims = dim_spec
                elif isinstance(dim_spec, str):
                    dim_str = dim_spec.strip()
                    if dim_str.lower().startswith('dim '):
                        dim_str = dim_str[4:].strip()
                    if dim_str.startswith('{') and dim_str.endswith('}'):
                        dim_content = dim_str[1:-1].strip()
                        parts = [p.strip()
                                 for p in dim_content.split(',') if p.strip()]
                        parsed_dims = []
                        for part in parts:
                            if ':' in part:
                                name, size = map(str.strip, part.split(':', 1))
                                size_spec = self.compiler._parse_dim_size(
                                    size, line_number)
                                parsed_dims.append((name, size_spec))
                            else:
                                size_spec = self.compiler._parse_dim_size(
                                    part, line_number)
                                parsed_dims.append((None, size_spec))
                        dims = parsed_dims
                    elif dim_str:
                        size_spec = self.compiler._parse_dim_size(
                            dim_str, line_number)
                        dims = [(None, size_spec)]
                if dims is not None:
                    self.compiler.dimensions[var] = dims

        if dims is None:
            return value
        scope = self.compiler.current_scope().get_defining_scope(
            var) or self.compiler.current_scope()
        resolved_dims = []
        for name, size_spec in dims:
            if isinstance(size_spec, str):
                try:
                    eval_scope = scope.get_full_scope()
                    size_val = self.compiler.expr_evaluator.eval_expr(
                        size_spec, eval_scope, line_number)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to evaluate dimension size '{size_spec}' for '{var}' at line {line_number}: {exc}")
                if isinstance(size_val, bool):
                    raise ValueError(
                        f"Invalid dimension size '{size_spec}' for '{var}' at line {line_number}")
                if isinstance(size_val, float) and not size_val.is_integer():
                    raise ValueError(
                        f"Dimension size '{size_spec}' for '{var}' is not an integer at line {line_number}")
                if not isinstance(size_val, (int, float)):
                    raise ValueError(
                        f"Invalid dimension size '{size_spec}' for '{var}' at line {line_number}")
                size_spec = int(size_val)
            resolved_dims.append((name, size_spec))
        dims = resolved_dims
        self.compiler.dimensions[var] = dims
        if isinstance(value, dict) and not value and any(
                self._is_unbounded_size_spec(size_spec)
                for _, size_spec in dims):
            # Fresh sparse array from a declaration without a value
            # expression; it grows on write and unset items read the
            # default (#N/A) on access.
            return value
        if dims == []:
            if isinstance(value, (list, tuple)) or (
                    isinstance(value, dict) and 'array' in value):
                value = list(value) if isinstance(value, (list, tuple)) else \
                    self.flatten_array(value, line_number)
            if isinstance(value, list):
                raise ConstraintError(
                    DIM_ERROR,
                    f"Dimension constraint for '{var}' expects a scalar, received array value at line {line_number}")
            return value

        # Unbounded multi-dimensional constraints ({*, *}, {0 to *, *}) have
        # no inferable sizes; keep the value as a sparse array instead of
        # reshaping a ragged literal into a dense buffer.
        if len(dims) > 1 and all(
                self._is_unbounded_size_spec(size_spec) for _, size_spec in dims):
            if isinstance(value, list):
                return self.materialize_list_array(value, line_number)
            return value

        # Handle scalar broadcasting
        var_type = None
        scope = self.compiler.current_scope().get_defining_scope(var)
        if scope:
            actual_key = scope._get_case_insensitive_key(
                var, scope.types) or var
            var_type = scope.types.get(actual_key)
        if var_type is None:
            var_type = self.compiler.types.get(var)

        if isinstance(value, dict) and 'array' not in value and var_type and var_type.lower() in self.compiler.types_defined:
            import copy
            shape = [self._dim_size(size_spec, star_size=1)
                     for _, size_spec in dims]
            total = 1
            for dim in shape:
                total *= dim
            flat_vals = [copy.deepcopy(value) for _ in range(total)]
            if len(shape) == 1:
                return flat_vals
            reshaped = []
            stride = shape[1] if len(shape) > 1 else 1
            for i in range(shape[0]):
                start = i * stride
                row = flat_vals[start:start + stride]
                reshaped.append(row)
            return reshaped

        is_sparse = is_sparse_array(value)
        is_nd_array = isinstance(value, dict) and 'array' in value
        if not is_nd_array and not is_sparse and (
                not isinstance(value, list) or isinstance(value, (int, float, str))):
            shape = [self._dim_size(size_spec, star_size=1)
                     for _, size_spec in dims]
            var_type = self.compiler.types.get(var)
            if var_type in ('number', 'array'):
                pa_type = 'number'
            elif var_type == 'logical':
                pa_type = 'logical'
            else:
                inferred = self.infer_type(value, line_number)
                if inferred in ('number', 'float64', 'int', 'int64'):
                    pa_type = 'number'
                elif inferred == 'logical' or isinstance(value, bool):
                    pa_type = 'logical'
                else:
                    pa_type = 'text'
            return self.create_array(shape, value, pa_type, line_number)

        # Compute expected shape
        expected_shape = []
        star_indices = [i for i, (_, size_spec) in enumerate(
            dims) if self._is_unbounded_size_spec(size_spec)]
        known_product = 1
        for i, (_, size_spec) in enumerate(dims):
            dim_size = self._dim_size(size_spec, star_size=None)
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
            if (total_elements and expected_total % total_elements == 0
                    and total_elements < expected_total):
                # Column-major row-broadcast: a shorter 1D literal spans the
                # slowest dim(s) and is repeated across the faster dims.
                # dim {row:2, col:3} = {1,2,3} -> buffer [1,1,2,2,3,3]
                repeat = expected_total // total_elements
                broadcast = []
                for v in flat_vals:
                    broadcast.extend([v] * repeat)
                flat_vals = broadcast
                total_elements = expected_total
            else:
                raise ConstraintError(
                    DIM_ERROR,
                    f"Element count mismatch for '{var}': expected {expected_total} elements, got {total_elements} at line {line_number}")

        # Reshape to expected shape (column-major flat buffer + declared shape)
        if len(expected_shape) == 1:
            return flat_vals
        return {'array': list(flat_vals), 'shape': list(expected_shape), 'original_shape': list(expected_shape)}

    def set_array_element(self, array, indices, value, line_number=None):
        """
        Set an element in an array of any dimension.
        :param array: Flat list, list + shape dict, or nested list.
        :param indices: List of indices corresponding to array dimensions.
        :param value: Value to set.
        :param line_number: Line number for error reporting.
        :return: Updated array.
        """

        # N-D arrays are stored as a flat Python list buffer wrapped in a dict
        # with the declared shape ({'array', 'shape', 'original_shape'}).  The
        # first declared dim is the fastest: flat = i0 + s0*i1 + s0*s1*i2 + ...
        nd_shape = None
        if isinstance(array, dict) and 'array' in array:
            nd_shape = array.get('shape') or array.get('original_shape')
            if nd_shape is not None:
                nd_shape = list(nd_shape)
            array = array['array']

        # Allow dynamic list-backed arrays (None init).
        # A flat list whose first element is not itself a list is treated as a
        # 1D flat array (bounded), which falls through to the flat path below.
        if array is None:
            if not indices:
                raise ValueError(
                    f"Expected at least one index at line {line_number}")
            return {tuple(indices): value}

        # Sparse unbounded (star-dim) arrays: dict keyed by 0-based index
        # tuples. Writes grow the array freely.
        if isinstance(array, dict) and 'array' not in array:
            if not indices:
                raise ValueError(
                    f"Expected at least one index at line {line_number}")
            if any(idx < 0 for idx in indices):
                raise ConstraintError(
                    REF_ERROR,
                    f"Negative index {indices} for sparse array at line {line_number}")
            array[tuple(indices)] = value
            return array

        if not isinstance(array, list):
            raise TypeError(
                f"Expected list for multi-dimensional arrays, got {type(array)} at line {line_number}")

        shape = nd_shape if nd_shape is not None else self.get_array_shape(
            array, line_number)
        if len(indices) != len(shape):
            raise ValueError(
                f"Expected {len(shape)} indices, got {len(indices)} at line {line_number}")

        for i, idx in enumerate(indices):
            if idx < 0 or idx >= shape[i]:
                raise IndexError(
                    f"Index {idx} out of bounds for dimension {i} with size {shape[i]} at line {line_number}")

        # Column-major flat index
        flat_idx = 0
        stride = 1
        for i, idx in enumerate(indices):
            flat_idx += idx * stride
            stride *= shape[i]

        # Get flat array and update value
        flat_arr = list(array)
        flat_arr[flat_idx] = float(value) if isinstance(
            value, (int, float)) else value

        if nd_shape is not None:
            return {'array': flat_arr, 'shape': list(nd_shape),
                    'original_shape': list(nd_shape)}
        if len(shape) == 1:
            return flat_arr
        return {'array': flat_arr, 'shape': list(shape), 'original_shape': list(shape)}

    def reshape_array(self, arr, new_dims, line_number=None):
        """
        Reshape an array to new dimensions, padding with zeros if needed.
        :param arr: Array to reshape.
        :param new_dims: New dimension specs.
        :param line_number: Line number.
        :return: Reshaped array.
        """
        flat = self.flatten_array(arr, line_number)
        shape = []
        for d in new_dims:
            if d is None or (
                    isinstance(d, tuple) and len(d) == 2 and d[1] is None):
                shape.append(
                    len(flat) if not shape else len(flat) // shape[-1])
            else:
                size = d[1] if isinstance(d, tuple) else d
                shape.append(size)
        total_size = 1
        for s in shape:
            total_size *= s
        flat.extend([0] * (total_size - len(flat)))
        resized = flat[:total_size]
        if len(shape) > 1:
            return {'array': list(resized),
                    'shape': list(shape), 'original_shape': list(shape)}
        return list(resized)

    def _grid_source_extent(self, grid_source):
        """Return (max_row, max_col) of a grid source: a sparse array keyed
        by 0-based (row, col) tuples, a dense dict-form array, or a cell-ref
        dict. Coordinates are 1-based in the returned extent."""
        if not grid_source:
            return 0, 0
        if isinstance(grid_source, dict) and 'array' in grid_source:
            shape = grid_source.get('shape') or grid_source.get('original_shape') or []
            return (shape[0] if len(shape) > 0 else 0,
                    shape[1] if len(shape) > 1 else 0)
        max_row = max_col = 0
        for cell in grid_source:
            if isinstance(cell, str):
                try:
                    col, row = split_cell(cell)
                    row = int(row)
                    col = col_to_num(col)
                except ValueError:
                    continue
            elif isinstance(cell, tuple) and len(cell) == 2:
                row = int(cell[0]) + 1
                col = int(cell[1]) + 1
            else:
                continue
            max_row = max(max_row, row)
            max_col = max(max_col, col)
        return max_row, max_col

    def get_grid_row(self, row_index, grid_source=None, line_number=None):
        """Return a dense list of values in the given (1-based) grid row.

        ``grid_source`` is the grid array (sparse or dense) or a cell-ref
        dict. Unset cells within the grid's used extent read as the unset
        value (#N/A, or the declared default).
        """
        grid_source = grid_source if grid_source is not None else self.compiler.grid
        max_row, max_col = self._grid_source_extent(grid_source)
        if row_index < 1 or row_index > max_row or max_col == 0:
            return []
        if isinstance(grid_source, dict) and 'array' in grid_source:
            shape = list(grid_source.get('shape') or grid_source.get('original_shape') or [])
            flat = grid_source['array']
            rows = shape[0] if shape else 0
            cols = shape[1] if len(shape) > 1 else 0
            if row_index < 1 or row_index > rows:
                return []
            return [flat[row_index - 1 + rows * c] for c in range(cols)]
        unset = object()
        row = []
        for col in range(1, max_col + 1):
            key = (row_index - 1, col - 1)
            value = grid_source.get(key, unset)
            if value is unset:
                value = self._array_unset_value('grid', line_number)
            row.append(value)
        return row

    def get_grid_column(self, col_ref, grid_source=None, line_number=None):
        """Return a dense list of values in the given grid column.

        ``col_ref`` is a 0-based column index (e.g. 0 for column A) or a
        column letter (e.g. 'A') for compatibility; unset cells within the
        grid's used extent read as the unset value (#N/A, or the declared
        default).
        """
        grid_source = grid_source if grid_source is not None else self.compiler.grid
        if isinstance(col_ref, str) and re.match(r'^[A-Za-z]+$', col_ref):
            col_num = col_to_num(col_ref)
        else:
            col_num = int(col_ref) + 1
        max_row, max_col = self._grid_source_extent(grid_source)
        if col_num < 1 or col_num > max_col or max_row == 0:
            return []
        if isinstance(grid_source, dict) and 'array' in grid_source:
            shape = list(grid_source.get('shape') or grid_source.get('original_shape') or [])
            flat = grid_source['array']
            rows = shape[0] if shape else 0
            cols = shape[1] if len(shape) > 1 else 0
            if col_num < 1 or col_num > cols:
                return []
            return [flat[r + rows * (col_num - 1)] for r in range(rows)]
        unset = object()
        values = []
        for r in range(max_row):
            key = (r, col_num - 1)
            value = grid_source.get(key, unset)
            if value is unset:
                value = self._array_unset_value('grid', line_number)
            values.append(value)
        return values

    def get_array_element(self, arr, indices, line_number=None, return_struct=False, original_shape=None, var_name=None):
        """
        Get an element from an array using indices.
        :param arr: Array or dict with 'array' key.
        :param indices: List of indices.
        :param line_number: Line number.
        :param return_struct: Return struct if True.
        :param original_shape: Original shape if provided.
        :param var_name: Array variable name, used to apply a 'default'
            constraint to unset items.
        :return: Element value.
        """
        # Handle dictionary with array (e.g., from grid DIM or N-D storage)
        if isinstance(arr, dict) and 'array' in arr:
            original_shape = arr.get(
                'shape') or arr.get('original_shape')
            original_shape = list(original_shape)
            arr = arr['array']

        # Sparse unbounded (star-dim) arrays: dict keyed by index tuples.
        # Any non-negative index is a valid address (no upper bound), so a
        # key that was never set reads as #N/A, while a wrong number of
        # indices or a negative index is an invalid address (#REF).
        if isinstance(arr, dict):
            first_key = next(iter(arr.keys()), None)
            rank = len(first_key) if first_key is not None else len(indices)
            if len(indices) != rank or any(i < 0 for i in indices):
                raise ConstraintError(
                    REF_ERROR,
                    f"Expected {rank} indices for array, got {len(indices)} at line {line_number}")
            result = arr.get(tuple(indices))
            if result is None:
                if var_name:
                    return self._array_unset_value(var_name, line_number)
                return error_value(NA_ERROR)
            return result

        # Get the shape to use for indexing - prefer the declared shape
        shape = self.get_array_shape(arr, line_number)
        indexing_shape = original_shape if original_shape is not None else getattr(
            arr, 'original_shape', shape)

        # Validate indices against the declared shape
        if len(indices) != len(indexing_shape):
            raise ConstraintError(
                REF_ERROR,
                f"Expected {len(indexing_shape)} indices for array with shape {indexing_shape}, got {len(indices)} at line {line_number}")

        validation_shape = indexing_shape
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= validation_shape[i]:
                raise ConstraintError(
                    REF_ERROR,
                    f"Index {idx} out of bounds for dimension {i} with size {validation_shape[i]} at line {line_number}")

        # Handle array indexing (column-major: first dim is fastest)
        if isinstance(arr, list):
            if arr and isinstance(arr[0], list):
                # Python nested lists (e.g. object arrays) are stored outermost
                # = first declared dim
                result = arr
                for idx in indices:
                    result = result[idx]
                if isinstance(result, dict) and 'value' in result:
                    return result['value']
                return result
            flat_arr = list(arr)
            flat_idx = 0
            stride = 1
            for i, idx in enumerate(indices):
                flat_idx += idx * stride
                stride *= indexing_shape[i]

            if flat_idx < 0 or flat_idx >= len(flat_arr):
                raise ConstraintError(
                    REF_ERROR,
                    f"Calculated index {flat_idx} out of bounds for array length {len(flat_arr)} at line {line_number}")

            result = flat_arr[flat_idx]
            if isinstance(result, dict) and 'value' in result:
                return result['value']
            if result is None:
                if var_name:
                    return self._array_unset_value(var_name, line_number)
                return error_value(NA_ERROR)
            return result
        else:
            raise TypeError(
                f"Cannot index non-array type {type(arr)} at line {line_number}")

    def read_array_element(self, arr, indices, line_number=None, return_struct=False, original_shape=None, var_name=None):
        """Read an element from an array; an invalid address produces the
        sticky ``#REF`` error value instead of raising."""
        if is_error_value(arr):
            code = arr if isinstance(arr, str) else arr.error_code
            return error_value(code)
        try:
            return self.get_array_element(
                arr, indices, line_number, return_struct, original_shape,
                var_name)
        except (IndexError, ConstraintError):
            return error_value(REF_ERROR)

    def read_array_slice(self, arr, specs, line_number=None, var_name=None):
        """Read a sub-array selected by ``*`` / ``n to m`` / scalar specs.

        ``specs`` holds one selector per declared dimension, already adjusted
        to storage (0-based) coordinates:
          - ``'*'``                : take all values in that dimension
          - ``(start, end)``       : inclusive range; ``end`` may be None
                                     (open-ended)
          - ``int``                : single index

        Scalar dimensions are reduced (excluded from the result): the result
        rank equals the number of ``*``/range selectors.  Invalid addresses
        produce the sticky ``#REF`` error value, mirroring element reads.
        """
        if is_error_value(arr):
            code = arr if isinstance(arr, str) else arr.error_code
            return error_value(code)
        if all(isinstance(s, int) for s in specs):
            try:
                return self.get_array_element(
                    arr, specs, line_number, var_name=var_name)
            except (IndexError, ConstraintError):
                return error_value(REF_ERROR)
        # Sparse unbounded (star-dim) arrays: dict keyed by index tuples.
        if isinstance(arr, dict) and 'array' not in arr:
            return self._read_sparse_slice(arr, specs, line_number)

        original_shape = None
        if isinstance(arr, dict) and 'array' in arr:
            original_shape = arr.get('shape') or arr.get('original_shape')
            original_shape = list(original_shape)
            arr = arr['array']

        shape = self.get_array_shape(arr, line_number)
        indexing_shape = original_shape if original_shape is not None else getattr(
            arr, 'original_shape', shape)

        if arr and isinstance(arr[0], list):
            return self._slice_nested_lists(arr, specs, line_number)

        if len(specs) != len(indexing_shape):
            raise ConstraintError(
                REF_ERROR,
                f"Expected {len(indexing_shape)} indices for array with shape "
                f"{indexing_shape}, got {len(specs)} at line {line_number}")

        ranges = []
        kept = []
        result_shape = []
        for i, spec in enumerate(specs):
            size = indexing_shape[i]
            if isinstance(spec, int):
                s = e = spec
            elif spec == '*':
                s, e = 0, size - 1
            else:
                s, e = spec
                if e is None:
                    e = size - 1
            if s < 0 or e >= size or s > e:
                raise ConstraintError(
                    REF_ERROR,
                    f"Index {spec} out of bounds for dimension {i} with size "
                    f"{size} at line {line_number}")
            ranges.append((s, e))
            if not isinstance(spec, int):
                kept.append(i)
                result_shape.append(e - s + 1)

        # Iterate full index combos with the first declared dim fastest, to
        # match the flat storage order of get_array_element.
        axes = [range(s, e + 1) for s, e in reversed(ranges)]
        result = []
        for combo in itertools.product(*axes):
            idx = tuple(reversed(combo))
            flat_idx = 0
            stride = 1
            for j, v in enumerate(idx):
                flat_idx += v * stride
                stride *= indexing_shape[j]
            val = arr[flat_idx]
            if isinstance(val, dict) and 'value' in val:
                val = val['value']
            result.append(val)
        if len(result_shape) <= 1:
            return result
        return {'array': result, 'shape': result_shape,
                'original_shape': result_shape}

    def _read_sparse_slice(self, arr, specs, line_number=None):
        first_key = next(iter(arr.keys()), None)
        rank = len(first_key) if first_key is not None else len(specs)
        if len(specs) != rank:
            raise ConstraintError(
                REF_ERROR,
                f"Expected {rank} indices for array, got {len(specs)} at line {line_number}")
        for spec in specs:
            if isinstance(spec, int) and spec < 0:
                raise ConstraintError(
                    REF_ERROR,
                    f"Negative index {spec} for sparse array at line {line_number}")
            if isinstance(spec, tuple) and spec[0] < 0:
                raise ConstraintError(
                    REF_ERROR,
                    f"Negative index {spec[0]} for sparse array at line {line_number}")

        matched = []
        for key, val in arr.items():
            ok = True
            for i, spec in enumerate(specs):
                k = key[i]
                if isinstance(spec, int):
                    if k != spec:
                        ok = False
                        break
                elif spec == '*':
                    continue
                else:
                    s, e = spec
                    if e is not None:
                        if k < s or k > e:
                            ok = False
                            break
                    elif k < s:
                        ok = False
                        break
            if ok:
                matched.append((key, val))

        kept = [i for i, spec in enumerate(specs)
                if not isinstance(spec, int)]
        if not kept:
            try:
                return arr.get(tuple(int(s) for s in specs))
            except (IndexError, ConstraintError):
                return error_value(REF_ERROR)
        # A slice of a sparse array stays sparse: keys are the kept storage
        # coordinates re-based so each kept dimension starts at 0.
        result = {}
        for key, val in matched:
            new_key = []
            for i in kept:
                spec = specs[i]
                base = 0 if spec == '*' else spec[0]
                new_key.append(key[i] - base)
            result[tuple(new_key)] = val
        return result

    def _slice_nested_lists(self, arr, specs, line_number=None):
        def walk(node, dim):
            if isinstance(node, dict) and 'value' in node:
                node = node['value']
            if dim >= len(specs):
                return node
            spec = specs[dim]
            if isinstance(spec, int):
                if spec < 0 or spec >= len(node):
                    raise ConstraintError(
                        REF_ERROR,
                        f"Index {spec} out of bounds for dimension {dim} "
                        f"with size {len(node)} at line {line_number}")
                return walk(node[spec], dim + 1)
            size = len(node)
            if spec == '*':
                s, e = 0, size - 1
            else:
                s, e = spec
                if e is None:
                    e = size - 1
            if s < 0 or e >= size or s > e:
                raise ConstraintError(
                    REF_ERROR,
                    f"Index {spec} out of bounds for dimension {dim} with "
                    f"size {size} at line {line_number}")
            return [walk(node[k], dim + 1) for k in range(s, e + 1)]

        return walk(arr, 0)

    def fill_array(self, array, value, line_number=None):
        """
        Fill an existing array with a value (scalar or list).
        :param array: Array to fill.
        :param value: Fill value.
        :param line_number: Line number.
        :return: Filled array.
        """
        nd_shape = None
        if isinstance(array, dict) and 'array' in array:
            nd_shape = array.get('shape') or array.get('original_shape')
            if nd_shape is not None:
                nd_shape = list(nd_shape)
            array = array['array']
        if not isinstance(array, list):
            raise ValueError(
                f"Expected list, got {type(array)} at line {line_number}")

        shape = nd_shape if nd_shape is not None else self.get_array_shape(
            array, line_number)
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
            return list(values)
        return {'array': list(values), 'shape': list(shape), 'original_shape': list(shape)}

    def get_array_dimensions(self, arr):
        """
        Get the dimensions of an array.
        """
        if isinstance(arr, dict) and ('shape' in arr or 'original_shape' in arr):
            return list(arr.get('shape') or arr.get('original_shape'))
        if isinstance(arr, dict) and 'array' in arr:
            return [len(arr['array'])]
        if isinstance(arr, dict) and arr and all(
                isinstance(k, tuple) for k in arr.keys()):
            shape = [0] * len(next(iter(arr.keys())))
            for k in arr.keys():
                for i, idx in enumerate(k):
                    if idx + 1 > shape[i]:
                        shape[i] = idx + 1
            return shape
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
