import re
import math
import pyarrow as pa
from expression import ExpressionEvaluator
from array_handler import ArrayHandler
from utils import col_to_num, num_to_col, split_cell, offset_cell, validate_cell_ref


class Scope:
    def __init__(self, compiler, parent=None, is_private=False):
        self.compiler = compiler
        self.variables = {}
        self.types = {}
        self.constraints = {}
        self.uninitialized = set()
        self.parent = parent
        self.is_private = is_private
        self.pending_assignments = {}

    def define(self, name, value=None, type=None, constraints=None, is_uninitialized=False):
        if name in self.variables and not is_uninitialized:
            raise ValueError(
                f"Variable '{name}' already defined in this scope")
        self.variables[name] = value
        self.types[name] = type
        self.constraints[name] = constraints or {}
        if is_uninitialized:
            self.uninitialized.add(name)
        else:
            self.uninitialized.discard(name)

    def update(self, name, value, line_number=None):
        defining_scope = self.get_defining_scope(name)
        if defining_scope:
            defining_scope._check_constraints(name, value, line_number)
            defining_scope.variables[name] = value
            defining_scope.uninitialized.discard(name)
        else:
            if self.is_shadowed(name) and not self.is_private:
                print(
                    f"Warning: '{name}' shadows a variable in an outer scope at line {line_number}")
            self.define(name, value)

    def get(self, name):
        if name in self.variables:
            return self.variables[name]
        if self.parent and not self.is_private:
            return self.parent.get(name)
        raise NameError(f"Variable '{name}' not defined")

    def is_uninitialized(self, name):
        if name in self.uninitialized:
            return True
        if self.parent and not self.is_private:
            return self.parent.is_uninitialized(name)
        return False

    def get_defining_scope(self, var):
        current = self
        while current:
            if var in current.variables:
                return current
            current = current.parent
        return None

    def is_shadowed(self, name):
        current = self.parent
        while current:
            if name in current.variables:
                return True
            current = current.parent
        return False

    def get_evaluation_scope(self):
        full_scope = {}
        current = self
        full_scope.update(current.variables)
        current = current.parent
        while current and not current.is_private:
            full_scope.update(current.variables)
            current = current.parent
        return full_scope

    def _check_constraints(self, name, value, line_number=None):
        constraints = self.constraints.get(name, {})
        for constraint_type, constraint_expr in constraints.items():
            if constraint_type == 'constant':
                if isinstance(constraint_expr, str):
                    constraint_val = self.compiler.expr_evaluator.eval_expr(
                        constraint_expr, self.get_full_scope(), line_number)
                else:
                    constraint_val = constraint_expr
                if value != constraint_val:
                    raise ValueError(
                        f"Cannot change constant '{name}' at line {line_number}")
            elif constraint_type in ('<=', '>=', '<', '>'):
                constraint_val = float(self.compiler.expr_evaluator.eval_expr(
                    constraint_expr, self.get_full_scope(), line_number))
                if constraint_type == '<=' and value > constraint_val:
                    raise ValueError(
                        f"'{name}' exceeds maximum {constraint_val} at line {line_number}")
                elif constraint_type == '>=' and value < constraint_val:
                    raise ValueError(
                        f"'{name}' is below minimum {constraint_val} at line {line_number}")
                elif constraint_type == '<' and value >= constraint_val:
                    raise ValueError(
                        f"'{name}' is not less than {constraint_val} at line {line_number}")
                elif constraint_type == '>' and value <= constraint_val:
                    raise ValueError(
                        f"'{name}' is not greater than {constraint_val} at line {line_number}")
            elif constraint_type == 'in':
                if value not in constraint_expr:
                    raise ValueError(
                        f"'{name}' value {value} not in allowed values {constraint_expr} at line {line_number}")
            elif constraint_type == 'type':
                expected_type = constraint_expr.lower()
                actual_type = self.compiler.array_handler.infer_type(
                    value, line_number)
                if expected_type == 'number' and actual_type not in ('number', 'float64'):
                    raise ValueError(
                        f"'{name}' must be a number, got {actual_type} at line {line_number}")
                elif expected_type == 'text' and actual_type != 'string':
                    raise ValueError(
                        f"'{name}' must be text, got {actual_type} at line {line_number}")
            elif constraint_type == 'unit':
                pass

    def get_full_scope(self):
        full_scope = {}
        current = self
        while current and not current.is_private:
            full_scope.update(current.variables)
            current = current.parent
        return full_scope


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
        self.types_defined = {}
        self.expr_evaluator = ExpressionEvaluator(self)
        self.array_handler = ArrayHandler(self)
        self.handled_assignments = set()

    def current_scope(self):
        return self.scopes[-1]

    def push_scope(self, is_private=False):
        self.scopes.append(
            Scope(self, parent=self.current_scope(), is_private=is_private))

    def pop_scope(self):
        if len(self.scopes) > 1:
            self.scopes.pop()
        else:
            raise RuntimeError("Cannot pop global scope")

    def run(self, code):
        self._reset_state()
        lines, label_lines, dim_lines = self._preprocess_code(code)
        self._process_declarations_and_labels(lines, label_lines, dim_lines)

        i = 0
        while i < len(lines):
            line, line_number = lines[i]
            if not line.strip():
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
            elif line.lower().startswith("for "):
                m = re.match(r'^\s*FOR\s+(.+?)(?:\s+do\s*$|\s*$)', line, re.I)
                if not m:
                    raise SyntaxError(
                        f"Invalid FOR syntax at line {line_number}")
                var_defs = m.group(1).strip()
                is_block = line.strip().lower().endswith('do')

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
                        self.current_scope().define(var, evaluated_value, inferred_type, constraints, False)
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

                # Handle FOR blocks or LET followers
                for var, _, _, _ in var_list:
                    defining_scope = self.current_scope().get_defining_scope(var)
                    if defining_scope and (is_block or not (i + 1 < len(lines) and lines[i + 1][0].lower().startswith('let '))):
                        raise ValueError(
                            f"Variable '{var}' already defined in scope at line {line_number}")

                self.push_scope(is_private=True)
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
                            self.current_scope().define(var, None, type_name, constraints, is_uninitialized=True)
                    else:
                        self.current_scope().define(var, None, type_name, constraints, is_uninitialized=True)

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
                    self._process_block(for_block_lines)
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
                        self._process_block(for_block_lines)
                        self.pop_scope()
                    else:
                        self.pop_scope()
                i += 1
            elif line.lower().startswith("let "):
                m = re.match(
                    r'^\s*LET\s+(.+?)(?:\s+then\s*$|\s*$)', line, re.I)
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
                    defining_scope = self.current_scope().get_defining_scope(var)
                    if defining_scope:
                        if constraints:
                            defining_scope.constraints[var] = constraints
                        if expr is None and var in defining_scope.variables and defining_scope.variables[var] is not None:
                            continue
                    else:
                        if self.current_scope().is_shadowed(var):
                            pass
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
                    block_pending = self._process_block(block_lines)
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
                            pass
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
                    if re.match(r'^\[[A-Z]+\d+\]$', target):
                        cell_ref = target[1:-1].strip()
                        try:
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
                else:
                    i += 1

        self._resolve_pending_assignments()
        return self.grid

    def process_for_statement(self, line, line_number, scope):

        m = re.match(r'For\s+([\w_]+)\s+dim\s+(\d+)\s+to\s+(\d+)', line, re.I)
        if m:
            var_name, start, end = m.groups()
            start, end = int(start), int(end)
            if var_name in scope.variables:
                scope.constraints[var_name]['dim'] = [(None, (start, end))]
            else:
                scope.define(var_name, None, None, {
                             'dim': [(None, (start, end))]}, True)
            return

        m = re.match(r'For\s+([\w_]+)\s*=\s*(.+)', line, re.I)
        if m:
            var_name, expr = m.groups()

            value = self.expr_evaluator.eval_or_eval_array(
                expr, scope.get_evaluation_scope(), line_number)

            defining_scope = scope.get_defining_scope(var_name)
            if defining_scope:
                if defining_scope.types.get(var_name) == 'number' and not isinstance(value, (int, float)):
                    raise TypeError(
                        f"Cannot assign non-numeric value {value} to '{var_name}' at line {line_number}")
                defining_scope.update(var_name, float(value) if isinstance(
                    value, (int, float)) else value, line_number)
            else:
                inferred_type = self.array_handler.infer_type(
                    value, line_number)
                if inferred_type == 'int':
                    inferred_type = 'number'
                scope.define(var_name, value, inferred_type,
                             {'constant': value}, False)

            pending = list(scope.pending_assignments.items())
            for key, (expr, ln, deps) in pending:
                unresolved = any(self.current_scope().is_uninitialized(
                    dep) or dep in self.pending_assignments for dep in deps)
                if not unresolved:
                    try:
                        self.array_handler.evaluate_line_with_assignment(
                            expr, ln, scope.get_evaluation_scope())
                        del scope.pending_assignments[key]
                    except Exception as e:
                        pass
            return

        raise SyntaxError(f"Invalid FOR syntax at line {line_number}")

    def _resolve_global_dependency(self, var, line_number, target_scope=None):
        if var not in self.pending_assignments:
            return False
        expr, assign_line, deps = self.pending_assignments[var]
        scope = target_scope if target_scope is not None else self.current_scope()
        unresolved = any(scope.is_uninitialized(dep) or (
            dep in self.pending_assignments and dep != var) for dep in deps)
        if unresolved:
            return False
        try:
            value = self.expr_evaluator.eval_or_eval_array(
                expr, scope.get_full_scope(), assign_line)
            value = self.array_handler.check_dimension_constraints(
                var, value, assign_line)
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

    def _process_block(self, block_lines):
        block_pending = {}
        i = 0
        while i < len(block_lines):
            line, line_number = block_lines[i]
            if not line.strip():
                i += 1
                continue
            if line.strip().lower().startswith("for "):
                m = re.match(r'^\s*FOR\s+(.+?)(?:\s+do\s*$|\s*$)', line, re.I)
                if m:
                    var_defs = m.group(1).strip()
                    is_block = line.strip().lower().endswith('do')
                    var_list = []
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

                    for var, _, _, _ in var_list:
                        defining_scope = self.current_scope().get_defining_scope(var)
                        if defining_scope:
                            if is_block or not (i + 1 < len(block_lines) and block_lines[i + 1][0].lower().startswith('let ')):
                                raise ValueError(
                                    f"Variable '{var}' already defined in scope at line {line_number}")

                    self.push_scope(is_private=True)
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
                                self.current_scope().define(var, None, type_name, constraints, is_uninitialized=True)
                        else:
                            self.current_scope().define(var, None, type_name, constraints, is_uninitialized=True)

                    if is_block:
                        for_block_lines = []
                        i += 1
                        depth = 1
                        while i < len(block_lines) and depth > 0:
                            next_line, next_line_number = block_lines[i]
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
                            for_block_lines.append(
                                (next_line, next_line_number))
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
                        self._process_block(for_block_lines)
                    else:
                        if len(var_list) == 1 and i + 1 < len(block_lines) and block_lines[i + 1][0].lower().startswith('let '):
                            for_block_lines = []
                            i += 1
                            for_block_lines.append(block_lines[i])
                            if i + 1 < len(block_lines) and not block_lines[i + 1][0].strip().lower() == "end":
                                for_block_lines.append(block_lines[i + 1])
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
                            self._process_block(for_block_lines)
                    self.pop_scope()
            elif ':=' in line:
                target, rhs = line.split(':=')
                target, rhs = target.strip(), rhs.strip()
                rhs_vars = set(re.findall(
                    r'\b[\w_]+\b(?=\s*(?:[\[\{]|!\w+\s*\(|(?:\.\w+)?\s*$))', rhs))
                if '$"' in rhs:
                    placeholders = re.findall(r'\{\s*([^}]*?)\s*\}', rhs)
                    for ph in placeholders:
                        rhs_vars.update(re.findall(r'\b[\w_]+\b', ph))
                field_vars = set(re.findall(r'\b[\w_]+\b(?=\.\w+\s*$)', rhs))
                rhs_vars.update(field_vars)
                unresolved = any(self.current_scope().is_uninitialized(
                    var) or var in self.pending_assignments for var in rhs_vars)
                if unresolved:
                    constraints = {}
                    for var in rhs_vars:
                        if var in self.current_scope().constraints:
                            constraints[var] = self.current_scope(
                            ).constraints[var]
                        defining_scope = self.current_scope().get_defining_scope(var)
                        if defining_scope and var in defining_scope.constraints:
                            constraints[var] = defining_scope.constraints[var]
                    block_pending[f"__block_line_{line_number}"] = (
                        line, line_number, rhs_vars, constraints)
                    self.current_scope().pending_assignments[f"__block_line_{line_number}"] = (
                        line, line_number, rhs_vars, constraints)
                    i += 1
                    continue
                violations = []
                for var in rhs_vars:
                    defining_scope = self.current_scope().get_defining_scope(var)
                    if defining_scope and var in defining_scope.constraints:
                        try:
                            var_value = defining_scope.get(var)
                            if var_value is not None:
                                defining_scope._check_constraints(
                                    var, var_value, line_number)
                        except ValueError as e:
                            violations.append(var)
                if not violations:
                    self.array_handler.evaluate_line_with_assignment(
                        line, line_number, self.current_scope().get_evaluation_scope())
                else:
                    self.grid.clear()
                i += 1
            elif line.strip().lower().startswith('let '):
                m = re.match(
                    r'^\s*LET\s+(.+?)(?:\s+then\s*$|\s*$)', line, re.I)
                if not m:
                    raise SyntaxError(
                        f"Invalid LET syntax at line {line_number}")
                var_def = m.group(1).strip()
                try:
                    var, type_name, constraints, value = self._parse_variable_def(
                        var_def, line_number)
                    defining_scope = self.current_scope().get_defining_scope(var)
                    if defining_scope and value is None:
                        if var in defining_scope.variables and defining_scope.variables[var]:
                            value = defining_scope.variables[var]
                    else:
                        if self.current_scope().is_shadowed(var):
                            pass
                        self.current_scope().define(var, value, type_name, constraints)
                        if value is not None:
                            try:
                                evaluated_value = self.expr_evaluator.eval_expr(
                                    str(value), self.current_scope().get_evaluation_scope(), line_number)
                                self.current_scope().update(var, evaluated_value, line_number)
                            except NameError as e:
                                pass
                except Exception as e:
                    pass
                i += 1
            else:
                i += 1
        if block_pending:
            self.scopes[0].pending_assignments.update(block_pending)
        return block_pending

    def _resolve_pending_assignments(self):
        max_attempts = len(self.pending_assignments) + 10
        attempt = 0
        while self.pending_assignments and attempt < max_attempts:
            unresolved_before = set(self.pending_assignments.keys())
            for var, assignment in sorted(self.pending_assignments.items(), key=lambda x: (x[0].startswith('__line_'), int(x[0].replace('__line_', '') if x[0].startswith('__line_') else '0'))):
                expr, line_number, deps = assignment[:3]
                if var in deps and not var.startswith('__line_'):
                    pass
                    raise ValueError(
                        f"Self-referential assignment '{var} = {expr}' at line {line_number}")
                if var.startswith("__line_"):
                    target, rhs = expr.split(':=')
                    target, rhs = target.strip(), rhs.strip()
                    unresolved = any(self.current_scope().is_uninitialized(
                        dep) or dep in self.pending_assignments for dep in deps)
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
                            self.array_handler.evaluate_line_with_assignment(
                                expr, line_number, self.current_scope().get_evaluation_scope())
                            del self.pending_assignments[var]
                        else:
                            self.grid.clear()
                            del self.pending_assignments[var]
                    except NameError as e:
                        continue
                    except Exception as e:
                        raise RuntimeError(
                            f"Error resolving '{var}' from '{expr}': {e} at line {line_number}")
                else:
                    cell_refs = self._extract_cell_refs(expr)
                    if cell_refs and not all(ref in self.grid for ref in cell_refs):
                        continue
                    unresolved = any(self.current_scope().is_uninitialized(
                        dep) or dep in self.pending_assignments for dep in deps)
                    if unresolved:
                        continue
                    try:
                        value = self.expr_evaluator.eval_or_eval_array(
                            expr, self.current_scope().get_full_scope(), line_number)
                        value = self.array_handler.check_dimension_constraints(
                            var, value, line_number)
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
            unresolved = any(self.current_scope().is_uninitialized(
                dep) or dep in self.pending_assignments for dep in deps)
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
                    self.array_handler.evaluate_line_with_assignment(
                        expr, line_number, self.current_scope().get_evaluation_scope())
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
        def_str = def_str.strip()
        constraints = {}
        expr = None
        type_name = None
        unit = None
        dims = None

        # Handle field assignments (e.g., p.x = value)
        field_match = re.match(r'^([\w_]+)\.(\w+)\s*=\s*(.+)$', def_str)
        if field_match:
            var, field, expr_str = field_match.groups()
            expr = expr_str
            constraints['constant'] = expr
            return f"{var}.{field}", None, constraints, expr

        # Extract 'with' clause early to preserve parentheses
        with_content_str = None
        with_match = re.search(r'\s+with\s+(\(.*\))', def_str, re.I)
        if with_match:
            with_content_str = with_match.group(1)
            def_str = def_str[:with_match.start()] + def_str[with_match.end():]

        # Split on keywords (as, of, dim, in, <=, >=, <, >, =, with)
        parts = re.split(r'\s+(as|of|dim|in|<=|>=|<|>|=)\s+',
                         def_str, flags=re.I)
        var = parts[0].strip()
        i = 1
        while i < len(parts):
            keyword = parts[i].lower()
            next_part = parts[i + 1].strip() if i + 1 < len(parts) else ''
            if keyword == 'as':
                type_name = next_part.lower()
            elif keyword == 'of':
                unit = next_part
                constraints['unit'] = unit
            elif keyword == 'dim':
                dims = next_part
                constraints['dim'] = dims
            elif keyword == 'in':
                if not (next_part.startswith('{') and next_part.endswith('}')):
                    raise SyntaxError(
                        f"Invalid 'in' constraint syntax: '{next_part}' at line {line_number}")
                values = [v.strip()
                          for v in next_part[1:-1].split(',') if v.strip()]
                constraints['in'] = values
            elif keyword in ('<=', '>=', '<', '>'):
                constraints[keyword] = next_part
            elif keyword == '=':
                if next_part.startswith('{'):
                    if ';' in next_part or '|' in next_part:
                        matrices = [m.strip()[1:-1]
                                    for m in next_part.split('|')]
                        matrix_data = []
                        for matrix in matrices:
                            rows = [r.strip().split(',')
                                    for r in matrix.split(';')]
                            matrix_data.append(
                                [[float(v.strip()) for v in row] for row in rows])
                        expr = matrix_data
                    else:
                        values_str = next_part[1:-1].split(',')
                        values = []
                        for v in values_str:
                            v_clean = v.strip()
                            if v_clean.startswith('"') and v_clean.endswith('"') or v_clean.startswith("'") and v_clean.endswith("'"):
                                val = v_clean[1:-1]
                            else:
                                try:
                                    val = float(v_clean)
                                except ValueError:
                                    val = v_clean
                            values.append(val)
                        expr = values
                else:
                    expr = next_part
            i += 2

        # Process 'with' clause
        if with_content_str:
            with_inner_content = with_content_str[1:-1].strip()
            with_constraints = {}
            dim_constraint = None
            wc_parts = []
            current = ""
            in_quotes = False
            paren_level, brace_level = 0, 0
            for char in with_inner_content + ',':
                if char == '"' and (len(current) == 0 or current[-1] != '\\'):
                    in_quotes = not in_quotes
                elif not in_quotes:
                    if char == '(':
                        paren_level += 1
                    elif char == ')':
                        paren_level -= 1
                    elif char == '{':
                        brace_level += 1
                    elif char == '}':
                        brace_level -= 1
                if char == ',' and not in_quotes and paren_level == 0 and brace_level == 0:
                    if current.strip():
                        wc_parts.append(current.strip())
                    current = ""
                else:
                    current += char
            for wc in wc_parts:
                wc = wc.strip()
                if wc.lower().startswith('grid dim'):
                    if dim_constraint is not None:
                        raise SyntaxError(
                            f"Multiple grid DIM statements not allowed in with clause at line {line_number}")
                    dim_match = re.match(
                        r'grid\s+dim\s*\{([^}]+)\}\s*(?:=\s*({.+?}(?:\s*\|\s*{.+?})*|[\d.]+|\w+))?', wc, re.I)
                    if not dim_match:
                        raise SyntaxError(
                            f"Invalid dimension syntax: '{wc}' at line {line_number}")
                    dim_str = dim_match.group(1)
                    grid_data = dim_match.group(2)
                    dim_parts = [p.strip() for p in dim_str.split(',')]
                    dim_constraint = {
                        'dims': [(1, int(p.strip())) for p in dim_parts]}
                    if grid_data:
                        if re.match(r'^{.+?}(?:\s*\|\s*{.+?})*$', grid_data):
                            matrices = [m.strip()[1:-1]
                                        for m in grid_data.split('|')]
                            matrix_data = []
                            for matrix in matrices:
                                rows = [r.strip().split(',')
                                        for r in matrix.split(';')]
                                matrix_data.append(
                                    [[float(v.strip()) for v in row] for row in rows])
                            dim_constraint['matrix_data'] = matrix_data
                        elif re.match(r'^[\d.]+$', grid_data):
                            dim_constraint['value'] = float(grid_data)
                        elif re.match(r'^\w+$', grid_data):
                            dim_constraint['data_var'] = grid_data
                        else:
                            raise SyntaxError(
                                f"Invalid grid data: '{grid_data}' at line {line_number}")
                elif '=' in wc:
                    k, v = [s.strip() for s in wc.split('=', 1)]
                    if v.startswith('"') and v.endswith('"'):
                        with_constraints[k] = v[1:-1]
                    else:
                        try:
                            with_constraints[k] = self.expr_evaluator.eval_expr(
                                v, self.current_scope().get_full_scope(), line_number)
                        except Exception:
                            v = v.strip('"')
                            with_constraints[k] = v
            if with_constraints:
                constraints['with'] = with_constraints
            if dim_constraint:
                constraints['dim'] = dim_constraint

        # Process dimensions
        if dims:
            dim_parts = dims[1:-
                             1].split(',') if dims.startswith('{') else [dims]
            dims_list = []
            for part in dim_parts:
                part = part.strip()
                if ':' in part:
                    name, size = map(str.strip, part.split(':'))
                    size_spec = self._parse_dim_size(size, line_number)
                    dims_list.append((name, size_spec))
                else:
                    size_spec = self._parse_dim_size(part, line_number)
                    dims_list.append((None, size_spec))
            constraints['dim'] = dims_list
        elif constraints.get('with', {}).get('dim'):
            dim_str = constraints['with']['dim']
            m = re.match(r'^\s*grid\s+dim\s*\{([^}]+)\}\s*$', dim_str, re.I)
            if not m:
                raise SyntaxError(
                    f"Invalid dimension syntax: '{dim_str}' at line {line_number}")
            dim_content = m.group(1).strip()
            dim_parts = [p.strip()
                         for p in dim_content.split(',') if p.strip()]
            dims_list = []
            for part in dim_parts:
                size_spec = self._parse_dim_size(part, line_number)
                dims_list.append((None, size_spec))
            constraints['dim'] = dims_list

        return var, type_name, constraints, expr

    def _extract_cell_refs(self, expr):
        cell_refs = set()
        single_matches = re.finditer(r'\[([A-Z]+\d+)\]', expr)
        for match in single_matches:
            cell_refs.add(match.group(1))
        range_matches = re.finditer(r'\[([A-Z]+\d+):([A-Z]+\d+)\]', expr)
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
                cell_refs.update(re.findall(r'\b[A-Z]+\d+\b', ph))
        return cell_refs

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
        self.types_defined.clear()
        self.handled_assignments.clear()

    def _preprocess_code(self, code):

        lines = []
        label_lines = []
        dim_lines = []
        type_def_lines = []

        in_type_def = False
        type_name = None
        line_number = 0
        current_line = ""
        in_multiline = False

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

            # Normalize brackets like [ A 12 ] → [A12]
            s = re.sub(r'\[\s*([A-Z]+)\s+[A-Z]*(\d+)\s*\]', r'[\1\2]', s)

            # Handle start of type definition
            if s.lower().startswith("define "):
                in_type_def = True
                m = re.match(r'^\s*define\s+([\w_]+)\s+as\s+type\s*$', s, re.I)
                if not m:
                    raise SyntaxError(
                        f"Invalid type definition syntax: '{s}' at line {line_number}")
                type_name = m.group(1).strip()
                type_def_lines = []
                continue

            # Handle end of type definition
            elif s.lower().startswith("end ") and in_type_def:
                in_type_def = False
                self.types_defined[type_name.lower()] = self._parse_type_def(
                    type_def_lines, line_number)
                continue

            # Inside type definition block
            elif in_type_def:
                type_def_lines.append(s.lstrip())
                continue

            # Handle multiline assignments (e.g., [@A1] := $" ... multiline ... ")
            if ':=' in s and s.startswith('[') and '$"' in s and not s.endswith('"'):
                current_line = s
                in_multiline = True
                continue
            elif in_multiline:
                current_line += "\n" + line.lstrip()
                if line.rstrip().endswith('"'):
                    lines.append((current_line, line_number))
                    current_line = ""
                    in_multiline = False
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

        return lines, label_lines, dim_lines

    def _parse_type_def(self, lines, line_number=None):
        fields = {}
        for line in lines:
            if line.startswith(':'):
                parts = line[1:].strip().split(' as ')
                if len(parts) != 2:
                    raise SyntaxError(
                        f"Invalid field definition: '{line}' at line {line_number}")
                field_name, field_type = parts
                fields[field_name.strip()] = field_type.strip()
        return fields

    def _process_declarations_and_labels(self, lines, label_lines, dim_lines):
        for line, line_number in dim_lines:
            self._collect_global_declarations(line, line_number)
        for line, line_number in lines:
            if line.startswith(':') and not line.lower().startswith(("for ", "let ")):
                self._collect_global_declarations(line, line_number)
        for line, line_number in label_lines:
            self._process_label_assignment(line, line_number)
        for line, line_number in lines:
            if not line.startswith(':') and ':=' not in line and '!' not in line:
                self._evaluate_cell_var_definition(line, line_number)

    def _collect_global_declarations(self, line, line_number=None):
        a = line[1:].strip()
        m_t = re.match(
            r'^([\w_]+)\s+as\s+(number|text|array)\s*(dim\s*\{[^{}]*\})?\s*(?:=\s*(.+))?$', a, re.I | re.S)
        if m_t:
            var, d_type, dim_part, v_expr = m_t.groups()
            var, d_type = var.strip(), d_type.lower()
            effective_type = 'array' if dim_part else d_type
            self.current_scope().types[var] = effective_type
            constraints = {}
            if dim_part:
                dims = []
                dim_content = dim_part[len('dim '):].strip()[1:-1].strip()
                if dim_content:
                    parts = [p.strip() for p in dim_content.split(',')]
                    for part in parts:
                        if ':' in part:
                            name, size = map(str.strip, part.split(':'))
                            size_spec = self._parse_dim_size(size, line_number)
                            dims.append((name, size_spec))
                        else:
                            size_spec = self._parse_dim_size(part, line_number)
                            dims.append((None, size_spec))
                    self.dimensions[var] = dims
                    self.dim_names[var] = {
                        name: idx for idx, (name, _) in enumerate(dims) if name}
                    self.dim_labels[var] = {}
                    constraints['dim'] = dims
                    if dims:
                        shape = []
                        for _, size_spec in dims:
                            if isinstance(size_spec, tuple):
                                start, end = size_spec
                                size = end - start + 1
                            elif size_spec is None:
                                size = 1
                            else:
                                size = size_spec
                            shape.append(size)
                        pa_type = pa.float64() if d_type in ('number', 'array') else pa.string()
                        self.current_scope().define(var, self.array_handler.create_array(shape, 0 if d_type in ('number', 'array')
                                                                                         else '', pa_type, line_number), effective_type, constraints, is_uninitialized=bool(v_expr))
            if v_expr:
                deps = set(re.findall(r'\b[\w_]+\b', v_expr))
                if var in deps:
                    raise ValueError(
                        f"Self-referential assignment '{var} = {v_expr}' at line {line_number}")
                # Evaluate simple literals immediately if no dependencies
                if not deps and (v_expr.startswith('"') and v_expr.endswith('"')):
                    try:
                        evaluated_value = self.expr_evaluator.eval_expr(
                            v_expr, self.current_scope().get_evaluation_scope(), line_number)
                        self.current_scope().define(var, evaluated_value, effective_type,
                                                    constraints, is_uninitialized=False)
                    except Exception as e:
                        self.pending_assignments[var] = (
                            v_expr, line_number, deps)
                else:
                    self.pending_assignments[var] = (v_expr, line_number, deps)
            return
        m_new = re.match(
            r'^([\w_]+)\s*=\s*new\s+(\w+)\s*\{([^}]*)\}$', a, re.I)
        if m_new:
            var, type_name, values_str = m_new.groups()
            if type_name.lower() not in self.types_defined:
                raise SyntaxError(
                    f"Type '{type_name}' not defined at line {line_number}")
            values = [v.strip() for v in values_str.split(',') if v.strip()]
            type_fields = self.types_defined[type_name.lower()]
            if len(values) != len(type_fields):
                raise ValueError(
                    f"Expected {len(type_fields)} values for type '{type_name}', got {len(values)} at line {line_number}")
            value_dict = {}
            all_literals = all(re.match(r'^-?\d*\.?\d+$|^".*"$', v)
                               for v in values)
            for (field_name, _), value in zip(type_fields.items(), values):
                val = self.expr_evaluator.eval_expr(
                    value, self.current_scope().get_full_scope(), line_number)
                value_dict[field_name] = val
            self.current_scope().define(var, value_dict, type_name)
            if not all_literals:
                deps = set(re.findall(r'\b[\w_]+\b', values_str))
                if var in deps:
                    raise ValueError(
                        f"Self-referential assignment '{var} = new {type_name}{{{values_str}}}' at line {line_number}")
                self.pending_assignments[var] = (
                    f"new {type_name}{{{values_str}}}", line_number, deps)
            return
        p = a.split('=', 1)
        if len(p) == 2:
            var_def, expr = map(str.strip, p)
            var, type_name, constraints, value = self._parse_variable_def(
                var_def, line_number)
            if value is not None:
                constraints['constant'] = expr
            self.current_scope().types.setdefault(var, type_name or 'unknown')
            self.current_scope().define(var, value, type_name or 'unknown',
                                        constraints, is_uninitialized=bool(expr and not value))
            if expr:
                deps = set(re.findall(r'\b[\w_]+\b', expr))
                if var in deps:
                    raise ValueError(
                        f"Self-referential assignment '{var} = {expr}' at line {line_number}")
                self.pending_assignments[var] = (expr, line_number, deps)
            return
        raise SyntaxError(
            f"Invalid global definition syntax: {line} at line {line_number}")

    def _parse_dim_size(self, size_str, line_number=None):
        """
        Parse dimension size, handling both standard and grid_dim formats.
        :param size_str: Dimension string.
        :param line_number: Line number for errors.
        :param grid_dim: If True, expect grid_dim format (optional label, fixed int size).
        :return: For grid_dim: (label, int) or (None, int); else: (label, size_spec) or size_spec where size_spec is int/(start,end)/None.
        """
        size_str = size_str.strip()
        label = None
        if ':' in size_str:
            label, size_str = [s.strip() for s in size_str.split(':', 1)]
        size_str = size_str.strip()

        if size_str == '*':
            size_spec = None
        elif re.match(r'^\d+\s+to\s+\d+$', size_str, re.I):
            m = re.match(r'^(\d+)\s+to\s+(\d+)$', size_str, re.I)
            start, end = map(int, m.groups())
            if start > end:
                raise SyntaxError(
                    f"Invalid range {start} to {end} at line {line_number}")
            size_spec = (start, end)
        else:
            try:
                size_spec = int(size_str)
            except ValueError:
                raise SyntaxError(
                    f"Invalid dimension size: '{size_str}' at line {line_number}")

        if label:
            return (label, size_spec)
        return size_spec

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
            shape = self.array_handler.get_array_shape(array, line_number)
            expected_size = shape[dim_idx]
            labels = [lbl.strip().strip('"')
                      for lbl in labels_str.split(',') if lbl.strip()]
            if len(labels) != expected_size:
                raise ValueError(
                    f"Number of labels ({len(labels)}) does not match dimension size ({expected_size}) at line {line_number}")
            self.array_handler.set_labels(
                var_name, dim_name, labels, line_number)
        else:
            raise SyntaxError(
                f"Invalid label assignment syntax: {line} at line {line_number}")

    def _evaluate_cell_var_definition(self, line, line_number=None):
        m = re.match(
            r'^\[([A-Z]+\d+)\]\s*:\s*([\w_]+)\s*=\s*(.+)$', line, re.S)
        if not m:
            return
        cell, var, expr = map(str.strip, m.groups())
        if not re.match(r'^[A-Z]+\d+$', cell):
            raise ValueError(
                f"Invalid cell reference '{cell}' at line {line_number}")
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
        value = self.expr_evaluator.eval_or_eval_array(
            expr, self.current_scope().get_full_scope(), line_number)
        value = self.array_handler.check_dimension_constraints(
            var, value, line_number)
        self.current_scope().update(var, value, line_number)
        self.grid[cell] = value.to_pylist() if isinstance(
            value, pa.Array) else value
        self._cell_var_map[cell] = var

    def run_tests_independent(self, tests):
        tests_data = [
            ("Test 1: Basic calculation", "[A1] : a = 51", {'A1': 51}),
            ("Test 2: Arithmetic with reference",
             "[A1] := 51\n[A2] := ([A1] + 15) / 1.2e3\n[A3] := sum[A1:A2]", {'A1': 51, 'A2': 0.055, 'A3': 51.055}),
            ("Test 3: Variable in cell definition",
             "[A1] : a = 51\n[A2] : b = (a + 15) / 1.2e3\n[A3] : c = sum([A1:A2])", {'A1': 51, 'A2': 0.055, 'A3': 51.055}),
            ("Test 4: Global variables and sum{}",
             ": a = 51\n: b = (a + 15) / 1.2e3\n: c = sum{a, b}\n[A1] := a\n[A2] := b\n[A3] := c", {'A1': 51, 'A2': 0.055, 'A3': 51.055}),
            ("Test 5: Inline comment parsing",
             "[AB2] := 33 '28th column", {'AB2': 33}),
            ("Test 6: Horizontal array assignment from inline var",
             "[A1] : a = 51\n[@B2] := {12, (15 + a) / 1.2e3, 1+[A1], 8}", {'A1': 51, 'B2': 12, 'C2': 0.055, 'D2': 52, 'E2': 8}),
            ("Test 7: Horizontal array from variable", "[A1] : a = 51\n: b = {12, (15 + a) / 1.2e3, 1+[A1], 8}\n[@B2] := b", {
             'A1': 51, 'B2': 12, 'C2': 0.055, 'D2': 52, 'E2': 8}),
            ("Test 8: Math function - SQRT",
             "[A1] := 2*( SQRT(100) - 7 )", {'A1': 6.0}),
            ("Test 9: Interpolation with cell reference",
             ": n = 2\n[A{ n }] := 33\n[A{n + 1}] := -4.9", {'A2': 33, 'A3': -4.9}),
            ("Test 10: 2D range assign and slice", "[A1:C2] := {1, 2, 3; 10, 11, 12}\n[@A3] := [A1:C1]", {
             'A1': 1, 'A2': 10, 'A3': 1, 'B1': 2, 'B2': 11, 'B3': 2, 'C1': 3, 'C2': 12, 'C3': 3}),
            ("Test 11: Range read and vertical assignment",
             "[@A1] := {1; 2; 3}\n[B1:B2] := [A2:A3]", {'A1': 1, 'A2': 2, 'A3': 3, 'B1': 2, 'B2': 3}),
            ("Test 12: Text type assignment",
             ": t as text = \"Hello\"\n[A1] := t", {'A1': 'Hello'}),
            ("Test 13: Cell var definition", "[A1] : e = 3", {'A1': 3}),
            ("Test 14: Number type declaration",
             ": x as number = 42\n[A1] := x", {'A1': 42}),
            ("Test 15: Number declaration with scientific notation",
             ": num as number = 12.34e-5\n[A1] := num", {'A1': 0.0001234}),
            ("Test 16: Number sequence with custom step",
             ": seq = 1 to 6 step 1.5\n[@B1] := seq", {'B1': 1.0, 'C1': 2.5, 'D1': 4.0, 'E1': 5.5}),
            ("Test 17: Interpolated text", ": name as text = \"world\"\n[A1] := $\"Hello, {name}!\"", {
             'A1': "Hello, world!"}),
            ("Test 18: Self-referential assignment (should fail)", ": x = x", None),
            ("Test 19: Variable before definition",
             ": y = x + 1\n: x as number = 5\n[A1] := y", {'A1': 6}),
            ("Test 20: Integer division", "[A1] := 10 \\ 3", {'A1': 3}),
            ("Test 21: Exponentiation", "[A1] := 2 ^ 3", {'A1': 8}),
            ("Test 22: Modulus", "[A1] := 10 mod 3", {'A1': 1}),
            ("Test 23: Special value #INF",
             "[A1] := #INF", {'A1': float('inf')}),
            ("Test 24: Special value -#INF",
             "[A1] := -#INF", {'A1': float('-inf')}),
            ("Test 25: Special value #N/A",
             "[A1] := #N/A", {'A1': float('nan')}),
            ("Test 26: Text concatenation", ": t1 as text = \"Hello\"\n: t2 as text = \"world\"\n[A1] := t1 & \", \" & t2", {
             'A1': "Hello, world"}),
            ("Test 27: Quote escaping", ": q as text = \"I say \"\"Hello\"\"\"\n[A1] := q", {
             'A1': "I say \"Hello\""}),
            ("Test 28: Padding right", ": num = 42\n[A1] := $\"Number: {num, 5}\"", {
             'A1': "Number:    42"}),
            ("Test 29: Padding left",
             ": num = 42\n[A1] := $\"Number: {num, -5}\"", {'A1': "Number: 42   "}),
            ("Test 30: Multi-line interpolation",
             ": name = \"world\"\n[A1] := $\"{*Loudly} I say:  \n\t\t\"\"Hello\"\", {name}!\"", {'A1': "{Loudly} I say:\n\"Hello\", world!"}),
            ("Test 31: Empty braces",
             "[A1] := $\"Empty: {}\"", {'A1': "Empty: {}"}),
            ("Test 32: Escaped brace",
             "[A1] := $\"Escaped: {{{*star\"", {'A1': "Escaped: {{{star"}),
            ("Test 33: Escaped brace with closing brace",
             "[A1] := $\"Escaped: {{{*star}\"", {'A1': "Escaped: {{{star}"}),
            ("Test 34: Number sequence default step", ": dice = 1 to 6\n[@A1] := dice", {
             'A1': 1, 'B1': 2, 'C1': 3, 'D1': 4, 'E1': 5, 'F1': 6}),
            ("Test 35: Point type spilling on grid",
             "Define Point as Type\n: x as number\n: y as number\nEnd Point\n: p = new Point{-4.3, 2.1}\n[@A3] := p", {'A3': -4.3, 'B3': 2.1}),
            ("Test 36: Array of Points spilling on grid",
             "Define Point as Type\n: x as number\n: y as number\nEnd Point\n: p1 = new Point{1, 2}\n: p2 = new Point{3, 4}\n[@C3] := {p1, p2}", {'C3': 1, 'C4': 3, 'D3': 2, 'D4': 4}),
            ("Test 37: Nested Rectangle type spilling on grid",
             "Define Point as Type\n: x as number\n: y as number\nEnd Point\nDefine Rectangle as Type\n: top as Point\n: bottom as Point\nEnd Rectangle\n[@A1] := new Rectangle{new Point{0, 10}, new Point{5, 3}}", {'A1': 0, 'B1': 10, 'C1': 5, 'D1': 3}),
            ("Test 38: Point type assigned to single cell",
             "Define Point as Type\n\t: x as number\n\t: y as number\nEnd Point\n: p = new Point{-4.3, 2.1}\n[A1] := p", {'A1': {'x': -4.3, 'y': 2.1}}),
            ("Test 39: Access field of point type",
             "Define Point as Type\n: x as number\n: y as number\nEnd Point\n: p = new Point{-4.3, 2.1}\n[A3] := p.y", {'A3': 2.1}),
            ("Test 40: Concatenation with interpolation",
             ": m = \"hello\"\n[A1] := \"I say-\" & $\" {m} world!\"", {'A1': "I say- hello world!"}),
            ("Test 41: Nested {* interpolation",
             "[A1] := $\"{{*\"", {'A1': "{{"}),
            ("Test 42: Dimension constraint",
             ": Weights as number dim {0 to 10, 0 to 10} = 0\n[A1] := Weights{0, 0}", {'A1': 0}),
            ("Test 43: Multi-dimensional addition",
             "[A1] := {0, 3; 10, 11} + {1, 2; 3, 4}", {'A1': [[1.0, 5.0], [13.0, 15.0]]}),
            ("Test 44: Multi-dimensional subtraction",
             "[A1] := {0, 3; 10, 11} - {1, 2; 3, 4}", {'A1': [[-1.0, 1.0], [7.0, 7.0]]}),
            ("Test 45: Multi-dimensional multiplication",
             "[A1] := {0, 3; 10, 11} * {1, 2; 3, 4}", {'A1': [[0.0, 6.0], [30.0, 44.0]]}),
            ("Test 46: Multi-dimensional division", "[A1] := {0, 3; 10, 11} / {1, 2; 3, 4}", {
             'A1': [[0.0, 1.5], [3.3333333333333335, 2.75]]}),
            ("Test 47: Multi-dimensional exponentiation",
             "[A1] := {0, 3; 10, 11} ^ {1, 2; 3, 4}", {'A1': [[0.0, 9.0], [1000.0, 14641.0]]}),
            ("Test 48: Multi-dimensional modulo",
             "[A1] := {0, 3; 10, 11} mod {1, 2; 3, 4}", {'A1': [[0.0, 1.0], [1.0, 3.0]]}),
            ("Test 49: Multi-dimensional integer division",
             "[A1] := {0, 3; 10, 11} \\ {1, 2; 3, 4}", {'A1': [[0.0, 1.0], [3.0, 2.0]]}),
            ("Test 50: Pipe operator", "[@A1] := {1, 2} | {3, 4} | {-5, -6}", {
             'A1': 1.0, 'B1': 2.0, 'A2': 3.0, 'B2': 4.0, 'A3': -5.0, 'B3': -6.0}),
            ("Test 51: Dim reshape",
             "[A1] := {10, 11} dim {*, 1}", {'A1': [[10], [11]]}),
            ("Test 52: Range with 1D vertical",
             "[B2:B4] := {9, 8, 7}", {'B2': 9, 'B3': 8, 'B4': 7}),
            ("Test 53: Range with 1D repeated", "[A2:B4] := {1, 2}", {
             'A2': 1, 'A3': 1, 'A4': 1, 'B2': 2, 'B3': 2, 'B4': 2}),
            ("Test 54: Named dimensions",
             ": Results as number dim {Dept: *, Quarter: 4} = {9, 4, 5, 1}\nResults!Quarter.Label{\"Q1\", \"Q2\", \"Q3\", \"Q4\"}\n[A1] := Results!Quarter(\"Q2\")", {'A1': 4}),
            ("Test 55: Assign and access dimensioned array",
             "[A1:D1] := {10, 20, 30, 40}\n: Results as number dim {Dept: *, Quarter: 4} = [A1:D1]\nResults!Quarter.Label{\"Q1\", \"Q2\", \"Q3\", \"Q4\"}\n[B2] := Results!Quarter(\"Q2\")", {'A1': 10, 'B1': 20, 'B2': 20, 'C1': 30, 'D1': 40}),
            ("Test 56: Multi-dimensional addressing", "[A2:D2] := {10, 20, 30, 40}\n: Results as number dim {Dept: *, Quarter: 4} = [A2:D2]\n[A1] := Results[B1]", {
             'A1': 20, 'A2': 10, 'B2': 20, 'C2': 30, 'D2': 40}),
            ("Test 57: FOR with already defined variable",
             ": x = 34\nFOR x AS NUMBER", None),
            ("Test 58: FOR with LET defining local variable",
             "For x = 34\nlet x as number\n[A1] := x", {'A1': 34}),
            ("Test 59: FOR block with LET defining local variable",
             "For x = 34 DO\n    Let x as number Then\n        [A1] := x\n    end\nend", {'A1': 34}),
            ("Test 60: LET block with FOR using already declared variable",
             "Let x as number Then\n    For x = 34 DO\n        [A1] := x\n    end\nend", None),
            ("Test 61: LET followed by FOR with local variable",
             "Let x as number\n[A1] := x\nFor x = 34", {'A1': 34}),
            ("Test 62: LET followed by global variable declaration",
             "Let x as number\n[A1] := x\n: x = 34", {'A1': 34}),
            ("Test 63: LET with constraint x > 10",
             ": x = 34\nLet x > 10 then\n    [A1] := x\nend", {'A1': 34}),
            ("Test 64: LET with constraint x < 10",
             ": x = 34\nLet x < 10 then\n    [A1] := x\nend", {}),
            ("Test 65: LET with constraint x < 10 halting execution",
             ": x = 34\nLet x < 10\n[A1] := x", {}),
            ("Test 66: LET chain with dependencies X -> Y -> Z",
             "Let X = 2 and Y = X * 5 and Z = Y - X\n[A1] := Z", {'A1': 8}),
            ("Test 67: LET chain with wrong order (Y uses X before defined)",
             "Let Y = X * 5 and X = 2\n[A1] := Y", None),
            ("Test 68: LET with array and named dimension", "For names dim 0 to 4\nLet names = {\"Alice\", \"Bob\", \"Carla\", \"Dylan\", \"Edith\"}\n[@A1] := names", {
             'A1': 'Alice', 'B1': 'Bob', 'C1': 'Carla', 'D1': 'Dylan', 'E1': 'Edith'}),
            ("Test 69: Global assignment used in FOR",
             ": n = (m + 10) / 10\n[A1] := n\nFor m = 32", {'A1': 4.2}),
            ("Test 70: LET with array access (index and label)",
             "For names dim 0 to 4\nLet names = {\"Alice\", \"Bob\", \"Carla\", \"Dylan\", \"Edith\"}\n[A1] := names(0)\n[B1] := names[3]", {'A1': 'Alice', 'B1': 'Carla'}),
            ("Test 71: Define custom type and array of objects",
             "define Tensor as type\n    : name as text\nend Tensor\n\nFor V as tensor with (name = \"V\", grid DIM {4, 4, 2} = 1.0) \n[A1] := V.grid{4, 4, 1}\n[B1] := V.name", {'A1': 1.0, 'B1': 'V'}),
            ("Test 72: Define custom type and array with no constraint for {4, 4, 2}",
             "define Tensor as type\n    : name as text\nend Tensor\n\nFor V as tensor with (name = \"V\", grid DIM {4, 4, 2} = {1,1,1,1;2,2,2,2;3,3,3,3;4,4,4,4} | {1,2,3,4;2,3,4,5;3,4,5,6;4,5,6,7})\n[A1] := V.grid{4, 4, 1}\n[B1] := V.grid{4, 4, 2}\n[C1] := V.name", {'A1': 4.0, 'B1': 7.0, 'C1': 'V'}),
            ("Test 73: Define custom type and array with variable assignment for {4, 4, 3}",
             "define Tensor as type\n    : name as text\nend Tensor\n\nFor var = {1,1,1,1;2,2,2,2;3,3,3,3;4,4,4,4} | {1,2,3,4;2,3,4,5;3,4,5,6;4,5,6,7} | {11,2,3,4;2,3,4,5;3,4,5,6;4,5,6,7}\nFor V as tensor with (name = \"V\", grid DIM {4, 4, 3} = var)\n[A1] := V.grid{3, 1, 2}\n[B1] := V.grid{4, 4, 2}\n[C1] := V.name\n[D1] := V.grid{1, 1, 3}", {'A1': 3.0, 'B1': 7.0, 'C1': 'V', 'D1': 11.0})
        ]
        passed = 0
        failed = 0
        total = 0
        failed_tests = []
        for name, code, expected in tests_data[:73]:
            if not tests or name.split(":")[0][5:] in tests:
                total += 1
                try:
                    result = self.run(code)
                    sorted_res = dict(sorted(result.items()))
                    if expected is None:
                        failed += 1
                        failed_tests.append(name)
                    else:
                        sorted_exp = dict(sorted(expected.items()))
                        match = True
                        if set(sorted_res.keys()) != set(sorted_exp.keys()):
                            match = False
                        else:
                            for k in sorted_exp:
                                a_val = sorted_res[k]
                                e_val = sorted_exp[k]
                                if isinstance(e_val, float) and math.isnan(e_val):
                                    if not (isinstance(a_val, float) and math.isnan(a_val)):
                                        match = False
                                elif isinstance(e_val, float) and isinstance(a_val, float):
                                    if not math.isclose(a_val, e_val, rel_tol=1e-9):
                                        match = False
                                elif isinstance(e_val, int) and isinstance(a_val, float) and a_val.is_integer():
                                    a_val = int(a_val)
                                    if a_val != e_val:
                                        match = False
                                elif isinstance(e_val, list) and isinstance(a_val, list):
                                    if not all(isinstance(a, list) and isinstance(e, list) and all(math.isclose(av, ev, rel_tol=1e-9) if isinstance(ev, float) else av == ev for av, ev in zip(a, e)) for a, e in zip(a_val, e_val)):
                                        match = False
                                elif isinstance(e_val, dict) and isinstance(a_val, dict):
                                    if not all(a_val.get(kk) == vv for kk, vv in e_val.items()):
                                        match = False
                                elif a_val != e_val:
                                    match = False
                        if match:
                            passed += 1
                        else:
                            failed += 1
                            failed_tests.append(name)
                except Exception as e:
                    if expected is None:
                        passed += 1
                    else:
                        failed += 1
                        failed_tests.append(name)
        print(
            f"Test summary - Total: {total}, Passed: {passed}, Failed: {failed}")
        if failed_tests:
            print("DEBUG: Failed tests:")
            for test_name in failed_tests:
                print(f"  - {test_name}")
