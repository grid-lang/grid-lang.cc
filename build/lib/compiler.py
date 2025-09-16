import re
import math
import pyarrow as pa
from expression import ExpressionEvaluator
from array_handler import ArrayHandler
from utils import col_to_num, num_to_col, split_cell, offset_cell, validate_cell_ref
from scope import Scope
from control_flow import GridLangControlFlow
from type_processor import GridLangTypeProcessor
from parser import GridLangParser


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
        self.control_flow = GridLangControlFlow(self)
        self.type_processor = GridLangTypeProcessor(self)
        self.parser = GridLangParser(self)
        self.handled_assignments = set()
        # Grid language features
        self.input_variables = []  # Ordered list of input variables
        self.output_variables = []  # List of output variables

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

    def run(self, code, args=None):
        """Delegates to the extracted run function."""
        from executor import GridLangExecutor

        extracted = GridLangExecutor()
        # Store reference to compiler for output values access
        extracted.compiler = self
        # Copy all necessary attributes from self to extracted
        for attr in ['grid', 'scopes', 'expr_evaluator', 'array_handler', 'types_defined',
                     'dimensions', 'pending_assignments', 'dim_labels']:
            if hasattr(self, attr):
                setattr(extracted, attr, getattr(self, attr))

        # Copy ALL methods from self to extracted (except run to avoid recursion)
        for method_name in dir(self):
            if callable(getattr(self, method_name)) and method_name != 'run' and not method_name.startswith('__'):
                setattr(extracted, method_name, getattr(self, method_name))

        # Call the extracted run function
        result = extracted.run(code, args)

        # Copy back any changes to attributes
        for attr in ['grid', 'scopes', 'expr_evaluator', 'array_handler', 'types_defined',
                     'dimensions', 'pending_assignments', 'dim_labels']:
            if hasattr(extracted, attr):
                setattr(self, attr, getattr(extracted, attr))

        return result

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
                            value = self.expr_evaluator.eval_or_eval_array(
                                expr, self.current_scope().get_full_scope(), line_number)
                            value = self.array_handler.check_dimension_constraints(
                                var, value, line_number)
                            self.current_scope().update(var, value, line_number)
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
                    value = self.expr_evaluator.eval_or_eval_array(
                        expr, self.current_scope().get_full_scope(), line_number)
                    value = self.array_handler.check_dimension_constraints(
                        var, value, line_number)
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
        self.root_scope = self.current_scope()  # Always set root scope here

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

        return lines, label_lines, dim_lines

    def _parse_type_def(self, lines, line_number=None):
        """Delegate to type processor."""
        return self.type_processor._parse_type_def(lines, line_number)

    def _execute_type_code(self, code_lines, var_name, value_dict, line_number):
        """Delegate to type processor."""
        return self.type_processor._execute_type_code(code_lines, var_name, value_dict, line_number)

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
        for line, line_number in lines:
            if (line.startswith(':') and not line.lower().startswith(("for ", "let "))) or line.lower().startswith(("input ", "output ")):
                self._collect_global_declarations(line, line_number)
        for line, line_number in label_lines:
            self._process_label_assignment(line, line_number)
        for line, line_number in lines:
            if not line.startswith(':') and ':=' not in line and '!' not in line:
                self._evaluate_cell_var_definition(line, line_number)

    def _collect_global_declarations(self, line, line_number=None):
        # Handle INPUT and OUTPUT declarations that don't start with ':'
        if line.lower().startswith(("input ", "output ")):
            a = line.strip()
        else:
            a = line[1:].strip()

        # Handle INPUT declarations
        m_input = re.match(
            r'^input\s+([\w_]+)\s+as\s+(number|text|array)\s*(?:or\s*=\s*(.+))?$', a, re.I | re.S)
        if m_input:
            var, d_type, default_value = m_input.groups()
            var, d_type = var.strip(), d_type.lower()
            effective_type = 'array' if 'dim' in a else d_type
            self.current_scope().define_input(var, effective_type, default_value, line_number)
            return

        # Handle OUTPUT declarations
        m_output = re.match(
            r'^OUTPUT\s+([\w_]+)(?:\s+as\s+(number|text|array))?$', a, re.I | re.S)
        if m_output:
            var, d_type = m_output.groups()
            var = var.strip()
            # Default to text if no type specified
            d_type = d_type.lower() if d_type else 'text'
            effective_type = 'array' if 'dim' in a else d_type
            self.current_scope().define_output(var, effective_type, line_number)
            return

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
            else:
                # For simple type declarations without dimensions, define the variable
                self.current_scope().define(var, None, effective_type,
                                            constraints, is_uninitialized=True)
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

            # Handle empty braces {} - initialize with default values
            if not values and values_str.strip() == '':
                value_dict = {}
                for field_name, field_type in type_fields.items():
                    if field_name == '_executable_code':
                        continue  # Skip executable code field
                    if field_type.lower() == 'number':
                        value_dict[field_name] = 0
                    elif field_type.lower() == 'text':
                        value_dict[field_name] = ""
                    elif field_type.lower() == 'array':
                        value_dict[field_name] = []
                    else:
                        value_dict[field_name] = None

                # Execute executable code if present
                if '_executable_code' in type_fields:
                    self._execute_type_code(
                        type_fields['_executable_code'], var, value_dict, line_number)

                self.current_scope().define(var, value_dict, type_name)
                return
            elif len(values) != len(type_fields):
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
            elif expr:  # Store constraint expression even when value is None
                constraints['constant'] = expr
            self.current_scope().types.setdefault(var, type_name or 'unknown')

            # Try to evaluate simple literals immediately
            evaluated_value = None
            is_uninitialized = True
            if expr:
                # Extract dependencies, but exclude quoted strings and numeric literals
                deps = set()
                # Skip quoted strings when looking for dependencies
                expr_no_quotes = re.sub(r'"[^"]*"', '', expr)
                expr_no_quotes = re.sub(r"'[^']*'", '', expr_no_quotes)

                # Find potential dependencies, but filter out numeric literals and built-in functions
                potential_deps = re.findall(r'\b[\w_]+\b', expr_no_quotes)
                built_in_functions = {
                    'sum', 'rows', 'sqrt', 'min', 'max', 'abs', 'int', 'float', 'str', 'len'}
                deps = set()
                for dep in potential_deps:
                    # Skip if it's a numeric literal (including negative numbers)
                    if not (dep.replace('-', '').replace('.', '').isdigit() or
                            (dep.startswith('-') and dep[1:].replace('.', '').isdigit())):
                        # Skip built-in functions
                        if dep.lower() not in built_in_functions:
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
                        is_uninitialized = False
                    except Exception as e:
                        self.pending_assignments[var] = (
                            expr, line_number, deps)
                else:
                    self.pending_assignments[var] = (expr, line_number, deps)

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
        if not re.match(r'^[A-Za-z]+\d+$', cell):
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

    def set_input_values(self, args):
        """Set input values from command line arguments or prompt for missing ones"""
        global_scope = self.get_global_scope()

        for i, input_var in enumerate(self.input_variables):
            if i < len(args):
                # Convert argument to appropriate type
                value = args[i]
                var_type = global_scope.types.get(input_var, 'text')

                if var_type == 'number':
                    try:
                        value = float(value)
                    except ValueError:
                        print(
                            f"Warning: Could not convert '{value}' to number for input '{input_var}', using default")
                        continue

                global_scope.update(input_var, value)
            else:
                # No more arguments, prompt for input
                self._prompt_for_input(input_var, global_scope)

    def _prompt_for_input(self, input_var, global_scope):
        """Prompt user for input value and set it in the global scope"""
        var_type = global_scope.types.get(input_var, 'text')

        while True:
            try:
                if var_type == 'number':
                    user_input = input(f"{input_var}: ")
                    value = float(user_input)
                else:
                    user_input = input(f"{input_var}: ")
                    value = user_input

                global_scope.update(input_var, value)
                break

            except ValueError:
                print(f"Invalid input. Please enter a valid {var_type}.")
            except KeyboardInterrupt:
                print("\nExiting...")
                import sys
                sys.exit(1)

    def collect_input_output_variables(self):
        """Collect all input and output variables from the current scope"""
        global_scope = self.get_global_scope()
        self.input_variables = list(global_scope.input_variables)
        self.output_variables = list(global_scope.output_variables)

    def export_to_csv(self, filename):
        """Export grid data to CSV file"""
        import csv
        import os

        # Get all variables and their values
        data = {}
        global_scope = self.get_global_scope()

        # Add all variables from the global scope
        for var_name in global_scope.variables:
            value = global_scope.get(var_name)
            if value is not None:
                data[var_name] = value

        # Add grid cell data
        for cell, value in self.grid.items():
            data[cell] = value

        if not data:
            return

        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Variable', 'Value'])
            for var, val in sorted(data.items()):
                writer.writerow([var, val])
