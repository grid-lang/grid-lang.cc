"""
Scope management for GridLang compiler.
Handles variable scoping, constraints, and pipe connections.
"""


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
        # New Grid language features
        # Maintain definition order for inputs/outputs (args rely on this)
        # Variables that can only receive values (case-insensitive, ordered)
        self.input_variables = []
        self.output_variables = set()  # Variables that can only push values
        self.pipe_connections = {}  # Maps outputs to connected inputs
        self.implicit_let = set()

    def _get_case_insensitive_key(self, name, dictionary):
        """Get a key from dictionary in a case-insensitive manner"""
        name_lower = name.lower()
        for key in dictionary:
            if key.lower() == name_lower:
                return key
        return None

    def define(self, name, value=None, type=None, constraints=None, is_uninitialized=False, line_number=None):
        # Check for case-insensitive conflicts
        existing_key = self._get_case_insensitive_key(name, self.variables)
        if existing_key and not is_uninitialized:
            raise ValueError(
                f"Variable '{name}' conflicts with existing variable '{existing_key}' in this scope")
        if value is not None and type and hasattr(self, 'compiler') and hasattr(self.compiler, 'types_defined'):
            if isinstance(value, list) and type.lower() in self.compiler.types_defined:
                value = self.compiler._convert_array_to_object(
                    type, value, line_number)
        if value is not None and not is_uninitialized:
            if constraints and constraints.get('dim') and hasattr(self, 'compiler'):
                value = self.compiler.array_handler.check_dimension_constraints(
                    name, value, line_number)
            self._check_constraints(name, value, line_number)
        self.variables[name] = value
        self.types[name] = type
        self.constraints[name] = constraints or {}
        if is_uninitialized:
            self.uninitialized.add(name)
        else:
            self.uninitialized.discard(name)
        if hasattr(self.compiler, 'mark_dependency_resolved'):
            self.compiler.mark_dependency_resolved(name)

    def update(self, name, value, line_number=None):
        defining_scope = self.get_defining_scope(name)
        if defining_scope:
            # Get the actual key for case-insensitive update
            actual_key = defining_scope._get_case_insensitive_key(
                name, defining_scope.variables)
            if actual_key:
                var_type = defining_scope.types.get(actual_key)
                if value is not None and var_type and hasattr(self, 'compiler') and hasattr(self.compiler, 'types_defined'):
                    if isinstance(value, list) and var_type.lower() in self.compiler.types_defined:
                        # If this variable is an array of custom types, keep the list.
                        if not defining_scope.constraints.get(actual_key, {}).get('dim'):
                            value = self.compiler._convert_array_to_object(
                                var_type, value, line_number)
                # Prevent updating input variables once initialized
                if defining_scope.constraints.get(actual_key, {}).get('input') and actual_key not in defining_scope.uninitialized:
                    raise ValueError(
                        f"Input variable '{actual_key}' cannot be updated at line {line_number}")
                if defining_scope.constraints.get(actual_key, {}).get('dim'):
                    value = self.compiler.array_handler.check_dimension_constraints(
                        actual_key, value, line_number)
                defining_scope._check_constraints(
                    actual_key, value, line_number)
                defining_scope.variables[actual_key] = value
                defining_scope.uninitialized.discard(actual_key)

                # Re-evaluate constraint expressions that depend on this variable
                self._re_evaluate_constraints(actual_key, line_number)
                if hasattr(self.compiler, 'mark_dependency_resolved'):
                    self.compiler.mark_dependency_resolved(actual_key)
                if hasattr(self.compiler, '_sync_cell_bindings'):
                    self.compiler._sync_cell_bindings(actual_key, value)
                if hasattr(self.compiler, '_record_output_value'):
                    self.compiler._record_output_value(actual_key, value)
            else:
                # Variable exists in types or constraints but not variables
                defining_scope._check_constraints(name, value, line_number)
                defining_scope.variables[name] = value
                defining_scope.uninitialized.discard(name)

                # Re-evaluate constraint expressions that depend on this variable
                self._re_evaluate_constraints(name, line_number)
                if hasattr(self.compiler, 'mark_dependency_resolved'):
                    self.compiler.mark_dependency_resolved(name)
                if hasattr(self.compiler, '_sync_cell_bindings'):
                    self.compiler._sync_cell_bindings(name, value)
                if hasattr(self.compiler, '_record_output_value'):
                    self.compiler._record_output_value(name, value)
        else:
            if self.is_shadowed(name) and not self.is_private:
                print(
                    f"Warning: '{name}' shadows a variable in an outer scope at line {line_number}")
            self.define(name, value)

    def get(self, name):
        # Case-insensitive lookup
        actual_key = self._get_case_insensitive_key(name, self.variables)
        if actual_key:
            value = self.variables[actual_key]
            # Lazily apply INIT defaults when the variable is first read
            if value is None:
                init_expr = self.constraints.get(actual_key, {}).get('init')
                if init_expr is not None and hasattr(self, 'compiler'):
                    try:
                        value = self.compiler.expr_evaluator.eval_or_eval_array(
                            str(init_expr), self.get_full_scope())
                        self.variables[actual_key] = value
                        self.uninitialized.discard(actual_key)
                    except Exception:
                        pass
            return value
        if self.parent and (not self.is_private or getattr(self, 'is_loop_scope', False)):
            return self.parent.get(name)
        raise NameError(f"Variable '{name}' not defined")

    def is_uninitialized(self, name):
        # Case-insensitive lookup
        actual_key = self._get_case_insensitive_key(name, self.uninitialized)
        if actual_key:
            return True
        # If the variable is defined in this scope (even if a parent has it),
        # treat it as initialized here.
        if (self._get_case_insensitive_key(name, self.variables) or
                self._get_case_insensitive_key(name, self.types) or
                self._get_case_insensitive_key(name, self.constraints)):
            return False
        if self.parent and (not self.is_private or getattr(self, 'is_loop_scope', False)):
            return self.parent.is_uninitialized(name)
        return False

    def get_defining_scope(self, var):
        current = self
        while current:
            # Case-insensitive lookup
            var_key = current._get_case_insensitive_key(var, current.variables)
            type_key = current._get_case_insensitive_key(var, current.types)
            constraint_key = current._get_case_insensitive_key(
                var, current.constraints)
            if (var_key or type_key or constraint_key):
                return current
            current = current.parent
        return None

    def define_input(self, name, type_name=None, default_value=None, line_number=None, extra_constraints=None):
        """Define an input variable that can only receive values through pipes"""
        name_lower = name.lower()
        if name_lower not in self.input_variables:
            self.input_variables.append(name_lower)
        constraints = {'input': True}
        if type_name:
            constraints['type'] = type_name.lower()
        if default_value is not None:
            constraints['default'] = default_value
        if extra_constraints:
            constraints.update(extra_constraints)
        # Always start uninitialized; defaults are applied during argument processing
        self.define(name, None, type_name, constraints, is_uninitialized=True)

    def define_output(self, name, type_name=None, line_number=None, constraints=None):
        """Define an output variable that can only push values through pipes"""
        self.output_variables.add(name.lower())
        constraints = constraints or {}
        constraints.setdefault('output', True)
        self.define(name, None, type_name, constraints, is_uninitialized=True)

    def is_input(self, name):
        """Check if a variable is an input variable"""
        name_lower = name.lower()
        if name_lower in self.input_variables:
            return True
        if self.parent and not self.is_private:
            return self.parent.is_input(name)
        return False

    def is_output(self, name):
        """Check if a variable is an output variable"""
        name_lower = name.lower()
        if name_lower in self.output_variables:
            return True
        if self.parent and not self.is_private:
            return self.parent.is_output(name)
        return False

    def connect_pipe(self, output_name, input_name, line_number=None):
        """Connect an output to an input through a pipe"""
        if output_name not in self.pipe_connections:
            self.pipe_connections[output_name] = []
        self.pipe_connections[output_name].append(input_name)

    def mark_implicit_let(self, name):
        self.implicit_let.add(name.lower())

    def is_implicit_let(self, name):
        return name.lower() in self.implicit_let

    def clear_implicit_let(self, name):
        self.implicit_let.discard(name.lower())

    def get_connected_inputs(self, output_name):
        """Get all inputs connected to a given output"""
        return self.pipe_connections.get(output_name, [])

    def push_value(self, output_name, value, line_number=None, _visited_outputs=None):
        """Push a value through an output to all connected inputs"""
        if not self.is_output(output_name):
            raise ValueError(
                f"'{output_name}' is not an output variable at line {line_number}")

        connected_inputs = self.get_connected_inputs(output_name)
        if not connected_inputs:
            return

        # Propagate value to all connected inputs
        for input_name in connected_inputs:
            try:
                self.update(input_name, value, line_number)
            except Exception as e:
                pass

        # Trigger wave propagation if any connected inputs have their own outputs
        if _visited_outputs is None:
            _visited_outputs = set()
        _visited_outputs.add(output_name.lower())
        self._propagate_wave(connected_inputs, line_number, _visited_outputs)

    def _propagate_wave(self, updated_variables, line_number, _visited_outputs=None):
        """Propagate value updates through the network (wave)"""
        for var_name in updated_variables:
            # Check if this variable has outputs connected to it
            for output_name, connected_inputs in self.pipe_connections.items():
                if var_name in connected_inputs:
                    # This variable is connected to an output, propagate the wave
                    var_value = self.get(var_name)
                    # Re-entrancy guard to avoid infinite loops
                    if _visited_outputs and output_name.lower() in _visited_outputs:
                        continue
                    self.push_value(output_name, var_value,
                                    line_number, _visited_outputs)

    def is_shadowed(self, name):
        current = self.parent
        while current:
            if current._get_case_insensitive_key(name, current.variables):
                return True
            current = current.parent
        return False

    def get_evaluation_scope(self):
        full_scope = {}
        current = self

        # Add variables with case-insensitive mappings
        for var_name, var_value in current.variables.items():
            full_scope[var_name] = var_value
            # Add lowercase version for case-insensitive access
            full_scope[var_name.lower()] = var_value
            # Add uppercase version for case-insensitive access
            full_scope[var_name.upper()] = var_value

        current = current.parent
        while current:
            # Include variables from all parent scopes, including private ones
            # This is necessary for nested FOR loops where outer loop variables
            # need to be accessible to inner loops
            for var_name, var_value in current.variables.items():
                # Only add if not already present (to avoid overriding local variables)
                if var_name not in full_scope:
                    full_scope[var_name] = var_value
                    # Add case-insensitive versions only if not already present
                    if var_name.lower() not in full_scope:
                        full_scope[var_name.lower()] = var_value
                    if var_name.upper() not in full_scope:
                        full_scope[var_name.upper()] = var_value
            current = current.parent
        return full_scope

    def _re_evaluate_constraints(self, changed_var, line_number=None):
        """Re-evaluate constraint expressions that depend on the changed variable"""

        # Find all variables that have constraint expressions depending on changed_var
        for var_name, constraints in self.constraints.items():
            for constraint_type, constraint_expr in constraints.items():
                if constraint_type == 'constant' and isinstance(constraint_expr, str):
                    # Check if this constraint expression depends on the changed variable
                    if self._expression_depends_on(constraint_expr, changed_var):
                        try:
                            # Re-evaluate the constraint expression
                            new_value = self.compiler.expr_evaluator.eval_or_eval_array(
                                constraint_expr, self.get_full_scope(), line_number)

                            # Update the variable with the new value
                            self.variables[var_name] = new_value
                            self.uninitialized.discard(var_name)

                            # Recursively re-evaluate constraints that depend on this variable
                            self._re_evaluate_constraints(
                                var_name, line_number)

                        except Exception as e:
                            pass

    def _expression_depends_on(self, expr, var_name):
        """Check if an expression depends on a specific variable"""
        # Simple dependency check - look for the variable name in the expression
        # This is a basic implementation; could be enhanced with proper parsing
        import re
        expr_text = expr
        if isinstance(expr, str) and ('$"' in expr or "$'" in expr):
            # Keep only interpolation segments for dependency detection.
            brace_parts = re.findall(r'\{([^{}]*)\}', expr)
            expr_text = ' '.join(brace_parts)
        else:
            # Strip quoted strings to avoid false positives from literals.
            expr_text = re.sub(
                r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', ' ', expr)
        # Create a pattern that matches the variable name as a whole word
        pattern = r'\b' + re.escape(var_name) + r'\b'
        return bool(re.search(pattern, expr_text))

    def _check_constraints(self, name, value, line_number=None):
        # Case-insensitive constraint lookup
        actual_key = self._get_case_insensitive_key(name, self.constraints)
        key_for_constraints = actual_key if actual_key is not None else name
        constraints = self.constraints.get(key_for_constraints, {})
        for constraint_type, constraint_expr in constraints.items():
            if constraint_type == 'constant':
                if isinstance(constraint_expr, str):
                    try:
                        constraint_val = self.compiler.expr_evaluator.eval_or_eval_array(
                            constraint_expr, self.get_full_scope(), line_number)
                    except Exception:
                        # Skip constant validation if the expression can't be resolved in this scope.
                        continue
                else:
                    constraint_val = constraint_expr
                if constraints.get('dim'):
                    try:
                        constraint_val = self.compiler.array_handler.check_dimension_constraints(
                            key_for_constraints, constraint_val, line_number)
                    except Exception:
                        pass
                if value != constraint_val:
                    raise ValueError(
                        f"Cannot change constant '{key_for_constraints}' at line {line_number}")
            elif constraint_type in ('<=', '>=', '<', '>'):
                constraint_val = float(self.compiler.expr_evaluator.eval_or_eval_array(
                    constraint_expr, self.get_full_scope(), line_number))
                if constraint_type == '<=' and value > constraint_val:
                    raise ValueError(
                        f"'{key_for_constraints}' exceeds maximum {constraint_val} at line {line_number}")
                elif constraint_type == '>=' and value < constraint_val:
                    raise ValueError(
                        f"'{key_for_constraints}' is below minimum {constraint_val} at line {line_number}")
                elif constraint_type == '<' and value >= constraint_val:
                    raise ValueError(
                        f"'{key_for_constraints}' is not less than {constraint_val} at line {line_number}")
                elif constraint_type == '>' and value <= constraint_val:
                    raise ValueError(
                        f"'{key_for_constraints}' is not greater than {constraint_val} at line {line_number}")
            elif constraint_type == '<>':
                constraint_val = self.compiler.expr_evaluator.eval_or_eval_array(
                    constraint_expr, self.get_full_scope(), line_number)
                if isinstance(value, (list, tuple, set)):
                    if constraint_val in value:
                        raise ValueError(
                            f"'{key_for_constraints}' contains disallowed value {constraint_val} at line {line_number}")
                elif value == constraint_val:
                    raise ValueError(
                        f"'{key_for_constraints}' must not equal {constraint_val} at line {line_number}")
            elif constraint_type.startswith('not_') and constraint_type[4:] in ('<=', '>=', '<', '>'):
                op = constraint_type[4:]
                constraint_val = float(self.compiler.expr_evaluator.eval_or_eval_array(
                    constraint_expr, self.get_full_scope(), line_number))
                if op == '<' and value < constraint_val:
                    raise ValueError(
                        f"'{key_for_constraints}' must not be less than {constraint_val} at line {line_number}")
                elif op == '<=' and value <= constraint_val:
                    raise ValueError(
                        f"'{key_for_constraints}' must be greater than {constraint_val} at line {line_number}")
                elif op == '>' and value > constraint_val:
                    raise ValueError(
                        f"'{key_for_constraints}' must not be greater than {constraint_val} at line {line_number}")
                elif op == '>=' and value >= constraint_val:
                    raise ValueError(
                        f"'{key_for_constraints}' must be less than {constraint_val} at line {line_number}")
            elif constraint_type == 'in':
                if isinstance(value, (list, tuple, set)):
                    if not all(item in constraint_expr for item in value):
                        raise ValueError(
                            f"'{key_for_constraints}' values {value} not in allowed values {constraint_expr} at line {line_number}")
                elif value not in constraint_expr:
                    raise ValueError(
                        f"'{key_for_constraints}' value {value} not in allowed values {constraint_expr} at line {line_number}")
            elif constraint_type == 'range':
                start_expr = constraint_expr.get('start')
                end_expr = constraint_expr.get('end')
                step_expr = constraint_expr.get('step')
                start_val = float(self.compiler.expr_evaluator.eval_or_eval_array(
                    start_expr, self.get_full_scope(), line_number))
                end_val = float(self.compiler.expr_evaluator.eval_or_eval_array(
                    end_expr, self.get_full_scope(), line_number))
                val = float(value)
                if not (start_val <= val <= end_val):
                    raise ValueError(
                        f"'{key_for_constraints}' value {value} not in range {start_val} to {end_val} at line {line_number}")
                if step_expr is not None:
                    step_val = float(self.compiler.expr_evaluator.eval_or_eval_array(
                        step_expr, self.get_full_scope(), line_number))
                    if step_val == 0:
                        raise ValueError(
                            f"'{key_for_constraints}' range step cannot be 0 at line {line_number}")
                    steps = (val - start_val) / step_val
                    if abs(steps - round(steps)) > 1e-9:
                        raise ValueError(
                            f"'{key_for_constraints}' value {value} not aligned to step {step_val} starting at {start_val} at line {line_number}")
                else:
                    if start_val.is_integer() and end_val.is_integer():
                        if not val.is_integer():
                            raise ValueError(
                                f"'{key_for_constraints}' value {value} must be an integer in range {start_val} to {end_val} at line {line_number}")
            elif constraint_type == 'not_null':
                if value is None:
                    raise ValueError(
                        f"'{key_for_constraints}' must not be null at line {line_number}")
                if isinstance(value, str) and value == '':
                    raise ValueError(
                        f"'{key_for_constraints}' must not be empty at line {line_number}")
            elif constraint_type == 'type':
                expected_type = constraint_expr.lower()
                actual_type = self.compiler.array_handler.infer_type(
                    value, line_number)
                if expected_type == 'number' and actual_type not in ('number', 'float64', 'int', 'int64'):
                    raise ValueError(
                        f"'{key_for_constraints}' must be a number, got {actual_type} at line {line_number}")
                elif expected_type == 'text' and actual_type not in ('string', 'text'):
                    raise ValueError(
                        f"'{key_for_constraints}' must be text, got {actual_type} at line {line_number}")
            elif constraint_type == 'type_union':
                actual_type = self.compiler.array_handler.infer_type(
                    value, line_number)
                allowed = set(constraint_expr)
                type_matches = False
                if 'number' in allowed and actual_type in ('number', 'float64', 'int', 'int64'):
                    type_matches = True
                if 'text' in allowed and actual_type in ('string', 'text'):
                    type_matches = True
                if not type_matches:
                    raise ValueError(
                        f"'{key_for_constraints}' must be one of {sorted(allowed)} at line {line_number}")
            elif constraint_type == 'not_type':
                expected_type = constraint_expr.lower()
                actual_type = self.compiler.array_handler.infer_type(
                    value, line_number)
                if expected_type == 'number' and actual_type in ('number', 'float64', 'int', 'int64'):
                    raise ValueError(
                        f"'{key_for_constraints}' must not be a number at line {line_number}")
                elif expected_type == 'text' and actual_type in ('string', 'text'):
                    raise ValueError(
                        f"'{key_for_constraints}' must not be text at line {line_number}")
            elif constraint_type == 'unit':
                pass
            elif constraint_type == 'not_unit':
                if isinstance(value, str) and value == constraint_expr:
                    raise ValueError(
                        f"'{key_for_constraints}' must not be unit {constraint_expr} at line {line_number}")

    def get_full_scope(self):
        full_scope = {}
        current = self
        while current and not current.is_private:
            full_scope.update(current.variables)
            current = current.parent
        return full_scope
