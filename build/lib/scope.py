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
        self.input_variables = set()  # Variables that can only receive values
        self.output_variables = set()  # Variables that can only push values
        self.pipe_connections = {}  # Maps outputs to connected inputs

    def _get_case_insensitive_key(self, name, dictionary):
        """Get a key from dictionary in a case-insensitive manner"""
        name_lower = name.lower()
        for key in dictionary:
            if key.lower() == name_lower:
                return key
        return None

    def define(self, name, value=None, type=None, constraints=None, is_uninitialized=False):
        # Check for case-insensitive conflicts
        existing_key = self._get_case_insensitive_key(name, self.variables)
        if existing_key and not is_uninitialized:
            raise ValueError(
                f"Variable '{name}' conflicts with existing variable '{existing_key}' in this scope")
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
            # Get the actual key for case-insensitive update
            actual_key = defining_scope._get_case_insensitive_key(
                name, defining_scope.variables)
            if actual_key:
                defining_scope._check_constraints(
                    actual_key, value, line_number)
                defining_scope.variables[actual_key] = value
                defining_scope.uninitialized.discard(actual_key)

                # Re-evaluate constraint expressions that depend on this variable
                self._re_evaluate_constraints(actual_key, line_number)
            else:
                # Variable exists in types or constraints but not variables
                defining_scope._check_constraints(name, value, line_number)
                defining_scope.variables[name] = value
                defining_scope.uninitialized.discard(name)

                # Re-evaluate constraint expressions that depend on this variable
                self._re_evaluate_constraints(name, line_number)
        else:
            if self.is_shadowed(name) and not self.is_private:
                pass  # Shadowing warning removed with debug logs
            self.define(name, value)

    def get(self, name):
        # Case-insensitive lookup
        actual_key = self._get_case_insensitive_key(name, self.variables)
        if actual_key:
            return self.variables[actual_key]
        if self.parent and not self.is_private:
            return self.parent.get(name)
        raise NameError(f"Variable '{name}' not defined")

    def is_uninitialized(self, name):
        # Case-insensitive lookup
        actual_key = self._get_case_insensitive_key(name, self.uninitialized)
        if actual_key:
            return True
        if self.parent and not self.is_private:
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

    def define_input(self, name, type_name=None, default_value=None, line_number=None):
        """Define an input variable that can only receive values through pipes"""
        self.input_variables.add(name.lower())
        self.define(name, default_value, type_name, {
                    'input': True}, is_uninitialized=default_value is None)

    def define_output(self, name, type_name=None, line_number=None):
        """Define an output variable that can only push values through pipes"""
        self.output_variables.add(name.lower())
        self.define(name, None, type_name, {
                    'output': True}, is_uninitialized=True)

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
            except Exception:
                # Swallow errors silently since debug logging is removed
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

                        except Exception:
                            # Ignore evaluation errors silently without debug logs
                            pass

    def _expression_depends_on(self, expr, var_name):
        """Check if an expression depends on a specific variable"""
        import re
        pattern = r'\b' + re.escape(var_name) + r'\b'
        return bool(re.search(pattern, expr))

    def _check_constraints(self, name, value, line_number=None):
        # Case-insensitive constraint lookup
        actual_key = self._get_case_insensitive_key(name, self.constraints)
        key_for_constraints = actual_key if actual_key is not None else name
        constraints = self.constraints.get(key_for_constraints, {})
        for constraint_type, constraint_expr in constraints.items():
            if constraint_type == 'constant':
                if isinstance(constraint_expr, str):
                    constraint_val = self.compiler.expr_evaluator.eval_or_eval_array(
                        constraint_expr, self.get_full_scope(), line_number)
                else:
                    constraint_val = constraint_expr
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
            elif constraint_type == 'in':
                if value not in constraint_expr:
                    raise ValueError(
                        f"'{key_for_constraints}' value {value} not in allowed values {constraint_expr} at line {line_number}")
            elif constraint_type == 'type':
                expected_type = constraint_expr.lower()
                actual_type = self.compiler.array_handler.infer_type(
                    value, line_number)
                if expected_type == 'number' and actual_type not in ('number', 'float64'):
                    raise ValueError(
                        f"'{key_for_constraints}' must be a number, got {actual_type} at line {line_number}")
                elif expected_type == 'text' and actual_type != 'string':
                    raise ValueError(
                        f"'{key_for_constraints}' must be text, got {actual_type} at line {line_number}")
            elif constraint_type == 'unit':
                pass

    def get_full_scope(self):
        full_scope = {}
        current = self
        while current and not current.is_private:
            full_scope.update(current.variables)
            current = current.parent
        return full_scope
