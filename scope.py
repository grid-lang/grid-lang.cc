"""
Scope management for GridLang compiler.
Handles variable scoping, constraints, and pipe connections.
"""

import copy
import re

import pyarrow as pa

from units import UNIT_ERROR, UnitValue, strip_units
from utils import num_to_col, split_cell, col_to_num, iter_interpolation_placeholders

# Stack of compiler run contexts: ``run()`` pushes the executing compiler and
# pops it on exit. Used to detect writes that originate from a read-only
# function sub-compiler and target a scope in that compiler's parent chain.
_ACTIVE_RUNNERS = []


class _ListenerGrid(dict):
    """Grid backing store that notifies the owning compiler on cell writes.

    Every ``self.grid[cell] = value`` write funnels through here so that
    client variables listening on grid cells can be recomputed when a
    publisher (push/init) updates one of their dependency cells.
    """

    def __init__(self, owner=None):
        super().__init__()
        self._grid_owner = owner

    def __setitem__(self, key, value):
        super().__setitem__(key, strip_units(value))
        owner = self._grid_owner
        if owner is not None and hasattr(owner, '_notify_cell_changed'):
            owner._notify_cell_changed(key, value)


class GridLiveView(dict):
    """A live view of a grid, keyed by 1-based (row, col) tuples.

    The grid cells are stored in a backing dict keyed by cell references such
    as 'A1'; this view exposes the same contents through ``(row, col)`` keys,
    matching the ``grid{row, col}`` syntax used inside type definitions.

    When a compiler is passed, the backing store is the compiler's current
    grid. Otherwise the view keeps its own private store, so type instances
    can each have an isolated grid. When ``read_only`` is set, writes to the
    grid raise an error (functions may read, but not modify, the parent grid).
    """

    def __init__(self, store=None, compiler=None, read_only=False):
        super().__init__()
        if compiler is not None:
            store = compiler.grid
        self._store = store if store is not None else {}
        self._read_only = read_only

    @staticmethod
    def _cell_ref(key):
        row, col = key
        return f"{num_to_col(int(col))}{int(row)}"

    def _is_cell_key(self, key):
        return isinstance(key, tuple) and len(key) == 2

    def __getitem__(self, key):
        if not self._is_cell_key(key):
            return dict.__getitem__(self, key)
        grid = self._store
        cell = self._cell_ref(key)
        if cell in grid:
            return grid[cell]
        return 0

    def __setitem__(self, key, value):
        if self._read_only:
            raise TypeError(
                "The grid is read-only inside a function; functions cannot "
                "write to the parent grid")
        if not self._is_cell_key(key):
            dict.__setitem__(self, key, value)
            return
        self._store[self._cell_ref(key)] = value

    def __contains__(self, key):
        if not self._is_cell_key(key):
            return dict.__contains__(self, key)
        return self._cell_ref(key) in self._store

    def __len__(self):
        return len(self._store)

    def __iter__(self):
        for cell in self._store:
            col, row = split_cell(cell)
            yield (int(row), col_to_num(col))

    def get(self, key, default=None):
        if not self._is_cell_key(key):
            return dict.get(self, key, default)
        return self._store.get(self._cell_ref(key), default)

    def keys(self):
        return list(self.__iter__())

    def values(self):
        return list(self._store.values())

    def items(self):
        for key in self.__iter__():
            yield key, self[key]

    def copy(self):
        return dict(self.items())


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
        self.input_variables = []  # Variables that can only receive values (case-insensitive, ordered)
        self.output_variables = set()  # Variables that can only push values
        self.pipe_connections = {}  # Maps outputs to connected inputs
        self.implicit_let = set()
        # Runtime unit of each variable's current value (lowercase keys).
        # Values are stored stripped of the unit wrapper; reads re-wrap.
        self.value_units = {}

    def get_value_unit(self, name):
        """Return the runtime unit of a variable (or None)."""
        key = self._get_case_insensitive_key(name, self.value_units)
        if key is None:
            key = self._get_case_insensitive_key(name, self.variables)
        if key is None:
            return None
        return self.value_units.get(key.lower()) or None

    def _unit_convert(self, name, value, constraints=None, line_number=None):
        """Decompose an incoming (possibly unit-bearing) value for storage.

        Returns ``(stored_value, runtime_unit)``. A unit mismatch produces the
        sticky ``#UNIT`` error value instead of raising.
        """
        constraints = constraints or {}
        if isinstance(value, UnitValue):
            if value.error:
                return UNIT_ERROR, None
            incoming = value.unit
            plain = value.value
        else:
            incoming = None
            plain = value

        not_unit = constraints.get('not_unit')
        if not_unit and incoming and str(incoming).lower() == str(not_unit).lower():
            return UNIT_ERROR, None

        declared = constraints.get('unit')
        if declared:
            declared = str(declared).lower()
            if incoming and str(incoming).lower() != declared:
                return UNIT_ERROR, None
            return plain, declared
        return plain, incoming

    def _wrap_for_eval(self, name, value):
        """Wrap a stored value for expression evaluation (read side)."""
        if value == UNIT_ERROR:
            return UnitValue(None, None, error=True)
        if isinstance(value, UnitValue):
            return value
        unit = self.get_value_unit(name)
        if unit:
            return UnitValue(value, unit)
        return value

    def _get_case_insensitive_key(self, name, dictionary):
        """Get a key from dictionary in a case-insensitive manner"""
        name_lower = name.lower()
        for key in dictionary:
            if key.lower() == name_lower:
                return key
        return None

    def _coerce_custom_type_value(self, type_name, value, constraints=None, line_number=None):
        adjusted_value = value
        adjusted_constraints = constraints or {}
        if (
            adjusted_value is None
            or not type_name
            or not hasattr(self, 'compiler')
            or not hasattr(self.compiler, 'types_defined')
            or type_name.lower() not in self.compiler.types_defined
        ):
            return adjusted_value, adjusted_constraints

        if isinstance(adjusted_value, pa.Array) and not adjusted_constraints.get('dim'):
            adjusted_value = adjusted_value.to_pylist()
        if isinstance(adjusted_value, list) and not adjusted_constraints.get('dim'):
            adjusted_value = self.compiler._convert_array_to_object(
                type_name, adjusted_value, line_number)

        constant_expr = adjusted_constraints.get('constant')
        raw_is_typed_literal = isinstance(constant_expr, (list, tuple, pa.Array))
        if isinstance(constant_expr, str):
            constant_text = constant_expr.strip()
            raw_is_typed_literal = constant_text.startswith('{') and constant_text.endswith('}')
        if raw_is_typed_literal and isinstance(adjusted_value, dict):
            adjusted_constraints = dict(adjusted_constraints)
            adjusted_constraints['constant'] = adjusted_value

        return adjusted_value, adjusted_constraints

    def _strip_init_copy_immutability(self, value):
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                if key == '_immutable_fields':
                    continue
                cleaned[key] = self._strip_init_copy_immutability(item)
            return cleaned
        if isinstance(value, list):
            return [self._strip_init_copy_immutability(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._strip_init_copy_immutability(item) for item in value)
        return value

    def _materialize_lazy_init_value(self, name, value, line_number=None):
        actual_key = self._get_case_insensitive_key(name, self.variables) or name
        try:
            materialized = copy.deepcopy(value)
        except Exception:
            materialized = value
        materialized = self._strip_init_copy_immutability(materialized)

        constraints_key = self._get_case_insensitive_key(
            actual_key, self.constraints) or actual_key
        constraints = self.constraints.get(constraints_key, {})
        type_key = self._get_case_insensitive_key(actual_key, self.types) or actual_key
        var_type = self.types.get(type_key)

        materialized, runtime_unit = self._unit_convert(
            actual_key, materialized, constraints, line_number)
        if materialized is not None and materialized != UNIT_ERROR and var_type and hasattr(self, 'compiler'):
            materialized, constraints = self._coerce_custom_type_value(
                var_type, materialized, constraints, line_number)
            self.constraints[constraints_key] = constraints
        if materialized is not None and materialized != UNIT_ERROR and constraints and constraints.get('dim') and hasattr(self, 'compiler'):
            materialized = self.compiler.array_handler.check_dimension_constraints(
                actual_key, materialized, line_number)
        if materialized is not None:
            self._check_constraints(actual_key, materialized, line_number)

        self.variables[actual_key] = materialized
        self.value_units[actual_key.lower()] = runtime_unit
        self.uninitialized.discard(actual_key)
        if hasattr(self.compiler, 'mark_dependency_resolved'):
            self.compiler.mark_dependency_resolved(actual_key)
        return materialized

    def _validate_variable_name(self, name, line_number=None):
        """Reject names that are not valid GridLang variable identifiers.

        A variable name must start with a letter and may contain letters,
        digits, '_' and '.' (never in the last position).
        """
        valid = (
            name
            and name[0].isalpha()
            and not name.endswith('.')
            and all(ch.isalnum() or ch in '._' for ch in name)
        )
        if valid:
            return
        at = f" at line {line_number}" if line_number else ""
        raise SyntaxError(
            f"Invalid variable name '{name}'{at}.")

    def define(self, name, value=None, type=None, constraints=None, is_uninitialized=False, line_number=None):
        effective_constraints = constraints or {}
        self._validate_variable_name(name, line_number)
        # Check for case-insensitive conflicts
        existing_key = self._get_case_insensitive_key(name, self.variables)
        if existing_key and not is_uninitialized:
            raise ValueError(
                f"Variable '{name}' conflicts with existing variable '{existing_key}' in this scope")
        value, runtime_unit = self._unit_convert(
            name, value, effective_constraints, line_number)
        if value is not None and value != UNIT_ERROR and type and hasattr(self, 'compiler') and hasattr(self.compiler, 'types_defined'):
            value, effective_constraints = self._coerce_custom_type_value(
                type, value, effective_constraints, line_number)
        if value is not None and not is_uninitialized:
            if value != UNIT_ERROR and effective_constraints and effective_constraints.get('dim') and hasattr(self, 'compiler'):
                value = self.compiler.array_handler.check_dimension_constraints(
                    name, value, line_number)
            self._check_constraints(name, value, line_number)
        self.variables[name] = value
        self.value_units[name.lower()] = runtime_unit
        self.types[name] = type
        self.constraints[name] = effective_constraints
        if is_uninitialized:
            self.uninitialized.add(name)
        else:
            self.uninitialized.discard(name)
        if hasattr(self.compiler, 'mark_dependency_resolved'):
            self.compiler.mark_dependency_resolved(name)

    def update(self, name, value, line_number=None):
        defining_scope = self.get_defining_scope(name)
        if defining_scope:
            # Functions are read-only with respect to the caller's scope chain.
            if (getattr(self.compiler, '_outer_scope_read_only', False)
                    and self.compiler._is_outer_scope(defining_scope)):
                raise RuntimeError(
                    f"Cannot assign to '{name}': variables in an outer scope are read-only inside a function at line {line_number}")
            # Guard against writes routed through the defining scope object
            # itself (whose compiler does not carry the read-only flag).
            for runner in reversed(_ACTIVE_RUNNERS):
                if (getattr(runner, '_outer_scope_read_only', False)
                        and runner._is_outer_scope(defining_scope)):
                    raise RuntimeError(
                        f"Cannot assign to '{name}': variables in an outer scope are read-only inside a function at line {line_number}")
            # Get the actual key for case-insensitive update
            actual_key = defining_scope._get_case_insensitive_key(
                name, defining_scope.variables)
            if actual_key:
                var_type = defining_scope.types.get(actual_key)
                constraints = defining_scope.constraints.get(actual_key, {})
                value, runtime_unit = self._unit_convert(
                    name, value, constraints, line_number)
                if value is not None and value != UNIT_ERROR and var_type and hasattr(self, 'compiler') and hasattr(self.compiler, 'types_defined'):
                    value, constraints = defining_scope._coerce_custom_type_value(
                        var_type, value, constraints, line_number)
                    defining_scope.constraints[actual_key] = constraints
                # Prevent updating input variables once initialized
                if defining_scope.constraints.get(actual_key, {}).get('input') and actual_key not in defining_scope.uninitialized:
                    raise ValueError(
                        f"Input variable '{actual_key}' cannot be updated at line {line_number}")
                if value != UNIT_ERROR and defining_scope.constraints.get(actual_key, {}).get('dim'):
                    value = self.compiler.array_handler.check_dimension_constraints(
                        actual_key, value, line_number)
                defining_scope._check_constraints(actual_key, value, line_number)
                defining_scope.variables[actual_key] = value
                defining_scope.value_units[actual_key.lower()] = runtime_unit
                defining_scope.uninitialized.discard(actual_key)

                # Re-evaluate constraint expressions that depend on this variable
                self._re_evaluate_constraints(actual_key, line_number)
                if hasattr(self.compiler, 'mark_dependency_resolved'):
                    self.compiler.mark_dependency_resolved(actual_key)
                if hasattr(self.compiler, '_sync_cell_bindings'):
                    self.compiler._sync_cell_bindings(actual_key, value)
                if hasattr(self.compiler, '_record_output_value'):
                    self.compiler._record_output_value(actual_key, value)
                if hasattr(self.compiler, '_notify_var_changed'):
                    self.compiler._notify_var_changed(actual_key, value)
            else:
                # Variable exists in types or constraints but not variables
                value, runtime_unit = self._unit_convert(
                    name, value, defining_scope.constraints.get(name, {}), line_number)
                defining_scope._check_constraints(name, value, line_number)
                defining_scope.variables[name] = value
                defining_scope.value_units[name.lower()] = runtime_unit
                defining_scope.uninitialized.discard(name)

                # Re-evaluate constraint expressions that depend on this variable
                self._re_evaluate_constraints(name, line_number)
                if hasattr(self.compiler, 'mark_dependency_resolved'):
                    self.compiler.mark_dependency_resolved(name)
                if hasattr(self.compiler, '_sync_cell_bindings'):
                    self.compiler._sync_cell_bindings(name, value)
                if hasattr(self.compiler, '_record_output_value'):
                    self.compiler._record_output_value(name, value)
                if hasattr(self.compiler, '_notify_var_changed'):
                    self.compiler._notify_var_changed(name, value)
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
                        value = self._materialize_lazy_init_value(
                            actual_key, value)
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
            wrapped = self._wrap_for_eval(var_name, var_value)
            full_scope[var_name] = wrapped
            # Add lowercase version for case-insensitive access
            full_scope[var_name.lower()] = wrapped
            # Add uppercase version for case-insensitive access
            full_scope[var_name.upper()] = wrapped

        current = current.parent
        while current:
            # Include variables from all parent scopes, including private ones
            # This is necessary for nested FOR loops where outer loop variables
            # need to be accessible to inner loops
            for var_name, var_value in current.variables.items():
                # Only add if not already present (to avoid overriding local variables)
                if var_name not in full_scope:
                    wrapped = current._wrap_for_eval(var_name, var_value)
                    full_scope[var_name] = wrapped
                    # Add case-insensitive versions only if not already present
                    if var_name.lower() not in full_scope:
                        full_scope[var_name.lower()] = wrapped
                    if var_name.upper() not in full_scope:
                        full_scope[var_name.upper()] = wrapped
            current = current.parent
        return full_scope

    def _re_evaluate_constraints(self, changed_var, line_number=None):
        """Re-evaluate constraint expressions that depend on the changed variable"""

        # Find all variables that have constraint expressions depending on changed_var
        for var_name, constraints in list(self.constraints.items()):
            for constraint_type, constraint_expr in constraints.items():
                if constraint_type == 'constant' and isinstance(constraint_expr, str):
                    # Check if this constraint expression depends on the changed variable
                    if self._expression_depends_on(constraint_expr, changed_var):
                        try:
                            new_value = self.compiler.expr_evaluator.eval_or_eval_array(
                                constraint_expr, self.get_full_scope(), line_number)
                        except Exception:
                            # Expression cannot be resolved yet; keep waiting.
                            continue

                        # Assign through update() so dimension and type checks
                        # apply; validation errors propagate to the caller.
                        self.update(var_name, new_value, line_number)

    def _expression_depends_on(self, expr, var_name):
        """Check if an expression depends on a specific variable"""
        # Simple dependency check - look for the variable name in the expression
        # This is a basic implementation; could be enhanced with proper parsing
        import re
        expr_text = expr if isinstance(expr, str) else str(expr)
        extra = []
        if '$"' in expr_text or "$'" in expr_text:
            # Placeholders inside interpolated strings ($"...{expr}...") are
            # real dependencies, so collect them from the interpolation spans.
            extra = list(iter_interpolation_placeholders(expr_text))
        # Strip quoted strings to avoid false positives from literals (e.g. a
        # text array like {"b", "c"} must not be treated as a dependency on b).
        expr_text = re.sub(
            r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', ' ', expr_text)
        if extra:
            expr_text += ' ' + ' '.join(extra)
        # Create a pattern that matches the variable name as a whole word
        pattern = r'\b' + re.escape(var_name) + r'\b'
        return bool(re.search(pattern, expr_text))

    def _validate_base_type(self, name, value, line_number=None):
        """Validate that a scalar value matches the declared base type.

        Arrays and custom-typed objects are validated by the dimension checks
        and custom-type coercion respectively, so only scalar values are
        checked here.
        """
        if value is None or isinstance(value, (list, tuple, pa.Array, dict)):
            return
        if value == '':
            # Empty string is used as a placeholder by block predeclaration.
            return
        type_key = self._get_case_insensitive_key(name, self.types)
        if not type_key:
            return
        var_type = self.types.get(type_key)
        if var_type not in ('number', 'text'):
            return
        if hasattr(self, 'compiler') and hasattr(self.compiler, 'types_defined') and var_type in self.compiler.types_defined:
            return
        constraints = self.constraints.get(type_key, {}) or {}
        if constraints.get('dim'):
            return
        if constraints.get('input') or constraints.get('output'):
            # Inputs/outputs are validated by their own 'type' constraint or
            # left loosely typed (untyped OUTPUT defaults to 'text').
            return
        actual_type = self.compiler.array_handler.infer_type(
            value, line_number)
        if var_type == 'number' and actual_type not in ('number', 'float64', 'int', 'int64'):
            raise ValueError(
                f"'{name}' must be a number, got {actual_type} at line {line_number}")
        if var_type == 'text' and actual_type not in ('string', 'text'):
            raise ValueError(
                f"'{name}' must be text, got {actual_type} at line {line_number}")

    def _check_constraints(self, name, value, line_number=None):
        # Case-insensitive constraint lookup
        if value == UNIT_ERROR:
            # A sticky unit error bypasses all constraint/type validation.
            return
        actual_key = self._get_case_insensitive_key(name, self.constraints)
        key_for_constraints = actual_key if actual_key is not None else name
        constraints = self.constraints.get(key_for_constraints, {})
        self._validate_base_type(key_for_constraints, value, line_number)
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
                # WITH constraints are stored separately from '=' parsing.
                # Apply them before constant comparison so
                # "new Type with (...)" compares against the constrained value.
                if constraints.get('with'):
                    try:
                        type_name = None
                        actual_type_key = self._get_case_insensitive_key(
                            key_for_constraints, self.types)
                        if actual_type_key:
                            type_name = self.types.get(actual_type_key)
                        constraint_val = self.compiler._apply_with_constraints(
                            constraint_val,
                            constraints.get('with', {}),
                            self.get_full_scope(),
                            line_number,
                            type_name=type_name,
                        )
                    except Exception:
                        pass
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
                allowed_values = constraint_expr
                if isinstance(constraint_expr, str):
                    try:
                        allowed_values = self.compiler.expr_evaluator.eval_or_eval_array(
                            constraint_expr, self.get_full_scope(), line_number)
                    except Exception:
                        allowed_values = constraint_expr
                if isinstance(allowed_values, pa.Array):
                    allowed_values = allowed_values.to_pylist()
                if isinstance(allowed_values, str):
                    allowed_values = [allowed_values]
                if isinstance(value, (list, tuple, set)):
                    if not all(item in allowed_values for item in value):
                        raise ValueError(
                            f"'{key_for_constraints}' values {value} not in allowed values {allowed_values} at line {line_number}")
                elif value not in allowed_values:
                    raise ValueError(
                        f"'{key_for_constraints}' value {value} not in allowed values {allowed_values} at line {line_number}")
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
            elif constraint_type == 'not_unit':
                if isinstance(value, str) and value == constraint_expr:
                    raise ValueError(
                        f"'{key_for_constraints}' must not be unit {constraint_expr} at line {line_number}")

    def get_full_scope(self):
        full_scope = {}
        current = self
        while current and not current.is_private:
            for var_name, var_value in current.variables.items():
                full_scope[var_name] = current._wrap_for_eval(var_name, var_value)
            current = current.parent
        return full_scope
