"""
Scope management for GridLang compiler.
Handles variable scoping, constraints, and pipe connections.
"""

import copy
import re

from units import (
    DIM_ERROR, NA_ERROR, NUM_ERROR, TYPE_ERROR, UNIT_ERROR, UNIVERSAL_ZERO,
    VALUE_ERROR, ConstraintError, UnitValue, apply_conversion, error_value,
    is_error_value, strip_units,
)
from utils import (
    iter_interpolation_placeholders, is_sparse_array,
)
from array_handler import constant_value_matches

# Stack of compiler run contexts: ``run()`` pushes the executing compiler and
# pops it on exit. Used to detect writes that originate from a read-only
# function sub-compiler and target a scope in that compiler's parent chain.
_ACTIVE_RUNNERS = []


def _strip_meta(value):
    """Return a copy of ``value`` with transient meta keys (underscore-prefixed)
    removed recursively, so dict comparisons in constant checks ignore internal
    markers such as ``_fresh_key`` and ``_immutable_fields``."""
    if isinstance(value, dict):
        return {k: _strip_meta(v) for k, v in value.items()
                if not str(k).startswith('_')}
    if isinstance(value, (list, tuple)):
        return [_strip_meta(v) for v in value]
    return value


class _GridStore(dict):
    """The single grid backing store, held as the predefined ``grid`` variable.

    Storage is a sparse dict keyed strictly by 0-based numeric index tuples
    (``(row, col)`` or extended N-D tuples), matching how the grid array is
    read and written with ``grid{row, col}``. Address-string keys (``'A1'``)
    are NOT accepted here; the compiler's ``_to_index``/``_set_grid_cell``
    helpers perform that conversion at the call sites.

    Every write strips units and notifies the owning compiler so that client
    variables listening on a grid cell are recomputed when a publisher
    (push/init) updates one of their dependency cells.
    """

    def __init__(self, owner=None):
        super().__init__()
        self._grid_owner = owner

    @staticmethod
    def _normalize_key(key):
        if isinstance(key, tuple):
            return key
        raise TypeError(
            f"Grid store keys must be numeric index tuples, got {key!r}")

    def __setitem__(self, key, value):
        key = self._normalize_key(key)
        super().__setitem__(key, strip_units(value))
        owner = self._grid_owner
        if owner is not None and hasattr(owner, '_notify_cell_changed'):
            owner._notify_cell_changed(key, value)

    def __getitem__(self, key):
        return super().__getitem__(self._normalize_key(key))

    def __contains__(self, key):
        try:
            return super().__contains__(self._normalize_key(key))
        except TypeError:
            return False

    def get(self, key, default=None):
        try:
            return super().get(self._normalize_key(key), default)
        except TypeError:
            return default


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
        # Lazy conflict detection: variables pushed while having a constant
        # constraint.  Validated on first read to avoid eager #VALUE checks.
        self._conflict_flags = set()

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
            if value.error_code is not None:
                return value.error_code, None
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
                try:
                    conv = None
                    if hasattr(self, 'compiler') and hasattr(self.compiler, 'expr_evaluator'):
                        conv = apply_conversion(plain, incoming, declared, self.compiler.expr_evaluator._formula_eval)
                    else:
                        conv = apply_conversion(plain, incoming, declared)
                    if conv is not None:
                        return conv.value, conv.unit
                except Exception:
                    pass
                return UNIT_ERROR, None
            return plain, declared
        return plain, incoming

    def _has_pending_assignment(self, name):
        """True when a late assignment for ``name`` is still outstanding."""
        name_lower = name.lower()
        scope = self
        while scope is not None:
            for key in scope.pending_assignments:
                if key.lower() == name_lower:
                    return True
            scope = scope.parent
        for key in getattr(self.compiler, 'pending_assignments', {}):
            if key.lower() == name_lower:
                return True
        return False

    def _wrap_for_eval(self, name, value):
        """Wrap a stored value for expression evaluation (read side).

        Sticky error codes are re-wrapped as error values; a None value from an
        uninitialized variable (with no pending late assignment) reads as the
        ``#N/A`` error value.
        """
        if is_error_value(value):
            code = value if isinstance(value, str) else value.error_code
            return error_value(code)
        if isinstance(value, UnitValue):
            return value
        if value is None:
            if self.is_uninitialized(name) and not self._has_pending_assignment(name):
                if self._is_not_null_var(name):
                    return error_value(VALUE_ERROR)
                return error_value(NA_ERROR)
            return value
        # Lazy conflict validation: check if a pushed value matches its
        # constant constraint.  This avoids eager #VALUE checks on Push.
        actual_key = self._get_case_insensitive_key(name, self.variables)
        if actual_key and actual_key in self._conflict_flags:
            self._conflict_flags.discard(actual_key)
            constraints = self.constraints.get(actual_key, {})
            constant_expr = constraints.get('constant')
            if constant_expr is not None:
                try:
                    if isinstance(constant_expr, str):
                        expected = self.compiler.expr_evaluator.eval_or_eval_array(
                            constant_expr, self.get_full_scope())
                    else:
                        expected = constant_expr
                    # Apply WITH constraints if present
                    if constraints.get('with'):
                        try:
                            type_name = None
                            actual_type_key = self._get_case_insensitive_key(
                                actual_key, self.types)
                            if actual_type_key:
                                type_name = self.types.get(actual_type_key)
                            expected = self.compiler._apply_with_constraints(
                                expected,
                                constraints.get('with', {}),
                                self.get_full_scope(),
                                None,
                                type_name=type_name,
                            )
                        except Exception:
                            pass
                    if isinstance(expected, dict) and isinstance(value, dict):
                        # Type-instance snapshot: compare only public fields
                        if is_sparse_array(expected) or is_sparse_array(value):
                            if not constant_value_matches(
                                    value, expected,
                                    self.compiler.array_handler.to_display_value):
                                return error_value(VALUE_ERROR)
                        else:
                            pub_val = {k: v for k, v in value.items()
                                       if not str(k).startswith('_')}
                            if pub_val != expected:
                                return error_value(VALUE_ERROR)
                    elif not constant_value_matches(
                            value, expected,
                            self.compiler.array_handler.to_display_value):
                        return error_value(VALUE_ERROR)
                except Exception:
                    pass
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

        if isinstance(adjusted_value, dict) and 'array' in adjusted_value and not adjusted_constraints.get('dim'):
            adjusted_value = list(adjusted_value['array'])
        elif is_sparse_array(adjusted_value) and not adjusted_constraints.get('dim'):
            adjusted_value = [adjusted_value[k]
                              for k in sorted(adjusted_value.keys())]
        if isinstance(adjusted_value, list) and not adjusted_constraints.get('dim'):
            adjusted_value = self.compiler._convert_array_to_object(
                type_name, adjusted_value, line_number)

        constant_expr = adjusted_constraints.get('constant')
        raw_is_typed_literal = isinstance(constant_expr, (list, tuple, dict))
        if isinstance(constant_expr, str):
            constant_text = constant_expr.strip()
            raw_is_typed_literal = constant_text.startswith('{') and constant_text.endswith('}')
        if raw_is_typed_literal and isinstance(adjusted_value, dict) and isinstance(constant_expr, str):
            adjusted_constraints = dict(adjusted_constraints)
            adjusted_constraints['constant'] = adjusted_value

        return adjusted_value, adjusted_constraints

    def _materialize_no_dim_list(self, name, value, constraints, line_number=None):
        """Convert a plain list value into sparse index-keyed dict storage
        when the variable has no explicit dim constraint, so dim-less
        literals, ragged arrays and object arrays share the sparse path."""
        if (
            value is not None
            and not is_error_value(value)
            and isinstance(value, list)
            and not constraints.get('dim')
            and hasattr(self, 'compiler')
            and hasattr(self.compiler, 'array_handler')
        ):
            try:
                return self.compiler.array_handler.materialize_list_array(
                    value, line_number)
            except Exception:
                return value
        return value

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

    def _coerce_universal_zero(self, value, var_type):
        """Turn the `None` sentinel into the declared base type's zero.

        Applied to typed values: a ``number`` variable stores 0, a ``logical``
        variable stores ``False``, a ``text`` variable stores ``""``. For typed
        arrays, every element is coerced. Untyped variables keep the sentinel
        so operators/functions can coerce it contextually.
        """
        if value is UNIVERSAL_ZERO:
            if var_type == 'number':
                return 0
            if var_type == 'logical':
                return False
            if var_type == 'text':
                return ""
            return value
        if isinstance(value, dict) and 'array' in value:
            inner = value.get('array')
            if isinstance(inner, list):
                result = dict(value)
                result['array'] = [
                    self._coerce_universal_zero(item, var_type)
                    for item in inner]
                return result
            return value
        if isinstance(value, dict) and value and all(
                isinstance(k, tuple) for k in value.keys()):
            return {
                k: self._coerce_universal_zero(v, var_type)
                for k, v in value.items()}
        if isinstance(value, list):
            return [self._coerce_universal_zero(item, var_type)
                    for item in value]
        if isinstance(value, tuple):
            return tuple(self._coerce_universal_zero(item, var_type)
                         for item in value)
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
        materialized = self._coerce_universal_zero(materialized, var_type)
        if materialized is not None and not is_error_value(materialized) and var_type and hasattr(self, 'compiler'):
            materialized, constraints = self._coerce_custom_type_value(
                var_type, materialized, constraints, line_number)
            self.constraints[constraints_key] = constraints
        if materialized is not None and not is_error_value(materialized) and constraints and constraints.get('dim') and hasattr(self, 'compiler'):
            try:
                materialized = self.compiler.array_handler.check_dimension_constraints(
                    actual_key, materialized, line_number)
            except ConstraintError as exc:
                materialized = exc.code
                runtime_unit = None
        if materialized is not None:
            try:
                self._check_constraints(actual_key, materialized, line_number)
            except ConstraintError as exc:
                materialized = exc.code
                runtime_unit = None
        materialized = self._materialize_no_dim_list(
            actual_key, materialized, constraints, line_number)

        self.variables[actual_key] = materialized
        self.value_units[actual_key.lower()] = runtime_unit
        self.uninitialized.discard(actual_key)
        if hasattr(self.compiler, 'mark_dependency_resolved'):
            self.compiler.mark_dependency_resolved(actual_key)
        return materialized

    def _validate_variable_name(self, name, line_number=None, internal=False):
        """Reject names that are not valid GridLang variable identifiers.

        A variable name must start with a letter and may contain letters,
        digits, '_' and '.' (never in the last position).
        Names starting with '_' are reserved for internal use.
        """
        valid = (
            name
            and (name[0].isalpha() or (internal and name[0] == '_'))
            and not name.endswith('.')
            and all(ch.isalnum() or ch in '._' for ch in name)
        )
        if valid:
            return
        at = f" at line {line_number}" if line_number else ""
        raise SyntaxError(
            f"Invalid variable name '{name}'{at}.")

    def _is_keyed_type(self, type_name, compiler=None):
        """True if ``type_name`` is a declared Keytype (``as Keytype``)."""
        if not type_name:
            return False
        compiler = compiler or getattr(self, 'compiler', None)
        td = (getattr(compiler, 'types_defined', None)
              or {}).get(str(type_name).lower(), {})
        return bool(td.get('_keyed'))

    def _is_not_null_var(self, name):
        """True if ``name`` must not be null: a Keytype variable, an
        ``as ... key`` variable (e.g. ``x as text key``), a variable with a
        ``key`` constraint, or a variable declared ``not null``. Unset
        not-null variables read as ``#VALUE``."""
        actual_key = self._get_case_insensitive_key(name, self.types) or name
        var_type = self.types.get(actual_key)
        if isinstance(var_type, str) and var_type.strip().lower().endswith(' key'):
            return True
        constraints = self.constraints.get(actual_key, {})
        if constraints.get('key') or constraints.get('not_null'):
            return True
        return self._is_keyed_type(var_type, getattr(self, 'compiler', None))

    def _keytype_primitive_base(self, type_name, compiler=None):
        """Return the primitive base type of ``type_name`` if it is a Keytype
        primitive alias (e.g. ``L as Keytype(number)``), else None."""
        if not self._is_keyed_type(type_name, compiler):
            return None
        compiler = compiler or getattr(self, 'compiler', None)
        td = (getattr(compiler, 'types_defined', None)
              or {}).get(str(type_name).lower(), {})
        if td.get('_base_type') not in ('number', 'text', 'logical'):
            return None
        try:
            public = compiler._get_public_type_fields(td) if hasattr(
                compiler, '_get_public_type_fields') else {}
        except Exception:
            public = {}
        if public:
            return None
        return td.get('_base_type')

    def _wrap_keytype_primitive(self, value, type_name, compiler=None):
        """Wrap a plain primitive ``value`` as a fresh Keytype instance when the
        destination is a Keytype primitive alias (e.g. ``x as L = 5`` where
        ``L as Keytype(number)``). Returns the possibly-wrapped value."""
        base = self._keytype_primitive_base(type_name, compiler)
        if base is None or isinstance(value, UnitValue) or value is None \
                or is_error_value(value):
            return value
        if (base == 'number' and isinstance(value, (int, float)) and not isinstance(value, bool)) \
                or (base == 'text' and isinstance(value, str)) \
                or (base == 'logical' and isinstance(value, bool)):
            return UnitValue(value, unit=None, key_type=type_name.lower(), fresh_key=True)
        return value

    def _apply_keyed_not_null(self, value):
        """If ``value`` is a stored custom-type instance with a keyed field left
        unset (``None``), turn the whole instance into a sticky #VALUE error.

        Keyed fields (Keytype-typed or ``as ... key``) imply Not Null, so an
        instance whose keyed field is never assigned is invalid at storage.
        """
        if not isinstance(value, dict) or is_error_value(value):
            return value
        type_name = value.get('_type_name')
        if not type_name or not hasattr(self, 'compiler') \
                or not hasattr(self.compiler, 'types_defined') \
                or type_name.lower() not in self.compiler.types_defined:
            return value
        tdef = self.compiler.types_defined.get(type_name.lower())
        try:
            keyed = self.compiler._keyed_field_names(tdef)
        except Exception:
            keyed = set()
        if not keyed:
            return value
        for field in keyed:
            for key, item in value.items():
                if key.lower() == field and item is None:
                    return error_value(VALUE_ERROR)
        return value

    def define(self, name, value=None, type=None, constraints=None, is_uninitialized=False, line_number=None, internal=False, preserve_freshness=False):
        effective_constraints = constraints or {}
        self._validate_variable_name(name, line_number, internal=internal)
        # Check for case-insensitive conflicts
        existing_key = self._get_case_insensitive_key(name, self.variables)
        if existing_key and not is_uninitialized:
            raise ValueError(
                f"Variable '{name}' conflicts with existing variable '{existing_key}' in this scope")
        # Handle Keytype freshness and implicit as IdKey
        # Wrap plain primitive values as fresh Keytype instances when dest is a Keytype (e.g. x as L = 5 where L as Keytype(number))
        value = self._wrap_keytype_primitive(value, type)
        # If value is a fresh Keytype instance (UnitValue with key_type, or a
        # keyed object dict marked _fresh_key) and dest has no explicit type,
        # implicitly add `as <keytype>` (e.g. Let x = new IdKey -> x as IdKey).
        value_key_type = None
        value_is_fresh = False
        compiler_ref = getattr(self, 'compiler', None)
        if compiler_ref is not None and hasattr(compiler_ref, '_key_type_of_value'):
            value_key_type = compiler_ref._key_type_of_value(value)
            value_is_fresh = bool(value_key_type and compiler_ref._key_value_is_fresh(value))
        if value_key_type and value_is_fresh:
            if (not type or type.lower() in ('unknown', 'object')) and not effective_constraints.get('type'):
                # Implicitly add `as <keytype>` for storage vars (not Output, which doesn't store)
                is_output = effective_constraints.get('output') or self.is_output(name)
                if not is_output:
                    # Check if dest already has another type constraint (hard error)
                    existing_type = self.types.get(name) or self.types.get(self._get_case_insensitive_key(name, self.types) or "")
                    if existing_type and existing_type.lower() not in ('unknown', 'object', str(value_key_type).lower()):
                        raise ConstraintError(TYPE_ERROR, f"Variable '{name}' already has type '{existing_type}', cannot implicitly add '{value_key_type}' at line {line_number}")
                    type = str(value_key_type).lower()
                    effective_constraints = dict(effective_constraints)
                    effective_constraints['type'] = type.lower()
            elif type and type.lower() not in ('unknown', 'object', str(value_key_type).lower()):
                raise ConstraintError(TYPE_ERROR, f"Variable '{name}' already has type '{type}', cannot assign '{value_key_type}' instance at line {line_number}")
        # Copying a stored (non-fresh) keytype instance via Let/:/For/Input/new Copy is #TYPE/I.
        # `k as number key` field values (plain numbers, foreign keys) are not
        # keytype instances and remain copyable.
        if value_key_type and compiler_ref is not None and hasattr(compiler_ref, '_check_key_type_copy'):
            compiler_ref._check_key_type_copy(
                name, type, value_key_type, value, self, line_number)
        # Clear freshness when stored to a non-Output var (Output doesn't store)
        is_output_final = effective_constraints.get('output') or self.is_output(name)
        if not is_output_final and not preserve_freshness:
            if isinstance(value, UnitValue) and getattr(value, 'fresh_key', False):
                # Create a non-fresh copy for storage
                value = UnitValue(value.value, value.unit, error_code=value.error_code, key_type=value.key_type, fresh_key=False)
            if compiler_ref is not None and hasattr(compiler_ref, '_mark_keytype_stored'):
                compiler_ref._mark_keytype_stored(value)
        # For Keytype values, bypass _unit_convert's stripping of UnitValue wrapper (keep key_type)
        if isinstance(value, UnitValue) and getattr(value, 'key_type', None):
            runtime_unit = value.unit
            # Keep value as UnitValue with key_type (don't strip to plain value)
            # _coerce_universal_zero and other handling should preserve it
        else:
            value, runtime_unit = self._unit_convert(
            name, value, effective_constraints, line_number)
        value = self._coerce_universal_zero(value, type)
        if value is not None and not is_error_value(value) and type and hasattr(self, 'compiler') and hasattr(self.compiler, 'types_defined'):
            value, effective_constraints = self._coerce_custom_type_value(
                type, value, effective_constraints, line_number)
        # For primitive arrays (number/text with dim), set constraints early so _validate_base_type sees dim
        if type in ('number', 'text') and effective_constraints and effective_constraints.get('dim'):
            self.types[name] = type
            self.constraints[name] = effective_constraints
        if value is not None and not is_uninitialized:
            if not is_error_value(value) and effective_constraints and effective_constraints.get('dim') and hasattr(self, 'compiler'):
                try:
                    value = self.compiler.array_handler.check_dimension_constraints(
                        name, value, line_number)
                except ConstraintError as exc:
                    value = exc.code
                    runtime_unit = None
            if not is_error_value(value):
                try:
                    self._check_constraints(name, value, line_number)
                except ConstraintError as exc:
                    value = exc.code
                    runtime_unit = None
        value = self._materialize_no_dim_list(
            name, value, effective_constraints, line_number)
        if not preserve_freshness:
            value = self._apply_keyed_not_null(value)
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

    def update(self, name, value, line_number=None, preserve_freshness=False):
        defining_scope = self.get_defining_scope(name)
        if defining_scope:
            # Functions are read-only with respect to the caller's scope chain.
            if (getattr(self.compiler, '_outer_scope_read_only', False)
                    and self.compiler._is_outer_scope(defining_scope)):
                raise RuntimeError(
                    f"Cannot assign to '{name}': variables in an outer scope are read-only at line {line_number}")
            # Guard against writes routed through the defining scope object
            # itself (whose compiler does not carry the read-only flag).
            for runner in reversed(_ACTIVE_RUNNERS):
                if (getattr(runner, '_outer_scope_read_only', False)
                        and runner._is_outer_scope(defining_scope)):
                    raise RuntimeError(
                        f"Cannot assign to '{name}': variables in an outer scope are read-only at line {line_number}")
            # Get the actual key for case-insensitive update
            actual_key = defining_scope._get_case_insensitive_key(
                name, defining_scope.variables)
            if actual_key:
                var_type = defining_scope.types.get(actual_key)
                constraints = defining_scope.constraints.get(actual_key, {})
                # Wrap plain primitive values as fresh Keytype instances when dest is a Keytype (e.g. x as L = 5 where L as Keytype(number))
                value = defining_scope._wrap_keytype_primitive(value, var_type, defining_scope.compiler)
                # For key types (e.g. x as L where L as Keytype), the variable is immutable after first assignment
                # But Output variables are not storage - they can be pushed multiple times (freshness kept)
                is_output_check = constraints.get('output') or defining_scope.is_output(actual_key)
                if not is_output_check and defining_scope._is_keyed_type(var_type, defining_scope.compiler):
                    existing_val = defining_scope.variables.get(actual_key)
                    # Allow initial assignment from None (e.g. Let x = new IdKey where x was is_uninitialized with None)
                    # But any subsequent Push/For update where existing already has a non-None, non-error value should error
                    if existing_val is not None and not is_error_value(existing_val):
                        # Check if this is an update (not initial) - actual_key already exists and has value
                        # For key types, even Push with same value should be considered immutable
                        # However, allow the first builder assignment after Copy where old is None (nulled) -> but old is not None here, it's 5, so need to distinguish
                        # For Copy case, old is None (nulled) and new is value, so existing would be None, not here
                        # Here, existing is not None, so any update to a keytype var should error
                        raise ConstraintError(TYPE_ERROR, f"Cannot modify key type '{var_type}' at line {line_number}")
                # Handle Keytype freshness, implicit as IdKey, and copy check for Push/For updates
                compiler_ref = getattr(defining_scope, 'compiler', None) or getattr(self, 'compiler', None)
                value_key_type = None
                value_is_fresh = False
                if compiler_ref is not None and hasattr(compiler_ref, '_key_type_of_value'):
                    value_key_type = compiler_ref._key_type_of_value(value)
                    value_is_fresh = bool(value_key_type and compiler_ref._key_value_is_fresh(value))
                if value_key_type:
                    # Hard error if dest already has a different explicit type
                    if var_type and var_type.lower() not in ('unknown', 'object', str(value_key_type).lower()):
                        raise ConstraintError(TYPE_ERROR, f"Variable '{actual_key}' already has type '{var_type}', cannot assign '{value_key_type}' instance at line {line_number}")
                    if value_is_fresh:
                        # Implicit as Keytype if dest has no explicit type and value is fresh
                        if (not var_type or var_type.lower() in ('unknown', 'object')):
                            is_output_dest = constraints.get('output') or defining_scope.is_output(actual_key)
                            if not is_output_dest:
                                existing_type_for_check = defining_scope.types.get(actual_key)
                                if existing_type_for_check and existing_type_for_check.lower() not in ('unknown', 'object', str(value_key_type).lower()):
                                    raise ConstraintError(TYPE_ERROR, f"Variable '{actual_key}' already has type '{existing_type_for_check}', cannot implicitly add '{value_key_type}' at line {line_number}")
                                var_type = str(value_key_type).lower()
                                defining_scope.types[actual_key] = var_type
                    # Copying a stored (non-fresh) keytype instance via Push/For/Let -> #TYPE/I
                    if compiler_ref is not None and hasattr(compiler_ref, '_check_key_type_copy'):
                        compiler_ref._check_key_type_copy(actual_key, var_type, value_key_type, value, defining_scope, line_number)
                is_output = constraints.get('output') or defining_scope.is_output(actual_key)
                if not is_output and not preserve_freshness:
                    # Fresh keytype instance being stored to a storage var -> clear freshness
                    if isinstance(value, UnitValue) and getattr(value, 'fresh_key', False):
                        value = UnitValue(value.value, value.unit, error_code=value.error_code, key_type=value.key_type, fresh_key=False)
                    if compiler_ref is not None and hasattr(compiler_ref, '_mark_keytype_stored'):
                        compiler_ref._mark_keytype_stored(value)
                # For Keytype values, bypass _unit_convert's stripping of UnitValue wrapper (keep key_type)
                if isinstance(value, UnitValue) and getattr(value, 'key_type', None):
                    runtime_unit = value.unit
                else:
                    value, runtime_unit = self._unit_convert(
                    name, value, constraints, line_number)
                value = self._coerce_universal_zero(value, var_type)
                if value is not None and not is_error_value(value) and var_type and hasattr(self, 'compiler') and hasattr(self.compiler, 'types_defined'):
                    value, constraints = defining_scope._coerce_custom_type_value(
                        var_type, value, constraints, line_number)
                    defining_scope.constraints[actual_key] = constraints
                # Prevent updating input variables once initialized
                if defining_scope.constraints.get(actual_key, {}).get('input') and actual_key not in defining_scope.uninitialized:
                    raise ValueError(
                        f"Input variable '{actual_key}' cannot be updated at line {line_number}")
                if not is_error_value(value) and defining_scope.constraints.get(actual_key, {}).get('dim'):
                    try:
                        value = self.compiler.array_handler.check_dimension_constraints(
                            actual_key, value, line_number)
                    except ConstraintError as exc:
                        value = exc.code
                        runtime_unit = None
                if not is_error_value(value):
                    try:
                        defining_scope._check_constraints(actual_key, value, line_number)
                    except ConstraintError as exc:
                        value = exc.code
                        runtime_unit = None
                value = defining_scope._materialize_no_dim_list(
                    actual_key, value, constraints, line_number)
                if not preserve_freshness:
                    value = defining_scope._apply_keyed_not_null(value)
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
                value = self._coerce_universal_zero(
                    value, defining_scope.types.get(name))
                if not is_error_value(value):
                    try:
                        defining_scope._check_constraints(name, value, line_number)
                    except ConstraintError as exc:
                        value = exc.code
                        runtime_unit = None
                value = defining_scope._materialize_no_dim_list(
                    name, value, defining_scope.constraints.get(name, {}), line_number)
                if not preserve_freshness:
                    value = defining_scope._apply_keyed_not_null(value)
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
        # Preserve unit from existing Input/Let/For/: definition (e.g. Input a of m + Output a)
        existing = self.constraints.get(name) or self.constraints.get(name.lower()) or {}
        if 'unit' not in constraints and 'unit' in existing:
            constraints['unit'] = existing['unit']
        # Also preserve other unit-related constraints
        if 'not_unit' not in constraints and 'not_unit' in existing:
            constraints['not_unit'] = existing['not_unit']
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
        # For `new Type with (f1=e1, f2=e2)` the field NAMES inside the payload
        # are not variable references; only the value sides are. Reuse the
        # compiler's dependency extraction so e.g. `new P2 with (x=9, y=7)` is
        # not treated as depending on variables named x or y. The leading type
        # name in a construction is likewise not a variable reference.
        dep_text = expr_text
        splitter = getattr(self.compiler, '_split_new_with_expr', None)
        if splitter is not None:
            try:
                base_expr, with_text = splitter(expr_text)
            except Exception:
                base_expr, with_text = expr_text, None
            if base_expr is None:
                base_expr = expr_text
            dep_text = re.sub(
                r'^\s*new\s+[A-Za-z][A-Za-z0-9_.]*', '', base_expr)
            if with_text:
                getter = getattr(self.compiler, '_with_deps', None)
                if getter is not None:
                    try:
                        extra += list(getter(with_text))
                    except Exception:
                        pass
        needle = var_name.lower()
        for token in re.finditer(r'[A-Za-z][A-Za-z0-9_.]*', dep_text):
            token_l = token.group(0).lower()
            if token_l == needle or token_l.startswith(needle + '.'):
                return True
        for candidate in set(extra):
            candidate_l = candidate.lower()
            if candidate_l == needle or candidate_l.startswith(needle + '.'):
                return True
        return False

    def _validate_base_type(self, name, value, line_number=None):
        """Validate that a scalar value matches the declared base type.

        Arrays declared with a base type are validated element-by-element
        (see array_handler.validate_array_element_types).  Custom-typed
        objects are validated by custom-type coercion, so only plain scalar
        values fall through to the scalar checks below.
        """
        if value is None:
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
        if constraints.get('input') or constraints.get('output'):
            # Inputs/outputs are validated by their own 'type' constraint or
            # left loosely typed (untyped OUTPUT defaults to 'text').
            return
        if isinstance(value, (list, tuple, dict)):
            if constraints.get('dim'):
                self.compiler.array_handler.validate_array_element_types(
                    name, value, var_type, line_number)
            return
        if constraints.get('dim'):
            return
        actual_type = self.compiler.array_handler.infer_type(
            value, line_number)
        if var_type == 'number' and actual_type not in ('number', 'float64', 'int', 'int64'):
            raise ConstraintError(
                TYPE_ERROR,
                f"'{name}' must be a number, got {actual_type} at line {line_number}")
        if var_type == 'text' and actual_type not in ('string', 'text'):
            raise ConstraintError(
                TYPE_ERROR,
                f"'{name}' must be text, got {actual_type} at line {line_number}")

    def _check_constraints(self, name, value, line_number=None):
        # Case-insensitive constraint lookup
        if is_error_value(value):
            # A sticky error bypasses all constraint/type validation.
            return
        actual_key = self._get_case_insensitive_key(name, self.constraints)
        key_for_constraints = actual_key if actual_key is not None else name
        constraints = self.constraints.get(key_for_constraints, {})

        def _as_numeric(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                raise ConstraintError(
                    TYPE_ERROR,
                    f"'{key_for_constraints}' must be a number, got {type(v).__name__} at line {line_number}")

        self._validate_base_type(key_for_constraints, value, line_number)
        for constraint_type, constraint_expr in constraints.items():
            if constraint_type == 'constant':
                # Skip constant validation if conflict flag is set (lazy check)
                if key_for_constraints in self._conflict_flags:
                    continue
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
                # Unit-aware constant check for Let/For/Input/: with conversion (e.g. Let b of m = 5 of in)
                try:
                    if isinstance(constraint_val, UnitValue) or isinstance(value, UnitValue):
                        c = constraint_val.value if isinstance(constraint_val, UnitValue) else constraint_val
                        c_u = constraint_val.unit if isinstance(constraint_val, UnitValue) else None
                        v = value.value if isinstance(value, UnitValue) else value
                        v_u = value.unit if isinstance(value, UnitValue) else constraints.get('unit')
                        if c_u and v_u and str(c_u).lower() != str(v_u).lower():
                            conv = apply_conversion(c, c_u, str(v_u).lower(), self.compiler.expr_evaluator._formula_eval) if hasattr(self, 'compiler') else None
                            if conv is not None:
                                constraint_val = conv.value
                                value = v
                        elif isinstance(constraint_val, UnitValue):
                            constraint_val = c
                        elif isinstance(value, UnitValue):
                            value = v
                except Exception:
                    pass
                if isinstance(value, (list, dict)):
                    if not is_error_value(value):
                        flat = self.compiler.array_handler.flatten_array(
                            value, line_number)
                        if any(is_error_value(e) for e in flat):
                            continue
                if not constant_value_matches(
                        value, constraint_val,
                        self.compiler.array_handler.to_display_value):
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"Cannot change constant '{key_for_constraints}' at line {line_number}")
            elif constraint_type in ('<=', '>=', '<', '>'):
                constraint_val = _as_numeric(
                    self.compiler.expr_evaluator.eval_or_eval_array(
                        constraint_expr, self.get_full_scope(), line_number))
                num_value = _as_numeric(value)
                if constraint_type == '<=' and num_value > constraint_val:
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"'{key_for_constraints}' exceeds maximum {constraint_val} at line {line_number}")
                elif constraint_type == '>=' and num_value < constraint_val:
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"'{key_for_constraints}' is below minimum {constraint_val} at line {line_number}")
                elif constraint_type == '<' and num_value >= constraint_val:
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"'{key_for_constraints}' is not less than {constraint_val} at line {line_number}")
                elif constraint_type == '>' and num_value <= constraint_val:
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"'{key_for_constraints}' is not greater than {constraint_val} at line {line_number}")
            elif constraint_type == '<>':
                constraint_val = self.compiler.expr_evaluator.eval_or_eval_array(
                    constraint_expr, self.get_full_scope(), line_number)
                if isinstance(value, (list, tuple, set)):
                    if constraint_val in value:
                        raise ConstraintError(
                            VALUE_ERROR,
                            f"'{key_for_constraints}' contains disallowed value {constraint_val} at line {line_number}")
                elif value == constraint_val:
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"'{key_for_constraints}' must not equal {constraint_val} at line {line_number}")
            elif constraint_type.startswith('not_') and constraint_type[4:] in ('<=', '>=', '<', '>'):
                op = constraint_type[4:]
                constraint_val = _as_numeric(
                    self.compiler.expr_evaluator.eval_or_eval_array(
                        constraint_expr, self.get_full_scope(), line_number))
                num_value = _as_numeric(value)
                if op == '<' and num_value < constraint_val:
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"'{key_for_constraints}' must not be less than {constraint_val} at line {line_number}")
                elif op == '<=' and num_value <= constraint_val:
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"'{key_for_constraints}' must be greater than {constraint_val} at line {line_number}")
                elif op == '>' and num_value > constraint_val:
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"'{key_for_constraints}' must not be greater than {constraint_val} at line {line_number}")
                elif op == '>=' and num_value >= constraint_val:
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"'{key_for_constraints}' must be less than {constraint_val} at line {line_number}")
            elif constraint_type == 'in':
                allowed_values = constraint_expr
                if isinstance(constraint_expr, str):
                    try:
                        allowed_values = self.compiler.expr_evaluator.eval_or_eval_array(
                            constraint_expr, self.get_full_scope(), line_number)
                    except Exception:
                        allowed_values = constraint_expr
                if isinstance(allowed_values, dict) and 'array' in allowed_values:
                    allowed_values = list(allowed_values['array'])
                elif is_sparse_array(allowed_values):
                    allowed_values = [allowed_values[k]
                                      for k in sorted(allowed_values.keys())]
                if isinstance(allowed_values, str):
                    allowed_values = [allowed_values]
                if isinstance(value, (list, tuple, set)):
                    if not all(item in allowed_values for item in value):
                        raise ConstraintError(
                            VALUE_ERROR,
                            f"'{key_for_constraints}' values {value} not in allowed values {allowed_values} at line {line_number}")
                elif value not in allowed_values:
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"'{key_for_constraints}' value {value} not in allowed values {allowed_values} at line {line_number}")
            elif constraint_type == 'range':
                start_expr = constraint_expr.get('start')
                end_expr = constraint_expr.get('end')
                step_expr = constraint_expr.get('step')
                start_val = _as_numeric(self.compiler.expr_evaluator.eval_or_eval_array(
                    start_expr, self.get_full_scope(), line_number))
                end_val = _as_numeric(self.compiler.expr_evaluator.eval_or_eval_array(
                    end_expr, self.get_full_scope(), line_number))
                val = _as_numeric(value)
                if not (start_val <= val <= end_val):
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"'{key_for_constraints}' value {value} not in range {start_val} to {end_val} at line {line_number}")
                if step_expr is not None:
                    step_val = _as_numeric(self.compiler.expr_evaluator.eval_or_eval_array(
                        step_expr, self.get_full_scope(), line_number))
                    if step_val == 0:
                        raise ConstraintError(
                            VALUE_ERROR,
                            f"'{key_for_constraints}' range step cannot be 0 at line {line_number}")
                    steps = (val - start_val) / step_val
                    if abs(steps - round(steps)) > 1e-9:
                        raise ConstraintError(
                            VALUE_ERROR,
                            f"'{key_for_constraints}' value {value} not aligned to step {step_val} starting at {start_val} at line {line_number}")
                else:
                    if start_val.is_integer() and end_val.is_integer():
                        if not val.is_integer():
                            raise ConstraintError(
                                VALUE_ERROR,
                                f"'{key_for_constraints}' value {value} must be an integer in range {start_val} to {end_val} at line {line_number}")
            elif constraint_type == 'not_null':
                if value is None:
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"'{key_for_constraints}' must not be null at line {line_number}")
            elif constraint_type == 'null':
                if value is not None:
                    raise ConstraintError(
                        VALUE_ERROR,
                        f"'{key_for_constraints}' must be null at line {line_number}")
            elif constraint_type == 'type':
                expected_type = constraint_expr.lower()
                if isinstance(value, dict) and '_type_name' in value:
                    # Custom-type instances bypass scalar base-type checks.
                    continue
                actual_type = self.compiler.array_handler.infer_type(
                    value, line_number)
                if expected_type == 'number' and actual_type not in ('number', 'float64', 'int', 'int64'):
                    raise ConstraintError(
                        TYPE_ERROR,
                        f"'{key_for_constraints}' must be a number, got {actual_type} at line {line_number}")
                elif expected_type == 'text' and actual_type not in ('string', 'text'):
                    raise ConstraintError(
                        TYPE_ERROR,
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
                    raise ConstraintError(
                        TYPE_ERROR,
                        f"'{key_for_constraints}' must be one of {sorted(allowed)} at line {line_number}")
            elif constraint_type == 'not_type':
                expected_type = constraint_expr.lower()
                actual_type = self.compiler.array_handler.infer_type(
                    value, line_number)
                if expected_type == 'number' and actual_type in ('number', 'float64', 'int', 'int64'):
                    raise ConstraintError(
                        TYPE_ERROR,
                        f"'{key_for_constraints}' must not be a number at line {line_number}")
                elif expected_type == 'text' and actual_type in ('string', 'text'):
                    raise ConstraintError(
                        TYPE_ERROR,
                        f"'{key_for_constraints}' must not be text at line {line_number}")
            elif constraint_type == 'not_unit':
                if isinstance(value, str) and value == constraint_expr:
                    raise ConstraintError(
                        UNIT_ERROR,
                        f"'{key_for_constraints}' must not be unit {constraint_expr} at line {line_number}")

    def get_full_scope(self):
        full_scope = {}
        chain = []
        current = self
        while current and not current.is_private:
            chain.append(current)
            current = current.parent
        for scope in reversed(chain):
            for var_name, var_value in scope.variables.items():
                full_scope[var_name] = scope._wrap_for_eval(var_name, var_value)
        return full_scope
