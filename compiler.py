import csv
import os
import re
import math
import sys
import copy
from expression import ExpressionEvaluator
from array_handler import ArrayHandler
from utils import col_to_num, split_cell, offset_cell, validate_cell_ref, object_public_keys, public_object_view, format_display_value, iter_interpolation_placeholders, is_address, parse_address, indices_to_address, _ADDRESS_FRAGMENT, is_sparse_array, strip_array_cell_indices
from scope import Scope, _GridStore
from units import (
    UNIT_ERROR, UNIVERSAL_ZERO, TYPE_ERROR, VALUE_ERROR, UnitValue, ConstraintError, error_value,
    is_error_value, strip_units, register_conversion, has_conversion,
    lookup_conversions,
)
from control_flow import GridLangControlFlow
from type_processor import GridLangTypeProcessor, split_builder_chain
from parser import GridLangParser
from executor import GridLangExecutor
from grid_lang_common import GridLangBase


from grid_lang_common import (
    _STATEMENT_KEYWORDS, _first_keyword,
    _IDENTIFIER_TOKEN_PATTERN, _STRING_LITERAL_PATTERN, _DEPENDENCY_IGNORED_TOKENS,
    _strip_constraint_operands, _strip_builder_arrows, _strip_cell_address_tokens,
    _is_numeric_token, mask_text_constant_tokens,
)
from builtin_functions import BUILTINS, KEYWORDS, RESOURCES
from modules import (
    ModuleImportError, parse_module_header, parse_module_version_line,
    parse_version_block, parse_use_line, strip_version_tag,
    version_pin_matches, resolve_module_source,
)


class SubprocessResult:
    """Container for subprocess execution results."""

    def __init__(self, grid=None, variables=None, outputs=None):
        self.grid = grid
        self._variables = variables or {}
        self.outputs = outputs or {}

    def __getattr__(self, item):
        if item in self._variables:
            return self._variables[item]
        raise AttributeError(
            f"Attribute '{item}' not found in subprocess result")


class _UnitSourceNamespace:
    """Attribute-access namespace for a UnitSource block's materialized
    constants, so member reads like ``SILength.inch`` resolve in the AST
    walker (``getattr`` on the block object)."""

    __slots__ = ('_fields',)

    def __init__(self, fields):
        object.__setattr__(self, '_fields', dict(fields))

    def __getattr__(self, name):
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, name):
        return self._fields[name]


# Predefined standard-library resources come from builtin_functions.py
# (RESOURCES, registered via @register_resource). They are seeded into the
# engine at startup and can never be redefined by a program (``Define X as
# Resource`` for one of them is rejected). They rely on the normal Resource
# machinery, so their field constraints/member sets look exactly like a
# body-declared resource.


# Predefined control subprocesses for the standard-library Ticker resource. The
# resource name is the namespace (``Ticker.Stop``, ``Ticker.Start``); their
# behavior is engine-built, so they carry a ``_system`` marker instead of a
# body. A program cannot redefine them.
_PREDEFINED_SUBPROCESSES = {
    'ticker.stop': {'_system': 'ticker.stop', 'inputs': [], 'outputs': []},
    'ticker.start': {'_system': 'ticker.start', 'inputs': [], 'outputs': []},
}

# Dotted capability handle-creators the engine seeds into its callable
# registry (number.*, text.*, ticker.timer, ticker.counter, ...). They are
# engine-owned facilities exactly like the system subprocesses above: a
# program cannot redefine them. Derived from the authoritative BUILTINS
# registry so it can never drift from what the engine actually installs.
_DOTTED_ENGINE_CREATORS = frozenset(
    k.lower() for k in BUILTINS if '.' in k)

# Inline block statements with a single-instruction payload. They are rewritten
# into real block form during preprocessing so the standard block machinery
# handles them: ``For n in 1 to 3 do push x = n`` becomes a For block containing
# a Push, and ``If a > 2 then return a else return b`` becomes an If block.
_INLINE_BLOCK_LOOP_RE = re.compile(
    r'^\s*(for|when)\b(.+?)\bdo\b\s+(.+)$', re.I)
_INLINE_BLOCK_IF_RE = re.compile(
    r'^\s*(elseif|if)\b(.+?)\bthen\b\s+(.+)$', re.I)
_INLINE_ELSE_ACTION_RE = re.compile(r'^\s*else\s+(.+)$', re.I)
_INLINE_IF_CLAUSE_SPLIT_RE = re.compile(r'\b(elseif|else)\b', re.I)
_INLINE_IF_THEN_SPLIT_RE = re.compile(r'\b(?:then)\b', re.I)


class GridLangCompiler(GridLangExecutor):
    def __init__(self):
        super().__init__()
        # The merged engine is both compiler and executor: executor methods that
        # reference self.compiler (previously the owning compiler of a copied
        # executor) resolve back to this same object.
        self.compiler = self
        self.scopes = [Scope(self)]
        # Predefine the 'grid' variable containing the current grid, like in a
        # type definition, so programs can read and write cells with grid{row, col}.
        # ``self.grid`` aliases the seeded store (see _seed_grid_variable).
        self._seed_grid_variable()
        self.variables = self.current_scope().variables
        self.types = self.scopes[0].types
        # Client/publisher listener registry. Cells and variables written by a
        # client binding (: v = expr) register listeners; publisher updates
        # (init/push) propagate to those listeners immediately.
        self._listeners = {'cell': {}, 'var': {}}
        self._set_by = {}
        self._propagating = set()
        self._singletons = {}
        self.dimensions = {}
        self.dim_names = {}
        self.dim_labels = {}
        self.pending_assignments = {}
        self.deferred_lines = []
        self._cell_var_map = {}
        self._cell_array_map = {}
        self.types_defined = {}
        self._seed_predefined_resources()
        self.subprocesses = {}
        self._seed_predefined_subprocesses()
        self.unit_sources = {}
        self._top_level_converts = []
        self.expr_evaluator = ExpressionEvaluator(self)
        self.array_handler = ArrayHandler(self)
        self.control_flow = GridLangControlFlow(self)
        self.type_processor = GridLangTypeProcessor(self)
        self.parser = GridLangParser(self)
        self.handled_assignments = set()
        # Grid language features
        self.input_variables = []  # Ordered list of input variables
        self.output_variables = []  # List of output variables
        self.output_values = {}
        self._allow_hidden_field_access = False
        self._allow_hidden_member_calls = False
        self.undefined_dependencies = set()
        self.dependency_graph = {'nodes': [], 'by_variable': {}, 'by_line': {}}
        self.global_guard_entries = []
        self.global_for_line_numbers = set()
        self.executed_global_for_lines = set()
        self.global_for_entries = []
        # Control whether missing inputs should prompt the user (CLI mode only)
        self.prompt_missing_inputs = False
        # =========================================================================
        # Required-capability (permission) state. `Require name as Resource` binds
        # a capability handle: granted at startup it holds the resource parameters
        # the user chose; denied it holds the sticky #PERM error value.
        # =========================================================================
        # Ordered list of requirement entries collected from the program.
        self.requirements = []
        # name(lower) -> requirement entry for quick lookup.
        self.require_caps = {}
        # name(lower) -> granted parameter dict, or None when denied.
        # Populated from --grant / interactive prompts before execution.
        self.grants = {}
        # Control whether missing requirements should prompt the user (CLI mode)
        self.prompt_missing_requires = False
        # When True, `run` halts right after preparation (no main loop, no
        # outputs) so the host can inspect required capabilities (--list-required).
        self.halt_before_main_loop = False
        # ------------------------------------------------------------------
        # Module import state. `module_sources` is host input that survives the
        # per-run reset (like `grants`); the run-scoped sets below are cleared
        # in `_reset_state`.
        # ------------------------------------------------------------------
        # name(lower) -> module source text provided by the host. Falls back to
        # filesystem resolution (see modules.resolve_module_source) when empty.
        self.module_sources = {}
        # Namespace names introduced by `For <ns> use <module>.<tag>`: tokens
        # like `B` resolve through here instead of the variable scope.
        self.module_namespaces = set()
        # Saved harvested-from-module tables so repeated `use` of a module in
        # one program shares definitions (per (importer x physical copy)).
        self._module_harvests = {}
        # Line numbers of `use`/`For ... use` statements processed during
        # declaration collection; skipped in the main loop.
        self.module_use_line_numbers = set()
        # Live module instances (per importer x physical copy): key ->
        # {'scope': <module state scope>, 'module_vars': {name: meta},
        #  'exported': {program_key: instance_var}}.
        self.module_instances = {}
        # module_key -> {program_key: instance_var}; refreshed after a module
        # subprocess call so the importer's bound copies observe mutations.
        self._module_export_bindings = {}

    def _get_public_type_fields(self, type_name_or_def):
        """Return the declared fields for a type, excluding internal metadata keys."""
        from utils import public_type_fields
        if isinstance(type_name_or_def, dict):
            return public_type_fields(type_name_or_def)
        return public_type_fields(self.types_defined.get(type_name_or_def.lower(), {}))

    def _get_all_type_fields(self, type_name_or_def):
        """Return all declared fields for a type, including hidden ones."""
        if isinstance(type_name_or_def, dict):
            return {k: v for k, v in type_name_or_def.items() if not str(k).startswith('_')}
        type_def = self.types_defined.get(type_name_or_def.lower(), {})
        return {k: v for k, v in type_def.items() if not str(k).startswith('_')}

    def _is_hidden_field(self, obj, field_name):
        if not isinstance(obj, dict):
            return False
        hidden = obj.get('_hidden_fields', set())
        if not isinstance(hidden, (set, list, tuple)):
            hidden = set()
        return str(field_name).lower() in {str(h).lower() for h in hidden}

    def _parse_type_header(self, line, line_number=None):
        """Parse a type definition header, returning name, parent, and constraints.

        Supports `Define X as Type`, `Define X as Type(Parent)`,
        `Define X as Keytype` / `Define X as Keytype(Parent)`,
        `Define X as Resource` (a capability that can only be acquired
        with `Require`, never instantiated with `new`), and
        `Define MyRes.Inc as Handle` (a handle type derived from Resource
        `MyRes`, instantiated via a Resource member function
        ``MyRes.Inc``; the canonical type name ``MyRes!Inc`` may contain
        ``!`` to denote Resource!Handle nesting, e.g. ``Input h as MyRes!Inc``).
        """
        m = re.match(
            r'^\s*define\s+([\w_!\.]+)\s+as\s+(type|keytype|resource|handle)(?:\s*\(\s*([^)]*)\s*\))?\s*(.*)$', line, re.I)
        if not m:
            return None, None, None
        type_name = m.group(1).strip()
        kind = m.group(2).strip().lower()  # type, keytype, resource, or handle
        inner = m.group(3).strip() if m.group(3) else ""
        remainder = m.group(4).strip()
        parent = None
        keyed = False
        if kind == "keytype":
            keyed = True
            if inner:
                # Keytype(Parent) -> parent is inner
                parent = inner.split()[0].strip()
        elif kind == "resource":
            if inner:
                raise SyntaxError(
                    f"Resource definitions take no parent at line {line_number}: '{line}'")
        elif kind == "handle":
            if '.' not in type_name:
                raise SyntaxError(
                    f"Handle definitions require a resource name: 'Define Resource.Handle as Handle' at line {line_number}: '{line}'")
            parent = type_name.split('.', 1)[0].strip()
        else:
            # kind == type, only Type or Type(Parent) - Type(key) removed, use Keytype
            if inner and "key" in inner.lower().split():
                raise SyntaxError(f"Type(key) syntax removed, use Keytype at line {line_number}: '{line}'")
            if inner:
                parent = inner.split()[0].strip()
        constraints = {}
        if remainder:
            try:
                _, _, constraints, _ = self.parser._parse_variable_def(
                    f"_type {remainder}", line_number)
            except Exception as exc:
                raise SyntaxError(
                    f"Invalid type constraints in '{line}' at line {line_number}: {exc}")
        if keyed:
            constraints['key'] = True
        if kind == "resource":
            constraints['is_resource'] = True
        if kind == "handle":
            constraints['is_handle'] = True
            # parent variable holds the resource name for handles; persist it in
            # constraints and clear the inheritance parent.
            resource_for_handle = parent
            if resource_for_handle:
                constraints['handle_resource'] = resource_for_handle
            parent = None
        return type_name, parent, constraints

    def _parse_unit_source_header(self, line, line_number=None):
        """Parse a ``Define X as UnitSource of Target`` header.

        Returns ``(name, target_unit)`` or ``(None, None)`` when the line is
        not a UnitSource definition.
        """
        m = re.match(
            r'^\s*define\s+([\w_]+)\s+as\s+unitsource'
            r'(?:\s+of\s+(\w+))\s*$',
            line, re.I)
        if not m:
            return None, None
        return m.group(1).strip(), m.group(2).strip()

    def _finalize_unit_source(self, body_lines, name, target_unit, line_number=None):
        """Parse and register the Convert/constant lines of a UnitSource block.

        Each ``Convert`` is registered into the global conversion registry
        (targeting *target_unit*). Each ``: field as ... of <T> = <rhs>`` is
        recorded as a constant field (with declared unit *T*) exposed as
        ``<name>.<field>`` and materialized at runtime.
        """
        source = self.unit_sources.setdefault(name.lower(), {
            '_target_unit': target_unit,
            '_constants': {},
            '_orig_name': name,
        })
        source['_target_unit'] = target_unit
        source['_orig_name'] = name
        for raw in body_lines:
            line = raw.lstrip()
            if not line.strip():
                continue
            stripped = line.strip()
            # Constant field: ": field [as type] of <T> = <rhs>" (leading ':')
            if stripped.startswith(':'):
                field_line = stripped[1:].strip()
                field_match = re.match(
                    r'^([\w_]+)(?:\s+as\s+\w+)?\s+of\s+(\w+)\s*=\s*(.+)$',
                    field_line, re.I)
                if field_match:
                    field_name, field_unit, field_expr = field_match.groups()
                    source['_constants'][field_name.lower()] = {
                        'unit': field_unit, 'expr': field_expr,
                        'name': name, 'field': field_name,
                    }
                    continue
            self._register_convert_line(stripped, target_unit, line_number)

    def _register_convert_line(self, raw, target_unit, line_number=None, var_name=None):
        """Parse a single ``Convert <p1> to <val>`` line and register it.

        ``p1`` is either a constant (``"ox" of animal``) producing a category
        mapping, or a new variable with a source unit (``x as number of cm``)
        producing a formula conversion. *target_unit* may be None for a
        top-level Convert, in which case it is inferred from the RHS value's
        unit when possible.
        """
        line = raw.strip()
        m = re.match(
            r'^\s*convert\s+(.+?)\s+to\s+(.+)$', line, re.I)
        if not m:
            return
        source_spec, dest_expr = m.group(1).strip(), m.group(2).strip()

        # Constant mapping: "<value>" of <unit> -> <dest>
        cst = re.match(
            r'^("[^"]*"|\'[^\']*\'|[\d.eE+-]+)\s+of\s+(\w+)$',
            source_spec, re.I)
        if cst:
            raw_lit = cst.group(1)
            src_unit = cst.group(2)
            if raw_lit.startswith(('"', "'")):
                src_value = raw_lit[1:-1]
            else:
                try:
                    src_value = float(raw_lit)
                except ValueError:
                    src_value = raw_lit
            if target_unit is None:
                return
            register_conversion(src_unit, target_unit, {
                'kind': 'constant', 'src': src_value, 'dst': dest_expr.strip('"').strip("'"),
            })
            return

        # Formula: "x [as type] of <unit>" -> <dest expr>
        form = re.match(
            r'^([\w_]+)(?:\s+as\s+\w+)?\s+of\s+(\w+)$', source_spec, re.I)
        if not form:
            return
        var_name = form.group(1)
        src_unit = form.group(2)
        if target_unit is None:
            target_unit = self._infer_convert_target_unit(
                dest_expr, var_name, line_number)
            if target_unit is None:
                return
        register_conversion(src_unit, target_unit, {
            'kind': 'formula', 'var': var_name, 'expr': dest_expr,
            'target': target_unit,
        })

    def _infer_convert_target_unit(self, dest_expr, var_name, line_number=None):
        """Infer the target unit of a top-level Convert by evaluating its RHS.

        The RHS is evaluated with the formula's parameter bound to a plain
        number (unitless, so it composes cleanly with any unit-bearing factor
        such as ``x * SILength.inch``); the resulting value's unit is the
        conversion target. Falls back to declared units of RHS variables for
        Input/Let/For/:/Push/Init when evaluation needs values not yet in scope.
        Returns None when the RHS cannot be evaluated or carries no unit.
        """
        try:
            scope = self.current_scope()
            eval_scope = dict(scope.get_evaluation_scope())
        except Exception:
            return None
        eval_scope[str(var_name)] = 1.0
        try:
            result = self.expr_evaluator.eval_or_eval_array(
                dest_expr, eval_scope, line_number)
            unit = getattr(result, 'unit', None)
            if unit is not None:
                return unit
        except Exception:
            pass
        # Fallback: look at declared units of variables in RHS (for not-yet-bound Input/Let/For/:/Push/Init)
        try:
            for tok in re.findall(r'[A-Za-z][A-Za-z0-9_.]*', dest_expr):
                base = tok.split('.')[0].split('[')[0]
                if base.lower() == var_name.lower():
                    continue
                u = None
                try:
                    u = scope.get_value_unit(base)
                except Exception:
                    pass
                if u is None:
                    try:
                        c = scope.constraints.get(base) or scope.constraints.get(base.lower()) or {}
                        u = c.get('unit')
                    except Exception:
                        pass
                if u is not None:
                    return u
        except Exception:
            pass
        return None

    def _register_top_level_converts(self):
        """Register top-level Convert lines once the scope and UnitSource
        constants are available (so RHS target units can be inferred)."""
        for raw, line_number in getattr(self, '_top_level_converts', []):
            self._register_convert_line(raw, None, line_number)

    def _materialize_unit_source_constants(self):
        """Inject UnitSource constant values into the current scope.

        Each constant's RHS is evaluated (so ``2.54 of cm`` with a cm->m
        conversion becomes ``0.0254 of m``) and stored as a scope variable
        named ``<block>.<field>`` so member reads resolve at evaluation time.
        A namespace object for the block (e.g. ``SILength``) is also defined
        so ``SILength.inch`` attribute access resolves in the AST walker.
        """
        scope = self.current_scope()
        for blk, source in self.unit_sources.items():
            namespace = {}
            for field, const in list(source.get('_constants', {}).items()):
                if 'value' in const:
                    namespace[field] = const.get('value')
                    continue
                expr = const.get('expr')
                unit = const.get('unit')
                value = None
                try:
                    value = self.expr_evaluator.eval_or_eval_array(
                        expr, scope.get_evaluation_scope(), None,
                        expected_unit=unit)
                except Exception:
                    value = None
                if value is not None and not is_error_value(value):
                    if isinstance(value, UnitValue) and value.unit is None:
                        value = UnitValue(value.value, unit)
                    elif not isinstance(value, UnitValue):
                        value = UnitValue(value, unit)
                    const['value'] = value
                    namespace[field] = value
            for field, value in namespace.items():
                const = source.get('_constants', {}).get(field)
                var_name = f"{const['name']}.{const['field']}"
                scope.define(var_name, value, None, {'constant': const.get('expr')},
                             line_number=None)
            if namespace:
                ns = _UnitSourceNamespace(namespace)
                # Use original block name (preserving case) for the namespace
                # so 'SILength.inch' attribute access resolves (case-insensitive
                # fallback also handles mixed case).
                orig = source.get('_orig_name', blk)
                for key in {blk, orig, orig.lower(), orig.upper()}:
                    if not scope.get_defining_scope(key):
                        scope.define(key, ns, None, {'unit_source': blk},
                                     line_number=None)
                        break
        return scope

    def _resolve_type_inheritance(self):
        """Merge inherited fields and constraints into child type definitions."""
        primitives = {'number', 'text', 'logical'}
        progress = True
        while progress:
            progress = False
            for type_name, type_def in list(self.types_defined.items()):
                parent = type_def.get('_parent')
                if not parent or type_def.get('_inheritance_applied'):
                    continue
                parent_lower = parent.lower()
                parent_def = self.types_defined.get(parent_lower)
                if parent_lower in primitives:
                    base_type = parent_lower
                elif parent_def:
                    base_type = parent_def.get('_base_type')
                else:
                    continue

                if parent_def:
                    parent_fields = {k: v for k, v in parent_def.items()
                                     if not str(k).startswith('_')}
                    # Rebuild public field order: parent fields first, then child fields.
                    merged_fields = {}
                    for field, f_type in parent_fields.items():
                        merged_fields[field] = f_type
                    for field, f_type in type_def.items():
                        if str(field).startswith('_'):
                            continue
                        merged_fields[field] = f_type
                    internal_items = {k: v for k, v in type_def.items()
                                      if str(k).startswith('_')}
                    type_def.clear()
                    type_def.update(merged_fields)
                    type_def.update(internal_items)

                    hidden = set(type_def.get('_hidden_fields', set()))
                    parent_hidden = parent_def.get('_hidden_fields', set())
                    if isinstance(parent_hidden, (set, list, tuple)):
                        hidden.update(parent_hidden)
                    if hidden:
                        type_def['_hidden_fields'] = hidden

                    parent_constraints = parent_def.get('_constraints', {}) or {}
                    child_constraints = type_def.get('_constraints', {}) or {}
                    if parent_constraints or child_constraints:
                        merged = dict(parent_constraints)
                        merged.update(child_constraints)
                        type_def['_constraints'] = merged

                    parent_field_constraints = parent_def.get('_field_constraints', {}) or {}
                    child_field_constraints = type_def.get('_field_constraints', {}) or {}
                    if parent_field_constraints or child_field_constraints:
                        merged_fields = dict(parent_field_constraints)
                        merged_fields.update(child_field_constraints)
                        type_def['_field_constraints'] = merged_fields

                    # Option 2: subclasses of a Keytype are keyed (flexible)
                    if parent_def.get('_keyed'):
                        type_def['_keyed'] = True

                if base_type:
                    type_def['_base_type'] = base_type
                type_def['_inheritance_applied'] = True
                progress = True

    def _is_type_compatible(self, actual_type, expected_type):
        if not actual_type or not expected_type:
            return False
        actual = actual_type.lower()
        expected = expected_type.lower()
        if actual == expected:
            return True
        if '.' in actual and actual.rsplit('.', 1)[1] == expected:
            return True
        visited = set()
        while actual and actual not in visited:
            visited.add(actual)
            t_def = self.types_defined.get(actual)
            parent = t_def.get('_parent') if isinstance(t_def, dict) else None
            if not parent:
                break
            actual = parent.lower()
            if actual == expected:
                return True
        return False

    def _resolve_member_function(self, obj_type, method_name):
        if not obj_type:
            return None
        current = obj_type.lower()
        visited = set()
        while current and current not in visited:
            visited.add(current)
            func_key = f"{current}.{method_name}".lower()
            if func_key in getattr(self, 'functions', {}):
                return func_key
            t_def = self.types_defined.get(current)
            parent = t_def.get('_parent') if isinstance(t_def, dict) else None
            if not parent:
                break
            current = parent.lower()
        return None

    def _copy_instance(self, source, line_number=None):
        """Deep-copy an object instance for `new Copy(Obj)`.

        No constructor is run. Listeners internal/external are cloned so that
        a field `b = a + 1` stays internal (`dst.b -> dst.a`) and
        `b = global + 1` stays external (`dst.b -> global`). Keyed fields are
        nulled so the builder must assign them (Not Null).
        """
        import copy
        if not isinstance(source, dict):
            raise TypeError(f"Copy source must be an object instance at line {line_number}")
        # A stored (non-fresh) keytype instance cannot be copied at all.
        src_key_type = self._key_type_of_value(source)
        if src_key_type and not self._key_value_is_fresh(source):
            raise ConstraintError(TYPE_ERROR, f"Cannot copy instance of key type '{src_key_type}' at line {line_number}")
        dst = copy.deepcopy(source)
        dst.pop('_with_applied_fields', None)
        dst.pop('_with_conflict', None)
        type_name = dst.get('_type_name')
        if type_name:
            tdef = self.types_defined.get(type_name.lower(), {})
            # Null out key fields for keyed types (defer singleton: Type(key) with no fields)
            # Type-level key: null all fields that are declared as keyed or whose type is keyed
            if tdef.get('_keyed'):
                # For Type(key) with fields, null all fields? Specification: Define A as Type(key) means all instances distinct - likely whole object keyed, so builder must init something? For now null out fields that are keyed via field type.
                pass
            # Field-level key: null keyed fields (Not Null) so a following builder must assign them
            for field in self._keyed_field_names(tdef):
                for key in list(dst.keys()):
                    if key.lower() == field:
                        dst[key] = None
        # Clone listeners (internal/external)
        # self._clone_listeners_for_copy is called by caller when source/dest var names are known (global decl)
        # For templated expression copies, listeners will be cloned when the copy is bound to a variable
        return dst

    def _clone_listeners_for_copy(self, src_var, dst_var, src_scope=None, dst_scope=None):
        """Clone listener edges from src_var to dst_var.

        Internal `src.a -> src.b` becomes `dst.a -> dst.b`.
        External `src.a -> global` becomes `dst.a -> global`.
        No ctor is run, so this preserves arrangement.
        """
        if not src_var or not dst_var:
            return
        src_lower = src_var.lower()
        dst_lower = dst_var.lower()
        # Walk current _listeners table; duplicate records where var == src_var or var starts with src_var.
        for kind in ('var', 'cell'):
            for dep_key, holders in list(self._listeners.get(kind, {}).items()):
                for holder_key, rec in list(holders.items()):
                    var_name = rec.get('var')
                    if not var_name:
                        continue
                    # Field of src object: var_name == src_var or var_name startswith src_var + '.'
                    if var_name.lower() == src_lower or var_name.lower().startswith(src_lower + '.'):
                        suffix = var_name[len(src_var):]  # includes dot if present
                        new_var = dst_var + suffix
                        # Determine new dep keys: remap internal deps (src.* -> dst.*)
                        new_deps = set()
                        for dep in rec.get('deps', ()):
                            if dep.lower() == src_lower or dep.lower().startswith(src_lower + '.'):
                                new_dep = dst_var + dep[len(src_var):]
                                new_deps.add(new_dep)
                            else:
                                new_deps.add(dep)
                        # Re-register with same scope (dst_scope or rec scope)
                        scope = dst_scope if dst_scope is not None else rec.get('scope')
                        # Use original expr but with src->dst substitution for internal refs? Keep expr as is for now; recompute will resolve via new var
                        # Register under same dep keys but with dst var
                        for dep in new_deps:
                            # Skip dependency tokens that are types/functions
                            if dep.lower() in self.types_defined or dep.lower() in getattr(self, 'functions', {}) or dep.lower() in getattr(self, 'subprocesses', {}):
                                continue
                            key = ('var', dep.lower()) if not re.match(r'^[A-Za-z]+\d+$', dep) else ('cell', dep)
                            # For cell deps keep as is
                            if key[0] == 'cell':
                                key = ('cell', dep)
                            holder2 = self._listeners.setdefault(key[0], {}).setdefault(key[1], {})
                            # Use new_var as key
                            new_rec = dict(rec)
                            new_rec['var'] = new_var
                            new_rec['deps'] = new_deps
                            # scope stays same (dst scope)
                            new_rec['scope'] = scope
                            holder2[(new_var.lower(), id(scope) if scope else 0)] = new_rec
                        if new_var:
                            self._set_by[('var', new_var.lower())] = 'client'

    def _is_key_type(self, type_name):
        """Return True if type_name is a Keytype (declared ``as Keytype``) whose
        instances cannot be copied once stored."""
        if not type_name:
            return False
        tdef = self.types_defined.get(str(type_name).lower(), {})
        return bool(tdef and tdef.get('_keyed'))

    def _key_type_of_value(self, value):
        """Return the keytype name if ``value`` is a keytype instance (primitive
        ``UnitValue`` or keyed object dict), else None."""
        if isinstance(value, UnitValue) and getattr(value, 'key_type', None):
            return value.key_type
        if isinstance(value, dict):
            tn = value.get('_type_name')
            if tn and self._is_key_type(tn):
                return tn
        return None

    def _key_value_is_fresh(self, value):
        """Return True if ``value`` is a freshly-constructed keytype instance
        (allowed to flow to its first binding) rather than a stored one."""
        if isinstance(value, UnitValue) and getattr(value, 'key_type', None):
            return bool(getattr(value, 'fresh_key', False))
        if isinstance(value, dict):
            return bool(value.get('_fresh_key', False))
        return False

    def _mark_keytype_stored(self, value):
        """Recursively clear the fresh marker on keytype instances nested in a
        stored value (a Let/For/Push/`:` variable or a stored object field)."""
        if isinstance(value, UnitValue) and getattr(value, 'key_type', None):
            if getattr(value, 'fresh_key', False):
                value.fresh_key = False
            return
        if isinstance(value, dict):
            if self._is_key_type(value.get('_type_name')):
                value['_fresh_key'] = False
            for k in list(value.keys()):
                if str(k).startswith('_'):
                    continue
                self._mark_keytype_stored(value[k])
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._mark_keytype_stored(item)

    def _check_key_type_copy(self, dest_var, dest_type, src_type, source_value=None, src_scope=None, line_number=None):
        """Raise if a non-fresh key type instance is being copied.

        Called after the RHS is evaluated: ``src_type`` is the keytype name
        derived from the evaluated value. ``new Copy(x)``, ``For d as L = x``,
        ``Push d = x``, ``Let d = x`` where L is a Keytype and x is an already
        stored keytype instance are errors. Fresh values (``new L(...)``,
        ``new K ...``, wrapping imports) may flow to their first binding.
        ``k as number key`` field values (plain numbers, foreign keys) are not
        keytype instances and remain copyable.
        """
        if not src_type or not self._is_key_type(src_type):
            return
        if source_value is not None and self._key_value_is_fresh(source_value):
            return
        raise ConstraintError(TYPE_ERROR, f"Cannot copy instance of key type '{src_type}' at line {line_number}")

    def _convert_array_to_object(self, type_name, value, line_number=None):
        type_def = self.types_defined.get(type_name.lower())
        if not isinstance(type_def, dict):
            return value
        inputs_list = type_def.get('_inputs', [])
        if inputs_list:
            raise ValueError(
                f"Cannot convert array to type '{type_name}' with constructor inputs at line {line_number}")
        if isinstance(value, dict) and 'array' in value:
            value = list(value['array'])
        if not isinstance(value, list):
            return value
        public_fields = list(self._get_public_type_fields(type_def).keys())
        if len(value) != len(public_fields):
            raise ValueError(
                f"Expected {len(public_fields)} values for type '{type_name}', got {len(value)} at line {line_number}")
        obj = {}
        for field_name, field_val in zip(public_fields, value):
            obj[field_name] = field_val
        obj['_type_name'] = type_name.lower()
        hidden_fields = type_def.get('_hidden_fields', set())
        if hidden_fields:
            obj['_hidden_fields'] = set(hidden_fields)
        # Lazy grid: don't create unless needed
        try:
            self._recompute_computed_fields(obj, line_number=line_number)
        except Exception:
            pass
        return obj

    def _recompute_computed_fields(self, obj, line_number=None, changed_field=None):
        """Recompute computed fields for a custom type instance when dependencies change."""
        if not isinstance(obj, dict):
            return
        type_name = obj.get('_type_name')
        if not type_name:
            return
        type_def = self.types_defined.get(str(type_name).lower(), {})
        computed = type_def.get('_computed_fields') or {}
        if not computed:
            return
        eval_scope = self.type_processor._build_type_eval_scope(obj, {})
        for field_name, expr in computed.items():
            # Avoid self-referential loops for the changed field
            if changed_field and str(field_name).lower() == str(changed_field).lower():
                continue
            try:
                val = self.expr_evaluator.eval_or_eval_array(
                    str(expr), eval_scope, line_number)
            except Exception:
                continue
            obj[field_name] = val
            eval_scope[field_name] = val

    def _is_outer_scope(self, scope):
        """Return True when the given scope belongs to the caller's scope chain."""
        parent = getattr(self, '_parent_scope', None)
        if parent is None:
            return False
        cur = parent
        while cur is not None:
            if cur is scope:
                return True
            cur = cur.parent
        return False

    def run(self, code, args=None, suppress_output=False, return_output=False):
        """Compile and execute a GridLang program.

        The single engine (GridLangCompiler) owns all state and components and
        executes directly on itself; there is no separate executor object and
        no method/state copy handoff.
        """
        from scope import _ACTIVE_RUNNERS

        # Track this compiler as the currently executing context so that
        # read-only function sub-compilers can reject writes to outer scopes
        # routed through the defining scope object.
        _ACTIVE_RUNNERS.append(self)
        try:
            result = super().run(code, args, suppress_output=suppress_output,
                                 return_output=return_output)
        finally:
            _ACTIVE_RUNNERS.pop()

        # Capture the final root-scope variables for callers that need them
        try:
            self._last_scope_vars = self.current_scope().variables.copy()
            self._last_scope_types = self.current_scope().types.copy()
        except Exception:
            self._last_scope_vars = {}
            self._last_scope_types = {}

        # Keep variables reference aligned with the active root scope
        if hasattr(self, 'scopes') and self.scopes:
            self.variables = self.scopes[0].variables

        return result

    def _extract_functions(self, lines, label_lines, dim_lines):
        if not hasattr(self, '_program_defined_names'):
            self._program_defined_names = set()
        """Extract user-defined functions and remove them from main code."""
        functions = getattr(self, 'functions', {}) or {}
        subprocesses = getattr(self, 'subprocesses', {}) or {}
        new_lines = []
        i = 0
        while i < len(lines):
            line, line_number = lines[i]
            m = re.match(
                r'^\s*define\s+(\$?[\w\!\.]+)\s+as\s+(function|subprocess|privatehelper|operation)\b', line, re.I)
            m_builder = re.match(
                r'^\s*define\s+(\$?[\w.]+)\s+as\s+builder\s*\(\s*([A-Za-z][\w]*)\s*\)\s*$', line, re.I)
            if re.search(r'\bas\s+builder\b', line, re.I) and not m_builder:
                raise SyntaxError(
                    f"Invalid Builder definition at line {line_number}: "
                    f"use 'Define $<name> as Builder(<type>)'")
            if m and m.group(2).lower() == 'privatehelper':
                raise SyntaxError(
                    f"'PrivateHelper' cannot be defined at line {line_number}; "
                    f"rename it as a builder: 'Define $<name> as Builder(<type>)'")
            if m_builder:
                raw_name = m_builder.group(1).strip()
                builder_type = m_builder.group(2)
                def_kind = 'builder'
                hidden = raw_name.startswith('$')
                func_name = raw_name[1:] if hidden else raw_name
            elif m:
                raw_name = m.group(1).strip()
                def_kind = m.group(2).lower()
                hidden = raw_name.startswith('$')
                func_name = raw_name[1:] if hidden else raw_name
            else:
                raw_name = None
                def_kind = None
                hidden = False
                func_name = None
            if def_kind:
                body_lines = []
                block_depth = 0
                i += 1
                while i < len(lines):
                    body_line, body_ln = lines[i]
                    stripped = body_line.strip().lower()
                    # Track nested control blocks so generic END inside the body doesn't terminate the function
                    if stripped.startswith(('for ', 'when ', 'if ')):
                        block_depth += 1
                    if stripped.startswith('end'):
                        # Named end takes precedence
                        if re.match(r'^\s*end\s+\$?%s\s*$' % re.escape(func_name), body_line, re.I):
                            break
                        # Allow bare "end" (or "end function") to close the definition when not nested
                        if block_depth == 0 and re.match(r'^\s*end(?:\s+(function|subprocess))?\s*$', stripped, re.I):
                            break
                        block_depth = max(0, block_depth - 1)
                        body_lines.append(body_line)
                        i += 1
                        continue
                    body_lines.append(body_line)
                    i += 1
                func_code = "\n".join(body_lines)
                code_lines = [ln for ln in body_lines
                              if not ln.strip().lower().startswith(('input ', 'output '))]
                inputs, input_defs, outputs = [], [], []
                for b in body_lines:
                    m_in = re.match(r'^\s*input\s+(.+)$', b, re.I)
                    if m_in:
                        try:
                            parsed_var, parsed_type, parsed_constraints, _ = self.parser._parse_variable_def(
                                b.strip(), body_ln)
                        except Exception:
                            parsed_var, parsed_type, parsed_constraints = None, None, {}
                        if parsed_var:
                            var_list = (parsed_constraints or {}).get('var_list') if parsed_constraints else None
                            names = var_list if var_list else [parsed_var]
                            inputs.extend(names)
                            for name in names:
                                input_defs.append({
                                    'name': name,
                                    'type': parsed_type,
                                    'constraints': parsed_constraints or {}
                                })
                    m_out = re.match(r'^\s*output\s+(.+)$', b, re.I)
                    if m_out:
                        try:
                            parsed_var, parsed_type, parsed_constraints, _ = self.parser._parse_variable_def(
                                b.strip(), body_ln)
                        except Exception:
                            parsed_var, parsed_type, parsed_constraints = None, None, {}
                        if parsed_var:
                            var_list = (parsed_constraints or {}).get('var_list') if parsed_constraints else None
                            names = var_list if var_list else [parsed_var]
                            outputs.extend(names)
                member_of = (builder_type if def_kind == 'builder'
                             else (func_name.split('.')[0] if '.' in func_name else None))
                entry = {
                    'name': func_name,
                    'code': func_code,
                    'outputs': outputs,
                    'inputs': inputs,
                    'input_defs': input_defs,
                    'member_of': member_of,
                    # Keep the original casing so we can expose multiple aliases
                    'original': func_name,
                    'hidden': hidden,
                    'code_lines': code_lines,
                    'defining_scope': self.current_scope()
                }
                # Uniform redefinition guard -- applies to EVERY kind by
                # the time the entry dict is assembled, before any kind
                # dispatch. A program may not redefine an engine-owned
                # dotted capability handle-creator (engine dotted creators),
                # a predefined subprocess, or any name it already defined
                # elsewhere in the same program (cross-kind uniqueness,
                # exactly as testredef.grid expects).
                if func_name:
                    # Kind-aware redefinition focal: a builder is a
                    # capability MEMBER of a specific type, so the SAME
                    # builder name may legitimately serve DIFFERENT types
                    # (Test 260: 'bump' on A and on B are two distinct
                    # capabilities and must both be accepted). A builder is
                    # therefore keyed on BOTH its owner type and its name.
                    # Every NON-builder Define (Function / Subprocess /
                    # dotted capability members) is unique program-wide.
                    if def_kind == 'builder':
                        redef_key = ((builder_type or '').lower(),
                                     func_name.lower())
                    else:
                        redef_key = func_name.lower()
                    if (redef_key in _DOTTED_ENGINE_CREATORS
                            or redef_key in _PREDEFINED_SUBPROCESSES
                            or redef_key in self._program_defined_names):
                        raise SyntaxError(
                            f"'{func_name}' is already defined and cannot be "
                            f"redefined at line {line_number}. ' not allowed'")
                    self._program_defined_names.add(redef_key)

                if def_kind == 'builder':
                    type_name = builder_type
                    type_def = self.types_defined.get(type_name.lower())
                    if not type_def:
                        raise SyntaxError(
                            f"Type '{type_name}' not defined for builder '{func_name}' at line {line_number}")
                    if isinstance(type_def, dict):
                        helpers = type_def.setdefault('_builders', {})
                        helpers[func_name.lower()] = entry
                elif def_kind in ('function', 'operation'):
                    functions[func_name.lower()] = entry
                else:
                    subprocesses[func_name.lower()] = entry
                i += 1
                continue
            new_lines.append((line, line_number))
            i += 1
        self.functions = functions
        self.subprocesses = subprocesses
        # Register synthetic Resource.Handle factories now that both handle types
        # and user functions are known.
        try:
            self._register_handle_factories()
        except Exception:
            pass
        return new_lines, label_lines, dim_lines

    def call_function(self, name, args, instance_type=None, collect_all=False, vectorize=True):
        """Invoke a user-defined function by name with the given arguments.

        When collect_all is True, return all pushed output values (as lists)
        instead of collapsing to the last pushed value.
        """
        func_def = getattr(self, 'functions', {}).get(name.lower())
        if not func_def:
            raise NameError(f"Function '{name}' not defined")
        if func_def.get('hidden') and not getattr(self, '_allow_hidden_member_calls', False):
            raise PermissionError(
                f"Hidden member function '{name}' cannot be called here")
        member_of = func_def.get('member_of')
        if member_of:
            if not args:
                raise ValueError(
                    f"Member function '{name}' requires an instance of '{member_of}' as first argument")
            instance = args[0]
            inferred_type = instance_type
            # Fallback: infer from object keys
            if inferred_type is None and isinstance(instance, dict):
                inferred_type = instance.get('_type_name')
            if inferred_type is None and isinstance(instance, dict):
                for t_name, t_def in self.types_defined.items():
                    public_fields = self._get_public_type_fields(t_def)
                    if public_fields and object_public_keys(instance) == set(public_fields.keys()):
                        inferred_type = t_name
                        break
            if inferred_type and not self._is_type_compatible(inferred_type, member_of):
                raise TypeError(
                    f"Member function '{name}' expects instance of '{member_of}', got '{inferred_type}'")
        # Handle factory: synthetic Resource.Handle constructor – instantiate handle
        if func_def.get('is_handle_factory'):
            handle_type = func_def.get('handle_type')
            resource_instance = args[0] if args else None
            handle_args = args[1:] if len(args) > 1 else []
            # Instantiate handle with privileged resource access
            handle_val = self._instantiate_handle(
                handle_type, resource_instance, handle_args, None,
                var_name=handle_type)
            return handle_val

        input_defs = func_def.get('input_defs') or []
        if vectorize and not collect_all:
            array_args = []
            for idx, arg in enumerate(args):
                if isinstance(arg, dict) and 'array' in arg:
                    array_args.append((idx, list(arg['array'])))
                elif isinstance(arg, (list, tuple)):
                    array_args.append((idx, list(arg)))
            if array_args:
                should_vectorize = False
                for idx, _vals in array_args:
                    dim_spec = None
                    if idx < len(input_defs):
                        dim_spec = input_defs[idx].get('constraints', {}).get('dim')
                    if isinstance(dim_spec, str) and dim_spec.replace(' ', '') == '{}':
                        continue
                    should_vectorize = True
                    break
                if should_vectorize:
                    lengths = {len(vals) for _, vals in array_args}
                    if len(lengths) > 1:
                        should_vectorize = False
                        array_args = []
                        lengths = set()
                    if should_vectorize:
                        count = lengths.pop() if lengths else 0
                        results = []
                        for i in range(count):
                            elem_args = []
                            for arg in args:
                                if isinstance(arg, dict) and 'array' in arg:
                                    elem_args.append(list(arg['array'])[i])
                                elif isinstance(arg, (list, tuple)):
                                    elem_args.append(list(arg)[i])
                                else:
                                    elem_args.append(arg)
                            results.append(self.call_function(
                                name, elem_args, instance_type=instance_type,
                                collect_all=False, vectorize=False))
                        return results
        for idx, input_def in enumerate(input_defs):
            if idx >= len(args):
                break
            expected_type = input_def.get('type')
            if not expected_type:
                continue
            actual_val = args[idx]
            expected_lower = expected_type.lower()
            if expected_lower in self.types_defined:
                if not isinstance(actual_val, dict):
                    raise TypeError(
                        f"Input '{input_def.get('name')}' expects {expected_type}, got {type(actual_val).__name__}")
                actual_type = actual_val.get('_type_name')
                if actual_type is None:
                    actual_type = None
                    for t_name, t_def in self.types_defined.items():
                        public_fields = self._get_public_type_fields(t_def)
                        if public_fields and object_public_keys(actual_val) == set(public_fields.keys()):
                            actual_type = t_name
                            break
                if actual_type and not self._is_type_compatible(actual_type, expected_lower):
                    raise TypeError(
                        f"Input '{input_def.get('name')}' expects {expected_type}, got {actual_type}")
        sub_compiler = GridLangCompiler()
        sub_compiler.types_defined = getattr(self, 'types_defined', {})
        sub_compiler.functions = getattr(self, 'functions', {})
        sub_compiler.preserve_types_defined = True
        sub_compiler.preserve_functions = True
        sub_compiler._allow_hidden_field_access = True
        # Members of the declaring type may use private builders/member
        # functions; standalone functions may not.
        sub_compiler._allow_hidden_member_calls = bool(func_def.get('member_of'))
        # Functions reference the caller's scope chain live but read-only:
        # reads resolve through the parent, writes to caller variables are
        # rejected. The caller's 'grid' is just another caller variable: it
        # resolves through the same chain and is protected by the generic
        # outer-scope read-only rule.
        defining_scope = func_def.get('defining_scope')
        sub_compiler._parent_scope = defining_scope if defining_scope is not None else self.current_scope()
        sub_compiler._outer_scope_read_only = True
        # Function/operation bodies may use Return (the call-value channel);
        # the strict top-level Return check is skipped for these runners.
        sub_compiler._is_operation_runner = True
        # Only compiler-level dimension metadata is copied.
        try:
            if getattr(self, 'scopes', None):
                import copy
                sub_compiler._seed_globals = {
                    'dimensions': copy.deepcopy(getattr(self, 'dimensions', {})),
                    'dim_names': copy.deepcopy(getattr(self, 'dim_names', {})),
                    'dim_labels': copy.deepcopy(getattr(self, 'dim_labels', {})),
                }
        except Exception:
            pass
        func_result = sub_compiler.run(
            func_def['code'], list(args),
            suppress_output=True, return_output=True)
        outputs = func_result or {}

        # Merge declared outputs from function scope when not pushed explicitly.
        try:
            last_scope_vars = getattr(sub_compiler, '_last_scope_vars', {}) or {}
            for out_name in func_def.get('outputs', []) or []:
                out_key = out_name.lower()
                if out_key in outputs:
                    continue
                try:
                    merged_val = sub_compiler.current_scope().get(out_name)
                except Exception:
                    merged_val = None
                if merged_val is None:
                    for key, value in last_scope_vars.items():
                        if key.lower() == out_key:
                            merged_val = value
                            break
                if merged_val is not None:
                    outputs[out_key] = merged_val
        except Exception:
            pass

        # Normalize outputs to lists to preserve all pushed values
        normalized_outputs = {}
        for k, v in outputs.items():
            if isinstance(v, list):
                normalized_outputs[k] = v
            elif v is None:
                normalized_outputs[k] = []
            else:
                normalized_outputs[k] = [v]

        def _pick_default():
            target_names = func_def['outputs'] if func_def['outputs'] else ['output']
            for out_name in target_names:
                if out_name in normalized_outputs:
                    return normalized_outputs[out_name]
            if normalized_outputs:
                return next(iter(normalized_outputs.values()))
            return []

        if collect_all:
            return normalized_outputs

        default_values = _pick_default()
        if default_values:
            return default_values[-1]
        return None

    def _grid_to_matrix(self, grid_dict):
        """Convert a grid dict {'A1': val, ...} to a dense 2D list."""
        if not grid_dict:
            return []
        max_row = 0
        max_col = 0
        cells = []
        for cell in grid_dict:
            if isinstance(cell, str):
                try:
                    col, row = split_cell(cell)
                    row_i = int(row)
                    col_i = col_to_num(col)
                except ValueError:
                    continue
            elif isinstance(cell, tuple) and len(cell) == 2:
                try:
                    row_i = int(cell[0]) + 1
                    col_i = int(cell[1]) + 1
                except (TypeError, ValueError):
                    continue
            else:
                continue
            max_row = max(max_row, row_i)
            max_col = max(max_col, col_i)
            cells.append((row_i, col_i, grid_dict[cell]))

        if max_row == 0 or max_col == 0:
            return []

        matrix = [[0 for _ in range(max_col)] for _ in range(max_row)]
        for r, c, val in cells:
            matrix[r - 1][c - 1] = val
        return matrix

    def _keyed_field_names(self, type_def):
        """Field names that are keyed: declared as a Keytype type (composite or
        primitive alias) or with an ``as ... key`` field constraint.

        Keyed fields imply Not Null, so an instance whose keyed field is left
        unset is a sticky #VALUE error at storage."""
        keyed = set()
        for fname, ftype in self._get_public_type_fields(type_def).items():
            fdef = self.types_defined.get(str(ftype).lower(), {}) if ftype else {}
            if fdef.get('_keyed'):
                keyed.add(fname.lower())
            fcons = (type_def.get('_field_constraints', {}) or {}).get(fname) \
                or (type_def.get('_field_constraints', {}) or {}).get(fname.lower(), {})
            if fcons.get('key'):
                keyed.add(fname.lower())
        return keyed

    def _is_keyed_primitive_field_type(self, ftype):
        """True if ``ftype`` is a Keytype primitive alias (e.g. ``L as Keytype(number)``),
        i.e. a Keytype with no composite fields based on a primitive base type."""
        fdef = self.types_defined.get(str(ftype).lower(), {}) if ftype else {}
        return bool(fdef.get('_keyed') and not self._get_public_type_fields(fdef)
                    and fdef.get('_base_type') in ('number', 'text', 'logical'))

    def _wrap_keytype_primitive_fields(self, value_dict, public_fields, type_def=None, ensure_fresh=False):
        """Wrap values of Keytype primitive-alias fields as fresh UnitValues so
        they read as fresh key instances (e.g. ``k as L`` where ``L as Keytype(number)``).
        Returns the (possibly reused) ``value_dict``."""
        for fname, ftype in list(public_fields.items()):
            if not self._is_keyed_primitive_field_type(ftype):
                continue
            raw = value_dict.get(fname)
            if raw is not None and not isinstance(raw, UnitValue):
                from units import UnitValue
                if isinstance(raw, UnitValue) and getattr(raw, 'key_type', None):
                    continue
                base = self.types_defined.get(str(ftype).lower(), {}).get('_base_type')
                if raw is None:
                    base_val = 0 if base == 'number' else ("" if base == 'text' else False)
                    raw = base_val
                value_dict[fname] = UnitValue(raw, unit=None, key_type=str(ftype).lower(), fresh_key=True)
            elif isinstance(raw, UnitValue) and ensure_fresh and not getattr(raw, 'fresh_key', False):
                raw.fresh_key = True
        return value_dict

    def _mark_keyed_fields_immutable(self, value_dict, public_fields, type_def=None):
        """Mark keyed fields (Keytype-typed fields and ``as ... key`` field
        constraints) immutable so Push on them is a compile error."""
        keyed = self._keyed_field_names(type_def) if type_def else set()
        if type_def is None:
            for fname, ftype in public_fields.items():
                fdef = self.types_defined.get(str(ftype).lower(), {}) if ftype else {}
                if fdef.get('_keyed'):
                    keyed.add(fname.lower())
        if keyed:
            immk = value_dict.setdefault('_immutable_fields', set())
            immk.update(keyed)
        return value_dict

    def _materialize_unset_type_fields(self, value_dict, type_name):
        """Ensure every declared public field key exists on a fresh instance,
        leaving any that the constructor did not set as unset (None) so reads
        wrap to #N/A rather than raising 'field does not exist'."""
        type_def = self.types_defined.get(str(type_name).lower(), {}) or {}
        for fname in self._get_public_type_fields(type_def):
            if fname not in value_dict:
                value_dict[fname] = None
        return value_dict

    def _instantiate_type(self, type_name, args, line_number, allow_default_if_empty=False, var_name=None, execute_code=True, input_values_out=None):
        """Create an instance dict for a user-defined type, honoring inputs and constructor code."""
        type_def = self.types_defined[type_name.lower()]
        public_fields = self._get_public_type_fields(type_def)
        all_fields = self._get_all_type_fields(type_def)
        inputs_list = type_def.get('_inputs', [])
        exec_lines = type_def.get('_executable_code', [])
        base_type = type_def.get('_base_type')
        expected_args = len(inputs_list) if inputs_list else len(public_fields)

        def _normalize_inputs(raw_inputs):
            normalized = []
            for entry in raw_inputs or []:
                if isinstance(entry, dict):
                    normalized.append(entry)
                else:
                    normalized.append({'name': entry, 'default': None, 'type': None})
            return normalized

        inputs_list = _normalize_inputs(inputs_list)

        args = args or []
        if base_type and not all_fields:
            # Keytype primitive alias (e.g. L as Keytype(number)) wraps the primitive as a fresh key instance
            is_keyed_primitive = type_def.get('_keyed') and base_type in ('number', 'text', 'logical')
            if is_keyed_primitive:
                from units import UnitValue
                if not args and allow_default_if_empty:
                    base_val = 0 if base_type == 'number' else ("" if base_type == 'text' else False if base_type == 'logical' else None)
                    return UnitValue(base_val, unit=None, key_type=type_name.lower(), fresh_key=True)
                if len(args) != 1:
                    raise ValueError(
                        f"Expected 1 value for type '{type_name}', got {len(args)} at line {line_number}")
                # Wrap the primitive value as a fresh key instance
                raw = args[0]
                # Unwrap if already a UnitValue (e.g. from 5 of unit) - keep value, add key
                if isinstance(raw, UnitValue):
                    # Preserve unit, add key_type and fresh
                    return UnitValue(raw.value, raw.unit, key_type=type_name.lower(), fresh_key=True)
                return UnitValue(raw, unit=None, key_type=type_name.lower(), fresh_key=True)
            if not args and allow_default_if_empty:
                if base_type == 'number':
                    return 0
                if base_type == 'text':
                    return ""
                if base_type == 'array':
                    return []
                return None
            if len(args) != 1:
                raise ValueError(
                    f"Expected 1 value for type '{type_name}', got {len(args)} at line {line_number}")
            return args[0]

        if not args and allow_default_if_empty:
            value_dict = {}
            input_values = {}
            if inputs_list:
                for entry in inputs_list:
                    default_expr = entry.get('default')
                    if default_expr is None:
                        raise ValueError(
                            f"Expected {expected_args} values for type '{type_name}', got 0 at line {line_number}")
                    eval_scope = self.current_scope().get_full_scope()
                    input_values[entry.get('name')] = self.expr_evaluator.eval_or_eval_array(
                        str(default_expr), eval_scope, line_number)
            else:
                for field_name in public_fields:
                    value_dict[field_name] = None
            value_dict['_type_name'] = type_name.lower()
            hidden_fields = type_def.get('_hidden_fields', set())
            if hidden_fields:
                value_dict['_hidden_fields'] = set(hidden_fields)
            self._materialize_unset_type_fields(value_dict, type_name)
            # Wrap Keytype fields (e.g. k as L where L as Keytype(number)) as fresh UnitValue
            self._wrap_keytype_primitive_fields(value_dict, public_fields, type_def)
            if input_values_out is not None:
                input_values_out.clear()
                input_values_out.update(input_values)
            self._init_instance_grid(value_dict, type_def, line_number)
            if exec_lines and execute_code:
                self._execute_type_code(
                    exec_lines, var_name or type_name, value_dict, line_number, input_values)
            elif inputs_list:
                value_dict.update(
                    {name: val for name, val in input_values.items() if name in public_fields})
                if input_values:
                    immutable = value_dict.setdefault('_immutable_fields', set())
                    immutable.update(n.lower() for n in input_values
                                     if n in public_fields)
            # A constructor that hit a conflict invalidates the instance
            if isinstance(value_dict, dict) and value_dict.pop('_with_conflict', False):
                return '#VALUE'
            # Keyed fields are immutable after construction (Push on key → compile error)
            self._mark_keyed_fields_immutable(value_dict, public_fields, type_def)
            if type_def.get('_keyed'):
                value_dict.setdefault('_fresh_key', True)
            return value_dict

        if inputs_list and len(args) > expected_args:
            raise ValueError(
                f"Expected {expected_args} values for type '{type_name}', got {len(args)} at line {line_number}")
        if not inputs_list and args:
            raise ValueError(
                f"Type '{type_name}' has no Input declarations; positional arguments are not allowed at line {line_number}")

        value_dict = {}
        input_values = {}
        target_names = [entry['name'] for entry in inputs_list]
        for field_name, val in zip(target_names, args):
            if inputs_list:
                input_values[field_name] = val
            else:
                value_dict[field_name] = val

        if inputs_list:
            for entry in inputs_list[len(args):]:
                default_expr = entry.get('default')
                if default_expr is None:
                    raise ValueError(
                        f"Missing value for input '{entry.get('name')}' in type '{type_name}' at line {line_number}")
                eval_scope = self.current_scope().get_full_scope()
                input_values[entry.get('name')] = self.expr_evaluator.eval_or_eval_array(
                    str(default_expr), eval_scope, line_number)

        value_dict['_type_name'] = type_name.lower()
        hidden_fields = type_def.get('_hidden_fields', set())
        if hidden_fields:
            value_dict['_hidden_fields'] = set(hidden_fields)
        self._materialize_unset_type_fields(value_dict, type_name)
        # Wrap Keytype fields (e.g. k as L where L as Keytype) as fresh UnitValue
        self._wrap_keytype_primitive_fields(value_dict, public_fields, type_def)
        if input_values_out is not None:
            input_values_out.clear()
            input_values_out.update(input_values)
        self._init_instance_grid(value_dict, type_def, line_number)
        if exec_lines and execute_code:
            # Track immutability for fields derived from inputs
            if inputs_list:
                value_dict.setdefault('_immutable_fields', set())
            self._execute_type_code(
                exec_lines, var_name or type_name, value_dict, line_number, input_values)
        elif inputs_list:
            # Map inputs directly to matching fields when no executable code is provided
            value_dict.update(
                {name: val for name, val in input_values.items() if name in public_fields})
            if input_values:
                immutable = value_dict.setdefault('_immutable_fields', set())
                immutable.update(n.lower() for n in input_values
                                 if n in public_fields)

        # A constructor that hit a conflict (e.g. Push onto an equality-constant
        # field) invalidates the whole instance: never materialise it.
        if isinstance(value_dict, dict) and value_dict.pop('_with_conflict', False):
            return '#VALUE'

        # Wrap any remaining Keytype fields that were set via with/args after exec (e.g. k as L with k=5)
        self._wrap_keytype_primitive_fields(value_dict, public_fields, type_def, ensure_fresh=True)

        # Keyed fields are immutable after construction (Push on key → compile error)
        self._mark_keyed_fields_immutable(value_dict, public_fields, type_def)
        # Keyed object instances are fresh on construction; storing clears it.
        if type_def.get('_keyed'):
            value_dict.setdefault('_fresh_key', True)

        return value_dict

    def _instantiate_handle(self, handle_type, resource_instance, args, line_number, var_name=None):
        """Create a handle instance derived from a Resource instance.

        Handle types are declared ``Define Resource.Handle as Handle`` and
        behave like Type constructors but with privileged access to the parent
        Resource: bare reads (``x``) resolve to Resource fields, and ``Push``
        to a Resource field mutates the Resource instance (builder-style).
        ``args`` are the handle's own Input parameters (excluding the implicit
        Resource receiver). The handle's ``_type_name`` is the canonical
        ``resource!handle`` string.
        """
        handle_def = self.types_defined.get(handle_type.lower())
        if not isinstance(handle_def, dict):
            raise ValueError(f"Handle type '{handle_type}' not defined at line {line_number}")
        if not (handle_def.get('_constraints') or {}).get('is_handle'):
            raise ValueError(f"Type '{handle_type}' is not a Handle at line {line_number}")
        public_fields = self._get_public_type_fields(handle_def)
        inputs_list = handle_def.get('_inputs', []) or []
        exec_lines = handle_def.get('_executable_code', []) or []
        # Normalize inputs
        def _normalize_inputs(raw):
            norm = []
            for entry in raw or []:
                if isinstance(entry, dict):
                    norm.append(entry)
                else:
                    norm.append({'name': entry, 'default': None, 'type': None})
            return norm
        inputs_list = _normalize_inputs(inputs_list)
        args = list(args or [])
        if inputs_list and len(args) > len(inputs_list):
            raise ValueError(
                f"Expected {len(inputs_list)} values for handle '{handle_type}', got {len(args)} at line {line_number}")
        if not inputs_list and args:
            raise ValueError(
                f"Handle '{handle_type}' has no Input declarations; positional arguments are not allowed at line {line_number}")
        input_values = {}
        for idx, entry in enumerate(inputs_list):
            name = entry.get('name')
            if idx < len(args):
                input_values[name] = args[idx]
            else:
                default_expr = entry.get('default')
                if default_expr is None:
                    raise ValueError(
                        f"Missing value for input '{name}' in handle '{handle_type}' at line {line_number}")
                eval_scope = self.current_scope().get_full_scope()
                # Include resource fields for default expression evaluation
                if isinstance(resource_instance, dict):
                    for rk, rv in resource_instance.items():
                        if not str(rk).startswith('_') and rk not in eval_scope:
                            eval_scope[rk] = rv
                input_values[name] = self.expr_evaluator.eval_or_eval_array(
                    str(default_expr), eval_scope, line_number)
        # Determine resource type name for dispatch
        resource_type = handle_type.split('!', 1)[0]
        # Prepare handle instance dict
        value_dict = {}
        value_dict['_type_name'] = handle_type.lower()
        value_dict['_handle'] = True
        value_dict['_handle_type'] = handle_type.lower()
        if isinstance(resource_instance, dict):
            value_dict['_resource'] = resource_instance
            # Parent linkage for engine-owned handle tracking (if needed)
            parent_name = resource_instance.get('_name')
            if parent_name:
                value_dict['_parent'] = parent_name
        hidden_fields = handle_def.get('_hidden_fields', set())
        if hidden_fields:
            value_dict['_hidden_fields'] = set(hidden_fields)
        self._materialize_unset_type_fields(value_dict, handle_type)
        self._wrap_keytype_primitive_fields(value_dict, public_fields, handle_def)
        self._init_instance_grid(value_dict, handle_def, line_number)
        # Execute handle constructor code with resource access
        handle_resource_type = resource_type or ''
        prev_res_instance = getattr(self, '_handle_resource_instance', None)
        prev_res_type = getattr(self, '_handle_resource_type', None)
        prev_member_keys = getattr(self, '_type_member_keys', None)
        self._handle_resource_instance = resource_instance
        self._handle_resource_type = handle_resource_type
        # Set member keys to handle's keys so _process_type_assignment knows Handle fields
        self._type_member_keys = set(handle_def.get('_member_keys', set()))
        # Use a member name that reflects the handle for error messages
        exec_var_name = 'this'
        try:
            if exec_lines:
                self._execute_type_code(exec_lines, exec_var_name, value_dict, line_number, input_values)
            else:
                # No executable code – map inputs directly to matching handle fields
                if inputs_list:
                    value_dict.update({name: val for name, val in input_values.items() if name in public_fields})
                    if input_values:
                        immutable = value_dict.setdefault('_immutable_fields', set())
                        immutable.update(n.lower() for n in input_values if n in public_fields)
            if isinstance(value_dict, dict) and value_dict.pop('_with_conflict', False):
                return '#VALUE'
        finally:
            # Restore previous handle resource context
            if prev_res_instance is None:
                try:
                    delattr(self, '_handle_resource_instance')
                except AttributeError:
                    pass
            else:
                self._handle_resource_instance = prev_res_instance
            if prev_res_type is None:
                try:
                    delattr(self, '_handle_resource_type')
                except AttributeError:
                    pass
            else:
                self._handle_resource_type = prev_res_type
            self._type_member_keys = prev_member_keys
        self._mark_keyed_fields_immutable(value_dict, public_fields, handle_def)
        if handle_def.get('_keyed'):
            value_dict.setdefault('_fresh_key', True)
        return value_dict

    def _register_handle_factories(self):
        """Synthesize Resource.Handle factory functions for each Handle type.

        For a handle ``Files!Open`` whose resource is ``Files``, a factory
        ``Files.Open`` (member_of = Files) is registered. Calling
        ``file.open(...)`` then dispatches to this factory which instantiates
        the handle with the resource receiver prepended.
        """
        for type_name, type_def in list(self.types_defined.items()):
            if not isinstance(type_def, dict):
                continue
            constraints = type_def.get('_constraints') or {}
            if not constraints.get('is_handle'):
                continue
            handle_type = type_name  # already lower-cased key (e.g. files!open)
            # Preserve original case for display? Use type_def original if available
            canonical = type_name
            # Derive resource and handle suffix
            resource = constraints.get('handle_resource')
            if not resource:
                if '!' in canonical:
                    resource = canonical.split('!', 1)[0]
                else:
                    continue
            if '!' in canonical:
                suffix = canonical.split('!', 1)[1]
            else:
                suffix = canonical
                canonical = f"{resource}!{suffix}"
            factory_name = f"{resource}.{suffix}"
            key = factory_name.lower()
            if key in self.functions or key in getattr(self, 'subprocesses', {}):
                continue
            input_defs = type_def.get('_inputs', []) or []
            # Build function entry for factory
            entry = {
                'name': factory_name,
                'original': factory_name,
                'code': '',
                'code_lines': [],
                'inputs': [d.get('name') for d in input_defs if isinstance(d, dict)],
                'input_defs': list(input_defs),
                'outputs': [],
                'member_of': resource,
                'hidden': False,
                'is_handle_factory': True,
                'handle_type': canonical,
                'defining_scope': self.current_scope(),
            }
            # Track as defined name for redefinition guard
            if not hasattr(self, '_program_defined_names'):
                self._program_defined_names = set()
            self._program_defined_names.add(key)
            self.functions[key] = entry

    def _init_instance_grid(self, value_dict, type_def, line_number):
        """Populate an instance's 'grid' field from the type's grid dims.

        A type declared ``with (grid dim {3, 3, 3})`` gives each instance a
        dense grid array of that shape (1-based). Without grid dims the field
        is a standard sparse array (dict keyed by 0-based index tuples), like
        any other unbounded array.
        """
        if not isinstance(value_dict, dict) or not isinstance(type_def, dict):
            return
        type_constraints = type_def.get('_constraints', {}) or {}
        grid_dim = type_constraints.get('dim')
        if isinstance(grid_dim, dict) and 'dims' in grid_dim:
            raw_dims = grid_dim['dims']
            has_unbounded = any(end is None for _, end in raw_dims)
            grid_type = grid_dim.get('grid_type', 'number')
            default_key = grid_dim.get('default')
            if default_key is not None:
                if default_key.lower() == 'none':
                    fill_value = UNIVERSAL_ZERO
                else:
                    try:
                        fill_value = float(default_key)
                    except (ValueError, TypeError):
                        fill_value = None
            else:
                fill_value = None
            if has_unbounded:
                grid_store = {}
                if fill_value is not None:
                    grid_store['_default'] = fill_value
                else:
                    exec_code = type_def.get('_executable_code', [])
                    for code_line in exec_code:
                        or_match = re.match(
                            r'^Let\s+grid\b.*\bor\s*=\s*(.+)$', code_line, re.I)
                        if or_match:
                            or_val = or_match.group(1).strip()
                            if or_val.lower() == 'none':
                                grid_store['_default'] = UNIVERSAL_ZERO
                            else:
                                try:
                                    grid_store['_default'] = float(or_val)
                                except (ValueError, TypeError):
                                    pass
                            break
                value_dict['grid'] = grid_store
            else:
                shape = [end - start + 1 for start, end in raw_dims]
                grid_store = self.array_handler.create_array(
                    shape, fill_value, grid_type, line_number,
                    template=(fill_value is None))
                value_dict['grid'] = grid_store
        else:
            # Lazy grid: don't create 'grid' if not explicitly dimensioned
            # and not touched by constructor/builders. Access returns {}
            # and grid{...} returns #N/A.
            pass

    def resolve_with_value(self, raw_value, line_number=None):
        """Finalize a single WITH-clause value against the current scope.

        Quoted strings become their literal, bare identifiers resolve as
        variable references (falling back to the literal when the variable is
        uninitialized), and non-strings pass through unchanged.
        """
        scope = self.current_scope().get_full_scope()
        return self._evaluate_with_value(raw_value, scope, line_number)

    def _evaluate_with_value(self, raw_value, scope, line_number=None):
        """Evaluate a WITH clause value when it looks like an expression."""
        if isinstance(raw_value, str):
            if raw_value.startswith('"') and raw_value.endswith('"'):
                return self.expr_evaluator.eval_or_eval_array(
                    raw_value, scope, line_number)
            if re.match(r'^[A-Za-z][A-Za-z0-9_]*$', raw_value):
                resolved = scope.get(raw_value, None)
                if resolved is None or (
                        isinstance(resolved, UnitValue)
                        and is_error_value(resolved.error_code)):
                    return raw_value
                return resolved
            try:
                return self.expr_evaluator.eval_or_eval_array(
                    raw_value, scope, line_number)
            except Exception:
                # Fallback: treat brace values as literal lists when identifiers are unresolved
                text = raw_value.strip()
                if text.startswith('{') and text.endswith('}'):
                    inner = text[1:-1].strip()
                    if not inner:
                        return []
                    items = [item.strip() for item in inner.split(',')]
                    resolved = []
                    for item in items:
                        if not item:
                            continue
                        if (item.startswith('"') and item.endswith('"')) or (
                                item.startswith("'") and item.endswith("'")):
                            resolved.append(item[1:-1])
                        elif re.match(r'^[A-Za-z][A-Za-z0-9_]*$', item) and item in scope:
                            resolved.append(scope.get(item))
                        else:
                            resolved.append(item)
                    return resolved
                return raw_value
        return raw_value

    def _coerce_with_field_value(self, field_type, field_value, line_number=None):
        """Coerce WITH-assigned values into declared custom field types when possible."""
        if not isinstance(field_type, str):
            return field_value
        expected_type = field_type.lower()
        if expected_type not in self.types_defined:
            return field_value
        if isinstance(field_value, (list, dict)):
            return self._convert_array_to_object(expected_type, field_value, line_number)
        if isinstance(field_value, dict):
            actual_type = field_value.get('_type_name')
            if actual_type and self._is_type_compatible(actual_type, expected_type):
                return field_value
            public_fields = self._get_public_type_fields(expected_type)
            if public_fields and object_public_keys(field_value) == set(public_fields.keys()):
                coerced = dict(field_value)
                coerced['_type_name'] = expected_type
                hidden_fields = self.types_defined.get(expected_type, {}).get('_hidden_fields', set())
                if hidden_fields and '_hidden_fields' not in coerced:
                    coerced['_hidden_fields'] = set(hidden_fields)
                # Lazy grid
                return coerced
        return field_value

    def _apply_with_constraints(self, value, with_constraints, scope, line_number=None, type_name=None):
        """Apply WITH clause values to a newly created object."""
        if not with_constraints or not isinstance(value, dict):
            return value
        field_map = {}
        public_fields = {}
        assigned_fields = set()
        if type_name and type_name.lower() in self.types_defined:
            public_fields = self._get_public_type_fields(type_name)
            field_map = {k.lower(): k for k in public_fields}
        else:
            field_map = {k.lower(): k for k in object_public_keys(value)}
        for key, raw_value in with_constraints.items():
            key_name = field_map.get(str(key).lower(), key)
            coerced_value = self._evaluate_with_value(raw_value, scope, line_number)
            declared_field_type = public_fields.get(key_name)
            if declared_field_type:
                coerced_value = self._coerce_with_field_value(
                    declared_field_type, coerced_value, line_number)
            # Handle unit conversion for field (e.g. : f of m = 5 of in)
            try:
                from units import UnitValue as UV, apply_conversion as AC
                if isinstance(coerced_value, UV) and coerced_value.unit:
                    # Find field's unit from type definition
                    f_cons = None
                    if type_name:
                        td = self.types_defined.get(type_name.lower(), {})
                        f_cons = (td.get('_field_constraints', {}) or {}).get(key_name) or (td.get('_field_constraints', {}) or {}).get(key_name.lower())
                    if f_cons and f_cons.get('unit'):
                        tgt = str(f_cons.get('unit')).lower()
                        if str(coerced_value.unit).lower() != tgt:
                            conv = AC(coerced_value.value, coerced_value.unit, tgt, self.expr_evaluator._formula_eval)
                            if conv is not None:
                                coerced_value = conv
            except Exception:
                pass
            value[key_name] = coerced_value
            assigned_fields.add(str(key_name).lower())
            if type_name:
                self._check_type_field_constraints(
                    type_name, key_name, value[key_name], value, scope, line_number)
        # Remember precisely which fields the WITH clause set, so constructor
        # code that runs afterwards does not overwrite them. The marker is
        # removed once the constructor body has executed.
        marker = value.setdefault('_with_applied_fields', set())
        marker.update(assigned_fields)
        if type_name and type_name.lower() in self.types_defined:
            self._recompute_type_fields_after_with(
                type_name, value, scope, line_number)
        return value

    def _check_type_field_constraints(self, type_name, field_name, value, value_dict, scope, line_number=None):
        type_def = self.types_defined.get(type_name.lower())
        if not type_def or not isinstance(type_def, dict):
            return
        constraints_map = type_def.get('_field_constraints', {}) or {}
        actual_key = None
        for key in constraints_map.keys():
            if str(key).lower() == str(field_name).lower():
                actual_key = key
                break
        if actual_key is None:
            return
        constraints = constraints_map.get(actual_key, {})
        if not constraints:
            return
        tmp_scope = Scope(self)
        if hasattr(scope, 'get_full_scope'):
            tmp_scope.variables.update(scope.get_full_scope())
        elif isinstance(scope, dict):
            tmp_scope.variables.update(scope)
        if hasattr(self.current_scope(), 'get_full_scope'):
            tmp_scope.variables.update(self.current_scope().get_full_scope())
        if isinstance(value_dict, dict):
            for k, v in value_dict.items():
                if not str(k).startswith('_'):
                    tmp_scope.variables[k] = v
        tmp_scope.constraints[actual_key] = constraints
        declared_type = constraints.get('type')
        if isinstance(declared_type, str) and declared_type.lower() in ('number', 'text'):
            tmp_scope.types[actual_key] = declared_type.lower()
        try:
            tmp_scope._check_constraints(actual_key, value, line_number)
        except ConstraintError:
            if isinstance(value_dict, dict):
                value_dict['_with_conflict'] = True
            else:
                raise

    def _apply_with_clause(self, value, with_kind, with_payload, scope, line_number=None, type_name=None):
        """Apply a parsed WITH clause (kind, payload) to a constructed object.

        The constructor has already run: only public fields are overwritten.
        Clone sources may be a single custom-type instance or an (nested)
        array of instances; arrays yield equally-shaped arrays of clones.
        """
        if with_kind == 'named':
            return self._apply_with_constraints(
                value, with_payload, scope, line_number, type_name)
        if with_kind == 'positional':
            fields = self._get_public_type_fields(type_name) if type_name and type_name.lower(
            ) in self.types_defined else {}
            names = list(fields.keys()) or [
                k for k in object_public_keys(value)]
            if len(with_payload) > len(names):
                raise ValueError(
                    f"Too many values in WITH clause for '{type_name or 'object'}' "
                    f"at line {line_number}")
            named = {}
            for i, raw in enumerate(with_payload):
                if i < len(names):
                    named[names[i]] = raw
            return self._apply_with_constraints(
                value, named, scope, line_number, type_name)
        if with_kind == 'clone':
            source = self.expr_evaluator.eval_or_eval_array(
                with_payload, scope, line_number)
            return self._clone_object_fields(
                value, source, type_name, scope, line_number)
        return value

    def _clone_object_fields(self, template, source, type_name, scope, line_number=None):
        """Copy a source object's public fields onto fresh instance(s)."""
        # An array variable bound by ``For`` is stored as a coordinate-keyed
        # dict (e.g. {(0,): obj, (1,): obj}); accept that shape as an array of
        # source instances.
        if isinstance(source, dict) and source:
            keys = list(source.keys())
            if all(isinstance(k, (int, tuple)) for k in keys):
                source = [source[k] for k in sorted(keys)]
        if isinstance(source, (list, tuple)):
            return [self._clone_object_fields(
                template, elem, type_name, scope, line_number) for elem in source]
        if not isinstance(source, dict):
            raise TypeError(
                f"Clone source must be an instance or an array of instances "
                f"at line {line_number}")
        fields = self._get_public_type_fields(type_name) if type_name and type_name.lower(
        ) in self.types_defined else {}
        names = list(fields.keys()) or [
            k for k in object_public_keys(source)]
        src_map = {
            str(k).lower(): v for k, v in source.items()
            if not str(k).startswith('_') and str(k) != 'grid'}
        named = {}
        for name in names:
            raw = src_map.get(str(name).lower())
            if raw is None:
                continue
            named[name] = raw
        target = copy.deepcopy(template)
        return self._apply_with_constraints(
            target, named, scope, line_number, type_name)

    def _with_deps(self, with_text, line_number=None):
        """Dependency identifiers referenced by the value side of a WITH clause."""
        with_kind, with_payload = self._parse_with_clause(with_text, line_number)
        deps = set()
        if with_kind == 'named':
            for raw in (with_payload or {}).values():
                deps |= self._extract_identifier_tokens(raw)
        elif with_kind == 'positional':
            for raw in (with_payload or []):
                deps |= self._extract_identifier_tokens(raw)
        elif with_kind == 'clone':
            deps |= self._extract_identifier_tokens(with_payload)
        return deps

    def _recompute_type_fields_after_with(self, type_name, value, scope, line_number=None):
        type_def = self.types_defined.get(type_name.lower())
        if not type_def or not isinstance(value, dict):
            return
        if value.get('_with_conflict'):
            return
        exec_lines = type_def.get('_executable_code', [])
        if not exec_lines:
            return
        input_names = {
            entry.get('name', '').lower()
            for entry in (type_def.get('_inputs') or [])
            if isinstance(entry, dict)
        }
        field_names = {k.lower() for k in self._get_public_type_fields(type_def)}
        eval_scope = {}
        if hasattr(self.current_scope(), 'get_full_scope'):
            eval_scope = self.current_scope().get_full_scope()
        elif isinstance(scope, dict):
            eval_scope = scope
        eval_scope = dict(eval_scope) if isinstance(eval_scope, dict) else {}
        for key, val in value.items():
            if not str(key).startswith('_'):
                eval_scope[key] = val

        string_pattern = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
        ident_pattern = re.compile(r'[A-Za-z][A-Za-z0-9_.]*')

        for raw_line in exec_lines:
            stripped = raw_line.strip()
            if not stripped or stripped.lower().startswith('super'):
                continue
            m = re.match(r'^\s*(\$?[\w_]+)\s*=\s*(.+)$', stripped)
            if not m:
                continue
            field_name, expr = m.groups()
            if field_name.startswith('$'):
                field_name = field_name[1:]
            cleaned = string_pattern.sub(' ', expr)
            tokens = ident_pattern.findall(cleaned)
            deps = {tok.split('.')[0].lower() for tok in tokens if tok}
            deps.discard(field_name.lower())
            if not deps:
                continue
            if deps & input_names:
                continue
            if not deps & field_names:
                continue
            value[field_name] = self.expr_evaluator.eval_expr(
                expr, eval_scope, line_number)
            eval_scope[field_name] = value[field_name]

    def _split_with_parts(self, text):
        """Split a WITH clause body into top-level comma-separated parts."""
        parts = []
        current = ""
        in_quotes = False
        paren_level = 0
        brace_level = 0
        for char in text + ',':
            if char == '"' and (len(current) == 0 or current[-1] != '\\'):
                in_quotes = not in_quotes
            elif not in_quotes:
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level = max(paren_level - 1, 0)
                elif char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level = max(brace_level - 1, 0)
            if char == ',' and not in_quotes and paren_level == 0 and brace_level == 0:
                if current.strip():
                    parts.append(current.strip())
                current = ""
            else:
                current += char
        return parts

    def _split_named_with_parts(self, text, line_number=None):
        """Split named 'key = value' WITH entries into a {field: expr} dict."""
        assignments = {}
        for part in text:
            if '=' in part:
                key, value = part.split('=', 1)
                assignments[key.strip()] = value.strip()
            else:
                name = part.strip()
                if name:
                    raise SyntaxError(
                        f"Ambiguous 'with ({name})' at line {line_number}: use "
                        "'<field> = <value>' for named constraints or 'with {…}' "
                        "for positional — by-name shorthand is not implied")
        return assignments

    def _parse_with_clause(self, with_text, line_number=None):
        """Parse a WITH clause body.

        Returns a (kind, payload) tuple:
          ('named', {field: expr, ...})  -- 'with (a = 1, b = foo)' or 'with a, b'
          ('positional', [expr, ...])    -- 'with {1, 2, dx}'
          ('clone', expr_str)            -- 'with identifier' (clone source)
          ('empty', None)                -- blank clause
        """
        if not with_text:
            return 'empty', None
        text = with_text.strip()
        if text.lower().startswith('with'):
            text = text[4:].strip()
        if not text:
            return 'empty', None
        if text.startswith('(') and text.endswith(')'):
            return 'named', self._split_named_with_parts(
                self._split_with_parts(text[1:-1].strip()), line_number)
        if text.startswith('{') and text.endswith('}'):
            inner = text[1:-1].strip()
            return 'positional', self._split_with_parts(inner)
        parts = self._split_with_parts(text)
        if len(parts) == 1 and '=' not in parts[0]:
            return 'clone', parts[0]
        return 'named', self._split_named_with_parts(parts, line_number)

    def _split_new_with_expr(self, expr):
        """Split a 'new Type ... with (...)' expression into base and with clause."""
        if not expr or not expr.lower().startswith('new '):
            return None, None
        in_quote = None
        paren_level = 0
        brace_level = 0
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
                    base_expr = expr[:idx].strip()
                    with_text = expr[idx:].strip()
                    return base_expr, with_text
        return None, None

    def _apply_single_binding(self, binding, values, scope, line_number=None):
        """Apply subprocess output binding to a target."""
        if values is None:
            return
        target = binding.strip()
        if not target:
            return
        if isinstance(values, dict) and 'array' in values:
            values = list(values['array'])
        simple_value = values[0] if isinstance(
            values, list) and len(values) == 1 else values

        if target.startswith('[') and target.endswith(']'):
            inner = target[1:-1].strip()
            is_horizontal = inner.startswith('^')
            inner = inner[1:].strip() if is_horizontal else inner
            cell_ref = self.expr_evaluator._resolve_column_interpolated_cell(
                inner, scope.get_evaluation_scope(), line_number) or inner
            validate_cell_ref(cell_ref)
            cell_key = self._to_index(cell_ref)
            if isinstance(values, list):
                all_scalars = all(not isinstance(v, (list, dict))
                                  for v in values)
                if all_scalars:
                    # For scalar pushes, keep the last value only
                    simple_value = values[-1]
                    self._set_grid_cell(cell_key, simple_value)
                    return
            assign_value = values if isinstance(values, list) else [values]
            expr_hint = '{' + ','.join(['x'] * len(assign_value)) + '}'
            if is_horizontal or len(assign_value) > 1:
                self.array_handler._assign_horizontal_array(
                    cell_key, assign_value, expr_hint, line_number=line_number)
            else:
                self._set_grid_cell(cell_key, simple_value)
            return

        try:
            scope.update(target, simple_value, line_number)
        except NameError:
            scope.define(target, simple_value)

    def _apply_subprocess_outputs(self, sp_def, output_values, output_bindings, caller_scope=None, line_number=None):
        """Map subprocess output variables to caller-provided bindings."""
        if not output_bindings or not sp_def.get('outputs'):
            return
        scope = caller_scope or self.current_scope()
        outputs = output_values or {}
        for idx, out_name in enumerate(sp_def.get('outputs', [])):
            if idx >= len(output_bindings):
                break
            binding = output_bindings[idx]
            if binding is None:
                continue
            out_key = out_name.lower()
            if out_key not in outputs:
                continue
            self._apply_single_binding(
                binding, outputs.get(out_key), scope, line_number=line_number)

    def call_subprocess(self, name, args, output_bindings=None, line_number=None, collect_all=False):
        """Invoke a user-defined subprocess by name with the given arguments."""
        sp_def = getattr(self, 'subprocesses', {}).get(name.lower())
        if not sp_def:
            raise NameError(f"Subprocess '{name}' not defined")
        # Predefined engine subprocesses (Ticker.Reset/Stop/Start) have no body.
        if sp_def.get('_system'):
            self._handle_ticker_system_call(
                name, list(args), line_number)
            if collect_all:
                return {}
            return SubprocessResult(grid=[], variables={}, outputs={})
        sub_compiler = GridLangCompiler()
        sub_compiler.types_defined = getattr(self, 'types_defined', {})
        sub_compiler.functions = getattr(self, 'functions', {})
        sub_compiler.subprocesses = getattr(self, 'subprocesses', {})
        sub_compiler.preserve_types_defined = True
        sub_compiler.preserve_functions = True
        sub_compiler.preserve_subprocesses = True
        # Subprocesses reference the caller's scope chain live: reads resolve to
        # caller variables and writes (push/assignments) flow through. Module
        # subprocesses run against the module's live instance scope instead,
        # so pushes mutate module state (docs §11). Only compiler-level
        # dimension metadata is copied.
        defining_scope = sp_def.get('defining_scope')
        sub_compiler._parent_scope = (
            defining_scope if defining_scope is not None
            else self.current_scope())
        # Function/operation bodies may use Return (the call-value channel);
        # the strict top-level Return check is skipped for these runners.
        sub_compiler._is_operation_runner = True
        try:
            if getattr(self, 'scopes', None):
                sub_compiler._seed_globals = {
                    'dimensions': copy.deepcopy(getattr(self, 'dimensions', {})),
                    'dim_names': copy.deepcopy(getattr(self, 'dim_names', {})),
                    'dim_labels': copy.deepcopy(getattr(self, 'dim_labels', {})),
                }
        except Exception:
            pass
        sub_output = sub_compiler.run(
            sp_def['code'], list(args),
            suppress_output=True, return_output=True)

        # A module subprocess may have mutated its instance state; refresh the
        # importer's copies of the module's exported variables.
        if sp_def.get('module_key'):
            try:
                self._sync_module_export_vars(sp_def['module_key'])
            except Exception:
                pass

        # Merge declared outputs from subprocess scope when not pushed explicitly.
        merged_outputs = dict(sub_output or {})
        try:
            # The executor's scope stack diverges from the compiler's after
            # _reset_state (a compiler-bound method), so fall back to the
            # pre-clobber capture of the subprocess's final variables.
            last_scope_vars = getattr(sub_compiler, '_last_scope_vars', {}) or {}
            deferred_inits = getattr(sub_compiler, '_deferred_output_inits', {}) or {}
            for out_name in sp_def.get('outputs', []) or []:
                out_key = out_name.lower()
                if out_key in merged_outputs:
                    continue
                try:
                    merged_val = sub_compiler.current_scope().get(out_name)
                except Exception:
                    merged_val = None
                if merged_val is None:
                    for key, value in last_scope_vars.items():
                        if key.lower() == out_key:
                            merged_val = value
                            break
                if merged_val is None and out_key in deferred_inits:
                    init_expr, init_constraints = deferred_inits[out_key]
                    try:
                        scope_dict = dict(last_scope_vars)
                        merged_val = self.expr_evaluator.eval_or_eval_array(
                            str(init_expr), scope_dict, line_number)
                    except Exception:
                        merged_val = None
                if merged_val is not None:
                    merged_outputs[out_key] = merged_val
        except Exception:
            pass

        if output_bindings:
            self._apply_subprocess_outputs(
                sp_def, merged_outputs, output_bindings, caller_scope=self.current_scope(), line_number=line_number)

        normalized_outputs = {}
        if merged_outputs:
            for k, v in merged_outputs.items():
                if isinstance(v, list):
                    normalized_outputs[k] = v
                elif v is None:
                    normalized_outputs[k] = []
                else:
                    normalized_outputs[k] = [v]

        last_scope_vars = getattr(sub_compiler, '_last_scope_vars', {}) or {}
        grid_source = (last_scope_vars.get('grid')
                       or sub_compiler.current_scope().variables.get('grid')
                       or sub_compiler.grid)

        def _normalize_grid(value):
            if value is None:
                return []
            if isinstance(value, dict) and 'array' in value:
                value = value['array']
            if isinstance(value, dict):
                return self._grid_to_matrix(value)
            if isinstance(value, list):
                if value and all(isinstance(r, list) for r in value):
                    return value
                return [value]
            return []

        result_grid = _normalize_grid(grid_source)

        # If caller requested raw outputs map, include grid as a flattened sequence for generator use
        if collect_all and result_grid:
            flat = []
            for row in result_grid:
                if isinstance(row, list):
                    flat.extend(row)
                else:
                    flat.append(row)
            # Preserve any existing outputs, but make the grid-derived sequence available
            normalized_outputs.setdefault('grid', flat)

        if collect_all:
            return normalized_outputs
        return SubprocessResult(
            grid=result_grid,
            variables=sub_compiler.current_scope().variables.copy(),
            outputs=normalized_outputs or {})

    def _resolve_global_dependency(self, var, line_number, target_scope=None):
        if var not in self.pending_assignments:
            return False
        assignment = self.pending_assignments[var]
        expr, assign_line, deps = assignment[:3]
        constraints = assignment[3] if len(assignment) > 3 else {}
        cell_refs = self._extract_cell_refs(str(expr))
        if cell_refs and not self._grid_cells_set(cell_refs):
            return False
        scope = target_scope if target_scope is not None else self.current_scope()
        unresolved = any(
            dep != var and self.has_unresolved_dependency(dep, scope=scope)
            for dep in deps)
        if unresolved:
            return False
        try:
            eval_scope = scope.get_full_scope()
            if hasattr(scope, 'get_evaluation_scope'):
                eval_scope = scope.get_evaluation_scope()
            value = self.expr_evaluator.eval_or_eval_array(
                expr, eval_scope, assign_line,
                expected_unit=(constraints or {}).get('unit'))
            value = self.array_handler.check_dimension_constraints(
                var, value, assign_line)
            if constraints.get('with'):
                defining_scope = scope.get_defining_scope(var)
                type_name = None
                if defining_scope:
                    actual_key = defining_scope._get_case_insensitive_key(
                        var, defining_scope.types)
                    if actual_key:
                        type_name = defining_scope.types.get(actual_key)
                value = self._apply_with_constraints(
                    value, constraints.get('with', {}),
                    scope.get_full_scope(), assign_line,
                    type_name=type_name)
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
        self._resolve_pending_assignments_main_loop()
        block_pending = self._resolve_block_pending_assignments()
        if self.pending_assignments or block_pending:
            unresolved = list(self.pending_assignments.keys()
                              ) + list(block_pending.keys())
            raise RuntimeError(f"Unresolved assignments: {unresolved}")

    def _resolve_pending_assignments_main_loop(self):
        max_attempts = len(self.pending_assignments) + 10
        attempt = 0
        while self.pending_assignments and attempt < max_attempts:
            unresolved_before = set(self.pending_assignments.keys())
            for var, assignment in sorted(self.pending_assignments.items(), key=lambda x: (x[0].startswith('__line_'), int(x[0].replace('__line_', '') if x[0].startswith('__line_') else '0'))):
                expr, line_number, deps = assignment[:3]
                self._validate_pending_assignment_not_self_referential(
                    var, expr, deps, line_number)
                if var.startswith("__line_"):
                    target, rhs = expr.split(':=')
                    target, rhs = target.strip(), rhs.strip()
                    unresolved = any(
                        self.has_unresolved_dependency(
                            dep, scope=self.current_scope())
                        for dep in deps)
                    if unresolved:
                        continue
                    is_array_indexing = bool(re.match(
                        r'^[\w_]+\s*(?:\[\w+\]|\{\s*\d+\s*(?:,\s*\d+\s*)*\}|!\w+\s*\(\s*"\w+"\s*\))$', rhs.strip()))
                    cell_refs = set()
                    if not is_array_indexing:
                        cell_refs = self._extract_cell_refs(rhs)
                    if cell_refs and not self._grid_cells_set(cell_refs):
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
                                expr, self.current_scope().get_full_scope(), line_number,
                                expected_unit=(constraints or {}).get('unit'))
                            value = self.array_handler.check_dimension_constraints(
                                var, value, line_number)
                            if constraints.get('with'):
                                scope = self.current_scope()
                                defining_scope = scope.get_defining_scope(var)
                                type_name = None
                                if defining_scope:
                                    actual_key = defining_scope._get_case_insensitive_key(
                                        var, defining_scope.types)
                                    if actual_key:
                                        type_name = defining_scope.types.get(actual_key)
                                value = self._apply_with_constraints(
                                    value, constraints.get('with', {}),
                                    scope.get_full_scope(), line_number,
                                    type_name=type_name)
                            value = self._to_sparse_undimmed(
                                value, constraints or {})
                            self.current_scope().update(var, value, line_number)
                            del self.pending_assignments[var]
                        else:
                            self.grid.clear()
                            del self.pending_assignments[var]
                    except NameError as e:
                        missing = self.extract_missing_dependencies(e)
                        if missing:
                            for dep in missing:
                                self.mark_dependency_missing(dep)
                            updated_deps = set(deps) | set(missing)
                            self.pending_assignments[var] = (
                                expr, line_number, updated_deps, constraints)
                        continue
                    except Exception as e:
                        raise RuntimeError(
                            f"Error resolving '{var}' from '{expr}': {e} at line {line_number}")
                else:
                    constraints = assignment[3] if len(assignment) > 3 else {}
                    cell_refs = self._extract_cell_refs(expr)
                    if cell_refs and not self._grid_cells_set(cell_refs):
                        continue
                    unresolved = any(
                        self.has_unresolved_dependency(
                            dep, scope=self.current_scope())
                        for dep in deps)
                    if unresolved:
                        continue
                    try:
                        value = self.expr_evaluator.eval_or_eval_array(
                            expr, self.current_scope().get_full_scope(), line_number,
                            expected_unit=(constraints or {}).get('unit'))
                        value = self.array_handler.check_dimension_constraints(
                            var, value, line_number)
                        if constraints.get('with') and not target.startswith('['):
                            scope = self.current_scope()
                            defining_scope = scope.get_defining_scope(target)
                            type_name = None
                            if defining_scope:
                                actual_key = defining_scope._get_case_insensitive_key(
                                    target, defining_scope.types)
                                if actual_key:
                                    type_name = defining_scope.types.get(actual_key)
                            value = self._apply_with_constraints(
                                value, constraints.get('with', {}),
                                scope.get_full_scope(), line_number,
                                type_name=type_name)
                        value = self._to_sparse_undimmed(
                            value, constraints or {})
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
                        del self.pending_assignments[var]
                    except ValueError as e:
                        del self.pending_assignments[var]
                        self.grid.clear()
                    except NameError as e:
                        missing = self.extract_missing_dependencies(e)
                        if missing:
                            for dep in missing:
                                self.mark_dependency_missing(dep)
                            updated_deps = set(deps) | set(missing)
                            self.pending_assignments[var] = (
                                expr, line_number, updated_deps, constraints)
                        continue
                    except Exception as e:
                        raise RuntimeError(
                            f"Error resolving '{var}' from '{expr}': {e} at line {line_number}")
            if set(self.pending_assignments.keys()) == unresolved_before:
                break
            attempt += 1

    def _validate_pending_assignment_not_self_referential(self, var, expr, deps, line_number):
        if var in deps and not var.startswith('__line_'):
            raise ValueError(
                f"Self-referential assignment '{var} = {expr}' at line {line_number}")

    def _resolve_block_pending_assignments(self):
        block_pending = {}
        for scope in self.scopes:
            if hasattr(scope, 'pending_assignments'):
                block_pending.update(scope.pending_assignments)
        for var, assignment in sorted(block_pending.items(), key=lambda x: int(x[0].replace('__block_line_', '')) if x[0].startswith('__block_line_') else '0'):
            expr, line_number, deps = assignment[:3]
            constraints = assignment[3] if len(assignment) > 3 else {}
            scope = self.current_scope()
            unresolved = any(
                self.has_unresolved_dependency(dep, scope=scope) for dep in deps)
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
                            expr, self.current_scope().get_full_scope(), line_number,
                            expected_unit=(constraints or {}).get('unit'))
                    value = self.array_handler.check_dimension_constraints(
                        var, value, line_number)
                    if constraints.get('with'):
                        scope = self.current_scope()
                        defining_scope = scope.get_defining_scope(var)
                        type_name = None
                        if defining_scope:
                            actual_key = defining_scope._get_case_insensitive_key(
                                var, defining_scope.types)
                            if actual_key:
                                type_name = defining_scope.types.get(actual_key)
                        value = self._apply_with_constraints(
                            value, constraints.get('with', {}),
                            scope.get_full_scope(), line_number,
                            type_name=type_name)
                    value = self._to_sparse_undimmed(
                        value, constraints or {})
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
        return block_pending

    def _parse_variable_def(self, def_str, line_number):
        """Delegate to parser."""
        return self.parser._parse_variable_def(def_str, line_number)

    def _has_star_dim(self, constraints):
        """Return True if *constraints* contain an unbounded (star) dim spec."""
        dim_spec = (constraints or {}).get('dim')
        if isinstance(dim_spec, dict) and 'dims' in dim_spec:
            dim_spec = dim_spec['dims']
        if isinstance(dim_spec, list):
            for _, size_spec in dim_spec:
                if self.array_handler._is_unbounded_size_spec(size_spec):
                    return True
        if isinstance(dim_spec, str):
            return '*' in dim_spec or re.search(r'to\s+\*', dim_spec, re.I)
        return False

    def _apply_dim_base_offsets(self, var_name, indices, line_number=None):
        """Convert parsed (1-based) source indices to storage indices for a
        declared dim, honoring a range lower bound ('n to *' / 'n to m').
        """
        if not indices:
            return indices
        dims = getattr(self, 'dimensions', {}) or {}
        dim_specs = dims.get(var_name, [])
        adjusted = []
        for i, idx in enumerate(indices):
            base = 1
            if i < len(dim_specs):
                size_spec = dim_specs[i][1]
                if isinstance(size_spec, tuple) and len(size_spec) == 2:
                    base = size_spec[0]
            adjusted.append(idx - (base - 1))
        return adjusted

    def _try_let_index_assignment(self, var, expr, scope_dict, line_number,
                                  local_vars=None):
        """Shared indexed-write: ``var{a,b} = expr``, ``var![addr] = expr``,
        ``var(1) = expr``. Returns True when the target was an index target.

        When *local_vars* is provided (e.g. the type-constructor ``value_dict``),
        the variable is looked up / written back there instead of the scope
        chain.
        """
        var_name, indices = self.expr_evaluator._parse_index_target(
            var, scope_dict, line_number)
        if var_name is None:
            return False

        value = self.expr_evaluator.eval_or_eval_array(
            expr, scope_dict, line_number)

        # In a type constructor the variable may live in value_dict, not the
        # compiler scope chain.  Always prefer local_vars when provided so
        # writes go to the correct store.
        if local_vars is not None:
            local_key = None
            for k in local_vars:
                if str(k).lower() == var_name.lower():
                    local_key = k
                    break
            if local_key is not None:
                arr = local_vars[local_key]
                if arr is None:
                    arr = {}
                    local_vars[local_key] = arr
                indices = self._apply_dim_base_offsets(
                    var_name, indices, line_number)
                updated_array = self.array_handler.set_array_element(
                    arr, indices, value, line_number)
                local_vars[local_key] = updated_array
                scope_dict[local_key] = updated_array
                return True
            raise NameError(
                f"Array variable '{var_name}' not defined at line {line_number}")

        defining_scope = self.current_scope().get_defining_scope(var_name)
        if not defining_scope:
            raise NameError(
                f"Array variable '{var_name}' not defined at line {line_number}")
        if (getattr(self, '_outer_scope_read_only', False)
                and self._is_outer_scope(defining_scope)):
            raise RuntimeError(
                f"Cannot assign to '{var_name}': variables in an outer scope "
                f"are read-only inside a function at line {line_number}")

        actual_key = defining_scope._get_case_insensitive_key(
            var_name, defining_scope.variables)
        if not actual_key:
            raise NameError(
                f"Array variable '{var_name}' not defined at line {line_number}")
        arr = defining_scope.variables[actual_key]
        constraints = defining_scope.constraints.get(actual_key, {})
        if constraints and self._has_star_dim(constraints):
            if defining_scope.is_uninitialized(actual_key) or arr is None:
                arr = {}
                defining_scope.variables[actual_key] = arr
        indices = self._apply_dim_base_offsets(
            var_name, indices, line_number)
        updated_array = self.array_handler.set_array_element(
            arr, indices, value, line_number)
        defining_scope.variables[actual_key] = updated_array
        scope_dict[actual_key] = updated_array
        return True

    def _infer_declared_type(self, expr, evaluated_value, line_number):
        """Infer a declared variable's type from a constructor or its value."""
        constructor_match = None
        try:
            constructor_match = re.match(
                r'new\s+([A-Za-z][A-Za-z0-9_]*)\s*\(', str(expr), re.I)
        except Exception:
            constructor_match = None
        inferred_type = (
            constructor_match.group(1) if constructor_match
            else self.array_handler.infer_type(evaluated_value, line_number))
        if inferred_type == 'int':
            return 'number'
        return inferred_type

    def _let_values_match(self, a, b):
        """Compare two bound values, tolerating numeric vs. unit forms."""
        try:
            if isinstance(a, dict) and ('array' in a or is_sparse_array(a)):
                a = self.array_handler.flatten_array(a)
            if isinstance(b, dict) and ('array' in b or is_sparse_array(b)):
                b = self.array_handler.flatten_array(b)
            result = (a == b)
        except Exception:
            return False
        if isinstance(result, (list, tuple)):
            items = list(result)
            return bool(items) and all(bool(item) for item in items)
        return bool(result)

    def _create_declared_dim_array(self, var, type_name, constraints, line_number):
        """Materialize an array from a ``dim`` constraint with no value."""
        dim_spec = constraints.get('dim')
        if isinstance(dim_spec, dict) and 'dims' in dim_spec:
            dims = dim_spec['dims']
        elif isinstance(dim_spec, list):
            dims = dim_spec
        else:
            return None
        if not dims:
            return None
        if any(self.array_handler._is_unbounded_size_spec(size_spec)
               for _, size_spec in dims):
            return {}
        shape = []
        for _, size_spec in dims:
            if not isinstance(size_spec, int):
                return None
            shape.append(size_spec)
        if type_name and type_name.lower() in self.types_defined:
            return self.array_handler.create_object_array(
                shape, None, line_number)
        pa_type = (type_name or '').lower()
        if pa_type not in ('number', 'text', 'logical'):
            pa_type = 'number'
        return self.array_handler.create_array(
            shape, None, pa_type, line_number, template=True)

    def _process_let_binding(
            self,
            var,
            type_name,
            constraints,
            expr,
            line_number,
            search_scope=None,
            define_scope=None,
            scope_dict=None,
            shadow_keyword=None):
        """Shared logic for LET / FOR variable binding.

        Handles constraint parsing, dim/default materialization, expression
        evaluation and type coercion. Returns ``'bound'`` or ``None``.
        """
        constraints = dict(constraints or {})
        if (
            expr is not None
            and isinstance(expr, str)
            and expr.strip()
            and 'constant' not in constraints
            and 'init' not in constraints
        ):
            constraints['constant'] = expr.strip()
        elif (
            expr is not None
            and isinstance(expr, list)
            and 'constant' not in constraints
            and 'init' not in constraints
        ):
            constraints['constant'] = expr

        search_scope = search_scope or self.current_scope()
        define_scope = define_scope or search_scope
        defining_scope = search_scope.get_defining_scope(var)
        if defining_scope and self._is_outer_scope(defining_scope):
            defining_scope = None
        old_constant = None
        old_value = None
        if defining_scope:
            old_key = defining_scope._get_case_insensitive_key(
                var, defining_scope.variables)
            old_constraints_key = defining_scope._get_case_insensitive_key(
                var, defining_scope.constraints)
            if old_constraints_key:
                old_constant = defining_scope.constraints[old_constraints_key].get(
                    'constant')
            if old_key:
                old_value = defining_scope.variables[old_key]
            if var in defining_scope.constraints:
                merged = dict(defining_scope.constraints[var])
                merged.update(constraints)
                constraints = merged
            if constraints:
                defining_scope.constraints[var] = constraints
            if expr is None and var in defining_scope.variables and defining_scope.variables[var] is not None:
                return None
        else:
            if shadow_keyword and self.current_scope().is_shadowed(var):
                print(
                    f"Warning: {shadow_keyword} defines '{var}' which shadows a variable in an outer scope at line {line_number}")
            define_scope.define(
                var, None, type_name, constraints, is_uninitialized=True,
                line_number=line_number)
            defining_scope = define_scope

        if expr is None and constraints.get('dim'):
            dim_value = self._create_declared_dim_array(
                var, type_name, constraints, line_number)
            if dim_value is not None:
                search_scope.update(var, dim_value, line_number)
                if scope_dict is not None:
                    scope_dict[var] = dim_value
                return 'bound'

        if expr is None and constraints.get('default') is not None:
            try:
                default_value = self.expr_evaluator.eval_expr(
                    str(constraints['default']),
                    search_scope.get_evaluation_scope(),
                    line_number)
                if type_name:
                    default_value, constraints = defining_scope._coerce_custom_type_value(
                        type_name, default_value, constraints, line_number)
                    actual_constraint_key = defining_scope._get_case_insensitive_key(
                        var, defining_scope.constraints) or var
                    defining_scope.constraints[actual_constraint_key] = constraints
                search_scope.update(var, default_value, line_number)
                if scope_dict is not None:
                    scope_dict[var] = default_value
                return 'bound'
            except Exception:
                return None

        if expr is None:
            return None

        if 'init' not in constraints and expr is not None:
            self._register_listeners(var, expr, search_scope)

        evaluated_value = self.expr_evaluator.eval_or_eval_array(
            expr, scope_dict or search_scope.get_evaluation_scope(),
            line_number)
        if constraints.get('with'):
            evaluated_value = self._apply_with_constraints(
                evaluated_value,
                constraints.get('with', {}),
                search_scope.get_full_scope(),
                line_number,
                type_name=type_name)
        if type_name:
            evaluated_value, constraints = defining_scope._coerce_custom_type_value(
                type_name, evaluated_value, constraints, line_number)
            actual_constraint_key = defining_scope._get_case_insensitive_key(
                var, defining_scope.constraints) or var
            defining_scope.constraints[actual_constraint_key] = constraints
        elif not type_name:
            inferred_type = self._infer_declared_type(
                expr, evaluated_value, line_number)
            actual_type_key = defining_scope._get_case_insensitive_key(
                var, defining_scope.types) or var
            defining_scope.types[actual_type_key] = inferred_type
        if (
            old_constant is not None
            and constraints.get('constant') is not None
            and old_value is not None
            and not self._let_values_match(old_value, evaluated_value)
        ):
            search_scope.update(
                var, error_value(VALUE_ERROR), line_number)
            return 'bound'
        if isinstance(constraints.get('constant'), (list, tuple, dict)):
            constraints['constant'] = evaluated_value
            actual_constraint_key = defining_scope._get_case_insensitive_key(
                var, defining_scope.constraints) or var
            defining_scope.constraints[actual_constraint_key] = constraints
        search_scope.update(var, evaluated_value, line_number)
        if scope_dict is not None:
            scope_dict[var] = evaluated_value
        return 'bound'

    def _extract_identifier_tokens(self, expr):
        """Extract identifier-like tokens ignoring string literals and numeric literals."""
        if not expr:
            return set()
        cleaned = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', ' ', expr)
        cleaned = mask_text_constant_tokens(cleaned)
        # Remove builder-call names ('-> name(') so they are not treated as deps.
        cleaned = re.sub(r'->\s*\$?[A-Za-z][A-Za-z0-9_.]*\s*\(', '(', cleaned)
        # Remove '<number> of <unit>' RHS literals so the unit name and the
        # 'of' connector are not mistaken for dependency variables.
        cleaned = re.sub(
            r'(?<![\w.])(\d+(?:\.\d*)?|\.\d+)\s+of\s+(?:[A-Za-z_][A-Za-z0-9_]*|1)',
            r'\1', cleaned, flags=re.I)
        # Remove member accesses like "obj.field" or "obj.method" to avoid
        # treating field/method names as standalone dependencies.
        cleaned = re.sub(r'\.\s*[A-Za-z][A-Za-z0-9_]*', ' ', cleaned)
        tokens = re.findall(r'[A-Za-z][A-Za-z0-9_]*', cleaned)
        filtered = set()
        for tok in tokens:
            if re.match(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$', tok, re.I):
                continue
            if re.match(r'^e[+-]?\d*$', tok, re.I):
                continue
            lower_tok = tok.lower()
            if lower_tok in KEYWORDS:
                continue
            if lower_tok in getattr(self, 'types_defined', {}):
                continue
            if lower_tok in getattr(self, 'functions', {}):
                continue
            if lower_tok in getattr(self, 'subprocesses', {}):
                continue
            filtered.add(tok)
        return filtered

    def _to_index(self, cell_ref):
        """Convert a cell reference (address string or index tuple) to a
        0-based numeric index tuple for the grid store."""
        if isinstance(cell_ref, (tuple, list)):
            return tuple(int(i) for i in cell_ref)
        return tuple(i - 1 for i in parse_address(cell_ref))

    def _extract_cell_refs(self, expr):
        """Extract the grid cells an expression depends on.

        Returns 0-based numeric index tuples (the grid store's key space),
        converted from address strings: ``[A1]`` -> ``(0, 0)``,
        ``[A1:A3]`` -> ``{(0, 0), (0, 1), (0, 2)}``,
        ``grid{2, 2}`` -> ``(1, 1)``.
        """
        cell_refs = set()
        single_matches = re.finditer(rf'\[({_ADDRESS_FRAGMENT})\]', expr)
        for match in single_matches:
            cell_refs.add(tuple(i - 1 for i in parse_address(match.group(1))))
        range_matches = re.finditer(rf'\[({_ADDRESS_FRAGMENT}):({_ADDRESS_FRAGMENT})\]', expr)
        for match in range_matches:
            start, end = match.group(1), match.group(2)
            if '.' in start or '.' in end:
                # Extended (N-D) range: register every enumerated cell.
                try:
                    s_idx = parse_address(start)
                    e_idx = parse_address(end)
                except ValueError:
                    continue
                if len(s_idx) != len(e_idx):
                    continue
                starts = [min(a, b) for a, b in zip(s_idx, e_idx)]
                shape = [abs(a - b) + 1 for a, b in zip(s_idx, e_idx)]
                total = 1
                for dim in shape:
                    total *= dim
                for flat_i in range(total):
                    rem = flat_i
                    idxs = []
                    for dim_size in shape:
                        idxs.append(rem % dim_size)
                        rem //= dim_size
                    cell_refs.add(tuple(
                        starts[i] + idxs[i] - 1 for i in range(len(shape))))
                continue
            start_col, start_row = split_cell(start)
            end_col, end_row = split_cell(end)
            start_col_num = col_to_num(start_col)
            end_col_num = col_to_num(end_col)
            for col_num in range(start_col_num, end_col_num + 1):
                for row in range(int(start_row), int(end_row) + 1):
                    cell_refs.add((row - 1, col_num - 1))
        # grid{row, col} references the current grid; treat literal indices
        # as cell dependencies so assignments wait until the cell is written.
        grid_matches = re.finditer(
            r'\bgrid\s*\{\s*(\d+)\s*,\s*(\d+)\s*\}', expr, re.I)
        for match in grid_matches:
            row, col = int(match.group(1)), int(match.group(2))
            cell_refs.add((row - 1, col - 1))
        if '$"' in expr:
            # Only scan placeholders inside interpolated strings ($"...{...}").
            # Naively scanning all '{...}' groups would also match array literal
            # braces and quoted text values that look like cell refs (e.g. a text
            # array like {"q6", "x7"}).
            for ph in iter_interpolation_placeholders(expr):
                for ref in re.findall(r'\b[A-Za-z]+\d+\b', ph):
                    cell_refs.add(tuple(i - 1 for i in parse_address(ref)))
        return cell_refs

    def _register_listeners(self, var_name, expr, scope):
        """Register var_name as a client listener on every cell/var its expr reads.

        Called during a client binding; the registry lives on this compiler so
        notifications (reached via compiler references from scope/_GridStore)
        always touch the same objects.
        """
        if not expr or scope is None:
            return
        deps = set()
        cleaned = strip_array_cell_indices(_strip_constraint_operands(expr))
        cleaned = mask_text_constant_tokens(cleaned)
        for token in _IDENTIFIER_TOKEN_PATTERN.findall(cleaned):
            if not token:
                continue
            base = token.split('.')[0]
            lower = base.lower()
            if lower in _DEPENDENCY_IGNORED_TOKENS:
                # 'e' (exponent notation) is only a real dependency when it
                # stands alone as a variable reference.
                if not (lower == 'e' and self._is_standalone_var_ref(cleaned, base)):
                    continue
            if lower in self.types_defined:
                continue
            if lower in self.functions or lower in self.subprocesses:
                continue
            if re.match(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$', base, re.I):
                continue
            deps.add(base)
        cell_refs = self._extract_cell_refs(str(expr))
        keys = []
        for ref in sorted(cell_refs):
            keys.append(('cell', ref))
        for dep in deps:
            if re.match(r'^[A-Za-z]+\d+$', dep):
                continue
            keys.append(('var', dep.lower()))
        record = {'var': var_name, 'expr': str(expr), 'scope': scope,
                  'deps': deps, 'cell_refs': cell_refs}
        for kind, key in keys:
            holder = self._listeners.setdefault(kind, {}).setdefault(key, {})
            holder[(var_name.lower(), id(scope))] = record
        if var_name:
            self._set_by[('var', var_name.lower())] = 'client'

    def _register_cell_spill_listener(self, line, rhs_var, line_number, scope=None):
        """Register a cell-spill := line as a listener on rhs_var.

        When rhs_var changes (e.g. via push from a subprocess), the
        entire ``:=`` line is re-evaluated so the cell write sees the
        updated value. A ``scope`` of None means the line is a
        pre-registered constraint whose scope is resolved at recompute
        time (used so that Push notifies the constraint regardless of
        statement order).
        """
        if not rhs_var:
            return
        cell_refs = self._extract_cell_refs(str(line))
        record = {'var': None, 'expr': rhs_var, 'scope': scope,
                  'line': line, 'line_number': line_number,
                  'deps': {rhs_var}, 'cell_refs': cell_refs}
        key = ('var', rhs_var.lower())
        holder = self._listeners.setdefault(key[0], {}).setdefault(key[1], {})
        holder[('__cell_spill_' + line, id(scope))] = record

    def _notify_var_changed(self, name, value):
        self._propagate(('var', name.lower()), value)

    def _notify_cell_changed(self, cell_ref, value):
        self._propagate(('cell', cell_ref), value)

    def _propagate(self, dep_key, value):
        """Recompute every client listener that depends on dep_key."""
        if dep_key in self._propagating:
            return
        entries = self._listeners.get(dep_key[0], {}).get(dep_key[1], {})
        if not entries:
            return
        self._propagating.add(dep_key)
        try:
            for record in list(entries.values()):
                try:
                    self._recompute_client(record)
                except Exception:
                    pass
        finally:
            self._propagating.discard(dep_key)

    def _propagate_transient(self, var_name, error_code):
        """Emit a transient error to all listeners of var_name.

        The variable's stored value is temporarily replaced with the error,
        listeners recompute and store the error, then the original value
        is restored. The constraint enforces the read value on future access.
        """
        dep_key = ('var', var_name.lower())
        entries = self._listeners.get('var', {}).get(var_name.lower(), {})
        if not entries:
            return
        scope = self.current_scope()
        defining_scope = scope.get_defining_scope(var_name)
        if defining_scope is None:
            return
        actual_key = defining_scope._get_case_insensitive_key(
            var_name, defining_scope.variables)
        if actual_key is None:
            actual_key = var_name
        original = defining_scope.variables.get(actual_key)
        from units import error_value
        defining_scope.variables[actual_key] = error_value(error_code)
        self._transient_active = True
        try:
            self._propagate(dep_key, error_value(error_code))
        finally:
            defining_scope.variables[actual_key] = original
            self._transient_active = False

    def _recompute_client(self, record):
        """Recompute a client variable from its stored expression."""
        line = record.get('line')
        if line:
            scope = record.get('scope')
            if scope is None:
                scope = self.current_scope()
                for dep in record.get('deps', ()):
                    ds = scope.get_defining_scope(dep)
                    if ds is not None:
                        scope = ds
                        break
            for dep in record.get('deps', ()):
                if self.has_unresolved_dependency(dep, scope=scope):
                    return
            # Cell mirrors are equality constraints on their expression: while
            # the source holds a transient error (a rejected push), the mirror
            # keeps its last consistent value instead of storing the error.
            prev_cells = {}
            if getattr(self, '_transient_active', False):
                for ref in record.get('cell_refs', ()):
                    prev_cells[ref] = self.grid.get(ref)
            was_recomputing = getattr(self, '_in_cell_spill_recompute', False)
            self._in_cell_spill_recompute = True
            try:
                self.array_handler.evaluate_line_with_assignment(
                    line, record.get('line_number'), scope.get_evaluation_scope())
            except Exception:
                pass
            finally:
                self._in_cell_spill_recompute = was_recomputing
            if prev_cells:
                from units import is_error_value
                for ref, old in prev_cells.items():
                    new = self.grid.get(ref)
                    if is_error_value(new) and not is_error_value(old):
                        self._set_grid_cell(ref, old)
            return
        var_name = record.get('var')
        expr = record.get('expr')
        scope = record.get('scope')
        if not var_name or not expr or scope is None:
            return
        defining_scope = scope.get_defining_scope(var_name)
        if defining_scope is None or defining_scope.is_uninitialized(var_name):
            return
        # Engine-owned handles hold all their state on the Python side: the
        # client `Let h = cap.member(...)` only created the handle, so a parent
        # capability tick must never re-evaluate it as a new handle.
        try:
            cur_key = defining_scope._get_case_insensitive_key(
                var_name, defining_scope.variables)
            cur_val = defining_scope.variables.get(cur_key)
        except Exception:
            cur_val = None
        if isinstance(cur_val, dict) and cur_val.get('_handle'):
            return
        for dep in record.get('deps', ()):
            if self.has_unresolved_dependency(dep, scope=scope):
                return
        for ref in record.get('cell_refs', ()):
            if ref not in self.grid:
                return
        val = self.expr_evaluator.eval_or_eval_array(
            expr, scope.get_evaluation_scope(), None)
        actual_key = defining_scope._get_case_insensitive_key(
            var_name, defining_scope.types) or var_name
        var_type = defining_scope.types.get(actual_key)
        if val is not None and var_type and var_type.lower() in self.types_defined:
            constraints = defining_scope.constraints.get(actual_key, {})
            val, constraints = defining_scope._coerce_custom_type_value(
                var_type, val, constraints, None)
            defining_scope.constraints[actual_key] = constraints
        defining_scope.update(var_name, val, None)

    def _split_assignment_expr(self, text):
        """Split on the first standalone '=' not part of comparison operators."""
        in_quote = False
        quote_char = None
        paren_level = 0
        brace_level = 0
        bracket_level = 0
        for i, ch in enumerate(text):
            if ch in ('"', "'") and (i == 0 or text[i - 1] != '\\'):
                if not in_quote:
                    in_quote = True
                    quote_char = ch
                elif quote_char == ch:
                    in_quote = False
                    quote_char = None
            if not in_quote:
                if ch == '(':
                    paren_level += 1
                elif ch == ')':
                    paren_level = max(0, paren_level - 1)
                elif ch == '{':
                    brace_level += 1
                elif ch == '}':
                    brace_level = max(0, brace_level - 1)
                elif ch == '[':
                    bracket_level += 1
                elif ch == ']':
                    bracket_level = max(0, bracket_level - 1)
            if ch == '=' and not in_quote and paren_level == 0 and brace_level == 0 and bracket_level == 0:
                prev = text[i - 1] if i > 0 else ''
                if prev in ('<', '>', '!'):
                    continue
                return text[:i], text[i + 1:]
        return text, None

    # Dependency tracking helpers
    def mark_dependency_missing(self, name):
        if not name:
            return
        self.undefined_dependencies.add(name.lower())

    def mark_dependency_resolved(self, name):
        if not name:
            return
        self.undefined_dependencies.discard(name.lower())

    def extract_missing_dependencies(self, error):
        """Return identifiers mentioned in NameError messages for deferral."""
        pattern = re.compile(r"name '([^']+)' is not defined", re.I)
        stack = [error]
        seen = set()
        missing = set()
        while stack:
            err = stack.pop()
            if err in seen:
                continue
            seen.add(err)
            message = str(err)
            for match in pattern.finditer(message):
                missing.add(match.group(1))
            cause = getattr(err, '__cause__', None)
            context = getattr(err, '__context__', None)
            if cause:
                stack.append(cause)
            if context:
                stack.append(context)
        return missing

    def has_unresolved_dependency(self, name, scope=None, include_global_pending=True, scope_pending=None):
        if not name:
            return False
        normalized = name.lower()
        if normalized.startswith("__line_"):
            return False
        # User-defined functions are not variable dependencies
        if hasattr(self, 'functions') and normalized in (self.functions or {}):
            return False
        # Module namespace bases from `For <ns> use <module>.<vN>` are never
        # variable dependencies: they are dotted-name prefixes (`B.Foo`)
        # resolved through the function/type tables.
        if hasattr(self, 'module_namespaces') and normalized in self.module_namespaces:
            return False
        # Treat direct cell references (e.g., A1, B2) as resolved if the grid
        # already contains the cell value, even though they are not tracked in
        # the variable scope dictionary.
        if re.match(r'^[A-Za-z]+\d+$', name):
            if self._grid_cell_is_set(name):
                return False
            if self._grid_has_default():
                # With a grid default (``Let grid not null or = None``) unset
                # cells are readable (they yield the default), so they never
                # block dependent lines.
                return False
        scope = scope or self.current_scope()
        # Treat entirely undefined names as unresolved so dependent lines are deferred
        if not scope.get_defining_scope(name):
            return True
        if normalized in self.undefined_dependencies:
            return True
        if scope.is_uninitialized(name):
            return True
        if include_global_pending and name in self.pending_assignments:
            return True
        if scope_pending and name in scope_pending:
            return True
        return False

    def _seed_grid_variable(self):
        # The predefined 'grid' variable is the single grid backing store: a
        # sparse array keyed by 0-based (row, col) tuples. ``self.grid`` always
        # aliases this store, so cell-reference writes (``[A1] := x``) and
        # array writes (``grid{row, col}``) land in the same object. Main,
        # subprocess and type compilers each get their own store. Function call
        # contexts do not seed a local 'grid': it resolves through the caller's
        # scope chain like any other caller variable, so reads are live and
        # writes are rejected by the generic outer-scope read-only rule.
        if getattr(self, '_outer_scope_read_only', False):
            return
        store = getattr(self, 'grid', None)
        if not isinstance(store, _GridStore):
            store = _GridStore(self)
            self.grid = store
        self.scopes[0].variables['grid'] = store
        self.scopes[0].constraints['grid'] = {
            'dim': [('row', None), ('col', None)]}

    def _seed_predefined_resources(self):
        # Register the standard-library Resource types (e.g. Ticker) so a
        # program can `Require` them without declaring them. They are seeded
        # once per engine and survive per-run resets like all type definitions.
        for name, type_def in RESOURCES.items():
            self.types_defined[name] = dict(type_def)

    def _seed_predefined_subprocesses(self):
        # Standard-library resource control subprocesses (Ticker.Reset/Stop/
        # Start). Seeded once per engine; _extract_functions preserves them and
        # programs cannot redefine them.
        for name, defn in _PREDEFINED_SUBPROCESSES.items():
            self.subprocesses[name] = dict(defn)

    def _get_grid_store(self):
        """Return the live grid store (the predefined 'grid' variable)."""
        # Inside a type body, grid operations target the instance grid.
        ctx_stack = getattr(self, '_context_grid_stack', None)
        if ctx_stack and ctx_stack[-1] is not None:
            return ctx_stack[-1]
        # Functions (and unit sources) use the global grid via parent chain
        if getattr(self, '_outer_scope_read_only', False):
            parent = getattr(self, '_parent_scope', None)
            # Walk parent scope chain to find the caller's grid variable
            cur = parent
            while cur is not None:
                g = cur.variables.get('grid')
                if isinstance(g, _GridStore):
                    return g
                cur = getattr(cur, 'parent', None)
            # Fallback to parent compiler's store if parent scope is from caller
            if parent is not None and hasattr(parent, 'compiler'):
                try:
                    return parent.compiler._get_grid_store()
                except Exception:
                    pass
        scopes = getattr(self, 'scopes', None)
        if scopes and scopes[0].variables.get('grid') is not None:
            return scopes[0].variables['grid']
        store = getattr(self, 'grid', None)
        if isinstance(store, _GridStore):
            return store
        store = _GridStore(self)
        self.grid = store
        return store

    def _set_grid_cell(self, cell_ref, value):
        """Write a single cell (address string or index tuple) into the grid
        store."""
        self._get_grid_store()[self._to_index(cell_ref)] = value

    def _spill_value_to_cells(self, cell_ref, value, line_number=None):
        """Auto-spill an array value into the grid starting at cell_ref.

        A list of typed objects places one object per cell (horizontally);
        any other array value is spilled through the horizontal array
        assigner. A single object or scalar is written into one cell only.
        """
        if isinstance(value, dict) and 'array' in value:
            self.array_handler._assign_horizontal_array(
                cell_ref, value, "{}", line_number)
            return
        if isinstance(value, dict) and value and all(isinstance(k, tuple) for k in value.keys()):
            self.array_handler._assign_horizontal_array(
                cell_ref, value, "{}", line_number)
            return
        if isinstance(value, list):
            is_obj_array = (
                value
                and all(isinstance(item, dict) for item in value)
                and self.array_handler._find_object_array_type(value) is not None
            )
            if is_obj_array:
                for i, item in enumerate(value):
                    self._set_grid_cell(
                        offset_cell(cell_ref, i, 0), public_object_view(item))
                return
            self.array_handler._assign_horizontal_array(
                cell_ref, value, "{}", line_number)
            return
        self._set_grid_cell(cell_ref, value)

    def _grid_cell_is_set(self, cell_ref):
        """Whether a cell has been explicitly written (address string or index
        tuple)."""
        try:
            return self._to_index(cell_ref) in self._get_grid_store()
        except (TypeError, ValueError):
            return False

    def _grid_has_default(self):
        """Whether the 'grid' variable carries a default constraint (e.g.
        ``Let grid not null or = None``), so unset cells read the default
        instead of #N/A."""
        try:
            scope = self.scopes[0]
            key = scope._get_case_insensitive_key('grid', scope.constraints)
            if key:
                return scope.constraints[key].get('default') is not None
        except Exception:
            pass
        return False

    def _grid_cells_set(self, cell_refs):
        """Whether every cell reference has been explicitly written."""
        return all(self._grid_cell_is_set(ref) for ref in cell_refs)

    def _build_output_dict(self):
        """Snapshot the grid store as a cell-ref keyed output dict.

        Tuple keys are converted back to ``'A1'``-style references (extended
        addresses stay dotted, e.g. ``'A1.B1'``) and each value is converted
        to its display form, matching the shape of the previous sheet store
        returned by ``run()``.
        """
        result = {}
        for key, value in self._get_grid_store().items():
            result[indices_to_address([i + 1 for i in key])] = (
                self.array_handler.to_display_value(value))
        return result

    def _reset_state(self):
        self.grid.clear()
        # Reset the set of program-defined names for this run. Early redefinition
        # guards (dotted handle/type creators) populate this during preprocessing;
        # seeding it fresh here keeps names from leaking between runs when a single
        # compiler instance is reused (as the test runner does).
        self._program_defined_names = set()
        # This method runs on the executor object (self.compiler is the owning
        # compiler); listener/mechanism state must be reset on the compiler
        # because notification hooks are reached through compiler references.
        owner = getattr(self, 'compiler', None) or self
        owner._listeners = {'cell': {}, 'var': {}}
        owner._set_by = {}
        owner._propagating = set()
        parent_scope = getattr(self, '_parent_scope', None)
        if parent_scope is not None:
            # Subprocesses and functions reference the caller's scope chain live
            # instead of deep-copying global variables: reads resolve through the
            # parent, writes flow through to caller variables.
            self.scopes = [Scope(self, parent=parent_scope)]
        else:
            self.scopes = [Scope(self)]
        # Re-seed the predefined 'grid' variable containing the current grid.
        self._seed_grid_variable()
        self.variables = self.scopes[0].variables
        self.types = self.scopes[0].types
        self.pending_assignments = {}
        self._deferred_output_inits = {}
        self.dimensions.clear()
        self.dim_names.clear()
        self.dim_labels.clear()
        self._cell_var_map.clear()
        self._cell_array_map.clear()
        if not getattr(self, 'preserve_types_defined', False):
            self.types_defined.clear()
            # Standard-library Resources (e.g. Ticker) survive the per-run reset
            # like the 'grid' variable: programs Require them without declaring.
            if hasattr(self, '_seed_predefined_resources'):
                self._seed_predefined_resources()
        self.handled_assignments.clear()
        self.root_scope = self.current_scope()  # Always set root scope here
        if hasattr(self, 'output_values'):
            self.output_values.clear()
        # Seed only compiler-level metadata (dimensions) for function/subprocess
        # calls; variables are shared live through the parent scope chain.
        seed = getattr(self, '_seed_globals', None)
        if seed:
            try:
                self.dimensions = seed.get('dimensions', {}) or {}
                self.dim_names = seed.get('dim_names', {}) or {}
                self.dim_labels = seed.get('dim_labels', {}) or {}
            except Exception:
                pass
        # Execution scheduling metadata
        if hasattr(self, 'global_guard_line_numbers'):
            self.global_guard_line_numbers.clear()
        else:
            self.global_guard_line_numbers = set()
        self.global_guard_allows_execution = True
        if hasattr(self, 'undefined_dependencies'):
            self.undefined_dependencies.clear()
        else:
            self.undefined_dependencies = set()
        if hasattr(self, 'dependency_graph'):
            self.dependency_graph['nodes'].clear()
            self.dependency_graph['by_variable'].clear()
            self.dependency_graph['by_line'].clear()
        else:
            self.dependency_graph = {'nodes': [],
                                     'by_variable': {}, 'by_line': {}}
        if hasattr(self, 'global_guard_entries'):
            self.global_guard_entries.clear()
        else:
            self.global_guard_entries = []
        if hasattr(self, 'global_for_line_numbers'):
            self.global_for_line_numbers.clear()
        else:
            self.global_for_line_numbers = set()
        if hasattr(self, 'executed_global_for_lines'):
            self.executed_global_for_lines.clear()
        else:
            self.executed_global_for_lines = set()
        if hasattr(self, 'global_for_entries'):
            self.global_for_entries.clear()
        else:
            self.global_for_entries = []
        # Required-capability state is per-run (grants come from the host and
        # survive across runs so the same grant file can drive multiple programs).
        if hasattr(self, 'requirements'):
            self.requirements.clear()
        else:
            self.requirements = []
        if hasattr(self, 'require_caps'):
            self.require_caps.clear()
        else:
            self.require_caps = {}
        # Ticker firing state (tickers advance via the ordinary When/Push
        # machinery; only the clock schedule is kept here).
        if hasattr(self, '_ticker_last_fire'):
            self._ticker_last_fire.clear()
        else:
            self._ticker_last_fire = {}
        # When-blocks and their Push queues belong to a single run: clear them
        # so stale entries from a previous program never fire again.
        self.when_blocks = []
        self._push_queues = {}
        self._processing_when = False
        self._loop_iteration = 0
        # Module-import state is run-scoped except `module_sources` (host input,
        # preserved across runs like `grants`).
        if hasattr(self, 'module_namespaces'):
            self.module_namespaces.clear()
        else:
            self.module_namespaces = set()
        if hasattr(self, '_module_harvests'):
            self._module_harvests.clear()
        else:
            self._module_harvests = {}
        if hasattr(self, 'module_use_line_numbers'):
            self.module_use_line_numbers.clear()
        else:
            self.module_use_line_numbers = set()
        if hasattr(self, 'module_instances'):
            self.module_instances.clear()
        else:
            self.module_instances = {}
        if hasattr(self, '_module_export_bindings'):
            self._module_export_bindings.clear()
        else:
            self._module_export_bindings = {}

    def _preprocess_code(self, code):

        if not hasattr(self, '_program_defined_names'):
            self._program_defined_names = set()
        lines = []
        label_lines = []
        dim_lines = []
        type_def_lines = []

        in_type_def = False
        type_name = None
        type_parent = None
        type_constraints = {}
        in_unit_source = False
        unit_source_name = None
        unit_source_target = None
        unit_source_lines = []
        line_number = 0
        current_line = ""
        in_multiline = False
        continuation_line = ""
        in_continuation = False

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

            # Normalize INIT statements into LET assignments.
            init_match = re.match(r'^(\s*)init\b(.*)$', s, re.I)
            if init_match:
                leading_ws, rest = init_match.groups()
                s = f"{leading_ws}Let{rest}"

            # Preserve WHEN blocks for runtime handling.

            # Normalize brackets like [ A 12 ] → [A12]
            s = re.sub(r'\[\s*([A-Z]+)\s+[A-Z]*(\d+)\s*\]', r'[\1\2]', s)

            if not in_multiline:
                # Handle underscore line continuation (e.g., array literals)
                if in_continuation:
                    s = f"{continuation_line} {s.lstrip()}".rstrip()
                    if s.endswith('_'):
                        continuation_line = s[:-1].rstrip()
                        continue
                    continuation_line = ""
                    in_continuation = False
                elif s.endswith('_'):
                    continuation_line = s[:-1].rstrip()
                    in_continuation = True
                    continue

            # Handle start of type definition
            if s.lower().startswith("define "):
                # UnitSource definitions are collected and registered eagerly
                # so top-level Convert lines can reference their constants.
                us_name, us_target = self._parse_unit_source_header(
                    s, line_number)
                if us_name:
                    in_unit_source = True
                    unit_source_name = us_name
                    unit_source_target = us_target
                    unit_source_lines = []
                    continue
                parsed_name, parsed_parent, parsed_constraints = self._parse_type_header(
                    s, line_number)
                if parsed_name:
                    if parsed_name.lower() in RESOURCES:
                        raise SyntaxError(
                            f"'{parsed_name}' is a predefined resource and cannot "
                            f"be redefined at line {line_number}")
                    # Early redefinition guard for handles/types with dotted/bang names
                    # (e.g. ticker.counter / ticker!counter). This catches the
                    # engine-owned handle before any truncation to the bare
                    # resource prefix (ticker) could happen. Check both dot and
                    # bang forms against the engine sets and already-defined names.
                    _dotted = parsed_name.lower().replace('!', '.')
                    _bang = parsed_name.lower().replace('.', '!')
                    _prog = getattr(self, '_program_defined_names', set())
                    if (_dotted in _DOTTED_ENGINE_CREATORS
                            or _bang in _DOTTED_ENGINE_CREATORS
                            or _dotted in _PREDEFINED_SUBPROCESSES
                            or _bang in _PREDEFINED_SUBPROCESSES
                            or _dotted in _prog
                            or _bang in _prog
                            or parsed_name.lower() in _prog):
                        raise SyntaxError(
                            f"'{parsed_name}' is already defined and cannot "
                            f"be redefined at line {line_number}. ' not allowed'")
                    # Reserve the name immediately so later defines (type or
                    # function) see it via _program_defined_names, even though
                    # the actual storage happens at End.
                    _prog = getattr(self, '_program_defined_names', set())
                    if not hasattr(self, '_program_defined_names'):
                        self._program_defined_names = set()
                        _prog = self._program_defined_names
                    _prog.add(_dotted)
                    _prog.add(_bang)
                    _prog.add(parsed_name.lower())
                    in_type_def = True
                    type_name = parsed_name
                    type_parent = parsed_parent
                    type_constraints = parsed_constraints or {}
                    type_def_lines = []
                    type_inner_requires = []
                    type_block_depth = 0
                    continue
                # Other definitions (functions/subprocesses) are handled later
                in_type_def = False

            # Handle lines inside a UnitSource definition block
            if in_unit_source:
                stripped_us = s.strip()
                if stripped_us.lower().startswith('end'):
                    in_unit_source = False
                    self._finalize_unit_source(
                        unit_source_lines, unit_source_name,
                        unit_source_target, line_number)
                    continue
                unit_source_lines.append(s.lstrip())
                continue

            # Handle lines inside a type definition block
            if in_type_def:
                stripped = s.strip()
                stripped_lower = stripped.lower()
                end_pattern = rf'^\s*end(\s+type|\s+{re.escape(type_name)})?\s*$'
                is_handle_def = bool(type_constraints and type_constraints.get('is_handle'))
                if (_first_keyword(stripped) in ('for','when') and stripped_lower.endswith('do')) or (
                    _first_keyword(stripped) == 'if' and stripped_lower.endswith('then')
                ) or (
                    re.match(r'^\s*let\b', stripped, re.I) and stripped_lower.endswith('then')
                ):
                    type_block_depth += 1
                if stripped_lower.startswith('end'):
                    is_end_match = bool(re.match(end_pattern, s, re.I))
                    # For handles relax name check: any End at depth 0 closes the handle
                    if is_handle_def and not is_end_match and type_block_depth == 0:
                        if stripped_lower == 'end' or stripped_lower.startswith('end '):
                            is_end_match = True
                    if is_end_match and type_block_depth == 0:
                        in_type_def = False
                        type_def = self._parse_type_def(
                            type_def_lines, line_number, type_name)
                        if type_parent:
                            type_def['_parent'] = type_parent
                        if type_constraints:
                            type_def['_constraints'] = type_constraints
                        if type_constraints and type_constraints.get('key'):
                            type_def['_keyed'] = True
                        if type_inner_requires:
                            type_def['_inner_requires'] = type_inner_requires
                        # Handle types: canonicalize to Resource!Handle form for storage
                        store_name = type_name
                        if type_constraints and type_constraints.get('is_handle'):
                            # Canonicalize: Resource.Name -> Resource!Name
                            resource_for_handle = ""
                            if '.' in store_name:
                                parts = store_name.split('.', 1)
                                resource_for_handle = parts[0]
                                store_name = f"{parts[0]}!{parts[1]}"
                            # Ensure handle_resource is persisted (derived from
                            # the dot/bang prefix of the define name)
                            if resource_for_handle and not type_def.get('_constraints', {}).get('handle_resource'):
                                type_def.setdefault('_constraints', {})['handle_resource'] = resource_for_handle
                        self.types_defined[store_name.lower()] = type_def
                        continue
                    if type_block_depth > 0:
                        type_block_depth = max(0, type_block_depth - 1)
                # Capture inner Require statements inside a Resource definition
                # as templates (not executable code). They are materialized as
                # grants when the resource is itself required with parameters.
                if type_constraints.get('is_resource') and type_block_depth == 0 and stripped_lower.startswith('require '):
                    raw = stripped[len('Require'):].strip() if stripped.lower().startswith('require ') else stripped[7:].strip()
                    # Re-use the same Require syntax as top-level
                    m = re.match(
                        r'^([\w_]+(?:\s*,\s*[\w_]+)*)\s+as\s+([A-Za-z][\w_]*)(?:\s+with\s*\((.*)\)\s*)?$',
                        raw, re.I | re.S)
                    if not m:
                        raise SyntaxError(
                            f"Invalid Require syntax inside resource '{type_name}' at line {line_number}: '{s}'\n"
                            "Expected: Require <name> as <Resource> [with (param = value, ...)]")
                    names_part = m.group(1).strip()
                    resource_type = m.group(2).strip()
                    with_content = (m.group(3) or '').strip()
                    params = {}
                    if with_content:
                        with_kind, with_payload = self._parse_with_clause(
                            with_content, line_number)
                        if with_kind != 'named':
                            raise SyntaxError(
                                f"Invalid parameter in Require inside resource '{type_name}' at line {line_number}: "
                                "'with (...)' must list 'field = value' pairs")
                        for field, val_expr in with_payload.items():
                            params[field.lower()] = val_expr.strip()
                    var_names = [n.strip() for n in names_part.split(',') if n.strip()]
                    for var_name in var_names:
                        if not re.match(r'^[\w_]+$', var_name):
                            raise SyntaxError(
                                f"Invalid capability name '{var_name}' in Require inside resource '{type_name}' at line {line_number}")
                        type_inner_requires.append({
                            'var_name': var_name,
                            'name': var_name,
                            'resource': resource_type,
                            'resource_lower': resource_type.lower(),
                            'params': dict(params),
                            'line_number': line_number,
                        })
                    continue
                type_def_lines.append(s.lstrip())
                continue

            # Handle top-level Convert instruction (outside a UnitSource block):
            # keep it out of the normal declaration/dependency flow and defer
            # registration until the scope / UnitSource constants are ready, so
            # the RHS target unit can be inferred by evaluation.
            if re.match(r'^\s*convert\s+', s, re.I):
                self._top_level_converts.append((s, line_number))
                continue

            def _has_unclosed_interpolation(text):
                start = text.find('$"')
                if start == -1:
                    return False
                i = start + 2
                while i < len(text):
                    ch = text[i]
                    if ch == '"' and (i == 0 or text[i - 1] != '\\'):
                        return False
                    i += 1
                return True

            # Handle multiline declarations (e.g., : template = $" ... multiline ... ")
            if s.startswith(':') and _has_unclosed_interpolation(s):
                current_line = s
                in_multiline = True
                continue
            # Handle multiline assignments (e.g., [^A1] := $" ... multiline ... ")
            if ':=' in s and s.startswith('[') and _has_unclosed_interpolation(s):
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

        self._resolve_type_inheritance()
        lines = self._normalize_inline_blocks(lines)
        return lines, label_lines, dim_lines

    def _normalize_inline_blocks(self, lines):
        """Rewrite inline block statements with a single-instruction payload into
        real block form so the standard block machinery executes them identically
        to the multi-line spelling.

        ``For n in 1 to 3 do push x = n`` becomes::

            For n in 1 to 3 do
              Push x = n
            End

        and ``If a > 2 then push x = a * 10 else push x = 7`` becomes an If
        block with Then/Else bodies. Inline If clauses are expanded to full
        block form, so ``If a > 2 then ... elseif ... then ... else ...`` and
        the mixed forms where a clause body is inline (``Else push x = 3``, or
        an inline Then followed by a block ``Else`` terminated by ``End``)
        reduce to the same canonical ``If / ElseIf / Else / End`` shape. The
        payload is normalized for any single instruction (Return, Push, ``:=``,
        Let, Output, ...); the whole line is left untouched when it does not
        describe a reducible inline block. Generated lines keep the original
        line number so diagnostics and dependency bookkeeping stay anchored.
        """
        normalized = []
        for raw_line, line_number in lines:
            expanded = self._expand_inline_block_line(raw_line, line_number)
            if expanded is None:
                normalized.append((raw_line, line_number))
            else:
                normalized.extend(expanded)
        return normalized

    def _expand_inline_block_line(self, raw_line, line_number):
        """Return the block-form lines for an inline block statement with a
        single-instruction payload, else None."""
        stripped = raw_line.strip()
        if not stripped:
            return None
        indent = raw_line[:len(raw_line) - len(raw_line.lstrip())]
        body_indent = indent + '  '

        loop_match = _INLINE_BLOCK_LOOP_RE.match(stripped)
        if loop_match:
            keyword, header, action = loop_match.groups()
            action = action.strip()
            if not self._is_single_inline_instruction(action):
                return None
            return [
                (f"{indent}{keyword} {header.strip()} do", line_number),
                (f"{body_indent}{action}", line_number),
                (f"{indent}End", line_number),
            ]

        # Standalone inline Else clause: ``Else push x = 3`` closes the If with
        # an inline body (the grammar puts no End after it).
        else_match = _INLINE_ELSE_ACTION_RE.match(stripped)
        if else_match:
            action = else_match.group(1).strip()
            if not self._is_single_inline_instruction(action):
                return None
            return [
                (f"{indent}Else", line_number),
                (f"{body_indent}{action}", line_number),
                (f"{indent}End", line_number),
            ]

        if_match = _INLINE_BLOCK_IF_RE.match(stripped)
        if if_match:
            keyword, condition, payload = if_match.groups()
            clauses = self._split_inline_if_clauses(payload)
            if clauses is None:
                return None
            return self._expand_inline_if_clauses(
                indent, body_indent, keyword, condition, clauses, line_number)

        return None

    def _expand_inline_if_clauses(self, indent, body_indent, keyword,
                                  condition, clauses, line_number):
        """Build block-form lines for an inline If/ElseIf chain."""
        expanded = [(f"{indent}{keyword} {condition.strip()} then",
                     line_number)]
        for clause in clauses:
            kind, cond, action = clause
            if kind == 'elseif':
                expanded.append(
                    (f"{indent}ElseIf {cond.strip()} then", line_number))
            elif kind == 'else':
                expanded.append((f"{indent}Else", line_number))
            if action:
                expanded.append((f"{body_indent}{action.strip()}", line_number))
        # An ElseIf is a clause of the enclosing If (its End comes from the
        # enclosing block), but a top-level inline If chain is closed by an End
        # only when the last clause body is a block. When the last clause ends
        # with a bare ``else`` (a block body follows on the next lines), the
        # End comes from the enclosing block.
        if keyword.lower() == 'if':
            last_kind = clauses[-1][0]
            last_action = clauses[-1][2]
            has_block_else = last_kind == 'else' and not last_action
            if not has_block_else:
                expanded.append((f"{indent}End", line_number))
        return expanded

    @staticmethod
    def _is_single_inline_instruction(action):
        """True when *action* can be a block body: one instruction line that is
        not itself starting an inline block statement."""
        return bool(
            action
            and not re.match(r'^\s*(for|when)\b', action, re.I)
            and not re.match(r'^\s*(elseif|if)\b', action, re.I)
            and not re.match(r'^\s*else\b', action, re.I))

    @classmethod
    def _split_inline_if_clauses(cls, payload):
        """Split an inline If payload into clause tuples.

        Returns a list of ``(kind, cond, action)`` where ``kind`` is one of
        ``'then'``, ``'elseif'`` or ``'else'``. ``cond`` is only set for
        ``elseif``; ``action`` is the clause body. A trailing bare ``else``
        (no action) means a block body follows on the next lines. Returns None
        when the payload is not a reducible inline If chain (for example an
        action that is itself an inline block statement). Clause separators
        inside string literals do not split the chain.
        """
        pieces = cls._split_clause_keywords(payload)
        head = pieces[0].strip() if pieces else ''
        if not cls._is_single_inline_instruction(head):
            return None
        clauses = [('then', None, head)]
        for idx in range(1, len(pieces), 2):
            sep = pieces[idx].strip().lower()
            text = pieces[idx + 1].strip()
            if sep == 'elseif':
                then_parts = cls._split_after_then(text)
                if len(then_parts) != 2:
                    return None
                cond, action = then_parts
                if not cls._is_single_inline_instruction(action):
                    return None
                clauses.append(('elseif', cond, action))
            else:
                if text and not cls._is_single_inline_instruction(text):
                    return None
                clauses.append(('else', None, text))
        return clauses

    @staticmethod
    def _mask_string_literals(text, mask_char='\x00'):
        """Replace string literals in *text* with *mask_char* repeats of the same
        length so keyword splitting never fires on the literal contents and slice
        positions stay aligned with the original text."""
        return _STRING_LITERAL_PATTERN.sub(
            lambda m: mask_char * (m.end() - m.start()), text)

    @classmethod
    def _split_clause_keywords(cls, payload):
        """Split *payload* on ``elseif``/``else`` outside string literals,
        returning a list with the separators at odd indexes (``re.split`` of
        ``_INLINE_IF_CLAUSE_SPLIT_RE`` shape). Match positions come from a
        masked copy so literal contents never split the chain; the returned
        pieces keep the original text."""
        masked = cls._mask_string_literals(payload)
        pieces = []
        start = 0
        for m in _INLINE_IF_CLAUSE_SPLIT_RE.finditer(masked):
            pieces.append(payload[start:m.start()])
            pieces.append(m.group(0))
            start = m.end()
        pieces.append(payload[start:])
        return pieces

    @classmethod
    def _split_after_then(cls, text):
        """Split an ElseIf clause on its ``then`` keyword outside string
        literals, returning ``[cond, action]`` with the original text kept.
        Returns a single-element list when no ``then`` is found."""
        masked = cls._mask_string_literals(text)
        m = _INLINE_IF_THEN_SPLIT_RE.search(masked)
        if not m:
            return [text.strip()]
        return [text[:m.start()].strip(), text[m.end():].strip()]

    def _parse_type_def(self, lines, line_number=None, type_name=None):
        """Delegate to type processor."""
        return self.type_processor._parse_type_def(lines, line_number, type_name)

    def _execute_type_code(self, code_lines, var_name, value_dict, line_number, input_values=None):
        """Delegate to type processor."""
        return self.type_processor._execute_type_code(code_lines, var_name, value_dict, line_number, input_values)

    def _process_grid_assignment(self, line, var_name, value_dict, line_number):
        """Delegate to type processor."""
        return self.type_processor._process_grid_assignment(line, var_name, value_dict, line_number)

    def _process_type_for_loop(self, loop_line, all_lines, var_name, value_dict, line_number):
        """Delegate to type processor."""
        return self.type_processor._process_type_for_loop(loop_line, all_lines, var_name, value_dict, line_number)

    def _process_type_let_statement(self, line, var_name, value_dict, line_number):
        """Delegate to type processor."""
        return self.type_processor._process_type_let_statement(line, var_name, value_dict, line_number)

    # ======================================================================
    # Module loading (see docs/modules.md). The "Import/load first (use)"
    # milestone: `use Module.<tag>` / `For <ns> use Module.<tag>` bind exported
    # definitions (functions, subprocesses, types, unit categories) at load
    # time. `use` lines never reach the dependency network, global-for
    # machinery, or the main loop: they are collected as a pre-pass and
    # filtered out of the executable line stream.
    # ======================================================================

    def _load_module_harvest(self, module_name, line_number=None):
        """Locate, parse and harvest a module's definitions (load, never run).

        The module body is parsed by a fresh compiler that never executes a
        main loop and never runs `_process_declarations_and_labels`: it only
        preprocesses (registering types/unit sources) and extracts function
        and subprocess definitions. Returns a dict with:

          - header: parsed `Module <name>` header or None,
          - module_version: parsed `: version` entry or None,
          - exports_by_tag: tag.lower() -> list of exported definition names,
          - functions / subprocesses: code tables of the module (raw names),
          - types_defined / unit_sources: type and unit-category tables.

        Results are cached per import (per compiler x physical copy).
        """
        key = module_name.lower()
        if key in self._module_harvests:
            return self._module_harvests[key]

        if not hasattr(self, '_module_harvests'):
            self._module_harvests = {}
        source = resolve_module_source(
            module_name, getattr(self, 'module_sources', None) or {})
        if not source or not source.strip():
            raise ModuleImportError(
                f"Module '{module_name}' is empty (no source text but an "
                "empty module cannot export definitions)")
        metadata = {'header': None, 'module_version': None,
                    'exports_by_tag': {}}
        body_source = []
        for raw_line in source.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("'"):
                continue
            header = parse_module_header(stripped)
            if header is not None:
                metadata['header'] = header
                continue
            version = parse_module_version_line(stripped)
            if version is not None:
                metadata['module_version'] = version
                continue
            version_block = parse_version_block(stripped)
            if version_block is not None:
                metadata['exports_by_tag'].setdefault(
                    version_block['tag'].lower(), version_block['exports'])
                continue
            body_source.append(raw_line)
        if not metadata['exports_by_tag']:
            # A module with no Version blocks exposes no importable API.
            raise ModuleImportError(
                f"Module '{module_name}' declares no 'Version' exports; it "
                "cannot be imported" +
                (f" at line {line_number}" if line_number else ""))

        # Harvest definitions with a fresh compiler that loads but never runs.
        loader = GridLangCompiler()
        loader.module_sources = getattr(self, 'module_sources', None) or {}
        loader._is_module_loader = True
        try:
            all_lines, label_lines, dim_lines = loader._preprocess_code(
                "\n".join(body_source))
            remaining, label_lines, dim_lines = loader._extract_functions(
                all_lines, label_lines, dim_lines)
            # Instantiate the module's top-level declarations into the loader's
            # global scope to obtain the per-instance initial state. Blocks and
            # Push-family statements are left for the (never-run) main loop,
            # so equality-bound variables get values and push-only ones stay
            # uninitialized (#N/A), exactly as §6 of docs/modules.md requires.
            loader._process_declarations_and_labels(
                remaining, label_lines, dim_lines)
            try:
                loader._resolve_pending_assignments()
            except Exception:
                pass
        except ModuleImportError:
            raise
        except Exception as exc:
            raise ModuleImportError(
                f"Module '{module_name}' is not a valid loadable module: "
                f"{exc}") from exc

        harvest = dict(metadata)
        harvest['functions'] = getattr(loader, 'functions', {})
        harvest['subprocesses'] = getattr(loader, 'subprocesses', {})
        harvest['types_defined'] = getattr(loader, 'types_defined', {})
        harvest['unit_sources'] = getattr(loader, 'unit_sources', {})
        harvest['state_scope'] = loader.current_scope()
        harvest['module_vars'] = self._snapshot_module_vars(loader)
        self._module_harvests[key] = harvest
        return harvest

    def _snapshot_module_vars(self, loader):
        """Capture the module's top-level variable state for one instance."""
        scope = loader.current_scope()
        vars_ = {}
        for key in list(scope.variables.keys()):
            if key.lower() == 'grid':
                continue
            if scope.is_input(key) or scope.is_output(key):
                continue
            try:
                value = scope.get(key)
            except Exception:
                value = None
            vars_[key.lower()] = {
                'value': value,
                'uninitialized': scope.is_uninitialized(key),
                'type': scope.types.get(
                    scope._get_case_insensitive_key(key, scope.types) or key),
            }
        return vars_

    def _ensure_module_instance(self, module_name, harvest):
        """Return the live instance scope for a module (created on first use)."""
        key = module_name.lower()
        inst = self.module_instances.get(key)
        if inst is None:
            inst = {'scope': harvest.get('state_scope'),
                    'module_vars': harvest.get('module_vars') or {},
                    'exported': {}}
            self.module_instances[key] = inst
        return inst

    def _sync_module_export_vars(self, module_key):
        """Refresh the importer's copies of a module's exported variables."""
        inst = self.module_instances.get(module_key)
        bindings = self._module_export_bindings.get(module_key)
        if not inst or not bindings or inst['scope'] is None:
            return
        scope = self.current_scope()
        for program_key, instance_var in bindings.items():
            if inst['scope'].is_uninitialized(instance_var):
                continue
            try:
                value = inst['scope'].get(instance_var)
            except Exception:
                value = None
            if value is None:
                continue
            actual = scope._get_case_insensitive_key(program_key, scope.variables)
            if actual:
                scope.variables[actual] = value

    def _process_module_imports(self, lines):
        """Bind `use` / `For <ns> use` statements and drop them from `lines`.

        Processed before any block depth is established here (they are
        top-level-only, like `Require`); the remaining lines are filtered by
        line number so downstream machinery never sees an import statement.
        """
        remaining = []
        depth = 0
        type_depth = 0
        if getattr(self, '_is_module_loader', False):
            for line, line_number in lines:
                if parse_use_line(line) is not None:
                    raise ModuleImportError(
                        f"A module cannot import another module yet; 'use' "
                        f"inside module source at line {line_number} is not "
                        f"supported")
        for line, line_number in lines:
            stripped = line.strip()
            lower = stripped.lower()
            is_block_start = (
                (lower.startswith('if ') and lower.endswith('then')) or
                (lower.startswith('for ') and lower.endswith('do')) or
                (lower.startswith('when ') and lower.endswith('do'))
            )
            is_end = lower == 'end'
            if lower.startswith("define ") and (
                    " as type" in lower or " as keytype" in lower or
                    " as resource" in lower or " as handle" in lower):
                type_depth += 1
            elif lower.startswith("end") and type_depth > 0:
                type_depth -= 1

            parsed = parse_use_line(line)
            if parsed is not None:
                if depth != 0 or type_depth != 0:
                    raise ModuleImportError(
                        f"'use' is only allowed at the top (global) level "
                        f"at line {line_number}")
                self.module_use_line_numbers.add(line_number)
                self._collect_module_import(line, line_number, parsed)
                continue
            remaining.append((line, line_number))
            if is_block_start:
                depth += 1
            elif is_end and depth > 0:
                depth -= 1
        lines[:] = remaining

    def _collect_module_import(self, line, line_number, parsed):
        """Bind a single import statement, exporting a whole API version.

        Raises `ModuleImportError` for unknown modules/tags, unsatisfied
        version pins, collisions, and definitions outside this milestone.
        """
        module_name = parsed['module']
        tag = parsed['tag']
        namespace = parsed['namespace']
        harvest = self._load_module_harvest(module_name, line_number)

        if parsed['pin_op'] is not None:
            matched = version_pin_matches(
                parsed, harvest['module_version'], line_number)
            if not matched:
                raise ModuleImportError(
                    f"Module '{module_name}' does not satisfy version pin "
                    f"'version{parsed['pin_op']}{parsed['pin_value']}' at "
                    f"line {line_number}")

        exports = harvest['exports_by_tag'].get(tag.lower())
        if exports is None:
            raise ModuleImportError(
                f"Module '{module_name}' has no 'Version {tag}' to import at "
                f"line {line_number}")

        if namespace:
            self.module_namespaces.add(namespace.lower())

        for export_name in exports:
            bound_name, _stripped = strip_version_tag(export_name, tag)
            if namespace:
                bound_name = f"{namespace}.{bound_name}"
            raw_key = export_name.lower()
            if raw_key in harvest['types_defined']:
                self._bind_module_type(module_name, export_name, bound_name,
                                       harvest, line_number)
            elif raw_key in harvest['subprocesses']:
                self._bind_module_def(module_name, export_name, bound_name,
                                      harvest, line_number, 'subprocesses')
            elif raw_key in harvest['functions']:
                self._bind_module_def(module_name, export_name, bound_name,
                                      harvest, line_number, 'functions')
            elif raw_key in harvest['unit_sources']:
                if namespace:
                    raise ModuleImportError(
                        f"Namespaced unit-category export '{export_name}' of "
                        f"module '{module_name}' is not yet supported at line "
                        f"{line_number}")
                self._bind_module_units(module_name, export_name, bound_name,
                                        harvest, line_number)
            elif raw_key in harvest['module_vars']:
                if namespace:
                    raise ModuleImportError(
                        f"Namespaced top-level variable export '{export_name}' "
                        f"of module '{module_name}' is not yet supported at "
                        f"line {line_number}")
                self._bind_module_var(module_name, export_name, bound_name,
                                      harvest, line_number)
            else:
                raise ModuleImportError(
                    f"Module '{module_name}' Version '{tag}' exports "
                    f"'{export_name}' which is not defined at line "
                    f"{line_number}")

    def _bind_module_def(self, module_name, raw_name, bound_name, harvest,
                         line_number, table):
        """Bind a harvested function/subprocess entry under the export name."""
        target_table = {'functions': self.functions,
                        'subprocesses': self.subprocesses}[table]
        bound_key = bound_name.lower()
        if bound_key in target_table:
            raise ModuleImportError(
                f"Name collision on import at line {line_number}: "
                f"'{bound_name}' is already defined")
        entry = dict(harvest[table][raw_name.lower()])
        if table == 'subprocesses':
            # Subprocesses are the module's mutation channel: they run against
            # the module's live instance so `Push`/assignments modify module
            # state (docs §11). Functions keep caller-scope resolution.
            inst = self._ensure_module_instance(module_name, harvest)
            entry['defining_scope'] = inst['scope']
            entry['module_key'] = module_name.lower()
        else:
            # Imported definitions bind at load time and execute against the
            # importer's scope, like locally-defined ones: drop the loader's
            # captured defining scope so call_function falls back to the caller.
            entry['defining_scope'] = None
        target_table[bound_key] = entry

    def _bind_module_var(self, module_name, raw_name, bound_name, harvest,
                         line_number):
        """Bind an exported top-level variable read-only into the scope."""
        inst = self._ensure_module_instance(module_name, harvest)
        bound_key = bound_name.lower()
        scope = self.current_scope()
        defining = scope.get_defining_scope(bound_key)
        if defining is not None and not self._is_outer_scope(defining):
            raise ModuleImportError(
                f"Name collision on import at line {line_number}: "
                f"'{bound_name}' is already defined")
        meta = inst['module_vars'][raw_name.lower()]
        self._module_export_bindings.setdefault(
            module_name.lower(), {})[bound_key] = raw_name.lower()
        inst.setdefault('exported', {})[bound_key] = raw_name.lower()
        scope.define(
            bound_key, meta.get('value'), meta.get('type') or 'unknown',
            {'module_export': module_name.lower()},
            is_uninitialized=bool(meta.get('uninitialized')),
            line_number=line_number)

    def _bind_module_type(self, module_name, raw_name, bound_name, harvest,
                          line_number):
        """Bind an exported type and its member functions under one name."""
        bound_key = bound_name.lower()
        if bound_key in self.types_defined:
            raise ModuleImportError(
                f"Name collision on import at line {line_number}: type "
                f"'{bound_name}' is already defined")
        self.types_defined[bound_key] = dict(harvest['types_defined'][raw_name.lower()])
        raw_key = raw_name.lower()
        for func_key, entry in harvest['functions'].items():
            if (entry.get('member_of') or '').lower() != raw_key:
                continue
            if not func_key.startswith(raw_key):
                continue
            suffix = func_key[len(raw_key):]
            new_key = f"{bound_key}{suffix}".lower()
            if new_key in self.functions:
                raise ModuleImportError(
                    f"Name collision on import at line {line_number}: member "
                    f"'{bound_key}{suffix}' is already defined")
            new_entry = dict(entry)
            new_entry['defining_scope'] = None
            new_entry['member_of'] = bound_key
            self.functions[new_key] = new_entry

    def _bind_module_units(self, module_name, raw_name, bound_name, harvest,
                           line_number):
        """Bind an exported unit-category source under its export name."""
        bound_key = bound_name.lower()
        if bound_key in self.unit_sources:
            raise ModuleImportError(
                f"Name collision on import at line {line_number}: unit "
                f"category '{bound_name}' is already defined")
        self.unit_sources[bound_key] = harvest['unit_sources'][raw_name.lower()]

    def _process_declarations_and_labels(self, lines, label_lines, dim_lines):
        self._process_module_imports(lines)
        for line, line_number in dim_lines:
            self._collect_global_declarations(line, line_number)

        for line, line_number in lines:
            stripped = line.strip()
            lowered = stripped.lower()
            if not lowered.startswith('for '):
                continue
            if re.search(r'\bdo\b', lowered):
                continue
            if re.search(r'\bgrid\s+dim\b', lowered):
                continue
            var_match = re.match(r'^\s*for\s+([A-Za-z][A-Za-z0-9_]*)\b', stripped, re.I)
            dim_match = re.search(r'\bdim\s+(\{[^}]+\})', stripped, re.I)
            if not (var_match and dim_match):
                continue
            var = var_match.group(1)
            dim_expr = dim_match.group(1).strip()
            dim_parts = dim_expr[1:-1].split(',') if dim_expr.startswith('{') else [dim_expr]
            dims = []
            for part in dim_parts:
                part = part.strip()
                if not part:
                    continue
                if ':' in part:
                    name, size = map(str.strip, part.split(':', 1))
                    size_spec = self._parse_dim_size(size, line_number)
                    dims.append((name, size_spec))
                else:
                    size_spec = self._parse_dim_size(part, line_number)
                    dims.append((None, size_spec))
            if dims:
                self.dimensions.setdefault(var, dims)
                self.dim_names.setdefault(
                    var, {name: idx for idx, (name, _) in enumerate(dims) if name})
                self.dim_labels.setdefault(var, {})

        # Track block depth to avoid treating nested ':' declarations as global
        depth = 0
        type_depth = 0
        for line, line_number in lines:
            lstrip_line = line.lstrip()
            stripped = lstrip_line.strip().lower()
            is_block_start = (
                (stripped.startswith('if ') and stripped.endswith('then')) or
                (stripped.startswith('for ') and stripped.endswith('do')) or
                (stripped.startswith('when ') and stripped.endswith('do'))
            )
            is_end = stripped == 'end'

            # Track entering/exiting type definitions to avoid treating inner lines as globals
            if stripped.startswith("define ") and (" as type" in stripped or " as keytype" in stripped or " as resource" in stripped or " as handle" in stripped):
                type_depth += 1
            elif stripped.startswith("end") and type_depth > 0:
                type_depth -= 1
                continue

            # Require is a top-level-only statement: it may not appear inside a
            # block (loop/condition/when) or inside a type definition.
            if stripped.startswith("require "):
                if depth != 0 or type_depth != 0:
                    raise SyntaxError(
                        f"Require is only allowed at the top (global) level at line {line_number}")
                self._collect_global_declarations(lstrip_line, line_number)
                if is_block_start:
                    depth += 1
                elif is_end and depth > 0:
                    depth -= 1
                continue

            if depth == 0 and type_depth == 0:
                if (lstrip_line.startswith(':') and not lstrip_line.lower().startswith(("for ", "let "))) or stripped.startswith(("input ", "output ")):
                    self._collect_global_declarations(lstrip_line, line_number)

            if is_block_start:
                depth += 1
            elif is_end and depth > 0:
                depth -= 1

        for line, line_number in label_lines:
            self._process_label_assignment(line, line_number)
        for line, line_number in lines:
            self._process_cell_binding_declaration(line, line_number)
        for line, line_number in lines:
            if not line.startswith(':') and ':=' not in line and '!' not in line:
                self._evaluate_cell_var_definition(line, line_number, defer=True)

    def _collect_require_declaration(self, line, line_number=None):
        """Collect a `Require <name> as <Resource> [with (param = value, ...)]`
        declaration into the capability registry.

        Requirement capabilities are resolved (granted or denied) during
        `_resolve_require_stage`; the handle variable is defined either way.
        """
        raw = line.lstrip()
        if raw.lower().startswith('require '):
            raw = raw[len('require'):].strip()
        m = re.match(
            r'^([\w_]+(?:\s*,\s*[\w_]+)*)\s+as\s+([A-Za-z][\w]*(?:\.[A-Za-z][\w]*)*)'
            r'(?:\s+with\s*\((.*)\)\s*)?$',
            raw, re.I | re.S)
        if not m:
            raise SyntaxError(
                f"Invalid Require syntax at line {line_number}: '{line}'\n"
                "Expected: Require <name> as <Resource> [with (param = value, ...)]")
        names_part = m.group(1).strip()
        resource_type = m.group(2).strip()
        with_content = (m.group(3) or '').strip()
        res_lower = resource_type.lower()
        type_def = self.types_defined.get(res_lower) if res_lower else None
        if not type_def and '.' in res_lower:
            # Dotted resource (e.g. mod.timer): resolve against the
            # namespaced binding, and if that fails, against the base name
            # (a flat import of the same physical type).
            base_lower = res_lower.rsplit('.', 1)[1]
            if base_lower in self.types_defined:
                type_def = self.types_defined.get(base_lower)
        if not type_def:
            raise SyntaxError(
                f"Unknown resource type '{resource_type}' in Require at line {line_number}; "
                "declare it with 'Define <Name> as Resource'")
        if not (type_def.get('_constraints') or {}).get('is_resource'):
            raise SyntaxError(
                f"'{resource_type}' is a Type, not a Resource; only resources declared "
                f"with 'Define <Name> as Resource' can be required (line {line_number})")
        params = {}
        if with_content:
            field_names = {str(f).lower() for f in type_def.get('_member_keys', set())}
            with_kind, with_payload = self._parse_with_clause(
                with_content, line_number)
            if with_kind != 'named':
                raise SyntaxError(
                    f"Invalid parameter in Require at line {line_number}: "
                    "'with (...)' must list 'field = value' pairs")
            for field, val_expr in with_payload.items():
                if field.lower() not in field_names:
                    raise SyntaxError(
                        f"Unknown parameter '{field}' for resource '{resource_type}' at line {line_number}; "
                        f"valid parameters: {', '.join(sorted(field_names)) or '(none)'}")
                params[field.lower()] = val_expr.strip()
        var_names = [n.strip() for n in names_part.split(',') if n.strip()]
        for name in var_names:
            if not re.match(r'^[\w_]+$', name):
                raise SyntaxError(
                    f"Invalid capability name '{name}' in Require at line {line_number}")
            key = name.lower()
            if key in self.require_caps:
                raise SyntaxError(
                    f"Capability '{name}' already required at line {line_number}")
            entry = {
                'var_name': name,
                'name': name,
                'resource': resource_type,
                'resource_lower': res_lower,
                'type_def': type_def,
                'params': params,
                'params_evaluated': None,
                'line_number': line_number,
            }
            self.require_caps[key] = entry
            self.requirements.append(entry)

    def _collect_global_declarations(self, line, line_number=None):

        # Handle REQUIRED (capability) declarations: bind a handle on a resource
        # as long as the user grants it, otherwise the handle is the sticky
        # #PERM error value. `Require <name> as <Resource> [with (param = value, ...)]`
        # is only valid at the top (global) level.
        if line.lstrip().lower().startswith('require '):
            self._collect_require_declaration(line, line_number)
            return

        # Handle INPUT and OUTPUT declarations that don't start with ':'
        line_stripped = line.lstrip()
        if line_stripped.lower().startswith(("input ", "output ")):
            a = line_stripped.strip()
        else:
            a = line_stripped[1:].strip()

        parsed = None

        # Immediately fail on simple self-referential assignments like "x = x"
        m_self = re.match(r'^([\w_]+)\s*=\s*\1\s*$', a, re.I)
        if m_self:
            raise ValueError(
                f"Self-referential assignment '{m_self.group(1)} = {m_self.group(1)}' at line {line_number}")

        # Handle INPUT declarations
        if a.lower().startswith('input '):
            var, type_name, constraints, expr = self.parser._parse_variable_def(
                a, line_number)
            constraints = constraints or {}
            default_value = expr
            var_names = constraints.pop('var_list', [var])
            for var_name in var_names:
                is_custom_type = type_name and type_name.lower() in self.types_defined
                effective_type = type_name.lower(
                ) if is_custom_type else ('array' if constraints.get('dim') else (type_name or None))
                self.current_scope().define_input(
                    var_name, effective_type, default_value, line_number, constraints)
            return

        # Handle OUTPUT declarations
        m_output = re.match(
            r'^OUTPUT\s+(.+)$', a, re.I | re.S)
        if m_output:
            def_str = m_output.group(1).strip()
            var, type_name, constraints, expr = self.parser._parse_variable_def(
                def_str, line_number)
            constraints = constraints or {}
            constraints['output'] = True
            if expr is not None and 'init' not in constraints:
                constraints['init'] = expr
                expr = None
            parsed = (var, type_name or 'text', constraints, expr)

        if parsed:
            var, type_name, constraints, expr = parsed
            var_names = constraints.pop('var_list', [var])
            for var_name in var_names:
                self.current_scope().define_output(
                    var_name, type_name, line_number, constraints)
                if constraints.get('init') is not None:
                    init_expr = constraints.get('init')
                    deps = set(self._extract_identifier_tokens(init_expr))
                    func_names = set(getattr(self, 'functions', {}).keys())
                    deps = {d for d in deps if d.lower() not in func_names}
                    unresolved = False
                    for dep in deps:
                        dep_scope = self.current_scope().get_defining_scope(dep)
                        if dep_scope and dep_scope.is_uninitialized(dep):
                            unresolved = True
                            break
                        try:
                            self.current_scope().get(dep)
                        except Exception:
                            unresolved = True
                            break
                    if unresolved:
                        self.pending_assignments[var_name] = (
                            init_expr, line_number, deps, constraints)
                    elif getattr(self, '_parent_scope', None) is not None:
                        # In subprocess context, defer OUTPUT init to output
                        # collection time so body statements (For/Let) can
                        # shadow variables first.
                        if not hasattr(self, '_deferred_output_inits'):
                            self._deferred_output_inits = {}
                        self._deferred_output_inits[var_name.lower()] = (
                            init_expr, constraints)
                    else:
                        eval_scope = self.current_scope().get_full_scope()
                        import copy
                        init_val = self.expr_evaluator.eval_expr(
                            init_expr, eval_scope, line_number)
                        init_val = copy.deepcopy(init_val)
                        if constraints.get('with'):
                            init_val = self._apply_with_constraints(
                                init_val, constraints.get('with', {}),
                                eval_scope, line_number,
                                type_name=type_name)
                        self.current_scope().update(
                            var_name, init_val, line_number)
            return

        var_def, expr = self._split_assignment_expr(a)
        if not parsed and expr is None:
            try:
                var, type_name, constraints, expr = self._parse_variable_def(
                    a, line_number)
            except Exception:
                var = None
            if var and not type_name and not constraints and expr is None:
                existing_scope = self.current_scope().get_defining_scope(var)
                if self._is_outer_scope(existing_scope):
                    existing_scope = None
                if (not existing_scope) or (var not in existing_scope.variables):
                    self.current_scope().define(
                        var, None, 'unknown', {},
                        is_uninitialized=True, line_number=line_number)
                return
            if var and (type_name or constraints):
                constraints = constraints or {}
                is_custom_type = type_name and type_name.lower() in self.types_defined
                effective_type = type_name.lower(
                ) if is_custom_type else ('array' if constraints.get('dim') else (type_name or 'unknown'))
                self.current_scope().types.setdefault(var, effective_type)
                if constraints.get('dim'):
                    dims = constraints['dim']
                    if isinstance(dims, dict) and 'dims' in dims:
                        dims = dims['dims']
                    if isinstance(dims, list):
                        self.dimensions[var] = dims
                        self.dim_names[var] = {
                            name: idx for idx, (name, _) in enumerate(dims) if name}
                        self.dim_labels[var] = {}
                if constraints.get('dim') and isinstance(constraints.get('dim'), list):
                    if any(isinstance(size_spec, str) for _, size_spec in constraints['dim']):
                        self.current_scope().define(
                            var, None, effective_type, constraints, is_uninitialized=True, line_number=line_number)
                        return
                    if constraints.get('init') is not None:
                        self.current_scope().define(
                            var, None, effective_type, constraints, is_uninitialized=True, line_number=line_number)
                        return
                    shape = []
                    for _, size_spec in constraints['dim']:
                        if isinstance(size_spec, tuple):
                            start, end = size_spec
                            size = None if end is None else end - start + 1
                        elif size_spec is None:
                            size = 1
                        else:
                            size = size_spec
                        shape.append(size)
                    if any(size is None for size in shape):
                        self.current_scope().define(
                            var, {}, effective_type, constraints,
                            is_uninitialized=False, line_number=line_number)
                        return
                    if is_custom_type:
                        value = self.array_handler.create_object_array(
                            shape, None, line_number)
                        self.current_scope().define(
                            var, value, type_name.lower(), constraints, is_uninitialized=False, line_number=line_number)
                    else:
                        pa_type = 'number' if effective_type in (
                            'number', 'array') else 'text'
                        value = self.array_handler.create_array(
                            shape, None, pa_type, line_number, template=True)
                        self.current_scope().define(
                            var, value, effective_type, constraints, is_uninitialized=False, line_number=line_number)
                else:
                    self.current_scope().define(
                        var, None, effective_type, constraints, is_uninitialized=True, line_number=line_number)
                return
        if self._handle_constructor_global_declaration(a, line_number):
            return

        # Fallback: handle INIT-only declarations (no '=')
        try:
            var, type_name, constraints, value = self._parse_variable_def(
                a, line_number)
        except Exception:
            var = None
        if var and constraints.get('init') is not None:
            # Leave INIT for runtime evaluation to preserve execution order.
            self.current_scope().types.setdefault(var, type_name or 'unknown')
            existing_scope = self.current_scope().get_defining_scope(var)
            if self._is_outer_scope(existing_scope):
                # Subprocesses/functions shadow caller variables locally.
                existing_scope = None
            if (not existing_scope) or (var not in existing_scope.variables):
                self.current_scope().define(var, None, type_name or 'unknown',
                                            constraints, is_uninitialized=True)
            else:
                existing_scope.constraints[existing_scope._get_case_insensitive_key(
                    var, existing_scope.constraints) or var] = constraints
            return

        if expr is not None:
            self._handle_global_assignment_expression(
                var_def, expr, line_number)
            return
        raise SyntaxError(
            f"Invalid global definition syntax: {line} at line {line_number}")

    def _handle_constructor_global_declaration(self, declaration, line_number=None):
        m_new = re.match(
            r'^([\w_]+)\s*=\s*new\s+(\w+)\s*(\{|\()(.*)$', declaration, re.I)
        if not m_new:
            return False
        var, type_name, opener, _ = m_new.groups()
        if type_name.lower() == "copy":
            # Copy is a pseudo-type handled specially after parsing
            pass
        elif type_name.lower() not in self.types_defined:
            raise SyntaxError(
                f"Type '{type_name}' not defined at line {line_number}")
        pairs = {'{': '}', '(': ')'}
        closer = pairs[opener]
        start_pos = declaration.find(opener, m_new.start(3))
        stack = [closer]
        values_str = None
        trailing = ""
        for i, ch in enumerate(declaration[start_pos + 1:], start_pos + 1):
            if ch in pairs:
                stack.append(pairs[ch])
            elif stack and ch == stack[-1]:
                stack.pop()
                if not stack:
                    values_str = declaration[start_pos + 1:i]
                    trailing = declaration[i + 1:].strip()
                    break
            elif ch in pairs.values():
                raise SyntaxError(
                    f"Mismatched delimiter in constructor at line {line_number}: {declaration}")
        if values_str is None:
            raise SyntaxError(
                f"Unclosed constructor for '{type_name}' at line {line_number}: {declaration}")
        with_kind = 'empty'
        with_payload = None
        chain_text = None
        if trailing:
            pre_trailing, chain_text = split_builder_chain(trailing)
            if chain_text is not None:
                trailing = pre_trailing.strip()
            if trailing:
                if trailing.lower().startswith('with'):
                    with_kind, with_payload = self._parse_with_clause(
                        trailing, line_number=line_number)
                else:
                    raise SyntaxError(
                        f"Unexpected characters after constructor at line {line_number}: {trailing}")

        def _split_args(arg_text):
            args = []
            current = ""
            nest_stack = []
            for ch in arg_text + ',':
                if ch == ',' and not nest_stack:
                    if current.strip():
                        args.append(current.strip())
                    current = ""
                    continue
                current += ch
                if ch in pairs:
                    nest_stack.append(pairs[ch])
                elif nest_stack and ch == nest_stack[-1]:
                    nest_stack.pop()
                elif ch in pairs.values():
                    raise SyntaxError(
                        f"Mismatched delimiter in constructor arguments at line {line_number}: {declaration}")
            if nest_stack:
                raise SyntaxError(
                    f"Unbalanced constructor arguments at line {line_number}: {declaration}")
            return [arg for arg in args if arg.strip()]

        values = _split_args(values_str)
        if type_name.lower() == "copy":
            if len(values) != 1:
                raise ValueError(f"Copy expects exactly one argument at line {line_number}")
            src_expr = values[0]
            # Instances of a key type (Keytype) cannot be copied at all - e.g. L as Keytype(number)
            # Check statically via src var type
            m_key_src = re.match(r'^\s*([A-Za-z_][\w]*)(?:\.([\w_]+))?\s*$', src_expr)
            if m_key_src:
                base_var = m_key_src.group(1)
                field = m_key_src.group(2)
                try:
                    def_scope = self.current_scope().get_defining_scope(base_var)
                    if def_scope:
                        if field:
                            base_type = def_scope.types.get(def_scope._get_case_insensitive_key(base_var, def_scope.types) or base_var)
                            if base_type:
                                base_tdef = self.types_defined.get(base_type.lower(), {})
                                field_type = None
                                for fname, ftype in self._get_public_type_fields(base_tdef).items():
                                    if fname.lower() == field.lower():
                                        field_type = ftype
                                        break
                                if field_type:
                                    if self._is_keyed_primitive_field_type(field_type):
                                        raise TypeError(f"Cannot copy instance of key type '{field_type}' at line {line_number}")
                                # field `k as number key` - the field itself is key, but its value is ordinary number
                                # Copying the field value via `new Copy(p.k)` where p.k is number key should be allowed as ordinary number?
                                # For now, only block L-type (Keytype) instances, not `number key` field values
                        else:
                            var_key = def_scope._get_case_insensitive_key(base_var, def_scope.types)
                            var_type = def_scope.types.get(var_key) if var_key else None
                            if var_type:
                                if self._is_keyed_primitive_field_type(var_type):
                                    raise TypeError(f"Cannot copy instance of key type '{var_type}' at line {line_number}")
                except TypeError:
                    raise
                except Exception:
                    pass
            source = self.expr_evaluator.eval_expr(src_expr, self.current_scope().get_full_scope(), line_number)
            value_dict = self._copy_instance(source, line_number)
            effective_type = value_dict.get('_type_name') if isinstance(value_dict, dict) else None
            # Clone listeners: internal/external edges from src var to dst var
            m_src_var = re.match(r'^\s*([A-Za-z_][\w]*)\s*$', src_expr)
            if m_src_var:
                src_var = m_src_var.group(1)
                try:
                    self._clone_listeners_for_copy(src_var, var, self.current_scope(), self.current_scope())
                except Exception:
                    pass
            if with_kind != 'empty':
                value_dict = self._apply_with_clause(value_dict, with_kind, with_payload, self.current_scope().get_full_scope(), line_number, type_name=effective_type)
                if isinstance(value_dict, list):
                    for elem in value_dict:
                        if isinstance(elem, dict):
                            elem.pop('_with_applied_fields', None)
                elif isinstance(value_dict, dict):
                    value_dict.pop('_with_applied_fields', None)
            if chain_text:
                value_dict = self.type_processor._apply_builder_chain(value_dict, chain_text, self.current_scope().get_full_scope(), line_number)
            # Keyed fields become immutable after Copy+builder (Push on key → compile error)
            if isinstance(value_dict, dict):
                tdef_eff = self.types_defined.get((effective_type or '').lower(), {})
                if tdef_eff:
                    self._mark_keyed_fields_immutable(value_dict, self._get_public_type_fields(tdef_eff), tdef_eff)
            elif isinstance(value_dict, list):
                for elem in value_dict:
                    if not isinstance(elem, dict):
                        continue
                    eff = elem.get('_type_name', effective_type)
                    tdef2 = self.types_defined.get((eff or '').lower(), {})
                    if not tdef2:
                        continue
                    self._mark_keyed_fields_immutable(elem, self._get_public_type_fields(tdef2), tdef2)
            if isinstance(value_dict, list):
                self.current_scope().define(var, value_dict, effective_type or 'object', {}, is_uninitialized=False)
                return True
            snapshot = {k: copy.deepcopy(v) for k, v in value_dict.items() if not str(k).startswith('_')}
            constraints = {'constant': snapshot}
            self.current_scope().define(var, value_dict, effective_type or 'object', constraints)
            return True
        type_fields = self.types_defined[type_name.lower()]
        actual_fields = self._get_public_type_fields(type_fields)
        inputs_list = type_fields.get('_inputs', [])
        _ = len(inputs_list) if inputs_list else len(actual_fields)
        if not values and values_str.strip() == '':
            with_input_values = {}
            value_dict = self._instantiate_type(
                type_name, [], line_number, allow_default_if_empty=True, var_name=var,
                execute_code=False, input_values_out=with_input_values)
            if with_kind != 'empty':
                value_dict = self._apply_with_clause(
                    value_dict, with_kind, with_payload,
                    self.current_scope().get_full_scope(),
                    line_number, type_name=type_name)
            exec_lines = type_fields.get('_executable_code', []) or []
            if isinstance(value_dict, list):
                for idx, elem in enumerate(value_dict):
                    if isinstance(elem, dict):
                        self.type_processor._execute_type_code(
                            exec_lines, var, elem, line_number, with_input_values)
                        if elem.pop('_with_conflict', False):
                            value_dict[idx] = '#VALUE'
                self.current_scope().define(
                    var, value_dict, type_name, {}, is_uninitialized=False)
                return True
            # Constructor code runs after the WITH clause, so WITH values act
            # as constraints on the fields the constructor computes.
            self.type_processor._execute_type_code(
                exec_lines, var, value_dict, line_number, with_input_values)
            if value_dict.pop('_with_conflict', False):
                self.current_scope().define(var, '#VALUE', type_name, {})
                return True
            if chain_text:
                value_dict = self.type_processor._apply_builder_chain(
                    value_dict, chain_text,
                    self.current_scope().get_full_scope(), line_number)
            snapshot = {k: copy.deepcopy(v) for k, v in value_dict.items()
                        if not str(k).startswith('_')}
            constraints = {'constant': snapshot}
            self.current_scope().define(var, value_dict, type_name, constraints)
            return True
        all_literals = all(re.match(r'^-?\d*\.?\d+$|^\".*\"$', v)
                           for v in values)
        evaluated_args = [self.expr_evaluator.eval_expr(
            value, self.current_scope().get_full_scope(), line_number) for value in values]
        with_input_values = {}
        value_dict = self._instantiate_type(
            type_name, evaluated_args, line_number, allow_default_if_empty=False, var_name=var,
            execute_code=False, input_values_out=with_input_values)
        if with_kind != 'empty':
            value_dict = self._apply_with_clause(
                value_dict, with_kind, with_payload,
                self.current_scope().get_full_scope(),
                line_number, type_name=type_name)
        exec_lines = type_fields.get('_executable_code', []) or []
        if isinstance(value_dict, list):
            for idx, elem in enumerate(value_dict):
                if isinstance(elem, dict):
                    self.type_processor._execute_type_code(
                        exec_lines, var, elem, line_number, with_input_values)
                    if elem.pop('_with_conflict', False):
                        value_dict[idx] = '#VALUE'
            self.current_scope().define(
                var, value_dict, type_name, {}, is_uninitialized=False)
            return True
        # Constructor code runs after the WITH clause, so WITH values act as
        # constraints on the fields the constructor computes.
        self.type_processor._execute_type_code(
            exec_lines, var, value_dict, line_number, with_input_values)
        if value_dict.pop('_with_conflict', False):
            self.current_scope().define(var, '#VALUE', type_name, {})
            return True
        if not all_literals:
            # Deferred: keep the chain in the stored expression so re-resolution
            # applies it exactly once; do NOT apply it eagerly here.
            pending_expr = f"new {type_name}{{{values_str}}}"
            deps = self._extract_identifier_tokens(values_str)
            if with_kind != 'empty' and trailing:
                pending_expr += ' ' + trailing
                deps |= self._with_deps(trailing)
            if chain_text:
                pending_expr += ' ' + chain_text
                deps |= self._extract_identifier_tokens(chain_text)
            if var in deps:
                raise ValueError(
                    f"Self-referential assignment '{var} = {pending_expr}' at line {line_number}")
            snapshot = {k: copy.deepcopy(v) for k, v in value_dict.items()
                        if not str(k).startswith('_')}
            constraints = {'constant': snapshot}
            self.current_scope().define(var, value_dict, type_name, constraints)
            self.pending_assignments[var] = (
                pending_expr, line_number, deps)
            return True
        if chain_text:
            value_dict = self.type_processor._apply_builder_chain(
                value_dict, chain_text,
                self.current_scope().get_full_scope(), line_number)
        snapshot = {k: copy.deepcopy(v) for k, v in value_dict.items()
                    if not str(k).startswith('_')}
        constraints = {'constant': snapshot}
        self.current_scope().define(var, value_dict, type_name, constraints)
        return True

    def _handle_global_assignment_expression(self, var_def, expr, line_number=None):
        var_def, expr = map(str.strip, (var_def, expr))
        var, type_name, constraints, value = self._parse_variable_def(
            var_def, line_number)
        dim_spec = constraints.get('dim')
        if dim_spec:
            dims = dim_spec
            if isinstance(dims, dict) and 'dims' in dims:
                dims = dims['dims']
            if isinstance(dims, list):
                self.dimensions[var] = dims
                self.dim_names[var] = {
                    name: idx for idx, (name, _) in enumerate(dims) if name}
                self.dim_labels[var] = {}
        if value is not None:
            constraints['constant'] = expr
        elif expr:
            constraints['constant'] = expr
        self.current_scope().types.setdefault(var, type_name or 'unknown')
        evaluated_value = None
        is_uninitialized = True
        if expr:
            base_expr, with_text = self._split_new_with_expr(expr)
            if with_text:
                deps = self._with_deps(with_text, line_number)
                deps |= self._extract_identifier_tokens(base_expr or '')
            else:
                deps = set()
                interpolation_only = False
                if expr.strip().startswith('$"') and expr.strip().endswith('"'):
                    for match in re.finditer(r'(?<!\{)\{(?![\{\*])([^{}]*)\}', expr):
                        deps |= self._extract_identifier_tokens(match.group(1))
                    interpolation_only = True
                if not interpolation_only:
                    expr_no_quotes = re.sub(r'"[^"]*"', '', expr)
                    expr_no_quotes = re.sub(r"'[^']*'", '', expr_no_quotes)
                    # Remove builder-call names ('-> name(') so they are not
                    # treated as dependency variables.
                    expr_no_quotes = re.sub(
                        r'->\s*\$?[A-Za-z][A-Za-z0-9_.]*\s*\(', '(', expr_no_quotes)
                    expr_no_numbers = re.sub(
                        r'(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?(?![\w.])',
                        ' ', expr_no_quotes, flags=re.I)
                    expr_no_numbers = re.sub(r'\[[^\]]*\]', ' ', expr_no_numbers)
                    # Remove member accesses like "obj.field" so the field name
                    # is not treated as a standalone dependency.
                    expr_no_numbers = re.sub(
                        r'\.\s*[A-Za-z][A-Za-z0-9_]*', ' ', expr_no_numbers)
                    expr_no_numbers = mask_text_constant_tokens(expr_no_numbers)
                    potential_deps = re.findall(r'\b[\w_]+\b', expr_no_numbers)
                    built_in_functions = set(BUILTINS.keys()) | KEYWORDS
                    known_funcs = set(getattr(self, 'functions', {}).keys())
                    known_subs = set(getattr(self, 'subprocesses', {}).keys())
                    known_types = set(getattr(self, 'types_defined', {}).keys())
                    member_suffixes = {name.split('.', 1)[1]
                                       for name in known_funcs if '.' in name}
                    deps = set()
                    for dep in potential_deps:
                        if re.match(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$', dep, re.I) or re.match(r'^e[+-]?\d+$', dep, re.I):
                            continue
                        dep_lower = dep.lower()
                        if dep_lower in built_in_functions:
                            continue
                        if dep_lower in known_funcs or dep_lower in known_subs or dep_lower in known_types:
                            continue
                        if dep_lower in member_suffixes:
                            continue
                        deps.add(dep)
            if var in deps:
                raise ValueError(
                    f"Self-referential assignment '{var} = {expr}' at line {line_number}")
            is_simple_literal = (
                expr.startswith('"') and expr.endswith('"') or
                expr.startswith("'") and expr.endswith("'") or
                expr.replace('.', '').replace('-', '').isdigit() or
                (expr.startswith('{') and expr.endswith('}') and
                 all(
                     (item.strip().replace('-', '').replace('.', '').isdigit() or
                      (item.strip().startswith('"') and item.strip().endswith('"')) or
                      (item.strip().startswith("'") and item.strip().endswith("'")))
                     for item in expr[1:-1].split(',')
                     if item.strip()
                 ))
            )
            if not deps and is_simple_literal:
                try:
                    evaluated_value = self.expr_evaluator.eval_expr(
                        expr, self.current_scope().get_evaluation_scope(), line_number)
                    if constraints.get('with'):
                        evaluated_value = self._apply_with_constraints(
                            evaluated_value,
                            constraints.get('with', {}),
                            self.current_scope().get_full_scope(),
                            line_number,
                            type_name=type_name)
                    is_uninitialized = False
                except Exception as e:
                    self.pending_assignments[var] = (
                        expr, line_number, deps, constraints)
            else:
                self.pending_assignments[var] = (
                    expr, line_number, deps, constraints)
        self.current_scope().define(var, evaluated_value, type_name or 'unknown',
                                    constraints, is_uninitialized=is_uninitialized)
        if expr:
            self._register_listeners(var, expr, self.current_scope())

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
            try:
                array = self.current_scope().get(var_name)
            except Exception:
                array = None
            expected_size = None
            if array is not None:
                shape = self.array_handler.get_array_shape(array, line_number)
                if dim_idx < len(shape):
                    expected_size = shape[dim_idx]
            if expected_size is None:
                dims = self.dimensions.get(var_name, [])
                if dim_idx < len(dims):
                    _, size_spec = dims[dim_idx]
                    if isinstance(size_spec, tuple):
                        start, end = size_spec
                        if end is not None:
                            expected_size = end - start + 1
                    elif isinstance(size_spec, int):
                        expected_size = size_spec
            labels = [lbl.strip().strip('"')
                      for lbl in labels_str.split(',') if lbl.strip()]
            if expected_size is not None and len(labels) != expected_size:
                raise ValueError(
                    f"Number of labels ({len(labels)}) does not match dimension size ({expected_size}) at line {line_number}")
            self.array_handler.set_labels(
                var_name, dim_name, labels, line_number)
        else:
            raise SyntaxError(
                f"Invalid label assignment syntax: {line} at line {line_number}")

    def _evaluate_cell_var_definition(self, line, line_number=None, defer=False):
        m = re.match(r'^\[\s*\^?([A-Z]+\d+)\s*\]\s*:\s*(.+)$', line, re.S)
        if not m:
            return
        cell, rhs = map(str.strip, m.groups())
        if not re.match(r'^[A-Za-z]+\d+$', cell):
            raise ValueError(
                f"Invalid cell reference '{cell}' at line {line_number}")
        var_def, expr = self._split_assignment_expr(rhs)
        if expr is None:
            return
        var, type_name, constraints, _ = self._parse_variable_def(
            var_def, line_number)
        if not re.match(r'^[\w_]+$', var):
            raise SyntaxError(
                f"Invalid variable name: '{var}' at line {line_number}")
        cell_key = self._to_index(cell)
        if cell_key in self._cell_var_map and self._cell_var_map[cell_key] != var:
            raise SyntaxError(
                f"Cell '{cell}' already mapped to '{self._cell_var_map[cell_key]}' at line {line_number}")
        for c, v in self._cell_var_map.items():
            if v == var and c != cell_key:
                raise SyntaxError(
                    f"Variable '{var}' already mapped to cell '{c}' at line {line_number}")

        deps = self._extract_dependencies_from_expression(expr)
        if deps:
            # Cell-bound derived variable like '[A1] : f = eg'. Register f as
            # a client of its dependencies and make the cell mirror f live.
            # During the pre-pass (defer=True) the value is resolved later at
            # runtime; unresolved dependencies are also deferred and filled in
            # by the notification machinery once they are defined.
            scope = self.current_scope()
            self._register_listeners(var, expr, scope)
            self._register_cell_spill_listener(
                f'[{cell}] := {var}', var, line_number, scope)
            self._cell_var_map[cell_key] = var
            if defer:
                return
            try:
                value = self.expr_evaluator.eval_or_eval_array(
                    expr, scope.get_full_scope(), line_number,
                    expected_unit=(constraints or {}).get('unit'))
            except NameError as e:
                missing = self.extract_missing_dependencies(e)
                if not missing or not any(
                        dep.lower() in expr.lower() for dep in missing):
                    return
                for dep in missing:
                    self.mark_dependency_missing(dep)
                self.pending_assignments[var] = (
                    expr, line_number, set(missing), constraints)
                return
        else:
            if 'constant' not in constraints:
                constraints['constant'] = expr
            value = self.expr_evaluator.eval_or_eval_array(
                                expr, self.current_scope().get_full_scope(), line_number,
                                expected_unit=(constraints or {}).get('unit'))
        value = self.array_handler.check_dimension_constraints(
            var, value, line_number)
        if constraints.get('with'):
            value = self._apply_with_constraints(
                value, constraints.get('with', {}),
                self.current_scope().get_full_scope(), line_number,
                type_name=type_name)
        inferred_type = type_name or self.array_handler.infer_type(
            value, line_number)
        if inferred_type == 'int':
            inferred_type = 'number'
        value = self._to_sparse_undimmed(value, constraints)
        defining_scope = self.current_scope().get_defining_scope(var)
        if defining_scope:
            if constraints:
                defining_scope.constraints[var] = constraints
            defining_scope.update(var, value, line_number)
        else:
            self.current_scope().define(
                var, value, inferred_type, constraints, is_uninitialized=False, line_number=line_number, internal=True)
        self._spill_value_to_cells(cell_key, value, line_number)
        self._cell_var_map[cell_key] = var

    def _process_cell_binding_declaration(self, line, line_number=None):
        """Process lines like [A1]: width as number to bind variables to cells."""
        m = re.match(r'^\[\s*(\^?)([A-Za-z]+\d+)\s*\]\s*:\s*(.+)$', line, re.I)
        if not m:
            return False
        caret_flag, cell_ref, rhs = m.groups()
        cell_ref = cell_ref.upper()
        rhs = rhs.strip()
        parse_address(cell_ref)

        # If this is an assignment, let the assignment handler manage it.
        var_def, expr = self._split_assignment_expr(rhs)
        if expr is not None:
            return False

        var, type_name, constraints, _ = self.parser._parse_variable_def(
            var_def.strip(), line_number)
        constraints = constraints or {}
        if not re.match(r'^[\w_]+$', var):
            raise SyntaxError(
                f"Invalid variable name: '{var}' at line {line_number}")

        # Ensure variable exists (case-insensitive) with the right type metadata.
        defining_scope = self.current_scope().get_defining_scope(var)
        if not defining_scope:
            self.current_scope().define(
                var, None, type_name, constraints, is_uninitialized=True)
        else:
            actual_key = defining_scope._get_case_insensitive_key(
                var, defining_scope.constraints) or var
            defining_scope.constraints[actual_key] = constraints
            if type_name:
                type_key = defining_scope._get_case_insensitive_key(
                    var, defining_scope.types) or var
                defining_scope.types[type_key] = type_name

        # Prevent conflicting mappings.
        cell_key = self._to_index(cell_ref)
        if caret_flag:
            existing = self._cell_array_map.get(cell_key)
            if existing and existing.lower() != var.lower():
                raise SyntaxError(
                    f"Cell '{cell_ref}' already mapped to '{existing}' at line {line_number}")
            for mapped_cell, mapped_var in self._cell_array_map.items():
                if mapped_var.lower() == var.lower() and mapped_cell != cell_key:
                    raise SyntaxError(
                        f"Variable '{var}' already mapped to cell '{mapped_cell}' at line {line_number}")
            self._cell_array_map[cell_key] = var
        else:
            existing = self._cell_var_map.get(cell_key)
            if existing and existing.lower() != var.lower():
                raise SyntaxError(
                    f"Cell '{cell_ref}' already mapped to '{existing}' at line {line_number}")
            for mapped_cell, mapped_var in self._cell_var_map.items():
                if mapped_var.lower() == var.lower() and mapped_cell != cell_key:
                    raise SyntaxError(
                        f"Variable '{var}' already mapped to cell '{mapped_cell}' at line {line_number}")
            self._cell_var_map[cell_key] = var

        # If the variable already has a value, reflect it in the grid immediately.
        try:
            current_value = self.current_scope().get(var)
            if current_value is not None:
                self._spill_value_to_cells(cell_key, current_value, line_number)
                self._record_output_value(var, current_value)
        except NameError:
            pass
        return True

    def _sync_cell_bindings(self, var_name, value):
        """Propagate variable updates to any bound cells."""
        if value is None:
            return
        var_lower = var_name.lower()
        for cell, mapped_var in self._cell_array_map.items():
            if mapped_var.lower() == var_lower:
                try:
                    self.array_handler._assign_horizontal_array(
                        cell, value, "{}", line_number=None)
                except Exception:
                    self._set_grid_cell(cell, value)
        for cell, mapped_var in self._cell_var_map.items():
            if mapped_var.lower() == var_lower:
                self._spill_value_to_cells(cell, value, None)

    def _record_output_value(self, var_name, value):
        """Record values for declared output variables."""
        if value is None:
            return
        global_scope = self.get_global_scope()
        if not global_scope.is_output(var_name):
            return
        var_key = var_name.lower()
        if isinstance(value, dict) and 'array' in value:
            value = list(value['array'])
        elif is_sparse_array(value):
            value = [value[k] for k in sorted(value.keys())]
        # Keep the unit attached so function/subprocess outputs retain it when
        # they are returned to a caller (grid writes strip it instead).
        if is_error_value(value):
            value = error_value(value if isinstance(value, str) else value.error_code)
        else:
            scope = self.current_scope()
            while scope is not None:
                unit = scope.get_value_unit(var_name)
                if unit:
                    value = UnitValue(value, unit)
                    break
                scope = getattr(scope, 'parent', None)
        # Keep all pushed values in order
        existing = self.output_values.get(var_key, [])
        if not isinstance(existing, list):
            existing = [] if existing is None else [existing]
        existing.append(value)
        self.output_values[var_key] = existing

    def truncate_output(self, output_dict, max_length=100):
        """Truncate long test output and add '...' if needed"""
        output_str = str(output_dict)
        if len(output_str) <= max_length:
            return output_str
        return output_str[:max_length-3] + "..."

    def _debug_export_value(self, value):
        """Format runtime values for debug CSV output."""
        if isinstance(value, dict) and 'array' in value:
            value = list(value['array'])
        elif is_sparse_array(value):
            value = [value[k] for k in sorted(value.keys())]
        if isinstance(value, dict):
            value = public_object_view(value)
        elif isinstance(value, list):
            normalized = []
            for item in value:
                if isinstance(item, dict):
                    normalized.append(public_object_view(item))
                elif isinstance(item, (list, dict)):
                    normalized.append(self._debug_export_value(item))
                else:
                    normalized.append(item)
            value = normalized
        return format_display_value(value)

    def _iter_debug_output_rows(self):
        """Yield pushed output values as a single-column fallback export."""
        output_values = getattr(self, 'output_values', {}) or {}
        for _, values in output_values.items():
            if values is None:
                continue
            if not isinstance(values, list):
                values = [values]
            for value in values:
                yield [self._debug_export_value(value)]

    def _build_debug_grid_rows(self):
        """Return the populated grid laid out as a CSV matrix."""
        cell_values = {}
        max_row = 0
        max_col = 0
        for cell, value in (self.grid or {}).items():
            if isinstance(cell, str):
                cell_text = cell.upper()
                try:
                    column_text, row_text = split_cell(cell_text)
                except Exception:
                    continue
                row_num = int(row_text)
                col_num = col_to_num(column_text)
            elif isinstance(cell, tuple) and len(cell) == 2:
                row_num = int(cell[0]) + 1
                col_num = int(cell[1]) + 1
            else:
                continue
            max_row = max(max_row, row_num)
            max_col = max(max_col, col_num)
            cell_values[(row_num, col_num)] = self._debug_export_value(value)

        rows = []
        for row_num in range(1, max_row + 1):
            row_values = []
            for col_num in range(1, max_col + 1):
                row_values.append(cell_values.get((row_num, col_num), ''))
            rows.append(row_values)
        return rows

    def export_to_csv(self, grid_file):
        """Export the grid as a CSV matrix, or outputs as a single column if no grid exists."""
        output_path = os.path.splitext(os.path.abspath(grid_file))[0] + '.csv'
        rows = self._build_debug_grid_rows()
        if not rows:
            rows = list(self._iter_debug_output_rows() or [])
        with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(rows)
        return output_path

    def _is_keyword(self, line, keyword):
        """Case-insensitive keyword check"""
        return line.strip().lower() == keyword.lower()

    def _starts_with_keyword(self, line, keyword):
        """Case-insensitive keyword start check"""
        return line.strip().lower().startswith(keyword.lower())

    def _ends_with_keyword(self, line, keyword):
        """Case-insensitive keyword end check"""
        return line.strip().lower().endswith(keyword.lower())

    def set_input_values(self, args, prompt_missing=False):
        """Set input values from args/defaults. Optionally prompt for missing ones."""
        global_scope = self.get_global_scope()
        if args is None:
            args = []
        can_prompt = prompt_missing and sys.stdin and sys.stdin.isatty()

        for i, input_var in enumerate(self.input_variables):
            value_assigned = False
            provided_arg = i < len(args)
            if provided_arg:
                value = args[i]
                actual_key = global_scope._get_case_insensitive_key(
                    input_var, global_scope.types) or input_var
                var_type = global_scope.types.get(actual_key, 'text')
                constraints = global_scope.constraints.get(actual_key, {})
                comparison_keys = ('<', '<=', '>', '>=', '<>')
                type_union = constraints.get('type_union')
                union_allows_text = isinstance(type_union, (list, tuple, set)) and 'text' in type_union
                union_allows_number = isinstance(type_union, (list, tuple, set)) and 'number' in type_union
                union_number_only = union_allows_number and not union_allows_text
                needs_number = union_number_only or (var_type == 'number' and not union_allows_text) or any(
                    key in constraints for key in comparison_keys + ('range',)) or any(
                    key.startswith('not_') and key[4:] in comparison_keys for key in constraints)
                not_type = constraints.get('not_type')
                if needs_number:
                    try:
                        value = float(value)
                    except ValueError:
                        print(
                            f"Warning: Could not convert '{value}' to number for input '{input_var}', skipping provided arg")
                        value = None
                elif union_allows_text and union_allows_number:
                    try:
                        value = float(value)
                    except ValueError:
                        value = value
                elif not_type == 'text':
                    try:
                        value = float(value)
                    except ValueError:
                        value = value
                elif isinstance(value, str) and not type_union and constraints.get('type') is None and not any(
                        key in constraints for key in comparison_keys + ('range',)):
                    try:
                        value = float(value)
                    except ValueError:
                        value = value
                if value is not None:
                    # Handle unit literals like "5 of in" for Input with conversion
                    if isinstance(value, str) and ' of ' in value.lower():
                        try:
                            ev = self.expr_evaluator.eval_or_eval_array(value, global_scope.get_evaluation_scope())
                            if ev is not None:
                                value = ev
                        except Exception:
                            pass
                    global_scope.update(input_var, value)
                    value_assigned = True

            default_expr = global_scope.constraints.get(
                input_var, {}).get('default')
            default_value = None
            if (not value_assigned) and default_expr is not None:
                try:
                    default_value = self.expr_evaluator.eval_expr(
                        str(default_expr), global_scope.get_evaluation_scope())
                except Exception as exc:
                    print(
                        f"Warning: Failed to evaluate default for input '{input_var}': {exc}")

            # Prompt if allowed and still unset
            if not value_assigned and prompt_missing:
                if can_prompt:
                    try:
                        self._prompt_for_input(
                            input_var, global_scope, default_value)
                        value_assigned = True
                    except RuntimeError as exc:
                        print(f"Warning: {exc}")
                        break

            # Apply default silently if not prompted or no prompt available
            if (not value_assigned) and default_value is not None:
                global_scope.update(input_var, default_value)
                value_assigned = True

            actual_key = global_scope._get_case_insensitive_key(
                input_var, global_scope.variables) or input_var
            current_value = global_scope.variables.get(actual_key)
            if isinstance(current_value, str):
                constraints = global_scope.constraints.get(actual_key, {})
                comparison_keys = ('<', '<=', '>', '>=', '<>')
                if constraints.get('type') is None and not any(
                        key in constraints for key in comparison_keys + ('range',)):
                    try:
                        coerced = float(current_value)
                        global_scope.update(input_var, coerced)
                    except ValueError:
                        pass

    def _prompt_for_input(self, input_var, global_scope, default_value=None):
        """Prompt user for input value and set it in the global scope"""
        actual_key = global_scope._get_case_insensitive_key(
            input_var, global_scope.types) or input_var
        var_type = global_scope.types.get(actual_key, 'text')
        constraints = global_scope.constraints.get(actual_key, {})
        comparison_keys = ('<', '<=', '>', '>=', '<>')
        type_union = constraints.get('type_union')
        union_allows_text = isinstance(type_union, (list, tuple, set)) and 'text' in type_union
        union_allows_number = isinstance(type_union, (list, tuple, set)) and 'number' in type_union
        union_number_only = union_allows_number and not union_allows_text
        needs_number = union_number_only or (var_type == 'number' and not union_allows_text) or any(
            key in constraints for key in comparison_keys + ('range',)) or any(
            key.startswith('not_') and key[4:] in comparison_keys for key in constraints)
        if not (sys.stdin and sys.stdin.isatty()):
            raise RuntimeError(
                f"Cannot prompt for input '{input_var}' (no interactive input available). "
                "Please supply arguments when running the program."
            )

        default_display = ''
        if default_value is not None:
            default_display = f" [{default_value}]"

        while True:
            try:
                not_type = constraints.get('not_type')
                if needs_number:
                    user_input = input(f"{input_var}{default_display}: ")
                    if not user_input.strip() and default_value is not None:
                        global_scope.update(input_var, default_value)
                        return default_value
                    value = float(user_input)
                elif union_allows_text and union_allows_number:
                    user_input = input(f"{input_var}{default_display}: ")
                    if not user_input.strip() and default_value is not None:
                        global_scope.update(input_var, default_value)
                        return default_value
                    try:
                        value = float(user_input)
                    except ValueError:
                        value = user_input
                elif not_type == 'text':
                    user_input = input(f"{input_var}{default_display}: ")
                    if not user_input.strip() and default_value is not None:
                        global_scope.update(input_var, default_value)
                        return default_value
                    try:
                        value = float(user_input)
                    except ValueError:
                        value = user_input
                else:
                    user_input = input(f"{input_var}{default_display}: ")
                    if not user_input.strip() and default_value is not None:
                        global_scope.update(input_var, default_value)
                        return default_value
                    if constraints.get('type') is None and not any(
                            key in constraints for key in comparison_keys + ('range',)):
                        try:
                            value = float(user_input)
                        except ValueError:
                            value = user_input
                    else:
                        value = user_input

                global_scope.update(input_var, value)
                return value

            except EOFError:
                raise RuntimeError(
                    f"Cannot prompt for input '{input_var}' (no interactive input available). "
                    "Please supply arguments when running the program."
                )
            except ValueError:
                if union_allows_text and union_allows_number:
                    display_type = 'text or number'
                else:
                    display_type = 'number' if needs_number else (var_type or 'value')
                print(f"Invalid input. Please enter a valid {display_type}.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(1)

