
import re
import math
import copy
import time
from collections import deque
from contextlib import contextmanager
from expression import ExpressionEvaluator
from array_handler import ArrayHandler, constant_value_matches
from control_flow import GridLangControlFlow, block_nesting_delta
from parser import GridLangParser
from units import DIM_ERROR, VALUE_ERROR, TYPE_ERROR, ConstraintError, error_value, is_error_value
from utils import col_to_num, split_cell, offset_cell, parse_address, public_type_fields, object_public_keys, format_display_value, split_var_defs, is_address, is_sparse_array, strip_array_cell_indices, is_wildcard_address
from grid_lang_common import GridLangBase, _STATEMENT_KEYWORDS, _first_keyword, _DEPENDENCY_IGNORED_TOKENS, _strip_constraint_operands, _strip_builder_arrows, _strip_cell_address_tokens, resolve_text_constant_token, mask_text_constant_tokens, match_push_assignment, unpack_push_assignment, match_return_statement, is_definition_body_error
DEPENDENCY_IGNORED_TOKENS = _DEPENDENCY_IGNORED_TOKENS

IDENTIFIER_TOKEN_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9_.]*')
STRING_LITERAL_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def infer_array_rank(value):
    """Infer the internal rank of an array value that has no explicit dim
    constraint.  Dict-form arrays use their shape, nested lists use their
    nesting depth, flat lists are 1-D.  Returns None for non-array values."""
    if isinstance(value, dict) and 'array' in value:
        shape = value.get('shape') or value.get('original_shape')
        return len(shape) if shape else 1
    if isinstance(value, (list, tuple)):
        depth = 0
        for item in value:
            if isinstance(item, (list, tuple)):
                depth = max(depth, infer_array_rank(item))
        return 1 + depth
    return None


def _strip_cell_address_tokens(line, var_set):
    """Remove cell-address segment tokens (A1, B4, ...) from a variable-token
    set so dotted addresses like [A1.B1] or ranges [A1.B1:A2.B2] are not
    misread as field accesses on variables named A1/A2."""
    if not var_set:
        return var_set
    stripped = set()
    for group in re.findall(r'\[([^\[\]]*)\]', line):
        for piece in group.split(':'):
            piece = piece.strip().lstrip('^')
            if is_address(piece) or is_wildcard_address(piece):
                for seg in piece.split('.'):
                    if re.match(r'^[A-Za-z]+\d*$', seg):
                        stripped.add(seg)
    return var_set - stripped


def _is_numeric_token(token: str) -> bool:
    """Return True if token looks like a numeric literal (int/float/scientific)."""
    if not token:
        return False
    if re.match(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$', token, re.I):
        return True
    if re.match(r'^e[+-]?\d+$', token, re.I):
        return True
    return False


def _filter_var_tokens(tokens):
    """Filter out numeric literals and ignored builtins from a token set."""
    filtered = set()
    for tok in tokens:
        if not tok:
            continue
        lower = tok.lower()
        if lower in DEPENDENCY_IGNORED_TOKENS:
            continue
        if _is_numeric_token(tok):
            continue
        filtered.add(tok)
    return filtered


def _extract_spill_root_var(rhs):
    """Root variable of a cell-spill RHS: a plain identifier ('x'), the
    base of a dotted member path ('q.x' -> 'q'), or the base of an element
    access ('arr(1)' / 'm{2,1}' -> 'arr'/'m'). Returns None otherwise."""
    rhs = rhs.strip()
    if re.match(r'^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$', rhs):
        return rhs.split('.')[0]
    if re.match(r'^[A-Za-z_]\w*$', rhs):
        return rhs
    if re.match(r'^[A-Za-z_]\w*\s*[({]', rhs):
        return re.match(r'^[A-Za-z_]\w*', rhs).group(0)
    return None


class GridLangExecutor(GridLangBase):
    """Runtime-execution layer of the GridLang engine.

    This is no longer a separate engine object: GridLangCompiler inherits it and
    runs directly on itself. Instances of this class represent the runtime loop
    (parsing, scopes, dependencies, for/when/push handling, output) shared by the
    single public engine.
    """

    def __init__(self):
        self.control_flow = GridLangControlFlow(self)
        self.exit_loop = False  # Simple boolean flag for breaking out of loops
        self.output_variables = []  # List of output variables
        self.output_values = {}  # Dictionary to store output values
        self.undefined_dependencies = set()
        self.global_guard_line_numbers = set()
        self.global_guard_allows_execution = True
        self.dependency_graph = {'nodes': [], 'by_variable': {}, 'by_line': {}}
        self.global_guard_entries = []
        self._global_guards_pre_evaluated = False
        self.global_for_line_numbers = set()
        self.executed_global_for_lines = set()
        self.hoisted_binding_lines = set()
        self.global_for_entries = []
        self.needed_line_numbers = set()
        self.when_blocks = []
        self._push_queues = {}
        self._processing_when = False
        # Periodic-Ticker clock state (see _maybe_fire_tickers): tickers fire
        # through the ordinary Push/When machinery, they just need a schedule
        # and a counter so listeners see the tick.
        self._ticker_last_fire = {}
        self._loop_iteration = 0
        # Ticker clock source: 'wall' fires on elapsed seconds (real programs);
        # 'loop' fires per main-loop iteration (deterministic test clock). The
        # test runner opts the latter in; production keeps the default.
        self._ticker_timebase = 'wall'
        # Started net!server transports (see _pump_server_requests): accept
        # threads append parsed requests to a plain deque; the main thread
        # drains it and fires matched `When <route> do` blocks.
        self._net_servers = []

    def _is_outer_defining_scope(self, scope):
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

    def _reset_dependency_graph(self):
        """Reset dependency graph storage for a new run."""
        self.dependency_graph = {'nodes': [], 'by_variable': {}, 'by_line': {}}

    def _is_standalone_var_ref(self, cleaned, name):
        """Whether 'name' occurs standalone in an expression (not glued to
        digits in scientific notation like 2e5, nor inside an identifier)."""
        return re.search(
            r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])',
            cleaned, re.I) is not None

    def _extract_dependencies_from_expression(self, expr):
        """Return identifiers referenced in an expression string."""
        if not expr:
            return set()
        cleaned = _strip_builder_arrows(
            strip_array_cell_indices(_strip_constraint_operands(expr)))
        tokens = IDENTIFIER_TOKEN_PATTERN.findall(cleaned)
        dependencies = set()
        for token in tokens:
            if not token:
                continue
            base = token.split('.')[0]
            lower = base.lower()
            if lower in DEPENDENCY_IGNORED_TOKENS:
                # 'e' (exponent notation) is only a real dependency when it
                # stands alone as a variable reference.
                if not (lower == 'e' and self._is_standalone_var_ref(cleaned, base)):
                    continue
            if hasattr(self, 'compiler') and getattr(self.compiler, 'types_defined', None):
                if lower in self.compiler.types_defined:
                    continue
            if hasattr(self, 'compiler'):
                if lower in getattr(self.compiler, 'functions', {}) or lower in getattr(self.compiler, 'subprocesses', {}):
                    continue
            if _is_numeric_token(base):
                continue
            dependencies.add(base)
        return dependencies

    def _collect_dependencies_from_constraints(self, constraints):
        """Walk constraint structures to collect embedded dependencies."""
        deps = set()

        def _walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    _walk(item)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    _walk(item)
            elif isinstance(value, str):
                deps.update(self._extract_dependencies_from_expression(value))

        if constraints:
            _walk(constraints)
        return deps

    def _extract_defined_names(self, target_part, line_number):
        """Infer variable names defined by a declaration/assignment header."""
        names = []
        try:
            var, type_name, constraints, expr = self._parse_variable_def(
                target_part, line_number)
            if var:
                names.append(var)
            var_list = constraints.get('var_list') if constraints else None
            if var_list:
                names.extend(var_list)
        except Exception:
            cleaned = target_part.strip()
            if cleaned.startswith('[') and ']' in cleaned:
                names.append(cleaned)
            elif ',' in cleaned:
                parts = [part.strip() for part in cleaned.split(',')]
                names.extend([p for p in parts if p])
            elif cleaned:
                names.append(cleaned)
        expanded = []
        for name in names:
            if not name:
                continue
            expanded.append(name)
            base_match = re.match(r'([A-Za-z][A-Za-z0-9_]*)', name)
            if base_match:
                base = base_match.group(1)
                expanded.append(base)
        dedup = []
        seen = set()
        for name in expanded:
            if not name:
                continue
            lowered = name.lower()
            if lowered in seen:
                continue
            dedup.append(name)
            seen.add(lowered)
        return dedup

    def _match_push_assignment(self, text):
        return match_push_assignment(text)

    def _unpack_push_assignment(self, push_match):
        return unpack_push_assignment(push_match)

    def _match_return_statement(self, text):
        return match_return_statement(text)

    def _has_push_action(self, text):
        if not text:
            return False
        lowered = text.lower()
        if self._match_push_assignment(text):
            return True
        if re.search(r'\breturn\b', text, re.I):
            return True
        return False

    def _has_star_dim(self, constraints):
        from grid_lang_common import _has_star_dim as _common_has_star
        return _common_has_star(constraints, getattr(self, 'array_handler', None) or getattr(getattr(self, 'compiler', None), 'array_handler', None))

    def _apply_dim_base_offsets(self, var_name, indices, line_number=None):
        from grid_lang_common import _apply_dim_base_offsets as _common_apply
        dims = getattr(self, 'dimensions', None) or getattr(getattr(self, 'compiler', None), 'dimensions', None)
        ah = getattr(self, 'array_handler', None) or getattr(getattr(self, 'compiler', None), 'array_handler', None)
        return _common_apply(var_name, indices, dims, ah, line_number)


    def _register_when_block(self, condition, block_lines, line_number):
        deps = self._extract_dependencies_from_expression(condition)
        deps = [dep for dep in deps if dep]
        entry = {
            'condition': condition,
            'deps': [d.lower() for d in deps],
            'block_lines': block_lines,
            'line_number': line_number,
        }
        self.when_blocks.append(entry)
        self._process_when_triggers()

    def _ticker_now(self):
        """Current Ticker clock value for the active time base.

        'wall' (the default, used by real programs) returns elapsed monotonic
        seconds so a Ticker interval is a real duration; 'loop' (selected by
        the test runner for deterministic tests) returns the running main-loop
        iteration counter so an interval is a number of statements executed.
        """
        if getattr(self, '_ticker_timebase', 'wall') == 'loop':
            return self._loop_iteration
        return time.monotonic()

    def _maybe_fire_tickers(self):
        """Advance periodic-Ticker capabilities and notify their listeners.

        A granted Ticker is a clock object: every ``interval`` clock units it
        increments its ``value`` counter (triggering any reactive binding that
        reads ``<cap>.value`` through the ordinary listener mechanism) and, when
        a `When <cap> do` block is registered, enqueues the tick on that
        capability's Push queue so the *existing* When machinery fires it. No
        part of When is special-cased for tickers. Each fire also advances
        every Timer/Counter handle derived from the ticker (see _advance_handles
        and notify_resource_updated). Denied (#PERM) or invalid tickers never
        advance (dead blocks).
        """
        from units import is_error_value
        caps = getattr(self, 'require_caps', None) or {}
        targets = [req for req in caps.values()
                   if req.get('resource_lower') == 'ticker']
        if not targets:
            return
        now = self._ticker_now()
        loop_units = getattr(self, '_ticker_timebase', 'wall') == 'loop'
        min_interval = 1 if loop_units else 0.001
        # Lazily compute the set of capability/handle names with a registered
        # When consumer so we only enqueue ticks someone is waiting for.
        when_deps = None
        for req in targets:
            cap_name = req['var_name'].lower()
            try:
                capability = self.current_scope().get(req['var_name'])
            except Exception:
                continue
            if (is_error_value(capability) or not isinstance(capability, dict)
                    or not capability.get('_capability')):
                continue
            params = capability.get('_params') or {}
            interval = params.get('interval')
            try:
                interval = float(interval)
            except (TypeError, ValueError):
                continue
            if not interval or interval < min_interval:
                continue
            # `disabled` is a hidden ticker attribute, settable from the Require
            # clause (never from a grant): while true the ticker does not advance.
            if capability.get('disabled'):
                continue
            if loop_units:
                # Loop clock: first fire as soon as one iteration elapses
                # (interval units are whole statements), then every interval.
                last = self._ticker_last_fire.get(cap_name, 0)
            else:
                # Wall clock: first observation fires immediately, then every
                # `interval` seconds (fractional values allowed). Start()/Reset
                # reposition the clock so a resumed ticker waits a full interval
                # instead of catching up instantly.
                last = self._ticker_last_fire.get(cap_name, now - interval)
            # A ticker with a fixed interval never fires before the interval
            # elapses, so short programs simply see no extra tick.
            if (now - last) < interval:
                continue
            self._ticker_last_fire[cap_name] = now
            # Rewrite the capability with an incremented engine-private counter
            # (`_ticks`). Public `value` is deliberately not registered (see
            # permissions build_capability): Grid observes elapsed ticks only
            # through a derived `tick.counter()`/`tick.timer()` handle. The
            # rewrite still changes the variable, which is what makes listener /
            # reactive bindings that depend on the ticker fire.
            new_cap = dict(capability)
            new_cap['_ticks'] = capability.get('_ticks', 0) + 1
            self._set_var_value(
                req['var_name'], new_cap, req['line_number'])
            # Serve `When <cap> do ...`: the ordinary queue-based trigger path
            # sees the enqueued tick exactly like a Push would.
            if when_deps is None:
                when_deps = self._when_dependency_names()
            if cap_name in when_deps:
                self._enqueue_push(req['var_name'], new_cap)
            # Engine-owned Timer/Counter handles derived from this ticker
            # advance in lockstep with the fire.
            self._advance_handles(cap_name, req['line_number'])
            self._process_when_triggers()

    def _when_dependency_names(self):
        """Set of variable names that a registered `When <name> do` waits on."""
        deps = set()
        for entry in getattr(self, 'when_blocks', []):
            deps.update(entry.get('deps') or [])
        return deps

    def _scope_handle_vars(self):
        """Return {var_lower: (var_name, handle_dict)} for every engine-owned
        handle stored in the current scope chain."""
        found = {}
        scope = self.current_scope()
        visited = set()
        while scope is not None and id(scope) not in visited:
            visited.add(id(scope))
            variables = getattr(scope, 'variables', None) or {}
            for name, val in variables.items():
                low = name.lower()
                if low in found:
                    continue
                if isinstance(val, dict) and val.get('_handle'):
                    found[low] = (name, val)
            scope = getattr(scope, 'parent', None)
        return found

    def _net_server_entry(self, server_id):
        """The started-server entry for ``server_id`` or None."""
        for entry in getattr(self, '_net_servers', []) or []:
            if entry.get('server_id') == server_id:
                return entry
        return None

    def _register_net_server(self, server_id, pinned):
        """Register a started net!server so _pump_server_requests drains it.

        Accept threads only append to the entry's plain ``queue`` deque (GIL
        makes single append/popleft atomic); every other field is owned by the
        main thread. Returns the entry for the Start builtin to plug its
        listener into.
        """
        entry = {
            'server_id': server_id,
            'queue': deque(),
            'listener': None,
            'started': True,
            'stopped': False,
            'pinned': list(pinned or ()),
        }
        self._net_servers.append(entry)
        return entry

    def _match_route(self, server_id, req, line_number):
        """Find the most-specific registered route for a request.

        Walks the scope chain for ``net!route`` handles belonging to that
        server, keeps those whose method/path template match, and returns the
        winning ``(var_name, route_handle)`` — or None (auto-404).
        """
        import net
        method = (req.get('method') or '').upper()
        path = req.get('raw_path') or req.get('path') or '/'
        best = None
        best_score = None
        for low, (name, value) in self._scope_handle_vars().items():
            if (value.get('_handle_type') or '').lower() != 'net!route':
                continue
            if str(value.get('_server_id') or '') != server_id:
                continue
            template = value.get('_template')
            if template is None:
                continue
            allowed = value.get('_methods') or []
            if not net.route_matches(template, path, method, allowed):
                continue
            score = net.route_specificity(template)
            if best_score is None or score > best_score:
                best = (name, value)
                best_score = score
        return best

    def _reply_unmatched(self, req, pinned):
        """Answer a request that matched no route with an automatic 404."""
        import net
        conn = req.get('conn')
        if conn is None:
            return
        try:
            conn.write(net.make_response(404, 'Not found', None, pinned))
            try:
                conn.flush()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        except Exception:
            pass

    def _pump_server_requests(self):
        """Deliver queued inbound requests to their matching route When blocks.

        Runs every main-loop unit next to ``_maybe_fire_tickers``. For each
        started server, every queued request is matched against the registered
        routes (most-specific wins); on a match the route handle is rewritten
        with a fresh ``net!request`` handle under its ``request`` member and
        enqueued on the route variable's Push queue so the ordinary When
        machinery fires it. Unmatched requests are answered 404 automatically.
        """
        from builtin_functions import create_handle
        fired = False
        for entry in list(getattr(self, '_net_servers', []) or []):
            if entry.get('stopped'):
                continue
            queue = entry.get('queue')
            pinned = entry.get('pinned') or []
            while queue:
                try:
                    req = queue.popleft()
                except IndexError:
                    break
                import sys as _sys
                try:
                    hit = self._match_route(entry['server_id'], req, 0)
                except Exception as e:
                    hit = None
                if hit is None:
                    self._reply_unmatched(req, pinned)
                    continue
                name, route_val = hit
                request_handle = create_handle(
                    "net!request",
                    _type_name="net!request",
                    method=req.get('method'),
                    path=req.get('path'),
                    headers='\n'.join(
                        f"{k}: {v}" for k, v in (req.get('headers') or {}).items()),
                    _body=req.get('body') or '',
                    _conn=req.get('conn'),
                    _pinned=pinned,
                    _replied=False,
                )
                new_route = dict(route_val)
                new_route['request'] = request_handle
                self._set_var_value(name, new_route, 0)
                self._enqueue_push(name, new_route)
                fired = True
        if fired:
            try:
                self._process_when_triggers()
            except Exception as e:
                import sys as _sys, traceback as _tb
                _tb.print_exc(file=_sys.stderr)

    def _advance_handles(self, parent_cap_name, line_number):
        """Advance Timer/Counter handles derived from the ticker that just fired.

        A Timer counts down once per parent tick and signals once (via its
        pending-Push queue when a `When <handle> do` waits on it) when
        `remaining` reaches 0, then stops. A Counter counts every parent tick
        into `now` for read-only elapsed-tick measurement. All updates happen
        engine-side; the notify callback propagates each change to listeners.
        """
        when_deps = self._when_dependency_names()
        for low, (name, value) in self._scope_handle_vars().items():
            if (value.get('parent') or '').lower() != parent_cap_name:
                continue
            htype = (value.get('_handle_type') or '').lower()
            new_value = dict(value)
            if htype == 'timer':
                if new_value.get('_fired'):
                    continue
                try:
                    rem = int(new_value.get('remaining') or 0)
                except (TypeError, ValueError):
                    continue
                rem -= 1
                if rem <= 0:
                    new_value['remaining'] = 0
                    new_value['_fired'] = True
                else:
                    new_value['remaining'] = rem
                self._set_var_value(name, new_value, line_number)
                # A Timer is a one-shot: its `When <timer> do ...` signals only
                # once, when `remaining` reaches 0 (the timeout). Intermediate
                # countdown updates still propagate to reactive listeners but
                # are not enqueued as triggers.
                if new_value.get('_fired') and low in when_deps:
                    self._enqueue_push(name, new_value)
            elif htype == 'counter':
                new_value['now'] = (new_value.get('now') or 0) + 1
                self._set_var_value(name, new_value, line_number)
                if low in when_deps:
                    self._enqueue_push(name, new_value)
            else:
                continue

    def notify_resource_updated(self, var_name, line_number=None):
        """Notify the engine that an engine-owned handle value changed.

        Builtin resource/member code calls this with the handle's variable name
        after it rewrites the handle in scope. The value itself never crosses
        the boundary: the engine reads the updated handle and (a) propagates it
        to reactive listeners and (b) fires any `When <handle> do ...` block by
        enqueuing the handle on its Push queue.
        """
        if var_name is None:
            return
        low = str(var_name).lower()
        try:
            value = self.current_scope().get(var_name)
        except Exception:
            return
        self._set_var_value(var_name, value, line_number)
        if low in self._when_dependency_names():
            self._enqueue_push(str(var_name), value)
        self._process_when_triggers()

    def _handle_ticker_system_call(self, name, args, line_number):
        """Run the predefined Ticker.Stop/Start control subprocesses.

        The resource name is used as the namespace: the call applies to every
        granted Ticker capability. ``Ticker.Stop()`` pauses ticking by setting
        the ticker's hidden ``disabled`` attribute (settable from a Require
        clause, never grantable); ``Ticker.Start()`` clears it and resumes with
        a fresh interval. The tick scalar itself is engine-private (``_ticks``)
        and is never readable from Grid; elapsed ticks are observed only
        through derived ``tick.counter()``/``tick.timer()`` handles. Denied
        (#PERM) tickers are left untouched.
        """
        from units import UnitValue, is_error_value
        action = name.lower().rsplit('.', 1)[-1]
        now = self._ticker_now()
        caps = getattr(self, 'require_caps', None) or {}
        targets = [req for req in caps.values()
                   if req.get('resource_lower') == 'ticker']
        for req in targets:
            cap_name = req['var_name'].lower()
            var_name = req['var_name']
            try:
                capability = self.current_scope().get(var_name)
            except Exception:
                continue
            if (is_error_value(capability) or not isinstance(capability, dict)
                    or not capability.get('_capability')):
                continue
            new_cap = dict(capability)
            if action == 'stop':
                new_cap['disabled'] = True
            elif action == 'start':
                new_cap['disabled'] = False
                # A fresh interval: the first resumed tick waits a full
                # interval after Start instead of catching up instantly.
                self._ticker_last_fire[cap_name] = now
            else:
                continue
            self._set_var_value(var_name, new_cap, line_number)

    def _set_var_value(self, var_name, value, line_number):
        # For key types, even client-owned variables are immutable after first assignment
        # Check before the client check so Push on a keytype still errors
        try:
            owner_check = getattr(self, 'compiler', None) or self
            def_scope_check = self.current_scope().get_defining_scope(var_name)
            if def_scope_check:
                actual_key_check = def_scope_check._get_case_insensitive_key(var_name, def_scope_check.variables)
                if actual_key_check:
                    var_type_check = def_scope_check.types.get(actual_key_check)
                    if def_scope_check._is_keyed_type(var_type_check, owner_check):
                        # Output variables are not storage - they can be pushed multiple times
                        is_output_check2 = False
                        try:
                            is_output_check2 = def_scope_check.is_output(actual_key_check) or def_scope_check.constraints.get(actual_key_check, {}).get('output')
                        except Exception:
                            pass
                        if not is_output_check2:
                            existing_val_check = def_scope_check.variables.get(actual_key_check)
                            from units import is_error_value
                            if existing_val_check is not None and not is_error_value(existing_val_check):
                                # Any Push/For update that tries to change a keytype var should error
                                # Even if value is plain number 6 (not UnitValue), it's still an attempt to modify key
                                raise ValueError(f"Cannot modify key type '{var_type_check}' at line {line_number}")
        except ValueError:
            raise
        except Exception:
            pass
        # Check equality constraint before attempting write: a Push onto a
        # constant-bound variable whose value does not match the constraint
        # emits a transient #VALUE to listeners instead of overwriting.
        owner = getattr(self, 'compiler', None) or self
        def_scope = self.current_scope().get_defining_scope(var_name)
        has_constant = False
        if def_scope:
            actual_key = def_scope._get_case_insensitive_key(
                var_name, def_scope.variables)
            if actual_key:
                constraints = def_scope.constraints.get(actual_key, {})
                constant_expr = constraints.get('constant')
                if constant_expr is not None:
                    has_constant = True
                    try:
                        from units import (strip_units, is_error_value,
                                           VALUE_ERROR)
                        if isinstance(constant_expr, str):
                            expected = self.expr_evaluator.eval_or_eval_array(
                                constant_expr, def_scope.get_full_scope(),
                                line_number)
                        else:
                            expected = constant_expr
                        if constraints.get('with'):
                            try:
                                type_name = None
                                actual_type_key = def_scope._get_case_insensitive_key(
                                    actual_key, def_scope.types)
                                if actual_type_key:
                                    type_name = def_scope.types.get(
                                        actual_type_key)
                                expected = owner._apply_with_constraints(
                                    expected,
                                    constraints.get('with', {}),
                                    def_scope.get_full_scope(),
                                    None,
                                    type_name=type_name,
                                )
                            except Exception:
                                pass
                        if not is_error_value(expected):
                            if not constant_value_matches(
                                    expected, value,
                                    self.array_handler.to_display_value):
                                owner._propagate_transient(
                                    var_name, VALUE_ERROR)
                                ctx = getattr(
                                    self, '_context_grid_stack', None) or getattr(
                                    owner, '_context_grid_stack', None)
                                if ctx:
                                    raise ValueError(
                                        f"Cannot modify constant field at line {line_number}")
                                return
                    except ValueError:
                        raise
                    except Exception:
                        pass
        # Publisher-side write: a client equality binding wins over the pushed
        # value, so skip the overwrite for client-owned variables. Engine-owned
        # handles are the exception: their state lives entirely on the Python
        # side (the client `Let h = cap.member(...)` only created the handle),
        # so the engine must be able to rewrite them on every update.
        if (not has_constant
                and owner._set_by.get(('var', var_name.lower())) == 'client'
                and not (isinstance(value, dict) and value.get('_handle'))):
            return
        defining_scope = self.current_scope().get_defining_scope(var_name)
        if defining_scope:
            defining_scope.update(var_name, value, line_number)
        else:
            self.current_scope().define(
                var_name, value, None, {}, is_uninitialized=False)
        owner._set_by[('var', var_name.lower())] = 'publisher'

    def _enqueue_push(self, var_name, value):
        key = var_name.lower()
        if key not in self._push_queues:
            self._push_queues[key] = deque()
        self._push_queues[key].append(value)

    def _process_when_triggers(self):
        if self._processing_when:
            return
        if not self.when_blocks:
            return
        self._processing_when = True
        try:
            progress = True
            while progress:
                progress = False
                for entry in list(self.when_blocks):
                    if self._run_when_block(entry):
                        progress = True
        finally:
            self._processing_when = False

    def _run_when_block(self, entry):
        deps = entry.get('deps') or []
        if not deps:
            return False
        queues = self._push_queues
        if any(len(queues.get(dep, deque())) == 0 for dep in deps):
            return False
        consumed = {}
        for dep in deps:
            val = queues[dep].popleft()
            consumed[dep] = val
            self._set_var_value(dep, val, entry['line_number'])
        try:
            cond_ok = self.control_flow._evaluate_if_condition(
                entry['condition'], entry['line_number'])
        except Exception as exc:
            return True
        if not cond_ok:
            return True
        self.push_scope(is_private=True, is_loop_scope=True)
        loop_scope = self.current_scope()
        self.control_flow._process_block(entry['block_lines'])
        parent_scope = loop_scope.parent
        if parent_scope:
            loop_local_lets = getattr(
                loop_scope, 'local_let_declarations', set())
            for name, val in loop_scope.variables.items():
                if name.lower() in loop_local_lets:
                    continue
                try:
                    parent_scope.update(name, val)
                except Exception:
                    inferred_type = self.array_handler.infer_type(
                        val, entry['line_number'])
                    parent_scope.define(
                        name, val, inferred_type, {}, is_uninitialized=False)
        self.pop_scope()
        return True

    def _parse_assignment_like_line(self, normalized, original_line, line_number, *, is_declaration=False, allow_split=True):
        """Capture assignment style statements for dependency analysis."""
        working = normalized
        kind = 'assignment'
        lowered = working.lower()
        if lowered.startswith('let '):
            kind = 'let'
            working = working[4:].strip()
            lowered = working.lower()
        elif lowered.startswith('for '):
            if ' do' in lowered:
                return None
            kind = 'for_assignment'
            working = working[4:].strip()
            lowered = working.lower()
        if lowered.startswith('if '):
            return None
        if ' then ' in lowered:
            return None
        if kind in {'let', 'for_assignment'} and allow_split and re.search(r'\s+and\s+', working, re.I):
            nodes = []
            parts = split_var_defs(working)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                sub_node = self._parse_assignment_like_line(
                    part, original_line, line_number,
                    is_declaration=(is_declaration and kind == 'let'),
                    allow_split=False)
                if not sub_node:
                    continue
                if isinstance(sub_node, list):
                    nodes.extend(sub_node)
                else:
                    nodes.append(sub_node)
            return nodes if nodes else None
        match = re.match(r'^(.*?)(?::=|=)\s*(.+)$', working)
        if not match:
            return None
        target_part = match.group(1).strip()
        expr = match.group(2).strip()
        defined_names = self._extract_defined_names(target_part, line_number)
        dependencies = self._extract_dependencies_from_expression(expr)
        defined_lower = {name.lower() for name in defined_names}
        filtered_deps = [
            dep for dep in dependencies if dep.lower() not in defined_lower]
        if not defined_names and not filtered_deps:
            return None
        return {
            'kind': 'declaration' if is_declaration else 'assignment',
            'line': line_number,
            'source': original_line,
            'defines': defined_names,
            'depends_on': filtered_deps,
            'data': {
                'target': target_part,
                'expression': expr,
                'category': kind
            }
        }

    def _split_call_arguments(self, args_str):
        """Split call argument list while respecting nesting and quotes."""
        args = []
        current = ""
        depth_paren = depth_brace = depth_bracket = 0
        in_quote = None

        def _flush():
            nonlocal current
            if current.strip():
                args.append(current.strip())
            current = ""

        for idx, ch in enumerate(args_str):
            if in_quote:
                current += ch
                if ch == in_quote and (idx == 0 or args_str[idx - 1] != '\\'):
                    in_quote = None
                continue
            if ch in ('"', "'"):
                in_quote = ch
                current += ch
                continue
            if ch == ',' and depth_paren == depth_brace == depth_bracket == 0:
                _flush()
                continue
            if ch == '(':
                depth_paren += 1
            elif ch == ')':
                depth_paren = max(depth_paren - 1, 0)
            elif ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace = max(depth_brace - 1, 0)
            elif ch == '[':
                depth_bracket += 1
            elif ch == ']':
                depth_bracket = max(depth_bracket - 1, 0)
            current += ch
        _flush()
        return args

    def _constructing_type_builder_entry(self, name):
        """Builder of that name owned by the type under construction, or None.

        Bare builder calls (``prepare(24)``) resolve here while a type's own
        constructor or builder body executes; the ``$`` prefix is optional
        because the call sits in the type's own construction scope.
        """
        type_name = getattr(self, '_constructing_type', None)
        if not type_name:
            return None
        type_def = self.types_defined.get(str(type_name).lower(), {})
        if not isinstance(type_def, dict):
            return None
        builders = type_def.get('_builders') or {}
        return builders.get(str(name).lower().lstrip('$'))

    def _call_construction_builder(self, name, arg_values, line_number=None):
        """Run a bare builder call against the in-progress instance.

        The instance is the constructor's ``this``; the builder mutates it
        in place and the instance is returned (the same result a ``new``
        chain yields).  Raises NameError when the name is not a builder of
        the constructing type or the call escapes construction scope.
        """
        if line_number is None:
            line_number = getattr(self, '_current_call_line', None) or 0
        type_name = getattr(self, '_constructing_type', None)
        entry = self._constructing_type_builder_entry(name)
        if entry is None:
            raise NameError(
                f"Builder '{name}' is not defined for type "
                f"'{type_name}' at line {line_number}")
        try:
            instance = self.current_scope().get('this')
        except Exception:
            instance = None
        if not isinstance(instance, dict):
            raise NameError(
                f"Builder '{name}' can only run during construction of type "
                f"'{type_name}' at line {line_number}")
        self.type_processor._execute_builder(
            str(type_name), name, instance, line_number,
            list(arg_values or []))
        # The builder ran with its own scope chain; mirror the mutated
        # fields back into the constructor's scope so a later field sync
        # does not clobber the builder's writes with stale scope values.
        self.type_processor._push_type_fields_to_scope(instance)
        return instance

    def _maybe_handle_subprocess_call(self, line, line_number):
        """Detect and execute bare subprocess call lines."""
        m = re.match(r'^\s*([\w\.]+)\s*\((.*)\)\s*$', line)
        if not m:
            return False
        name, args_str = m.groups()
        subprocess_defs = getattr(self, 'subprocesses', {}) or {}
        if (not subprocess_defs or name.lower() not in subprocess_defs) and hasattr(self, 'compiler'):
            subprocess_defs = getattr(self.compiler, 'subprocesses', {}) or {}
        if name.lower() not in subprocess_defs:
            if self._constructing_type_builder_entry(name) is not None:
                arg_parts = self._split_call_arguments(args_str)
                arg_values = [
                    self.expr_evaluator.eval_or_eval_array(
                        a, self.current_scope().get_evaluation_scope(),
                        line_number)
                    for a in arg_parts]
                self._call_construction_builder(
                    name, arg_values, line_number)
                return True
            if self._bare_handle_member_call(name, args_str, line, line_number):
                return True
            return False
        arg_parts = self._split_call_arguments(args_str)
        sp_def = subprocess_defs[name.lower()]
        inputs = sp_def.get('inputs', []) or []
        # Predefined engine subprocesses may declare optional trailing inputs
        # (e.g. Ticker.Reset() defaults the counter to 0).
        if len(arg_parts) < len(inputs) and not sp_def.get('_system'):
            raise ValueError(
                f"Subprocess '{name}' expects at least {len(inputs)} arguments at line {line_number}")
        call_fn = getattr(self, 'call_subprocess', None) or getattr(
            getattr(self, 'compiler', None), 'call_subprocess', None)
        if not call_fn:
            raise AttributeError("call_subprocess not available on executor")
        # Live signatures map args in declaration order: live inputs keep their
        # raw expression (lazily re-evaluated) and live outputs become
        # write-through bindings. Non-live-only subprocesses keep the legacy
        # count-then-bind handling below.
        route_fn = getattr(self, '_route_live_call', None) or getattr(
            getattr(self, 'compiler', None), '_route_live_call', None)
        if route_fn is not None:
            live_route = route_fn(
                sp_def, arg_parts,
                lambda raw: self.expr_evaluator.eval_or_eval_array(
                    raw, self.current_scope().get_evaluation_scope(),
                    line_number),
                line_number)
            if live_route is not None:
                input_vals, live_sources, live_bindings = live_route
                call_fn(
                    name, input_vals,
                    live_input_sources=live_sources,
                    live_output_bindings=live_bindings,
                    caller_scope=self.current_scope(),
                    line_number=line_number)
                return True
        input_vals = []
        for idx, raw in enumerate(arg_parts):
            if idx < len(inputs):
                val = self.expr_evaluator.eval_or_eval_array(
                    raw, self.current_scope().get_evaluation_scope(), line_number)
                input_vals.append(val)
            else:
                break
        output_bindings = [a for a in arg_parts[len(inputs):]]
        sp_result = call_fn(
            name, input_vals, output_bindings, line_number=line_number)
        # Fallback: bind subprocess outputs from the returned result if they were not applied.
        if output_bindings and sp_def.get('outputs'):
            try:
                outputs_map = getattr(sp_result, 'outputs', {}) or {}
                vars_map = getattr(sp_result, '_variables', {}) or {}
                apply_fn = getattr(self, '_apply_single_binding', None) or getattr(
                    getattr(self, 'compiler', None), '_apply_single_binding', None)
                if apply_fn:
                    for idx, out_name in enumerate(sp_def.get('outputs', [])):
                        if idx >= len(output_bindings):
                            break
                        binding = output_bindings[idx]
                        if binding is None:
                            continue
                        out_key = out_name.lower()
                        val = outputs_map.get(out_key)
                        if val is None and vars_map:
                            for key, value in vars_map.items():
                                if key.lower() == out_key:
                                    val = value
                                    break
                        if val is not None:
                            apply_fn(
                                binding, val, self.current_scope(), line_number=line_number)
            except Exception:
                pass
        return True

    def _bare_handle_member_call(self, name, args_str, line, line_number):
        """Run a stand-alone ``handle.Method(args)`` statement.

        Engine handles own their members, so a bare call on one is a
        side-effecting subprocess call (``srv.Start()``, ``srv.Stop()``,
        ``rq.Reply(...)``), not a no-op read. Resolves the base variable to
        its handle, verifies the member is a real engine builtin for that
        handle type, then evaluates the line and discards the value.
        """
        if '.' not in name:
            return False
        base, _, member = name.partition('.')
        if not member or '.' in member:
            return False
        handles = self._scope_handle_vars()
        handle = handles.get(base.lower())
        if handle is None:
            return False
        htype = str(handle[1].get('_handle_type') or '').lower()
        if not htype:
            return False
        try:
            from builtin_functions import BUILTINS
            key = f"{htype}.{member}".lower()
            if key not in BUILTINS:
                return False
        except ImportError:
            return False
        try:
            self.expr_evaluator.eval_or_eval_array(
                line, self.current_scope().get_evaluation_scope(), line_number)
        except Exception:
            return False
        return True

    def _analyze_loop_dependencies(self, normalized, original_line, line_number):
        """Create a dependency node for a FOR loop header."""
        body = normalized[3:].strip()
        header_part = re.split(r'\bdo\b', body, maxsplit=1, flags=re.I)[
            0].strip()
        defined_names = []
        dependencies = set()
        index_name = None
        if re.search(r'\bin\b', header_part, re.I):
            iterator_part, expr_part = re.split(
                r'\bin\b', header_part, maxsplit=1, flags=re.I)
            defined_names = [name.strip()
                             for name in iterator_part.split(',') if name.strip()]
            index_match = re.search(
                r'\bindex\s+([A-Za-z][A-Za-z0-9_.]*)\b', expr_part, re.I)
            if index_match:
                index_name = index_match.group(1)
                expr_part = re.sub(
                    r'\bindex\s+[A-Za-z][A-Za-z0-9_.]*\b', '', expr_part, flags=re.I)
            dependencies = self._extract_dependencies_from_expression(
                expr_part)
        else:
            try:
                var, type_name, constraints, expr = self._parse_variable_def(
                    header_part, line_number)
                if var:
                    defined_names.append(var)
                var_list = constraints.get('var_list') if constraints else None
                if var_list:
                    defined_names.extend(var_list)
                dependencies |= self._collect_dependencies_from_constraints(
                    constraints)
                dependencies |= self._extract_dependencies_from_expression(
                    expr)
            except Exception:
                dependencies = self._extract_dependencies_from_expression(
                    header_part)
        defined_lower = {name.lower() for name in defined_names}
        filtered_deps = [
            dep for dep in dependencies if dep.lower() not in defined_lower]
        if index_name:
            filtered_deps = [
                dep for dep in filtered_deps if dep.lower() != index_name.lower()]
        return {
            'kind': 'loop',
            'line': line_number,
            'source': original_line,
            'defines': defined_names,
            'depends_on': filtered_deps,
            'data': {
                'header': normalized,
                'loop_type': 'for'
            }
        }

    def _analyze_when_dependencies(self, normalized, original_line, line_number):
        """Create a dependency node for a WHEN block header."""
        header_expr = re.split(
            r'\bdo\b', normalized[4:].strip(), maxsplit=1, flags=re.I)[0].strip()
        dependencies = self._extract_dependencies_from_expression(header_expr)
        return {
            'kind': 'loop',
            'line': line_number,
            'source': original_line,
            'defines': [],
            'depends_on': list(dependencies),
            'data': {
                'header': normalized,
                'loop_type': 'when'
            }
        }

    def _register_dependency_node(self, node):
        """Record a node inside the dependency graph."""
        defines = sorted({name for name in node.get('defines', []) if name})
        depends_on = sorted({dep for dep in node.get('depends_on', []) if dep})
        node['defines'] = defines
        node['depends_on'] = depends_on
        node['depth'] = node.get('depth', 0)
        self.dependency_graph['nodes'].append(node)
        self.dependency_graph['by_line'][node['line']] = node
        for defined in defines:
            key = defined.lower()
            info = self.dependency_graph['by_variable'].setdefault(
                key, {'definitions': set(), 'dependents': set()})
            info['definitions'].add(node['line'])
        for dep in depends_on:
            key = dep.lower()
            info = self.dependency_graph['by_variable'].setdefault(
                key, {'definitions': set(), 'dependents': set()})
            info['dependents'].add(node['line'])

    def _build_dependency_network(self, lines, guard_entries=None):
        """Construct the dependency graph for the provided program lines."""
        self._reset_dependency_graph()
        guard_entries = guard_entries or []
        guard_map = {entry['line_number']: entry for entry in guard_entries}
        depth = 0
        for raw_line, line_number in lines:
            stripped_line = self.control_flow._strip_inline_comment(raw_line)
            normalized_line = stripped_line.rstrip()
            clean_line = normalized_line.strip()
            lowered_clean = clean_line.lower()
            is_block_start = (
                (lowered_clean.startswith('if ') and lowered_clean.endswith('then')) or
                (lowered_clean.startswith('for ') and lowered_clean.endswith('do')) or
                (lowered_clean.startswith('when ') and lowered_clean.endswith('do'))
            )
            is_end = lowered_clean == 'end'
            depth_change = (1 if is_block_start else 0) + (-1 if is_end else 0)
            current_depth = depth

            if not clean_line:
                depth = max(0, depth + depth_change)
                continue
            if line_number in guard_map:
                condition = guard_map[line_number]['condition']
                deps = self._extract_dependencies_from_expression(condition)
                node = {
                    'kind': 'guard',
                    'line': line_number,
                    'source': raw_line,
                    'defines': [],
                    'depends_on': list(deps),
                    'data': {'condition': condition}
                }
                node['depth'] = current_depth
                self._register_dependency_node(node)
                depth = max(0, depth + depth_change)
                continue
            normalized = clean_line
            lowered = normalized.lower()
            if lowered in {'end', 'else', 'elseif'}:
                depth = max(0, depth + depth_change)
                continue
            is_declaration = normalized.startswith(':')
            if is_declaration:
                normalized = normalized[1:].strip()
                lowered = normalized.lower()
            if not normalized:
                depth = max(0, depth + depth_change)
                continue
            if lowered.startswith('define '):
                depth = max(0, depth + depth_change)
                continue
            if lowered.startswith('input '):
                depth = max(0, depth + depth_change)
                continue
            if lowered.startswith('output '):
                depth = max(0, depth + depth_change)
                continue
            if lowered.startswith('label '):
                depth = max(0, depth + depth_change)
                continue
            if lowered.startswith('type '):
                depth = max(0, depth + depth_change)
                continue
            if lowered.startswith('dim '):
                depth = max(0, depth + depth_change)
                continue
            if lowered.startswith('push ') or lowered.startswith('return ') or lowered.startswith('print '):
                depth = max(0, depth + depth_change)
                continue
            # Skip declarations that appear inside a block; they are handled at runtime.
            if is_declaration and current_depth > 0:
                depth = max(0, depth + depth_change)
                continue
            if lowered.startswith('for ') and ' do' in lowered:
                node = self._analyze_loop_dependencies(
                    normalized, raw_line, line_number)
                if node:
                    node['depth'] = current_depth
                    self._register_dependency_node(node)
                depth = max(0, depth + depth_change)
                continue
            if lowered.startswith('when ') and ' do' in lowered:
                node = self._analyze_when_dependencies(
                    normalized, raw_line, line_number)
                if node:
                    node['data']['loop_type'] = 'when'
                    node['depth'] = current_depth
                    self._register_dependency_node(node)
                depth = max(0, depth + depth_change)
                continue
            assignment_node = self._parse_assignment_like_line(
                normalized, raw_line, line_number, is_declaration=is_declaration)
            if assignment_node:
                if isinstance(assignment_node, list):
                    for node in assignment_node:
                        if node:
                            node['depth'] = current_depth
                            self._register_dependency_node(node)
                else:
                    assignment_node['depth'] = current_depth
                    self._register_dependency_node(assignment_node)
            depth = max(0, depth + depth_change)
        for info in self.dependency_graph['by_variable'].values():
            info['definitions'] = sorted(info['definitions'])
            info['dependents'] = sorted(info['dependents'])
    def _evaluate_global_guards_pre_execution(self, guard_entries):
        """Evaluate global guard conditions before executing main body."""
        if not guard_entries:
            self._global_guards_pre_evaluated = True
            return True


        unresolved_overall = []
        current_scope = self.current_scope()
        for entry in guard_entries:
            line_number = entry['line_number']
            node = self.dependency_graph['by_line'].get(line_number, {})
            deps = node.get('depends_on', [])
            missing = []
            for dep in deps:
                defining_scope = current_scope.get_defining_scope(dep)
                if not defining_scope:
                    missing.append(dep)
                    continue
                if defining_scope.is_uninitialized(dep):
                    missing.append(dep)
                    continue
                if self.has_unresolved_dependency(dep, scope=current_scope):
                    missing.append(dep)
            if missing:
                unresolved_overall.extend(missing)

        if unresolved_overall:
            self._global_guards_pre_evaluated = False
            return True

        resolver = getattr(self, '_resolve_pending_assignments', None)
        if callable(resolver):
            try:
                resolver()
            except Exception as exc:
                pass

        guard_passed = self._evaluate_guard_conditions(guard_entries)
        if not guard_passed:
            self.global_guard_allows_execution = False
            self.grid.clear()
            self.output_values.clear()
        self._global_guards_pre_evaluated = True
        return guard_passed

    def _execute_simple_for_assignment(self, line, line_number):
        """Handle simple FOR declarations like 'For x as number = expr'."""
        stripped = line.strip()
        if not stripped.lower().startswith('for '):
            return False
        if 'grid dim' in stripped.lower():
            return False
        remainder = stripped[3:].strip()
        if ' and ' in remainder.lower():
            return False
        try:
            var, type_name, constraints, expr = self._parse_variable_def(
                remainder, line_number)
        except Exception:
            return False
        if expr is None:
            if (constraints or {}).get('default') is not None:
                return bool(self._bind_declared_var(
                    var, type_name, constraints, expr, line_number))
            return False
        status = self._bind_declared_var(
            var, type_name, constraints, expr, line_number)
        if not status:
            return False
        self._register_listeners(var, expr, self.current_scope())
        return True

    def _is_root_binding_line(self, stripped):
        """Whether a depth-0 line is a plain/`Let` binding that hoists.

        Block headers (``Let ... then``) and statement keywords are not
        bindings; member/index writes, multi-binding ``and`` forms, INIT
        forms, cell references and function/subprocess calls stay with the
        main loop because their evaluation is order-sensitive.
        """
        if not stripped or stripped.startswith('[') or ':=' in stripped:
            return False
        lower = stripped.lower()
        is_let_keyword = lower.startswith('let ')
        is_block = lower.endswith('then')
        if is_let_keyword and is_block:
            return False
        if not is_let_keyword and not self.control_flow._is_optional_let_line(
                stripped):
            return False
        body = stripped[4:].strip() if is_let_keyword else stripped
        if ' and ' in body.lower():
            return False
        try:
            var, type_name, constraints, expr = self._parse_variable_def(
                body, 0)
        except Exception:
            return False
        if not var or not re.fullmatch(r'[A-Za-z_]\w*', var):
            return False
        if not (expr or type_name or constraints):
            return False
        if constraints and 'init' in constraints:
            return False
        if expr is not None and not isinstance(expr, str):
            return False
        if expr is not None and self._match_fp_call_name(expr) is not None:
            return False
        return True

    def _hoist_root_bindings(self, lines):
        """Wave-0: apply root-level plain/`Let` bindings before the main loop.

        A binding applies first: ``x = 1`` written after a Let block or a
        For-do header is live when those headers run, so redefinition
        checks and Let proposals observe the finished binding.  Lines
        whose dependencies are not resolvable yet, or whose evaluation is
        order-sensitive, stay with the main loop at their source position.
        """
        self.hoisted_binding_lines = set()
        candidates = []
        depth = 0
        type_depth = 0
        for line, line_number in lines:
            stripped = line.strip()
            lowered = stripped.lower()
            is_block_start = (
                (lowered.startswith('if ') and lowered.endswith('then')) or
                (lowered.startswith('for ') and lowered.endswith('do')) or
                (lowered.startswith('when ') and lowered.endswith('do')) or
                (lowered.startswith('let ') and lowered.endswith('then'))
            )
            is_end = lowered == 'end'
            if lowered.startswith("define ") and (
                    " as type" in lowered or " as keytype" in lowered
                    or " as resource" in lowered or " as handle" in lowered):
                type_depth += 1
            elif lowered.startswith("end") and type_depth > 0:
                type_depth -= 1
                continue
            if depth == 0 and type_depth == 0 and stripped and (
                    self._is_root_binding_line(stripped)):
                candidates.append((stripped, line_number))
            if is_block_start:
                depth += 1
            elif is_end and depth > 0:
                depth -= 1
        remaining = candidates
        while remaining:
            still = []
            progressed = False
            for stripped, line_number in remaining:
                is_let_keyword = stripped.lower().startswith('let ')
                body = stripped[4:].strip() if is_let_keyword else stripped
                try:
                    var, _t, _c, expr = self._parse_variable_def(
                        body, line_number)
                except Exception:
                    continue
                deps = self._compute_assignment_deps(
                    expr) if expr is not None else set()
                if any(self.has_unresolved_dependency(
                        dep, scope=self.current_scope()) for dep in deps):
                    still.append((stripped, line_number))
                    continue
                source = stripped if is_let_keyword else 'Let ' + stripped
                try:
                    self._handle_main_loop_let_statement(
                        lines, 0, source, line_number,
                        source if is_let_keyword else 'Let ' + stripped)
                except Exception:
                    still.append((stripped, line_number))
                    continue
                self.hoisted_binding_lines.add(line_number)
                progressed = True
            if not progressed:
                break
            remaining = still

    def _execute_global_for_loops(self, lines):
        """Evaluate global FOR declarations prior to main execution."""
        if not self.global_for_line_numbers:
            return
        line_map = {line_number: line for line,
                    line_number in lines if line_number in self.global_for_line_numbers}
        remaining = set(self.global_for_line_numbers)
        max_passes = len(remaining) + 3
        passes = 0
        # Map line number to index for block extraction
        line_index_map = {ln: idx for idx, (_, ln) in enumerate(lines)}

        while remaining and passes < max_passes:
            progress = False
            for line_number in sorted(list(remaining)):
                if line_number in getattr(self, 'for_guard_conditions', {}):
                    remaining.discard(line_number)
                    continue
                line = line_map.get(line_number)
                if not line:
                    remaining.discard(line_number)
                    continue
                node = self.dependency_graph['by_line'].get(line_number)
                deps = node.get('depends_on', []) if node else []
                unresolved = any(self.has_unresolved_dependency(
                    dep, scope=self.current_scope()) for dep in deps)
                if unresolved:
                    continue
                line_lower = line.lower()
                executed = False
                # Only attempt simple assignment handling for non-block FOR statements
                if ' do' not in line_lower:
                    try:
                        executed = self._execute_simple_for_assignment(
                            line, line_number)
                    except NameError as exc:
                        missing = self.extract_missing_dependencies(exc)
                        if missing:
                            for dep in missing:
                                self.mark_dependency_missing(dep)
                            continue
                        raise

                if not executed and (' and ' in line_lower or ' dim ' in line_lower):
                    remaining.discard(line_number)
                    continue
                try:
                    if not executed:
                        if ' do' in line_lower:
                            # Execute block FOR once in pre-pass
                            idx = line_index_map.get(line_number)
                            if idx is None:
                                raise SyntaxError(
                                    f"Cannot locate FOR block start at line {line_number}")
                            body_lines, end_idx = self.control_flow._extract_block_body(
                                lines, idx)
                            # A block For bound to a function that delivers a
                            # stream wave-iterates: the body runs once per
                            # delivered value (Init materializes the whole
                            # stream into one value instead).  The header is
                            # read with the shared declaration parser, same
                            # as every other For-equality form.
                            wave_iterated = False
                            stream_var, stream_expr = None, None
                            for_defs = re.sub(
                                r'^\s*For\s+', '', line, flags=re.I)
                            for_defs = re.sub(
                                r'\s+do\s*$', '', for_defs, flags=re.I
                            ).strip()
                            try:
                                stream_var, _st, _sc, parsed_expr = (
                                    self._parse_variable_def(
                                        for_defs, line_number))
                                if isinstance(parsed_expr, str):
                                    stream_expr = parsed_expr
                            except Exception:
                                pass
                            # The For block owns its scope: everything it
                            # binds dies at End.
                            if self._match_fp_call_name(
                                    stream_expr) is not None:
                                stream_values = self._evaluate_push_expression(
                                    stream_expr, line_number)
                                if len(stream_values) > 1:
                                    wave_iterated = True
                                    self.push_scope(is_private=True)
                                    block_scope = self.current_scope()
                                    try:
                                        for stream_value in stream_values:
                                            if is_error_value(stream_value):
                                                continue
                                            inferred = self.array_handler.infer_type(
                                                stream_value, line_number)
                                            if inferred == 'int':
                                                inferred = 'number'
                                            def_scope = block_scope.get_defining_scope(
                                                stream_var)
                                            if def_scope is not None:
                                                def_scope.update(
                                                    stream_var, stream_value,
                                                    line_number)
                                            else:
                                                block_scope.define(
                                                    stream_var, stream_value,
                                                    inferred, {}, False)
                                            self.control_flow._process_block(
                                                body_lines)
                                    except BaseException:
                                        self.control_flow._unwind_block_scope(
                                            block_scope)
                                        raise
                                    self.control_flow._pop_for_block_scope(block_scope)
                            if not wave_iterated:
                                # Define the loop variable in the block's own
                                # scope so it does not outlive the block.
                                self.push_scope(is_private=True)
                                block_scope = self.current_scope()
                                try:
                                    self.control_flow.process_for_statement(
                                        line, line_number, block_scope)
                                    # Do not execute the block when the loop
                                    # variable holds an error value
                                    # (``For x = <error> do ... End``).
                                    skip_block = False
                                    if stream_var:
                                        def_scope = block_scope.get_defining_scope(
                                            stream_var)
                                        if def_scope is not None:
                                            var_value = def_scope.get(
                                                stream_var)
                                            if is_error_value(var_value):
                                                skip_block = True
                                    # Execute the block body once
                                    if not skip_block:
                                        self.control_flow._process_block(
                                            body_lines)
                                    else:
                                        # The loop never runs, but its
                                        # declarations still declare (unvalued).
                                        self.control_flow._predeclare_block_assignment_targets(
                                            body_lines)
                                except BaseException:
                                    self.control_flow._unwind_block_scope(
                                        block_scope)
                                    raise
                                self.control_flow._pop_for_block_scope(block_scope)
                            # Mark body lines as executed so main loop skips them
                            for _, body_ln in body_lines:
                                self.executed_global_for_lines.add(body_ln)
                            if end_idx is not None and end_idx < len(lines):
                                end_line_number = lines[end_idx][1]
                                self.executed_global_for_lines.add(
                                    end_line_number)
                        else:
                            self.control_flow.process_for_statement(
                                line, line_number, self.current_scope())
                    self.executed_global_for_lines.add(line_number)
                    remaining.discard(line_number)
                    progress = True
                except SyntaxError as exc:
                    remaining.discard(line_number)
                    continue
                except NameError as exc:
                    missing = self.extract_missing_dependencies(exc)
                    if missing:
                        for dep in missing:
                            self.mark_dependency_missing(dep)
                        continue
                    raise
                except Exception as exc:
                    raise
            if not progress:
                break
            passes += 1

    def _determine_needed_lines(self):
        """Compute the set of line numbers required to satisfy outputs."""
        if not self.dependency_graph['nodes']:
            return set()

        needed_lines = set()
        required_vars = set()

        for node in self.dependency_graph['nodes']:
            defines = node.get('defines', [])
            if any(name.startswith('[') or '.' in name for name in defines):
                needed_lines.add(node['line'])
                for name in defines:
                    if name:
                        required_vars.add(name.lower())

        for var in getattr(self, 'output_variables', []):
            if var:
                required_vars.add(var.lower())
        required_vars.add('output')

        queue = deque(required_vars)
        visited = set()

        for entry in getattr(self, 'global_guard_entries', []):
            line_number = entry.get('line_number')
            if line_number:
                needed_lines.add(line_number)
                node = self.dependency_graph['by_line'].get(line_number)
                if node:
                    for dep in node.get('depends_on', []):
                        queue.append(dep.lower())

        while queue:
            var = queue.popleft()
            if not var:
                continue
            var_lower = var.lower()
            if var_lower in visited:
                continue
            visited.add(var_lower)

            entry = self.dependency_graph['by_variable'].get(var_lower)
            if not entry:
                continue

            for line in entry.get('definitions', []):
                needed_lines.add(line)
                node = self.dependency_graph['by_line'].get(line)
                if node:
                    for dep in node.get('depends_on', []):
                        queue.append(dep.lower())

        return needed_lines

    def _evaluate_guard_conditions(self, guard_entries):
        """Evaluate recorded guard conditions after execution."""
        if not guard_entries:
            return True

        all_true = True
        for entry in guard_entries:
            condition = entry['condition']
            line_number = entry['line_number']
            try:
                result = self.control_flow._evaluate_if_condition(
                    condition, line_number)
                if not result:
                    all_true = False
                    break
            except Exception as exc:
                all_true = False
                break
        return all_true

    def _attempt_resolve_pending_var(self, var_name, trigger_line):
        """Try to resolve a global pending assignment for a specific variable."""
        if var_name.startswith('__line_') or var_name.startswith('__if_line_'):
            # A deferred statement, not a variable assignment: its "expr" is a
            # whole line, and it is retried by the deferred-statement passes.
            return False
        if var_name not in self.pending_assignments:
            return False
        assignment = self.pending_assignments[var_name]
        expr, assign_line, deps = assignment[:3]
        constraints = assignment[3] if len(assignment) > 3 else {}
        scope = self.current_scope()
        unresolved = any(
            dep != var_name and self.has_unresolved_dependency(
                dep, scope=scope)
            for dep in deps)
        if unresolved:
            return False
        try:
            value = self.expr_evaluator.eval_or_eval_array(
                expr, scope.get_full_scope(), assign_line)
            value = self.array_handler.check_dimension_constraints(
                var_name, value, assign_line)
            if constraints.get('with'):
                defining_scope = scope.get_defining_scope(var_name)
                type_name = None
                if defining_scope:
                    actual_key = defining_scope._get_case_insensitive_key(
                        var_name, defining_scope.types)
                    if actual_key:
                        type_name = defining_scope.types.get(actual_key)
                value = self._apply_with_constraints(
                    value, constraints.get('with', {}),
                    scope.get_full_scope(), assign_line,
                    type_name=type_name)
            defining_scope = scope.get_defining_scope(var_name) or scope
            defining_scope.update(var_name, value, assign_line)
            del self.pending_assignments[var_name]
            return True
        except ValueError as e:
            if is_definition_body_error(e):
                raise RuntimeError(
                    f"Error resolving pending '{var_name}' from '{expr}': {e} "
                    f"at line {assign_line}")
            # Not satisfiable right now; leave it pending for a later pass.
            return False
        except NameError as e:
            if is_definition_body_error(e):
                raise RuntimeError(
                    f"Error resolving pending '{var_name}' from '{expr}': {e} "
                    f"at line {assign_line}")
            missing = self.extract_missing_dependencies(e)
            if missing:
                for dep in missing:
                    self.mark_dependency_missing(dep)
                updated_deps = set(deps) | set(missing)
                self.pending_assignments[var_name] = (
                    expr, assign_line, updated_deps, constraints)
            return False
        except Exception as e:
            raise RuntimeError(
                f"Error resolving pending '{var_name}' from '{expr}': {e} "
                f"at line {assign_line}")

    def _resolve_ready_pending_vars(self, trigger_line):
        """Resolve any global pending variable assignments that are ready."""
        resolver = getattr(self, 'compiler', None) or self
        pending_map = getattr(resolver, 'pending_assignments', {}) or {}
        passes = 0
        max_passes = len(pending_map) + 3
        while passes < max_passes:
            pending_vars = [
                name for name in list(pending_map.keys())
                if not name.startswith('__line_') and not name.startswith('__if_line_')
            ]
            if not pending_vars:
                break
            progress = False
            for pending_var in pending_vars:
                resolved = False
                if hasattr(resolver, '_resolve_global_dependency'):
                    # _resolve_global_dependency already classifies failures:
                    # NameError/ValueError mean "not ready yet" and it returns
                    # False, anything else it re-raises. Swallowing that raise
                    # here hid every error raised inside a constructor body,
                    # leaving the variable unresolved and the program running
                    # on with a None it never mentions.
                    resolved = bool(resolver._resolve_global_dependency(
                        pending_var,
                        trigger_line,
                        target_scope=self.current_scope(),
                    ))
                else:
                    resolved = self._attempt_resolve_pending_var(
                        pending_var, trigger_line)
                if resolved:
                    progress = True
            pending_map = getattr(resolver, 'pending_assignments', {}) or {}
            if not progress:
                break
            passes += 1

    def _is_state_affecting_statement(self, stripped):
        """Whether a line can bind or mutate program state.

        The dead-code filter in ``_prepare_main_loop_line`` may only drop a
        line that reaches no requested output when dropping it cannot change
        observable state. That holds for bare value-producing expressions and
        nothing else, so every declaration, assignment and control statement
        is treated as state-affecting regardless of which keyword carries it.
        """
        if not stripped:
            return False
        if stripped.startswith('['):
            # A cell assignment or a cell declaration (``[A1]: value``).
            return True
        if '=' in stripped:
            return True
        if _first_keyword(stripped) in _STATEMENT_KEYWORDS:
            return True
        if self.control_flow._is_optional_let_line(stripped):
            return True
        return False

    def _prepare_main_loop_line(self, lines, i):
        """Prepare and pre-filter the current main-loop line."""
        # Resolve ready global declarations before INIT materialization so
        # FOR/LET/: INIT can consume newly available sources uniformly.
        self._resolve_ready_pending_vars(lines[i][1] if i < len(lines) else None)
        # Realize any INIT values whose dependencies are now satisfied before executing the next line
        self._materialize_inits()
        if not self.global_guard_allows_execution:
            return {'action': 'break'}

        line, line_number = lines[i]
        stripped = line.strip()
        stripped_lower = stripped.lower()

        if _first_keyword(stripped) == 'while':
            raise SyntaxError(
                f"'While' is not part of the language at line {line_number}; use 'When <condition> do ... End' instead")

        if line_number in self.executed_global_for_lines:
            return {'action': 'continue', 'next_i': i + 1}
        if line_number in getattr(self, 'hoisted_binding_lines', set()):
            return {'action': 'continue', 'next_i': i + 1}
        if not stripped:
            return {'action': 'continue', 'next_i': i + 1}
        if stripped_lower == 'end':
            return {'action': 'continue', 'next_i': i + 1}
        # Skip global declarations as they're already processed in _process_declarations_and_labels
        if stripped.startswith(':'):
            # Allow global ': var init ...' to run at executor time so
            # 'new Type' INIT expressions are materialised (compiler-only
            # handling doesn't bind values for executor's scope).
            if re.match(r'^:\s*[A-Za-z_][\w_\.]*\s+init\b', stripped, re.I):
                pass  # don't skip; let it be handled as a declaration
            else:
                return {'action': 'continue', 'next_i': i + 1}
        # Require lines resolve during preparation; nothing executes at runtime.
        if stripped_lower.startswith('require '):
            return {'action': 'continue', 'next_i': i + 1}

        if line_number in self.global_guard_line_numbers:
            return {'action': 'continue', 'next_i': i + 1}

        node = self.dependency_graph['by_line'].get(line_number)
        if (node and self.needed_line_numbers
                and line_number not in self.needed_line_numbers
                and not self._is_state_affecting_statement(stripped)):
            return {'action': 'continue', 'next_i': i + 1}

        return {
            'action': 'process',
            'line': line,
            'line_number': line_number,
            'stripped': stripped,
            'stripped_lower': stripped_lower,
        }

    def _handle_main_loop_quick_statements(self, lines, i, line, line_number, stripped, stripped_lower):
        """Handle single-line and light-weight statements before heavy branches."""
        # Phase B of the wave model: a bare Push/Print/Return collected as a
        # stream consumer runs one round - once per recorded wave, with that
        # wave's values bound - instead of a single execution.  Deferred
        # until every plain binding has run, so the effects observe the
        # finished bindings.  Init lines are skipped here: the wave hook
        # already queued their push effect into the deferred list.
        wave_effects = getattr(self, '_wave_effects', None)
        if wave_effects and getattr(self, '_wave_phase_a_done', False):
            wave_effect = wave_effects.get(line_number)
            if wave_effect is not None and line_number not in (
                    getattr(self, '_wave_init_lines', None) or set()):
                if getattr(self, '_wave_remaining_bindings', None):
                    self._wave_deferred_effects.append(wave_effect)
                else:
                    self._execute_wave_effect_round(wave_effect)
                return True, i + 1
        if stripped_lower.startswith("for ") and 'grid dim' in stripped_lower:
            handled = self._handle_for_grid_dim(
                stripped, line_number)
            if handled:
                return True, i + 1
        elif stripped_lower.startswith("return "):
            self._handle_return_statement(stripped, line_number)
            return True, i + 1
        elif stripped_lower == "print" or stripped_lower.startswith("print "):
            self._handle_print_statement(stripped, line_number)
            return True, i + 1
        elif stripped_lower.startswith("push "):
            self._handle_push_assignment_line(stripped, line_number)
            return True, i + 1
        elif line.lower().strip().startswith("for ") and " do " in line.lower() and self._has_push_action(line):
            handled, next_i = self._handle_for_push_loop(lines, i, line, line_number)
            if handled:
                return True, next_i
        elif re.search(r'\.push\s*\(', line, re.I):
            raise SyntaxError(
                f"'.push()' syntax is not supported. Use 'Push var = value' instead at line {line_number}")
        # Array access like names(0) and names[3] will be handled by the expression evaluator
        # No need for special handling here
        elif re.search(r'([\w_]+)!(\w+)\.Label\s*\{([^}]+)\}', line):
            if self._handle_dim_label_assignment(line, line_number):
                return True, i + 1
            return True, i + 1
        return False, i

    def _handle_bare_let_line(self, line, line_number):
        """Run an assignment-shaped line through the Let pipeline.

        Block bodies hand ``P.x = i`` to the cell-assignment path, which only
        understands addresses; giving the line the same treatment as an
        explicit ``Let P.x = i`` keeps construction identical wherever the
        statement appears.  Lines this cannot read as a member write are
        left to the caller.
        """
        if ':=' in line or '=' not in line:
            return False
        var, type_name, constraints, expr = self._parse_variable_def(
            line, line_number)
        if expr is None or '.' not in var:
            return False
        var_list = [(var, type_name, constraints, expr)]
        scope_dict = self.current_scope().get_evaluation_scope()
        member_writes = self._process_let_first_pass(
            var_list, scope_dict, line_number)
        self._run_let_second_pass(var_list, line_number)
        self._execute_let_inline(
            [], 0, line_number, var_list, member_writes)
        return True

    def _handle_main_loop_let_statement(self, lines, i, line, line_number, stripped):
        m = re.match(
            r'^\s*let\s+(.+?)(?:\s+then\s*$|\s*$)', stripped, re.I)
        if not m:
            raise SyntaxError(
                f"Invalid LET syntax at line {line_number}")
        var_def = m.group(1).strip()
        has_block = line.lower().strip().endswith('then')
        var_list = []
        if ' and ' in var_def.lower():
            var_parts = split_var_defs(var_def)
            for var_part in var_parts:
                var, type_name, constraints, expr = self._parse_variable_def(
                    var_part, line_number)
                var_list.append((var, type_name, constraints, expr))
        else:
            var, type_name, constraints, expr = self._parse_variable_def(
                var_def, line_number)
            var_list.append((var, type_name, constraints, expr))


        if has_block:
            # The block form binds inside its own scope, so the header
            # binding must be created after the scope push in
            # ``_execute_let_block`` rather than here in the parent.
            return self._execute_let_block(
                lines, i, line_number, var_list)
        scope_dict = self.current_scope().get_evaluation_scope()
        member_writes = self._process_let_first_pass(
            var_list, scope_dict, line_number)

        self._run_let_second_pass(var_list, line_number)

        return self._execute_let_inline(
            lines, i, line_number, var_list, member_writes)

    def _process_let_first_pass(self, var_list, scope_dict, line_number):
        """Bind every target of one Let line.

        Returns the lowercase names that were *not* declared as variables --
        index writes, field writes and object-member writes -- so the
        post-passes that constrain declared variables can skip them.
        """
        member_writes = set()
        let_var_order = [var.lower() for var, _, _, _ in var_list]
        for idx, (var, type_name, constraints, expr) in enumerate(var_list):
            var_lower = var.lower()
            remaining_let_vars = set(let_var_order[idx + 1:])
            # grid field is read-only outside type
            if '.grid' in var_lower:
                ctx = getattr(self, '_context_grid_stack', None) or getattr(getattr(self, 'compiler', None), '_context_grid_stack', None)
                if not ctx:
                    raise PermissionError(
                        f"Field 'grid' is read-only outside type at line {line_number}")

            if self._try_handle_let_field_assignment(
                    var, expr, scope_dict, line_number):
                member_writes.add(var_lower)
                continue

            if self._try_handle_let_index_assignment(
                    var, expr, scope_dict, line_number):
                member_writes.add(var_lower)
                continue

            if self._try_handle_dotted_member_let(
                    var, expr, scope_dict, line_number):
                member_writes.add(var_lower)
                continue

            self._process_let_standard_assignment(
                var,
                type_name,
                constraints,
                expr,
                scope_dict,
                line_number,
                remaining_let_vars,
            )
            remaining_let_vars.discard(var_lower)
        return member_writes

    def _try_handle_let_field_assignment(
            self, var, expr, scope_dict, line_number):
        # grid field is read-only outside type
        if '.grid' in var.lower():
            ctx = getattr(self, '_context_grid_stack', None) or getattr(getattr(self, 'compiler', None), '_context_grid_stack', None)
            if not ctx:
                raise PermissionError(
                    f"Field 'grid' is read-only outside type at line {line_number}")
        field_index_match = re.match(
            r'^([\w_]+)\.([\w_]+)\(([^)]+)\)\.(\w+)$', var)
        if not field_index_match:
            return False

        obj_name, array_field, index_expr, field_name = field_index_match.groups()
        value = self.expr_evaluator.eval_or_eval_array(
            expr, scope_dict, line_number)
        index_value = self.expr_evaluator.eval_expr(
            index_expr, scope_dict, line_number)
        if isinstance(index_value, float) and index_value.is_integer():
            index_value = int(index_value)
        if not isinstance(index_value, int):
            raise ValueError(
                f"Invalid index '{index_expr}' for '{var}' at line {line_number}")
        indices = [index_value - 1]
        obj_value = self.current_scope().get(obj_name)
        if not isinstance(obj_value, dict):
            raise ValueError(
                f"'{obj_name}' is not an object at line {line_number}")
        public_keys = {k.lower(): k for k in obj_value.keys()
                       if not str(k).startswith('_')}
        actual_array_field = public_keys.get(array_field.lower(), array_field)
        array_val = obj_value.get(actual_array_field)
        if array_val is None:
            raise ValueError(
                f"Field '{array_field}' is not defined on '{obj_name}' at line {line_number}")
        if isinstance(array_val, dict) and 'array' in array_val:
            array_storage = array_val['array']
        else:
            array_storage = array_val
        element = self.array_handler.get_array_element(
            array_storage, indices, line_number)
        if not isinstance(element, dict):
            raise ValueError(
                f"'{obj_name}.{array_field}({index_value})' is not an object at line {line_number}")
        element_keys = {k.lower(): k for k in element.keys()
                        if not str(k).startswith('_')}
        actual_field = element_keys.get(field_name.lower(), field_name)
        element[actual_field] = value
        updated_array = self.array_handler.set_array_element(
            array_storage, indices, element, line_number)
        if isinstance(array_val, dict) and 'array' in array_val:
            array_val['array'] = updated_array
            obj_value[actual_array_field] = array_val
        else:
            obj_value[actual_array_field] = updated_array
        self.current_scope().update(obj_name, obj_value, line_number)
        scope_dict[obj_name] = obj_value
        return True

    def _try_handle_let_index_assignment(
            self, var, expr, scope_dict, line_number):
        return self._try_let_index_assignment(var, expr, scope_dict, line_number)

    def _repin_member_write_constant(self, scope, root_name, root_value,
                                     line_number):
        """Re-freeze an equality-bound root after a field was constructed on it.

        ``: p = new Point`` pins ``p`` to the object the constructor built, so
        the read path keeps comparing ``p`` against that pin.  ``p.x = 3``
        builds a field on the very same object, which would leave the pin and
        the value disagreeing -- the read would store ``#VALUE`` for the whole
        root.  Re-pinning to the object as it now stands keeps the freeze and
        lets the field bind.  A root with no ``constant`` binding has nothing
        to keep in step.
        """
        key = scope._get_case_insensitive_key(root_name, scope.variables)
        if key is None:
            return
        constraints = scope.constraints.get(key)
        if not constraints or 'constant' not in constraints:
            return
        constraint_key = scope._get_case_insensitive_key(
            root_name, scope.constraints) or key
        constraints = dict(constraints)
        constraints['constant'] = {
            k: copy.deepcopy(v) for k, v in root_value.items()
            if not str(k).startswith('_')}
        scope.constraints[constraint_key] = constraints

    def _try_handle_dotted_member_let(
            self, var, expr, scope_dict, line_number):
        """Write a dotted Let target into its object instead of naming a variable.

        ``p.x = 3`` constructs the field, the way ``D(2) = "hi"`` writes an
        array element: the object already carries the field, so the assignment
        goes into it.  Returns False when the target is not a member write so
        the caller falls back to the ordinary declaration path.

        Four shapes stay off this path:

        * no expression (``p.x as number``, ``p.x > 1``) -- those declare
          constraints, they do not write a field;
        * an unknown root -- with nothing to construct on, ``x.y = 100``
          declares a local named ``x.y``, which is what it has always done
          (Test 558b);
        * a root declared in an outer scope -- a subprocess or function
          declaring ``P.x`` binds a local that shadows the caller's member,
          which is what the shadow warning in ``_bind_declared_var`` reports;
        * a root holding no value, or one whose value came from ``init`` --
          only a root a ``=`` binding produced accepts a field Let, so those
          raise and point at ``Push``.
        """
        if expr is None or re.search(r'[\(\{!]', var):
            return False
        path_parts = [part.strip() for part in var.split('.') if part.strip()]
        if len(path_parts) < 2:
            return False
        if not all(re.match(r'^[\w_]+$', part) for part in path_parts):
            return False

        root_name = path_parts[0]
        defining_scope = self.current_scope().get_defining_scope(root_name)
        if defining_scope is None:
            return False
        if self._is_outer_defining_scope(defining_scope):
            return False

        root_key = defining_scope._get_case_insensitive_key(
            root_name, defining_scope.variables)
        if root_key is None:
            return False
        root_value = defining_scope.variables.get(root_key)
        root_constraints = defining_scope.constraints.get(root_key) or {}
        if root_value is None or 'init' in root_constraints:
            raise NameError(
                f"Cannot bind '{var}': '{root_name}' is not bound to a value; "
                f"use Push to update '{var}' at line {line_number}")

        value = self.expr_evaluator.eval_or_eval_array(
            expr, scope_dict, line_number)
        root_value, leaf_parent, owner_path, leaf_key, leaf_name = \
            self._walk_member_path(root_name, path_parts, line_number)

        type_def = None
        type_name = leaf_parent.get('_type_name')
        if isinstance(type_name, str):
            type_def = self.types_defined.get(type_name.lower())
            if not isinstance(type_def, dict):
                type_def = None
        member_keys = type_def.get('_member_keys') if type_def else None
        declared = any(
            not str(key).startswith('_')
            and str(key).lower() == leaf_name.lower()
            for key in leaf_parent)
        if not declared and member_keys:
            declared = leaf_name.lower() in member_keys
        if not declared:
            raise ValueError(
                f"Field '{leaf_name}' is not defined on '{owner_path}' "
                f"at line {line_number}")

        leaf_lower = leaf_name.lower()
        init_fields = type_def.get('_init_fields') if type_def else None
        default_fields = (type_def.get('_default_fields') or set()
                          if type_def else set())
        if init_fields and leaf_lower in init_fields and leaf_lower not in default_fields:
            # The constructor put a value in the field (``: z as number = 1``,
            # ``init``): that is a binding, and a Let may not rebind it, so the
            # field reads back the error value.  ``or =`` only supplies a
            # default, so those fields stay bindable (Test 556 vs 559).
            leaf_parent[leaf_key] = VALUE_ERROR
        else:
            # A field already bound by a Let must be bound again with the
            # identical value -- a different value is a conflict and the
            # field reads back as #VALUE, the same rule plain Let bindings
            # follow (``a.top = 3`` then ``a.top = 4``).
            let_bounds = leaf_parent.get('_let_bounds')
            prior_bound = (let_bounds.get(leaf_lower)
                           if isinstance(let_bounds, dict) else None)
            if (prior_bound is not None
                    and not constant_value_matches(
                        value, prior_bound,
                        self.array_handler.to_display_value)):
                leaf_parent[leaf_key] = VALUE_ERROR
            else:
                violation = self._validate_object_field_value(
                    leaf_parent, leaf_name, value, line_number)
                leaf_parent[leaf_key] = (
                    violation if violation is not None else value)
                if violation is None:
                    if not isinstance(let_bounds, dict):
                        let_bounds = {}
                        leaf_parent['_let_bounds'] = let_bounds
                    let_bounds[leaf_lower] = value

        self._repin_member_write_constant(
            defining_scope, root_name, root_value, line_number)
        self.current_scope().update(root_name, root_value, line_number)
        self._enqueue_push(root_name, root_value)
        self._process_when_triggers()
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

    def _is_call_expression(self, expr):
        """Whether expr is a subprocess/function/constructor call.

        Covers ``Name(args)``, dotted capability member calls like
        ``Cap.Member(args)`` / ``handle.member(args)``, and ``new Type``
        constructor forms (with or without an argument list).

        These are all computed values with side effects, so a Let binding
        holding one must never be pinned as a constant equality binding --
        otherwise ``_check_constraints`` re-evaluates the expression and the
        side effects run twice. ``tick`` in ``Let time = tick.counter()`` is
        a granted capability variable rather than a callable name, so bare
        and dotted forms are both checked against known callables.
        """
        if not isinstance(expr, str):
            return False
        stripped = expr.strip()
        if re.match(r'^new\s+[A-Za-z][A-Za-z0-9_.]*', stripped, re.I):
            return True
        m = re.match(
            r'^([A-Za-z][A-Za-z0-9_.]*)\s*\(.*\)\s*$', stripped, re.S)
        if not m:
            return False
        full_name = m.group(1)
        if '.' in full_name:
            # Dotted capability-member call (e.g. `tick.counter()`).
            base = full_name.split('.', 1)[0].lower()
            return bool(self._scope_handle_vars().get(base) or
                        base in (getattr(self, 'require_caps', None) or {}) or
                        base in (self.subprocesses or {}) or
                        base in (self.functions or {}))
        name = full_name.lower()
        return name in (self.subprocesses or {}) or name in (self.functions or {})

    def _leading_subprocess_call(self, expr):
        """Return the subprocess name if expr starts with a call to it.

        Matches 'SubName(...)' at the start of the expression even when the
        result is further accessed, e.g. 'SpellReverseSub("DIRG").grid'.
        """
        if not isinstance(expr, str):
            return None
        m = re.match(r'^\s*([A-Za-z][A-Za-z0-9_.]*)\s*\(', expr, re.S)
        if not m:
            return None
        name = m.group(1).lower()
        if name in (self.subprocesses or {}):
            return m.group(1)
        return None

    def _literal_rank(self, value):
        """Rank of a nested-list literal (how many levels hold lists)."""
        rank = 0
        node = value
        while isinstance(node, list) and node:
            rank += 1
            if isinstance(node[0], list):
                node = node[0]
            else:
                break
        return rank

    def _to_sparse_undimmed(self, value, constraints):
        """Store undimmed rank>=2 array literals as sparse star-dim arrays.

        Rank (from ';' rows / '|' planes), not element count, decides the
        storage: a literal with no declared dim is an unbounded sparse array
        like a `dim {*,*}` declaration would produce, so ragged rows are legal.
        """
        if not isinstance(value, list):
            return value
        constraints = constraints or {}
        if constraints.get('dim') or constraints.get(
                'shape') or constraints.get('matrix_data'):
            return value
        if self._literal_rank(value) < 2:
            return value
        return self.array_handler.materialize_list_array(value)

    def _existing_member_field(self, var):
        """Return ``(root, field)`` when ``var`` already names a live field.

        ``a.top = 3`` is a LET, so it binds a name rather than writing a
        field. When ``a`` is an object that already exposes ``top``, ``a.top``
        is resolvable and so is not a new variable name: the LET would declare
        a second, unrelated variable literally called ``a.top`` and the field
        write would be silently dropped. Push is the instruction that mutates
        an existing field.
        """
        if '.' not in (var or ''):
            return None, None
        path = [part.strip() for part in var.split('.') if part.strip()]
        if len(path) < 2:
            return None, None
        try:
            value = self.current_scope().get(path[0])
        except NameError:
            # The root is not a variable at all, so there is no member to
            # shadow and the dotted name is free to become a local
            # (Test 558b).
            return None, None
        for part in path[1:-1]:
            if not isinstance(value, dict) or part not in value:
                return None, None
            value = value[part]
        if not isinstance(value, dict) or path[-1] not in value:
            return None, None
        return path[0], path[-1]

    def _bind_declared_var(
            self,
            var,
            type_name,
            constraints,
            expr,
            line_number,
            search_scope=None,
            define_scope=None,
            scope_dict=None,
            shadow_keyword=None,
            mark_implicit=False,
            infer_type=False,
            remaining_let_vars=()):
        """Evaluate and bind a single declared variable.

        Shared by LET and FOR '=' declarations. The binding is deferred to
        pending_assignments when the expression references a name that is not
        yet defined. Returns 'bound', 'deferred', or None when expr is None.
        """
        constraints = dict(constraints or {})
        # `= new Type(...)` is an equality binding to a freshly constructed
        # object, exactly like the ':' constructor declaration. It is excluded
        # from the string-expression pin below (a call expression is not a
        # constant) and is pinned instead to the evaluated object afterwards,
        # so every declaration form freezes the bound object the same way.
        is_new_binding = (
            isinstance(expr, str)
            and re.match(r'^\s*new\s+[A-Za-z_]', expr, re.I) is not None
        )
        # A WITH clause still has to reach the object even when an earlier
        # eager INIT pass already produced a value, otherwise the per-instance
        # field constraints are silently dropped.
        has_with = bool(constraints.get('with') or constraints.get('with_specs'))
        # Declarations with '=' behave like an equality binding, not a mutable
        # assignment. Keep this limited to string expressions to avoid changing
        # literal-list behavior parsed into Python lists by the parser.
        # Subprocess/function calls are not pinned: the result is a computed
        # value that may be read through (e.g. myword.grid).
        if (
            expr is not None
            and isinstance(expr, str)
            and expr.strip()
            and 'constant' not in constraints
            and 'init' not in constraints
            and not is_new_binding
            and not self._is_call_expression(expr)
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
        if defining_scope and self._is_outer_defining_scope(defining_scope):
            # Subprocesses and functions reference the caller's scope chain
            # live, so declarations shadow caller variables locally instead of
            # writing through to the caller's scope.
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
                # Re-declaring a variable with a different dim is a conflict,
                # not a silent re-shape. Store a sticky #DIM so every later
                # read of the variable reports it.
                old_dim = defining_scope.constraints[old_constraints_key].get(
                    'dim')
                new_dim = constraints.get('dim')
                if old_dim is not None and new_dim is not None and old_dim != new_dim:
                    err = error_value(DIM_ERROR)
                    defining_scope.update(var, err, line_number)
                    if scope_dict is not None:
                        scope_dict[var] = err
                    return 'bound'
            if old_key:
                old_value = defining_scope.variables[old_key]
            if var in defining_scope.constraints:
                merged = dict(defining_scope.constraints[var])
                merged.update(constraints)
                constraints = merged
            if constraints:
                defining_scope.constraints[var] = constraints
            if (expr is None
                    and not has_with
                    and var in defining_scope.variables
                    and defining_scope.variables[var] is not None):
                return None
        else:
            if shadow_keyword and self.current_scope().is_shadowed(var):
                print(
                    f"Warning: {shadow_keyword} defines '{var}' which shadows a variable in an outer scope at line {line_number}")
            member_root, member_field = self._existing_member_field(var)
            if shadow_keyword and member_root:
                print(
                    f"Warning: {shadow_keyword} defines '{var}' which shadows a member field of '{member_root}' at line {line_number}")
            define_scope.define(
                var, None, type_name, constraints, is_uninitialized=True,
                line_number=line_number)
            if mark_implicit:
                define_scope.mark_implicit_let(var)
            if ((getattr(self.current_scope(), 'is_loop_scope', False)
                    or getattr(self.current_scope(), 'let_block', False))
                    and define_scope is self.current_scope()):
                # A declaration inside a loop body or Let block is scoped
                # to that body and must not be copied back to the parent
                # scope when the iteration or block ends. ``implicit_let``
                # is not consulted by the copy-back, so record it where
                # that copy-back looks.
                define_scope.local_let_declarations.add(var.lower())
            defining_scope = define_scope

        # A declaration like `Let d not null dim {*,*} or = none` has no value
        # expression, but the dim constraint still materializes the array;
        # unset items then read the default (or #N/A without one).
        if expr is None and constraints.get('dim'):
            dim_value = self._create_declared_dim_array(
                var, type_name, constraints, line_number)
            if dim_value is not None:
                search_scope.update(var, dim_value, line_number)
                if scope_dict is not None:
                    scope_dict[var] = dim_value
                return 'bound'

        if expr is None and constraints.get('default') is not None:
            # A declaration like `Let x not null or = None` binds the
            # evaluated default so the value behaves like 0 / False in
            # boolean and equality contexts (None evaluates to the universal
            # zero sentinel, not plain Python None).
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

        if expr is None and constraints.get('init') is not None:
            # 'Let a init new Box' has no '=' expression: materialize the INIT
            # value now, the way the For path does.  INIT is a Push, so the
            # binding stays mutable -- unlike '=' it pins no equality.
            try:
                init_value = self.expr_evaluator.eval_or_eval_array(
                    str(constraints['init']),
                    search_scope.get_evaluation_scope(),
                    line_number, expected_unit=constraints.get('unit'))
                init_value = self.array_handler.check_dimension_constraints(
                    var, init_value, line_number)
                if constraints.get('with') or constraints.get('with_specs'):
                    init_value = self._apply_with_constraints(
                        init_value, constraints.get('with', {}),
                        search_scope.get_full_scope(), line_number,
                        with_specs=constraints.get('with_specs'),
                        type_name=type_name)
                if type_name:
                    init_value, constraints = defining_scope._coerce_custom_type_value(
                        type_name, init_value, constraints, line_number)
                actual_constraint_key = defining_scope._get_case_insensitive_key(
                    var, defining_scope.constraints) or var
                defining_scope.constraints[actual_constraint_key] = constraints
                search_scope.update(var, init_value, line_number)
                if scope_dict is not None:
                    scope_dict[var] = init_value
                return 'bound'
            except NameError:
                return None
            except Exception:
                return None

        if expr is None:
            return None

        # Equality binding to a subprocess result is not allowed; the INIT
        # keyword materializes subprocess results into a variable instead.
        if 'init' not in constraints:
            sub_name = self._leading_subprocess_call(expr)
            if sub_name:
                raise RuntimeError(
                    f"Equality assignment to subprocess '{sub_name}' is not allowed; use INIT instead at line {line_number}")

        # Client-style equality bindings (no INIT) register as listeners so the
        # expression re-evaluates whenever a dependency cell/var is pushed.
        if 'init' not in constraints and expr is not None:
            self._register_listeners(var, expr, search_scope)

        try:
            expected_unit = (constraints or {}).get('unit')
            evaluated_value = self.expr_evaluator.eval_or_eval_array(
                expr, scope_dict or search_scope.get_evaluation_scope(),
                line_number, expected_unit=expected_unit)
            evaluated_value = self.array_handler.check_dimension_constraints(
                var, evaluated_value, line_number)
            if constraints.get('with') or constraints.get('with_specs'):
                evaluated_value = self._apply_with_constraints(
                    evaluated_value,
                    constraints.get('with', {}),
                    search_scope.get_full_scope(),
                    line_number,
                    with_specs=constraints.get('with_specs'),
                    type_name=type_name)
            if type_name:
                evaluated_value, constraints = defining_scope._coerce_custom_type_value(
                    type_name, evaluated_value, constraints, line_number)
                actual_constraint_key = defining_scope._get_case_insensitive_key(
                    var, defining_scope.constraints) or var
                defining_scope.constraints[actual_constraint_key] = constraints
            elif infer_type:
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
            value = self._to_sparse_undimmed(evaluated_value, constraints)
            if isinstance(value, dict) and value.get('_handle'):
                # Engine-owned handle: the client binding (`Let h = cap.fn()`) only
                # minted the handle. It is not an immutable constant — the engine
                # rewrites it as the handle advances (`\.timer`/`\.counter`) — so
                # strip any constant pin that the generic equality-binding logic
                # attached to the member-call expression.
                constraints = dict(constraints)
                constraints.pop('constant', None)
                actual_constraint_key = defining_scope._get_case_insensitive_key(
                    var, defining_scope.constraints) or var
                defining_scope.constraints[actual_constraint_key] = constraints
            if isinstance(constraints.get('constant'), (list, tuple, dict)):
                # Array/object literal constant: replace raw parser tokens
                # (text items arrive quoted) with the evaluated value so
                # constraint checks compare real values.
                constraints['constant'] = value
                actual_constraint_key = defining_scope._get_case_insensitive_key(
                    var, defining_scope.constraints) or var
                defining_scope.constraints[actual_constraint_key] = constraints
            if is_new_binding and isinstance(value, dict) and isinstance(value.get('_type_name'), str):
                # Pin a public-field snapshot of the constructed object, the
                # same snapshot the ':' constructor declaration stores, so
                # bare/Let/For and ':' bindings freeze identically.
                constraints = dict(constraints)
                constraints['constant'] = {
                    k: copy.deepcopy(v) for k, v in value.items()
                    if not str(k).startswith('_')}
                actual_constraint_key = defining_scope._get_case_insensitive_key(
                    var, defining_scope.constraints) or var
                defining_scope.constraints[actual_constraint_key] = constraints
            search_scope.update(var, value, line_number)
            if scope_dict is not None:
                scope_dict[var] = value
            # The binding just evaluated its own expression; dep changes it
            # caused (live outputs streaming into its inputs) are not pending
            # work for this record.
            owner = getattr(self, 'compiler', None) or self
            clear = getattr(owner, '_clear_stale_for', None)
            if clear is not None:
                clear(var)
            return 'bound'
        except NameError as e:
            missing = self.extract_missing_dependencies(e)
            if not missing:
                raise
            expr_lower = (expr or '').lower()
            real_missing = {dep for dep in missing if dep.lower() in expr_lower}
            if not real_missing:
                raise
            missing = real_missing
            if remaining_let_vars:
                future_refs = {
                    dep for dep in missing if dep.lower() in remaining_let_vars}
                if future_refs:
                    raise NameError(
                        f"Forward LET reference to {future_refs} at line {line_number}")
            for dep in missing:
                self.mark_dependency_missing(dep)
            self.pending_assignments[var] = (
                expr, line_number, set(missing), constraints)
            return 'deferred'

    def _process_let_standard_assignment(
            self,
            var,
            type_name,
            constraints,
            expr,
            scope_dict,
            line_number,
            remaining_let_vars):
        status = self._bind_declared_var(
            var,
            type_name,
            constraints,
            expr,
            line_number,
            scope_dict=scope_dict,
            shadow_keyword='LET',
            mark_implicit=True,
            remaining_let_vars=remaining_let_vars,
        )
        if status != 'bound':
            return
        pending_vars = [
            pending_var for pending_var in list(self.pending_assignments.keys())
            if not pending_var.startswith('__line_')
        ]
        for pending_var in pending_vars:
            self._attempt_resolve_pending_var(
                pending_var, line_number)

    def _run_let_second_pass(self, var_list, line_number):
        for var, _, _, expr in var_list:
            if expr is None and self.current_scope().is_uninitialized(var):
                try:
                    var_value = self.current_scope().get(var)
                    if var_value is not None:
                        self.current_scope()._check_constraints(
                            var, var_value, line_number)
                except Exception as e:
                    pass

    def _resolve_pending_let_var(self, var, defining_scope):
        """Resolve a pending assignment for a LET variable, if possible.

        Returns the current value of the variable (or None if unresolved).
        """
        if var in self.pending_assignments and defining_scope.get(var) is None:
            expr, ln, deps, _ = self.pending_assignments[var]
            unresolved = any(
                dep != var and self.has_unresolved_dependency(
                    dep, scope=self.current_scope())
                for dep in deps)
            if not unresolved:
                try:
                    value = self.expr_evaluator.eval_or_eval_array(
                        expr, defining_scope.get_full_scope(), ln)
                    defining_scope.update(var, value, ln)
                    del self.pending_assignments[var]
                except Exception:
                    pass
        return defining_scope.get(var)

    def _execute_let_block(self, lines, i, line_number, var_list,
                           member_writes=None):
        for var, _type_name, _constraints, expr in var_list:
            if expr is None:
                continue
            if not (isinstance(var, str)
                    and re.match(r'^[A-Za-z_]\w*$', var)):
                if self.control_flow._let_header_target_missing(var):
                    raise SyntaxError(
                        f"Name '{var}' is not defined at line {line_number}")
        self.push_scope(is_private=True)
        self.current_scope().let_block = True
        block_lines = []
        depth = 1
        scan_i = i + 1
        while scan_i < len(lines) and depth > 0:
            next_line, next_line_number = lines[scan_i]
            next_line_clean = next_line.strip().lower()
            if next_line_clean == "end":
                depth -= 1
                if depth == 0:
                    break
            else:
                depth += block_nesting_delta(next_line_clean)
            block_lines.append((next_line, next_line_number))
            scan_i += 1
        if depth > 0:
            raise SyntaxError(
                f"Unclosed LET block starting at line {line_number}")
        block_end_i = scan_i
        # Snapshot existing names for rollback.
        snapshots = {}
        for var, _type_name, _constraints, expr in var_list:
            if expr is None or not isinstance(var, str):
                continue
            if not re.match(r'^[A-Za-z_]\w*$', var):
                # A member/index header writes through to its target; it
                # neither reserves a name nor snapshots one.
                continue
            parent_scope = self.current_scope().parent
            parent_def = parent_scope.get_defining_scope(var) \
                if parent_scope is not None else None
            if parent_def is not None:
                snapshots[var] = (
                    parent_def,
                    self.control_flow._snapshot_var_entry(parent_def, var),
                )

        def _rollback_existing():
            for snap_var, (snap_scope, snap) in snapshots.items():
                self.control_flow._restore_var_entry(
                    snap_scope, snap_var, snap)
                self.pending_assignments.pop(snap_var, None)

        def _skip_block():
            # Roll back existing names, declare body decls unvalued,
            # drop block-locals with the scope.  A member header has
            # already written through to its object and stays written.
            _rollback_existing()
            self.control_flow._predeclare_block_assignment_targets(
                block_lines)
            self.pop_scope()
            return block_end_i + 1

        scope_dict = self.current_scope().get_evaluation_scope()
        member_writes = self._process_let_first_pass(
            var_list, scope_dict, line_number)
        self._run_let_second_pass(var_list, line_number)
        # A deferred update failed: roll back and skip.
        for snap_var in snapshots:
            if snap_var in self.pending_assignments:
                return _skip_block()
        # Rebinds of enclosing names stay where they were defined.
        for var, _, _, _ in var_list:
            if var.lower() in member_writes:
                continue
            if self.current_scope().get_defining_scope(var) is self.current_scope():
                self.current_scope().local_let_declarations.add(var.lower())

        for var, _, constraints, _ in var_list:
            if constraints and var.lower() not in member_writes:
                defining_scope = self.current_scope().get_defining_scope(var) or self.current_scope()
                var_value = self._resolve_pending_let_var(var, defining_scope)
                if var_value is None:
                    return _skip_block()
                for op, threshold in constraints.items() if constraints else []:
                    try:
                        threshold_value = float(threshold)
                        if isinstance(var_value, (int, float)):
                            if op == '<' and var_value >= threshold_value:
                                return _skip_block()
                            elif op == '>' and var_value <= threshold_value:
                                return _skip_block()
                    except ValueError:
                        pass
                if defining_scope.constraints.get(var, {}):
                    try:
                        defining_scope._check_constraints(
                            var, var_value, line_number)
                    except ValueError as e:
                        return _skip_block()
        # Skip the block when any declared variable holds an error value
        # (``Let x = <error> then ... End`` must not execute its body).
        for var, _, _, _ in var_list:
            if var.lower() in member_writes:
                # Written into an object rather than declared here.
                continue
            defining_scope = self.current_scope(
            ).get_defining_scope(var) or self.current_scope()
            var_value = self._resolve_pending_let_var(
                var, defining_scope)
            if var_value is None:
                var_value = defining_scope.variables.get(var)
            if is_error_value(var_value):
                return _skip_block()

        # Collect Let-defined variable names so we can exclude them from
        # promotion.  For and : define in the enclosing scope; only Let
        # is truly local to the block.
        let_var_names = set()
        for bl, _ in block_lines:
            bl_stripped = bl.strip()
            if _first_keyword(bl_stripped) == 'let' and 'init' in bl_stripped.lower().split():
                m_let = re.match(r'^let\s+(\w+)', bl_stripped, re.I)
                if m_let:
                    let_var_names.add(m_let.group(1).lower())

        block_pending = self.control_flow._process_block(block_lines)
        if block_pending:
            self.scopes[0].pending_assignments.update(block_pending)

        # Promote For- and :-defined variables to the enclosing scope so
        # they persist after the Let block ends (mirrors type-constructor
        # behaviour).  Let-defined variables stay block-local.
        parent_scope = self.current_scope().parent
        if parent_scope is not None:
            block_scope = self.current_scope()
            for key in list(block_scope.variables.keys()):
                if key.startswith('_'):
                    continue
                if key.lower() in let_var_names:
                    continue
                if key.lower() in block_scope.local_let_declarations:
                    continue
                existing = parent_scope.get_defining_scope(key)
                if existing:
                    try:
                        existing.update(key, block_scope.variables[key], line_number)
                    except Exception:
                        pass
                else:
                    try:
                        parent_scope.define(
                            key, block_scope.variables[key],
                            block_scope.types.get(key, ''),
                            block_scope.constraints.get(key, {}),
                            is_uninitialized=False,
                            line_number=line_number)
                    except Exception:
                        pass

        # A member header writes through to the object it names, so the
        # field stays as the block left it.
        self.pop_scope()
        return block_end_i + 1

    def _execute_let_inline(self, lines, i, line_number, var_list,
                            member_writes=None):
        member_writes = member_writes or set()
        for var, _, constraints, _ in var_list:
            if constraints and var.lower() not in member_writes:
                defining_scope = self.current_scope().get_defining_scope(var) or self.current_scope()
                var_value = self._resolve_pending_let_var(var, defining_scope)
                if var_value is None:
                    # The variable is declared but not valued yet (it will be filled
                    # by a later push). Downstream cell assignments defer through
                    # the pending mechanism and re-run once the value arrives.
                    continue
                for op, threshold in constraints.items() if constraints else []:
                    if op not in ('<', '>', '<=', '>='):
                        continue
                    if isinstance(threshold, (list, dict, tuple)):
                        continue
                    try:
                        threshold_value = float(threshold)
                        if isinstance(var_value, (int, float)):
                            if op == '<' and var_value >= threshold_value:
                                defining_scope.update(
                                    var, error_value(VALUE_ERROR), line_number)
                                continue
                            elif op == '>' and var_value <= threshold_value:
                                defining_scope.update(
                                    var, error_value(VALUE_ERROR), line_number)
                                continue
                    except (TypeError, ValueError):
                        pass
                if defining_scope.constraints.get(var, {}):
                    try:
                        defining_scope._check_constraints(
                            var, var_value, line_number)
                    except ConstraintError as exc:
                        defining_scope.update(
                            var, error_value(exc.code), line_number)
                        continue
        return i + 1

    def _evaluate_generator_values(self, init_expr, line_number):
        """Evaluate INIT/generator expressions used by FOR fallback handling."""
        try:
            values = self._evaluate_push_expression(str(init_expr), line_number)
        except Exception:
            try:
                values = [self.expr_evaluator.eval_or_eval_array(
                    str(init_expr), self.current_scope().get_evaluation_scope(), line_number)]
            except Exception:
                values = []
        if not values:
            call_match = re.match(
                r'^([A-Za-z][A-Za-z0-9_.]*)\s*\((.*)\)$', str(init_expr))
            if call_match:
                call_name, call_args = call_match.groups()
                args_list = self._split_call_arguments(
                    call_args) if call_args else []
                arg_values = [self.expr_evaluator.eval_or_eval_array(
                    a, self.current_scope().get_evaluation_scope(), line_number) for a in args_list] if args_list else []
                try:
                    if hasattr(self, 'subprocesses') and call_name.lower() in getattr(self, 'subprocesses', {}):
                        values = [self.call_subprocess(
                            call_name, arg_values, collect_all=False, line_number=line_number)]
                    elif hasattr(self, 'functions') and call_name.lower() in getattr(self, 'functions', {}):
                        values = [self.call_function(
                            call_name, arg_values,
                            instance_type=self.functions[call_name.lower()].get('member_of'),
                            line_number=line_number)]
                except Exception:
                    values = []
        if not values:
            try:
                values = [self.expr_evaluator.eval_or_eval_array(
                    str(init_expr), self.current_scope().get_evaluation_scope(), line_number)]
            except Exception:
                values = []
        return values

    def _is_bare_subprocess_call(self, expr):
        call_match = re.match(
            r'^\s*([A-Za-z][A-Za-z0-9_.]*)\s*\(.*\)\s*$',
            str(expr),
            re.I,
        )
        if not call_match:
            return False
        name = call_match.group(1).lower()
        subprocess_defs = getattr(self, 'subprocesses', {}) or {}
        if name in subprocess_defs:
            return True
        compiler = getattr(self, 'compiler', None)
        if compiler is not None:
            compiler_subs = getattr(compiler, 'subprocesses', {}) or {}
            return name in compiler_subs
        return False

    def _apply_init_values(
            self,
            var_name,
            type_name,
            constraints,
            values,
            line_number,
            scope=None):
        """Apply INIT values with the same update/trigger path used by PUSH."""
        if values is None:
            return False
        if not isinstance(values, list):
            values = [values]
        if not values:
            return False

        target_scope = scope or self.current_scope()
        defining_scope = target_scope.get_defining_scope(var_name) or target_scope
        actual_key = defining_scope._get_case_insensitive_key(
            var_name, defining_scope.variables) or var_name
        committed = False

        for raw_value in values:
            try:
                value = copy.deepcopy(raw_value)
            except Exception:
                value = raw_value
            value = self._strip_init_copy_immutability(value)
            value = self._to_sparse_undimmed(value, constraints)
            if isinstance(value, dict) and (
                    constraints.get('with') or constraints.get('with_specs')):
                # WITH carries per-instance field constraints.  Applying it
                # here keeps every INIT path (eager pass, FOR, LET, ':') on the
                # same footing: without it the object is stored bare and member
                # pushes never validate against the clause.
                try:
                    value = self._apply_with_constraints(
                        value,
                        constraints.get('with', {}),
                        target_scope.get_full_scope(),
                        line_number,
                        with_specs=constraints.get('with_specs'),
                        type_name=type_name)
                except Exception:
                    pass
            owner = getattr(self, 'compiler', None) or self
            if actual_key in defining_scope.variables:
                if owner._set_by.get(('var', actual_key.lower())) == 'client':
                    committed = True
                    continue
                defining_scope.update(actual_key, value, line_number)
            else:
                defining_scope.define(
                    var_name,
                    value,
                    type_name,
                    constraints,
                    is_uninitialized=False,
                    line_number=line_number,
                )
                actual_key = defining_scope._get_case_insensitive_key(
                    var_name, defining_scope.variables) or var_name
            owner._set_by[('var', actual_key.lower())] = 'publisher'
            self._enqueue_push(actual_key, value)
            committed = True

        if committed:
            self._process_when_triggers()
        return committed

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

    def _handle_main_loop_for_fallback_statement(self, lines, i, line, line_number, stripped, stripped_lower):
        m = re.match(r'^\s*FOR\s+(.+?)(?:\s+DO\s*$|\s*$)',
                     stripped, re.I)
        if not m:
            raise SyntaxError(
                f"Invalid FOR syntax at line {line_number}")
        var_def = m.group(1).strip()
        has_block = stripped_lower.endswith('do')
        index_fallback = None
        index_match_line = re.search(
            r'\bindex\s+([A-Za-z][A-Za-z0-9_.]*)\b', var_def, re.I)
        if index_match_line:
            index_fallback = index_match_line.group(1)
        var_list = []
        if ' and ' in var_def.lower():
            var_parts = split_var_defs(var_def)
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
            if defining_scope and self._is_outer_defining_scope(defining_scope):
                # Subprocesses/functions shadow caller variables locally.
                defining_scope = None
            if defining_scope:
                # Rule 2: a ``For ... do`` header over any live variable is
                # a redefinition, whichever nature declared it first.
                if has_block:
                    raise SyntaxError(
                        f"Variable '{var}' already defined in scope at line "
                        f"{line_number}")
                if constraints:
                    defining_scope.constraints[var] = constraints
                if expr is None and var in defining_scope.variables and defining_scope.variables[var] is not None:
                    continue
            else:
                if self.current_scope().is_shadowed(var):
                    print(
                        f"Warning: FOR defines '{var}' which shadows a variable in an outer scope at line {line_number}")
                self.current_scope().define(var, None, type_name, constraints, is_uninitialized=True)

        # If there is a DO block and INIT references a generator, run the block per pushed value
        if has_block:
            init_entries = [(var, type_name, constraints) for var, type_name,
                            constraints, expr in var_list if constraints and 'init' in constraints]
            if init_entries:
                # Collect block lines until matching END
                block_lines = []
                depth = 1
                while i < len(lines) and depth > 0:
                    next_line, next_ln = lines[i]
                    clean = next_line.strip().lower()
                    if clean == 'end':
                        depth -= 1
                        if depth == 0:
                            break
                    else:
                        depth += block_nesting_delta(clean)
                    block_lines.append((next_line, next_ln))
                    i += 1
                # Execute block for each value produced by INIT expression of the first entry
                raw_init_expr = init_entries[0][2].get('init')
                index_name = init_entries[0][2].get(
                    'index') or index_fallback
                # Fallback: allow "expr index var" inside init string if parser missed the keyword
                init_match = None
                try:
                    init_match = re.match(
                        r'^(.*)\s+index\s+([A-Za-z][A-Za-z0-9_.]*)\s*$', str(raw_init_expr), re.I)
                except Exception:
                    init_match = None
                if init_match and not index_name:
                    raw_init_expr = init_match.group(1)
                    index_name = init_match.group(2)
                init_expr = raw_init_expr
                values = self._evaluate_generator_values(init_expr, line_number)
                for idx_val, val in enumerate(values, start=1):
                    self.push_scope(is_private=True,
                                    is_loop_scope=True)
                    loop_scope = self.current_scope()
                    if index_name:
                        defining = loop_scope.get_defining_scope(
                            index_name)
                        if defining:
                            defining.update(
                                index_name, idx_val, line_number)
                        else:
                            loop_scope.define(index_name, idx_val, 'number', {},
                                              is_uninitialized=False)
                    fallback_name = index_name or index_fallback
                    if fallback_name:
                        loop_scope.variables[fallback_name] = idx_val
                    for var, type_name, constraints in init_entries:
                        if var in loop_scope.variables:
                            loop_scope.update(var, val, line_number)
                        else:
                            loop_scope.define(
                                var, val, type_name, constraints, is_uninitialized=False)
                        idx_var = (constraints or {}).get(
                            'index') or index_fallback
                        if idx_var:
                            if idx_var in loop_scope.variables:
                                loop_scope.update(
                                    idx_var, idx_val, line_number)
                            else:
                                loop_scope.define(
                                    idx_var, idx_val, 'number', {}, is_uninitialized=False)
                    if index_fallback and index_fallback not in loop_scope.variables:
                        # Ensure index fallback is present even if constraints missed it
                        loop_scope.define(
                            index_fallback, idx_val, 'number', {}, is_uninitialized=False)
                    if index_name and index_name not in loop_scope.variables:
                        loop_scope.define(index_name, idx_val, 'number',
                                          {}, is_uninitialized=False)
                    self.control_flow._process_block(
                        block_lines, loop_scope)
                    self.pop_scope()
                return i + 1

        # If there is no DO block, handle INIT generator semantics immediately
        if not has_block:
            init_handled = False
            for var, type_name, constraints, expr in var_list:
                if not constraints or 'init' not in constraints:
                    continue
                init_expr = constraints.get('init')
                values = self._evaluate_generator_values(init_expr, line_number)
                target_scope = self.current_scope().get_defining_scope(
                    var) or self.current_scope()
                if values:
                    val = values[-1]
                    owner = getattr(self, 'compiler', None) or self
                    if owner._set_by.get(('var', var.lower())) == 'client':
                        init_handled = True
                        continue
                    try:
                        target_scope.update(var, val, line_number)
                    except NameError:
                        target_scope.define(
                            var, val, type_name, constraints, is_uninitialized=False)
                    owner._set_by[('var', var.lower())] = 'publisher'
                    self._enqueue_push(var, val)
                    self._process_when_triggers()
                init_handled = True
            if init_handled:
                return i + 1

        # Handle WITH clause and array initialization
        if var_list[0][1] and 'with' in var_list[0][2]:
            var, type_name, constraints, value = var_list[0]
            if type_name.lower() in self.types_defined:
                with_constraints = constraints.get('with', {})
                dims = constraints.get('dim', [])
                if dims:
                    shape = []
                    for _, size_spec in dims:
                        if isinstance(size_spec, tuple) and size_spec[1] is not None:
                            start, end = size_spec
                            size = end - start + 1
                        else:
                            size = size_spec
                        shape.append(size)
                    public_fields = public_type_fields(
                        self.types_defined[type_name.lower()])
                    if any(size is None for size in shape):
                        self.current_scope().define(
                            var, {}, type_name.lower(), constraints,
                            is_uninitialized=False, line_number=line_number)
                        return i + 1
                    array = self.array_handler.create_array(
                        shape, None, 'number', line_number)
                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            for k in range(shape[2]):
                                obj = {f: (self.compiler.resolve_with_value(
                                    with_constraints.get(f, None), line_number)
                                    if with_constraints.get(f, None) is not None else None)
                                    for f in public_fields}
                                obj['value'] = float(
                                    value) if value else 1.0
                                array = self.array_handler.set_array_element(
                                    array, [i, j, k], obj, line_number)
                    self.current_scope().define(var, array, 'array', constraints, is_uninitialized=False)
                    return i + 1

        return i + 1


    def run(self, code, args=None, suppress_output=False, return_output=False):
        lines, label_lines, dim_lines = self._run_setup(code, args)
        guard_entries = self._run_prepare_execution(
            lines, label_lines, dim_lines, args)
        if guard_entries is None:
            return self._build_output_dict()
        if getattr(self, 'halt_before_main_loop', False):
            # Host requested stop after preparation (e.g. --list-required):
            # requirements are collected and parameters evaluated, nothing runs.
            return self._build_output_dict()
        guard_conditions = {entry['line_number']: entry['condition']
                            for entry in guard_entries}
        self._run_main_loop(lines, guard_conditions)
        # Safety flush: if a tracked binding line was never reached, its
        # deferred effects would otherwise be stranded.
        self._flush_deferred_wave_effects()
        if guard_entries and not self._global_guards_pre_evaluated:
            guard_passed = self._evaluate_guard_conditions(guard_entries)
            if not guard_passed:
                self.global_guard_allows_execution = False
                self.grid.clear()
                self.output_values.clear()
                return self._build_output_dict()

        self._resolve_pending_assignments()

        # Process any remaining deferred assignments
        self._process_deferred_assignments()

        # End-of-program wave: output collection reads variables and grid
        # cells directly, so flush every pending recompute first.
        owner = getattr(self, 'compiler', None) or self
        settle = getattr(owner, '_settle_stale', None)
        if settle is not None:
            settle()

        # A deferred IF whose dependencies never resolved is still a failure.
        # It is only reported here, after the retry pass had its chance.
        still_deferred = [key for key in (self.pending_assignments or {})
                          if key.startswith('__if_line_')]
        if still_deferred:
            raise RuntimeError(f"Unresolved assignments: {still_deferred}")

        # Print all output variables
        if not suppress_output:
            self._print_outputs()

        if return_output:
            return self.output_values
        return self._build_output_dict()

    def _run_main_loop(self, lines, guard_conditions):
        """Thin orchestrator for main execution loop."""
        return self._run_main_loop_impl(lines, guard_conditions)

    def _handle_for_array_and_dim_declarations(self, line, line_number, i):
        # Handle special case: for element = haystack(mid) (array element assignment)
        m = re.match(
            r'^\s*for\s+([\w_]+)\s*=\s*([\w_]+)\(([^)]+)\)\s*$', line, re.I)
        if m:
            var_name, array_name, index_expr = m.groups()
            # Don't handle if array_name is a known builtin or user function (e.g., For h = Power(...))
            try:
                from builtin_functions import BUILTINS
                if array_name.lower() in BUILTINS:
                    return False, i
            except ImportError:
                pass
            if array_name.lower() in getattr(self.compiler, 'functions', {}):
                return False, i

            try:
                # Evaluate the index expression
                index_value = self.expr_evaluator.eval_expr(
                    index_expr, self.current_scope().get_evaluation_scope(), line_number)

                # Get the array
                array = self.current_scope().get(array_name)
                if array is None:
                    raise ValueError(
                        f"Array '{array_name}' is not defined at line {line_number}")

                # Convert to list if it's a set
                if isinstance(array, set):
                    array = list(array)

                # Access the array element (1-based indexing)
                if isinstance(index_value, (int, float)):
                    idx = int(index_value) - 1  # Convert to 0-based
                    if 0 <= idx < len(array):
                        element_value = array[idx]

                        # Define the variable with the element value
                        self.current_scope().define(var_name, element_value,
                                                    'number', {}, is_uninitialized=False)
                    else:
                        raise IndexError(
                            f"Index {index_value} out of range for array '{array_name}' at line {line_number}")
                else:
                    raise ValueError(
                        f"Invalid index type {type(index_value)} for array access at line {line_number}")

            except Exception as e:
                raise ValueError(
                    f"Failed to process array element assignment at line {line_number}: {e}")

            return True, i + 1

        # Handle For var as type dim {dimensions} syntax
        m = re.match(
            r'^\s*for\s+([\w_]+)\s+as\s+(\w+)\s+(?:of\s+\w+\s+)?(?:not\s+null\s+)?dim\s*(\{[^}]*\})', line, re.I)
        if m:
            # Declarations carrying an initializer ('init'/'=') or any
            # unbounded ('*'/named/empty) dimension are handled by the
            # general For path, which materializes sparse arrays and
            # applies the INIT values.
            if re.search(r'\binit\b', line, re.I):
                return False, i
            # 'or = <expr>' is a default constraint, not an initializer.
            # Strip it before checking for standalone '='.
            line_no_or_eq = re.sub(r'\bor\s*=\s*.+', '', line, flags=re.I).strip()
            if '=' in line_no_or_eq:
                return False, i
            var_name, type_name, dim_str = m.groups()
            is_custom_type = type_name.lower() in getattr(self, 'types_defined', {})

            # Parse the dimension string
            dim_str = dim_str.strip()
            if dim_str.startswith('{') and dim_str.endswith('}'):
                dim_content = dim_str[1:-1].strip()
                # Parse dimensions like {30, 2}
                dim_parts = [part.strip()
                             for part in dim_content.split(',')]
                shape = []
                bounded = True
                for part in dim_parts:
                    try:
                        shape.append(int(part))
                    except ValueError:
                        # Named dimensions, '*' or empty parts denote
                        # unbounded dimensions handled by the general path.
                        bounded = False
                        shape.append(part)
                if not bounded:
                    return False, i

                # Create the array with the specified shape
                if is_custom_type:
                    array_data = self.array_handler.create_object_array(
                        shape, None, line_number)
                else:
                    array_data = self.array_handler.create_array(
                        shape, None, type_name.lower() if type_name.lower() in ('number', 'text', 'logical') else 'number', line_number, template=True)

                # Define the variable with the created array
                constraints = {'dim': dim_str}
                or_eq_match = re.search(r'\bor\s*=\s*(.+?)\s*$', line, re.I)
                if or_eq_match:
                    constraints['default'] = or_eq_match.group(1).strip()
                self.current_scope().define(var_name, array_data,
                                            type_name.lower(), constraints, False)
            else:
                constraints = {}
                self.current_scope().define(var_name, None, type_name.lower(), constraints, True)
            return True, i + 1

        # Handle For var as type dim number syntax
        m = re.match(
            r'^\s*for\s+([\w_]+)\s+as\s+(\w+)\s+(?:of\s+\w+\s+)?(?:not\s+null\s+)?dim\s+(\d+)', line, re.I)
        if m:
            var_name, type_name, dim_size = m.groups()

            # Create an array with the specified size
            dim_size = int(dim_size)
            if type_name.lower() in getattr(self, 'types_defined', {}):
                initial_array = [
                    self.compiler._instantiate_type(
                        type_name.lower(), [], line_number, allow_default_if_empty=True)
                    for _ in range(dim_size)
                ]
            else:
                # A blank cell reads as #N/A, not zero.
                initial_array = [None] * dim_size

            # Define the variable with the initialized array
            self.current_scope().define(var_name, initial_array, type_name.lower(),
                                        {'dim': [(None, dim_size)]}, False)
            return True, i + 1

        return False, i

    def _handle_for_simple_declarations(self, var_defs, line_number, i):
        if ' in ' in var_defs.lower() or '=' in var_defs or re.search(r'\binit\b', var_defs, re.I):
            return False, i

        var_list = []
        if ' and ' in var_defs.lower():
            var_parts = split_var_defs(var_defs)
            for var_part in var_parts:
                var, type_name, constraints, expr = self._parse_variable_def(
                    var_part, line_number)
                var_list.append((var, type_name, constraints, expr))
        else:
            var, type_name, constraints, expr = self._parse_variable_def(
                var_defs, line_number)
            if constraints and 'var_list' in constraints:
                for name in constraints['var_list']:
                    per_constraints = constraints.copy()
                    per_constraints.pop('var_list', None)
                    var_list.append((name, type_name, per_constraints, expr))
            else:
                var_list.append((var, type_name, constraints, expr))

        for var, type_name, constraints, expr in var_list:
            defining_scope = self.current_scope().get_defining_scope(var)
            if defining_scope and self._is_outer_defining_scope(defining_scope):
                # Subprocesses/functions shadow caller variables locally.
                defining_scope = None
            if defining_scope and not defining_scope.is_implicit_let(var):
                raise ValueError(
                    f"Variable '{var}' already defined in scope at line {line_number}")
            if not defining_scope:
                if self.current_scope().is_shadowed(var):
                    print(
                        f"Warning: FOR defines '{var}' which shadows a variable in an outer scope at line {line_number}")
                self.current_scope().define(
                    var, None, type_name, constraints, is_uninitialized=True)

        return True, i + 1

    def _handle_for_single_line_executable(self, line, line_number, i):
        if ' do ' not in line.lower():
            return True, i + 1

        parts = line.split(' do ', 1)
        if len(parts) != 2:
            return True, i + 1

        for_part = parts[0].strip()
        executable_part = parts[1].strip()

        for_match = re.match(
            r'^\s*for\s+(.+?)(?:\s+do\s*$|\s*$)', for_part, re.I)
        if not for_match:
            return True, i + 1

        var_defs = for_match.group(1).strip()

        range_match = re.match(
            r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', var_defs, re.I)
        if not range_match:
            return True, i + 1

        var_name, range_expr, step_str, index_var = range_match.groups()
        step = int(step_str) if step_str else 1

        if ' to ' not in range_expr:
            return True, i + 1

        range_parts = range_expr.split(' to ')
        start = int(range_parts[0].strip())
        end = int(range_parts[1].strip())
        if step < 0:
            values = list(range(start, end - 1, step))
        else:
            values = list(range(start, end + 1, step))

        if not self._has_push_action(executable_part):
            return True, i + 1

        for idx, value in self._iter_for_values_skipping_errors(values):
            self.push_scope(is_private=True, is_loop_scope=True)
            self.current_scope().define(var_name, value, 'number')
            if index_var:
                self.current_scope().define(index_var, idx + 1, 'number')

            return_match = self._match_return_statement(executable_part)
            push_match = self._match_push_assignment(executable_part)
            if return_match:
                value_expr = return_match.group(1).strip()
                output_values = self._evaluate_push_expression(
                    value_expr, line_number)
                for output_value in output_values:
                    self.output_values.setdefault('output', []).append(
                        self._snapshot_value(output_value))
                if 'output' not in self.output_variables:
                    self.output_variables.append('output')
            elif push_match:
                target, value_expr = self._unpack_push_assignment(push_match)
                self._handle_push_assignment(target, value_expr, line_number)
            else:
                raise SyntaxError(
                    f"Unsupported push syntax in For loop at line {line_number}. Use 'Push var = value' syntax instead.")
            self.pop_scope()

        return True, i + 1

    def _execute_for_assignment_combinations(
            self, value_combinations, var_specs, assignment_line, assignment_line_number):
        """Apply a ':=' assignment line over every loop value combination.

        Each combo is run in its own private loop scope, defining each loop
        variable and optional index variable before evaluating the line.
        """
        for combo in value_combinations:
            if self._combo_has_error(combo):
                continue
            self.push_scope(is_private=True, is_loop_scope=True)
            for loop_idx, (var_name, index_var) in enumerate(var_specs):
                self.current_scope().define(
                    var_name, combo[loop_idx], 'number')
                if index_var:
                    if index_var in self.current_scope().variables:
                        self.current_scope().update(
                            index_var, loop_idx + 1)
                    else:
                        self.current_scope().define(
                            index_var, loop_idx + 1, 'number')
            self.array_handler.evaluate_line_with_assignment(
                assignment_line, assignment_line_number, self.current_scope().get_evaluation_scope())
            self.pop_scope()

    def _handle_for_and_syntax(self, lines, i, var_defs, is_block, line_number):
        if ' and ' not in var_defs.lower() or ' in ' not in var_defs.lower():
            return False, i

        and_parts = re.split(r'\s+and\s+', var_defs, flags=re.I)
        and_loops = []

        for part in and_parts:
            part = part.strip()
            range_end = part.find(' step ')
            if range_end == -1:
                range_end = part.find(' index ')
            if range_end == -1:
                range_end = part.find(' Index ')
            if range_end == -1:
                range_end = len(part)

            range_expr = part[part.find(' in ') + 4:range_end].strip()
            var_name = part[:part.find(' in ')].strip()

            step_str = None
            index_var = None
            remaining = part[range_end:].strip()

            if ' index ' in range_expr:
                range_expr = range_expr[:range_expr.find(' index ')].strip()
            elif ' Index ' in range_expr:
                range_expr = range_expr[:range_expr.find(' Index ')].strip()


            step_match = re.search(r'step\s+(\d+)', remaining, re.I)
            if step_match:
                step_str = step_match.group(1)
                remaining = remaining[:step_match.start()] + remaining[step_match.end():]

            index_match = re.search(r'index\s+([\w_]+)', remaining, re.I)
            if index_match:
                index_var = index_match.group(1)

            step = int(step_str) if step_str else 1
            start_expr = None
            end_expr = None
            if ' to ' in range_expr:
                range_parts = range_expr.split(' to ')
                start_expr = range_parts[0].strip()
                end_expr = range_parts[1].strip()
                is_dynamic = False
                if not start_expr.replace('.', '').replace('-', '').isdigit() or not end_expr.replace('.', '').replace('-', '').replace('+', '').replace('*', '').replace('/', '').replace('(', '').replace(')', '').isdigit():
                    is_dynamic = True
                    values = None
                else:
                    start = int(start_expr)
                    end = int(end_expr)
                    values = list(range(start, end + 1, step))
            elif range_expr.startswith('{') and range_expr.endswith('}'):
                values_str = range_expr[1:-1].split(',')
                values = []
                for v in values_str:
                    v_clean = v.strip()
                    if v_clean.startswith('"') and v_clean.endswith('"'):
                        values.append(v_clean[1:-1])
                    elif v_clean.startswith("'") and v_clean.endswith("'"):
                        values.append(v_clean[1:-1])
                    else:
                        try:
                            values.append(float(v_clean))
                        except ValueError:
                            values.append(v_clean)
            else:
                try:
                    values = self._evaluate_for_range_values(
                        range_expr, step, line_number)
                except SyntaxError:
                    raise
                except Exception:
                    raise SyntaxError(
                        f"Invalid range expression: {range_expr} at line {line_number}")

            if ' to ' not in range_expr:
                is_dynamic = False

            and_loops.append({
                'var_name': var_name,
                'values': values,
                'index_var': index_var,
                'step': step,
                'is_dynamic': is_dynamic,
                'start_expr': start_expr if is_dynamic else None,
                'end_expr': end_expr if is_dynamic else None
            })

        if len(and_loops) < 2:
            return False, i


        def generate_combinations(loop_idx=0, current_combo=[]):
            if loop_idx >= len(and_loops):
                yield current_combo
                return
            loop = and_loops[loop_idx]
            if loop['is_dynamic']:
                self.push_scope(is_private=True, is_loop_scope=True)
                for prev_idx, prev_loop in enumerate(and_loops[:loop_idx]):
                    self.current_scope().define(
                        prev_loop['var_name'], current_combo[prev_idx], 'number')
                start = int(self.expr_evaluator.eval_expr(
                    loop['start_expr'], self.current_scope().get_evaluation_scope(), line_number))
                end = int(self.expr_evaluator.eval_expr(
                    loop['end_expr'], self.current_scope().get_evaluation_scope(), line_number))
                dynamic_values = list(range(start, end + 1, loop['step']))
                self.pop_scope()
                for val in dynamic_values:
                    yield from generate_combinations(loop_idx + 1, current_combo + [val])
            else:
                for val in loop['values']:
                    yield from generate_combinations(loop_idx + 1, current_combo + [val])

        if is_block:
            for_block_lines = []
            i += 1
            depth = 1
            while i < len(lines) and depth > 0:
                next_line, next_line_number = lines[i]
                next_line_clean = next_line.strip().lower()
                if self._is_keyword(next_line_clean, "end"):
                    depth -= 1
                    if depth == 0:
                        break
                else:
                    depth += block_nesting_delta(next_line_clean)
                for_block_lines.append((next_line, next_line_number))
                i += 1

            value_combinations = list(generate_combinations())
            for combo in value_combinations:
                if self._combo_has_error(combo):
                    continue
                self.push_scope(is_private=True, is_loop_scope=True)
                for loop_idx, loop in enumerate(and_loops):
                    self.current_scope().define(
                        loop['var_name'], combo[loop_idx], 'number')
                    if loop['index_var']:
                        loop_values = loop['values']
                        current_value = combo[loop_idx]
                        value_index = loop_values.index(current_value) + 1
                        if loop['index_var'] in self.current_scope().variables:
                            self.current_scope().update(
                                loop['index_var'], value_index)
                        else:
                            self.current_scope().define(
                                loop['index_var'], value_index, 'number')
                self.control_flow._process_block(for_block_lines)
                if self.exit_loop:
                    self.exit_loop = False
                    self.pop_scope()
                    break
                self.pop_scope()
            return True, i

        if i + 1 < len(lines):
            next_line, next_line_number = lines[i + 1]
            if ':=' in next_line:
                value_combinations = list(generate_combinations())
                self._execute_for_assignment_combinations(
                    value_combinations,
                    [(loop['var_name'], loop['index_var']) for loop in and_loops],
                    next_line, next_line_number)
                return True, i + 2

        return False, i

    def _handle_consecutive_nested_for_shortcuts(self, lines, i):
        nested_for_loops = []
        current_i = i
        while current_i < len(lines) and lines[current_i][0].strip().lower().startswith("for "):
            current_line, current_line_number = lines[current_i]
            m = re.match(r'^\s*FOR\s+(.+?)(?:\s+do\s*$|\s*$)', current_line, re.I)
            if not m:
                break
            var_defs = m.group(1).strip()
            is_block = current_line.strip().lower().endswith('do')

            if ' and ' in var_defs.lower():
                break

            range_match = re.match(
                r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', var_defs, re.I)
            if not range_match:
                break
            var_name, range_expr, step_str, index_var = range_match.groups()
            step = int(step_str) if step_str else 1

            if ' to ' in range_expr:
                clean_range_expr = range_expr.split(' step ')[0] if ' step ' in range_expr else range_expr
                range_parts = clean_range_expr.split(' to ')
                try:
                    start_expr = range_parts[0].strip()
                    end_expr = range_parts[1].strip()
                    start_value = self.expr_evaluator.eval_expr(
                        start_expr, self.current_scope().get_evaluation_scope(), current_line_number)
                    end_value = self.expr_evaluator.eval_expr(
                        end_expr, self.current_scope().get_evaluation_scope(), current_line_number)
                    start = int(start_value)
                    end = int(end_value)
                    values = list(range(start, end + 1, step))
                except Exception as e:
                    start = int(range_parts[0].strip())
                    end = int(range_parts[1].strip())
                    values = list(range(start, end + 1, step))
            elif range_expr.startswith('{') and range_expr.endswith('}'):
                values = []
                for v in range_expr[1:-1].split(','):
                    v_clean = v.strip()
                    if v_clean.startswith('"') and v_clean.endswith('"'):
                        values.append(v_clean[1:-1])
                    elif v_clean.startswith("'") and v_clean.endswith("'"):
                        values.append(v_clean[1:-1])
                    else:
                        try:
                            values.append(float(v_clean))
                        except ValueError:
                            values.append(v_clean)
            else:
                break

            nested_for_loops.append({
                'var_name': var_name,
                'values': values,
                'index_var': index_var,
                'is_block': is_block,
                'line_number': current_line_number
            })
            current_i += 1

        if len(nested_for_loops) == 2 and not any(loop['is_block'] for loop in nested_for_loops):
            assignment_line = None
            assignment_line_number = None
            if current_i < len(lines):
                next_line, next_line_number = lines[current_i]
                if ':=' in next_line:
                    assignment_line = next_line
                    assignment_line_number = next_line_number
                    current_i += 1
            if assignment_line:
                vals1 = nested_for_loops[0]['values']
                vals2 = nested_for_loops[1]['values']
                var1 = nested_for_loops[0]['var_name']
                var2 = nested_for_loops[1]['var_name']
                for a, b in self._iterate_for_zip(vals1, vals2):
                    self.push_scope(is_private=True, is_loop_scope=True)
                    self.current_scope().define(var1, a, 'number')
                    self.current_scope().define(var2, b, 'number')
                    self.array_handler.evaluate_line_with_assignment(
                        assignment_line, assignment_line_number, self.current_scope().get_evaluation_scope())
                    self.pop_scope()
                return True, current_i

        if len(nested_for_loops) > 1:
            assignment_line = None
            assignment_line_number = None
            if current_i < len(lines):
                next_line, next_line_number = lines[current_i]
                if ':=' in next_line:
                    assignment_line = next_line
                    assignment_line_number = next_line_number
                    current_i += 1
            if assignment_line:
                from itertools import product
                value_combinations = list(product(*[loop['values'] for loop in nested_for_loops]))
                self._execute_for_assignment_combinations(
                    value_combinations,
                    [(loop['var_name'], loop['index_var']) for loop in nested_for_loops],
                    assignment_line, assignment_line_number)
                return True, current_i

        return False, i

    def _run_main_loop_impl(self, lines, guard_conditions):
        return self._run_main_loop_impl_body(lines, guard_conditions)

    def _cell_spill_dependencies(self, rhs):
        """Variables a ``[cell] := rhs`` mirror depends on.

        Every free variable in the expression is a dependency, so a mirror
        like ``[C6] := a * b`` is recomputed when either ``a`` or ``b`` is
        published to. The spill root (the base of a dotted, called or indexed
        reference) is included as well, so ``[C1] := grid{a}`` still reacts
        to ``grid``. Function, subprocess and type names are not variables.
        """
        deps = set(self._extract_dependencies_from_expression(rhs) or ())
        root = _extract_spill_root_var(rhs)
        if root:
            deps.add(root)
        functions = self.functions or {}
        subprocesses = self.subprocesses or {}
        types = getattr(self, 'types_defined', None) or {}
        return {dep for dep in deps
                if dep and dep.lower() not in functions
                and dep.lower() not in subprocesses
                and dep.lower() not in types}

    def _pre_register_cell_spill_constraints(self, lines):
        """Register ``[cell] := expr`` spill constraints before the main loop
        runs, so that Push statements notify them regardless of statement
        order. Re-evaluation of a pre-registered line only takes effect once
        the target cell has been materialized (its write has executed).

        Lines inside a block body are skipped: they register when the block
        executes, so a body whose guard fails installs no listener at all."""
        block_body_lines = getattr(
            self.control_flow, 'block_body_lines', None) or set()
        for line, line_number in lines:
            if ':=' not in line:
                continue
            if line_number in block_body_lines:
                continue
            target, rhs = map(str.strip, line.split(':=', 1))
            if not re.match(r'^\[\^?[A-Za-z{]', target) or not target.endswith(']'):
                continue
            deps = self._cell_spill_dependencies(rhs)
            if not deps:
                continue
            self._register_cell_spill_listener(
                line, deps, line_number, None)

    def _run_main_loop_impl_body(self, lines, guard_conditions):
        self._pre_register_cell_spill_constraints(lines)
        i = 0
        while i < len(lines):
            self._loop_iteration += 1
            self._maybe_fire_tickers()
            self._pump_server_requests()
            # Live inputs are lazily re-evaluated in the caller's scope before
            # each statement the callee runs, so mid-run reads observe the
            # caller's current state (feedback loop).
            refresh = getattr(self, '_refresh_live_inputs', None)
            if refresh is not None:
                refresh()
            prep = self._prepare_main_loop_line(lines, i)
            if prep['action'] == 'break':
                break
            if prep['action'] == 'continue':
                i = prep['next_i']
                continue
            line = prep['line']
            line_number = prep['line_number']
            stripped = prep['stripped']
            stripped_lower = prep['stripped_lower']

            wave_next = self._try_execute_wave_machinery(lines, i)
            if wave_next is not None:
                i = wave_next
                continue

            if stripped_lower.startswith("when ") and stripped_lower.endswith("do"):
                handled, i = self._handle_when_block(
                    lines, i, line, line_number)
                if handled:
                    continue

            handled, next_i = self._handle_main_loop_quick_statements(
                lines, i, line, line_number, stripped, stripped_lower)
            if handled:
                i = next_i
                continue
            if line.lower().strip().startswith("for "):
                precheck_result = self._handle_main_loop_for_prechecks(
                    lines, i, line, line_number)
                guard_condition = precheck_result['guard_condition']
                m = precheck_result['match']
                if precheck_result['handled']:
                    i = precheck_result['next_i']
                    continue

                # Process as single FOR loop (original logic)
                if not m:
                    raise SyntaxError(
                        f"Invalid FOR syntax at line {line_number}")
                if guard_condition is None:
                    guard_match = re.match(
                        r'^\s*for\s+(.+?)\s+do\s+when\s+(.+?)\s+do\s*$',
                        line, re.I)
                    if guard_match:
                        var_defs = guard_match.group(1).strip()
                        guard_condition = guard_match.group(2).strip()
                        is_block = True
                    else:
                        var_defs = m.group(1).strip()
                        is_block = line.strip().lower().endswith('do')
                else:
                    var_defs = m.group(1).strip()
                    is_block = line.strip().lower().endswith('do')
                # Check for FOR loops with ranges or sets
                range_match = re.match(
                    r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', var_defs, re.I)
                if range_match:
                    i = self._handle_main_loop_for_range_case(
                        lines,
                        i,
                        line_number,
                        guard_condition,
                        range_match,
                        is_block,
                    )
                    continue

                elif ' and ' in var_defs.lower() and ' in ' in var_defs.lower():
                    i = self._handle_main_loop_nested_for_case(
                        lines,
                        i,
                        line_number,
                        var_defs,
                        is_block,
                    )
                    continue

                i = self._handle_main_loop_for_declaration_case(
                    lines,
                    i,
                    line_number,
                    var_defs,
                    is_block,
                )
                continue
            else:
                i = self._handle_main_loop_post_for_branches(
                    lines, i, line, line_number, stripped, stripped_lower)
                continue

    def _handle_main_loop_for_declaration_case(
            self,
            lines,
            i,
            line_number,
            var_defs,
            is_block):
        import sys as _sys
        var_list = self._parse_for_declaration_var_list(var_defs, line_number)

        next_i = self._try_handle_for_declaration_special_cases(
            var_list,
            var_defs,
            is_block,
            line_number,
            i,
        )
        if next_i is not None:
            return next_i

        next_i = self._try_handle_for_declaration_single_assignment(
            var_list,
            is_block,
            line_number,
            i,
        )
        if next_i is not None:
            return next_i

        next_i = self._try_handle_for_declaration_multi_assignments(
            var_list,
            is_block,
            line_number,
            i,
        )
        if next_i is not None:
            return next_i


        self._validate_for_declaration_targets(
            var_list,
            is_block,
            lines,
            i,
            line_number,
        )

        self._define_for_declaration_implicit_vars(
            var_list,
            is_block,
            line_number,
        )

        if not is_block:
            self._initialize_for_declaration_non_block(
                var_list,
                line_number,
            )
            return i + 1

        return self._execute_for_declaration_block_scope(
            lines,
            i,
            line_number,
            var_list,
        )

    def _parse_for_declaration_var_list(self, var_defs, line_number):
        var_list = []
        try:
            if ' and ' in var_defs.lower():
                var_parts = split_var_defs(var_defs)
                for var_part in var_parts:
                    var, type_name, constraints, value = self._parse_variable_def(
                        var_part, line_number)
                    var_list.append(
                        (var, type_name, constraints, value))
            else:
                var, type_name, constraints, value = self._parse_variable_def(
                    var_defs, line_number)
                var_list.append((var, type_name, constraints, value))
                if 'var_list' in constraints:
                    var_names = constraints['var_list']
                    var_list.clear()
                    for var_name in var_names:
                        var_list.append(
                            (var_name, type_name, constraints.copy(), value))
        except Exception as e:
            raise SyntaxError(
                f"Invalid FOR variable definition at line {line_number}")
        return var_list

    def _try_handle_for_declaration_special_cases(
            self,
            var_list,
            var_defs,
            is_block,
            line_number,
            i):
        if not is_block and len(var_list) == 1 and var_list[0][3] is None:
            var, type_name, constraints, _ = var_list[0]
            defining_scope = self.current_scope().get_defining_scope(var)
            if defining_scope and defining_scope.is_implicit_let(var):
                if type_name:
                    defining_scope.types[var] = type_name
                if constraints:
                    defining_scope.constraints[var] = constraints
                defining_scope.clear_implicit_let(var)
                return i + 1

        if 'dim' in var_defs.lower():
            m_dim = re.match(
                r'^([\w_]+)\s+dim\s+(\d+)\s+to\s+(\d+)$', var_defs, re.I)
            if m_dim:
                var_name, start, end = m_dim.groups()
                start, end = int(start), int(end)
                constraints = {'dim': [(None, (start, end))]}
                self.current_scope().define(
                    var_name, None, 'array', constraints, is_uninitialized=True)
                self.dimensions[var_name] = [(None, (start, end))]
                return i + 1

        if (var_list[0][1] and 'with' in var_list[0][2]
                and var_list[0][3] is None):
            var, type_name, constraints, value = var_list[0]
            if type_name.lower() in self.types_defined:
                with_constraints = constraints.get('with', {})
                dims = constraints.get('dim', [])
                if dims:
                    shape = [size_spec for _, size_spec in dims]
                    public_fields = public_type_fields(
                        self.types_defined[type_name.lower()])
                    default_struct = {f.lower(): (self.compiler.resolve_with_value(
                        with_constraints.get(f, None), line_number)
                        if with_constraints.get(f, None) is not None else None)
                        for f in public_fields}
                    default_struct['value'] = float(
                        value) if value else 1.0
                    array = self.array_handler.create_array(
                        shape, default_struct, 'number', line_number)
                    self.current_scope().define(
                        var, array, 'array', constraints, is_uninitialized=False)
                    for field in public_fields:
                        if field in with_constraints:
                            self.current_scope().define(
                                f"{var}.{field}",
                                self.compiler.resolve_with_value(
                                    with_constraints[field], line_number),
                                'text')
                return i + 1

        return None

    def _resolve_for_assignment_pending(self, line_number):
        pending = list(self.current_scope().pending_assignments.items())
        for key, (expr, ln, deps) in pending:
            scope_pending = self.current_scope().pending_assignments
            unresolved = any(
                self.has_unresolved_dependency(
                    dep, scope=self.current_scope(), scope_pending=scope_pending)
                for dep in deps)
            if not unresolved:
                try:
                    self.array_handler.evaluate_line_with_assignment(
                        expr, ln, self.current_scope().get_evaluation_scope())
                    del self.current_scope().pending_assignments[key]
                except Exception as e:
                    pass
        pending_vars = [
            pending_var for pending_var in list(self.pending_assignments.keys())
            if not pending_var.startswith('__line_')
        ]
        for pending_var in pending_vars:
            self._attempt_resolve_pending_var(
                pending_var, line_number)

    def _try_handle_for_declaration_single_assignment(
            self,
            var_list,
            is_block,
            line_number,
            i):
        if is_block or len(var_list) != 1 or var_list[0][3] is None:
            return None

        var, type_name, constraints, value = var_list[0]
        status = self._bind_declared_var(
            var,
            type_name,
            constraints,
            value,
            line_number,
            search_scope=self.current_scope(),
            define_scope=self.get_global_scope(),
            infer_type=True,
        )
        if status != 'deferred':
            self._resolve_for_assignment_pending(line_number)
        return i + 1

    def _try_handle_for_declaration_multi_assignments(
            self,
            var_list,
            is_block,
            line_number,
            i):
        if is_block or len(var_list) <= 1:
            return None

        all_have_assignments = all(v[3] is not None for v in var_list)
        if all_have_assignments:
            for var, type_name, constraints, value in var_list:
                self._bind_declared_var(
                    var,
                    type_name,
                    constraints,
                    value,
                    line_number,
                    search_scope=self.current_scope(),
                    define_scope=self.get_global_scope(),
                    infer_type=True,
                )
            self._resolve_for_assignment_pending(line_number)
            return i + 1

        for var, type_name, constraints, _ in var_list:
            if var == 'low':
                default_value = 1
            elif var == 'high':
                try:
                    haystack = self.current_scope().get('haystack')
                    if haystack and hasattr(haystack, '__len__'):
                        default_value = len(haystack)
                    else:
                        default_value = 10
                except Exception:
                    default_value = 10
            else:
                default_value = 0
            self.current_scope().define(
                var,
                default_value,
                type_name,
                constraints,
                is_uninitialized=False,
            )
        return i + 1

    def _validate_for_declaration_targets(
            self,
            var_list,
            is_block,
            lines,
            i,
            line_number):
        for var, _, _, _ in var_list:
            defining_scope = self.current_scope().get_defining_scope(var)
            if defining_scope and self._is_outer_defining_scope(defining_scope):
                defining_scope = None
            if (defining_scope and not defining_scope.is_implicit_let(var) and
                    (not is_block and not (i + 1 < len(lines) and lines[i + 1][0].lower().startswith('let ')))):
                raise ValueError(
                    f"Variable '{var}' already defined in scope at line {line_number}")

    def _define_for_declaration_implicit_vars(
            self,
            var_list,
            is_block,
            line_number):
        if not is_block and len(var_list) > 1 and all(v[3] is None for v in var_list):
            for var, type_name, constraints, _ in var_list:
                self.current_scope().define(
                    var, None, type_name, constraints, is_uninitialized=True)
                self.current_scope().mark_implicit_let(var)

    def _initialize_for_declaration_non_block(self, var_list, line_number):
        init_entries = [
            (v, t, c) for v, t, c, _ in var_list
            if c and 'init' in c
        ]
        if init_entries:
            for var, type_name, constraints in init_entries:
                init_expr = constraints.get('init')
                defining = self.current_scope().get_defining_scope(var)
                if defining and self._is_outer_defining_scope(defining):
                    defining = None
                target_scope = defining or self.current_scope()
                actual_key = target_scope._get_case_insensitive_key(
                    var, target_scope.variables)

                if actual_key:
                    merged_constraints = dict(
                        target_scope.constraints.get(actual_key, {}) or {})
                    merged_constraints.update(constraints or {})
                    target_scope.constraints[actual_key] = merged_constraints
                    if type_name:
                        target_scope.types[actual_key] = type_name
                    target_scope.uninitialized.add(actual_key)
                else:
                    target_scope.define(
                        var,
                        None,
                        type_name,
                        constraints,
                        is_uninitialized=True,
                        line_number=line_number,
                    )
                    actual_key = target_scope._get_case_insensitive_key(
                        var, target_scope.variables) or var

                if self._is_bare_subprocess_call(init_expr):
                    try:
                        init_value = self.expr_evaluator.eval_or_eval_array(
                            str(init_expr),
                            target_scope.get_evaluation_scope(),
                            line_number,
                        )
                        values = [init_value]
                    except Exception:
                        values = []
                else:
                    values = self._evaluate_generator_values(
                        init_expr, line_number)
                if not values:
                    # Not a generator/list: evaluate the init expression as a
                    # single value so object construction binds too
                    # (e.g. ': a init new Box').  Mirrors the block-scope
                    # fallback in _execute_for_declaration_block_scope.
                    try:
                        values = self._evaluate_push_expression(
                            str(init_expr), line_number)
                    except Exception:
                        values = []
                    if not values:
                        try:
                            values = [self.expr_evaluator.eval_or_eval_array(
                                str(init_expr),
                                target_scope.get_evaluation_scope(),
                                line_number)]
                        except Exception:
                            values = []
                if not values:
                    continue

                effective_type = target_scope.types.get(actual_key, type_name)
                effective_constraints = target_scope.constraints.get(
                    actual_key, constraints or {})
                self._apply_init_values(
                    actual_key,
                    effective_type,
                    effective_constraints,
                    values,
                    line_number,
                    scope=target_scope,
                )

    def _execute_for_declaration_block_scope(
            self,
            lines,
            i,
            line_number,
            var_list):
        for var, _, _, _ in var_list:
            defining_scope = self.current_scope().get_defining_scope(var)
            if defining_scope is not None and \
                    not self._is_outer_defining_scope(defining_scope):
                # Rule 2: a ``For ... do`` header over any live variable is
                # a redefinition, whichever nature declared it first.
                raise SyntaxError(
                    f"Variable '{var}' already defined in scope at line "
                    f"{line_number}")
        self.push_scope(is_private=True, is_loop_scope=True)
        # The loop variables are local to the For block: they are declared
        # in the block's own scope and die at End (everything else the body
        # binds is copied back out).
        block_loop_vars = {var.lower() for var, _, _, _ in var_list}
        block_scope = self.current_scope()
        evaluated_values = []
        for var, type_name, constraints, value in var_list:
            if value is not None:
                try:
                    evaluated_value = self.expr_evaluator.eval_expr(
                        str(value), self.current_scope().get_evaluation_scope(), line_number)
                    inferred_type = type_name or self.array_handler.infer_type(
                        evaluated_value, line_number)
                    if inferred_type == 'int':
                        inferred_type = 'number'
                    self.current_scope().define(
                        var, evaluated_value, inferred_type, constraints)
                    self.current_scope().update(
                        var, evaluated_value, line_number)
                    evaluated_values.append(evaluated_value)
                except NameError as e:
                    self.current_scope().define(
                        var, None, type_name, constraints, is_uninitialized=True)
            else:
                self.current_scope().define(
                    var, None, type_name, constraints, is_uninitialized=True)

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
            else:
                depth += block_nesting_delta(next_line_clean)
            for_block_lines.append((next_line, next_line_number))
            i += 1
        if depth > 0:
            raise SyntaxError(
                f"Unclosed FOR block starting at line {line_number}")
        if any(is_error_value(v) for v in evaluated_values):
            # The loop value is itself in error, so there is no iteration to
            # run - the same rule that skips an error element inside a For
            # list. The loop variable dies with the block like any other.
            self.pop_scope()
            return i
        init_entry = next(
            ((v, t, c) for v, t, c, _ in var_list if c and 'init' in c), None)
        import sys as _sys
        if init_entry:
            init_expr = init_entry[2].get('init')
            values = []
            call_match = re.match(
                r'^([A-Za-z][A-Za-z0-9_.]*)\s*\((.*)\)$', str(init_expr))
            if call_match and call_match.group(1).lower() in getattr(self, 'subprocesses', {}):
                call_name, call_args = call_match.groups()
                args_list = self._split_call_arguments(
                    call_args) if call_args else []
                arg_values = [self.expr_evaluator.eval_or_eval_array(
                    a, self.current_scope().get_evaluation_scope(), line_number) for a in args_list] if args_list else []
                try:
                    values = [self.call_subprocess(
                        call_name, arg_values, collect_all=False, line_number=line_number)]
                except Exception:
                    values = []
            if not values:
                try:
                    values = self._evaluate_push_expression(
                        str(init_expr), line_number)
                except Exception:
                    try:
                        values = [self.expr_evaluator.eval_or_eval_array(
                            str(init_expr), self.current_scope().get_evaluation_scope(), line_number)]
                    except Exception:
                        values = []
            for val in values:
                self.push_scope(is_private=True,
                                is_loop_scope=True)
                loop_scope = self.current_scope()
                for var, tname, constraints, _ in var_list:
                    if var in loop_scope.variables:
                        loop_scope.update(var, val, line_number)
                    else:
                        loop_scope.define(
                            var, val, tname, constraints, is_uninitialized=False)
                self.control_flow._process_block(
                    for_block_lines, loop_scope)
                self.control_flow._pop_for_block_scope(loop_scope)
            return i
        for var, _, constraints, _ in var_list:
            if constraints:
                try:
                    defining_scope = self.current_scope().get_defining_scope(var)
                    if defining_scope and not defining_scope.is_uninitialized(var):
                        var_value = defining_scope.get(var)
                        if var_value is not None:
                            defining_scope._check_constraints(
                                var, var_value, line_number)
                except ValueError as e:
                    for_block_lines = []
                    break
        self.control_flow._process_block(for_block_lines)
        self.control_flow._pop_for_block_scope(block_scope)
        return i

    def _handle_main_loop_for_range_case(
            self,
            lines,
            i,
            line_number,
            guard_condition,
            range_match,
            is_block):
        var_name, range_expr, step_str, index_var = range_match.groups()
        step = int(step_str) if step_str else 1

        values = self._evaluate_for_range_values(
            range_expr,
            step,
            line_number,
        )


        if is_block:
            # A block FOR owns the lines that follow it up to its END, so its
            # body is never zipped with the producers that come after it.
            return self._execute_range_for_block(
                lines,
                i,
                line_number,
                var_name,
                index_var,
                values,
                guard_condition,
            )

        # A bodyless 'For x in ...' is a row-stream producer: the wave
        # engine claims it, and when no wave applies the line only
        # evaluates its range - legacy dispatch handles what follows,
        # one line at a time.
        return i + 1

    def _evaluate_for_range_values(self, range_expr, step, line_number):
        if ' to ' in range_expr:
            range_parts = range_expr.split(' to ')
            try:
                start_expr = range_parts[0].strip()
                end_expr = range_parts[1].strip()
                start_value = self.expr_evaluator.eval_expr(
                    start_expr, self.current_scope().get_evaluation_scope(), line_number)
                start = int(start_value)
                end_value = self.expr_evaluator.eval_expr(
                    end_expr, self.current_scope().get_evaluation_scope(), line_number)
                end = int(end_value)
            except Exception as e:
                start = int(range_parts[0].strip())
                end = int(range_parts[1].strip())
            if step < 0:
                return list(range(start, end - 1, step))
            return list(range(start, end + 1, step))

        if range_expr.startswith('{') and range_expr.endswith('}'):
            values = []
            for raw_value in range_expr[1:-1].split(','):
                value = raw_value.strip()
                if value.startswith('"') and value.endswith('"'):
                    values.append(value[1:-1])
                elif value.startswith("'") and value.endswith("'"):
                    values.append(value[1:-1])
                else:
                    try:
                        values.append(float(value))
                    except ValueError:
                        values.append(value)
            return values

        try:
            eval_scope = self.current_scope().get_evaluation_scope()
            evaluated = self.expr_evaluator.eval_or_eval_array(
                range_expr, eval_scope, line_number)
        except Exception:
            raise SyntaxError(
                f"Invalid range expression: {range_expr} at line {line_number}")
        if isinstance(evaluated, dict):
            return self.array_handler.flatten_array(
                evaluated, line_number)
        if isinstance(evaluated, (list, tuple)):
            return list(evaluated)
        if evaluated is None:
            return []
        return [evaluated]

    @staticmethod
    def _wave_identifiers(text):
        """Every identifier token in an address or expression fragment."""
        return re.findall(r'[A-Za-z_]\w*', text or '')

    def _match_fp_call_name(self, expr):
        """Bare function/subprocess name called in *expr*, or None.

        Shared by the wave collector and the global-For pre-pass: an
        assignment whose RHS calls a defined function or subprocess may
        deliver a stream of values.
        """
        if not isinstance(expr, str) or not expr:
            return None
        call_match = re.search(r'([A-Za-z_]\w*)\s*\(', expr)
        if not call_match:
            return None
        name = call_match.group(1).lower()
        functions = self.functions or {}
        subprocesses = self.subprocesses or {}
        if name in functions or name in subprocesses:
            return name
        return None

    def _parse_wave_binding_candidate(self, stripped, line_number):
        """Classify a depth-0 line as a binding, via the main-line parsers.

        Returns ``(var, expr, init_expr, compile_time)``, or None when the
        line is not a plain binding.  ``For``/``Let`` keywords are stripped
        the same way the main loop strips them before handing the rest to
        ``_parse_variable_def``; bare and colon lines are gated by the same
        ``_is_optional_let_line`` predicate the main loop uses, so the wave
        claims exactly the lines the legacy path executes as declarations.
        """
        lower = stripped.lower()
        if lower.endswith('do') or lower.endswith('then'):
            return None
        compile_time = stripped.startswith(':')
        body = stripped
        if lower.startswith('for ') or lower.startswith('let '):
            body = stripped[4:].strip()
        elif not compile_time and not self.control_flow._is_optional_let_line(
                stripped):
            return None
        if not body or ' and ' in body.lower():
            # Multi-binding 'and' forms stay with the legacy path.
            return None
        # A WITH clause is stripped by the parser (its '=' may sit inside the
        # parens); keep its raw text so init/fp expressions can be re-emitted
        # as written, the way the legacy path re-reads the source line.
        with_match = self.parser._find_with_clause_balanced(body)
        raw_with = body[with_match[1]:with_match[2] + 1] if with_match else None
        try:
            var, _type_name, constraints, expr = self._parse_variable_def(
                body, line_number)
        except Exception:
            return None
        if 'var_list' in (constraints or {}):
            # Comma-separated declaration: legacy path only.
            return None
        if var and var.startswith('$'):
            var = var[1:]
        if not var or not re.fullmatch(r'[A-Za-z_]\w*', var):
            # Member/indexed targets are writes, not plain bindings.
            return None
        init_expr = (constraints or {}).get('init')
        if init_expr is not None and raw_with is not None:
            # Init never carries the parser's new-with fold (no '='), so the
            # written WITH text is always re-appended here.
            init_expr = f"{init_expr} with {raw_with}"
        elif (isinstance(expr, str) and raw_with is not None
                and not re.search(r'\bwith\s*\(', expr, re.I)):
            # Non-new forms store WITH in constraints only; re-attach it so
            # fp evaluation sees the expression as written.
            expr = f"{expr} with {raw_with}"
        return var, expr, init_expr, compile_time

    def _compute_wave_tainted_vars(self, seed_vars):
        """Close the wave-tied var set over the dependency graph.

        Seeds are the wave-tied vars (bodyless For-in vars, their index vars,
        function-producer binding vars); anything that transitively depends on
        a wave-tied var inherits the taint.  Graph edges run var -> dependent
        line numbers -> vars that line defines.
        """
        tainted = {name.lower() for name in seed_vars if name}
        by_variable = (self.dependency_graph or {}).get('by_variable') or {}
        by_line = (self.dependency_graph or {}).get('by_line') or {}
        changed = True
        while changed:
            changed = False
            for name in list(tainted):
                info = by_variable.get(name)
                if not info:
                    continue
                for line_number in info.get('dependents', ()) or ():
                    node = by_line.get(line_number)
                    if not node:
                        continue
                    for defined in node.get('defines', ()) or ():
                        if not isinstance(defined, str):
                            continue
                        dep_key = defined.lower()
                        if dep_key and dep_key not in tainted:
                            tainted.add(dep_key)
                            changed = True
        return tainted

    def _collect_global_wave_machinery(self, lines):
        """Collect the global bodyless wave machinery, order-independent.

        Depth-0 For-in producers, bare If stream guards, ``[...] :=`` rules,
        bare Push/Print/Return effects, and function-producer candidate
        bindings (``var = Func(...)`` whose var is referenced by the other
        machinery).  Returns the collection dict, or None when the program
        does not qualify - function/subprocess bodies, programs with global
        When blocks, or programs without a bodyless producer - and the
        legacy adjacency-based row-stream handling applies instead.
        """
        if getattr(self, '_is_operation_runner', False) or \
                getattr(self, '_parent_scope', None) is not None:
            return None
        from row_stream import parse_cell_rule, parse_guard, parse_producer
        producers = []
        guards = []
        rules = []
        effects = []
        candidates = []
        init_lines = set()
        has_global_when = False
        depth = 0
        control = self.control_flow
        for raw_line, line_number in lines:
            stripped = control._strip_inline_comment(raw_line).strip()
            if not stripped:
                continue
            lower = stripped.lower()
            is_block_if = bool(control._header_if.match(stripped))
            is_block_for = bool(control._header_for.match(stripped))
            is_block_when = bool(control._header_when.match(stripped))
            is_block_let = bool(control._header_let.match(stripped))
            is_define = _first_keyword(stripped) == 'define'
            # Definitions close with the declared name ('End Point'), blocks
            # with the bare 'end' token; both leave the enclosing depth.
            is_end = bool(re.match(r'^\s*end\b', stripped, re.I))
            if is_block_when and depth == 0:
                has_global_when = True
            at_depth_zero = depth == 0 and not (
                is_block_if or is_block_for or is_block_when or is_block_let
                or is_define or is_end)
            if at_depth_zero:
                producer = parse_producer(stripped)
                if producer is not None:
                    producers.append({'ln': line_number, 'data': producer})
                else:
                    guard = parse_guard(stripped)
                    if guard is not None:
                        guards.append({'ln': line_number, 'data': guard})
                    else:
                        # Static rules ([B2] := expr) are claimed here too:
                        # the wave fires them per wave when their value is
                        # wave-tainted, unlike the legacy row-stream group.
                        rule = parse_cell_rule(stripped, allow_static=True)
                        if rule is not None:
                            rules.append({
                                'ln': line_number, 'target': rule[0],
                                'value_expr': rule[1]})
                        elif lower == 'print' or lower.startswith('print '):
                            effects.append({
                                'ln': line_number, 'kind': 'print',
                                'data': stripped})
                        elif lower.startswith('return'):
                            return_match = self._match_return_statement(
                                stripped)
                            if return_match is not None:
                                effects.append({
                                    'ln': line_number, 'kind': 'return',
                                    'data': return_match.group(1).strip()})
                        elif lower.startswith('push '):
                            push_match = self._match_push_assignment(stripped)
                            if push_match is not None:
                                target, value_expr = (
                                    self._unpack_push_assignment(push_match))
                                effects.append({
                                    'ln': line_number, 'kind': 'push',
                                    'target': target,
                                    'value_expr': value_expr})
                        else:
                            binding = self._parse_wave_binding_candidate(
                                stripped, line_number)
                            claimed = False
                            if binding is not None:
                                var, expr, init_expr, cand_compile_time = (
                                    binding)
                                if expr is not None and (
                                        not str(expr).strip()):
                                    # Empty assignment RHS: not a binding.
                                    claimed = True
                                elif self._match_fp_call_name(expr) is not None:
                                    candidates.append({
                                        'ln': line_number,
                                        'var': var,
                                        'expr': expr, 'kind': 'fp'})
                                    claimed = True
                                if not claimed:
                                    if init_expr is not None:
                                        # Init splits into a declaration
                                        # (Phase-2 binding) and an assignment
                                        # (Phase-3 push effect), like ``For
                                        # x`` + ``Push x = expr``.
                                        candidates.append({
                                            'ln': line_number,
                                            'var': var,
                                            'expr': None,
                                            'kind': 'line',
                                            'init': True,
                                            'compile_time': cand_compile_time})
                                        effects.append({
                                            'ln': line_number,
                                            'kind': 'push',
                                            'target': var,
                                            'value_expr': init_expr})
                                        init_lines.add(line_number)
                                    else:
                                        candidates.append({
                                            'ln': line_number,
                                            'var': var,
                                            'expr': expr,
                                            'kind': 'line',
                                            # Colon declarations bind at
                                            # compile time (prep skips them);
                                            # they are already in place before
                                            # the main loop and are not
                                            # pending runtime bindings.
                                            'compile_time': cand_compile_time})
            if is_block_if or is_block_for or is_block_when or is_block_let \
                    or is_define:
                depth += 1
            if is_end and depth > 0:
                depth -= 1
        if has_global_when or (not producers and not candidates):
            return None
        if not rules and not effects:
            return None
        # A function-producer candidate is only claimed when its var is
        # referenced by the wave machinery; unrelated bindings stay with the
        # main loop.
        corpus_parts = []
        for rule in rules:
            corpus_parts.append(rule['target'])
            corpus_parts.append(rule['value_expr'])
        for guard in guards:
            corpus_parts.append(guard['data'])
        for effect in effects:
            corpus_parts.append(effect.get('target') or '')
            corpus_parts.append(effect.get('value_expr') or '')
            corpus_parts.append(effect.get('data') or '')
        for producer in producers:
            for _var, range_expr, _index_var in producer['data']:
                corpus_parts.append(range_expr)
        corpus = '\n'.join(corpus_parts)
        candidates = [
            cand for cand in candidates
            if re.search(r'\b' + re.escape(cand['var']) + r'\b',
                         corpus, re.I)]
        return {
            'producers': producers,
            'guards': guards,
            'rules': rules,
            'effects': effects,
            'candidates': candidates,
            'init_lines': init_lines,
        }

    def _prepare_wave_collection(self, collected):
        """Split candidates, settle streamless/taint/rule modes.  Shared by
        the top-level wave hook and the in-block wave."""
        line_candidates = [
            cand for cand in collected['candidates']
            if cand.get('kind') == 'line']
        collected['candidates'] = [
            cand for cand in collected['candidates']
            if cand.get('kind') != 'line']
        collected['line_candidates'] = line_candidates
        # No stream or function producer means no waves: effects run
        # once at their source position, and rules and guards stay with
        # the legacy main loop.
        streamless = (
            not collected['producers'] and not collected['candidates'])
        collected['streamless'] = streamless
        # Wave taint decides rule classification: a wave-tainted address
        # snapshots (establishes per wave, never listens); otherwise a
        # wave-tainted value follows in place; unrelated rules are handed
        # back to the main loop.
        seed_vars = set()
        for producer in collected['producers']:
            for var, _range_expr, index_var in producer['data']:
                seed_vars.add(var)
                if index_var:
                    seed_vars.add(index_var)
        for cand in collected['candidates']:
            seed_vars.add(cand['var'])
        self._wave_tainted = self._compute_wave_tainted_vars(seed_vars)
        kept_rules = []
        for rule in collected['rules']:
            target_ids = self._wave_identifiers(rule['target'])
            if any(name.lower() in self._wave_tainted
                   for name in target_ids):
                mode = 'snapshot'
            else:
                value_ids = self._wave_identifiers(rule['value_expr'])
                if any(name.lower() in self._wave_tainted
                       for name in value_ids):
                    mode = 'listener'
                else:
                    continue
            self._wave_rule_modes[rule['ln']] = mode
            kept_rules.append(rule)
        collected['rules'] = kept_rules
        return collected

    def _install_wave_claim_state(self, collected):
        """Build the claim set and effect/bindings bookkeeping from a
        prepared collection."""
        claim = (
            {p['ln'] for p in collected['producers']}
            | {r['ln'] for r in collected['rules']}
            | {c['ln'] for c in collected['candidates']}
            | {e['ln'] for e in collected['effects']})
        if not collected.get('streamless'):
            claim |= {g['ln'] for g in collected['guards']}
        self._wave_claim_set = claim
        self._wave_effects = {
            e['ln']: e for e in collected['effects']}
        self._wave_remaining_bindings = {
            c['ln'] for c in collected['line_candidates']
            if not c.get('compile_time')
            and c['ln'] not in getattr(self, 'hoisted_binding_lines', set())}
        self._wave_init_lines = collected.get('init_lines') or set()
        self._wave_deferred_effects = []
        self._wave_flush_pending = False

    def _engage_wave_streams(self):
        """Expand producer streams and evaluate fp candidates; sets
        ``_wave_engaged``.  Returns False when a range is unresolvable so
        the caller can settle the wave before any state mutates."""
        from row_stream import row_sequences
        collected = self._wave_collection
        env = self.current_scope().get_evaluation_scope()
        streams = []
        for producer in collected['producers']:
            rows = row_sequences(
                producer['data'], env=env,
                resolve=self._resolve_stream_range)
            if rows is None:
                return False
            streams.append(rows)
        collected['streams'] = streams
        # Function producers: candidates whose call delivers a stream
        # self-identify here; each wave rebinds the var to the next
        # delivered value, and it ends on the last one delivered.
        fp_producers = []
        scope = self.current_scope()
        for cand in collected['candidates']:
            values = self._evaluate_push_expression(
                cand['expr'], cand['ln'])
            value = values[0] if values else None
            inferred = self.array_handler.infer_type(value, cand['ln'])
            if inferred == 'int':
                inferred = 'number'
            def_scope = scope.get_defining_scope(cand['var'])
            if def_scope is not None:
                def_scope.update(cand['var'], value, cand['ln'])
            else:
                scope.define(cand['var'], value, inferred, {}, False)
            if len(values) > 1:
                fp_producers.append({
                    'var': cand['var'], 'values': values,
                    'ln': cand['ln']})
            else:
                self._register_listeners(
                    cand['var'], cand['expr'], scope)
        collected['fp_producers'] = fp_producers
        self._wave_engaged = True
        # A stream guard filters rows inside the wave loop; it is not a
        # whole-program output guard, so drop it from the global guard
        # bookkeeping (which re-evaluates guards after the loop and would
        # clear the grid, because the row variables do not exist globally).
        if not collected.get('streamless'):
            stream_guard_lines = {g['ln'] for g in collected['guards']}
            self.global_guard_line_numbers -= stream_guard_lines
            entries = getattr(self, 'global_guard_entries', None)
            if entries is not None:
                # In-place: run() captured the same list for its end-of-run
                # guard re-evaluation.
                entries[:] = [
                    entry for entry in entries
                    if entry.get('line_number') not in stream_guard_lines]
        return True

    def _try_execute_wave_machinery(self, lines, i):
        """Engage the global two-phase wave engine at the first claimed line.

        Phase A (once, here): expand every global bodyless producer, evaluate
        function-producer candidates, then advance all producers in lockstep,
        firing rules per wave (guards gate every fire) with zero side
        effects.  Phase B: bare Push/Print/Return lines run one round per
        recorded wave when the main loop reaches them; blocks and inits run
        self-contained in source order, as today.  Returns the next index, or
        None so the legacy dispatch handles the line.
        """
        if getattr(self, '_is_operation_runner', False) or \
                getattr(self, '_parent_scope', None) is not None:
            return None
        line_number = lines[i][1]
        if not self._wave_engaged and self._wave_collection is None:
            collected = self._collect_global_wave_machinery(lines)
            if collected is None:
                # Settled: never engage; do not re-collect per line.
                self._wave_collection = {'producers': []}
                return None
            # Classification is side-effect free and runs at the first line
            # dispatch.  Plain bindings (Let/bare/For-equality/colon
            # declarations) stay with the legacy main loop at their source
            # position; only effects defer until every one of them has run,
            # so Print/Push always observe the finished bindings.
            collected = self._prepare_wave_collection(collected)
            self._wave_collection = collected
            self._install_wave_claim_state(collected)
        if self._wave_collection is None or (
                not self._wave_engaged and not self._wave_claim_set and
                not getattr(self, '_wave_remaining_bindings', None)):
            return None
        # Flush deferred effects on the line after the last binding ran, so
        # the binding itself has executed before its consumers observe it.
        if getattr(self, '_wave_flush_pending', False):
            self._wave_flush_pending = False
            self._flush_deferred_wave_effects()
        if line_number in (getattr(self, '_wave_remaining_bindings', None)
                           or set()):
            self._wave_remaining_bindings.discard(line_number)
            # An Init binding line also carries a Phase-3 push effect: queue
            # it so the assignment runs after every binding is in place.
            if line_number in self._wave_effects:
                self._wave_deferred_effects.append(
                    self._wave_effects[line_number])
            if not self._wave_remaining_bindings:
                self._wave_flush_pending = True
            return None
        if line_number not in self._wave_claim_set:
            return None
        if not self._wave_engaged:
            # Expand streams first: an unresolvable range falls back to the
            # legacy path before any state is mutated.
            if not self._engage_wave_streams():
                self._wave_collection = {'producers': []}
                self._wave_claim_set = set()
                self._wave_effects = {}
                self._wave_remaining_bindings = set()
                self._wave_deferred_effects = []
                return None
        if not self._wave_phase_a_done:
            self._run_wave_phase_a()
        if line_number in self._wave_effects:
            # Effects run as Phase-B rounds; if plain bindings are still
            # ahead in the main loop, defer this effect until they finish.
            entry = self._wave_effects[line_number]
            if getattr(self, '_wave_remaining_bindings', None):
                self._wave_deferred_effects.append(entry)
            else:
                self._execute_wave_effect_round(entry)
            self.executed_global_for_lines.add(line_number)
        return i + 1

    def _flush_deferred_wave_effects(self):
        """Run Phase-B effects that waited for the last binding to finish."""
        deferred = getattr(self, '_wave_deferred_effects', None) or []
        self._wave_deferred_effects = []
        for entry in deferred:
            self._execute_wave_effect_round(entry)
            if entry.get('ln') is not None:
                self.executed_global_for_lines.add(entry['ln'])

    def _run_wave_phase_a(self):
        """Phase A: advance every producer in lockstep and fire rules per
        wave (guards gate fires, not advancement).  No Push/Print effects
        run here - waves are recorded for the Phase-B consumer round.
        Candidates, taint, and rule classification were settled at engage
        time so a dropped line is never swallowed."""
        self._wave_phase_a_done = True
        collected = self._wave_collection
        streams = collected['streams']
        fp_producers = collected['fp_producers']
        kept_rules = collected['rules']
        guards = collected['guards']
        # Lockstep wave loop: all producers advance together and stop at the
        # first exhaustion.
        min_len = min((len(stream) for stream in streams), default=0)
        for fp in fp_producers:
            min_len = min(min_len, len(fp['values']))
        records = []
        owner = getattr(self, 'compiler', None) or self
        for w in range(min_len):
            row = {}
            for stream in streams:
                row.update(stream[w])
            for fp in fp_producers:
                value = fp['values'][w]
                self._set_var_value(fp['var'], value, fp['ln'])
                row[fp['var']] = value
            if any(is_error_value(value) for value in row.values()):
                continue
            self.push_scope(is_private=True, is_loop_scope=True)
            try:
                scope = self.current_scope()
                for var, value in row.items():
                    vtype = self.array_handler.infer_type(value, 0)
                    if vtype == 'int':
                        vtype = 'number'
                    scope.define(var, value, vtype)
                eval_scope = scope.get_evaluation_scope()
                row_blocked = False
                for guard in guards:
                    if not self.control_flow._evaluate_if_condition(
                            guard['data'], guard['ln']):
                        row_blocked = True
                        break
                if row_blocked:
                    continue
                for rule in kept_rules:
                    was_snapshot = getattr(
                        owner, '_wave_snapshot_mode', False)
                    was_listener = getattr(
                        owner, '_wave_listener_mode', False)
                    rule_mode = self._wave_rule_modes.get(rule['ln'])
                    owner._wave_snapshot_mode = (rule_mode == 'snapshot')
                    owner._wave_listener_mode = (rule_mode == 'listener')
                    try:
                        self.array_handler.evaluate_line_with_assignment(
                            f"[{rule['target']}] := {rule['value_expr']}",
                            rule['ln'], eval_scope)
                    finally:
                        owner._wave_snapshot_mode = was_snapshot
                        owner._wave_listener_mode = was_listener
                records.append(dict(row))
            finally:
                self.pop_scope()
        self._wave_records = records
        consumed = (
            {p['ln'] for p in collected['producers']}
            | {r['ln'] for r in kept_rules}
            | {c['ln'] for c in collected['candidates']})
        if not collected.get('streamless'):
            consumed |= {g['ln'] for g in collected['guards']}
        self.executed_global_for_lines |= consumed

    def _execute_wave_effect_round(self, entry):
        """Phase B consumer round: run one bare effect once per recorded
        wave with that wave's values bound.  A streamless (bindings-only)
        program has no waves: the effect runs once at its source position."""
        records = getattr(self, '_wave_records', None) or []
        if not records:
            if not (getattr(self, '_wave_collection', None) or {}).get(
                    'streamless'):
                return
            if entry['kind'] == 'push':
                self._handle_push_assignment(
                    entry['target'], entry['value_expr'], entry['ln'])
            elif entry['kind'] == 'print':
                self._handle_print_statement(entry['data'], entry['ln'])
            else:
                push_values = self._evaluate_push_expression(
                    entry['data'], entry['ln'])
                for value in push_values:
                    self.output_values.setdefault('output', []).append(
                        self._snapshot_value(value))
                if 'output' not in self.output_variables:
                    self.output_variables.append('output')
            return
        for row in records:
            self.push_scope(is_private=True, is_loop_scope=True)
            try:
                scope = self.current_scope()
                for var, value in row.items():
                    vtype = self.array_handler.infer_type(value, 0)
                    if vtype == 'int':
                        vtype = 'number'
                    scope.define(var, value, vtype)
                if entry['kind'] == 'push':
                    self._handle_push_assignment(
                        entry['target'], entry['value_expr'], entry['ln'])
                elif entry['kind'] == 'print':
                    self._handle_print_statement(entry['data'], entry['ln'])
                else:
                    push_values = self._evaluate_push_expression(
                        entry['data'], entry['ln'])
                    for value in push_values:
                        self.output_values.setdefault('output', []).append(
                            self._snapshot_value(value))
                    if 'output' not in self.output_variables:
                        self.output_variables.append('output')
            finally:
                self.pop_scope()

    _WAVE_STATE_ATTRS = (
        '_wave_collection', '_wave_engaged', '_wave_claim_set',
        '_wave_effects', '_wave_records', '_wave_tainted',
        '_wave_rule_modes', '_wave_phase_a_done',
        '_wave_remaining_bindings', '_wave_init_lines',
        '_wave_deferred_effects', '_wave_flush_pending',
    )

    def _wave_state_snapshot(self):
        """Capture every wave attribute so an enclosing wave (top-level or
        outer block) survives an inner block wave."""
        return {name: getattr(self, name, None)
                for name in self._WAVE_STATE_ATTRS}

    def _wave_state_reset(self):
        """Install a fresh wave state (the values run() pre-seeds)."""
        self._wave_collection = None
        self._wave_engaged = False
        self._wave_claim_set = set()
        self._wave_effects = {}
        self._wave_records = []
        self._wave_tainted = set()
        self._wave_rule_modes = {}
        self._wave_phase_a_done = False
        self._wave_remaining_bindings = set()
        self._wave_init_lines = set()
        self._wave_deferred_effects = []
        self._wave_flush_pending = False

    def _wave_state_restore(self, snapshot):
        for name, value in snapshot.items():
            setattr(self, name, value)

    def _try_execute_block_wave(self, block_lines):
        """Engage a self-contained wave inside a block body.

        Mirrors the top-level engine, but eagerly: the body's depth-0 lines
        are classified, streams expanded, and Phase A runs at body entry so
        rules fire per wave before any body statement executes.  Bindings
        then run one-by-one at their source positions and effects fire one
        round per recorded wave at their position.  Returns the claim set,
        or None when the body does not qualify and the plain one-by-one
        walk applies.  The caller owns swapping ``self._wave*`` state
        around this call.
        """
        collected = self._collect_global_wave_machinery(block_lines)
        if collected is None:
            return None
        collected = self._prepare_wave_collection(collected)
        if not collected['producers']:
            # Bindings-only bodies have no waves: keep the one-by-one walk.
            return None
        self._wave_collection = collected
        self._install_wave_claim_state(collected)
        if not self._engage_wave_streams():
            self._wave_collection = {'producers': []}
            self._wave_claim_set = set()
            self._wave_effects = {}
            self._wave_remaining_bindings = set()
            self._wave_deferred_effects = []
            return None
        self._run_wave_phase_a()
        return self._wave_claim_set

    def _wave_block_step(self, line_number):
        """Advance the in-block wave for one body line.  Returns 'skip' when
        the line was claimed (already Phase-A'd or its effect round ran),
        'run' when the block handlers should execute it."""
        if getattr(self, '_wave_flush_pending', False):
            self._wave_flush_pending = False
            self._flush_deferred_wave_effects()
        if line_number in (getattr(self, '_wave_remaining_bindings', None)
                           or set()):
            self._wave_remaining_bindings.discard(line_number)
            if line_number in self._wave_effects:
                self._wave_deferred_effects.append(
                    self._wave_effects[line_number])
            if not self._wave_remaining_bindings:
                self._wave_flush_pending = True
            return 'run'
        if line_number not in self._wave_claim_set:
            return 'run'
        if line_number in self._wave_effects:
            entry = self._wave_effects[line_number]
            if getattr(self, '_wave_remaining_bindings', None):
                self._wave_deferred_effects.append(entry)
            else:
                self._execute_wave_effect_round(entry)
            self.executed_global_for_lines.add(line_number)
        return 'skip'


    def _resolve_stream_range(self, range_expr, row):
        """Resolve a producer range that is a bare identifier to an array of
        values (a plain list/tuple or a grid-array dict flattened), used when
        ``iter_range`` rejects it."""
        range_expr = range_expr.strip()
        if not re.match(r'^[A-Za-z_]\w*$', range_expr):
            return None
        value = row.get(range_expr)
        if value is None:
            try:
                value = self.current_scope().get(range_expr)
            except Exception:
                value = None
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, dict):
            try:
                return self.array_handler.flatten_array(value, 0)
            except Exception:
                return None
        if value is None:
            return None
        return [value]

    def _iter_for_values_skipping_errors(self, values):
        """Yield ``(idx, value)`` for For-loop iteration. Error elements are
        skipped, but the index still counts all elements (its position in the
        full value list)."""
        for idx, value in enumerate(values):
            if is_error_value(value):
                continue
            yield idx, value

    def _combo_has_error(self, combo):
        """True when any element of an ``and``-joined For combination is an
        error value (the whole combination is skipped)."""
        return any(is_error_value(item) for item in combo)

    def _iterate_for_zip(self, values_a, values_b):
        """Yield aligned ``(a, b)`` pairs from consecutive For loops, skipping
        pairs where either element is an error value."""
        for a, b in zip(values_a, values_b):
            if is_error_value(a) or is_error_value(b):
                continue
            yield a, b

    def _execute_range_for_block(
            self,
            lines,
            i,
            line_number,
            var_name,
            index_var,
            values,
            guard_condition):
        for_block_lines = []
        i += 1
        depth = 1
        while i < len(lines) and depth > 0:
            next_line, next_line_number = lines[i]
            next_line_clean = next_line.strip().lower()
            if next_line_clean.lower() == "end":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            else:
                depth += block_nesting_delta(next_line_clean)
            for_block_lines.append(
                (next_line, next_line_number))
            i += 1

        if depth > 0:
            raise SyntaxError(
                f"Unclosed FOR block starting at line {line_number}")

        for idx, value in self._iter_for_values_skipping_errors(values):
            self.push_scope(is_private=True,
                            is_loop_scope=True)
            loop_scope = self.current_scope()
            if var_name in loop_scope.variables:
                loop_scope.update(var_name, value, line_number)
            else:
                loop_scope.define(var_name, value, 'number')
            if index_var:
                if index_var in loop_scope.variables:
                    loop_scope.update(index_var, idx + 1)
                else:
                    loop_scope.define(index_var, idx + 1, 'number')
            if guard_condition:
                if not self.control_flow._evaluate_if_condition(
                        guard_condition, line_number):
                    self.pop_scope()
                    break
            self.control_flow._process_block(for_block_lines)
            parent_scope = loop_scope.parent
            if parent_scope:
                loop_local_lets = getattr(
                    loop_scope, 'local_let_declarations', set())
                # The loop and index variables are local to the For block
                # and die with it; everything else the body bound copies out.
                block_vars = {var_name.lower()}
                if index_var:
                    block_vars.add(index_var.lower())
                for name, val in loop_scope.variables.items():
                    if name.lower() in loop_local_lets:
                        continue
                    if name.lower() in block_vars:
                        continue
                    try:
                        parent_scope.update(name, val)
                    except Exception:
                        inferred_type = self.array_handler.infer_type(
                            val, line_number)
                        parent_scope.define(
                            name, val, inferred_type, {}, is_uninitialized=False)
            self.pop_scope()
            if self.exit_loop:
                self.exit_loop = False
                break
        return i


    def _handle_main_loop_nested_for_case(
            self,
            lines,
            i,
            line_number,
            var_defs,
            is_block):
        var_parts = split_var_defs(var_defs)
        loop_configs = []

        for var_part in var_parts:
            range_match = re.match(
                r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(\d+))?(?:\s+index\s+([\w_]+))?$',
                var_part.strip(),
                re.I,
            )
            if not range_match:
                raise SyntaxError(
                    f"Invalid FOR loop syntax: {var_part} at line {line_number}")
            var_name, range_expr, step_str, index_var = range_match.groups()
            step = int(step_str) if step_str else 1
            if ' to ' in range_expr:
                range_parts = range_expr.split(' to ')
                start_expr = range_parts[0].strip()
                end_expr = range_parts[1].strip()
                is_dynamic = False
                if not start_expr.replace('.', '').replace('-', '').isdigit() or not end_expr.replace('.', '').replace('-', '').replace('+', '').replace('*', '').replace('/', '').replace('(', '').replace(')', '').isdigit():
                    is_dynamic = True
                    values = None
                else:
                    start = int(start_expr)
                    end = int(end_expr)
                    values = list(range(start, end + 1, step))
            elif range_expr.startswith('{') and range_expr.endswith('}'):
                values_str = range_expr[1:-1].split(',')
                values = []
                for v in values_str:
                    v_clean = v.strip()
                    if v_clean.startswith('"') and v_clean.endswith('"'):
                        values.append(v_clean[1:-1])
                    elif v_clean.startswith("'") and v_clean.endswith("'"):
                        values.append(v_clean[1:-1])
                    else:
                        try:
                            values.append(float(v_clean))
                        except ValueError:
                            values.append(v_clean)
                is_dynamic = False
                start_expr = None
                end_expr = None
            else:
                try:
                    eval_scope = self.current_scope().get_evaluation_scope()
                    evaluated = self.expr_evaluator.eval_or_eval_array(
                        range_expr, eval_scope, line_number)
                    if isinstance(evaluated, dict):
                        values = self.array_handler.flatten_array(
                            evaluated, line_number)
                    elif isinstance(evaluated, (list, tuple)):
                        values = list(evaluated)
                    elif evaluated is None:
                        values = []
                    else:
                        values = [evaluated]
                    is_dynamic = False
                    start_expr = None
                    end_expr = None
                except Exception:
                    raise SyntaxError(
                        f"Invalid range expression: {range_expr} at line {line_number}")

            loop_configs.append(
                (
                    var_name,
                    values,
                    index_var,
                    is_dynamic,
                    start_expr if is_dynamic else None,
                    end_expr if is_dynamic else None,
                    step,
                )
            )


        value_combinations = list(
            self._generate_nested_for_combinations(loop_configs, line_number))

        if is_block:
            for_block_lines = []
            i += 1
            depth = 1
            while i < len(lines) and depth > 0:
                next_line, next_line_number = lines[i]
                next_line_clean = next_line.strip().lower()
                if self._is_keyword(next_line_clean, "end"):
                    depth -= 1
                    if depth > 0:
                        for_block_lines.append(
                            (next_line, next_line_number))
                        i += 1
                        continue
                    i += 1
                    break
                else:
                    depth += block_nesting_delta(next_line_clean)
                for_block_lines.append(
                    (next_line, next_line_number))
                i += 1
            if depth > 0:
                raise SyntaxError(
                    f"Unclosed FOR block starting at line {line_number}")

            for combo in value_combinations:
                if self.exit_loop:
                    self.exit_loop = False
                    break
                if self._combo_has_error(combo):
                    continue
                # The For block owns its scope: the loop variables are
                # local to the block (they die at End), everything else
                # the body binds copies back out.
                self.push_scope(is_private=True, is_loop_scope=True)
                loop_scope = self.current_scope()
                for var_idx, (var_name, _, index_var, _, _, _, _) in enumerate(loop_configs):
                    if var_name in loop_scope.variables:
                        loop_scope.update(
                            var_name, combo[var_idx], line_number)
                    else:
                        loop_scope.define(
                            var_name, combo[var_idx], 'number')
                    if index_var:
                        if index_var in loop_scope.variables:
                            loop_scope.update(
                                index_var, var_idx + 1, line_number)
                        else:
                            loop_scope.define(
                                index_var, var_idx + 1, 'number')
                self.control_flow._process_block(for_block_lines)
                parent_scope = loop_scope.parent
                if parent_scope:
                    loop_local_lets = getattr(
                        loop_scope, 'local_let_declarations', set())
                    block_vars = {cfg[0].lower() for cfg in loop_configs}
                    block_vars.update(
                        cfg[2].lower() for cfg in loop_configs if cfg[2])
                    for name, value in loop_scope.variables.items():
                        if name.lower() in loop_local_lets:
                            continue
                        if name.lower() in block_vars:
                            continue
                        try:
                            parent_scope.update(
                                name, value)
                        except Exception:
                            inferred_type = self.array_handler.infer_type(
                                value, line_number)
                            parent_scope.define(
                                name, value, inferred_type, {}, is_uninitialized=False)
                self.pop_scope()
                if self.exit_loop:
                    self.exit_loop = False
                    break
            return i

        if i + 1 < len(lines):
            next_line, next_line_number = lines[i + 1]
            if ':=' in next_line:
                self._execute_for_assignment_combinations(
                    value_combinations,
                    [(cfg[0], cfg[2]) for cfg in loop_configs],
                    next_line, next_line_number)

        return i + 1

    def _generate_nested_for_combinations(
            self,
            loop_configs,
            line_number,
            loop_idx=0,
            current_combo=None):
        if current_combo is None:
            current_combo = []
        if loop_idx >= len(loop_configs):
            yield current_combo
            return
        _, values, _, is_dynamic, start_expr, end_expr, step = loop_configs[
            loop_idx]
        if is_dynamic:
            self.push_scope(
                is_private=True, is_loop_scope=True)
            for combo_idx, prev_config in enumerate(loop_configs[:loop_idx]):
                prev_var_name = prev_config[0]
                self.current_scope().define(
                    prev_var_name, current_combo[combo_idx], 'number')
            start = int(self.expr_evaluator.eval_expr(
                start_expr, self.current_scope().get_evaluation_scope(), line_number))
            end = int(self.expr_evaluator.eval_expr(
                end_expr, self.current_scope().get_evaluation_scope(), line_number))
            dynamic_values = list(
                range(start, end + 1, step))
            self.pop_scope()
            for val in dynamic_values:
                yield from self._generate_nested_for_combinations(
                    loop_configs,
                    line_number,
                    loop_idx + 1,
                    current_combo + [val],
                )
            return
        for val in values:
            yield from self._generate_nested_for_combinations(
                loop_configs,
                line_number,
                loop_idx + 1,
                current_combo + [val],
            )

    def _handle_main_loop_for_prechecks(self, lines, i, line, line_number):
        guard_condition = self.for_guard_conditions.get(line_number)
        handled, next_i = self._handle_for_array_and_dim_declarations(
            line, line_number, i)
        if handled:
            return {
                'handled': True,
                'next_i': next_i,
                'guard_condition': guard_condition,
                'match': None,
            }

        m = re.match(r'^\s*for\s+(.+?)(?:\s+do\s*$|\s*$)', line, re.I)
        if m:
            var_defs = m.group(1).strip()
            has_executable = self._has_push_action(line) or ':=' in line
            is_block = line.strip().lower().endswith('do') and not has_executable

            handled, next_i = self._handle_for_simple_declarations(
                var_defs, line_number, i)
            if handled:
                return {
                    'handled': True,
                    'next_i': next_i,
                    'guard_condition': guard_condition,
                    'match': m,
                }

            if has_executable:
                handled, next_i = self._handle_for_single_line_executable(
                    line, line_number, i)
                if handled:
                    return {
                        'handled': True,
                        'next_i': next_i,
                        'guard_condition': guard_condition,
                        'match': m,
                    }

            handled, next_i = self._handle_for_and_syntax(
                lines, i, var_defs, is_block, line_number)
            if handled:
                return {
                    'handled': True,
                    'next_i': next_i,
                    'guard_condition': guard_condition,
                    'match': m,
                }

        handled, next_i = self._handle_consecutive_nested_for_shortcuts(lines, i)
        if handled:
            return {
                'handled': True,
                'next_i': next_i,
                'guard_condition': guard_condition,
                'match': m,
            }

        return {
            'handled': False,
            'next_i': i,
            'guard_condition': guard_condition,
            'match': m,
        }

    def _compute_assignment_deps(self, expr):
        """Dependencies for an assignment RHS, WITH-aware.

        For 'new Type ... with ...' expressions only the constructor
        argument variables and the with-clause VALUE expressions are
        dependencies; field names and the type name are not.
        """
        compiler = getattr(self, 'compiler', None)
        if compiler is not None:
            base_expr, with_text = compiler._split_new_with_expr(expr)
            if with_text:
                deps = compiler._with_deps(with_text)
                deps |= compiler._extract_identifier_tokens(base_expr or '')
                return deps
        return _filter_var_tokens(
            set(re.findall(r'\b[\w_]+\b', _strip_builder_arrows(
                mask_text_constant_tokens(expr)))))

    def _handle_main_loop_post_for_branches(self, lines, i, line, line_number, stripped, stripped_lower):
        if stripped_lower.startswith("if "):
            if_block = None
            for block in self.control_flow.if_blocks:
                if block['start_line'] == line_number:
                    if_block = block
                    break

            if if_block and len(if_block.get('clauses', [])) > 1:
                skip_lines = self.control_flow._process_if_statement_rich(
                    line, line_number, lines, i)
            else:
                skip_lines = self.control_flow._process_if_statement(
                    line, line_number, lines, i)
            return i + skip_lines

        if stripped_lower.startswith("let "):
            return self._handle_main_loop_let_statement(
                lines, i, line, line_number, stripped)

        if stripped_lower.startswith('for '):
            return self._handle_main_loop_for_fallback_statement(
                lines, i, line, line_number, stripped, stripped_lower)

        if stripped.startswith(":") and "=" in stripped and not stripped_lower.startswith(("for ", "let ")):
            var_def, expr = map(str.strip, line[1:].split("=", 1))
            var, type_name, constraints, value = self._parse_variable_def(
                var_def, line_number)
            if 'init' not in (constraints or {}):
                sub_name = self._leading_subprocess_call(expr)
                if sub_name:
                    raise RuntimeError(
                        f"Equality assignment to subprocess '{sub_name}' is not allowed; use INIT instead at line {line_number}")
            deps = self._compute_assignment_deps(expr)
            if not deps:
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
                    inferred_type = type_name or self.array_handler.infer_type(
                        evaluated_value, line_number)
                    if inferred_type == 'int':
                        inferred_type = 'number'
                    evaluated_value = self._to_sparse_undimmed(
                        evaluated_value, constraints)
                    self.current_scope().define(var, evaluated_value, inferred_type,
                                                constraints, is_uninitialized=False)
                    if var in self.pending_assignments:
                        del self.pending_assignments[var]
                except Exception as e:
                    self.pending_assignments[var] = (
                        expr, line_number, deps, constraints)
            else:
                self.pending_assignments[var] = (
                    expr, line_number, deps, constraints)
            return i + 1

        return self._handle_main_loop_misc_statement(
            line, line_number, i, lines=lines)

    def _handle_main_loop_misc_statement(self, line, line_number, i, lines=None):
        handled_grid, next_i = self._handle_main_loop_grid_assignment(
            line, line_number, i)
        if handled_grid:
            return next_i

        if (re.match(r'^\[\s*\^?[A-Za-z]+\d+\s*\]\s*:\s*', line) and '=' in line and ':=' not in line):
            try:
                self.compiler._evaluate_cell_var_definition(
                    line, line_number)
            except Exception as e:
                raise RuntimeError(
                    f"Error processing cell binding line: {e} at line {line_number}")
            return i + 1

        if (
            ':=' not in line
            and not re.match(r'^\[\s*\^?[A-Za-z]+\d+\s*\]\s*:\s*', line)
            and self.control_flow._is_optional_let_line(line.strip())
        ):
            # A declaration with the LET keyword left out binds exactly as an
            # explicit LET does, including constraint-only forms that carry no
            # '=' at all ('x > 1', 'v as number').
            if lines is not None:
                return self._handle_main_loop_let_statement(
                    lines, i, 'Let ' + line, line_number,
                    'Let ' + line.strip())
            if '=' not in line:
                # No line context to defer against; the predicate already
                # confirmed this is a declaration.
                return i + 1
            target, rhs = line.split('=', 1)
            target, rhs = target.strip(), rhs.strip()
            rhs_vars = None
            if rhs.lstrip('+-').startswith('#'):
                rhs_vars = set()
            if '$"' in rhs:
                if rhs_vars is None:
                    rhs_vars = set()
                placeholders = re.findall(r'\{\s*([^}]*?)\s*\}', rhs)
                for ph in placeholders:
                    clean_ph = ph.lstrip('{')
                    if clean_ph.strip().startswith('*'):
                        continue
                    if resolve_text_constant_token(clean_ph) is not None:
                        continue
                    rhs_vars.update(re.findall(
                        r'\b[\w_]+\b', clean_ph))
            if rhs_vars is None:
                rhs_vars = set(re.findall(
                    r'\b[\w_]+\b(?=\s*(?:[\[\{]|!\w+\s*\(|(?:\.\w+)?\s*$))', _strip_constraint_operands(mask_text_constant_tokens(rhs))))
            field_pairs = re.findall(r'\b([\w_]+)\.\s*([\w_]+)', rhs)
            if field_pairs:
                base_fields = {base for base, _ in field_pairs}
                attr_fields = {attr for _, attr in field_pairs}
                current = rhs_vars or set()
                rhs_vars = (current | base_fields) - attr_fields
            col_tokens = re.findall(r'\[\s*([A-Za-z]+)\s*\{', rhs)
            col_interp_pattern = re.compile(
                r'\[\{\s*([^}:]+?)\s*:\s*([A-Za-z]+)\s*\}\s*(\d+|\{[^}]+\})\s*\]')
            col_interp_base_cols = []
            for index_expr, base_col, row_part in col_interp_pattern.findall(rhs):
                rhs_vars.update(re.findall(r'\b[\w_]+\b', index_expr))
                if row_part.startswith('{') and row_part.endswith('}'):
                    rhs_vars.update(
                        re.findall(r'\b[\w_]+\b', row_part[1:-1]))
                col_interp_base_cols.append(base_col)
            if col_tokens or col_interp_base_cols:
                rhs_vars = {v for v in rhs_vars if v not in set(
                    col_tokens + col_interp_base_cols)}
            field_vars = set(re.findall(
                r'\b[\w_]+\b(?=\.\w+\s*$)', rhs))
            rhs_vars.update(field_vars)
            target_vars = set()
            if '{' in target:
                for match in re.finditer(r'\{([^}]+)\}', target):
                    expr = match.group(1).strip()
                    target_vars.update(re.findall(r'\b[\w_]+\b', expr))
            col_interp_pattern = re.compile(
                r'\[\{\s*([^}:]+?)\s*:\s*([A-Za-z]+)\s*\}\s*(\d+|\{[^}]+\})\s*\]')
            col_interp_base_cols = []
            for index_expr, base_col, row_part in col_interp_pattern.findall(target):
                target_vars.update(re.findall(
                    r'\b[\w_]+\b', index_expr))
                if row_part.startswith('{') and row_part.endswith('}'):
                    target_vars.update(
                        re.findall(r'\b[\w_]+\b', row_part[1:-1]))
                col_interp_base_cols.append(base_col)
            if col_interp_base_cols:
                target_vars = {
                    v for v in target_vars if v not in set(col_interp_base_cols)}
            rhs_vars = _filter_var_tokens(rhs_vars)
            target_vars = _filter_var_tokens(target_vars)
            rhs_vars = _strip_cell_address_tokens(line, rhs_vars)
            target_vars = _strip_cell_address_tokens(line, target_vars)
            if hasattr(self, 'compiler') and getattr(self.compiler, 'types_defined', None):
                type_keys = set(self.compiler.types_defined.keys())
                rhs_vars = {v for v in rhs_vars if v.lower()
                            not in type_keys}
                target_vars = {
                    v for v in target_vars if v.lower() not in type_keys}
            scope = self.current_scope()
            scope_pending = getattr(scope, 'pending_assignments', {})
            unresolved = any(self.has_unresolved_dependency(
                var, scope=scope, scope_pending=scope_pending) for var in rhs_vars | target_vars)
            if rhs_vars | target_vars:
                for var in rhs_vars | target_vars:
                    is_uninit = self.current_scope().is_uninitialized(var)
                    in_pending = var in self.pending_assignments
            if unresolved:
                self.pending_assignments[f"__line_{line_number}"] = (
                    line, line_number, rhs_vars | target_vars)
            else:
                violations = []
                for var in target_vars:
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
                    is_plain_var_target = re.match(
                        r'^[A-Za-z][A-Za-z0-9_.]*$', target) is not None
                    try:
                        if is_plain_var_target:
                            evaluated_value = self.expr_evaluator.eval_or_eval_array(
                                rhs, scope.get_evaluation_scope(), line_number)
                            defining_scope = scope.get_defining_scope(target)
                            if defining_scope:
                                defining_scope.update(
                                    target, evaluated_value, line_number)
                                self._register_listeners(
                                    target, rhs, scope)
                            else:
                                inferred_type = self.array_handler.infer_type(
                                    evaluated_value, line_number)
                                scope.define(
                                    target, evaluated_value, inferred_type, {}, is_uninitialized=False)
                                self._register_listeners(
                                    target, rhs, scope)
                        else:
                            self.array_handler.evaluate_line_with_assignment(
                                line, line_number, scope.get_evaluation_scope())
                            rhs_deps = self._cell_spill_dependencies(
                                rhs) if rhs_vars else set()
                            if rhs_deps:
                                self._register_cell_spill_listener(
                                    line, rhs_deps, line_number, scope)
                    except Exception as e:
                        missing = self.extract_missing_dependencies(e)
                        if missing:
                            for dep in missing:
                                self.mark_dependency_missing(dep)
                            deps = (rhs_vars | target_vars) | missing
                            self.pending_assignments[f"__line_{line_number}"] = (
                                line, line_number, deps)
                            return i + 1
                        raise
                else:
                    self.pending_assignments[f"__line_{line_number}"] = (
                        line, line_number, rhs_vars | target_vars)
            return i + 1

        if re.match(r'^\[\s*\^?[A-Za-z]+\d+\s*\]\s*:\s*', line):
            try:
                if '=' in line:
                    self.compiler._evaluate_cell_var_definition(
                        line, line_number)
                else:
                    self.compiler._process_cell_binding_declaration(
                        line, line_number)
            except Exception as e:
                raise RuntimeError(
                    f"Error processing cell binding line: {e} at line {line_number}")
            return i + 1

        if self._maybe_handle_subprocess_call(line, line_number):
            return i + 1

        self._reject_stray_pseudo_constraint(line, line_number)
        self._reject_unknown_call(line, line_number)
        return i + 1

    # 'live' and 'init' are declaration modifiers, not statements: they belong
    # on the Input/Output/For line that declares the slot.
    _PSEUDO_CONSTRAINT_RE = re.compile(
        r'^\s*([A-Za-z_][\w.]*)\s+(live|init)\b', re.I)

    def _reject_stray_pseudo_constraint(self, line, line_number):
        """Raise when ``live``/``init`` is used as a standalone statement."""
        m = self._PSEUDO_CONSTRAINT_RE.match(line.strip())
        if not m:
            return
        name, keyword = m.group(1), m.group(2).lower()
        rest = line.strip()[m.end():]
        # 'a init 5' is still a stray modifier on a bare line.
        raise SyntaxError(
            f"'{keyword}' is only valid on an Input/Output/For declaration "
            f"line, not on a standalone statement: '{name} {keyword}' at line "
            f"{line_number}"
            + (f" (unexpected trailing {rest.strip()!r})" if rest.strip() else ''))

    def _reject_unknown_call(self, line, line_number):
        """Turn a bare call to a non-existent subprocess/function into an error.

        A statement of the exact shape ``Name(args)`` reached the end of the
        main-loop dispatch with no handler: either it names nothing the engine
        knows (typo, missing ``use``/``Require``) or it is a call-shaped read
        that produces no value. Names that resolve to a function, subprocess,
        granted capability handle, or any in-scope variable keep today's
        no-op behavior; truly unknown callees raise instead of failing
        silently.
        """
        m = re.match(
            r'^\s*([A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)*)\s*\(.*\)\s*$',
            line, re.S)
        if not m:
            return
        name = m.group(1)
        base = name.split('.')[0].lower()
        try:
            func_defs = getattr(self, 'functions', {}) or {}
            sp_defs = getattr(self, 'subprocesses', {}) or {}
            compiler = getattr(self, 'compiler', None)
            if compiler is not None:
                func_defs = getattr(compiler, 'functions', {}) or {}
                sp_defs = getattr(compiler, 'subprocesses', {}) or {}
            if name.lower() in func_defs or name.lower() in sp_defs:
                return
            scope = self.current_scope()
            if scope is not None:
                if scope._get_case_insensitive_key(base, scope.variables):
                    return
            if base in (getattr(self, '_scope_handle_vars', lambda: {})()):
                return
            if base in (getattr(self, 'require_caps', {}) or {}):
                return
            try:
                from builtin_functions import BUILTINS
                if base in (BUILTINS or {}):
                    return
            except ImportError:
                pass
        except Exception:
            return
        raise NameError(
            f"Name '{name}' is not defined at line {line_number}")

    def _handle_main_loop_grid_assignment(self, line, line_number, i):
        if ':=' not in line:
            return False, i
        target, value = line.split(':=')
        target = target.strip()
        value = value.strip()
        if not re.match(r'^\[\^?[A-Za-z{]', target):
            return False, i
        if not target.endswith(']'):
            return False, i
        cell_ref = target[1:-1].strip()
        if not re.match(r'^\^?[A-Za-z{]', cell_ref):
            return False, i
        sub_name = self._leading_subprocess_call(value)
        if sub_name:
            raise RuntimeError(
                f"Equality assignment to subprocess '{sub_name}' is not allowed; use INIT instead at line {line_number}")
        if '{' in cell_ref or ':' in cell_ref:
            scope = self.current_scope()
            self.array_handler.evaluate_line_with_assignment(
                line, line_number, scope.get_evaluation_scope())
            return True, i + 1
        try:
            if cell_ref.startswith('^'):
                array_cell_ref = cell_ref[1:].strip()
                parse_address(array_cell_ref)
                array_cell_key = self.compiler._to_index(array_cell_ref)
                scope_value = self.current_scope().get_evaluation_scope()
                is_grid_value = '.grid{' in value
                evaluated_value = self.expr_evaluator.eval_or_eval_array(
                    value, scope_value, line_number, is_grid_dim=is_grid_value)
                self.array_handler._assign_horizontal_array(
                    array_cell_key, evaluated_value, value, line_number=line_number)
            else:
                indices = parse_address(cell_ref)
                grid_dims = self.current_scope().constraints.get('grid', {}).get('dim')
                if isinstance(grid_dims, list):
                    max_rank = len(grid_dims)
                else:
                    max_rank = 2
                if len(indices) > max_rank:
                    raise ValueError(
                        f"Cell address '{cell_ref}' has rank {len(indices)} but grid has rank {max_rank} at line {line_number}")
                scope = self.current_scope()
                self.array_handler.evaluate_line_with_assignment(
                    line, line_number, scope.get_evaluation_scope())
        except Exception as e:
            raise RuntimeError(
                f"Error evaluating '{value}': {e} at line {line_number}")
        rhs_deps = self._cell_spill_dependencies(value)
        if rhs_deps:
            scope = self.current_scope()
            self._register_cell_spill_listener(
                line, rhs_deps, line_number, scope)
        return True, i + 1

    def _handle_when_block(self, lines, i, line, line_number):
        when_match = re.match(
            r'^\s*when\s+(.+?)\s+do\s*$', line, re.I)
        if not when_match:
            raise SyntaxError(
                f"Invalid WHEN syntax at line {line_number}")
        condition = when_match.group(1).strip()
        block_lines = []
        i += 1
        depth = 1
        while i < len(lines) and depth > 0:
            next_line, next_ln = lines[i]
            clean = next_line.strip().lower()
            if clean == 'end':
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            elif clean.startswith('for ') and ' do' in clean:
                depth += 1
            elif clean.startswith('when ') and ' do' in clean:
                depth += 1
            elif clean.startswith('if ') and clean.endswith('then'):
                depth += 1
            block_lines.append((next_line, next_ln))
            i += 1
        self._register_when_block(condition, block_lines, line_number)
        return True, i

    def _handle_dim_label_assignment(self, line, line_number):
        # Handle named dimension label assignment like Results!Quarter.Label{"Q1", "Q2", "Q3", "Q4"}
        m = re.match(
            r'^\s*([\w_]+)!(\w+)\.Label\s*\{([^}]+)\}\s*$', line)
        if not m:
            return False
        var_name, dim_name, labels_str = m.groups()

        # Parse the labels
        labels = [label.strip().strip('"')
                  for label in labels_str.split(',')]

        # Store the labels in the compiler for later use
        if not hasattr(self, 'dim_labels'):
            self.dim_labels = {}
        if var_name not in self.dim_labels:
            self.dim_labels[var_name] = {}
        self.dim_labels[var_name][dim_name] = {
            label: i for i, label in enumerate(labels)}
        return True

    def _handle_for_grid_dim(self, stripped, line_number):
        m = re.match(r'^\s*FOR\s+(.+?)(?:\s+DO\s*$|\s*$)',
                     stripped, re.I)
        if not m:
            raise SyntaxError(
                f"Invalid FOR syntax at line {line_number}")
        var_def = m.group(1).strip()
        var, type_name, constraints, expr = self._parse_variable_def(
            var_def, line_number)

        if not type_name and isinstance(expr, str):
            ctor_match = re.match(
                r'^\s*new\s+([A-Za-z][A-Za-z0-9_]*)\s*$', expr, re.I)
            if ctor_match:
                type_name = ctor_match.group(1).lower()

        if type_name and ('with' in constraints or 'dim' in constraints) and type_name in self.types_defined:
            current_tensor = var
            with_constraints = constraints.get('with', {})
            dim_constraints = constraints.get('dim', {})
            if dim_constraints:
                dims = dim_constraints.get('dims', [])
                shape = [size for label, size in dims]
                pa_type = 'number'
                matrix_data = dim_constraints.get('matrix_data')
                data_var = dim_constraints.get('data_var')
                if data_var:
                    scope = self.current_scope().get_full_scope()
                    if data_var not in scope:
                        raise NameError(
                            f"Variable '{data_var}' not defined at line {line_number}")
                    matrix_data = scope[data_var]
                    if isinstance(matrix_data, dict) and 'array' in matrix_data:
                        matrix_data = self._dict_array_to_matrix_data(
                            matrix_data, line_number)
                default_value = dim_constraints.get('value')
                if default_value is None and expr and not isinstance(expr, list):
                    try:
                        default_value = float(expr)
                    except (TypeError, ValueError):
                        default_value = None
                grid_data = self.array_handler.create_array(
                    shape, default_value, pa_type, line_number, matrix_data=matrix_data, is_grid_dim=True)

                public_fields = public_type_fields(
                    self.types_defined[type_name])
                tensor_struct = {f: (self.compiler.resolve_with_value(
                    with_constraints.get(f, None), line_number)
                    if with_constraints.get(f, None) is not None else None)
                    for f in public_fields}
                tensor_struct['grid'] = grid_data
                tensor_struct['original_shape'] = grid_data['original_shape']
                tensor_struct['constraints'] = constraints
                tensor_struct['_type_name'] = type_name.lower()
                hidden_fields = self.types_defined[type_name].get(
                    '_hidden_fields', set())
                if hidden_fields:
                    tensor_struct['_hidden_fields'] = set(
                        hidden_fields)

                self.current_scope().define(var, tensor_struct, type_name,
                                            constraints, is_uninitialized=False)
                for field in public_fields:
                    if field in with_constraints:
                        self.current_scope().define(
                            f"{var}.{field}",
                            self.compiler.resolve_with_value(
                                with_constraints[field], line_number),
                            'text')
        return True

    def _dict_array_to_matrix_data(self, arr, line_number=None):
        """Convert a pipe dict-form array back into a list of display matrices.

        Pipe results (from _evaluate_pipe_array) have the new dim at index 1
        with column-major storage, so flat[i0 + s0*i1 + s0*s1*i2] = arr[i0][i1][i2].
        Grid DIM stacks the operand matrices along the last declared dim, so the
        conversion extracts each matrix as its display-form nested list.
        """
        flat = arr.get('array', arr.get('grid'))
        shape = list(arr.get('shape') or arr.get('original_shape') or [])
        if not shape or not isinstance(flat, list):
            raise ValueError(
                f"Invalid array data for grid DIM at line {line_number}")
        s0 = shape[0]
        num_operands = shape[1]
        col_size = 1
        for s in shape[2:]:
            col_size *= s
        matrix_data = []
        for k in range(num_operands):
            matrix = []
            for i0 in range(s0):
                row = []
                for i2 in range(col_size):
                    row.append(flat[i0 + s0 * k + s0 * num_operands * i2])
                matrix.append(row)
            matrix_data.append(matrix)
        return matrix_data

    def _snapshot_value(self, value):
        """Snapshot a mutable container so a later mutation of the source
        object cannot change an earlier Return/Push value. Return and Push
        store values, not variables: a future change to a variable must not
        affect what was already pushed."""
        if isinstance(value, (dict, list)):
            return copy.deepcopy(value)
        return value

    def _handle_return_statement(self, stripped, line_number):
        m_return = self._match_return_statement(stripped)
        if not m_return:
            raise SyntaxError(
                f"Invalid RETURN syntax at line {line_number}")
        value_expr = m_return.group(1).strip()
        self._append_console_output(value_expr, line_number)

    @contextmanager
    def _unpoisoned_reads(self):
        """Direct reads (Print) see through transient push poisons.

        A rejected Push latches #VALUE on the target so its listeners
        mirror the failure; a Print is a direct read, so the saved
        originals are shown for its duration and the errors go back
        afterwards.
        """
        owner = getattr(self, 'compiler', None) or self
        poisons = getattr(owner, '_transient_poisons', None)
        if not poisons:
            yield
            return
        restored = []
        for _poison_key, poison in list(poisons.items()):
            defining_scope, actual_key, original = poison
            restored.append(
                (defining_scope, actual_key,
                 defining_scope.variables.get(actual_key)))
            defining_scope.variables[actual_key] = original
        try:
            yield
        finally:
            for defining_scope, actual_key, latched in restored:
                defining_scope.variables[actual_key] = latched

    def _handle_print_statement(self, stripped, line_number):
        """Print <expr> — ambient console output (the module-level and
        top-level output statement). Pushes the evaluated value to the
        console channel."""
        m_print = re.match(r'^\s*print(?:\s+(.+))?\s*$', stripped, re.I)
        value_expr = (m_print.group(1) or '').strip() if m_print else ''
        if not value_expr:
            raise SyntaxError(
                f"Invalid PRINT syntax at line {line_number}. Use 'Print <expression>'")
        with self._unpoisoned_reads():
            self._append_console_output(value_expr, line_number, channel='_console')

    def _append_console_output(self, value_expr, line_number, channel='output'):
        """Evaluate an expression and append it to the given output channel
        (default output_values['output']). Print targets the ambient '_console'
        channel; Return (top level, transitional) uses 'output'."""
        try:
            resolver = getattr(self, 'compiler', None) or self
            pending = getattr(
                resolver, 'pending_assignments', {}) or {}
            if pending:
                pending_vars = [
                    pending_var for pending_var in list(pending.keys())
                    if not pending_var.startswith('__line_')
                ]
                for pending_var in pending_vars:
                    resolver._resolve_global_dependency(
                        pending_var, line_number,
                        target_scope=self.current_scope())
            values = self._evaluate_push_expression(
                value_expr, line_number)
            if any(v is None for v in values):
                pending = getattr(
                    resolver, 'pending_assignments', {}) or {}
                if pending:
                    pending_vars = [
                        pending_var for pending_var in list(pending.keys())
                        if not pending_var.startswith('__line_')
                    ]
                    for pending_var in pending_vars:
                        resolver._resolve_global_dependency(
                            pending_var, line_number,
                            target_scope=self.current_scope())
                    values = self._evaluate_push_expression(
                        value_expr, line_number)
            for value in values:
                self.output_values.setdefault(
                    channel, []).append(self._snapshot_value(value))
            if channel not in self.output_variables:
                self.output_variables.append(channel)
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate return expression at line {line_number}: {e}")

    def _handle_push_assignment_line(self, stripped, line_number):
        m_assign = self._match_push_assignment(stripped)
        if m_assign:
            target, value_expr = self._unpack_push_assignment(m_assign)
            self._handle_push_assignment(
                target, value_expr, line_number)
        else:
            raise SyntaxError(
                f"Invalid PUSH syntax at line {line_number}")

    def _handle_for_push_loop(self, lines, i, line, line_number):
        if line.count('"') % 2 == 1:
            return self._handle_for_push_multiline(lines, i, line, line_number)
        return self._handle_for_push_oneliner(i, line, line_number)

    def _handle_for_push_multiline(self, lines, i, line, line_number):
        complete_lines = [line]
        current_i = i
        current_line = line
        while not (current_line.strip().endswith(')') and current_line.count('"') % 2 == 0):
            current_i += 1
            if current_i >= len(lines):
                raise SyntaxError(
                    f"Unterminated string literal in FOR loop starting at line {line_number}")
            current_line = lines[current_i][0]
            complete_lines.append(current_line)

        executable_part = ' '.join(complete_lines)

        first_line = complete_lines[0]
        if ' do ' not in first_line.lower():
            return True, current_i + 1
        parts = first_line.split(' do ', 1)
        if len(parts) != 2:
            return True, current_i + 1
        for_part = parts[0].strip()
        executable_part = executable_part.replace(first_line, '').strip()
        if executable_part.startswith('do '):
            executable_part = executable_part[3:].strip()

        next_i = self._process_for_push_executable(
            for_part, executable_part, line_number, current_i + 1)
        return True, next_i

    def _handle_for_push_oneliner(self, i, line, line_number):
        parts = line.split(' do ', 1)
        if len(parts) != 2:
            return True, i + 1
        for_part = parts[0].strip()
        executable_part = parts[1].strip()
        next_i = self._process_for_push_executable(
            for_part, executable_part, line_number, i + 1)
        return True, next_i

    def _process_for_push_executable(self, for_part, executable_part, line_number, fallback_next_i):
        for_match = re.match(r'^\s*for\s+(.+?)$', for_part, re.I)
        if not for_match:
            return fallback_next_i
        var_defs = for_match.group(1).strip()

        range_match = re.match(
            r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$',
            var_defs, re.I)
        if not range_match:
            return fallback_next_i
        var_name, range_expr, step_str, index_var = range_match.groups()
        step = int(step_str) if step_str else 1

        if not self._process_for_push_range(
                var_name, range_expr, step, index_var,
                executable_part, line_number):
            return fallback_next_i
        return fallback_next_i

    def _process_for_push_range(self, var_name, range_expr, step, index_var, executable_part, line_number):
        if ' to ' not in range_expr:
            return False
        range_parts = range_expr.split(' to ')
        start = int(range_parts[0].strip())
        end = int(range_parts[1].strip())
        if step < 0:
            values = list(range(start, end - 1, step))
        else:
            values = list(range(start, end + 1, step))

        if self._has_push_action(executable_part):
            for idx, value in enumerate(values):
                self._process_for_push_iteration(
                    var_name, index_var, idx, value,
                    executable_part, line_number)
        return True

    def _process_for_push_iteration(self, var_name, index_var, idx, value, executable_part, line_number):
        self.push_scope(is_private=True, is_loop_scope=True)
        self.current_scope().define(var_name, value, 'number')
        if index_var:
            self.current_scope().define(index_var, idx + 1, 'number')

        return_match = self._match_return_statement(
            executable_part)
        push_match = self._match_push_assignment(
            executable_part)
        if return_match:
            value_expr = return_match.group(1).strip()
            values = self._evaluate_push_expression(
                value_expr, line_number)
            for value in values:
                self.output_values.setdefault(
                    'output', []).append(self._snapshot_value(value))
            if 'output' not in self.output_variables:
                self.output_variables.append(
                    'output')
        elif push_match:
            target, value_expr = self._unpack_push_assignment(push_match)
            self._handle_push_assignment(
                target, value_expr, line_number)
        else:
            raise SyntaxError(
                f"Unsupported push syntax in For loop at line {line_number}. Use 'Push var = value' syntax instead.")
        self.pop_scope()

    def _validate_no_top_level_return(self, lines):
        """Load-time check: ``Return`` is the call-value channel and is only
        valid inside functions and operations (named subprocesses). Any Return
        surviving in the top-level program stream is the legacy console-output
        pattern; reject it as a load error pointing at ``Print`` (the ambient
        console statement)."""
        for raw_line, line_number in lines:
            stripped = raw_line.strip()
            if re.match(r'return\b', stripped, re.I):
                raise SyntaxError(
                    f"Top-level RETURN is not allowed at line {line_number}: "
                    f"Return may only be used "
                    f"inside functions and operations. Use "
                    f"'Print <expression>' for console output.")

    def _run_setup(self, code, args):
        self._reset_state()
        self._global_guards_pre_evaluated = False
        self.needed_line_numbers = set()
        lines, label_lines, dim_lines = self._preprocess_code(code)
        if hasattr(self, 'compiler') and hasattr(self.compiler, '_extract_functions'):
            lines, label_lines, dim_lines = self.compiler._extract_functions(
                lines, label_lines, dim_lines)
            # Mirror extracted function tables onto executor for downstream checks
            self.functions = getattr(self.compiler, 'functions', {}) or {}
            self.subprocesses = getattr(
                self.compiler, 'subprocesses', {}) or {}
        else:
            lines, label_lines, dim_lines = self._extract_functions(
                lines, label_lines, dim_lines)
        # Keep compiler/executor function tables in sync for expression evaluation
        if hasattr(self, 'compiler'):
            self.functions = getattr(self, 'functions', {}) or {}
            self.subprocesses = getattr(self, 'subprocesses', {}) or {}
            self.compiler.functions = self.functions
            self.compiler.subprocesses = self.subprocesses
        # Return is only legal inside function/operation bodies, which are
        # extracted above; whatever remains is the top-level program stream.
        if not getattr(self, '_is_operation_runner', False):
            self._validate_no_top_level_return(lines)
        self.for_guard_conditions = {}
        normalized_lines = []
        for raw_line, line_number in lines:
            guard_match = re.match(
                r'^\s*for\s+(.+?)\s+do\s+when\s+(.+?)\s+do\s*$',
                raw_line, re.I)
            if guard_match:
                var_defs = guard_match.group(1).strip()
                condition = guard_match.group(2).strip()
                self.for_guard_conditions[line_number] = condition
                raw_line = f"For {var_defs} do"
            normalized_lines.append((raw_line, line_number))
        return normalized_lines, label_lines, dim_lines

    def _run_prepare_execution(self, lines, label_lines, dim_lines, args):
        self._process_declarations_and_labels(lines, label_lines, dim_lines)
        # Materialize UnitSource constants into scope so references like
        # SILength.inch resolve during evaluation.
        if hasattr(self, '_materialize_unit_source_constants'):
            self._materialize_unit_source_constants()
        if hasattr(self, '_register_top_level_converts'):
            self._register_top_level_converts()
        # Store the root scope after declarations
        self.root_scope = self.current_scope()

        # Collect input/output variables and set input values
        self.collect_input_output_variables()
        prompt_missing = getattr(self, 'prompt_missing_inputs', False)
        self.set_input_values(args, prompt_missing=prompt_missing)
        # Resolve REQUIRED capabilities (grant/deny/prompt). Capability handles
        # are always defined - granted with the chosen parameters, or bound to
        # the sticky #PERM error when denied - so the rest of the dependency
        # network treats them as resolved and execution continues either way.
        self._resolve_require_stage(
            can_prompt=getattr(self, 'prompt_missing_requires', False))
        # Resolve global pending declarations before first INIT materialization.
        self._resolve_ready_pending_vars(None)
        # Materialize INIT values early so they capture pre-execution state
        self._materialize_inits()

        # Pre-scan blocks to map structure before processing
        self.control_flow.pre_scan_blocks(lines)
        guard_entries = self.control_flow.identify_global_guard_lines(lines)
        self.global_guard_line_numbers = {
            entry['line_number'] for entry in guard_entries}
        self.global_guard_entries = guard_entries
        for_entries = self.control_flow.identify_global_for_declarations(lines)
        self.global_for_entries = for_entries
        self.global_for_line_numbers = {
            entry['line_number'] for entry in for_entries}
        self._build_dependency_network(lines, guard_entries)
        if not self._evaluate_global_guards_pre_execution(guard_entries):
            return None
        self._execute_global_for_loops(lines)
        # A binding is a binding: root-level plain/`Let` bindings apply
        # before the main loop, so a Let block or For-do header written
        # before them still observes the live name.
        self._hoist_root_bindings(lines)
        # Wave-model state: the two-phase engine collects the global bodyless
        # producers/rules/guards/effects on first encounter in the main loop.
        self._wave_state_reset()
        self.needed_line_numbers = self._determine_needed_lines()
        return guard_entries

    # ------------------------------------------------------------------
    # Required-capability resolution (Require / Grant)
    # ------------------------------------------------------------------

    def _resolve_require_stage(self, can_prompt=False):
        """Bind every required capability: granted capabilities hold the chosen
        resource parameters, denied ones hold the sticky #PERM error value.

        Requirements are never optional and never abort execution; a denied or
        default-deny capability simply taints every use with #PERM.

        The resolution policy lives in ``permissions.RequirementResolver``; this
        host supplies its evaluation services and defines the resulting bindings
        in scope.
        """
        from permissions import RequirementResolver
        grant_map, bindings = RequirementResolver(self).resolve(
            self.requirements, self.grants, can_prompt=can_prompt)
        self.grants = grant_map
        for var_name, value, vtype, line_number in bindings:
            # Inner handles of a user resource instance bind as members of the
            # owning handle (``p.ticks``) instead of global names: the builtin
            # can be parameterised per owning resource and no two handles
            # collide in scope. The owner is defined earlier (parent first).
            # Denied handles still attach, carrying the sticky #PERM value.
            if '.' in var_name:
                parent_name, _, member_name = var_name.partition('.')
                try:
                    parent_val = self.current_scope().get(parent_name)
                except NameError:
                    parent_val = None
                if parent_val is not None and isinstance(parent_val, dict):
                    parent_val[str(member_name)] = value
                    continue
            self.current_scope().define(
                var_name, value, vtype, {}, is_uninitialized=False,
                line_number=line_number)

    def _evaluate_push_expression(self, value_expr, line_number, expected_unit=None):
        """Evaluate a PUSH expression, expanding generator outputs into a sequence."""
        base_scope = self.current_scope().get_evaluation_scope()
        func_defs = getattr(self, 'functions', {}) or {}
        subprocess_defs = getattr(self, 'subprocesses', {}) or {}

        call_pattern = re.compile(r'([A-Za-z][A-Za-z0-9_.]*)\s*\(')

        def _find_calls(expr):
            calls = []
            idx = 0
            while True:
                m = call_pattern.search(expr, idx)
                if not m:
                    break
                name = m.group(1)
                paren_start = m.end() - 1
                depth = 0
                in_quote = None
                end_idx = -1
                for pos in range(paren_start, len(expr)):
                    ch = expr[pos]
                    if in_quote:
                        if ch == in_quote and expr[pos - 1] != '\\':
                            in_quote = None
                        continue
                    if ch in ('"', "'"):
                        in_quote = ch
                        continue
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                        if depth == 0:
                            end_idx = pos
                            break
                if end_idx == -1:
                    idx = m.end()
                    continue
                args_str = expr[paren_start + 1:end_idx]
                calls.append({
                    'name': name,
                    'start': m.start(1),
                    'end': end_idx,
                    'args': args_str
                })
                idx = end_idx + 1
            return calls

        def _pick_sequence(def_entry, outputs_map):
            if not outputs_map:
                return []
            seq = None
            target_names = def_entry.get('outputs') if def_entry else None
            if target_names:
                for out_name in target_names:
                    if out_name in outputs_map:
                        seq = outputs_map[out_name]
                        break
            if seq is None:
                seq = outputs_map.get('output')
            if seq is None and outputs_map:
                seq = next(iter(outputs_map.values()))
            if seq is None:
                return []
            return list(seq) if isinstance(seq, list) else ([seq] if seq is not None else [])

        raw_calls = _find_calls(value_expr)
        call_entries = []
        for call in raw_calls:
            lname = call['name'].lower()
            if lname in func_defs or lname in subprocess_defs:
                call_entries.append(call)

        if not call_entries:
            return [self.expr_evaluator.eval_or_eval_array(
                value_expr, base_scope, line_number, expected_unit=expected_unit)]

        replacements = []
        sequences = []
        for idx, call in enumerate(call_entries):
            placeholder = f"__gen{idx}__"
            args_parts = self._split_call_arguments(call['args'])
            # Evaluate each argument; if an argument yields multiple values, treat it as a sequence for zipping
            eval_args = []
            seq_lengths = []
            has_sequence_arg = False
            for a in args_parts:
                try:
                    vals = self._evaluate_push_expression(a, line_number, expected_unit)
                except Exception:
                    vals = [self.expr_evaluator.eval_or_eval_array(
                        a, base_scope, line_number, expected_unit=expected_unit)]
                if isinstance(vals, list) and len(vals) > 1:
                    has_sequence_arg = True
                    eval_args.append(list(vals))
                    seq_lengths.append(len(vals))
                else:
                    eval_args.append(
                        vals[0] if isinstance(vals, list) else vals)
            def_entry = func_defs.get(call['name'].lower()) if call['name'].lower(
            ) in func_defs else subprocess_defs.get(call['name'].lower(), {})
            member_of = def_entry.get('member_of') if def_entry else None

            if has_sequence_arg:
                iters = min(seq_lengths) if seq_lengths else 0
                seq = []
                for iter_idx in range(iters):
                    call_args = []
                    for v in eval_args:
                        call_args.append(
                            v[iter_idx] if isinstance(v, list) else v)
                    if call['name'].lower() in func_defs:
                        outputs_map = self.call_function(
                            call['name'], call_args, instance_type=member_of,
                            collect_all=True, line_number=line_number) or {}
                        seq.extend(_pick_sequence(def_entry, outputs_map))
                    else:
                        outputs_map = self.call_subprocess(
                            call['name'], call_args, collect_all=True, line_number=line_number) or {}
                        picked = _pick_sequence(def_entry, outputs_map)
                        if picked:
                            seq.extend(picked)
                        else:
                            try:
                                sp_result = self.call_subprocess(
                                    call['name'], call_args, collect_all=False, line_number=line_number)
                                seq.append(sp_result)
                            except Exception:
                                pass
            else:
                live_route = None
                route_fn = getattr(self, '_route_live_call', None)
                if route_fn is not None:
                    live_route = route_fn(
                        def_entry, args_parts,
                        lambda raw, _parts=args_parts, _vals=eval_args:
                        _vals[_parts.index(raw)] if raw in _parts else None,
                        line_number)
                if live_route is not None:
                    input_vals, live_sources, live_bindings = live_route
                    if call['name'].lower() in func_defs:
                        outputs_map = self.call_function(
                            call['name'], input_vals, instance_type=member_of, collect_all=True,
                            live_input_sources=live_sources,
                            live_output_bindings=live_bindings,
                            caller_scope=self.current_scope(),
                            line_number=line_number) or {}
                        seq = _pick_sequence(def_entry, outputs_map)
                    else:
                        try:
                            outputs_map = self.call_subprocess(
                                call['name'], input_vals, collect_all=True, line_number=line_number,
                                live_input_sources=live_sources,
                                live_output_bindings=live_bindings,
                                caller_scope=self.current_scope()) or {}
                        except Exception:
                            outputs_map = {}
                        seq = _pick_sequence(def_entry, outputs_map)
                        if not seq:
                            try:
                                sp_result = self.call_subprocess(
                                    call['name'], input_vals, collect_all=False, line_number=line_number,
                                    live_input_sources=live_sources,
                                    live_output_bindings=live_bindings,
                                    caller_scope=self.current_scope())
                                seq = [sp_result]
                            except Exception:
                                seq = []
                elif call['name'].lower() in func_defs:
                    outputs_map = self.call_function(
                        call['name'], eval_args, instance_type=member_of,
                        collect_all=True, line_number=line_number) or {}
                    seq = _pick_sequence(def_entry, outputs_map)
                else:
                    outputs_map = self.call_subprocess(
                        call['name'], eval_args, collect_all=True, line_number=line_number) or {}
                    seq = _pick_sequence(def_entry, outputs_map)
                    if not seq:
                        try:
                            sp_result = self.call_subprocess(
                                call['name'], eval_args, collect_all=False, line_number=line_number)
                            seq = [sp_result]
                        except Exception:
                            seq = []
            sequences.append((placeholder, seq))
            replacements.append((call['start'], call['end'], placeholder))

        replacements.sort(key=lambda x: x[0])
        rebuilt = []
        last = 0
        for start, end, placeholder in replacements:
            rebuilt.append(value_expr[last:start])
            rebuilt.append(placeholder)
            last = end + 1
        rebuilt.append(value_expr[last:])
        template_expr = "".join(rebuilt)

        if not sequences:
            return [self.expr_evaluator.eval_or_eval_array(
                template_expr, base_scope, line_number)]

        max_iters = min(len(seq) for _, seq in sequences) if sequences else 1
        results = []
        for idx in range(max_iters):
            iter_scope = dict(base_scope)
            missing = False
            for placeholder, seq in sequences:
                if idx >= len(seq):
                    missing = True
                    break
                iter_scope[placeholder] = seq[idx]
            if missing:
                break
            results.append(self.expr_evaluator.eval_or_eval_array(
                template_expr, iter_scope, line_number))
        return results

    def _notify_push_listeners(self, target, line_number):
        """Handle a void Push (``Push <target>`` with no expression).

        The stored value is never touched. Whatever the target already holds
        is re-published to its listeners - derived cells recompute, When
        blocks whose dependencies include it re-run - which is the same
        notification a value Push sends, minus the write.
        """
        owner = getattr(self, 'compiler', None) or self
        if target.startswith('['):
            cell_map = getattr(owner, '_cell_var_map', None) or {}
            bound_var = cell_map.get(target)
            if bound_var is None:
                to_index = getattr(owner, '_to_index', None)
                if to_index is not None:
                    try:
                        bound_var = cell_map.get(
                            to_index(target.strip('[]')))
                    except Exception:
                        bound_var = None
            if not bound_var:
                try:
                    cell_value = self.array_handler.lookup_cell(
                        target.strip('[]'), line_number)
                except Exception:
                    return
                if hasattr(owner, '_notify_cell_changed'):
                    owner._notify_cell_changed(target, cell_value)
                return
            target = bound_var
        elif re.search(r'[\(\{!]', target):
            indexed = re.match(r'^[\w_]+', target)
            if not indexed:
                raise SyntaxError(
                    f"Invalid Push target '{target}' at line {line_number}")
            target = indexed.group(0)
        else:
            target = target.split('.')[0].strip()
        if self.current_scope().get_defining_scope(target) is None:
            raise NameError(
                f"Cannot Push to '{target}': variable is not defined at line {line_number}")
        value = self.current_scope().get(target)
        if hasattr(owner, '_notify_var_changed'):
            owner._notify_var_changed(target, value)
        self._enqueue_push(target, value)
        self._process_when_triggers()

    def _handle_push_assignment(self, target, value_expr, line_number):
        if not target:
            # Only reachable from an '=' with no right-hand side: the '=' is
            # extra, and there is nothing here to Push.
            raise SyntaxError(
                f"Invalid PUSH syntax at line {line_number}: '=' has no value. "
                f"Use 'Push <target> = <expression>', or write "
                f"'Push <target>' with no '=' to re-notify listeners")
        # A Push to the rejected target is what recovers a value a rejected
        # Push left reading the error: clear only that target's outstanding
        # poison before this Push reads or notifies anything
        # (compiler._clear_transient_poisons).  A Push to anything else is a
        # no-op on the poisoned readers.
        owner = getattr(self, 'compiler', None) or self
        boundary = getattr(owner, '_wave_push_boundary', None)
        if boundary is not None:
            boundary(target)
        clear = getattr(owner, '_clear_transient_poisons', None)
        if clear is not None:
            clear(target)
        if value_expr is None:
            # A void Push: the line names a target with no '=' expression at
            # all ('Push x').  Nothing is stored - the target keeps the value
            # it already holds - but every listener of it is re-notified with
            # that value, so dependent cells and When blocks recompute.
            self._notify_push_listeners(target, line_number)
            return
        import sys as _sys
        # Key type variables are immutable: Push to a stored keytype var is forbidden
        try:
            compiler = getattr(self, 'compiler', None) or self
            if hasattr(compiler, '_is_key_type') and hasattr(compiler, 'current_scope'):
                dest_type = None
                is_output_dest = False
                try:
                    def_scope = self.current_scope().get_defining_scope(target)
                    if def_scope:
                        var_key = def_scope._get_case_insensitive_key(target, def_scope.types)
                        dest_type = def_scope.types.get(var_key) if var_key else None
                        con_key = def_scope._get_case_insensitive_key(target, def_scope.constraints)
                        con = def_scope.constraints.get(con_key, {}) if con_key else {}
                        is_output_dest = bool(con.get('output')) or def_scope.is_output(target)
                except Exception:
                    pass
                if dest_type and compiler._is_key_type(dest_type) and not is_output_dest:
                    raise ConstraintError(TYPE_ERROR, f"Cannot modify key type '{dest_type}' at line {line_number}")
        except ConstraintError:
            raise
        except Exception:
            pass
        # grid field is read-only outside type constructors/builders
        if '.grid' in target.lower():
            ctx = getattr(self, '_context_grid_stack', None) or getattr(getattr(self, 'compiler', None), '_context_grid_stack', None)
            if not ctx:
                raise PermissionError(
                    f"Field 'grid' is read-only outside type at line {line_number}")
        if target.startswith('['):
            assignment_line = f"{target} := {value_expr}"
            # A Push into a cell is mutation (last value wins), not a new
            # binding: flag it so the cell's equality-binding registry is
            # left alone.
            owner = getattr(self, 'compiler', None) or self
            was_push = getattr(owner, '_in_push_cell', False)
            owner._in_push_cell = True
            try:
                self.array_handler.evaluate_line_with_assignment(
                    assignment_line, line_number,
                    self.current_scope().get_evaluation_scope())
            finally:
                owner._in_push_cell = was_push
            return
        if re.search(r'[\(\{!]', target):
            self._assign_indexed_target(target, value_expr, line_number)
            return
        member_path_match = re.match(r'^[\w_]+(?:\.[\w_]+)+$', target)
        # A Push to a plain variable that does not exist is an error: variables
        # must be declared (or otherwise bound) before they can be pushed.
        if not member_path_match:
            try:
                if self.current_scope().get_defining_scope(target) is None:
                    raise NameError(
                        f"Cannot Push to '{target}': variable is not defined at line {line_number}")
            except NameError:
                raise
            except Exception:
                pass
        # Get target's declared unit for LHS-informed conversion (Push/For etc.)
        target_unit = None
        try:
            t_scope = self.current_scope().get_defining_scope(target)
            if t_scope:
                ak = t_scope._get_case_insensitive_key(target, t_scope.constraints)
                if ak:
                    target_unit = t_scope.constraints.get(ak, {}).get('unit')
                else:
                    for k in t_scope.constraints:
                        if k.lower() == target.lower():
                            target_unit = t_scope.constraints[k].get('unit')
                            break
        except Exception:
            pass
        if not member_path_match and t_scope is None:
            t_scope = self.current_scope().get_defining_scope(target)
        if not member_path_match and t_scope is not None:
            # Materialize a declared-but-uninitialized target's 'init' default
            # before evaluating the RHS: the RHS reads run against the merged
            # evaluation-scope dict (which stores the raw None), so an init
            # value like `Push count = count + 1` would otherwise evaluate
            # None + 1. The scope-chain get() applies lazy init and stores it.
            ak = t_scope._get_case_insensitive_key(target, t_scope.variables)
            if ak and t_scope.variables.get(ak) is None:
                try:
                    init_constraints = t_scope.constraints.get(ak, {}) or {}
                    if init_constraints.get('init') is not None:
                        t_scope.get(target)
                except Exception:
                    pass
        try:
            if self._is_bare_subprocess_call(value_expr):
                values = [self.expr_evaluator.eval_or_eval_array(
                    str(value_expr),
                    self.current_scope().get_evaluation_scope(),
                    line_number,
                    expected_unit=target_unit,
                )]
            else:
                values = self._evaluate_push_expression(
                    value_expr, line_number, target_unit)
            global_scope = self.get_global_scope()
            for value in values:
                value = self._snapshot_value(value)
                if member_path_match:
                    self._update_member_path_target(
                        target, value, line_number)
                    continue
                self._set_var_value(target, value, line_number)
                collecting_via_compiler = hasattr(
                    self, 'compiler') and self.compiler is not None
                if (target.lower() in global_scope.output_variables) and not collecting_via_compiler:
                    self.output_values.setdefault(
                        target.lower(), []).append(value)
                # Live outputs stream their value to the caller's binding
                # immediately (write-through), not only when the call returns.
                self._maybe_publish_live_output(target, value, line_number)
                self._enqueue_push(target, value)
                self._process_when_triggers()
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate push expression at line {line_number}: {e}")

    def _update_member_path_target(self, target, value, line_number):
        # grid field is read-only outside type
        if '.grid' in target.lower():
            ctx = getattr(self, '_context_grid_stack', None) or getattr(getattr(self, 'compiler', None), '_context_grid_stack', None)
            if not ctx:
                raise PermissionError(
                    f"Field 'grid' is read-only outside type at line {line_number}")
        path_parts = [part.strip() for part in target.split('.') if part.strip()]
        if len(path_parts) < 2:
            return False

        root_name = path_parts[0]
        member_parts = path_parts[1:]
        root_value, current_obj, _owner_path, actual_field, field_name = \
            self._walk_member_path(root_name, path_parts, line_number)

        existing = current_obj.get(actual_field)
        if isinstance(existing, list):
            existing.append(value)
        else:
            # The root is equality-bound ('a = new Box'): a Push into
            # any field must not rewrite the constant.  Reject the
            # write, keep the stored value, and emit a transient
            # #VALUE to the root's listeners -- the member-Push
            # counterpart of the check in _set_var_value (Test 331/337).
            if self._member_push_conflicts_with_constant(
                    root_name, root_value, member_parts, value,
                    line_number):
                owner = getattr(self, 'compiler', None) or self
                owner._propagate_transient(root_name, VALUE_ERROR)
                ctx = getattr(
                    self, '_context_grid_stack', None) or getattr(
                        owner, '_context_grid_stack', None)
                if ctx:
                    raise ValueError(
                        f"Cannot modify constant field at line {line_number}")
                return True
            # Otherwise the field's own constraints (inequality, set,
            # range, type) apply, exactly as a constrained variable
            # does: the error value is stored in the field, so reads of
            # the whole object propagate it.
            violation = self._validate_object_field_value(
                current_obj, actual_field, value, line_number)
            if violation is not None:
                current_obj[actual_field] = violation
                self.current_scope().update(
                    root_name, root_value, line_number)
                self._enqueue_push(root_name, root_value)
                self._process_when_triggers()
                return True
            current_obj[actual_field] = value

        self.current_scope().update(root_name, root_value, line_number)
        self._enqueue_push(root_name, root_value)
        self._process_when_triggers()
        return True

    def _walk_member_path(self, root_name, path_parts, line_number):
        """Resolve a dotted member path down to the object that holds its leaf.

        Returns ``(root_value, leaf_parent, owner_path, leaf_key, leaf_name)``.
        Every part before the leaf must already exist and hold an object, and
        hidden or immutable fields are rejected at every level including the
        leaf.  Callers own the write policy for the leaf itself: a Push creates
        a field the instance does not carry yet, a Let does not.
        """
        if len(path_parts) < 2:
            raise ValueError(
                f"Member path needs at least one field at line {line_number}")
        # Resolve the root through its defining scope: a private block scope
        # (a Let block) does not chain reads to its parent, but the binding
        # still has to be reachable from inside the block.
        defining = self.current_scope().get_defining_scope(root_name)
        root_key = None
        if defining is not None:
            root_key = defining._get_case_insensitive_key(
                root_name, defining.variables)
        if defining is None or root_key is None:
            root_value = self.current_scope().get(root_name)
        else:
            root_value = defining.variables.get(root_key)
        if not isinstance(root_value, dict):
            raise ValueError(
                f"'{root_name}' is not an object at line {line_number}")

        current_obj = root_value
        owner_path = root_name
        member_parts = path_parts[1:]
        last_index = len(member_parts) - 1
        for idx, field_name in enumerate(member_parts):
            public_keys = {
                k.lower(): k for k in current_obj.keys()
                if not str(k).startswith('_')
            }
            actual_field = public_keys.get(field_name.lower(), field_name)
            if hasattr(self, '_is_hidden_field') and self._is_hidden_field(current_obj, actual_field):
                if not getattr(self, '_allow_hidden_field_access', False):
                    raise PermissionError(
                        f"Hidden field '{field_name}' is not accessible at line {line_number}")
            immutable_fields = {
                f.lower() for f in current_obj.get('_immutable_fields', set())
            }
            if actual_field.lower() in immutable_fields:
                # Allow initialisation of a nulled key (None) via builder following Copy
                from units import is_error_value
                existing_val = current_obj.get(actual_field)
                if existing_val is not None and not is_error_value(existing_val):
                    raise ValueError(
                        f"Field '{field_name}' of '{owner_path}' is immutable at line {line_number}")

            if idx == last_index:
                return root_value, current_obj, owner_path, actual_field, field_name

            if actual_field not in current_obj:
                raise ValueError(
                    f"Field '{field_name}' is not defined on '{owner_path}' at line {line_number}")
            next_obj = current_obj.get(actual_field)
            if not isinstance(next_obj, dict):
                raise ValueError(
                    f"'{owner_path}.{actual_field}' is not an object at line {line_number}")
            current_obj = next_obj
            owner_path = f"{owner_path}.{actual_field}"
        raise ValueError(
            f"Member path needs at least one field at line {line_number}")

    def _assign_indexed_target(self, target, value_expr, line_number):
        paren_match = re.match(r'^([\w_]+)\s*\(([^)]+)\)$', target)
        brace_match = re.match(r'^([\w_]+)\s*\{([^}]+)\}$', target)
        bang_match = re.match(r'^([\w_]+)\s*!\[([^\]]+)\]$', target)
        if not (paren_match or brace_match or bang_match):
            raise SyntaxError(
                f"Invalid indexed assignment target: '{target}' at line {line_number}")
        scope_dict = self.current_scope().get_evaluation_scope()
        if paren_match:
            var_name, index_expr = paren_match.groups()
            index_value = self.expr_evaluator.eval_expr(
                index_expr, scope_dict, line_number)
            if isinstance(index_value, float) and index_value.is_integer():
                index_value = int(index_value)
            indices = [index_value - 1]
        elif bang_match:
            var_name, index_expr = bang_match.groups()
            try:
                index_value = self.expr_evaluator.eval_expr(
                    index_expr, scope_dict, line_number)
                if isinstance(index_value, float) and index_value.is_integer():
                    index_value = int(index_value)
                indices = [index_value - 1]
            except Exception:
                try:
                    indices = [i - 1 for i in parse_address(index_expr)]
                except Exception:
                    indices = self.array_handler.cell_ref_to_indices(
                        index_expr, line_number)
        else:
            var_name, indices_str = brace_match.groups()
            index_exprs = [idx.strip()
                           for idx in indices_str.split(',') if idx.strip()]
            indices = []
            for index_expr in index_exprs:
                index_value = self.expr_evaluator.eval_expr(
                    index_expr, scope_dict, line_number)
                if isinstance(index_value, float) and index_value.is_integer():
                    index_value = int(index_value)
                indices.append(index_value - 1)
        value = self.expr_evaluator.eval_or_eval_array(
            value_expr, scope_dict, line_number)
        defining_scope = self.current_scope().get_defining_scope(var_name)
        if not defining_scope:
            raise NameError(
                f"Array variable '{var_name}' not defined at line {line_number}")
        if (getattr(self.compiler, '_outer_scope_read_only', False)
                and self.compiler._is_outer_scope(defining_scope)):
            raise RuntimeError(
                f"Cannot assign to '{var_name}': variables in an outer scope "
                f"are read-only inside a function at line {line_number}")
        actual_key = defining_scope._get_case_insensitive_key(
            var_name, defining_scope.variables) or var_name
        constraints = defining_scope.constraints.get(actual_key, {})
        if not constraints or not constraints.get('dim'):
            rank = infer_array_rank(defining_scope.variables.get(actual_key))
            if rank is not None and len(indices) != rank:
                raise ValueError(
                    f"Expected {rank} indices for array variable '{var_name}', got {len(indices)} at line {line_number}")
        indices = self._apply_dim_base_offsets(
            var_name, indices, line_number)
        arr = defining_scope.variables.get(actual_key)
        if isinstance(arr, dict) and 'grid' in arr:
            updated_grid = self.array_handler.set_array_element(
                arr['grid'], indices, value, line_number)
            arr['grid'] = updated_grid
        else:
            updated_array = self.array_handler.set_array_element(
                arr, indices, value, line_number)
            defining_scope.variables[actual_key] = updated_array
        defining_scope.uninitialized.discard(actual_key)
        self._notify_var_changed(actual_key, defining_scope.variables.get(actual_key))
        return

    def _print_outputs(self):
        """Print all output variables as required by Grid language"""
        # Check if we have a compiler reference with output variables
        if hasattr(self, 'compiler') and hasattr(self.compiler, 'output_variables'):
            # Only use compiler's output variables if we don't have any of our own
            if not self.output_variables:
                self.output_variables = self.compiler.output_variables
        try:

            # Get output values from self.output_values (the executor's output_values)
            output_values = self.output_values

            for output_var in self.output_variables:
                condition_result = output_var in output_values and output_values[output_var]
                if condition_result:
                    # Print all collected values for this output variable
                    for value in output_values[output_var]:
                        display = format_display_value(value)
                        if output_var in ('output', '_console'):
                            print(display)
                        else:
                            print(f"{output_var}: {display}")
        except Exception as e:
            import traceback
            traceback.print_exc()

    def _materialize_inits(self):
        """Evaluate INIT constraints for uninitialized vars when dependencies are satisfied."""
        try:
            scope = self.current_scope()
        except Exception:
            return
        deferred = getattr(self, '_deferred_output_inits', None) or getattr(
            getattr(self, 'compiler', None), '_deferred_output_inits', None) or {}
        seen_scopes = set()
        while scope and id(scope) not in seen_scopes:
            seen_scopes.add(id(scope))
            for var_name, constraints in list(scope.constraints.items()):
                if not constraints or 'init' not in constraints:
                    continue
                if not scope.is_uninitialized(var_name):
                    continue
                if var_name.lower() in deferred:
                    continue
                init_expr = constraints.get('init')
                init_line = constraints.get('init_line')
                # A wave-claimed Init defers its assignment to Phase 3 (as a
                # push effect); do not materialize it lazily here.
                if init_line in (getattr(self, '_wave_init_lines', None)
                                 or set()):
                    continue
                deps = self._extract_dependencies_from_expression(
                    str(init_expr))
                if any(self.has_unresolved_dependency(dep, scope=scope) for dep in deps):
                    continue
                try:
                    val = self.expr_evaluator.eval_or_eval_array(
                        str(init_expr), scope.get_full_scope(), init_line)
                    type_key = scope._get_case_insensitive_key(
                        var_name, scope.types) or var_name
                    var_type = scope.types.get(type_key)
                    self._apply_init_values(
                        var_name,
                        var_type,
                        constraints,
                        [val],
                        init_line,
                        scope=scope)
                except Exception as exc:
                    pass
            for var_name, constraints in list(scope.constraints.items()):
                if not constraints or 'dim' not in constraints:
                    continue
                if not scope.is_uninitialized(var_name):
                    continue
                dim_spec = constraints.get('dim')
                if isinstance(dim_spec, dict) and 'dims' in dim_spec:
                    dims = dim_spec['dims']
                elif isinstance(dim_spec, list):
                    dims = dim_spec
                else:
                    continue
                if not any(isinstance(size_spec, str) for _, size_spec in dims):
                    continue
                deps = set()
                for _, size_spec in dims:
                    if isinstance(size_spec, str):
                        deps |= self._extract_dependencies_from_expression(
                            size_spec)
                if any(self.has_unresolved_dependency(dep, scope=scope) for dep in deps):
                    continue
                try:
                    resolved_dims = []
                    for name, size_spec in dims:
                        if isinstance(size_spec, str):
                            size_val = self.expr_evaluator.eval_expr(
                                size_spec, scope.get_full_scope(), None)
                            if isinstance(size_val, bool):
                                raise ValueError(
                                    "dimension size is not numeric")
                            if isinstance(size_val, float) and not size_val.is_integer():
                                raise ValueError(
                                    "dimension size is not an integer")
                            if not isinstance(size_val, (int, float)):
                                raise ValueError(
                                    "dimension size is not numeric")
                            size_spec = int(size_val)
                        resolved_dims.append((name, size_spec))
                    if any(self.array_handler._is_unbounded_size_spec(size_spec)
                           for _, size_spec in resolved_dims):
                        continue
                    shape = []
                    for _, size_spec in resolved_dims:
                        if isinstance(size_spec, tuple):
                            start, end = size_spec
                            size = end - start + 1
                        else:
                            size = size_spec
                        shape.append(size)
                    actual_key = scope._get_case_insensitive_key(
                        var_name, scope.types) or var_name
                    var_type = scope.types.get(actual_key)
                    if var_type and var_type.lower() in getattr(self, 'types_defined', {}):
                        value = self.array_handler.create_object_array(
                            shape, None, None)
                    else:
                        pa_type = 'number' if var_type in (
                            'number', 'array') else 'text'
                        value = self.array_handler.create_array(
                            shape, None, pa_type, None, template=True)
                    self.dimensions[var_name] = resolved_dims
                    if actual_key in scope.constraints:
                        scope.constraints[actual_key]['dim'] = resolved_dims
                    elif var_name in scope.constraints:
                        scope.constraints[var_name]['dim'] = resolved_dims
                    scope.update(var_name, value)
                except Exception as exc:
                    pass
            scope = scope.parent if (getattr(scope, 'parent', None)
                                     and not scope.is_private
                                     and getattr(scope.parent, 'compiler',
                                                 None) is scope.compiler) else None

    def _deferred_dependency_unresolved(self, dep, scope):
        """Whether a deferred assignment must keep waiting on ``dep``.

        A declared variable that is merely uninitialized does not block
        processing: reading it yields the sticky ``#N/A`` error value. Only
        genuinely missing names, variables with an outstanding late
        assignment, and dependencies still being computed keep a line pending.
        """
        if not dep:
            return False
        if re.match(r'^[A-Za-z]+\d+$', dep):
            if self._grid_cell_is_set(dep):
                return False
            # Unset cells only stay pending while they are truly missing; with
            # a grid default they are readable and never block.
            return not self._grid_has_default()
        if dep.lower() in (self.functions or {}):
            return False
        if dep.lower() in self.undefined_dependencies:
            return True
        if dep in self.pending_assignments or dep in getattr(scope, 'pending_assignments', {}):
            return True
        if not scope.get_defining_scope(dep):
            return True
        if scope.is_uninitialized(dep):
            return False
        return self.has_unresolved_dependency(dep, scope=scope)

    def _process_deferred_assignments(self):
        """Process any deferred assignments stored with __line_ keys."""

        # First, materialize any INIT-constrained variables whose dependencies are now resolved
        self._materialize_inits()

        # First, opportunistically evaluate any simple assignments in the dependency graph
        # whose dependencies are now satisfied but which were skipped due to earlier deferral.
        if getattr(self, 'dependency_graph', {}).get('nodes'):
            for node in sorted(self.dependency_graph['nodes'], key=lambda n: n.get('line', 0)):
                defines = node.get('defines') or []
                if len(defines) != 1:
                    continue
                var_name = defines[0]
                if not var_name or var_name.startswith('['):
                    continue
                current_scope = self.current_scope()
                if current_scope.get_defining_scope(var_name):
                    continue
                if node.get('depth', 0) > 0:
                    # Nodes inside a block body are executed conditionally by
                    # the block machinery at runtime; materializing them here
                    # would ignore If/When gates (e.g. a LET under a false If).
                    continue
                data = node.get('data') or {}
                expr = data.get('expression')
                if not expr:
                    continue
                deps = node.get('depends_on') or []
                unresolved = any(self.has_unresolved_dependency(
                    dep, scope=current_scope) for dep in deps)
                if unresolved:
                    continue
                try:
                    value = self.expr_evaluator.eval_or_eval_array(
                        expr, current_scope.get_evaluation_scope(), node.get('line'))
                    defining_scope = current_scope.get_defining_scope(
                        var_name) or current_scope
                    defining_scope.define(
                        var_name, value, None, {}, is_uninitialized=False)
                    self.mark_dependency_resolved(var_name)
                except Exception as exc:
                    missing = self.extract_missing_dependencies(exc)
                    for dep in missing:
                        self.mark_dependency_missing(dep)
                    continue

        deferred_assignments = []
        deferred_ifs = []
        for key, assignment in self.pending_assignments.items():
            if key.startswith('__line_'):
                line_content, line_number, deps = assignment[:3]
                deferred_assignments.append((line_number, line_content, deps))
            elif key.startswith('__if_line_'):
                line_content, line_number, deps = assignment[:3]
                deferred_ifs.append((line_number, line_content, deps))

        # Sort by line number to process in order
        deferred_assignments.sort(key=lambda x: x[0])
        deferred_ifs.sort(key=lambda x: x[0])

        # Retry deferred IF blocks once dependencies are available
        for line_number, line_content, deps in deferred_ifs:
            try:
                scope = self.current_scope()
                unresolved = any(
                    self._deferred_dependency_unresolved(dep, scope=scope)
                    for dep in deps)
                if unresolved:
                    continue
                # Locate the index of the IF line in the cached preprocessed
                # lines. Match content as well: a renumbered constructor IF
                # can share a line number with an unrelated main IF.
                line_index = next(
                    (idx for idx, (txt, ln) in enumerate(
                        self.control_flow._preprocessed_lines)
                     if ln == line_number and txt == line_content),
                    None)
                if line_index is None:
                    continue
                if_block = self.control_flow._match_global_if_block(
                    line_number, self.control_flow._preprocessed_lines,
                    line_index)
                if if_block and len(if_block.get('clauses', [])) > 1:
                    self.control_flow._process_if_statement_rich(
                        line_content, line_number, self.control_flow._preprocessed_lines, line_index)
                else:
                    self.control_flow._process_if_statement(
                        line_content, line_number, self.control_flow._preprocessed_lines, line_index)
                if f"__if_line_{line_number}" in self.pending_assignments:
                    del self.pending_assignments[f"__if_line_{line_number}"]
            except Exception as e:
                missing = self.extract_missing_dependencies(e)
                if missing:
                    for dep in missing:
                        self.mark_dependency_missing(dep)
                    updated_deps = set(deps) | set(missing)
                    self.pending_assignments[f"__if_line_{line_number}"] = (
                        line_content, line_number, updated_deps)
                    continue
                if f"__if_line_{line_number}" in self.pending_assignments:
                    del self.pending_assignments[f"__if_line_{line_number}"]

        for line_number, line_content, deps in deferred_assignments:
            try:
                scope = self.current_scope()
                unresolved = any(
                    self._deferred_dependency_unresolved(dep, scope=scope)
                    for dep in deps)
                if not unresolved:
                    # Process the assignment
                    self.array_handler.evaluate_line_with_assignment(
                        line_content, line_number, self.current_scope().get_evaluation_scope())
                    # Remove from pending assignments
                    if f"__line_{line_number}" in self.pending_assignments:
                        del self.pending_assignments[f"__line_{line_number}"]
            except Exception as e:
                missing = self.extract_missing_dependencies(e)
                if missing:
                    for dep in missing:
                        self.mark_dependency_missing(dep)
                    updated_deps = set(deps) | set(missing)
                    self.pending_assignments[f"__line_{line_number}"] = (
                        line_content, line_number, updated_deps)
                    continue
                # Remove from pending assignments even on error to avoid infinite loops
                if f"__line_{line_number}" in self.pending_assignments:
                    del self.pending_assignments[f"__line_{line_number}"]
                raise
