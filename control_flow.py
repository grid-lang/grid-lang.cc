"""
Control flow processing for GridLang compiler.
Handles FOR loops, IF statements, LET statements, and block processing.
"""

import re
from units import UNIVERSAL_ZERO, ERROR_CODES, is_error_value
from utils import num_to_col, split_var_defs, is_sparse_array
from grid_lang_common import _STATEMENT_KEYWORDS, _first_keyword, match_push_assignment, unpack_push_assignment, match_return_statement, is_definition_body_error

# Regex patterns for block parsing
HEADER_IF = re.compile(r'^\s*if\b(.+?)\bthen\s*$', re.I)
HEADER_ELSEIF = re.compile(r'^\s*elseif\b(.+?)\bthen\s*$', re.I)
HEADER_ELSE = re.compile(r'^\s*else\s*$', re.I)
HEADER_FOR = re.compile(r'^\s*for\b(.+?)\bdo\s*$', re.I)
HEADER_WHEN = re.compile(r'^\s*when\b(.+?)\bdo\s*$', re.I)
HEADER_LET = re.compile(r'^\s*let\b(.+?)\bthen\s*$', re.I)
TOKEN_END = re.compile(r'^\s*end\s*$', re.I)
# 'End For', 'End If' and friends: a keyword after 'end' is never a block
# terminator (the terminator is the bare 'end' token), and it is never a
# definition close either, because definitions close with the name they
# declared - 'End Point', 'End Sub'.  These are rejected outright rather
# than falling through, which would silently leave the block open.
TOKEN_END_KEYWORD = re.compile(
    r'^\s*end\s+(?:for|if|else|elseif|while|when|select|case|each|next|do|let|with)\b',
    re.I)


def block_nesting_delta(clean):
    """Return 1 when a cleaned line opens a block closed by End, else 0.

    Shared by every For/Let block body collector so nested blocks
    (For/When/Let/If) never close the outer block early.
    """
    line = clean.strip().lower()
    if line.startswith('for ') and line.endswith('do'):
        return 1
    if line.startswith('when ') and line.endswith('do'):
        return 1
    if HEADER_LET.match(line):
        return 1
    if HEADER_IF.match(line):
        return 1
    return 0


class GridLangControlFlow:
    """Handles control flow constructs in GridLang."""

    def __init__(self, compiler=None):
        self.compiler = compiler
        self.executor = compiler  # The compiler is actually the executor
        self.block_map = {}  # Maps start line numbers to end line numbers
        self.if_blocks = []  # List of IF blocks with clause information
        self.loops = []      # List of loop blocks
        self.exit_for_requested = False  # Flag to break out of FOR loops
        self._preprocessed_lines = []  # Cache of the most recent preprocessed lines

        # Regex patterns for block parsing
        self._header_if = HEADER_IF
        self._header_elseif = HEADER_ELSEIF
        self._header_else = HEADER_ELSE
        self._header_for = HEADER_FOR
        self._header_when = HEADER_WHEN
        self._header_let = HEADER_LET
        self._token_end = TOKEN_END
        self._token_end_keyword = TOKEN_END_KEYWORD

    def _strip_inline_comment(self, s: str) -> str:
        """Strip an inline comment.

        A comment starts with an apostrophe: the syntax is VB-based and
        there are no single-quoted string literals, so the first apostrophe
        outside a double-quoted string begins the comment.

        '#' is *not* a comment marker -- it is the error-literal sigil
        ('#N/A', '#DIV/0', ...).  '//' is integer division, not a comment.
        """
        in_string = False
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = not in_string
            elif ch == "'" and not in_string:
                return s[:i].rstrip()
            i += 1
        return s.rstrip()

    def _get_parser(self):
        parser = getattr(self.compiler, 'parser', None)
        if parser:
            return parser
        if hasattr(self.compiler, 'compiler'):
            return getattr(self.compiler.compiler, 'parser', None)
        return None

    def _process_output_statement(self, def_str, line_number):
        parser = self._get_parser()
        if not parser:
            raise AttributeError("Parser not available for OUTPUT parsing")
        var, type_name, constraints, expr = parser._parse_variable_def(
            def_str, line_number)
        constraints = constraints or {}
        constraints['output'] = True
        current_scope = self.compiler.current_scope()
        target_scope = current_scope
        if hasattr(self.compiler, 'get_global_scope'):
            target_scope = self.compiler.get_global_scope() or current_scope
        defining_scope = target_scope.get_defining_scope(var)
        if not defining_scope:
            target_scope.define_output(
                var, type_name, line_number, constraints)
            defining_scope = target_scope
        else:
            defining_scope.output_variables.add(var.lower())
            actual_key = defining_scope._get_case_insensitive_key(
                var, defining_scope.constraints) or var
            defining_scope.constraints[actual_key] = constraints
            if type_name:
                type_key = defining_scope._get_case_insensitive_key(
                    var, defining_scope.types) or var
                defining_scope.types[type_key] = type_name
        if expr is not None:
            value = self.compiler.expr_evaluator.eval_or_eval_array(
                expr, current_scope.get_evaluation_scope(), line_number)
            defining_scope.update(var, value, line_number)
            # A live output's init seeds the write-through binding so the
            # caller observes the value without an explicit Push.
            live_pub = getattr(self.compiler, '_maybe_publish_live_output', None)
            if live_pub is not None:
                live_pub(var, value, line_number)

    def _match_push_assignment(self, text):
        return match_push_assignment(text)

    def _unpack_push_assignment(self, push_match):
        return unpack_push_assignment(push_match)

    def _match_return_statement(self, text):
        return match_return_statement(text)

    def _match_print_statement(self, text):
        return re.match(r'^\s*print\s+(.+)$', text, re.I)

    def _handle_print_statement(self, value_expr, line_number):
        """Print <expr> — ambient console output inside block bodies. Uses the
        dedicated '_console' channel (distinct from the 'output' call-value
        channel used by Return) so prints survive the sub-call boundary; the
        difference is that Print is legal anywhere while Return is the
        call-value channel restricted to functions and operations."""
        resolver_targets = [self.compiler]
        if hasattr(self.compiler, 'compiler'):
            resolver_targets.append(self.compiler.compiler)
        for target in resolver_targets:
            pending = getattr(target, 'pending_assignments', {}) or {}
            pending_vars = [
                pending_var for pending_var in list(pending.keys())
                if not pending_var.startswith('__line_')
            ]
            for pending_var in pending_vars:
                target._resolve_global_dependency(
                    pending_var, line_number,
                    target_scope=self.compiler.current_scope())
        values = self.compiler._evaluate_push_expression(
            value_expr, line_number)
        for value in values:
            snapshot = value
            if hasattr(self.compiler, '_snapshot_value'):
                snapshot = self.compiler._snapshot_value(value)
            self.compiler.output_values.setdefault('_console', []).append(snapshot)
        if '_console' not in self.compiler.output_variables:
            self.compiler.output_variables.append('_console')

    def _handle_return_statement(self, value_expr, line_number):
        resolver_targets = [self.compiler]
        if hasattr(self.compiler, 'compiler'):
            resolver_targets.append(self.compiler.compiler)
        for target in resolver_targets:
            pending = getattr(target, 'pending_assignments', {}) or {}
            pending_vars = [
                pending_var for pending_var in list(pending.keys())
                if not pending_var.startswith('__line_')
            ]
            for pending_var in pending_vars:
                target._resolve_global_dependency(
                    pending_var, line_number,
                    target_scope=self.compiler.current_scope())
        values = self.compiler._evaluate_push_expression(
            value_expr, line_number)
        for value in values:
            snapshot = value
            if hasattr(self.compiler, '_snapshot_output_value'):
                snapshot = self.compiler._snapshot_output_value(value)
            self.compiler.output_values.setdefault('output', []).append(snapshot)
        if 'output' not in self.compiler.output_variables:
            self.compiler.output_variables.append('output')

    def process_for_statement(self, line, line_number, scope):

        # Handle For var as type dim {dimensions} syntax
        m = re.match(
            r'For\s+([\w_]+)\s+as\s+(\w+)\s+dim\s*(\{[^}]*\})', line, re.I)
        if m:
            var_name, type_name, dim_str = m.groups()

            # Parse the dimension string
            dim_str = dim_str.strip()
            if dim_str.startswith('{') and dim_str.endswith('}'):
                dim_content = dim_str[1:-1].strip()
                # For now, just store the dimension constraint as a string
                # The actual parsing can be done later when needed
                constraints = {'dim': dim_str}
            else:
                constraints = {}

            existing_scope = scope.get_defining_scope(var_name)
            if existing_scope:
                existing_scope.types[var_name] = type_name.lower()
                existing_scope.constraints[var_name] = constraints
                return

            # Define the variable with the type and constraints
            scope.define(var_name, None, type_name.lower(), constraints, True)
            return

        # Handle For var as type dim number syntax
        m = re.match(
            r'For\s+([\w_]+)\s+as\s+(\w+)\s+dim\s+(\d+)', line, re.I)
        if m:
            var_name, type_name, dim_size = m.groups()

            existing_scope = scope.get_defining_scope(var_name)
            if existing_scope:
                existing_scope.types[var_name] = type_name.lower()
                existing_scope.constraints[var_name] = {
                    'dim': [(None, int(dim_size))]}
                return

            # Create an array with the specified size
            dim_size = int(dim_size)
            # Initialize array with zeros
            initial_array = [0.0] * dim_size

            # Define the variable with the initialized array
            scope.define(var_name, initial_array, type_name.lower(),
                         {'dim': [(None, dim_size)]}, False)
            return

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

        m = re.match(
            r'^\s*For\s+([\w_]+)\s*=\s*(.+?)(?:\s+DO\s*$|\s*$)', line, re.I)
        if m:
            var_name, expr = m.groups()

            # Check if variable is already declared in the current scope or in a parent scope
            # FOR assignment statements should be allowed to reuse variables from outer loop scopes
            # but not from the same logical block (like LET blocks)
            current_check_scope = scope
            while current_check_scope:
                if var_name in current_check_scope.variables:
                    # If this is a loop scope, allow it (variables can be reused across loop iterations)
                    # If this is not a loop scope, it's an error (like in LET blocks)
                    if not hasattr(current_check_scope, 'is_loop_scope') or not current_check_scope.is_loop_scope:
                        # If we're in a FOR loop context, allow variable reuse (this is the normal case for nested loops)
                        # Only fail if we're not in a loop context (like in LET blocks)
                        if not (hasattr(scope, 'is_loop_scope') and scope.is_loop_scope):
                            error_msg = f"Error: FOR variable '{var_name}' is already declared in an outer scope at line {line_number}"
                            print(error_msg)
                            raise ValueError(error_msg)
                    break
                current_check_scope = current_check_scope.parent

            value = self.compiler.expr_evaluator.eval_or_eval_array(
                expr, scope.get_evaluation_scope(), line_number)

            defining_scope = scope.get_defining_scope(var_name)
            if defining_scope:
                if defining_scope.types.get(var_name) == 'number' and not isinstance(value, (int, float)):
                    raise TypeError(
                        f"Cannot assign non-numeric value {value} to '{var_name}' at line {line_number}")
                defining_scope.update(var_name, float(value) if isinstance(
                    value, (int, float)) else value, line_number)
            else:
                inferred_type = self.compiler.array_handler.infer_type(
                    value, line_number)
                if inferred_type == 'int':
                    inferred_type = 'number'
                constraints = {'constant': value}
                if isinstance(value, dict) and value.get('_handle'):
                    # Engine-owned handle (e.g. `For t = clock.timer(3)`):
                    # the engine rewrites it as it advances (`\.timer` /
                    # `\.counter`), so it is not an immutable constant -
                    # mirror the LET-handle binding which strips the pin.
                    constraints = {}
                scope.define(var_name, value, inferred_type,
                             constraints, False)

            pending = list(scope.pending_assignments.items())
            for key, (expr, ln, deps) in pending:
                current_scope = self.compiler.current_scope()
                scope_pending = getattr(
                    current_scope, 'pending_assignments', {})
                unresolved = any(
                    self.compiler.has_unresolved_dependency(
                        dep, scope=current_scope, scope_pending=scope_pending)
                    for dep in deps)
                if not unresolved:
                    try:
                        self.compiler.array_handler.evaluate_line_with_assignment(
                            expr, ln, scope.get_evaluation_scope())
                        del scope.pending_assignments[key]
                    except Exception as e:
                        pass
            return

        raise SyntaxError(f"Invalid FOR syntax at line {line_number}")

    def _is_error_literal_for_header(self, line):
        """True when a block ``For x = <expr> do`` header uses an error code
        literal on the right side (e.g. ``For x = #N/A do``)."""
        m_eq = re.match(
            r'^\s*For\s+[\w_]+\s*=\s*(.+?)\s+do\s*$', line, re.I)
        if not m_eq:
            return False
        return m_eq.group(1).strip() in ERROR_CODES

    def _skip_equal_for_block_body(self, block_lines, i):
        """Skip the body of a ``For x = <expr> do ... End`` block and return
        the next index past its matching End."""
        _, end_index = self._extract_block_body(block_lines, i)
        next_index = end_index + 1 if end_index < len(
            block_lines) else len(block_lines)
        return True, next_index

    def identify_global_guard_lines(self, lines):
        """Return metadata for IF guard statements at global scope (no THEN)."""
        guard_lines = []
        depth = 0
        for idx, (raw_line, line_number) in enumerate(lines):
            stripped = self._strip_inline_comment(raw_line).strip()
            if not stripped:
                continue
            lower = stripped.lower()

            is_block_if = bool(self._header_if.match(stripped))
            is_block_for = bool(self._header_for.match(stripped))
            is_block_when = bool(self._header_when.match(stripped))
            is_end = bool(self._token_end.match(stripped))

            has_inline_then = " then " in lower or lower.endswith(" then")
            if depth == 0 and lower.startswith("if ") and not is_block_if and not has_inline_then:
                m = re.match(r'^\s*if\s+(.+?)\s*$', stripped, re.I)
                if m:
                    condition = m.group(1).strip()
                    guard_lines.append({
                        'index': idx,
                        'line_number': line_number,
                        'condition': condition,
                    })
                continue

            if is_block_if or is_block_for or is_block_when:
                depth += 1
                continue

            if is_end and depth > 0:
                depth -= 1

        return guard_lines

    def identify_global_for_declarations(self, lines):
        """Return global FOR statements that behave like declarations or single-pass blocks."""
        for_entries = []
        depth = 0
        for idx, (raw_line, line_number) in enumerate(lines):
            stripped = self._strip_inline_comment(raw_line).strip()
            if not stripped:
                continue
            lower = stripped.lower()

            is_block_if = bool(self._header_if.match(stripped))
            is_block_for = bool(self._header_for.match(stripped))
            is_block_when = bool(self._header_when.match(stripped))
            is_end = bool(self._token_end.match(stripped))

            is_top_level_for = depth == 0 and lower.startswith("for ")

            # Treat both simple FOR declarations and top-level FOR blocks as global pre-exec
            if is_top_level_for and (" do" not in lower or is_block_for):
                for_entries.append(
                    {'line_number': line_number, 'line': stripped})

            if is_block_if or is_block_for or is_block_when:
                depth += 1
            if is_end and depth > 0:
                depth -= 1
        return for_entries

    def _normalize_block_lines(self, block_lines, line_number):
        """Coerce a block body into the ``(text, line_number)`` pairs the engine uses.

        The engine identifies statements -- guards especially -- by line
        number, so every body must be numbered. Most callers already hold
        numbered lines from ``prepare()``; a constructor or builder body holds
        bare strings and a single ``line_number`` for the whole body, so each
        line is given a distinct offset number. A shared number would make the
        engine treat every statement as the guard line and skip the body.
        """
        normalized = []
        for offset, entry in enumerate(block_lines, start=1):
            if isinstance(entry, tuple):
                normalized.append(entry)
            else:
                normalized.append((entry, line_number + offset))
        return normalized

    def _extract_block_body(self, block_lines, start_index):
        """Return the body lines and matching END index for a block starting at start_index."""
        depth = 0
        body_start = start_index + 1
        j = body_start
        while j < len(block_lines):
            raw_line, _ = block_lines[j]
            line_clean = self._strip_inline_comment(raw_line).strip()
            if not line_clean:
                j += 1
                continue
            if (self._header_if.match(line_clean) or
                    self._header_for.match(line_clean) or
                    self._header_when.match(line_clean) or
                    self._header_let.match(line_clean)):
                depth += 1
            elif self._token_end.match(line_clean):
                if depth == 0:
                    return block_lines[body_start:j], j
                depth -= 1
            j += 1
        return block_lines[body_start:], len(block_lines)

    def _should_break_block_processing(self):
        """Handle EXIT LOOP flag and indicate whether block processing should stop."""
        if self.executor.exit_loop:
            self.executor.exit_loop = False
            return True
        return False

    def _prepare_block_line(self, block_lines, i):
        """Return the raw line tuple and cleaned content for the current block index."""
        line, line_number = block_lines[i]
        return line, line_number, line.strip()

    def _handle_block_when_statement(self, block_lines, i, line, line_number, line_clean):
        """Handle WHEN ... DO block registration."""
        if not (line_clean.lower().startswith("when ") and line_clean.lower().endswith("do")):
            return False, i
        when_match = re.match(
            r'^\s*when\s+(.+?)\s+do\s*$', line, re.I)
        if not when_match:
            raise SyntaxError(
                f"Invalid WHEN syntax at line {line_number}")
        condition = when_match.group(1).strip()
        block_body, end_index = self._extract_block_body(
            block_lines, i)
        self.executor._register_when_block(
            condition, block_body, line_number)
        return True, end_index + 1

    def _run_constructor_block(self, block_lines, value_dict, line_number):
        """Execute constructor/builder block lines with the engine's walkers.

        A constructor body is ordinary language code that additionally reads and
        writes its instance's fields, so it is executed by this engine rather
        than by a second interpreter. Fields are mirrored into the scope for
        the duration and written back afterwards.
        """
        if not block_lines:
            return
        self._push_type_fields_to_scope(value_dict)
        try:
            self._process_block(block_lines, self.compiler.current_scope())
        finally:
            self._sync_type_fields_from_scope(value_dict)

    def _process_type_if_block_via_engine(self, block_lines, i, value_dict, line_number):
        """Run an ``If ... then [elseif/else] ... End`` from a constructor body.

        Parsing, condition evaluation and branch selection are the engine's;
        the selected branch runs through ``_process_block``. The single-line
        form ``if cond then action`` takes the block dispatcher's inline path.
        """
        line, ln, line_clean = self._prepare_block_line(block_lines, i)
        if re.match(r'^\s*if\s+.+?\s+then\s+\S', line, re.I):
            handled, end = self._handle_block_if_dispatch(
                block_lines, i, line, ln, line_clean)
            self._sync_type_fields_from_scope(value_dict)
            return end if handled else i + 1
        consumed = self._process_if_statement_rich(line, ln, block_lines, i)
        end = min(i + consumed + 1, len(block_lines))
        self._sync_type_fields_from_scope(value_dict)
        return end

    def _process_type_for_block_via_engine(self, block_lines, i, value_dict, line_number):
        """Run a ``For ... do ... End`` from a constructor body via the engine."""
        line, ln, line_clean = self._prepare_block_line(block_lines, i)
        # Multi-variable ``AND`` ranges need the cartesian/dynamic-bound walk
        # the main loop owns; it drives the body through _process_block too.
        m_vars = re.match(r'^\s*for\s+(.+?)\s+do\s*$', line, re.I)
        if m_vars and ' and ' in m_vars.group(1).lower() \
                and ' in ' in m_vars.group(1).lower():
            handled, next_i = self.compiler._handle_for_and_syntax(
                block_lines, i, m_vars.group(1).strip(), True, ln)
            self._sync_type_fields_from_scope(value_dict)
            return next_i if handled else i + 1
        handled, next_i = self._handle_block_for_statement(block_lines, i, line, ln)
        if not handled:
            body, end_index = self._extract_block_body(block_lines, i)
            self.process_for_statement(line, ln, self.compiler.current_scope())
            self._run_constructor_block(body, value_dict, line_number)
            next_i = end_index + 1
        self._sync_type_fields_from_scope(value_dict)
        return next_i

    def _process_type_let_then_via_engine(self, block_lines, i, value_dict, line_number):
        """Run a ``Let <cond> then ... End`` from a constructor body."""
        return self._handle_let_cond_block(
            block_lines, i, line_number,
            on_done=lambda: self._sync_type_fields_from_scope(value_dict))

    def _handle_let_cond_block(self, block_lines, i, line_number,
                               value_dict=None, on_done=None):
        """Run a conditional ``Let <cond> then ... End`` block.

        Declaration headers (``Let x = 2 then``) bind their variable first,
        exactly like the main-loop LET, so the body can read it; guard
        headers (``Let x > 10 then``) run the body only when the current
        value satisfies the comparison. The body itself is ordinary code and
        is executed by :meth:`_process_block`.
        """
        line, ln, line_clean = self._prepare_block_line(block_lines, i)
        header = re.sub(r'^Let\s+', '', line_clean, count=1, flags=re.I)
        header = re.sub(r'\s+then\s*$', '', header, flags=re.I).strip()
        body, end_index = self._extract_block_body(block_lines, i)
        var, type_name, constraints, expr = self.compiler._parse_variable_def(
            header, ln)
        condition_passed = True
        block_local = False
        plain_var = bool(isinstance(var, str) and re.match(
            r'^[A-Za-z_]\w*$', var or ''))
        if expr is not None and plain_var:
            fresh = self.compiler.current_scope().get_defining_scope(var) is None
            status = self.executor._bind_declared_var(
                var, type_name, constraints, expr, ln,
                shadow_keyword='LET', mark_implicit=True)
            if status == 'deferred':
                condition_passed = False
            else:
                if fresh and status == 'bound':
                    # A block-form Let declares a new variable that is only
                    # visible inside its block: record it where the enclosing
                    # scope copy-back looks so it is dropped at End.
                    self.compiler.current_scope().local_let_declarations.add(
                        var.lower())
                    block_local = True
                try:
                    bound_value = self.compiler.current_scope().get(var)
                except NameError:
                    bound_value = None
                # An error-valued or unresolved declaration skips the body,
                # matching the main-loop LET block.
                if bound_value is None or is_error_value(bound_value):
                    condition_passed = False
        elif expr is not None:
            # Index/field writes are not bound here; keep the legacy
            # truthiness gate so the body only runs on a real value.
            try:
                value = self.compiler.expr_evaluator.eval_or_eval_array(
                    str(expr),
                    self.compiler.current_scope().get_evaluation_scope(), ln)
                condition_passed = bool(value)
            except Exception:
                condition_passed = False
        elif constraints:
            # Guard headers (``x > 10``, ``name <> "bob"``, ``q of peso``)
            # are ordinary conditions: evaluate them with the shared If/'?'
            # evaluator instead of re-implementing the operator table.
            try:
                condition_passed = bool(self._evaluate_if_condition(
                    header, ln, warn_boolean=True))
            except NameError:
                condition_passed = False
        if condition_passed and body:
            self._process_block(body, self.compiler.current_scope())
        if block_local:
            # The header variable lives only inside the block. The main-loop
            # LET drops it by popping the block scope; this path binds into
            # the current scope, so drop it here instead of waiting for the
            # enclosing scope's copy-back.
            scope = self.compiler.current_scope()
            old_key = scope._get_case_insensitive_key(
                var, scope.variables)
            if old_key is not None:
                scope.variables.pop(old_key, None)
                scope.types.pop(old_key, None)
                scope.constraints.pop(old_key, None)
            scope.local_let_declarations.discard(var.lower())
            scope.clear_implicit_let(var)
        if callable(on_done):
            on_done()
        return end_index + 1

    def _push_type_fields_to_scope(self, value_dict, only=None):
        """Constructor-side helper: mirror instance fields into the scope."""
        return self.compiler.type_processor._push_type_fields_to_scope(
            value_dict, only=only)

    def _sync_type_fields_from_scope(self, value_dict):
        """Constructor-side helper: copy engine field writes into the instance."""
        return self.compiler.type_processor._sync_type_fields_from_scope(value_dict)

    def _is_optional_let_line(self, line_clean):
        """True when a line may be read as a ``Let`` with the keyword left out.

        The keyword is optional only when the line is unambiguously a
        declaration. Every constraint kind is accepted -- a type
        (``a as number``), a unit (``a of peso``), a dim
        (``arr dim 3 as logical = {...}``), a choice or default
        (``grid not null or = 0``), and an inequality (``x > 1``) -- because
        the parser, not this predicate, decides what a declaration looks like.
        A bare ``a`` with neither a value nor a constraint stays mandatory.

        Two shapes stay mandatory here: a leading instruction keyword, and a
        comparison, whose ``=`` is part of ``>=``/``<=``/``==``/``<>``/``!=``
        rather than an assignment.
        """
        if not line_clean:
            return False
        if re.search(r'\bthen\b', line_clean, re.I):
            return False
        first = re.match(r'^\s*(\$?[A-Za-z_][\w.]*)', line_clean)
        if not first:
            return False
        if _first_keyword(first.group(1).lstrip('$').split('.')[0]) in _STATEMENT_KEYWORDS:
            return False
        try:
            _var, type_name, constraints, expr = \
                self.compiler._parse_variable_def(line_clean, 0)
        except Exception:
            return False
        # A bare name declares nothing on its own, so `Let a` is required.
        return bool(expr or constraints or type_name)

    def _handle_block_optional_let_statement(self, block_lines, i, line,
                                            line_number, line_clean):
        """Run ``name = expr`` as ``Let name = expr`` in block context.

        This reuses the in-block ``Let`` binder, so bare assignment and an
        explicit ``Let`` in the same context are the same instruction.
        """
        if not self._is_optional_let_line(line_clean):
            return False, i
        self._process_let_statement_inline('Let ' + line, line_number)
        return True, i + 1

    def _handle_block_push_or_output_statement(self, i, line, line_number, line_clean):
        """Handle PUSH and OUTPUT statements in block context."""
        if line_clean.lower().startswith('push '):
            try:
                m_assign = self._match_push_assignment(line_clean)
                if not m_assign:
                    raise SyntaxError(
                        f"Invalid PUSH syntax at line {line_number}")
                target, value_expr = self._unpack_push_assignment(m_assign)
                self.compiler._handle_push_assignment(
                    target, value_expr, line_number)
            except Exception as e:
                raise
            return True, i + 1
        if line_clean.lower().startswith('output '):
            try:
                def_str = line_clean[len('output '):].strip()
                self._process_output_statement(def_str, line_number)
            except Exception as e:
                raise
            return True, i + 1
        return False, i

    def _handle_block_direct_assignment_statement(self, i, line, line_number, line_clean):
        """Handle direct := assignment lines in block context."""
        if ':=' not in line_clean:
            return False, i
        try:
            self.compiler.array_handler.evaluate_line_with_assignment(
                line, line_number, self.compiler.current_scope().get_evaluation_scope())
        except Exception as e:
            raise
        return True, i + 1

    def _handle_block_if_dispatch(self, block_lines, i, line, line_number, line_clean):
        if not line_clean.lower().startswith('if '):
            return False, i

        # Inline guard:  if <cond> then <action>  (single line, no block).
        # The guard is evaluated first; if its dependencies still need
        # evaluating, it is deferred without running any of the action.
        inline_if_match = re.match(
            r'^\s*if\s+(.+?)\s+then\s+(.+)$', line, re.I)
        if inline_if_match:
            cond, action = inline_if_match.groups()
            if self._block_if_guard_deferred(
                    cond.strip(), line, line_number):
                return True, i + 1
            try:
                if self._evaluate_if_condition(
                        cond.strip(), line_number, warn_boolean=True):
                    action = action.strip()
                    if action.lower().startswith('let '):
                        self._process_let_statement_inline(action, line_number)
                    elif action.lower().startswith('output '):
                        try:
                            def_str = action.strip()[len('output '):].strip()
                            self._process_output_statement(
                                def_str, line_number)
                        except Exception as e:
                            raise
                    elif ':=' in action:
                        self.compiler.array_handler.evaluate_line_with_assignment(
                            action, line_number, self.compiler.current_scope().get_evaluation_scope())
                    else:
                        push_match = self._match_push_assignment(action)
                        return_match = self._match_return_statement(action)
                        if push_match:
                            target, value_expr = self._unpack_push_assignment(push_match)
                            self.compiler._handle_push_assignment(
                                target, value_expr, line_number)
                        elif return_match:
                            self._handle_return_statement(
                                return_match.group(1).strip(), line_number)
                        else:
                            # Any other single-line action (bare subprocess or
                            # engine-handle member call, plain expression) runs
                            # like a normal block line
                            callable_fn = getattr(
                                self.compiler, '_maybe_handle_subprocess_call', None)
                            if callable_fn is not None and callable_fn(
                                    action, line_number):
                                return True, i + 1
                            self.compiler.expr_evaluator.eval_expr(
                                action,
                                self.compiler.current_scope().get_evaluation_scope(),
                                line_number)
            except NameError:
                return True, i + 1
            except Exception as e:
                return True, i + 1
            return True, i + 1

        # Block form:  if <cond> then <body> end
        if line_clean.lower().endswith('then'):
            cond_match = re.match(
                r'^\s*if\s+(.+?)\s*then\s*$', line, re.I)
            if not cond_match:
                return True, i + 1
            condition = cond_match.group(1).strip()

            # Locate the whole block so the enclosing walker skips it
            # entirely (the body is executed exactly once here).
            body_lines, end_index = self._extract_block_body(
                block_lines, i)

            # Blocks with ELSEIF / ELSE clauses keep the rich handler.
            if_block = self._match_global_if_block(
                line_number, block_lines, i)
            clause_count = len(if_block.get('clauses', [])) if if_block else 0
            if not if_block:
                # Constructor bodies are renumbered and absent from the
                # pre-scan; without this an ELSE would run as part of the
                # taken clause's span instead of being mutually exclusive.
                local_block = self._scan_local_if_block(
                    line, line_number, block_lines, i)
                clause_count = len(local_block.get(
                    'clauses', [])) if local_block else 0
            if clause_count > 1:
                try:
                    consumed = self._process_if_statement_rich(
                        line, line_number, block_lines, i)
                    return True, i + consumed + 1
                except Exception as e:
                    return True, end_index + 1

            # Guard first: when a dependency is defined but still pending,
            # defer the whole block without running any push.
            if self._block_if_guard_deferred(
                    condition, line, line_number):
                return True, end_index + 1

            try:
                condition_result = self._evaluate_if_condition(
                    condition, line_number, warn_boolean=True)
            except NameError as exc:
                dep_extractor = getattr(
                    self.compiler, 'extract_missing_dependencies',
                    lambda e: set())
                missing = dep_extractor(exc)
                if missing:
                    current_scope = self.compiler.current_scope()
                    pending_deps = [
                        dep for dep in missing
                        if self.compiler.has_unresolved_dependency(
                            dep, scope=current_scope)
                        and dep in getattr(
                            self.compiler, 'pending_assignments', {})
                    ]
                    if pending_deps:
                        self._defer_pending_if(
                            line, line_number,
                            set(missing) | set(pending_deps))
                        return True, end_index + 1
                # A dependency that will never resolve simply reads as False.
                condition_result = False

            # The clause condition guards its body for reactive
            # recomputation too, not just for this walk, so it is recorded
            # whether or not the clause runs this time.
            self._add_block_guard_entries(
                body_lines, [(condition, line_number)])
            if condition_result:
                self._process_block(
                    body_lines, self.compiler.current_scope())
            return True, end_index + 1

        # Inline THEN without a terminating END: legacy single-line IF.
        if re.match(r'^if\b.+\bthen\b', line_clean, re.I):
            try:
                consumed = self._process_if_statement(
                    line, line_number, block_lines, i)
                return True, i + consumed
            except Exception as e:
                return True, i + 1
        return False, i

    def _handle_block_for_statement(self, block_lines, i, line, line_number):
        if not line.strip().lower().startswith("for "):
            return False, i

        if '=' in line and not re.search(r'\bin\b|\bas\b|\bdim\b', line.lower()):
            try:
                self.process_for_statement(
                    line, line_number, self.compiler.current_scope())
                # ``For x = <expr> do ... End`` inside a block: when the loop
                # variable holds an error value, skip the block body.
                if line.strip().lower().endswith('do'):
                    m_var = re.match(
                        r'^\s*For\s+([\w_]+)\s*=', line, re.I)
                    if m_var:
                        var_name = m_var.group(1)
                        def_scope = self.compiler.current_scope(
                        ).get_defining_scope(var_name)
                        if def_scope is not None and is_error_value(
                                def_scope.get(var_name)):
                            return self._skip_equal_for_block_body(
                                block_lines, i)
                return True, i + 1
            except ValueError as e:
                if "already declared in an outer scope" in str(e):
                    raise e
                if self._is_error_literal_for_header(line):
                    return self._skip_equal_for_block_body(
                        block_lines, i)
                return True, i + 1
            except Exception:
                if self._is_error_literal_for_header(line):
                    return self._skip_equal_for_block_body(
                        block_lines, i)
                return True, i + 1

        m = re.match(r'^\s*FOR\s+(.+?)(?:\s+do\s*$|\s*$)', line, re.I)
        if not m:
            return True, i + 1
        var_defs = m.group(1).strip()
        is_block = line.strip().lower().endswith('do')
        index_fallback = None
        index_match_line = re.search(
            r'\bindex\s+([A-Za-z_][A-Za-z0-9_]*)\b', var_defs, re.I)
        if index_match_line:
            index_fallback = index_match_line.group(1)

        var_list = []
        if ' and ' in var_defs.lower():
            var_parts = re.split(
                r'\s+and\s+', var_defs, flags=re.I)
            for var_part in var_parts:
                var, type_name, constraints, value = self.compiler._parse_variable_def(
                    var_part, line_number)
                var_list.append(
                    (var, type_name, constraints, value))
        else:
            var, type_name, constraints, value = self.compiler._parse_variable_def(
                var_defs, line_number)
            var_list.append((var, type_name, constraints, value))

        for var, _, constraints, value in var_list:
            defining_scope = self.compiler.current_scope().get_defining_scope(var)
            if defining_scope and value is None and constraints and not getattr(self.compiler.current_scope(), 'is_loop_scope', False):
                defining_scope.constraints[var] = constraints
                continue
            if defining_scope and not getattr(self.compiler.current_scope(), 'is_loop_scope', False):
                raise ValueError(
                    f"Variable '{var}' already defined in scope at line {line_number}")

        loop_body = block_lines[i + 1:]
        loop_end_index = None
        if is_block:
            loop_body, loop_end_index = self._extract_block_body(
                block_lines, i)

        init_entries = [(var, type_name, constraints) for var, type_name,
                        constraints, _ in var_list if constraints and 'init' in constraints]
        handled_init, next_i = self._handle_block_for_init_entries(
            init_entries, var_list, is_block, loop_body, loop_end_index,
            block_lines, i, line_number, index_fallback)
        if handled_init:
            return True, next_i

        for var, _, constraints, _ in var_list:
            if 'in' in constraints:
                values = constraints['in']
                for val in values:
                    if is_error_value(val):
                        # Skip error elements in the loop
                        continue
                    try:
                        if val.isdigit():
                            val = int(val)
                        elif val.replace('.', '').replace('-', '').isdigit():
                            val = float(val)
                    except Exception:
                        pass

                    self.compiler.push_scope(
                        is_private=True, is_loop_scope=True)
                    self.compiler.current_scope().define(var, val, 'number')
                    self._process_block(
                        loop_body)
                    parent_scope = self.compiler.current_scope().parent
                    if parent_scope:
                        loop_local_lets = getattr(
                            self.compiler.current_scope(),
                            'local_let_declarations', set())
                        for name, value in self.compiler.current_scope().variables.items():
                            if name.lower() in loop_local_lets:
                                continue
                            try:
                                parent_scope.update(name, value)
                            except Exception:
                                inferred_type = self.compiler.array_handler.infer_type(
                                    value, line_number)
                                parent_scope.define(
                                    name, value, inferred_type, {}, is_uninitialized=False)
                    self.compiler.pop_scope()
            elif 'range' in constraints:
                range_constraint = constraints['range']
                start_expr = range_constraint['start']
                end_expr = range_constraint['end']
                try:
                    start_val = self.compiler.expr_evaluator.eval_expr(
                        start_expr, self.compiler.current_scope().get_evaluation_scope(), line_number)
                    end_val = self.compiler.expr_evaluator.eval_expr(
                        end_expr, self.compiler.current_scope().get_evaluation_scope(), line_number)
                    start_val = int(start_val)
                    end_val = int(end_val)
                    for val in range(start_val, end_val + 1):
                        if self.exit_for_requested:
                            self.exit_for_requested = False
                            break
                        self.compiler.push_scope(
                            is_private=True, is_loop_scope=True)
                        self.compiler.current_scope().define(var, val, 'number')
                        self._process_block(
                            loop_body, self.compiler.current_scope())
                        parent_scope = self.compiler.current_scope().parent
                        if parent_scope:
                            loop_local_lets = getattr(
                                self.compiler.current_scope(),
                                'local_let_declarations', set())
                            for name, value in self.compiler.current_scope().variables.items():
                                if name.lower() in loop_local_lets:
                                    continue
                                try:
                                    parent_scope.update(name, value)
                                except Exception:
                                    inferred_type = self.compiler.array_handler.infer_type(
                                        value, line_number)
                                    parent_scope.define(
                                        name, value, inferred_type, {}, is_uninitialized=False)
                        self.compiler.pop_scope()
                except Exception as e:
                    raise SyntaxError(
                        f"Invalid range expressions in FOR loop at line {line_number}")
            else:
                if constraints:
                    try:
                        defining_scope = self.compiler.current_scope().get_defining_scope(var)
                        if defining_scope and not defining_scope.is_uninitialized(var):
                            var_value = defining_scope.get(var)
                            if var_value is not None:
                                defining_scope._check_constraints(
                                    var, var_value, line_number)
                    except ValueError as e:
                        break
                self._process_block(loop_body)

        if is_block and loop_end_index is not None:
            next_index = loop_end_index + \
                1 if loop_end_index < len(block_lines) else len(block_lines)
            return True, next_index
        return True, i + 1

    def _handle_block_for_init_entries(
            self, init_entries, var_list, is_block, loop_body, loop_end_index,
            block_lines, i, line_number, index_fallback):
        if not init_entries:
            return False, i
        if is_block:
            raw_init_expr = init_entries[0][2].get('init')
            index_name = init_entries[0][2].get('index') or index_fallback
            try:
                values = self.compiler._evaluate_push_expression(
                    str(raw_init_expr), line_number)
            except Exception:
                values = [self.compiler.expr_evaluator.eval_or_eval_array(
                    str(raw_init_expr), self.compiler.current_scope().get_evaluation_scope(), line_number)]
            for idx_val, val in enumerate(values, start=1):
                self.compiler.push_scope(
                    is_private=True, is_loop_scope=True)
                loop_scope = self.compiler.current_scope()
                if index_name:
                    defining = loop_scope.get_defining_scope(index_name)
                    if defining:
                        defining.update(index_name, idx_val, line_number)
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
                    idx_var = (constraints or {}).get('index') or index_fallback
                    if idx_var:
                        if idx_var in loop_scope.variables:
                            loop_scope.update(idx_var, idx_val, line_number)
                        else:
                            loop_scope.define(
                                idx_var, idx_val, 'number', {}, is_uninitialized=False)
                if index_fallback and index_fallback not in loop_scope.variables:
                    loop_scope.define(
                        index_fallback, idx_val, 'number', {}, is_uninitialized=False)
                if index_name and index_name not in loop_scope.variables:
                    loop_scope.define(index_name, idx_val, 'number',
                                      {}, is_uninitialized=False)
                self._process_block(loop_body)
                parent_scope = loop_scope.parent
                if parent_scope:
                    loop_local_lets = getattr(
                        loop_scope, 'local_let_declarations', set())
                    for name, value in loop_scope.variables.items():
                        if name.lower() in loop_local_lets:
                            continue
                        try:
                            parent_scope.update(name, value)
                        except Exception:
                            inferred_type = self.compiler.array_handler.infer_type(
                                value, line_number)
                            parent_scope.define(
                                name, value, inferred_type, {}, is_uninitialized=False)
                self.compiler.pop_scope()
            if is_block and loop_end_index is not None:
                next_index = loop_end_index + 1 if loop_end_index < len(
                    block_lines) else len(block_lines)
                return True, next_index
            return True, i + 1
        for var, type_name, constraints, _ in var_list:
            if not constraints or 'init' not in constraints:
                continue
            init_expr = constraints.get('init')
            try:
                values = self.compiler._evaluate_push_expression(
                    str(init_expr), line_number)
            except Exception:
                values = [self.compiler.expr_evaluator.eval_or_eval_array(
                    str(init_expr), self.compiler.current_scope().get_evaluation_scope(), line_number)]
            target_scope = self.compiler.current_scope().get_defining_scope(
                var) or self.compiler.current_scope()
            for val in values:
                try:
                    target_scope.update(var, val, line_number)
                except NameError:
                    target_scope.define(
                        var, val, type_name, constraints, is_uninitialized=False)
        return True, i + 1

    def _extract_rhs_vars(self, rhs):
        rhs_vars = set(re.findall(
            r'(?<![\w.])\b[\w_]+\b(?=\s*(?:[\[\{]|!\w+\s*\(|(?:\.\w+)?\s*$))', rhs))
        if '$"' in rhs:
            placeholders = re.findall(r'\{\s*([^}]*?)\s*\}', rhs)
            for ph in placeholders:
                rhs_vars.update(re.findall(r'\b[\w_]+\b', ph))
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
        field_vars = set(re.findall(r'\b[\w_]+\b(?=\.\w+\s*$)', rhs))
        rhs_vars.update(field_vars)
        rhs_vars = {
            var for var in rhs_vars
            if not (re.match(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$', var, re.I) or
                    re.match(r'^e[+-]?\d+$', var, re.I))
        }
        return rhs_vars

    def _is_index_element_write(self, line, line_number):
        """Whether a plain ``=`` line writes an element of an indexed array.

        ``grid{2, 2} = 7``, ``grid![A1] = 1`` and ``grid(1) = 9`` are element
        writes, not declarations, so they belong to the shared indexed-write
        path (``_try_process_let_indexed_assignment``) whichever keyword
        introduces them. Claiming them here instead left
        ``_parse_assignment_target_details`` to reject them with "Invalid
        assignment target" once the LET keyword was left out.

        Returns False for cell targets (``[A1] := ...``) and for plain
        identifiers, which stay this handler's business.
        """
        if ':=' in line or '=' not in line:
            return False
        target = line.split('=', 1)[0].strip()
        if not target or target.startswith('['):
            return False
        scope_dict = self.compiler.current_scope().get_evaluation_scope()
        try:
            var_name, _ = self.compiler.expr_evaluator._parse_index_target(
                target, scope_dict, line_number)
        except Exception:
            return False
        return var_name is not None

    def _handle_block_cell_binding_or_assignment(self, block_pending, i, line, line_number, line_clean):
        if re.match(r'^\[\s*\^?[A-Za-z]+\d+\s*\]\s*:\s*', line) and '=' in line and ':=' not in line:
            try:
                self.compiler._evaluate_cell_var_definition(
                    line, line_number)
            except Exception as e:
                raise RuntimeError(
                    f"Error processing cell binding line: {e} at line {line_number}")
            return True, i + 1
        if not (':=' in line or (
            '=' in line
            and ':=' not in line
            and not line_clean.lower().startswith((
                'if ', 'for ', 'when ', 'let ', 'output ', 'input ', 'define ', 'return ', 'push ', 'print '))
            and not line.strip().startswith(': ')
            and not re.match(r'^\[\s*\^?[A-Za-z]+\d+\s*\]\s*:\s*', line)
            and not self._is_index_element_write(line, line_number)
        )):
            return False, i
        if ':=' in line:
            target, rhs = line.split(':=', 1)
        else:
            target, rhs = line.split('=', 1)
        target, rhs = target.strip(), rhs.strip()
        rhs_vars = self._extract_rhs_vars(rhs)
        current_scope = self.compiler.current_scope()
        scope_pending = getattr(current_scope, 'pending_assignments', {})
        unresolved = any(
            self.compiler.has_unresolved_dependency(
                var, scope=current_scope, scope_pending=scope_pending)
            for var in rhs_vars)
        if unresolved:
            constraints = {}
            for var in rhs_vars:
                if var in self.compiler.current_scope().constraints:
                    constraints[var] = self.compiler.current_scope(
                    ).constraints[var]
                defining_scope = self.compiler.current_scope().get_defining_scope(var)
                if defining_scope and var in defining_scope.constraints:
                    constraints[var] = defining_scope.constraints[var]
            block_pending[f"__block_line_{line_number}"] = (
                line, line_number, rhs_vars, constraints)
            self.compiler.current_scope().pending_assignments[f"__block_line_{line_number}"] = (
                line, line_number, rhs_vars, constraints)
            return True, i + 1
        violations = []
        for var in rhs_vars:
            defining_scope = self.compiler.current_scope().get_defining_scope(var)
            if defining_scope and var in defining_scope.constraints:
                try:
                    var_value = defining_scope.get(var)
                    if var_value is not None:
                        defining_scope._check_constraints(
                            var, var_value, line_number)
                except ValueError as e:
                    violations.append(var)
        if not violations:
            is_plain_var_target = re.match(
                r'^[A-Za-z_][A-Za-z0-9_]*$', target) is not None
            if is_plain_var_target:
                evaluated_value = self.compiler.expr_evaluator.eval_or_eval_array(
                    rhs, self.compiler.current_scope().get_evaluation_scope(), line_number)
                defining_scope = self.compiler.current_scope().get_defining_scope(
                    target)
                if defining_scope:
                    defining_scope.update(
                        target, evaluated_value, line_number)
                else:
                    inferred_type = self.compiler.array_handler.infer_type(
                        evaluated_value, line_number)
                    self.compiler.current_scope().define(
                        target, evaluated_value, inferred_type, {}, is_uninitialized=False)
            elif (':=' not in line and re.match(
                    r'^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$',
                    target)):
                # ``P.x = i`` builds a field inside a block the same way it
                # does at the top level; the address path below only
                # understands cells.
                self.compiler._handle_bare_let_line(line, line_number)
            else:
                self.compiler.array_handler.evaluate_line_with_assignment(
                    line, line_number, self.compiler.current_scope().get_evaluation_scope())
        else:
            self.compiler.grid.clear()
        return True, i + 1

    def _handle_block_tail_statements(self, block_lines, i, line, line_number, line_clean):
        if line_clean.lower().startswith('if '):
            try:
                if re.match(r'^if\b.+\bthen\b', line_clean, re.I) and not line_clean.lower().endswith('then'):
                    consumed = self._process_if_statement(
                        line, line_number, block_lines, i)
                    return True, i + consumed, False
                consumed = self._process_if_statement_rich(
                    line, line_number, block_lines, i)
                return True, i + consumed + 1, False
            except Exception as e:
                return True, i + 1, False
        if line.strip().startswith(': '):
            assignment_line = line.strip()[2:]
            m = re.match(r'^([\w_]+)\s*=\s*(.+)$', assignment_line)
            if m:
                # A colon-let on a plain variable has the same shape as the
                # keyword-less assignment ``name = expr``; reuse its in-block
                # Let binder (the cell evaluator below rejects plain-variable
                # targets, and its failure is swallowed here).
                handled, next_i = self._handle_block_optional_let_statement(
                    block_lines, i, assignment_line, line_number, assignment_line)
                if handled:
                    return handled, next_i, False
            try:
                self.compiler.array_handler.evaluate_line_with_assignment(
                    assignment_line, line_number, self.compiler.current_scope().get_evaluation_scope())
            except Exception as e:
                pass
            return True, i + 1, False
        if line.strip().lower().startswith('let '):
            self._process_let_statement_inline(
                line, line_number)
            return True, i + 1, False
        if line.strip() == 'exit for':
            self.executor.exit_loop = True
            return True, i, True
        if line.strip().startswith('elseif ') and line.strip().endswith(' then'):
            return True, i + 1, False
        if line.strip() == 'else':
            return True, i + 1, False
        if self._match_push_assignment(line):
            push_match = self._match_push_assignment(line)
            target, value_expr = self._unpack_push_assignment(push_match)
            self.compiler._handle_push_assignment(
                target, value_expr, line_number)
            return True, i + 1, False
        if self._match_print_statement(line):
            print_match = self._match_print_statement(line)
            self._handle_print_statement(
                print_match.group(1).strip(), line_number)
            return True, i + 1, False
        if self._match_return_statement(line):
            return_match = self._match_return_statement(line)
            self._handle_return_statement(
                return_match.group(1).strip(), line_number)
            return True, i + 1, False
        return False, i, False

    def _block_header(self, line_clean):
        """True when the line opens a nested block (If/For/When/Let)."""
        return bool(
            self._header_if.match(line_clean)
            or self._header_for.match(line_clean)
            or self._header_when.match(line_clean)
            or self._header_let.match(line_clean))

    def _collect_block_guards(self, block_lines):
        """Return the standalone ``If <cond>`` guards of a block body.

        A guard belongs to the block that contains it, so a guard inside a
        nested block is left to that block. Guards are position-independent:
        one gates the whole body, whether it is written before or after the
        statements it guards.
        """
        guards = []
        depth = 0
        for index, (line, line_number) in enumerate(block_lines):
            line_clean = self._strip_inline_comment(line).strip()
            if not line_clean:
                continue
            if self._block_header(line_clean):
                # Some callers hand over a body that starts with its own header
                # line, which opens no nested block for this body.
                if index == 0 and depth == 0:
                    continue
                depth += 1
                continue
            if self._token_end.match(line_clean):
                if depth > 0:
                    depth -= 1
                continue
            if depth > 0:
                continue
            if re.match(r'^\s*if\s+\S', line_clean, re.I) and \
                    not re.search(r'\bthen\b', line_clean, re.I):
                condition = re.match(
                    r'^\s*if\s+(.+?)\s*$', line_clean, re.I).group(1).strip()
                if condition:
                    guards.append((line_clean, line_number, condition))
        return guards

    def _evaluate_block_guards(self, guards):
        """Evaluate a block's guards before any of its statements run.

        Returns True when the body may run. A guard whose dependency is still
        pending defers the whole body through the deferred-assignment machinery
        (the If line is retried once the binding resolves); a guard that is
        false - including one naming something that never resolves - skips the
        body. A guard naming a variable that nothing will ever define is an
        error: skipping the body would silently hide the typo.
        """
        pending_assignments = getattr(self.compiler, 'pending_assignments', {})
        for guard_line, line_number, condition in guards:
            dep_extractor = getattr(
                self.compiler, '_extract_dependencies_from_expression', None)
            deps = set(dep_extractor(condition)) if callable(
                dep_extractor) else set()
            current_scope = self.compiler.current_scope()
            if any(dep in pending_assignments for dep in deps
                   if current_scope.get_defining_scope(dep)):
                for dep in deps:
                    self.compiler.mark_dependency_missing(dep)
                pending_assignments[f"__if_line_{line_number}"] = (
                    guard_line, line_number, deps)
                return False
            for dep in sorted(deps):
                if current_scope.get_defining_scope(dep) is not None:
                    continue
                if dep in pending_assignments:
                    continue
                if dep.lower() in (getattr(self.compiler, 'functions', None) or {}):
                    continue
                if dep.lower() in (getattr(self.compiler, 'subprocesses', None) or {}):
                    continue
                if dep.lower() in (getattr(self.compiler, 'types_defined', None) or {}):
                    continue
                raise NameError(
                    f"Name '{dep}' is not defined at line {line_number}")
            try:
                result = self._evaluate_if_condition(
                    condition, line_number, warn_boolean=True)
            except NameError:
                # The condition names something that does not exist at all.
                # A guard is a condition on state the block depends on, so
                # running (or skipping) the body on an undefined name would
                # hide the typo: report it instead.
                raise
            if not result or self._guard_dependencies_are_errors(condition):
                return False
        return True

    def _register_block_guards(self, block_lines, guards):
        """Record a block's guards for every line of its body.

        The reactive machinery re-evaluates a ``:=`` line whenever one of its
        dependencies changes, which can happen outside the block walk (when the
        loop variable is set). Recording the guards per line lets that
        recomputation honor the same guards.
        """
        entries = [(condition, guard_line_number)
                   for _, guard_line_number, condition in guards]
        self._add_block_guard_entries(block_lines, entries)

    def _add_block_guard_entries(self, block_lines, entries):
        """Extend the guards recorded for each line of a block body."""
        target = getattr(self.compiler, '_block_guards_by_line', None)
        if target is None or not entries:
            return
        for _line, line_number in block_lines:
            existing = target.setdefault(line_number, [])
            for entry in entries:
                if entry not in existing:
                    existing.append(entry)

    def _guard_dependencies_are_errors(self, condition):
        """True when a condition reads a value that is an error.

        A guard over an error cannot hold, so the guarded body is skipped.
        """
        from units import is_error_value
        dep_extractor = getattr(
            self.compiler, '_extract_dependencies_from_expression', None)
        if not callable(dep_extractor):
            return False
        current_scope = self.compiler.current_scope()
        for dep in dep_extractor(condition) or ():
            try:
                defining_scope = current_scope.get_defining_scope(dep)
                if defining_scope is None:
                    continue
                if is_error_value(defining_scope.get(dep)):
                    return True
            except Exception:
                continue
        return False

    def _process_block(self, block_lines, current_scope=None):
        block_pending = {}
        compiler = self.compiler
        saved_wave = compiler._wave_state_snapshot()
        compiler._wave_state_reset()
        try:
            # A block body carries its own row stream, exactly as the program
            # does at the top level: producers, rules, guards and effects are
            # classified at body entry, rules fire per wave, and effects run
            # one round per wave at their source positions.
            wave_claim = compiler._try_execute_block_wave(block_lines)
            # Guards come first: they gate every statement of this body, so they
            # are evaluated before the walk and skipped during it.  Wave guards
            # filter rows inside the wave instead and are not body gates.
            guards = self._collect_block_guards(block_lines)
            if wave_claim is not None:
                guards = [
                    guard for guard in guards
                    if guard[1] not in wave_claim]
            self._register_block_guards(block_lines, guards)
            guard_line_numbers = {line_number for _, line_number, _ in guards}
            if guards and not self._evaluate_block_guards(guards):
                return block_pending
            i = 0
            while i < len(block_lines):
                if self._should_break_block_processing():
                    break
                refresh = getattr(self.compiler, '_refresh_live_inputs', None)
                if refresh is not None:
                    refresh()
                line, line_number, line_clean = self._prepare_block_line(
                    block_lines, i)
                if not line_clean:
                    i += 1
                    continue
                if wave_claim is not None:
                    if compiler._wave_block_step(line_number) == 'skip':
                        i += 1
                        continue
                if line_number in guard_line_numbers:
                    # The guard was already evaluated for the whole body above.
                    # A guard is a bare 'If <cond>' line: it has no 'then' and no
                    # 'End' of its own, so only the guard line is skipped here and
                    # the 'End' that follows still closes the enclosing block.
                    i += 1
                    continue
                if self._token_end.match(line_clean):
                    # A bare 'End' closes an enclosing block whose handler already
                    # ran its body; it is a terminator, not a statement.
                    i += 1
                    continue

                handled, next_i = self._handle_block_when_statement(
                    block_lines, i, line, line_number, line_clean)
                if handled:
                    i = next_i
                    continue
                handled, next_i = self._handle_block_push_or_output_statement(
                    i, line, line_number, line_clean)
                if handled:
                    i = next_i
                    continue
                handled, next_i = self._handle_block_direct_assignment_statement(
                    i, line, line_number, line_clean)
                if handled:
                    i = next_i
                    continue
                handled, next_i = self._handle_block_if_dispatch(
                    block_lines, i, line, line_number, line_clean)
                if handled:
                    i = next_i
                    continue
                handled, next_i = self._handle_block_for_statement(
                    block_lines, i, line, line_number)
                if handled:
                    i = next_i
                    continue
                # Conditional block: 'Let <cond> then' is a guarded body, unlike
                # the declaration form 'Let <var> = ...' handled by the tail.
                if self._header_let.match(line_clean):
                    i = self._handle_let_cond_block(block_lines, i, line_number)
                    continue
                handled, next_i = self._handle_block_cell_binding_or_assignment(
                    block_pending, i, line, line_number, line_clean)
                if handled:
                    i = next_i
                    continue
                handled, next_i, should_return = self._handle_block_tail_statements(
                    block_lines, i, line, line_number, line_clean)
                if should_return:
                    return
                if handled:
                    i = next_i
                    continue

                # 'name = expr' is a Let with the keyword left out. It is matched
                # only once every explicit statement has declined the line, so no
                # keyword form and no other assignment form can be shadowed by it.
                handled, next_i = self._handle_block_optional_let_statement(
                    block_lines, i, line, line_number, line_clean)
                if handled:
                    i = next_i
                    continue

                self.compiler.expr_evaluator.eval_expr(
                    line, self.compiler.current_scope().get_evaluation_scope(), line_number)

                i += 1
            if wave_claim is not None and getattr(
                    compiler, '_wave_flush_pending', False):
                # The body ended right after its last binding: run the
                # effects that were waiting for it.
                compiler._wave_flush_pending = False
                compiler._flush_deferred_wave_effects()
            if block_pending:
                self.compiler.scopes[0].pending_assignments.update(block_pending)
            return block_pending
        finally:
            compiler._wave_state_restore(saved_wave)

    def _process_if_statement(self, line, line_number, lines, current_index):
        """Thin wrapper to keep public IF entrypoint compact."""
        return self._process_if_statement_core(line, line_number, lines, current_index)

    def _process_if_statement_core(self, line, line_number, lines, current_index):
        """Orchestrate IF processing via extracted implementation."""
        return self._process_if_statement_core_impl(line, line_number, lines, current_index)
    def _process_if_statement_core_impl(self, line, line_number, lines, current_index):
        """Process an IF statement with conditions, blocks, and else/elseif clauses."""

        condition, has_block = self._parse_if_header(line, line_number)

        deferred = self._defer_if_if_unresolved(
            condition, line, line_number)
        if deferred is not None:
            return deferred

        try:
            condition_result = self._evaluate_if_condition(
                condition, line_number, warn_boolean=True)
        except NameError as exc:
            dep_extractor = getattr(
                self.compiler, 'extract_missing_dependencies', lambda e: set())
            missing = dep_extractor(exc)
            return self._defer_pending_if(line, line_number, missing)

        if has_block:
            return self._process_if_block_statement(
                condition_result, lines, current_index, line_number)

        self._process_single_line_if_statement(
            condition_result, line, line_number, lines, current_index)
        return 1

    def _parse_if_header(self, line, line_number):
        condition_match = re.match(
            r'^\s*if\s+(.+?)(?:\s+then\s*|\s*$)', line, re.I)
        if not condition_match:
            raise SyntaxError(f"Invalid IF syntax at line {line_number}")
        condition = condition_match.group(1).strip()
        has_block = line.lower().strip().endswith('then')
        return condition, has_block

    def _defer_pending_if(self, line, line_number, deps):
        for dep in deps:
            self.compiler.mark_dependency_missing(dep)
        self.compiler.pending_assignments[f"__if_line_{line_number}"] = (
            line, line_number, deps)
        # Prefer the index-based skip when the block is degenerate (generated
        # lines share the source line number) so the returned count cannot
        # collapse to zero or one and re-execute body lines prematurely.
        if self.if_blocks:
            for block in self.if_blocks:
                if block['start_line'] == line_number:
                    return max(block['end_i'] - block['start_i'] + 1, 1)
        end_line = self.block_map.get(line_number, line_number)
        return max(end_line - line_number + 1, 1)

    def _defer_if_if_unresolved(self, condition, line, line_number):
        dep_extractor = getattr(
            self.compiler, '_extract_dependencies_from_expression', None)
        deps = set(dep_extractor(condition)) if callable(
            dep_extractor) else set()
        current_scope = self.compiler.current_scope()
        unresolved = any(
            self.compiler.has_unresolved_dependency(dep, scope=current_scope)
            for dep in deps
        )
        if unresolved:
            return self._defer_pending_if(line, line_number, deps)
        return None

    def _block_if_guard_deferred(self, condition, line, line_number):
        """Conservative dependency check for a block IF guard.

        Defers the guard (without running any push) only when a dependency is
        a *defined* variable with a pending assignment — i.e. it genuinely
        needs evaluating.  Undefined names and builtins are not treated as
        unresolved here: an undefined name simply evaluates to False, while
        a builtin (mid, sin, ...) resolves against the expression globals.
        """
        dep_extractor = getattr(
            self.compiler, '_extract_dependencies_from_expression', None)
        deps = set(dep_extractor(condition)) if callable(
            dep_extractor) else set()
        current_scope = self.compiler.current_scope()
        for dep in deps:
            defining_scope = current_scope.get_defining_scope(dep)
            if defining_scope and dep in getattr(
                    self.compiler, 'pending_assignments', {}):
                for pending_dep in deps:
                    self.compiler.mark_dependency_missing(pending_dep)
                self.compiler.pending_assignments[
                    f'__if_line_{line_number}'] = (line, line_number, deps)
                return True
        return False

    def _process_if_block_statement(self, condition_result, lines, current_index, line_number):
        if_block_lines, else_block_lines, elseif_blocks, block_i = self._collect_if_blocks(
            lines, current_index, line_number)
        # Same contract as the rich handler: targets assigned inside the
        # block exist (as the ``""`` placeholder) even when the block is
        # skipped, so reads after the IF do not raise NameError.
        self._predeclare_block_assignment_targets(
            if_block_lines + else_block_lines
            + [ln for _, elseif_lines in elseif_blocks
               for ln in elseif_lines])
        self._execute_if_block_choice(
            condition_result, if_block_lines, elseif_blocks, else_block_lines, line_number)
        return block_i - current_index - 1

    def _collect_if_blocks(self, lines, current_index, line_number):
        if_block_lines = []
        else_block_lines = []
        elseif_blocks = []
        current_block = if_block_lines
        block_i = current_index + 1
        depth = 1

        while block_i < len(lines) and depth > 0:
            next_line, next_line_number = lines[block_i]
            next_line_clean = next_line.strip().lower()

            if self.compiler._is_keyword(next_line_clean, "end"):
                depth -= 1
                if depth == 0:
                    block_i += 1
                    break
            elif self.compiler._starts_with_keyword(next_line_clean, "if ") and self.compiler._ends_with_keyword(next_line_clean, "then"):
                depth += 1
            elif re.match(r'^\s*(?:let\b.+\bthen|for\b.+\bdo|when\b.+\bdo)\s*$',
                          next_line, re.I):
                # Inner blocks (``Let ... then``, ``For ... do``) own an ``End``
                # too; without this depth bump the inner ``End`` closes the
                # outer IF and the remaining clauses leak out of the block.
                depth += 1
            elif self.compiler._starts_with_keyword(next_line_clean, "elseif ") and self.compiler._ends_with_keyword(next_line_clean, "then") and depth == 1:
                elseif_condition = re.match(
                    r'^\s*elseif\s+(.+?)\s+then\s*$', next_line, re.I)
                if elseif_condition:
                    elseif_cond = elseif_condition.group(1).strip()
                    elseif_result = self._evaluate_if_condition(
                        elseif_cond, next_line_number, warn_boolean=True)
                    elseif_blocks.append((elseif_result, []))
                    current_block = elseif_blocks[-1][1]
            elif self.compiler._starts_with_keyword(next_line_clean, "else") and depth == 1:
                current_block = else_block_lines
                current_block.append((next_line, next_line_number))
                block_i += 1
                continue
            # A nested clause header (ELSEIF/ELSE of an inner IF) is ordinary
            # content here: the nested IF's own handler splits it when the
            # body runs, and it must not change this walk's depth.
            elif next_line_clean.startswith("for "):
                if self._handle_special_nested_for_element_line(next_line, next_line_number):
                    current_block.append((next_line, next_line_number))
                    block_i += 1
                    continue
            elif next_line_clean == "next":
                current_block.append((next_line, next_line_number))
                block_i += 1
                continue

            current_block.append((next_line, next_line_number))
            block_i += 1

        if depth > 0:
            raise SyntaxError(
                f"Unclosed IF block starting at line {line_number}")

        return if_block_lines, else_block_lines, elseif_blocks, block_i

    def _handle_special_nested_for_element_line(self, next_line, next_line_number):
        if "for element = haystack(mid)" not in next_line:
            return False
        try:
            existing_element = self.compiler.current_scope().get("element")
            if existing_element is not None:
                return True
        except NameError:
            pass

        try:
            scope = self.compiler.current_scope().get_evaluation_scope()
            index_value = self.compiler.expr_evaluator.eval_expr(
                "mid", scope, next_line_number)
            array = self.compiler.current_scope().get("haystack")
            if array is None:
                raise ValueError("Array 'haystack' is not defined")
            if isinstance(array, set):
                array = list(array)
            if not isinstance(index_value, (int, float)):
                raise ValueError(
                    f"Invalid index type {type(index_value)} for array access")
            idx = int(index_value) - 1
            if not (0 <= idx < len(array)):
                raise IndexError(
                    f"Index {index_value} out of range for array 'haystack'")
            element_value = array[idx]
            self.compiler.current_scope().define(
                "element", element_value, 'number', {}, is_uninitialized=False)
            return True
        except Exception as e:
            raise ValueError(
                f"Failed to process nested array element assignment: {e}")

    def _execute_if_block_choice(self, condition_result, if_block_lines, elseif_blocks, else_block_lines, line_number):
        # _collect_if_blocks keeps the ELSE/ELSEIF header line inside the
        # clause body; strip it so the body is pure statements.
        def strip_header(block_lines):
            if block_lines:
                first = block_lines[0][0].strip().lower()
                if first == 'else' or first.startswith('elseif '):
                    return block_lines[1:]
            return block_lines

        if condition_result:
            self._execute_block_with_scope_transfer(if_block_lines, "IF")
            return

        for elseif_result, elseif_lines in elseif_blocks:
            if elseif_result:
                self._execute_block_with_scope_transfer(
                    strip_header(elseif_lines), "ELSEIF")
                return

        if else_block_lines:
            self._execute_block_with_scope_transfer(
                strip_header(else_block_lines), "ELSE")

    def _execute_block_with_scope_transfer(self, block_lines, label):
        self.compiler.push_scope(is_private=False)
        self._process_block(block_lines, self.compiler.current_scope())
        block_scope = self.compiler.current_scope()
        self.compiler.pop_scope()
        parent_scope = self.compiler.current_scope()
        local_lets = getattr(
            block_scope, 'local_let_declarations', set())
        for var_name, var_value in block_scope.variables.items():
            if var_name.lower() in local_lets:
                # LET-declared variables are block-local: they must not leak
                # into the enclosing scope (unlike For and ":" declarations).
                continue
            parent_scope.variables[var_name] = var_value
            if var_name in parent_scope.uninitialized:
                parent_scope.uninitialized.remove(var_name)

    def _process_single_line_if_statement(self, condition_result, line, line_number, lines, current_index):
        if not condition_result:
            else_part = self._extract_inline_clause(line, 'else')
            if else_part:
                self._execute_inline_if_action(else_part, line_number)
            return

        then_part = self._extract_inline_clause(line, 'then')
        if then_part:
            self._execute_inline_if_action(then_part, line_number)

        if current_index + 2 < len(lines):
            next_line, _ = lines[current_index + 1]
            if next_line.strip().lower().startswith('else'):
                else_line, else_line_number = lines[current_index + 2]
                self._execute_inline_if_action(else_line, else_line_number)

    def _extract_inline_clause(self, line, keyword):
        match = re.search(rf'\b{keyword}\b\s*(.+)$', line, re.I)
        return match.group(1).strip() if match else None

    def _execute_inline_if_action(self, action_line, line_number):
        action = action_line.strip()
        if not action:
            return
        if action.lower().startswith('output '):
            try:
                def_str = action[len('output '):].strip()
                self._process_output_statement(def_str, line_number)
            except Exception as e:
                raise
            return
        if ':=' in action:
            self.compiler.array_handler.evaluate_line_with_assignment(
                action, line_number, self.compiler.current_scope().get_evaluation_scope())
            return
        if action.lower().startswith('let '):
            self._process_let_statement_inline(action, line_number)
            return

        push_match = self._match_push_assignment(action)
        return_match = self._match_return_statement(action)
        if push_match:
            target, value_expr = self._unpack_push_assignment(push_match)
            self.compiler._handle_push_assignment(target, value_expr, line_number)
            return
        if return_match:
            self._handle_return_statement(
                return_match.group(1).strip(), line_number)
            return

        # Any other single-line action (bare subprocess/handle member call,
        # plain expression) is evaluated like a normal block line.
        self.compiler.expr_evaluator.eval_expr(
            action, self.compiler.current_scope().get_evaluation_scope(),
            line_number)

    def _process_let_statement_inline(self, line, line_number):
        """Bind a LET declaration appearing inside a block body.

        A block body re-executes on every iteration of its enclosing loop, so
        a declaration there re-binds rather than pins: it is a mutable
        assignment with the LET declaration syntax, not the immutable
        equality binding the top-level ``_bind_declared_var`` performs.
        """
        m = re.match(r'^\s*LET\s+(.+?)(?:\s+then\s*$|\s*$)', line, re.I)
        if not m:
            raise SyntaxError(f"Invalid LET syntax at line {line_number}")

        var_def = m.group(1).strip()
        var_list = []
        if ' and ' in var_def.lower():
            var_parts = split_var_defs(var_def)
            for var_part in var_parts:
                var, type_name, constraints, expr = self.compiler._parse_variable_def(
                    var_part, line_number)
                var_list.append((var, type_name, constraints, expr))
        else:
            var, type_name, constraints, expr = self.compiler._parse_variable_def(
                var_def, line_number)
            var_list.append((var, type_name, constraints, expr))

        scope_dict = self.compiler.current_scope().get_evaluation_scope()
        for var, type_name, constraints, expr in var_list:
            if self._try_process_let_field_index_assignment(
                    var, expr, scope_dict, line_number):
                continue
            if self._try_process_let_indexed_assignment(
                    var, expr, scope_dict, line_number):
                continue
            self._bind_let_var_in_block(
                var, type_name, constraints, expr, scope_dict, line_number)
            # A loop body re-executes every iteration, so a `Let` there binds
            # fresh each time instead of pinning one immutable value. The
            # loop copy-back drops the variable, so re-binding a name across
            # iterations is a new binding rather than a conflicting one.

    def _bind_let_var_in_block(self, var, type_name, constraints, expr,
                               scope_dict, line_number):
        """Bind a LET that appears inside a block body.

        Uses the same binder as a top-level LET, so the two are one
        instruction. The difference is scope, not semantics: a block body
        re-executes on every iteration of its enclosing loop, so
        ``let a = f(i)`` binds a new value each pass. That works because
        ``_bind_declared_var`` marks a loop-body declaration in
        ``local_let_declarations``, and the loop's copy-back to the parent
        scope skips exactly that set -- the variable is dropped when the
        iteration ends, so each iteration binds afresh rather than colliding
        with its own previous value.
        """
        self.compiler._bind_declared_var(
            var, type_name, constraints, expr, line_number,
            scope_dict=scope_dict, shadow_keyword='LET', mark_implicit=True)

    def _try_process_let_field_index_assignment(self, var, expr, scope_dict, line_number):
        field_index_match = re.match(
            r'^([\w_]+)\.([\w_]+)\(([^)]+)\)\.(\w+)$', var)
        if not field_index_match:
            return False
        obj_name, array_field, index_expr, field_name = field_index_match.groups()
        value = self.compiler.expr_evaluator.eval_or_eval_array(
            expr, scope_dict, line_number)
        index_value = self.compiler.expr_evaluator.eval_expr(
            index_expr, scope_dict, line_number)
        if isinstance(index_value, float) and index_value.is_integer():
            index_value = int(index_value)
        if not isinstance(index_value, int):
            raise ValueError(
                f"Invalid index '{index_expr}' for '{var}' at line {line_number}")
        indices = [index_value - 1]
        obj_value = self.compiler.current_scope().get(obj_name)
        if not isinstance(obj_value, dict):
            raise ValueError(
                f"'{obj_name}' is not an object at line {line_number}")
        public_keys = {k.lower(): k for k in obj_value.keys()
                       if not str(k).startswith('_')}
        actual_array_field = public_keys.get(
            array_field.lower(), array_field)
        array_val = obj_value.get(actual_array_field)
        if array_val is None:
            raise ValueError(
                f"Field '{array_field}' is not defined on '{obj_name}' at line {line_number}")
        array_storage = array_val['array'] if isinstance(
            array_val, dict) and 'array' in array_val else array_val
        element = self.compiler.array_handler.get_array_element(
            array_storage, indices, line_number)
        if not isinstance(element, dict):
            raise ValueError(
                f"'{obj_name}.{array_field}({index_value})' is not an object at line {line_number}")
        element_keys = {k.lower(): k for k in element.keys()
                        if not str(k).startswith('_')}
        actual_field = element_keys.get(field_name.lower(), field_name)
        element[actual_field] = value
        updated_array = self.compiler.array_handler.set_array_element(
            array_storage, indices, element, line_number)
        if isinstance(array_val, dict) and 'array' in array_val:
            array_val['array'] = updated_array
            obj_value[actual_array_field] = array_val
        else:
            obj_value[actual_array_field] = updated_array
        self.compiler.current_scope().update(
            obj_name, obj_value, line_number)
        scope_dict[obj_name] = obj_value
        return True

    def _try_process_let_indexed_assignment(self, var, expr, scope_dict, line_number):
        var_name, indices = self.compiler.expr_evaluator._parse_index_target(
            var, scope_dict, line_number)
        if var_name is None:
            return False
        value = self.compiler.expr_evaluator.eval_or_eval_array(
            expr, scope_dict, line_number)
        defining_scope = self.compiler.current_scope().get_defining_scope(var_name)
        if defining_scope:
            if (getattr(self.compiler, '_outer_scope_read_only', False)
                    and self.compiler._is_outer_scope(defining_scope)):
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
            dim_spec = constraints.get('dim')
            has_star = False
            if isinstance(dim_spec, list):
                has_star = any(size_spec is None for _, size_spec in dim_spec)
            elif isinstance(dim_spec, str):
                has_star = '*' in dim_spec
            if has_star and (defining_scope.is_uninitialized(actual_key) or arr is None):
                arr = {}
                defining_scope.variables[actual_key] = arr
            updated_array = self.compiler.array_handler.set_array_element(
                arr, indices, value, line_number)
            defining_scope.variables[actual_key] = updated_array
            scope_dict[actual_key] = updated_array
            return True
        return True

    def _evaluate_if_condition(self, condition, line_number, warn_boolean=False):
        """
        Evaluate an IF condition, handling various types of constraints.
        """

        # A leading '?' is the truth-test prefix used in value context
        # ('[B1] := ? a = b'); inside a condition (If/When/guard) it is
        # redundant, so tolerate it as if it were absent.
        condition = condition.strip()
        if condition.startswith('?'):
            condition = condition[1:].strip()

        # Instructions (Let, For, If, ...) may only start a statement, so
        # one appearing inside an expression is invalid.
        instruction_match = re.match(
            r'^(let|for|if|when|input|output|declare)\b', condition, re.I)
        if instruction_match:
            raise SyntaxError(
                f"'{instruction_match.group(1)}' is an instruction and must "
                f"be at the start of a line, or after 'Then', 'Else', 'Do' "
                f"or ':' at line {line_number}")

        # '!=' is not a GridLang operator; use '<>' for inequality
        without_strings = re.sub(
            r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', '', condition)
        if '!=' in without_strings:
            raise SyntaxError(
                f"'!=' is not a GridLang operator; use '<>' for inequality "
                f"at line {line_number}")

        # '==' is not a GridLang operator; a bare comparison is an assignment,
        # so equality must be requested explicitly with '? a = b'.
        if '==' in without_strings:
            raise SyntaxError(
                f"'==' is not a GridLang operator; use '? a = b' to get a "
                f"logical value at line {line_number}")

        # Handle "not as" type constraints (e.g., "z not as text")
        not_type_match = re.match(
            r'^([\w_]+)\s+not\s+as\s+(\w+)(?:\s+dim\s+(.+))?$', condition, re.I)
        if not_type_match:
            var_name, type_name, dim_constraint = not_type_match.groups()
            positive = f"{var_name} as {type_name}"
            if dim_constraint:
                positive = f"{positive} dim {dim_constraint}"
            return not self._evaluate_if_condition(positive, line_number)

        # Handle "not of" unit constraints (e.g., "z not of dollar")
        not_unit_match = re.match(
            r'^([\w_]+)\s+not\s+of\s+(\w+)$', condition, re.I)
        if not_unit_match:
            var_name, unit_name = not_unit_match.groups()
            defining_scope = self.compiler.current_scope().get_defining_scope(var_name)
            if not defining_scope:
                raise NameError(
                    f"Variable '{var_name}' not defined at line {line_number}")
            return self._current_unit_of(defining_scope, var_name) != unit_name.lower()

        # Handle "not null" constraints (optionally with equality)
        not_null_eq_match = re.match(
            r'^([\w_]+)\s+not\s+null\s*=\s*(.+)$', condition, re.I)
        if not_null_eq_match:
            var_name, rhs = not_null_eq_match.groups()
            if not self._evaluate_if_condition(f"{var_name} not null", line_number):
                return False
            scope = self.compiler.current_scope().get_evaluation_scope()
            left_val = self.compiler.expr_evaluator.eval_expr(
                var_name, scope, line_number)
            right_val = self.compiler.expr_evaluator.eval_expr(
                rhs.strip(), scope, line_number)
            return left_val == right_val
        not_null_match = re.match(r'^([\w_]+)\s+not\s+null$', condition, re.I)
        if not_null_match:
            var_name = not_null_match.group(1)
            defining_scope = self.compiler.current_scope().get_defining_scope(var_name)
            if not defining_scope:
                raise NameError(
                    f"Variable '{var_name}' not defined at line {line_number}")
            value = defining_scope.get(var_name)
            return value is not None

        # A bare variable condition tests presence, not truthiness: a falsy
        # value (0, "") still triggers, equivalent to '<var> not null'. For a
        # logical variable this is ambiguous, so warn once suggesting an
        # explicit '<var> = true' or '<var> not null'.
        bare_match = re.match(r'^([\w_.]+)$', condition.strip(), re.I)
        if bare_match:
            var_name = bare_match.group(1)
            defining_scope = self.compiler.current_scope().get_defining_scope(
                var_name)
            if defining_scope:
                value = defining_scope.get(var_name)
                present = value is not None and not is_error_value(value)
                actual_key = defining_scope._get_case_insensitive_key(
                    var_name, defining_scope.types)
                var_type = defining_scope.types.get(
                    actual_key) if actual_key else None
                if present and (var_type == 'logical'
                                or isinstance(value, bool)):
                    self._warn_boolean_condition(var_name, line_number)
                return present

        # Handle negated comparisons (e.g., "x not <= 10")
        not_cmp_match = re.match(
            r'^(.+?)\s+not\s*(<=|>=|<|>|=)\s*(.+)$', condition.strip(), re.I)
        if not_cmp_match:
            left_expr, operator, right_expr = not_cmp_match.groups()
            positive = f"{left_expr.strip()} {operator} {right_expr.strip()}"
            return not self._evaluate_if_condition(positive, line_number)

        handled, result = self._evaluate_if_type_constraint(
            condition, line_number)
        if handled:
            return result
        handled, result = self._evaluate_if_dim_constraint(condition, line_number)
        if handled:
            return result
        handled, result = self._evaluate_if_unit_constraint(
            condition, line_number)
        if handled:
            return result
        handled, result = self._evaluate_if_in_constraint(condition, line_number)
        if handled:
            return result

        handled, result = self._evaluate_if_logical_or_comparison(
            condition, line_number)
        if handled:
            return result
        return self._evaluate_if_condition_default(
            condition, line_number, warn_boolean=warn_boolean)

    def _warn_boolean_condition(self, expr, line_number):
        """Warn once per line: a bare variable condition tests presence (any
        value, including falsy 0/""/false, satisfies it), which reads like a
        truthiness test for a boolean value; suggest an explicit equality or a
        'not null' presence test."""
        warned = getattr(self.compiler, '_bool_condition_warned_lines', None)
        if warned is None:
            warned = set()
            self.compiler._bool_condition_warned_lines = warned
        if line_number in warned:
            return
        warned.add(line_number)
        print(
            f"Warning: boolean condition at line {line_number}; "
            f"use '{expr} = true' to test the value, or '{expr} not null' "
            f"to test presence (any value, including {expr} = false)")

    def _resolve_implicit_operands(self, parts):
        """Rewrite comparison parts that omit the left operand (e.g. 'x >= 1 and < 8')
        by carrying over the left operand from the preceding part."""
        resolved = []
        last_left = None
        for part in parts:
            part = part.strip()
            if last_left is not None and re.match(
                    r'^(?:<>|<=|>=|=|<|>)\s*\S', part, re.I):
                resolved.append(f"{last_left} {part}")
                continue
            cmp_match = re.match(
                r'^(.+?)\s*(?:<>|<=|>=|=|<|>)\s*(.+)$', part, re.I)
            last_left = cmp_match.group(1).strip() if cmp_match else None
            resolved.append(part)
        return resolved

    def _evaluate_if_logical_or_comparison(self, condition, line_number):
        if ' and ' in condition.lower() and not re.search(r'[<>=!]\s*and\s*[<>=!]', condition):
            parts = re.split(r'\s+and\s+', condition, flags=re.I)
            parts = self._resolve_implicit_operands(parts)
            return True, all(self._evaluate_if_condition(part.strip(), line_number) for part in parts)
        if ' or ' in condition.lower() and not re.search(r'[<>=!]\s*or\s*[<>=!]', condition):
            parts = re.split(r'\s+or\s+', condition, flags=re.I)
            parts = self._resolve_implicit_operands(parts)
            return True, any(self._evaluate_if_condition(part.strip(), line_number) for part in parts)
        handled, result = self._evaluate_if_comparison(condition, line_number)
        if handled:
            return True, result
        handled, result = self._evaluate_if_not_equal(condition, line_number)
        if handled:
            return True, result
        return False, None

    def _evaluate_if_comparison(self, condition, line_number):
        comparison_match = re.match(
            r'^(.+?)\s*(?<!not\s)(<>|<=|>=|=|<|>)(?!\s*=)\s*(?!.*\s+(?:and|or)\s+)(.+)$', condition.strip(), re.I)
        if not comparison_match:
            return False, None
        left_expr, operator, right_expr = comparison_match.groups()
        current_scope = self.compiler.current_scope()
        scope = current_scope.get_evaluation_scope()
        scope.setdefault('true', True)
        scope.setdefault('false', False)
        scope.setdefault('none', UNIVERSAL_ZERO)
        if left_expr.strip() in current_scope.variables:
            scope[left_expr.strip()] = current_scope._wrap_for_eval(
                left_expr.strip(), current_scope.variables[left_expr.strip()])
        try:
            if operator == '=':
                dim_selector_present = (
                    re.search(r'[\w_]+!\w+\([^)]*\)', left_expr) or
                    re.search(r'[\w_]+!\w+\([^)]*\)', right_expr)
                )
                if dim_selector_present:
                    left_val = self.compiler.expr_evaluator.eval_expr(
                        left_expr.strip(), scope, line_number)
                    right_val = self.compiler.expr_evaluator.eval_expr(
                        right_expr.strip(), scope, line_number)
                    return True, left_val == right_val
                try:
                    full_expr = f"{left_expr.strip()} == {right_expr.strip()}"
                    result = self.compiler.expr_evaluator.eval_expr(
                        full_expr, scope, line_number)
                    return True, bool(result)
                except Exception as e:
                    left_val = self.compiler.expr_evaluator.eval_expr(
                        left_expr.strip(), scope, line_number)
                    right_val = self.compiler.expr_evaluator.eval_expr(
                        right_expr.strip(), scope, line_number)
                    return True, left_val == right_val
            left_val = self.compiler.expr_evaluator.eval_expr(
                left_expr.strip(), scope, line_number)
            right_val = self.compiler.expr_evaluator.eval_expr(
                right_expr.strip(), scope, line_number)
            if operator == '<>':
                return True, left_val != right_val
            if operator == '<=':
                return True, left_val <= right_val
            if operator == '>=':
                return True, left_val >= right_val
            if operator == '<':
                return True, left_val < right_val
            if operator == '>':
                return True, left_val > right_val
            return True, False
        except Exception as e:
            return True, False

    def _evaluate_if_not_equal(self, condition, line_number):
        not_equal_match = re.search(r'not\s*=', condition, re.I)
        if not not_equal_match:
            return False, None
        parts = re.split(r'\s+not\s*=\s+', condition, flags=re.I)
        if len(parts) != 2:
            return True, False
        left_expr, right_expr = parts
        scope = self.compiler.current_scope().get_evaluation_scope()
        try:
            left_val = self.compiler.expr_evaluator.eval_expr(
                left_expr.strip(), scope, line_number)
            right_val = self.compiler.expr_evaluator.eval_expr(
                right_expr.strip(), scope, line_number)
            return True, left_val != right_val
        except Exception as e:
            return True, False

    def _evaluate_if_condition_default(self, condition, line_number, warn_boolean=False):
        try:
            scope = self.compiler.current_scope().get_evaluation_scope()
            result = self.compiler.expr_evaluator.eval_expr(
                condition, scope, line_number)
            if warn_boolean and isinstance(result, bool):
                self._warn_boolean_condition(condition, line_number)
            return bool(result)
        except Exception as e:
            return False

    def _evaluate_if_type_constraint(self, condition, line_number):
        type_match = re.match(
            r'^([\w_]+)\s+as\s+(\w+)(?:\s+dim\s+(.+))?$', condition, re.I)
        if not type_match:
            return False, None
        var_name, type_name, dim_constraint = type_match.groups()
        defining_scope = self.compiler.current_scope().get_defining_scope(var_name)
        if not defining_scope:
            current_scope = self.compiler.current_scope()
            while current_scope:
                if var_name in current_scope.variables or var_name in current_scope.types or var_name in current_scope.constraints:
                    defining_scope = current_scope
                    break
                current_scope = current_scope.parent
            if not defining_scope:
                root_scope = self.compiler.current_scope()
                while root_scope.parent:
                    root_scope = root_scope.parent
                if var_name in root_scope.variables or var_name in root_scope.types or var_name in root_scope.constraints:
                    defining_scope = root_scope
                else:
                    raise NameError(
                        f"Variable '{var_name}' not defined at line {line_number}")
        var_value = defining_scope.get(var_name)
        if var_value is None:
            declared_type = ''
            if hasattr(defining_scope, 'types'):
                type_key = defining_scope._get_case_insensitive_key(
                    var_name, defining_scope.types)
                if type_key:
                    declared_type = (defining_scope.types.get(
                        type_key) or '').lower()
            return True, declared_type == type_name.lower()
        actual_type = self.compiler.array_handler.infer_type(var_value, line_number)
        flat_value = self.compiler.array_handler.flatten_array(
            var_value, line_number) if (
                isinstance(var_value, dict) and ('array' in var_value
                                                or is_sparse_array(var_value))
            ) else var_value
        if type_name.lower() == 'text':
            if not (actual_type == 'text' or (isinstance(flat_value, (list, tuple)) and all(isinstance(item, str) for item in flat_value))):
                return True, False
        elif type_name.lower() == 'number':
            if not (actual_type in ('number', 'int') or (isinstance(flat_value, (list, tuple)) and all(isinstance(item, (int, float)) for item in flat_value))):
                return True, False
        is_array = isinstance(var_value, (list, tuple)) or (
            isinstance(var_value, dict) and 'array' in var_value) or (
            isinstance(var_value, dict) and var_value
            and all(isinstance(k, tuple) for k in var_value.keys()))
        if dim_constraint:
            if dim_constraint == '{}':
                return True, not is_array
            if dim_constraint == '*':
                return True, is_array
            try:
                dim_parts = dim_constraint[1:-1].split(',')
                expected_dims = [int(d.strip()) for d in dim_parts]
                if is_array:
                    actual_dims = self.compiler.array_handler.get_array_dimensions(
                        var_value)
                    return True, actual_dims == expected_dims
                return True, expected_dims == [1]
            except Exception:
                return True, False
        return True, True

    def _evaluate_if_dim_constraint(self, condition, line_number):
        dim_match = re.match(r'^([\w_]+)\s+dim\s+(.+)$', condition, re.I)
        if not dim_match:
            return False, None
        var_name, dim_spec = dim_match.groups()
        eval_scope = self.compiler.current_scope().get_evaluation_scope()
        if var_name not in eval_scope:
            raise NameError(
                f"Variable '{var_name}' not defined at line {line_number}")
        var_value = eval_scope[var_name]
        is_array = isinstance(var_value, (list, tuple)) or (
            isinstance(var_value, dict) and 'array' in var_value) or (
            isinstance(var_value, dict) and var_value
            and all(isinstance(k, tuple) for k in var_value.keys()))
        if dim_spec.strip() == '*':
            return True, is_array
        if dim_spec.strip() == '{}':
            return True, not is_array
        if dim_spec.strip().isdigit():
            expected_dim = int(dim_spec.strip())
            if is_array:
                actual_dims = self.compiler.array_handler.get_array_dimensions(
                    var_value)
                return True, bool(actual_dims) and actual_dims[0] == expected_dim
            return True, expected_dim == 0
        if dim_spec.strip().startswith('{') and dim_spec.strip().endswith('}'):
            try:
                dim_str = dim_spec.strip()[1:-1]
                expected_dims = [int(d.strip()) for d in dim_str.split(',')]
                if is_array:
                    actual_dims = self.compiler.array_handler.get_array_dimensions(
                        var_value)
                    while actual_dims and actual_dims[0] == 1:
                        actual_dims = actual_dims[1:]
                    return True, actual_dims == expected_dims
                return True, expected_dims == []
            except Exception:
                return True, False
        return True, False

    def _current_unit_of(self, defining_scope, var_name):
        """Runtime unit of a variable, falling back to its declared unit."""
        runtime = defining_scope.get_value_unit(var_name)
        if runtime is not None:
            return str(runtime).lower()
        constraint_key = defining_scope._get_case_insensitive_key(
            var_name, defining_scope.constraints)
        if constraint_key:
            declared = defining_scope.constraints[constraint_key].get('unit')
            return str(declared).lower() if declared else None
        return None

    def _evaluate_if_unit_constraint(self, condition, line_number):
        unit_match = re.match(r'^([\w_]+)\s+of\s+(\w+)$', condition, re.I)
        if not unit_match:
            return False, None
        var_name, unit_name = unit_match.groups()
        defining_scope = self.compiler.current_scope().get_defining_scope(var_name)
        if not defining_scope:
            raise NameError(
                f"Variable '{var_name}' not defined at line {line_number}")
        return True, self._current_unit_of(defining_scope, var_name) == unit_name.lower()

    def _evaluate_if_in_constraint(self, condition, line_number):
        in_match = re.match(r'^([\w_]+)\s+in\s+(.+)$', condition, re.I)
        if not in_match:
            return False, None
        var_name, constraint_expr = in_match.groups()
        scope = self.compiler.current_scope().get_evaluation_scope()
        var_value = None
        for key in scope.keys():
            if key == var_name:
                var_value = scope[key]
                break
        if var_value is None:
            raise NameError(
                f"Variable '{var_name}' not defined at line {line_number}")
        if constraint_expr.startswith('{') and constraint_expr.endswith('}'):
            values_str = constraint_expr[1:-1].split(',')
            constraint_values = []
            for v in values_str:
                v_clean = v.strip()
                if v_clean.startswith('"') and v_clean.endswith('"'):
                    constraint_values.append(v_clean[1:-1])
                elif v_clean.startswith("'") and v_clean.endswith("'"):
                    constraint_values.append(v_clean[1:-1])
                else:
                    try:
                        constraint_values.append(float(v_clean))
                    except ValueError:
                        constraint_values.append(v_clean)
            return True, var_value in constraint_values
        if ' to ' in constraint_expr:
            range_match = re.match(r'^(\d+)\s+to\s+(\d+)$', constraint_expr)
            if range_match:
                start, end = map(int, range_match.groups())
                try:
                    var_num = float(var_value)
                    return True, start <= var_num <= end
                except (ValueError, TypeError):
                    return True, False
        return True, False

    def _resolve_global_dependency(self, var, line_number, target_scope=None):
        if var not in self.compiler.pending_assignments:
            return False
        expr, assign_line, deps = self.compiler.pending_assignments[var]
        scope = target_scope if target_scope is not None else self.compiler.current_scope()
        unresolved = any(
            dep != var and self.compiler.has_unresolved_dependency(
                dep, scope=scope)
            for dep in deps)
        if unresolved:
            return False
        try:
            value = self.compiler.expr_evaluator.eval_or_eval_array(
                expr, scope.get_full_scope(), assign_line)
            value = self.compiler.array_handler.check_dimension_constraints(
                var, value, assign_line)
            defining_scope = scope.get_defining_scope(var)
            if not defining_scope:
                defining_scope = self.compiler.current_scope()
            defining_scope.update(var, value, assign_line)
            del self.compiler.pending_assignments[var]
            return True
        except ValueError as e:
            if is_definition_body_error(e):
                raise RuntimeError(
                    f"Error resolving global dependency '{var}': {e} at line {assign_line}")
            del self.compiler.pending_assignments[var]
            self.compiler.grid.clear()
            return False
        except NameError as e:
            if is_definition_body_error(e):
                raise RuntimeError(
                    f"Error resolving global dependency '{var}': {e} at line {assign_line}")
            return False
        except Exception as e:
            raise RuntimeError(
                f"Error resolving global dependency '{var}': {e} at line {assign_line}")

    def pre_scan_blocks(self, lines):
        """First pass: build a rich structure of blocks and clause spans."""
        # {start_line: end_line} (kept for compatibility)
        self.block_map = {}
        # list of {start_line, end_line, clauses: [...]}
        self.if_blocks = []
        self.loops = []                  # list of {type, start_line, end_line}
        # Cache the full list of lines (line content + original numbers) so nested
        # IF handlers can refer back to the source regardless of the current slice.
        self._preprocessed_lines = list(lines)

        stack = []  # Stack of dicts: {type, start_i, start_line, clauses?}

        # Validation-only nesting stack used to reject an 'End' that closes
        # nothing.  It is deliberately separate from the structural stack
        # above: 'let ... then' blocks are executed by a different code path
        # and are not part of block_map, but they still consume an 'End'.
        open_stack = []

        # Lines that sit inside a structural block body. Their cell-spill
        # listeners are installed when the block actually executes, so a block
        # whose guard fails never registers one.
        block_body_lines = set()

        for i, (line, line_number) in enumerate(lines):
            raw = line
            line = self._strip_inline_comment(line)
            line_clean = line.strip()

            if not line_clean:
                continue

            # A block ends with exactly 'end'.  A keyword after 'End' matches
            # neither a block header nor the 'end' token, so it would be
            # skipped silently and the block it appears to close would stay
            # open (or, with nothing open, simply vanish).
            if self._token_end_keyword.match(line_clean):
                raise SyntaxError(
                    f"'{line_clean}' is not a terminator at line "
                    f"{line_number}: a block ends with 'end'")

            if open_stack and not self._token_end.match(line_clean):
                block_body_lines.add(line_number)

            # IF
            if self._header_if.match(line_clean):
                block = {
                    'type': 'if',
                    'start_i': i,
                    'start_line': line_number,
                    'clauses': [{'kind': 'if', 'header_i': i, 'header_line': line_number}]
                }
                stack.append(block)
                open_stack.append('if')
                continue

            # ELSEIF
            if self._header_elseif.match(line_clean):
                if stack and stack[-1]['type'] == 'if':
                    cur = stack[-1]
                    cur['clauses'].append(
                        {'kind': 'elseif', 'header_i': i, 'header_line': line_number})
                continue

            # ELSE
            if self._header_else.match(line_clean):
                if stack and stack[-1]['type'] == 'if':
                    cur = stack[-1]
                    cur['clauses'].append(
                        {'kind': 'else', 'header_i': i, 'header_line': line_number})
                continue

            # FOR
            if self._header_for.match(line_clean):
                stack.append({'type': 'for', 'start_i': i,
                              'start_line': line_number})
                open_stack.append('for')
                continue

            # WHEN
            if self._header_when.match(line_clean):
                stack.append({'type': 'when', 'start_i': i,
                              'start_line': line_number})
                open_stack.append('when')
                continue

            # LET ... then
            if self._header_let.match(line_clean):
                open_stack.append('let')
                continue


            # END
            if self._token_end.match(line_clean):
                if not open_stack:
                    raise SyntaxError(
                        f"'End' without a matching block header at line "
                        f"{line_number}")
                open_stack.pop()
                if not stack:
                    continue
                block = stack.pop()
                start_line = block['start_line']
                self.block_map[start_line] = line_number
                btype = block['type']

                if btype == 'if':
                    # finalize clause spans: each clause's body is between its header and the next clause header/end
                    clauses = block['clauses']
                    # Calculate body ranges using indices
                    boundaries = [c['header_i']
                                  # end index 'i' is END
                                  for c in clauses] + [i]
                    clause_spans = []
                    for idx, c in enumerate(clauses):
                        body_start = boundaries[idx] + 1
                        body_end = boundaries[idx+1] - 1
                        clause_spans.append({
                            'kind': c['kind'],
                            'header_line': c['header_line'],
                            'body_start_i': body_start,
                            'body_end_i': body_end
                        })
                    self.if_blocks.append({
                        'start_line': start_line,
                        'end_line': line_number,
                        'start_i': block['start_i'],
                        'end_i': i,
                        'clauses': clause_spans
                    })
                else:
                    self.loops.append({
                        'type': btype,
                        'start_line': start_line,
                        'end_line': line_number
                    })
                continue

            # otherwise just a normal line; nothing to do in pre-scan

        # Any unclosed blocks?
        self.block_body_lines = block_body_lines
        return self.block_map

    def _process_if_elseif_else_block(self, block_lines, if_condition_result, if_line_number, if_block, if_condition=None):
        """Process IFâ€“ELSEIFâ€“ELSE blocks, with correct clause selection and nested IF support."""

        active_clause_lines = []
        clause_found = False
        in_active_clause = False
        depth = 0  # nested IF depth
        # Condition of the clause that ends up running, so reactive
        # recomputation of its bindings honors it too.
        active_clause_condition = None
        active_clause_header = None
        if if_condition:
            active_clause_condition = if_condition.strip()
            active_clause_header = if_line_number
            # The IF clause body ends at the first depth-0 ELSEIF/ELSE. Its
            # condition guards that body on every iteration, including the
            # iterations where another clause runs, so record it for those
            # lines: the reactive recomputation of a binding happens outside
            # this walk and must honor the same condition.
            if_clause_lines = []
            clause_depth = 0
            scan_lines = block_lines[1:] if (
                block_lines and re.match(
                    r'^\s*if\s+(.+?)\s+then\s*$', block_lines[0][0], re.I)
            ) else block_lines
            for body_line, body_line_number in scan_lines:
                body_clean = self._strip_inline_comment(body_line).strip()
                if clause_depth == 0 and (
                        re.match(r'^\s*elseif\b.+?\bthen\s*$', body_clean, re.I)
                        or re.match(r'^\s*else\s*$', body_clean, re.I)):
                    break
                if self._block_header(body_clean):
                    clause_depth += 1
                elif self._token_end.match(body_clean) and clause_depth > 0:
                    clause_depth -= 1
                if_clause_lines.append((body_line, body_line_number))
            if if_clause_lines:
                self._add_block_guard_entries(
                    if_clause_lines,
                    [(active_clause_condition, active_clause_header)])

        # Check if the main IF clause should be active
        if if_condition_result:
            in_active_clause = True
            clause_found = True

        i = 0
        while i < len(block_lines):
            line, line_number = block_lines[i]
            lower = line.strip().lower()

            # --- Top-level clause headers (only if not inside a nested IF) ---
            if depth == 0 and lower.startswith("if ") and lower.endswith(" then"):
                # This is a nested IF, handle it as nested content
                depth += 1
                if in_active_clause:
                    active_clause_lines.append((line, line_number))

            elif depth == 0 and lower.startswith("elseif ") and lower.endswith(" then"):
                if not clause_found:
                    cond = re.match(r'^\s*elseif\s+(.+?)\s+then\s*$',
                                    line, re.I).group(1).strip()
                    cond_result = self._evaluate_if_condition(
                        cond, line_number, warn_boolean=True)
                    if cond_result:
                        in_active_clause = True
                        clause_found = True
                        active_clause_lines = []
                        active_clause_condition = cond
                        active_clause_header = line_number
                    else:
                        in_active_clause = False
                else:
                    in_active_clause = False

            elif depth == 0 and lower == "else":
                if not clause_found:
                    in_active_clause = True
                    clause_found = True
                    active_clause_lines = []
                    active_clause_condition = None
                    active_clause_header = line_number
                else:
                    in_active_clause = False

            elif depth == 0 and lower == "end":
                break

            else:
                # --- Nested block handling ---
                if lower == "end" and depth > 0:
                    if in_active_clause:
                        active_clause_lines.append((line, line_number))
                    depth -= 1
                else:
                    if in_active_clause:
                        active_clause_lines.append((line, line_number))
                    # Inner block headers (nested IF at depth > 0, ``Let ...
                    # then``, ``For ... do``, ``When ... do``) own an ``End``;
                    # bump depth so their ``End`` does not terminate the
                    # clause walk before the ELSE/ELSEIF is seen.
                    if re.match(r'^\s*(?:if|let|for|when)\b.+\b(?:then|do)\s*$',
                                line, re.I):
                        depth += 1

            i += 1

        # Execute the chosen clause
        if active_clause_lines:
            if active_clause_condition is not None:
                self._add_block_guard_entries(
                    active_clause_lines,
                    [(active_clause_condition, active_clause_header)])
            self._process_block(active_clause_lines,
                                self.compiler.current_scope())

    def _match_global_if_block(self, line_number, lines=None,
                               current_index=None):
        """Find the pre-scanned IF block for *line_number*.

        Constructor and builder bodies are renumbered as ``call_line +
        offset``, so their line numbers can collide with an unrelated main
        level IF. A block whose header in the preprocessed source differs
        from the line at hand is rejected and the caller falls back to a
        local scan.
        """
        for block in self.if_blocks:
            if block.get('start_line') != line_number:
                continue
            pre = self._preprocessed_lines
            if pre and lines is not None and current_index is not None:
                start_i = block.get('start_i', -1)
                if 0 <= current_index < len(lines) and 0 <= start_i < len(pre):
                    if lines[current_index] != pre[start_i]:
                        continue
            return block
        return None

    def _scan_local_if_block(self, line, line_number, lines, current_index):
        """Derive IF/ELSEIF/ELSE clause structure from *lines*.

        Bodies that are executed through :meth:`_process_block` without being
        part of the top-level pre-scan (constructor bodies, nested slices) have
        no entry in ``if_blocks``. This builds the same shape so the shared
        clause-selection code can run unchanged.
        """
        clauses = []
        header_re = re.compile(r'^\s*(if|elseif)\s+(.+?)\s+then\s*$', re.I)
        m = header_re.match(line)
        if not m:
            return None
        clauses.append({
            'kind': m.group(1).lower(),
            'condition': m.group(2).strip(),
            'start': current_index + 1,
        })
        end_i = None
        depth = 0
        for i in range(current_index + 1, len(lines)):
            body = lines[i][0].strip() if isinstance(lines[i], tuple) \
                else str(lines[i]).strip()
            if header_re.match(body) or re.match(r'^\s*for\b.+?\bdo\s*$',
                                                body, re.I) \
                    or re.match(r'^\s*let\b.+?\bthen\s*$', body, re.I) \
                    or re.match(r'^\s*when\b.+?\bdo\s*$', body, re.I):
                depth += 1
            elif self._token_end.match(body):
                if depth == 0:
                    end_i = i
                    break
                depth -= 1
            if depth == 0 and re.match(r'^\s*else\s*$', body, re.I):
                clauses.append({'kind': 'else', 'condition': None,
                                'start': i + 1})
        if end_i is None:
            raise SyntaxError(f"Unclosed IF block at line {line_number}")
        clauses[-1]['end'] = end_i
        return {
            'start_line': line_number,
            'end_line': lines[end_i][1] if isinstance(lines[end_i], tuple)
            else line_number,
            'start_i': current_index,
            'end_i': end_i,
            'clauses': clauses,
            # Indices are relative to *lines* (a constructor body or nested
            # slice), not to the program-wide preprocessed list.
            'local': True,
        }

    def _process_if_statement_rich(self, line, line_number, lines, current_index):
        """Process IF statement using the rich block structure with clause information"""

        # Find the IF block that starts at this line
        if_block = self._match_global_if_block(
            line_number, lines, current_index)

        if not if_block:
            # Blocks running through a nested body (e.g. a constructor body)
            # are not part of the top-level pre-scan, so derive the clause
            # structure from the lines at hand.
            if_block = self._scan_local_if_block(line, line_number,
                                                 lines, current_index)
        if not if_block:
            return 1

        # Parse the condition
        condition_match = re.match(
            r'^\s*if\s+(.+?)(?:\s+then\s*|\s*$)', line, re.I)
        if not condition_match:
            raise SyntaxError(f"Invalid IF syntax at line {line_number}")

        condition = condition_match.group(1).strip()

        dep_extractor = getattr(
            self.compiler, '_extract_dependencies_from_expression', None)
        deps = set(dep_extractor(condition)) if callable(
            dep_extractor) else set()
        current_scope = self.compiler.current_scope()
        unresolved = False
        for dep in deps:
            defining_scope = current_scope.get_defining_scope(dep)
            if defining_scope and dep in getattr(self.compiler, 'pending_assignments', {}):
                unresolved = True
                break
        if unresolved:
            for dep in deps:
                self.compiler.mark_dependency_missing(dep)
            self.compiler.pending_assignments[f"__if_line_{line_number}"] = (
                line, line_number, deps)
            return if_block['end_i'] - if_block['start_i']

        # Evaluate the condition. Missing variables simply yield False.
        try:
            condition_result = self._evaluate_if_condition(
                condition, line_number, warn_boolean=True)
        except NameError as exc:
            condition_result = False

        # Extract the block lines from the IF block. Prefer the cached full list of
        # preprocessed lines so nested IF statements still have access to their
        # ELSEIF/ELSE headers even when we're processing a sliced block. A
        # locally scanned block carries indices into *lines*, so slice there.
        if self._preprocessed_lines and not if_block.get('local'):
            block_lines = self._preprocessed_lines[
                if_block['start_i'] + 1: if_block['end_i'] + 1]
        else:
            block_lines = []
            for i in range(current_index + 1, len(lines)):
                line_content, line_num = lines[i]
                block_lines.append((line_content, line_num))
                if line_num == if_block['end_line']:
                    break

        # Predeclare variables that are assigned within the IF block so they exist
        # even if the condition evaluates to False. This keeps downstream references
        # (like cell bindings) consistent with programs that rely on defaulted values.
        self._predeclare_block_assignment_targets(block_lines)

        # Check if this is a simple IF statement (no ELSEIF/ELSE clauses)
        if len(if_block['clauses']) == 1 and if_block['clauses'][0]['kind'] == 'if':
            # Simple IF statement - process the body directly
            if condition_result:
                self._process_block(block_lines, self.compiler.current_scope())
        else:
            # Complex IF-ELSEIF-ELSE block
            self._process_if_elseif_else_block(
                block_lines, condition_result, line_number, if_block,
                if_condition=condition)

        # Return the number of lines consumed (from start to end of the IF block)
        return if_block['end_i'] - current_index

    def _predeclare_block_assignment_targets(self, block_lines):
        """
        Predeclare variables that are assigned within an IF block so they exist even
        when the block is skipped. This avoids NameErrors and keeps outputs stable
        for programs that expect these identifiers to be present with default values.
        """
        current_scope = self.compiler.current_scope()
        for line_content, _ in block_lines:
            stripped = line_content.strip()
            # ``Let`` targets are NOT predeclared: the LET handler binds them
            # when its block runs, and a placeholder here would carry a wrong
            # type into that bind (e.g. ``"" as text`` vs ``Let z = 1``).
            match = re.match(r'^:\s*([\w_]+)\s*=', stripped)
            if match:
                var_name = match.group(1)
                defining_scope = current_scope.get_defining_scope(var_name)
                if not defining_scope:
                    # The placeholder exists so a skipped block still reads
                    # (as ``#N/A``, an unvalued declaration) instead of
                    # raising NameError; it stays untyped so the real
                    # assignment (number, text, ...) is not rejected.
                    current_scope.define(
                        var_name, None, None, {}, is_uninitialized=True)
                # A variable already declared but unvalued keeps reading as
                # #N/A; do not force it to "".

    def _process_if_statement_new(self, line, line_number, lines, current_index):
        """New simple IF processing using pre-scanned block map"""

        # Parse the condition
        condition_match = re.match(
            r'^\s*if\s+(.+?)(?:\s+then\s*|\s*$)', line, re.I)
        if not condition_match:
            raise SyntaxError(f"Invalid IF syntax at line {line_number}")

        condition = condition_match.group(1).strip()
        has_block = line.lower().strip().endswith('then')

        # Evaluate the condition
        condition_result = self._evaluate_if_condition(
            condition, line_number, warn_boolean=True)

        if has_block:
            # Use pre-scanned block map to find the end of this IF block
            if line_number in self.block_map:
                end_line_number = self.block_map[line_number]

                # Extract the entire if-elseif-else block lines (including IF and END)
                block_lines = []
                for i, (block_line, block_line_number) in enumerate(lines):
                    if block_line_number >= line_number and block_line_number <= end_line_number:
                        block_lines.append((block_line, block_line_number))

                # Check if this is a simple IF statement (no ELSEIF/ELSE at the same indentation level)
                has_elseif_else = False
                # Get indentation of IF line
                if_line, if_line_number = block_lines[0]  # Extract the IF line
                if_line_indent = len(if_line) - len(if_line.lstrip())
                # Skip the IF line itself
                for block_line, block_line_number in block_lines[1:]:
                    line_clean = block_line.strip()
                    line_indent = len(block_line) - len(block_line.lstrip())
                    # Only check for ELSEIF/ELSE at the same indentation level as the IF statement
                    if line_indent == if_line_indent and (line_clean.startswith('elseif ') or line_clean == 'else'):
                        has_elseif_else = True
                        break

                if has_elseif_else:
                    # Process the if-elseif-else structure
                    self._process_if_elseif_else_block(
                        block_lines, condition_result, line_number,
                        if_block=None, if_condition=condition)
                else:
                    # Process as a simple IF statement
                    if condition_result:
                        # Process the IF block content (excluding the IF and END lines)
                        if_block_lines = []
                        # Skip IF and END lines
                        for block_line, block_line_number in block_lines[1:-1]:
                            if_block_lines.append(
                                (block_line, block_line_number))
                        if if_block_lines:
                            self._process_block(
                                if_block_lines, self.compiler.current_scope())
                # Return the number of lines consumed (from IF to END)
                # Only return full block length if this is a multi-line structure
                if len(block_lines) > 1:
                    # -1 because we already processed the IF line
                    return len(block_lines) - 1
                else:
                    return 1  # Single line IF
        return 1

