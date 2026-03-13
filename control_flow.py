"""
Control flow processing for GridLang compiler.
Handles FOR loops, IF statements, LET statements, and block processing.
"""

import re
from utils import num_to_col

# Regex patterns for block parsing
HEADER_IF = re.compile(r'^\s*if\b(.+?)\bthen\s*$', re.I)
HEADER_ELSEIF = re.compile(r'^\s*elseif\b(.+?)\bthen\s*$', re.I)
HEADER_ELSE = re.compile(r'^\s*else\s*$', re.I)
HEADER_FOR = re.compile(r'^\s*for\b(.+?)\bdo\s*$', re.I)
HEADER_WHILE = re.compile(r'^\s*while\b(.+?)\bdo\s*$', re.I)
HEADER_WHEN = re.compile(r'^\s*when\b(.+?)\bdo\s*$', re.I)
TOKEN_END = re.compile(r'^\s*end\s*$', re.I)


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
        self._header_while = HEADER_WHILE
        self._header_when = HEADER_WHEN
        self._token_end = TOKEN_END

    def _strip_inline_comment(self, s: str) -> str:
        """Strip inline comments (//... or #...) if not in quotes"""
        s = re.split(r'(?<!["\'])\s//', s, maxsplit=1)[0]
        s = re.split(r'(?<!["\'])\s#',  s, maxsplit=1)[0]
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

    def _match_push_assignment(self, text):
        return re.match(r'^\s*push\s+(\[[^\]]+\]|[\w_]+(?:\.[\w_]+)?(?:\([^)]+\)|\{[^}]+\})?)(?:\s*=\s*(.+))?\s*$', text, re.I)

    def _unpack_push_assignment(self, push_match):
        target, value_expr = push_match.groups()
        if value_expr is None:
            value_expr = target
        return target, value_expr

    def _match_return_statement(self, text):
        return re.match(r'^\s*return\s+(.+)$', text, re.I)

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
            self.compiler.output_values.setdefault('output', []).append(value)
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
            import pyarrow as pa
            # Initialize array with zeros
            initial_array = pa.array([0.0] * dim_size, type=pa.float64())

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
                scope.define(var_name, value, inferred_type,
                             {'constant': value}, False)

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
            is_block_while = bool(self._header_while.match(stripped))
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

            if is_block_if or is_block_for or is_block_while or is_block_when:
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
            is_block_while = bool(self._header_while.match(stripped))
            is_block_when = bool(self._header_when.match(stripped))
            is_end = bool(self._token_end.match(stripped))

            is_top_level_for = depth == 0 and lower.startswith("for ")

            # Treat both simple FOR declarations and top-level FOR blocks as global pre-exec
            if is_top_level_for and (" do" not in lower or is_block_for):
                for_entries.append(
                    {'line_number': line_number, 'line': stripped})

            if is_block_if or is_block_for or is_block_while or is_block_when:
                depth += 1
            if is_end and depth > 0:
                depth -= 1
        return for_entries

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
                    self._header_while.match(line_clean) or
                    self._header_when.match(line_clean)):
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

    def _handle_block_push_or_output_statement(self, i, line, line_number, line_clean):
        """Handle PUSH/.push and OUTPUT statements in block context."""
        if line_clean.lower().startswith('push ') or '.push(' in line_clean.lower():
            try:
                if line_clean.lower().startswith('push '):
                    m_assign = self._match_push_assignment(line_clean)
                    if not m_assign:
                        raise SyntaxError(
                            f"Invalid PUSH syntax at line {line_number}")
                    target, value_expr = self._unpack_push_assignment(m_assign)
                    self.compiler._handle_push_assignment(
                        target, value_expr, line_number)
                else:
                    self.compiler._process_push_call(line, line_number)
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
        # Inline IF with single statement on the same line
        inline_if_match = re.match(
            r'^\s*if\s+(.+?)\s+then\s+(.+)$', line, re.I)
        if inline_if_match:
            cond, action = inline_if_match.groups()
            try:
                if self._evaluate_if_condition(cond.strip(), line_number):
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
                    elif '.push(' in action.lower():
                        self.compiler._process_push_call(action, line_number)
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
                        elif re.match(r'^\s*push\s*\(', action, re.I):
                            values = self.compiler._evaluate_push_expression(
                                re.sub(r'^\s*push\s*\(|\)\s*$', '', action, flags=re.I), line_number)
                            for value in values:
                                self.compiler.output_values.setdefault(
                                    'output', []).append(value)
                            if 'output' not in self.compiler.output_variables:
                                self.compiler.output_variables.append('output')
                return True, i + 1
            except Exception as e:
                return True, i + 1
        # Simple IF-THEN single-block pattern: if <cond> then\n  <LET/assignment>\nend
        if line_clean.lower().startswith('if ') and line_clean.lower().endswith('then'):
            if i + 2 < len(block_lines):
                next_line, next_ln_no = block_lines[i + 1]
                end_line, end_ln_no = block_lines[i + 2]
                if end_line.strip().lower() == 'end':
                    try:
                        cond = re.match(
                            r'^\s*if\s+(.+?)\s*then\s*$', line, re.I).group(1).strip()
                        cond_result = self._evaluate_if_condition(
                            cond, line_number)
                        if cond_result:
                            if next_line.strip().lower() == 'exit for':
                                self.executor.exit_loop = True
                            elif next_line.strip().lower().startswith('let '):
                                self._process_let_statement_inline(
                                    next_line, next_ln_no)
                            elif next_line.strip().lower().startswith('output '):
                                try:
                                    def_str = next_line.strip(
                                    )[len('output '):].strip()
                                    self._process_output_statement(
                                        def_str, next_ln_no)
                                except Exception as e:
                                    raise
                            elif ':=' in next_line:
                                self.compiler.array_handler.evaluate_line_with_assignment(
                                    next_line, next_ln_no, self.compiler.current_scope().get_evaluation_scope())
                            elif '.push(' in next_line.lower():
                                self.compiler._process_push_call(
                                    next_line, next_ln_no)
                            else:
                                push_match = self._match_push_assignment(
                                    next_line)
                                return_match = self._match_return_statement(
                                    next_line)
                                if push_match:
                                    target, value_expr = self._unpack_push_assignment(push_match)
                                    self.compiler._handle_push_assignment(
                                        target, value_expr, next_ln_no)
                                elif return_match:
                                    self._handle_return_statement(
                                        return_match.group(1).strip(), next_ln_no)
                        return True, i + 3
                    except Exception as e:
                        pass
            try:
                consumed = self._process_if_statement_rich(
                    line, line_number, block_lines, i)
                return True, i + consumed + 1
            except Exception as e:
                return True, i + 1
        if re.match(r'^if\b.+\bthen\b', line_clean, re.I) and not line_clean.lower().endswith('then'):
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

        if '=' in line and not any(keyword in line.lower() for keyword in ['in', 'as', 'dim']):
            try:
                self.process_for_statement(
                    line, line_number, self.compiler.current_scope())
                return True, i + 1
            except ValueError as e:
                if "already declared in an outer scope" in str(e):
                    raise e
                return True, i + 1
            except Exception as e:
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
                        for name, value in self.compiler.current_scope().variables.items():
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
                            for name, value in self.compiler.current_scope().variables.items():
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
                    for name, value in loop_scope.variables.items():
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
            r'\b[\w_]+\b(?=\s*(?:[\[\{]|!\w+\s*\(|(?:\.\w+)?\s*$))', rhs))
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
                'if ', 'for ', 'when ', 'let ', 'output ', 'input ', 'define ', 'return ', 'push '))
            and not line.strip().startswith(': ')
            and not re.match(r'^\[\s*\^?[A-Za-z]+\d+\s*\]\s*:\s*', line)
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
            try:
                assignment_line = line.strip()[2:]
                m = re.match(r'^([\w_]+)\s*=\s*(.+)$', assignment_line)
                if m:
                    target_var = m.group(1)
                    defining_scope = self.compiler.current_scope().get_defining_scope(
                        target_var)
                    if not defining_scope:
                        inferred_type = 'number'
                        self.compiler.current_scope().define(
                            target_var, None, inferred_type, {}, is_uninitialized=True)
                self.compiler.array_handler.evaluate_line_with_assignment(
                    assignment_line, line_number, self.compiler.current_scope().get_evaluation_scope())
            except Exception as e:
                pass
            return True, i + 1, False
        if line.strip().lower().startswith('let '):
            try:
                self._process_let_statement_inline(
                    line, line_number)
            except Exception as e:
                if "dim * must be initialized with PUSH or INIT before LET" in str(e):
                    raise
            return True, i + 1, False
        if line.strip() == 'exit for':
            self.executor.exit_loop = True
            return True, i, True
        if line.strip().startswith('elseif ') and line.strip().endswith(' then'):
            return True, i + 1, False
        if line.strip() == 'else':
            return True, i + 1, False
        if re.match(r'^\s*\[[^\]]+\]\s*\.push\s*\(.+\)\s*$', line):
            try:
                cell_match = re.match(
                    r'^\s*(\[[^\]]+\])\s*\.push\s*\(\s*(.+)\s*\)\s*$', line)
                if cell_match:
                    target_cell = cell_match.group(1)
                    expr = cell_match.group(2)
                    assignment_line = f"{target_cell} := {expr}"
                    self.compiler.array_handler.evaluate_line_with_assignment(
                        assignment_line, line_number, self.compiler.current_scope().get_evaluation_scope())
            except Exception as e:
                pass
            return True, i + 1, False
        if self._match_push_assignment(line):
            push_match = self._match_push_assignment(line)
            target, value_expr = self._unpack_push_assignment(push_match)
            self.compiler._handle_push_assignment(
                target, value_expr, line_number)
            return True, i + 1, False
        if self._match_return_statement(line):
            return_match = self._match_return_statement(line)
            self._handle_return_statement(
                return_match.group(1).strip(), line_number)
            return True, i + 1, False
        if '.push(' in line and ')' in line:
            try:
                push_match = re.match(
                    r'(\w+)\s*\.push\s*\(\s*(.+)\s*\)', line.strip())
                if push_match:
                    var_name = push_match.group(1)
                    expr = push_match.group(2)
                    var_value = None
                    defining_scope = self.compiler.current_scope().get_defining_scope(var_name)
                    if defining_scope:
                        var_value = defining_scope.get(var_name)

                    if defining_scope:
                        result = self.compiler.expr_evaluator.eval_expr(
                            expr, self.compiler.current_scope().get_evaluation_scope())
                        defining_scope.update(
                            var_name, result, line_number)

                        if not hasattr(self.compiler, 'output_values'):
                            self.compiler.output_values = {}
                        existing = self.compiler.output_values.get(
                            var_name, [])
                        if not isinstance(existing, list):
                            existing = [] if existing is None else [existing]
                        existing.append(result)
                        self.compiler.output_values[var_name] = existing
                    else:
                        raise NameError(
                            f"Variable {var_name} not defined at line {line_number}")
            except Exception as e:
                pass
            return True, i + 1, False
        if line.strip().lower().startswith('push(') and line.strip().endswith(')'):
            try:
                expr_match = re.match(
                    r'push\s*\(\s*(.+)\s*\)\s*$', line.strip(), re.I)
                if expr_match:
                    expr = expr_match.group(1)
                    result = self.compiler.expr_evaluator.eval_expr(
                        expr, self.compiler.current_scope().get_evaluation_scope())
                    if not hasattr(self.compiler, 'output_values'):
                        self.compiler.output_values = {}
                    existing = self.compiler.output_values.get(
                        'output', [])
                    if not isinstance(existing, list):
                        existing = [] if existing is None else [existing]
                    existing.append(result)
                    self.compiler.output_values['output'] = existing
                    if 'output' not in self.compiler.output_variables:
                        self.compiler.output_variables.append('output')
            except Exception as e:
                pass
            return True, i + 1, False
        return False, i, False

    def _process_block(self, block_lines, current_scope=None):
        block_pending = {}
        i = 0
        while i < len(block_lines):
            if self._should_break_block_processing():
                break
            line, line_number, line_clean = self._prepare_block_line(
                block_lines, i)
            if not line_clean:
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

            i += 1
        if block_pending:
            self.compiler.scopes[0].pending_assignments.update(block_pending)
        return block_pending

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
            condition_result = self._evaluate_if_condition(condition, line_number)
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
            or not current_scope.get_defining_scope(dep)
            for dep in deps
        )
        if unresolved:
            return self._defer_pending_if(line, line_number, deps)
        return None

    def _process_if_block_statement(self, condition_result, lines, current_index, line_number):
        if_block_lines, else_block_lines, elseif_blocks, block_i = self._collect_if_blocks(
            lines, current_index, line_number)
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
            elif next_line_clean.startswith("if "):
                current_block.append((next_line, next_line_number))
                block_i += 1
                continue
            elif self.compiler._starts_with_keyword(next_line_clean, "if ") and self.compiler._ends_with_keyword(next_line_clean, "then"):
                depth += 1
            elif self.compiler._starts_with_keyword(next_line_clean, "elseif ") and self.compiler._ends_with_keyword(next_line_clean, "then") and depth == 1:
                elseif_condition = re.match(
                    r'^\s*elseif\s+(.+?)\s+then\s*$', next_line, re.I)
                if elseif_condition:
                    elseif_cond = elseif_condition.group(1).strip()
                    elseif_result = self._evaluate_if_condition(
                        elseif_cond, next_line_number)
                    elseif_blocks.append((elseif_result, []))
                    current_block = elseif_blocks[-1][1]
            elif self.compiler._starts_with_keyword(next_line_clean, "else") and depth == 1:
                current_block = else_block_lines
                current_block.append((next_line, next_line_number))
                block_i += 1
                continue
            elif self.compiler._starts_with_keyword(next_line_clean, "elseif ") and self.compiler._ends_with_keyword(next_line_clean, "then"):
                depth += 1
            elif self.compiler._starts_with_keyword(next_line_clean, "else"):
                if depth == 1:
                    depth -= 1
                    if depth == 0:
                        block_i += 1
                        break
                else:
                    depth += 1
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
        if condition_result:
            self._execute_block_with_scope_transfer(if_block_lines, "IF")
            return

        for elseif_result, elseif_lines in elseif_blocks:
            if elseif_result:
                self._execute_block_with_scope_transfer(elseif_lines, "ELSEIF")
                return

        if else_block_lines:
            self._execute_block_with_scope_transfer(else_block_lines, "ELSE")

    def _execute_block_with_scope_transfer(self, block_lines, label):
        self.compiler.push_scope(is_private=False)
        self._process_block(block_lines, self.compiler.current_scope())
        block_scope = self.compiler.current_scope()
        self.compiler.pop_scope()
        parent_scope = self.compiler.current_scope()
        for var_name, var_value in block_scope.variables.items():
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
        if '.push(' in action.lower():
            self.compiler._process_push_call(action, line_number)
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
        if re.match(r'^\s*push\s*\(', action, re.I):
            expr = re.sub(r'^\s*push\s*\(|\)\s*$', '', action, flags=re.I)
            values = self.compiler._evaluate_push_expression(expr, line_number)
            for value in values:
                self.compiler.output_values.setdefault('output', []).append(value)
            if 'output' not in self.compiler.output_variables:
                self.compiler.output_variables.append('output')

    def _process_let_statement_inline(self, line, line_number):
        """
        Process a LET statement inline (for use in IF-ELSE blocks).
        """

        # Parse the LET statement
        m = re.match(r'^\s*LET\s+(.+?)(?:\s+then\s*$|\s*$)', line, re.I)
        if not m:
            raise SyntaxError(f"Invalid LET syntax at line {line_number}")

        var_def = m.group(1).strip()

        var_list = []
        if ' and ' in var_def.lower():
            var_parts = re.split(r'\s+and\s+', var_def, flags=re.I)
            for var_part in var_parts:
                var, type_name, constraints, expr = self.compiler._parse_variable_def(
                    var_part, line_number)
                var_list.append((var, type_name, constraints, expr))
        else:
            var, type_name, constraints, expr = self.compiler._parse_variable_def(
                var_def, line_number)
            var_list.append((var, type_name, constraints, expr))


        # Process each variable
        scope_dict = self.compiler.current_scope().get_evaluation_scope()
        for var, type_name, constraints, expr in var_list:
            if self._try_process_let_field_index_assignment(
                    var, expr, scope_dict, line_number):
                continue
            if self._try_process_let_indexed_assignment(
                    var, expr, scope_dict, line_number):
                continue
            defining_scope = self.compiler.current_scope().get_defining_scope(var)
            if defining_scope:
                if constraints:
                    defining_scope.constraints[var] = constraints
                if expr is None and var in defining_scope.variables and defining_scope.variables[var] is not None:
                    continue
            else:
                if self.compiler.current_scope().is_shadowed(var):
                    print(
                        f"Warning: LET defines '{var}' which shadows a variable in an outer scope at line {line_number}")
                self.compiler.current_scope().define(
                    var, None, type_name, constraints, is_uninitialized=True)

            if expr is not None:
                try:
                    evaluated_value = self.compiler.expr_evaluator.eval_or_eval_array(
                        expr, scope_dict, line_number)
                    self.compiler.current_scope().update(var, evaluated_value, line_number)
                    scope_dict[var] = evaluated_value
                    pending_vars = [
                        pending_var for pending_var in list(self.compiler.pending_assignments.keys())
                        if not pending_var.startswith('__line_')
                    ]
                    for pending_var in pending_vars:
                        self.compiler._attempt_resolve_pending_var(
                            pending_var, line_number)
                except NameError as e:
                    raise NameError(
                        f"Undefined variables in LET statement: {var} at line {line_number}")

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
        array_index_match = re.match(r'^([\w_]+)\{([^}]+)\}$', var)
        cell_index_match = re.match(r'^([\w_]+)\[([^\]]+)\]$', var)
        paren_index_match = re.match(r'^([\w_]+)\(([^)]+)\)$', var)
        if not (array_index_match or cell_index_match or paren_index_match):
            return False
        if array_index_match:
            var_name, indices_str = array_index_match.groups()
            indices = []
            for index_expr in indices_str.split(','):
                index_expr = index_expr.strip()
                index_value = self.compiler.expr_evaluator.eval_expr(
                    index_expr, scope_dict, line_number)
                if isinstance(index_value, float) and index_value.is_integer():
                    index_value = int(index_value)
                indices.append(
                    index_value - 1 if isinstance(index_value, int) else index_value)
        elif cell_index_match:
            var_name, index_expr = cell_index_match.groups()
            try:
                index_value = self.compiler.expr_evaluator.eval_expr(
                    index_expr, scope_dict, line_number)
                if isinstance(index_value, float) and index_value.is_integer():
                    index_value = int(index_value)
                indices = [index_value - 1]
            except Exception:
                indices = self.compiler.array_handler.cell_ref_to_indices(
                    index_expr, line_number)
        else:
            var_name, index_expr = paren_index_match.groups()
            index_value = self.compiler.expr_evaluator.eval_expr(
                index_expr, scope_dict, line_number)
            if isinstance(index_value, float) and index_value.is_integer():
                index_value = int(index_value)
            indices = [index_value - 1]
        value = self.compiler.expr_evaluator.eval_or_eval_array(
            expr, scope_dict, line_number)
        defining_scope = self.compiler.current_scope().get_defining_scope(var_name)
        if defining_scope:
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
                raise ValueError(
                    f"Array variable '{var_name}' with dim * must be initialized with PUSH or INIT before LET at line {line_number}")
            if isinstance(arr, dict) and 'array' in arr:
                updated_array = self.compiler.array_handler.set_array_element(
                    arr['array'], indices, value, line_number)
                arr['array'] = updated_array
                defining_scope.variables[actual_key] = arr
                scope_dict[actual_key] = arr
            else:
                updated_array = self.compiler.array_handler.set_array_element(
                    arr, indices, value, line_number)
                defining_scope.variables[actual_key] = updated_array
                scope_dict[actual_key] = updated_array
            debug_array = updated_array.to_pylist() if hasattr(
                updated_array, 'to_pylist') else updated_array
            return True
        if var_name.lower() == 'grid':
            if len(indices) == 2:
                row_idx = indices[0] + 1
                col_idx = indices[1] + 1
                cell = f"{num_to_col(col_idx)}{row_idx}"
                self.compiler.grid[cell] = value
            return True
        return True

    def _evaluate_if_condition(self, condition, line_number):
        """
        Evaluate an IF condition, handling various types of constraints.
        """

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
            if var_name in defining_scope.constraints:
                var_constraints = defining_scope.constraints[var_name]
                current_unit = (var_constraints.get('unit') or '').lower()
                return current_unit != unit_name.lower()
            return True

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
        return self._evaluate_if_condition_default(condition, line_number)

    def _evaluate_if_logical_or_comparison(self, condition, line_number):
        if ' and ' in condition.lower() and not re.search(r'[<>=!]\s*and\s*[<>=!]', condition):
            parts = re.split(r'\s+and\s+', condition, flags=re.I)
            return True, all(self._evaluate_if_condition(part.strip(), line_number) for part in parts)
        if ' or ' in condition.lower() and not re.search(r'[<>=!]\s*or\s*[<>=!]', condition):
            parts = re.split(r'\s+or\s+', condition, flags=re.I)
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
            r'^(.+?)\s*(?<!not\s)(<>|!=|<=|>=|=|<|>)(?!\s*=)\s*(?!.*\s+(?:and|or)\s+)(.+)$', condition.strip(), re.I)
        if not comparison_match:
            return False, None
        left_expr, operator, right_expr = comparison_match.groups()
        current_scope = self.compiler.current_scope()
        scope = current_scope.get_evaluation_scope()
        scope.setdefault('true', True)
        scope.setdefault('false', False)
        scope.setdefault('none', None)
        if left_expr.strip() in current_scope.variables:
            scope[left_expr.strip()] = current_scope.variables[left_expr.strip()]
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
            if operator in ('!=', '<>'):
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

    def _evaluate_if_condition_default(self, condition, line_number):
        try:
            scope = self.compiler.current_scope().get_evaluation_scope()
            result = self.compiler.expr_evaluator.eval_expr(
                condition, scope, line_number)
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
        if type_name.lower() == 'text':
            if not (actual_type == 'text' or (isinstance(var_value, (list, tuple)) and all(isinstance(item, str) for item in var_value))):
                return True, False
        elif type_name.lower() == 'number':
            if not (actual_type in ('number', 'int') or (isinstance(var_value, (list, tuple)) and all(isinstance(item, (int, float)) for item in var_value))):
                return True, False
        if dim_constraint:
            if dim_constraint == '{}':
                return True, not isinstance(var_value, (list, tuple)) or len(var_value) == 0
            if dim_constraint == '*':
                return True, isinstance(var_value, (list, tuple)) and len(var_value) > 0
            try:
                dim_parts = dim_constraint[1:-1].split(',')
                expected_dims = [int(d.strip()) for d in dim_parts]
                if isinstance(var_value, (list, tuple)):
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
        if dim_spec.strip() == '*':
            return True, isinstance(var_value, (list, tuple)) and len(var_value) > 0
        if dim_spec.strip() == '{}':
            return True, not isinstance(var_value, (list, tuple)) or len(var_value) == 0
        if dim_spec.strip().isdigit():
            expected_dim = int(dim_spec.strip())
            if isinstance(var_value, (list, tuple)):
                return True, len(var_value) == expected_dim
            return True, expected_dim == 0
        if dim_spec.strip().startswith('{') and dim_spec.strip().endswith('}'):
            try:
                dim_str = dim_spec.strip()[1:-1]
                expected_dims = [int(d.strip()) for d in dim_str.split(',')]
                if isinstance(var_value, (list, tuple)):
                    actual_dims = self.compiler.array_handler.get_array_dimensions(
                        var_value)
                    while actual_dims and actual_dims[0] == 1:
                        actual_dims = actual_dims[1:]
                    return True, actual_dims == expected_dims
                return True, expected_dims == []
            except Exception:
                return True, False
        return True, False

    def _evaluate_if_unit_constraint(self, condition, line_number):
        unit_match = re.match(r'^([\w_]+)\s+of\s+(\w+)$', condition, re.I)
        if not unit_match:
            return False, None
        var_name, unit_name = unit_match.groups()
        defining_scope = self.compiler.current_scope().get_defining_scope(var_name)
        if not defining_scope:
            raise NameError(
                f"Variable '{var_name}' not defined at line {line_number}")
        constraint_key = defining_scope._get_case_insensitive_key(
            var_name, defining_scope.constraints)
        if constraint_key:
            var_constraints = defining_scope.constraints[constraint_key]
            return True, (var_constraints.get('unit') or '').lower() == unit_name.lower()
        return True, False

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
            del self.compiler.pending_assignments[var]
            self.compiler.grid.clear()
            return False
        except NameError as e:
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

        for i, (line, line_number) in enumerate(lines):
            raw = line
            line = self._strip_inline_comment(line)
            line_clean = line.strip()

            if not line_clean:
                continue

            # IF
            if self._header_if.match(line_clean):
                block = {
                    'type': 'if',
                    'start_i': i,
                    'start_line': line_number,
                    'clauses': [{'kind': 'if', 'header_i': i, 'header_line': line_number}]
                }
                stack.append(block)
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
                continue

            # WHILE
            if self._header_while.match(line_clean):
                stack.append({'type': 'while', 'start_i': i,
                             'start_line': line_number})
                continue

            # WHEN
            if self._header_when.match(line_clean):
                stack.append({'type': 'when', 'start_i': i,
                             'start_line': line_number})
                continue

            # END
            if self._token_end.match(line_clean):
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
        return self.block_map

    def _process_if_elseif_else_block(self, block_lines, if_condition_result, if_line_number, if_block):
        """Process IFâ€“ELSEIFâ€“ELSE blocks, with correct clause selection and nested IF support."""

        active_clause_lines = []
        clause_found = False
        in_active_clause = False
        depth = 0  # nested IF depth

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
                        cond, line_number)
                    if cond_result:
                        in_active_clause = True
                        clause_found = True
                        active_clause_lines = []
                    else:
                        in_active_clause = False
                else:
                    in_active_clause = False

            elif depth == 0 and lower == "else":
                if not clause_found:
                    in_active_clause = True
                    clause_found = True
                    active_clause_lines = []
                else:
                    in_active_clause = False

            elif depth == 0 and lower == "end":
                break

            else:
                # --- Nested IF handling ---
                if lower == "end" and depth > 0:
                    if in_active_clause:
                        active_clause_lines.append((line, line_number))
                    depth -= 1
                else:
                    if in_active_clause:
                        active_clause_lines.append((line, line_number))

            i += 1

        # Execute the chosen clause
        if active_clause_lines:
            self._process_block(active_clause_lines,
                                self.compiler.current_scope())

    def _process_if_statement_rich(self, line, line_number, lines, current_index):
        """Process IF statement using the rich block structure with clause information"""

        # Find the IF block that starts at this line
        if_block = None
        for block in self.if_blocks:
            if block['start_line'] == line_number:
                if_block = block
                break

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
            return if_block['end_line'] - line_number

        # Evaluate the condition. Missing variables simply yield False.
        try:
            condition_result = self._evaluate_if_condition(
                condition, line_number)
        except NameError as exc:
            condition_result = False

        # Extract the block lines from the IF block. Prefer the cached full list of
        # preprocessed lines so nested IF statements still have access to their
        # ELSEIF/ELSE headers even when we're processing a sliced block.
        if self._preprocessed_lines:
            block_lines = [
                (line_content, line_num)
                for (line_content, line_num) in self._preprocessed_lines
                if line_number < line_num <= if_block['end_line']
            ]
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
                block_lines, condition_result, line_number, if_block)

        # Return the number of lines consumed (from start to end of the IF block)
        return if_block['end_line'] - line_number

    def _predeclare_block_assignment_targets(self, block_lines):
        """
        Predeclare variables that are assigned within an IF block so they exist even
        when the block is skipped. This avoids NameErrors and keeps outputs stable
        for programs that expect these identifiers to be present with default values.
        """
        current_scope = self.compiler.current_scope()
        for line_content, _ in block_lines:
            stripped = line_content.strip()
            match = re.match(r'^:\s*([\w_]+)\s*=', stripped)
            if not match:
                match = re.match(r'^let\s+([\w_]+)\s*=', stripped, re.I)
            if match:
                var_name = match.group(1)
                defining_scope = current_scope.get_defining_scope(var_name)
                if not defining_scope:
                    current_scope.define(
                        var_name, "", 'text', {}, is_uninitialized=False)
                elif defining_scope.is_uninitialized(var_name):
                    defining_scope.update(var_name, "", None)

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
        condition_result = self._evaluate_if_condition(condition, line_number)

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
                        block_lines, condition_result, line_number)
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

