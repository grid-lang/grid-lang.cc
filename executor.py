
import re
import math
import copy
from collections import deque
import pyarrow as pa
from expression import ExpressionEvaluator
from array_handler import ArrayHandler
from control_flow import GridLangControlFlow
from parser import GridLangParser
from utils import col_to_num, num_to_col, split_cell, offset_cell, validate_cell_ref, public_type_fields, object_public_keys


IDENTIFIER_TOKEN_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_.]*')
STRING_LITERAL_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
DEPENDENCY_IGNORED_TOKENS = {
    'sum', 'rows', 'sqrt', 'min', 'max', 'abs', 'int', 'float', 'str', 'len',
    'textsplit', 'print', 'push', 'true', 'false', 'none', 'nan', 'inf', 'and', 'or', 'not',
    'if', 'then', 'else', 'elseif', 'end', 'do', 'for', 'while', 'step', 'return',
    'index', 'as', 'dim', 'with', 'grid', 'output', 'input', 'number', 'text',
    'array', 'mod', 'div', 'to', 'by', 'e', 'new', 'in', 'counta', 'rows'
}


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


class GridLangExecutor:
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
        self.global_for_entries = []
        self.needed_line_numbers = set()

    def get_global_scope(self):
        """Get the global scope from the compiler"""
        if hasattr(self, 'compiler') and self.compiler:
            return self.compiler.scopes[0]  # First scope is always global
        else:
            # Fallback if no compiler reference
            return self.current_scope()

    def collect_input_output_variables(self):
        """Collect all input and output variables from the current scope"""
        global_scope = self.get_global_scope()
        self.input_variables = list(global_scope.input_variables)
        self.output_variables = list(global_scope.output_variables)
        # Add 'output' as a default output variable for push() calls
        if 'output' not in self.output_variables:
            self.output_variables.append('output')

    def _reset_dependency_graph(self):
        """Reset dependency graph storage for a new run."""
        self.dependency_graph = {'nodes': [], 'by_variable': {}, 'by_line': {}}

    def _extract_dependencies_from_expression(self, expr):
        """Return identifiers referenced in an expression string."""
        if not expr:
            return set()
        cleaned = STRING_LITERAL_PATTERN.sub(' ', expr)
        tokens = IDENTIFIER_TOKEN_PATTERN.findall(cleaned)
        dependencies = set()
        for token in tokens:
            if not token:
                continue
            base = token.split('.')[0]
            lower = base.lower()
            if lower in DEPENDENCY_IGNORED_TOKENS:
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
            base_match = re.match(r'([A-Za-z_][A-Za-z0-9_]*)', name)
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
        return re.match(r'^\s*push\s+(\[[^\]]+\]|[\w_]+(?:\.[\w_]+)?(?:\([^)]+\)|\{[^}]+\})?)\s*=\s*(.+)$', text, re.I)

    def _match_return_statement(self, text):
        return re.match(r'^\s*return\s+(.+)$', text, re.I)

    def _has_push_action(self, text):
        if not text:
            return False
        lowered = text.lower()
        if '.push(' in lowered:
            return True
        if re.search(r'\bpush\s*\(', text, re.I):
            return True
        if self._match_push_assignment(text):
            return True
        if re.search(r'\breturn\b', text, re.I):
            return True
        return False

    def _has_star_dim(self, constraints):
        dim_spec = (constraints or {}).get('dim')
        if isinstance(dim_spec, dict) and 'dims' in dim_spec:
            dim_spec = dim_spec['dims']
        if isinstance(dim_spec, list):
            return any(size_spec is None for _, size_spec in dim_spec)
        if isinstance(dim_spec, str):
            return '*' in dim_spec
        return False

    def _parse_indexed_target(self, target, scope_dict, line_number):
        array_index_match = re.match(r'^([\w_]+)\{([^}]+)\}$', target)
        cell_index_match = re.match(r'^([\w_]+)\[([^\]]+)\]$', target)
        paren_index_match = re.match(r'^([\w_]+)\(([^)]+)\)$', target)
        if array_index_match:
            var_name, indices_str = array_index_match.groups()
            indices = []
            for index_expr in indices_str.split(','):
                index_expr = index_expr.strip()
                index_value = self.expr_evaluator.eval_expr(
                    index_expr, scope_dict, line_number)
                if isinstance(index_value, float) and index_value.is_integer():
                    index_value = int(index_value)
                indices.append(index_value - 1 if isinstance(
                    index_value, int) else index_value)
            return var_name, indices
        if cell_index_match:
            var_name, index_expr = cell_index_match.groups()
            try:
                index_value = self.expr_evaluator.eval_expr(
                    index_expr, scope_dict, line_number)
                if isinstance(index_value, float) and index_value.is_integer():
                    index_value = int(index_value)
                indices = [index_value - 1]
            except Exception:
                indices = self.array_handler.cell_ref_to_indices(
                    index_expr, line_number)
            return var_name, indices
        if paren_index_match:
            var_name, index_expr = paren_index_match.groups()
            index_value = self.expr_evaluator.eval_expr(
                index_expr, scope_dict, line_number)
            if isinstance(index_value, float) and index_value.is_integer():
                index_value = int(index_value)
            indices = [index_value - 1]
            return var_name, indices
        return None, None

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
            parts = re.split(r'\s+and\s+', working, flags=re.I)
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
            return False
        arg_parts = self._split_call_arguments(args_str)
        sp_def = subprocess_defs[name.lower()]
        inputs = sp_def.get('inputs', []) or []
        if len(arg_parts) < len(inputs):
            raise ValueError(
                f"Subprocess '{name}' expects at least {len(inputs)} arguments at line {line_number}")
        input_vals = []
        for idx, raw in enumerate(arg_parts):
            if idx < len(inputs):
                val = self.expr_evaluator.eval_or_eval_array(
                    raw, self.current_scope().get_evaluation_scope(), line_number)
                input_vals.append(val)
            else:
                break
        output_bindings = [a for a in arg_parts[len(inputs):]]
        call_fn = getattr(self, 'call_subprocess', None) or getattr(
            getattr(self, 'compiler', None), 'call_subprocess', None)
        if not call_fn:
            raise AttributeError("call_subprocess not available on executor")
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
                r'\bindex\s+([A-Za-z_][A-Za-z0-9_]*)\b', expr_part, re.I)
            if index_match:
                index_name = index_match.group(1)
                expr_part = re.sub(
                    r'\bindex\s+[A-Za-z_][A-Za-z0-9_]*\b', '', expr_part, flags=re.I)
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

    def _analyze_while_dependencies(self, normalized, original_line, line_number):
        """Create a dependency node for a WHILE loop header."""
        header_expr = re.split(
            r'\bdo\b', normalized[5:].strip(), maxsplit=1, flags=re.I)[0].strip()
        dependencies = self._extract_dependencies_from_expression(header_expr)
        return {
            'kind': 'loop',
            'line': line_number,
            'source': original_line,
            'defines': [],
            'depends_on': list(dependencies),
            'data': {
                'header': normalized,
                'loop_type': 'while'
            }
        }

    def _register_dependency_node(self, node):
        """Record a node inside the dependency graph."""
        defines = sorted({name for name in node.get('defines', []) if name})
        depends_on = sorted({dep for dep in node.get('depends_on', []) if dep})
        node['defines'] = defines
        node['depends_on'] = depends_on
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
                (lowered_clean.startswith('while ')
                 and lowered_clean.endswith('do'))
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
            if lowered.startswith('push ') or lowered.startswith('push(') or lowered.startswith('return '):
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
                    self._register_dependency_node(node)
                depth = max(0, depth + depth_change)
                continue
            if lowered.startswith('while ') and ' do' in lowered:
                node = self._analyze_while_dependencies(
                    normalized, raw_line, line_number)
                if node:
                    self._register_dependency_node(node)
                depth = max(0, depth + depth_change)
                continue
            assignment_node = self._parse_assignment_like_line(
                normalized, raw_line, line_number, is_declaration=is_declaration)
            if assignment_node:
                if isinstance(assignment_node, list):
                    for node in assignment_node:
                        if node:
                            self._register_dependency_node(node)
                else:
                    self._register_dependency_node(assignment_node)
            depth = max(0, depth + depth_change)
        for info in self.dependency_graph['by_variable'].values():
            info['definitions'] = sorted(info['definitions'])
            info['dependents'] = sorted(info['dependents'])
        for node in self.dependency_graph['nodes']:
            pass

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
            return False
        # Disallow equality binding to a subprocess; require INIT instead
        call_match = re.match(
            r'^([A-Za-z_][A-Za-z0-9_.]*)\\s*\\(.*\\)$', str(expr))
        if call_match:
            call_name = call_match.group(1).lower()
            if hasattr(self, 'subprocesses') and call_name in getattr(self, 'subprocesses', {}):
                raise RuntimeError(
                    f"Equality assignment to subprocess '{call_name}' is not allowed; use INIT instead at line {line_number}")
        scope = self.current_scope()
        defining_scope = scope.get_defining_scope(var)
        if not defining_scope:
            defining_scope = scope
            defining_scope.define(var, None, type_name or 'unknown',
                                  constraints, is_uninitialized=True)
        full_scope = defining_scope.get_full_scope(
        ) if hasattr(defining_scope, 'get_full_scope') else scope.get_full_scope()
        value = self.expr_evaluator.eval_or_eval_array(
            expr, full_scope, line_number)
        value = self.array_handler.check_dimension_constraints(
            var, value, line_number)
        # Infer type for constructor patterns or value shape
        inferred_type = type_name
        if not inferred_type:
            ctor_match = re.match(
                r'new\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', str(expr), re.I)
            if ctor_match:
                inferred_type = ctor_match.group(1)
        if not inferred_type:
            inferred_type = self.array_handler.infer_type(value, line_number)
        try:
            key = defining_scope._get_case_insensitive_key(
                var, defining_scope.types)
            if key:
                defining_scope.types[key] = inferred_type
            else:
                defining_scope.types[var] = inferred_type
        except Exception:
            pass
        defining_scope.update(var, value, line_number)
        return True

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
                            # Define the loop variable in current scope
                            self.control_flow.process_for_statement(
                                line, line_number, self.current_scope())
                            # Execute the block body once
                            self.control_flow._process_block(body_lines)
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
        except Exception as e:
            missing = self.extract_missing_dependencies(e)
            if missing:
                for dep in missing:
                    self.mark_dependency_missing(dep)
                updated_deps = set(deps) | set(missing)
                self.pending_assignments[var_name] = (
                    expr, assign_line, updated_deps, constraints)
                return False
            return False

    def run(self, code, args=None, suppress_output=False, return_output=False):
        if args:
            pass
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
        self._process_declarations_and_labels(lines, label_lines, dim_lines)
        # Store the root scope after declarations
        self.root_scope = self.current_scope()

        # Collect input/output variables and set input values
        self.collect_input_output_variables()
        prompt_missing = getattr(self, 'prompt_missing_inputs', False)
        self.set_input_values(args, prompt_missing=prompt_missing)
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
            return self.grid
        self._execute_global_for_loops(lines)
        self.needed_line_numbers = self._determine_needed_lines()
        guard_conditions = {entry['line_number']: entry['condition']
                            for entry in guard_entries}
        i = 0
        while i < len(lines):
            # Realize any INIT values whose dependencies are now satisfied before executing the next line
            self._materialize_inits()
            if not self.global_guard_allows_execution:
                break
            line, line_number = lines[i]
            stripped = line.strip()
            stripped_lower = stripped.lower()
            if line_number in self.executed_global_for_lines:
                i += 1
                continue
            if not stripped:
                i += 1
                continue
            if stripped_lower == 'end':
                i += 1
                continue
            # Skip global declarations as they're already processed in _process_declarations_and_labels
            if stripped.startswith(':'):
                i += 1
                continue

            if line_number in self.global_guard_line_numbers:
                i += 1
                continue

            node = self.dependency_graph['by_line'].get(line_number)
            if node and self.needed_line_numbers and line_number not in self.needed_line_numbers:
                if stripped_lower.startswith(('let ', 'for ', 'if ', 'elseif', 'else', 'while ')):
                    pass
                else:
                    i += 1
                    continue


            if stripped_lower.startswith("for ") and 'grid dim' in stripped_lower:
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
                        r'^\s*new\s+([A-Za-z_][A-Za-z0-9_]*)\s*$', expr, re.I)
                    if ctor_match:
                        type_name = ctor_match.group(1).lower()

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
                        tensor_struct = {f: with_constraints.get(
                            f) for f in public_fields}
                        tensor_struct['grid'] = grid_data['array']
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
                                    f"{var}.{field}", with_constraints[field], 'text')
                i += 1
            elif stripped_lower.startswith("return "):
                m_return = self._match_return_statement(stripped)
                if not m_return:
                    raise SyntaxError(
                        f"Invalid RETURN syntax at line {line_number}")
                value_expr = m_return.group(1).strip()
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
                            'output', []).append(value)
                    if 'output' not in self.output_variables:
                        self.output_variables.append('output')
                except Exception as e:
                    raise ValueError(
                        f"Failed to evaluate return expression at line {line_number}: {e}")
                i += 1
            elif stripped_lower.startswith("push "):
                m_assign = self._match_push_assignment(stripped)
                if m_assign:
                    target, value_expr = m_assign.groups()
                    self._handle_push_assignment(
                        target, value_expr, line_number)
                else:
                    raise SyntaxError(
                        f"Invalid PUSH syntax at line {line_number}")
                i += 1
            elif stripped_lower.startswith("push("):
                m_func = re.match(
                    r'^\s*push\s*\(\s*(.+)\s*\)\s*$', line, re.I)
                if m_func:
                    value_expr = m_func.group(1).strip()
                    try:
                        values = self._evaluate_push_expression(
                            value_expr, line_number)
                        for value in values:
                            self.output_values.setdefault(
                                'output', []).append(value)
                        if 'output' not in self.output_variables:
                            self.output_variables.append('output')
                    except Exception as e:
                        pass
                else:
                    raise SyntaxError(
                        f"Invalid PUSH syntax at line {line_number}")
                i += 1
            # Check for one-liner FOR loops with .push() first (before .push() detection)
            # Check for FOR loops with .push() (both one-liner and multi-line)
            elif line.lower().strip().startswith("for ") and ' do ' in line.lower() and self._has_push_action(line):

                # Check if this is a multi-line FOR loop by looking for incomplete string literals
                if line.count('"') % 2 == 1:  # Odd number of quotes means incomplete string

                    # Collect all lines until we find the complete .push() call
                    complete_lines = [line]
                    current_i = i
                    current_line = line

                    # Look for the closing parenthesis and quote
                    while not (current_line.strip().endswith(')') and current_line.count('"') % 2 == 0):
                        current_i += 1
                        if current_i >= len(lines):
                            raise SyntaxError(
                                f"Unterminated string literal in FOR loop starting at line {line_number}")

                        # Get the line content from the tuple
                        current_line = lines[current_i][0]
                        complete_lines.append(current_line)

                    # Combine all lines into a single executable part
                    executable_part = ' '.join(complete_lines)

                    # Extract the FOR loop part from the first line
                    first_line = complete_lines[0]
                    if ' do ' in first_line.lower():
                        parts = first_line.split(' do ', 1)
                        if len(parts) == 2:
                            for_part = parts[0].strip()
                            # Remove the partial executable part from the first line
                            executable_part = executable_part.replace(
                                first_line, '').strip()
                            if executable_part.startswith('do '):
                                executable_part = executable_part[3:].strip()


                            # Parse the FOR loop part
                            for_match = re.match(
                                r'^\s*for\s+(.+?)$', for_part, re.I)
                            if for_match:
                                var_defs = for_match.group(1).strip()

                                # Parse the range expression
                                range_match = re.match(
                                    r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', var_defs, re.I)
                                if range_match:
                                    var_name, range_expr, step_str, index_var = range_match.groups()
                                    step = int(step_str) if step_str else 1

                                    # Parse the range expression
                                    if ' to ' in range_expr:
                                        range_parts = range_expr.split(' to ')
                                        start = int(range_parts[0].strip())
                                        end = int(range_parts[1].strip())

                                        # Handle negative steps correctly
                                        if step < 0:
                                            values = list(
                                                range(start, end - 1, step))
                                        else:
                                            values = list(
                                                range(start, end + 1, step))

                                        # Process the executable part for each value
                                        if self._has_push_action(executable_part):
                                            for idx, value in enumerate(values):
                                                self.push_scope(
                                                    is_private=True, is_loop_scope=True)
                                                self.current_scope().define(var_name, value, 'number')
                                                if index_var:
                                                    self.current_scope().define(index_var, idx + 1, 'number')

                                                # Process the push/return call
                                                return_match = self._match_return_statement(
                                                    executable_part)
                                                push_match = self._match_push_assignment(
                                                    executable_part)
                                                if return_match:
                                                    value_expr = return_match.group(
                                                        1).strip()
                                                    values = self._evaluate_push_expression(
                                                        value_expr, line_number)
                                                    for value in values:
                                                        self.output_values.setdefault(
                                                            'output', []).append(value)
                                                    if 'output' not in self.output_variables:
                                                        self.output_variables.append(
                                                            'output')
                                                elif push_match:
                                                    target, value_expr = push_match.groups()
                                                    self._handle_push_assignment(
                                                        target, value_expr, line_number)
                                                else:
                                                    self._process_push_call(
                                                        executable_part, line_number)
                                                self.pop_scope()

                                        # Skip all the lines we processed in the main loop
                                        i = current_i + 1
                                        continue
                                    else:
                                        i = current_i + 1
                                        continue
                                else:
                                    i = current_i + 1
                                    continue
                            else:
                                i = current_i + 1
                                continue
                        else:
                            i = current_i + 1
                            continue
                    else:
                        i = current_i + 1
                        continue

                else:
                    # This is a one-liner FOR loop

                    # Split the line at " do " to separate loop definition and executable code
                    parts = line.split(' do ', 1)
                    if len(parts) == 2:
                        for_part = parts[0].strip()
                        executable_part = parts[1].strip()

                        # Parse the FOR loop part
                        for_match = re.match(
                            r'^\s*for\s+(.+?)$', for_part, re.I)
                        if for_match:
                            var_defs = for_match.group(1).strip()

                            # Parse the range expression
                            range_match = re.match(
                                r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', var_defs, re.I)
                            if range_match:
                                var_name, range_expr, step_str, index_var = range_match.groups()
                                step = int(step_str) if step_str else 1

                                # Parse the range expression
                                if ' to ' in range_expr:
                                    range_parts = range_expr.split(' to ')
                                    start = int(range_parts[0].strip())
                                    end = int(range_parts[1].strip())

                                    # Handle negative steps correctly
                                    if step < 0:
                                        values = list(
                                            range(start, end - 1, step))
                                    else:
                                        values = list(
                                            range(start, end + 1, step))

                                    # Process the executable part for each value
                                    if self._has_push_action(executable_part):
                                        for idx, value in enumerate(values):
                                            self.push_scope(
                                                is_private=True, is_loop_scope=True)
                                            self.current_scope().define(var_name, value, 'number')
                                            if index_var:
                                                self.current_scope().define(index_var, idx + 1, 'number')

                                            # Process the push/return call
                                            return_match = self._match_return_statement(
                                                executable_part)
                                            push_match = self._match_push_assignment(
                                                executable_part)
                                            if return_match:
                                                value_expr = return_match.group(
                                                    1).strip()
                                                values = self._evaluate_push_expression(
                                                    value_expr, line_number)
                                                for value in values:
                                                    self.output_values.setdefault(
                                                        'output', []).append(value)
                                                if 'output' not in self.output_variables:
                                                    self.output_variables.append(
                                                        'output')
                                            elif push_match:
                                                target, value_expr = push_match.groups()
                                                self._handle_push_assignment(
                                                    target, value_expr, line_number)
                                            else:
                                                self._process_push_call(
                                                    executable_part, line_number)
                                            self.pop_scope()

                                    # Skip this line in the main loop - it's been fully processed
                                    i += 1
                                    continue
                                else:
                                    i += 1
                                    continue
                            else:
                                i += 1
                                continue
                        else:
                            i += 1
                            continue
                    else:
                        i += 1
                        continue

            elif re.search(r'\.push\(', line, re.I) and not line.strip().lower().startswith("if "):
                # Use the push processor to handle this properly
                self._process_push_call(line, line_number)
                i += 1
            elif re.search(r'\bpush\s*\(', line, re.I) and not line.strip().lower().startswith("if "):
                # Handle push() function calls (e.g., push(mid))
                m = re.match(
                    r'^\s*push\s*\(\s*([^)]+)\s*\)\s*$', line, re.I)
                if m:
                    value_expr = m.group(1).strip()
                else:
                    raise SyntaxError(
                        f"Invalid push() syntax at line {line_number}")

                # Evaluate the expression and add it to output values
                try:
                    values = self._evaluate_push_expression(
                        value_expr, line_number)
                    for value in values:
                        self.output_values.setdefault(
                            'output', []).append(value)
                    if 'output' not in self.output_variables:
                        self.output_variables.append('output')
                except Exception as e:
                    raise ValueError(
                        f"Failed to evaluate push() expression at line {line_number}: {e}")
                i += 1
            # Array access like names(0) and names[3] will be handled by the expression evaluator
            # No need for special handling here
            elif re.search(r'([\w_]+)!(\w+)\.Label\s*\{([^}]+)\}', line):
                # Handle named dimension label assignment like Results!Quarter.Label{"Q1", "Q2", "Q3", "Q4"}
                m = re.match(
                    r'^\s*([\w_]+)!(\w+)\.Label\s*\{([^}]+)\}\s*$', line)
                if m:
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

                    i += 1
                    continue
                else:
                    # Not a valid label pattern, skip this line
                    i += 1
                    continue
            # Named dimension access like Results!Quarter("Q2") will be handled by the expression evaluator
            # No need for special handling here
            elif line.lower().strip().startswith("for "):
                # Continue with normal FOR loop processing for multi-line FOR loops
                pass

                # Handle special case: for element = haystack(mid) (array element assignment)
                m = re.match(
                    r'^\s*for\s+([\w_]+)\s*=\s*([\w_]+)\(([^)]+)\)\s*$', line, re.I)
                if m:
                    var_name, array_name, index_expr = m.groups()

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

                    i += 1
                    continue

                # Handle For var as type dim {dimensions} syntax
                m = re.match(
                    r'^\s*for\s+([\w_]+)\s+as\s+(\w+)\s+dim\s*(\{[^}]*\})', line, re.I)
                if m:
                    var_name, type_name, dim_str = m.groups()

                    # Parse the dimension string
                    dim_str = dim_str.strip()
                    if dim_str.startswith('{') and dim_str.endswith('}'):
                        dim_content = dim_str[1:-1].strip()
                        # Parse dimensions like {30, 2}
                        dim_parts = [part.strip()
                                     for part in dim_content.split(',')]
                        shape = []
                        for part in dim_parts:
                            try:
                                shape.append(int(part))
                            except ValueError:
                                # Handle named dimensions or other formats
                                shape.append(part)

                        # Create the array with the specified shape
                        default_value = 0 if type_name.lower() == 'number' else None
                        array_data = self.array_handler.create_array(
                            shape, default_value, pa.float64(), line_number)

                        # Define the variable with the created array
                        self.current_scope().define(var_name, array_data,
                                                    type_name.lower(), {'dim': dim_str}, False)
                    else:
                        constraints = {}
                        self.current_scope().define(var_name, None, type_name.lower(), constraints, True)
                    i += 1
                    continue

                # Handle For var as type dim number syntax
                m = re.match(
                    r'^\s*for\s+([\w_]+)\s+as\s+(\w+)\s+dim\s+(\d+)', line, re.I)
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
                        # Initialize array with zeros
                        initial_array = pa.array(
                            [0.0] * dim_size, type=pa.float64())

                    # Define the variable with the initialized array
                    self.current_scope().define(var_name, initial_array, type_name.lower(),
                                                {'dim': [(None, dim_size)]}, False)
                    i += 1
                    continue

                # Check for AND syntax in single FOR line first
                m = re.match(r'^\s*for\s+(.+?)(?:\s+do\s*$|\s*$)', line, re.I)
                if m:
                    var_defs = m.group(1).strip()
                    # Check if this is actually a single line FOR loop (contains push/return or :=)
                    # even if it ends with 'do'
                    has_executable = self._has_push_action(
                        line) or ':=' in line
                    is_block = line.strip().lower().endswith('do') and not has_executable

                    # Handle simple FOR declarations like "For z as text" or "For z of dollar"
                    if ' in ' not in var_defs.lower() and '=' not in var_defs and not re.search(r'\binit\b', var_defs, re.I):
                        var_list = []
                        if ' and ' in var_defs.lower():
                            var_parts = re.split(
                                r'\s+and\s+', var_defs, flags=re.I)
                            for var_part in var_parts:
                                var, type_name, constraints, expr = self._parse_variable_def(
                                    var_part, line_number)
                                var_list.append(
                                    (var, type_name, constraints, expr))
                        else:
                            var, type_name, constraints, expr = self._parse_variable_def(
                                var_defs, line_number)
                            var_list.append(
                                (var, type_name, constraints, expr))
                        for var, type_name, constraints, expr in var_list:
                            defining_scope = self.current_scope().get_defining_scope(var)
                            if defining_scope and not defining_scope.is_implicit_let(var):
                                raise ValueError(
                                    f"Variable '{var}' already defined in scope at line {line_number}")
                            if not defining_scope:
                                if self.current_scope().is_shadowed(var):
                                    print(
                                        f"Warning: FOR defines '{var}' which shadows a variable in an outer scope at line {line_number}")
                                self.current_scope().define(
                                    var, None, type_name, constraints, is_uninitialized=True)
                        i += 1
                        continue

                    # If this is a single line FOR loop with executable code, handle it specially
                    if has_executable:
                        # Extract the FOR loop part and the executable part
                        # The line format is: "For var in range do executable_code"
                        # We need to split at " do " and process both parts
                        if ' do ' in line.lower():
                            parts = line.split(' do ', 1)
                            if len(parts) == 2:
                                for_part = parts[0].strip()
                                executable_part = parts[1].strip()

                                # Parse the FOR loop part
                                for_match = re.match(
                                    r'^\s*for\s+(.+?)(?:\s+do\s*$|\s*$)', for_part, re.I)
                                if for_match:
                                    var_defs = for_match.group(1).strip()

                                    # Parse the range expression
                                    range_match = re.match(
                                        r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', var_defs, re.I)
                                    if range_match:
                                        var_name, range_expr, step_str, index_var = range_match.groups()
                                        step = int(step_str) if step_str else 1

                                        # Parse the range expression
                                        if ' to ' in range_expr:
                                            range_parts = range_expr.split(
                                                ' to ')
                                            start = int(range_parts[0].strip())
                                            end = int(range_parts[1].strip())
                                            # Handle negative steps correctly
                                            if step < 0:
                                                values = list(
                                                    range(start, end - 1, step))
                                            else:
                                                values = list(
                                                    range(start, end + 1, step))

                                            # Process the executable part for each value
                                            if self._has_push_action(executable_part):
                                                for idx, value in enumerate(values):
                                                    self.push_scope(
                                                        is_private=True, is_loop_scope=True)
                                                    self.current_scope().define(var_name, value, 'number')
                                                    if index_var:
                                                        self.current_scope().define(index_var, idx + 1, 'number')

                                                    # Process the push/return call
                                                    return_match = self._match_return_statement(
                                                        executable_part)
                                                    push_match = self._match_push_assignment(
                                                        executable_part)
                                                    if return_match:
                                                        value_expr = return_match.group(
                                                            1).strip()
                                                        values = self._evaluate_push_expression(
                                                            value_expr, line_number)
                                                        for value in values:
                                                            self.output_values.setdefault(
                                                                'output', []).append(value)
                                                        if 'output' not in self.output_variables:
                                                            self.output_variables.append(
                                                                'output')
                                                    elif push_match:
                                                        target, value_expr = push_match.groups()
                                                        self._handle_push_assignment(
                                                            target, value_expr, line_number)
                                                    else:
                                                        self._process_push_call(
                                                            executable_part, line_number)
                                                    self.pop_scope()

                                                # Skip this line in the main loop
                                                i += 1
                                                continue
                                            else:
                                                i += 1
                                                continue
                                        else:
                                            i += 1
                                            continue
                                    else:
                                        i += 1
                                        continue
                                else:
                                    i += 1
                                    continue
                            else:
                                i += 1
                                continue
                        else:
                            i += 1
                            continue

                    if ' AND ' in var_defs:
                        and_parts = var_defs.split(' AND ')
                        and_loops = []

                        for part in and_parts:
                            part = part.strip()
                            # Parse each part: "var in range" or "var in range index idx"
                            # First, find where the range expression ends
                            range_end = part.find(' step ')
                            if range_end == -1:
                                range_end = part.find(' index ')
                            if range_end == -1:
                                range_end = part.find(' Index ')
                            if range_end == -1:
                                range_end = len(part)

                            range_expr = part[part.find(
                                ' in ') + 4:range_end].strip()
                            var_name = part[:part.find(' in ')].strip()

                            # Extract step and index if present
                            step_str = None
                            index_var = None
                            remaining = part[range_end:].strip()

                            # Clean up range_expr if it contains index keyword
                            if ' index ' in range_expr:
                                range_expr = range_expr[:range_expr.find(
                                    ' index ')].strip()
                            elif ' Index ' in range_expr:
                                range_expr = range_expr[:range_expr.find(
                                    ' Index ')].strip()


                            # Check for step
                            step_match = re.search(
                                r'step\s+(\d+)', remaining, re.I)
                            if step_match:
                                step_str = step_match.group(1)
                                remaining = remaining[:step_match.start(
                                )] + remaining[step_match.end():]

                            # Check for index
                            index_match = re.search(
                                r'index\s+([\w_]+)', remaining, re.I)
                            if index_match:
                                index_var = index_match.group(1)
                            else:
                                pass

                            step = int(step_str) if step_str else 1

                            # Parse the range expression
                            if ' to ' in range_expr:
                                range_parts = range_expr.split(' to ')
                                start_expr = range_parts[0].strip()
                                end_expr = range_parts[1].strip()

                                # Check if this is a dynamic range first
                                is_dynamic = False
                                # Check if expressions contain variables (not just numbers)
                                if not start_expr.replace('.', '').replace('-', '').isdigit() or not end_expr.replace('.', '').replace('-', '').replace('+', '').replace('*', '').replace('/', '').replace('(', '').replace(')', '').isdigit():
                                    is_dynamic = True
                                    values = None  # Will be computed during execution
                                else:
                                    # Static range, compute now
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
                                raise SyntaxError(
                                    f"Invalid range expression: {range_expr} at line {line_number}")

                            # For sets, use the already computed values
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

                        if len(and_loops) >= 2:

                            if is_block:
                                # Process AND loops with block
                                # For dynamic ranges, we need to compute values during execution
                                # Start with the first loop's values
                                current_values = []
                                for loop in and_loops:
                                    if loop['is_dynamic']:
                                        # For dynamic loops, we'll compute values during execution
                                        current_values.append([])
                                    else:
                                        current_values.append(loop['values'])

                                # Generate combinations dynamically
                                def generate_combinations(loop_idx=0, current_combo=[]):
                                    if loop_idx >= len(and_loops):
                                        yield current_combo
                                        return

                                    loop = and_loops[loop_idx]
                                    if loop['is_dynamic']:
                                        # Compute dynamic range for this iteration
                                        # We need to evaluate the expressions with current variable values
                                        self.push_scope(
                                            is_private=True, is_loop_scope=True)
                                        # Set up variables from previous loops
                                        for i, prev_loop in enumerate(and_loops[:loop_idx]):
                                            self.current_scope().define(
                                                prev_loop['var_name'], current_combo[i], 'number')

                                        # Evaluate the dynamic range
                                        start = int(self.expr_evaluator.eval_expr(
                                            loop['start_expr'], self.current_scope().get_evaluation_scope(), line_number))
                                        end = int(self.expr_evaluator.eval_expr(
                                            loop['end_expr'], self.current_scope().get_evaluation_scope(), line_number))
                                        dynamic_values = list(
                                            range(start, end + 1, loop['step']))
                                        self.pop_scope()

                                        for val in dynamic_values:
                                            yield from generate_combinations(loop_idx + 1, current_combo + [val])
                                    else:
                                        # Static range
                                        for val in loop['values']:
                                            yield from generate_combinations(loop_idx + 1, current_combo + [val])

                                value_combinations = list(
                                    generate_combinations())

                                # Collect block lines
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
                                    elif next_line_clean.endswith("do"):
                                        depth += 1
                                    for_block_lines.append(
                                        (next_line, next_line_number))
                                    i += 1

                                # Execute for each combination
                                # Execute for each combination
                                for combo_idx, combo in enumerate(value_combinations):
                                    self.push_scope(
                                        is_private=True, is_loop_scope=True)
                                    for loop_idx, loop in enumerate(and_loops):
                                        self.current_scope().define(
                                            loop['var_name'], combo[loop_idx], 'number')
                                        if loop['index_var']:
                                            # For AND loops, we need to track the iteration count for each loop separately
                                            # This is more complex than regular loops because we're doing a product
                                            # We need to calculate which iteration this is for each individual loop
                                            loop_values = loop['values']
                                            current_value = combo[loop_idx]
                                            # Find the position of this value in the loop's values (1-based)
                                            value_index = loop_values.index(
                                                current_value) + 1
                                            # If index_var already exists in the current scope, update it instead of redefining
                                            if loop['index_var'] in self.current_scope().variables:
                                                self.current_scope().update(
                                                    loop['index_var'], value_index)
                                            else:
                                                self.current_scope().define(
                                                    loop['index_var'], value_index, 'number')
                                    self.control_flow._process_block(
                                        for_block_lines)

                                    # Check if exit loop was requested immediately after processing block
                                    if self.exit_loop:
                                        self.exit_loop = False
                                        self.pop_scope()
                                        break

                                    self.pop_scope()

                                continue
                            else:
                                # Single line AND loops - find assignment
                                if i + 1 < len(lines):
                                    next_line, next_line_number = lines[i + 1]
                                    if ':=' in next_line:
                                        # For dynamic ranges, we need to compute values during execution
                                        # Generate combinations dynamically
                                        def generate_combinations(loop_idx=0, current_combo=[]):
                                            if loop_idx >= len(and_loops):
                                                yield current_combo
                                                return

                                            loop = and_loops[loop_idx]
                                            if loop['is_dynamic']:
                                                # Compute dynamic range for this iteration
                                                # We need to evaluate the expressions with current variable values
                                                self.push_scope(
                                                    is_private=True, is_loop_scope=True)
                                                # Set up variables from previous loops
                                                for i, prev_loop in enumerate(and_loops[:loop_idx]):
                                                    self.current_scope().define(
                                                        prev_loop['var_name'], current_combo[i], 'number')

                                                # Evaluate the dynamic range
                                                start = int(self.expr_evaluator.eval_expr(
                                                    loop['start_expr'], self.current_scope().get_evaluation_scope(), line_number))
                                                end = int(self.expr_evaluator.eval_expr(
                                                    loop['end_expr'], self.current_scope().get_evaluation_scope(), line_number))
                                                dynamic_values = list(
                                                    range(start, end + 1, loop['step']))
                                                self.pop_scope()

                                                for val in dynamic_values:
                                                    yield from generate_combinations(loop_idx + 1, current_combo + [val])
                                            else:
                                                # Static range
                                                for val in loop['values']:
                                                    yield from generate_combinations(loop_idx + 1, current_combo + [val])

                                        value_combinations = list(
                                            generate_combinations())

                                        for combo in value_combinations:
                                            self.push_scope(
                                                is_private=True, is_loop_scope=True)
                                            for loop_idx, loop in enumerate(and_loops):
                                                self.current_scope().define(
                                                    loop['var_name'], combo[loop_idx], 'number')
                                                if loop['index_var']:
                                                    # If index_var already exists in the current scope, update it instead of redefining
                                                    if loop['index_var'] in self.current_scope().variables:
                                                        self.current_scope().update(
                                                            loop['index_var'], loop_idx + 1)
                                                    else:
                                                        self.current_scope().define(
                                                            loop['index_var'], loop_idx + 1, 'number')
                                            self.array_handler.evaluate_line_with_assignment(
                                                next_line, next_line_number, self.current_scope().get_evaluation_scope())
                                            self.pop_scope()

                                        i += 2
                                        continue

                # Check for nested FOR loops
                nested_for_loops = []
                current_i = i
                while current_i < len(lines) and lines[current_i][0].strip().lower().startswith("for "):
                    current_line, current_line_number = lines[current_i]
                    m = re.match(
                        r'^\s*FOR\s+(.+?)(?:\s+do\s*$|\s*$)', current_line, re.I)
                    if not m:
                        break
                    var_defs = m.group(1).strip()
                    is_block = current_line.strip().lower().endswith('do')

                    # Check for FOR loops with ranges or sets
                    range_match = re.match(
                        r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', var_defs, re.I)
                    if range_match:
                        var_name, range_expr, step_str, index_var = range_match.groups()
                        step = int(step_str) if step_str else 1

                        # Parse the range expression
                        if ' to ' in range_expr:
                            # Clean up the range expression by removing any step part
                            clean_range_expr = range_expr.split(
                                ' step ')[0] if ' step ' in range_expr else range_expr
                            range_parts = clean_range_expr.split(' to ')

                            # Evaluate start and end as expressions instead of literal integers
                            try:
                                start_expr = range_parts[0].strip()
                                end_expr = range_parts[1].strip()

                                # Evaluate start expression
                                start_value = self.expr_evaluator.eval_expr(
                                    start_expr, self.current_scope().get_evaluation_scope(), current_line_number)
                                start = int(start_value)

                                # Evaluate end expression
                                end_value = self.expr_evaluator.eval_expr(
                                    end_expr, self.current_scope().get_evaluation_scope(), current_line_number)
                                end = int(end_value)

                                values = list(range(start, end + 1, step))
                            except Exception as e:
                                # Fallback to original behavior for simple literals
                                start = int(range_parts[0].strip())
                                end = int(range_parts[1].strip())
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
                            break

                        nested_for_loops.append({
                            'var_name': var_name,
                            'values': values,
                            'index_var': index_var,
                            'is_block': is_block,
                            'line_number': current_line_number
                        })
                        current_i += 1
                    else:
                        break

                if len(nested_for_loops) == 2 and not any(loop['is_block'] for loop in nested_for_loops):
                    # Use zip logic for exactly two consecutive FORs
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
                        for a, b in zip(vals1, vals2):
                            self.push_scope(is_private=True,
                                            is_loop_scope=True)
                            self.current_scope().define(var1, a, 'number')
                            self.current_scope().define(var2, b, 'number')
                            self.array_handler.evaluate_line_with_assignment(
                                assignment_line, assignment_line_number, self.current_scope().get_evaluation_scope())
                            self.pop_scope()
                        i = current_i
                        continue

                if len(nested_for_loops) > 1:
                    # Handle nested FOR loops

                    # Find the assignment line after the nested loops
                    assignment_line = None
                    assignment_line_number = None
                    if current_i < len(lines):
                        next_line, next_line_number = lines[current_i]
                        if ':=' in next_line:
                            assignment_line = next_line
                            assignment_line_number = next_line_number
                            current_i += 1

                    if assignment_line:
                        # Generate all combinations of nested loop values
                        from itertools import product
                        value_combinations = list(
                            product(*[loop['values'] for loop in nested_for_loops]))

                        # Execute for each combination
                        for combo in value_combinations:
                            self.push_scope(is_private=True,
                                            is_loop_scope=True)
                            for loop_idx, loop in enumerate(nested_for_loops):
                                self.current_scope().define(
                                    loop['var_name'], combo[loop_idx], 'number')
                                if loop['index_var']:
                                    # If index_var already exists in the current scope, update it instead of redefining
                                    if loop['index_var'] in self.current_scope().variables:
                                        self.current_scope().update(
                                            loop['index_var'], loop_idx + 1)
                                    else:
                                        self.current_scope().define(
                                            loop['index_var'], loop_idx + 1, 'number')
                            self.array_handler.evaluate_line_with_assignment(
                                assignment_line, assignment_line_number, self.current_scope().get_evaluation_scope())
                            self.pop_scope()

                        # Skip all the FOR lines and the assignment line
                        i = current_i
                        continue
                    else:
                        # No assignment line found, process as regular FOR loops
                        pass
                    # Use zip logic for exactly two consecutive FORs
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
                        for a, b in zip(vals1, vals2):
                            self.push_scope(is_private=True,
                                            is_loop_scope=True)
                            self.current_scope().define(var1, a, 'number')
                            self.current_scope().define(var2, b, 'number')
                            self.array_handler.evaluate_line_with_assignment(
                                assignment_line, assignment_line_number, self.current_scope().get_evaluation_scope())
                            self.pop_scope()
                        i = current_i
                        continue

                # Process as single FOR loop (original logic)
                m = re.match(r'^\s*for\s+(.+?)(?:\s+do\s*$|\s*$)', line, re.I)
                if not m:
                    raise SyntaxError(
                        f"Invalid FOR syntax at line {line_number}")
                var_defs = m.group(1).strip()
                is_block = line.strip().lower().endswith('do')

                # Check for FOR loops with ranges or sets
                range_match = re.match(
                    r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', var_defs, re.I)
                if range_match:
                    var_name, range_expr, step_str, index_var = range_match.groups()
                    step = int(step_str) if step_str else 1

                    # Parse the range expression
                    if ' to ' in range_expr:
                        # Range like "1 to 3"
                        range_parts = range_expr.split(' to ')

                        # Evaluate start and end as expressions instead of literal integers
                        try:
                            start_expr = range_parts[0].strip()
                            end_expr = range_parts[1].strip()

                            # Evaluate start expression
                            start_value = self.expr_evaluator.eval_expr(
                                start_expr, self.current_scope().get_evaluation_scope(), current_line_number)
                            start = int(start_value)

                            # Evaluate end expression
                            end_value = self.expr_evaluator.eval_expr(
                                end_expr, self.current_scope().get_evaluation_scope(), current_line_number)
                            end = int(end_value)
                        except Exception as e:
                            # Fallback to original behavior for simple literals
                            start = int(range_parts[0].strip())
                            end = int(range_parts[1].strip())
                        # Handle negative steps correctly
                        if step < 0:
                            # For negative steps, range goes from start down to end (inclusive)
                            values = list(range(start, end - 1, step))
                        else:
                            # For positive steps, range goes from start up to end (inclusive)
                            values = list(range(start, end + 1, step))
                    elif range_expr.startswith('{') and range_expr.endswith('}'):
                        # Set like "{1, 2, 3}"
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
                            eval_scope = self.current_scope().get_evaluation_scope()
                            evaluated = self.expr_evaluator.eval_or_eval_array(
                                range_expr, eval_scope, current_line_number)
                        except Exception:
                            raise SyntaxError(
                                f"Invalid range expression: {range_expr} at line {line_number}")
                        if hasattr(evaluated, 'to_pylist'):
                            values = evaluated.to_pylist()
                        elif isinstance(evaluated, (list, tuple)):
                            values = list(evaluated)
                        elif evaluated is None:
                            values = []
                        else:
                            values = [evaluated]


                    # Handle consecutive FORs as zipped pairs
                    if i + 1 < len(lines):
                        next_line, next_line_number = lines[i + 1]
                        if next_line.strip().lower().startswith('for '):
                            m2 = re.match(
                                r'^\s*FOR\s+([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$', next_line, re.I)
                            if m2:
                                var2, range2, step2, _ = m2.groups()
                                step2 = int(step2) if step2 else 1
                                # Parse range for second FOR
                                if ' to ' in range2:
                                    s2, e2 = range2.split(' to ')
                                    vals2 = list(
                                        range(int(s2.strip()), int(e2.strip()) + 1, step2))
                                elif range2.startswith('{') and range2.endswith('}'):
                                    vals2 = [float(v.strip())
                                             for v in range2[1:-1].split(',')]
                                else:
                                    try:
                                        eval_scope = self.current_scope().get_evaluation_scope()
                                        evaluated = self.expr_evaluator.eval_or_eval_array(
                                            range2, eval_scope, next_line_number)
                                        if hasattr(evaluated, 'to_pylist'):
                                            vals2 = evaluated.to_pylist()
                                        elif isinstance(evaluated, (list, tuple)):
                                            vals2 = list(evaluated)
                                        elif evaluated is None:
                                            vals2 = []
                                        else:
                                            vals2 = [evaluated]
                                    except Exception:
                                        vals2 = []

                                # Look for the next executable line (assignment or push/return call)
                                next_executable_line = None
                                next_executable_line_number = None
                                for j in range(i + 2, len(lines)):
                                    candidate_line, candidate_line_number = lines[j]
                                    if ':=' in candidate_line or self._has_push_action(candidate_line):
                                        next_executable_line = candidate_line
                                        next_executable_line_number = candidate_line_number
                                        break

                                if next_executable_line:
                                    for a, b in zip(values, vals2):
                                        self.push_scope(
                                            is_private=True, is_loop_scope=True)
                                        self.current_scope().define(var_name, a, 'number')
                                        self.current_scope().define(var2, b, 'number')
                                        if ':=' in next_executable_line:
                                            self.array_handler.evaluate_line_with_assignment(
                                                next_executable_line, next_executable_line_number, self.current_scope().get_evaluation_scope())
                                        elif self._has_push_action(next_executable_line):
                                            return_match = self._match_return_statement(
                                                next_executable_line)
                                            push_match = self._match_push_assignment(
                                                next_executable_line)
                                            if return_match:
                                                value_expr = return_match.group(
                                                    1).strip()
                                                values = self._evaluate_push_expression(
                                                    value_expr, next_executable_line_number)
                                                for value in values:
                                                    self.output_values.setdefault(
                                                        'output', []).append(value)
                                                if 'output' not in self.output_variables:
                                                    self.output_variables.append(
                                                        'output')
                                            elif push_match:
                                                target, value_expr = push_match.groups()
                                                self._handle_push_assignment(
                                                    target, value_expr, next_executable_line_number)
                                            else:
                                                # Process the .push() call directly using the existing logic
                                                self._process_push_call(
                                                    next_executable_line, next_executable_line_number)
                                        self.pop_scope()
                                    # Skip both FOR lines and the executable line
                                    i = j + 1
                                    continue

                    if is_block:
                        # Process FOR block
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
                            elif next_line_clean.startswith("for ") and " do" in next_line_clean.lower():
                                depth += 1
                            elif next_line_clean.startswith("let ") and " then " in next_line_clean:
                                depth += 1
                            elif next_line_clean.startswith("if ") and next_line_clean.endswith("then"):
                                depth += 1
                            for_block_lines.append(
                                (next_line, next_line_number))
                            i += 1

                        if depth > 0:
                            raise SyntaxError(
                                f"Unclosed FOR block starting at line {line_number}")

                        # Execute the block for each value
                        for idx, value in enumerate(values):
                            # Create an isolated loop scope so iteration variables can be reused safely
                            self.push_scope(is_private=True,
                                            is_loop_scope=True)
                            loop_scope = self.current_scope()

                            # Define/update the loop variable in the loop scope
                            if var_name in loop_scope.variables:
                                loop_scope.update(var_name, value, line_number)
                            else:
                                loop_scope.define(var_name, value, 'number')

                            # Define/update index variable (1-based) inside the loop scope
                            if index_var:
                                if index_var in loop_scope.variables:
                                    loop_scope.update(index_var, idx + 1)
                                else:
                                    loop_scope.define(
                                        index_var, idx + 1, 'number')

                            # Process the block body with the loop scope active
                            self.control_flow._process_block(for_block_lines)

                            # Propagate variables defined in the loop scope back to the parent
                            parent_scope = loop_scope.parent
                            if parent_scope:
                                for name, val in loop_scope.variables.items():
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
                        # Skip the FOR statement in the main loop
                        continue
                    else:
                        # Single line FOR
                        # Look for the next executable line (assignment or push/return call)
                        next_executable_line = None
                        next_executable_line_number = None
                        skip_lines = 0
                        for j in range(i + 1, len(lines)):
                            candidate_line, candidate_line_number = lines[j]
                            if ':=' in candidate_line or self._has_push_action(candidate_line):
                                next_executable_line = candidate_line
                                next_executable_line_number = candidate_line_number
                                skip_lines = j - i
                                break

                        if next_executable_line:
                            if ':=' in next_executable_line:
                                # Execute for each value
                                for idx, value in enumerate(values):
                                    self.push_scope(
                                        is_private=True, is_loop_scope=True)
                                    self.current_scope().define(var_name, value, 'number')
                                    if index_var:
                                        # If index_var already exists in the current scope, update it instead of redefining
                                        if index_var in self.current_scope().variables:
                                            self.current_scope().update(index_var, idx + 1)
                                        else:
                                            self.current_scope().define(index_var, idx + 1, 'number')
                                    self.array_handler.evaluate_line_with_assignment(
                                        next_executable_line, next_executable_line_number, self.current_scope().get_evaluation_scope())
                                    self.pop_scope()
                                # Skip the FOR statement and all lines up to the executable line
                                i += skip_lines + 1
                                continue
                            elif self._has_push_action(next_executable_line):
                                # Execute for each value
                                for idx, value in enumerate(values):
                                    self.push_scope(
                                        is_private=True, is_loop_scope=True)
                                    self.current_scope().define(var_name, value, 'number')
                                    if index_var:
                                        # If index_var already exists in the current scope, update it instead of redefining
                                        if index_var in self.current_scope().variables:
                                            self.current_scope().update(index_var, idx + 1)
                                        else:
                                            self.current_scope().define(index_var, idx + 1, 'number')
                                    return_match = self._match_return_statement(
                                        next_executable_line)
                                    push_match = self._match_push_assignment(
                                        next_executable_line)
                                    if return_match:
                                        value_expr = return_match.group(
                                            1).strip()
                                        values = self._evaluate_push_expression(
                                            value_expr, next_executable_line_number)
                                        for value in values:
                                            self.output_values.setdefault(
                                                'output', []).append(value)
                                        if 'output' not in self.output_variables:
                                            self.output_variables.append(
                                                'output')
                                    elif push_match:
                                        target, value_expr = push_match.groups()
                                        self._handle_push_assignment(
                                            target, value_expr, next_executable_line_number)
                                    else:
                                        # Process the .push() call directly using the existing logic
                                        self._process_push_call(
                                            next_executable_line, next_executable_line_number)
                                    self.pop_scope()
                                # Skip the FOR statement and all lines up to the executable line
                                i += skip_lines + 1
                                continue
                        i += 1
                        continue

                # Handle nested FOR loops with AND
                elif ' and ' in var_defs.lower() and ' in ' in var_defs.lower():
                    # Parse multiple FOR loops like "a in {1, 2} AND b in 9 to 15 step 3"
                    var_parts = re.split(r'\s+and\s+', var_defs, flags=re.I)
                    loop_configs = []

                    for var_part in var_parts:
                        range_match = re.match(
                            r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(\d+))?(?:\s+index\s+([\w_]+))?$', var_part.strip(), re.I)
                        if range_match:
                            var_name, range_expr, step_str, index_var = range_match.groups()
                            step = int(step_str) if step_str else 1

                            # Parse the range expression
                            if ' to ' in range_expr:
                                range_parts = range_expr.split(' to ')
                                start_expr = range_parts[0].strip()
                                end_expr = range_parts[1].strip()

                                # Check if this is a dynamic range first
                                is_dynamic = False
                                # Check if expressions contain variables (not just numbers)
                                if not start_expr.replace('.', '').replace('-', '').isdigit() or not end_expr.replace('.', '').replace('-', '').replace('+', '').replace('*', '').replace('/', '').replace('(', '').replace(')', '').isdigit():
                                    is_dynamic = True
                                    values = None  # Will be computed during execution
                                else:
                                    # Static range, compute now
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
                                    eval_scope = self.current_scope().get_evaluation_scope()
                                    evaluated = self.expr_evaluator.eval_or_eval_array(
                                        range_expr, eval_scope, line_number)
                                    if hasattr(evaluated, 'to_pylist'):
                                        values = evaluated.to_pylist()
                                    elif isinstance(evaluated, (list, tuple)):
                                        values = list(evaluated)
                                    elif evaluated is None:
                                        values = []
                                    else:
                                        values = [evaluated]
                                except Exception:
                                    raise SyntaxError(
                                        f"Invalid range expression: {range_expr} at line {line_number}")

                            loop_configs.append((var_name, values, index_var, is_dynamic,
                                                start_expr if is_dynamic else None, end_expr if is_dynamic else None, step))
                        else:
                            raise SyntaxError(
                                f"Invalid FOR loop syntax: {var_part} at line {line_number}")


                    if is_block:
                        # Process FOR block
                        for_block_lines = []
                        i += 1
                        depth = 1
                        while i < len(lines) and depth > 0:
                            next_line, next_line_number = lines[i]
                            next_line_clean = next_line.strip().lower()
                            if self._is_keyword(next_line_clean, "end"):
                                depth -= 1
                                # Include 'end' if it closes a nested block, not the FOR itself
                                if depth > 0:
                                    for_block_lines.append(
                                        (next_line, next_line_number))
                                    i += 1
                                    continue
                                else:
                                    i += 1
                                    break
                            elif next_line_clean.startswith("for ") and " do" in next_line_clean.lower():
                                depth += 1
                            # Correctly account for nested IF blocks inside FOR blocks
                            elif next_line_clean.startswith("if ") and next_line_clean.endswith("then"):
                                depth += 1
                            elif next_line_clean.startswith("for ") and " do" in next_line_clean.lower():
                                depth += 1

                            for_block_lines.append(
                                (next_line, next_line_number))
                            i += 1

                        if depth > 0:
                            raise SyntaxError(
                                f"Unclosed FOR block starting at line {line_number}")

                        # Generate all combinations
                        from itertools import product

                        # For dynamic ranges, we need to compute values during execution
                        def generate_combinations(loop_idx=0, current_combo=[]):
                            if loop_idx >= len(loop_configs):
                                yield current_combo
                                return

                            var_name, values, index_var, is_dynamic, start_expr, end_expr, step = loop_configs[
                                loop_idx]

                            if is_dynamic:
                                # Compute dynamic range for this iteration
                                # We need to evaluate the expressions with current variable values
                                self.push_scope(
                                    is_private=True, is_loop_scope=True)
                                # Set up variables from previous loops
                                for i, prev_config in enumerate(loop_configs[:loop_idx]):
                                    prev_var_name = prev_config[0]
                                    self.current_scope().define(
                                        prev_var_name, current_combo[i], 'number')

                                # Evaluate the dynamic range
                                start = int(self.expr_evaluator.eval_expr(
                                    start_expr, self.current_scope().get_evaluation_scope(), line_number))
                                end = int(self.expr_evaluator.eval_expr(
                                    end_expr, self.current_scope().get_evaluation_scope(), line_number))
                                dynamic_values = list(
                                    range(start, end + 1, step))
                                self.pop_scope()

                                for val in dynamic_values:
                                    yield from generate_combinations(loop_idx + 1, current_combo + [val])
                            else:
                                # Static range
                                for val in values:
                                    yield from generate_combinations(loop_idx + 1, current_combo + [val])

                        value_combinations = list(generate_combinations())

                        # Execute the block for each combination
                        for combo_idx, combo in enumerate(value_combinations):
                            # Check if exit loop was requested
                            if self.exit_loop:
                                self.exit_loop = False
                                break

                            loop_scope = self.current_scope()
                            for var_idx, (var_name, _, index_var, _, _, _, _) in enumerate(loop_configs):
                                defining_scope = loop_scope.get_defining_scope(
                                    var_name)
                                if defining_scope:
                                    defining_scope.update(
                                        var_name, combo[var_idx], line_number)
                                else:
                                    loop_scope.define(
                                        var_name, combo[var_idx], 'number')
                                if index_var:
                                    defining_index_scope = loop_scope.get_defining_scope(
                                        index_var)
                                    if defining_index_scope:
                                        defining_index_scope.update(
                                            index_var, var_idx + 1, line_number)
                                    else:
                                        loop_scope.define(
                                            index_var, var_idx + 1, 'number')
                            self.control_flow._process_block(for_block_lines)
                            # After processing the loop block, propagate values defined in the loop scope
                            # back to the parent so they become available for subsequent lines.
                            parent_scope = loop_scope.parent
                            if parent_scope:
                                for var_name, var_value in loop_scope.variables.items():
                                    try:
                                        parent_scope.update(
                                            var_name, var_value)
                                    except Exception:
                                        inferred_type = self.array_handler.infer_type(
                                            var_value, line_number)
                                        parent_scope.define(
                                            var_name, var_value, inferred_type, {}, is_uninitialized=False)

                            # Check if exit loop was requested immediately after processing block
                            if self.exit_loop:
                                self.exit_loop = False
                                break

                    else:
                        # Single line nested FOR
                        if i + 1 < len(lines):
                            next_line, next_line_number = lines[i + 1]
                            if ':=' in next_line:
                                # Generate all combinations
                                from itertools import product

                                # For dynamic ranges, we need to compute values during execution
                                def generate_combinations(loop_idx=0, current_combo=[]):
                                    if loop_idx >= len(loop_configs):
                                        yield current_combo
                                        return

                                    var_name, values, index_var, is_dynamic, start_expr, end_expr, step = loop_configs[
                                        loop_idx]

                                    if is_dynamic:
                                        # Compute dynamic range for this iteration
                                        # We need to evaluate the expressions with current variable values
                                        self.push_scope(
                                            is_private=True, is_loop_scope=True)
                                        # Set up variables from previous loops
                                        for i, prev_config in enumerate(loop_configs[:loop_idx]):
                                            prev_var_name = prev_config[0]
                                            self.current_scope().define(
                                                prev_var_name, current_combo[i], 'number')

                                        # Evaluate the dynamic range
                                        start = int(self.expr_evaluator.eval_expr(
                                            start_expr, self.current_scope().get_evaluation_scope(), line_number))
                                        end = int(self.expr_evaluator.eval_expr(
                                            end_expr, self.current_scope().get_evaluation_scope(), line_number))
                                        dynamic_values = list(
                                            range(start, end + 1, step))
                                        self.pop_scope()

                                        for val in dynamic_values:
                                            yield from generate_combinations(loop_idx + 1, current_combo + [val])
                                    else:
                                        # Static range
                                        for val in values:
                                            yield from generate_combinations(loop_idx + 1, current_combo + [val])

                                value_combinations = list(
                                    generate_combinations())

                                # Execute for each combination
                                for combo_idx, combo in enumerate(value_combinations):
                                    self.push_scope(
                                        is_private=True, is_loop_scope=True)
                                    for var_idx, (var_name, _, index_var, _, _, _, _) in enumerate(loop_configs):
                                        self.current_scope().define(
                                            var_name, combo[var_idx], 'number')
                                        if index_var:
                                            # If index_var already exists in the current scope, update it instead of redefining
                                            if index_var in self.current_scope().variables:
                                                self.current_scope().update(index_var, var_idx + 1)
                                            else:
                                                self.current_scope().define(index_var, var_idx + 1, 'number')
                                    self.array_handler.evaluate_line_with_assignment(
                                        next_line, next_line_number, self.current_scope().get_evaluation_scope())
                                    self.pop_scope()
                        i += 1
                        continue

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

                        # Handle comma-separated variable lists (e.g., "low, high as number")
                        if 'var_list' in constraints:
                            var_names = constraints['var_list']
                            # Clear the original entry and add each variable separately
                            var_list.clear()
                            for var_name in var_names:
                                var_list.append(
                                    (var_name, type_name, constraints.copy(), value))

                except Exception as e:
                    raise SyntaxError(
                        f"Invalid FOR variable definition at line {line_number}")

                if (not is_block and len(var_list) == 1 and var_list[0][3] is None):
                    var, type_name, constraints, _ = var_list[0]
                    defining_scope = self.current_scope().get_defining_scope(var)
                    if defining_scope and defining_scope.is_implicit_let(var):
                        if type_name:
                            defining_scope.types[var] = type_name
                        if constraints:
                            defining_scope.constraints[var] = constraints
                        defining_scope.clear_implicit_let(var)
                        i += 1
                        continue

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
                            public_fields = public_type_fields(
                                self.types_defined[type_name.lower()])
                            struct_fields = [(f.lower(), pa.string() if public_fields.get(f.lower()) == 'text' else pa.float64())
                                             for f in public_fields]
                            struct_fields.append(('value', pa.float64()))
                            pa_type = pa.struct(struct_fields)
                            default_struct = {f.lower(): with_constraints.get(
                                f) for f in public_fields}
                            default_struct['value'] = float(
                                value) if value else 1.0
                            array = self.array_handler.create_array(
                                shape, default_struct, pa_type, line_number)
                            self.current_scope().define(var, array, 'array', constraints, is_uninitialized=False)
                            for field in public_fields:
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
                    if constraints.get('with'):
                        evaluated_value = self._apply_with_constraints(
                            evaluated_value,
                            constraints.get('with', {}),
                            self.current_scope().get_full_scope(),
                            line_number,
                            type_name=type_name)
                    # Infer type: explicit type, constructor pattern, or heuristic
                    constructor_match = None
                    try:
                        constructor_match = re.match(
                            r'new\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', str(value), re.I)
                    except Exception:
                        constructor_match = None
                    if type_name:
                        inferred_type = type_name
                    elif constructor_match:
                        inferred_type = constructor_match.group(1)
                    else:
                        inferred_type = self.array_handler.infer_type(
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
                        # Store constraint expression if it exists
                        if value and str(value) != str(evaluated_value):
                            constraints['constant'] = str(value)
                        self.get_global_scope().define(var, evaluated_value,
                                                       inferred_type, constraints, False)
                    pending = list(
                        self.current_scope().pending_assignments.items())
                    for key, (expr, ln, deps) in pending:
                        scope_pending = self.current_scope().pending_assignments
                        unresolved = any(
                            self.has_unresolved_dependency(dep, scope=self.current_scope(),
                                                           scope_pending=scope_pending)
                            for dep in deps)
                        if not unresolved:
                            try:
                                self.array_handler.evaluate_line_with_assignment(
                                    expr, ln, self.current_scope().get_evaluation_scope())
                                del self.current_scope(
                                ).pending_assignments[key]
                            except Exception as e:
                                pass
                    # Attempt to resolve global pending assignments impacted by this definition
                    pending_vars = [
                        pending_var for pending_var in list(self.pending_assignments.keys())
                        if not pending_var.startswith('__line_')
                    ]
                    for pending_var in pending_vars:
                        self._attempt_resolve_pending_var(
                            pending_var, line_number)
                    i += 1
                    continue

                # Handle multiple FOR assignments (e.g., For a = 20 and b = 4)
                if not is_block and len(var_list) > 1:
                    # Check if all variables have assignments
                    all_have_assignments = all(
                        v[3] is not None for v in var_list)

                    if all_have_assignments:
                        # Handle assignments
                        for var, type_name, constraints, value in var_list:
                            evaluated_value = self.expr_evaluator.eval_expr(
                                str(value), self.current_scope().get_evaluation_scope(), line_number)
                            if constraints.get('with'):
                                evaluated_value = self._apply_with_constraints(
                                    evaluated_value,
                                    constraints.get('with', {}),
                                    self.current_scope().get_full_scope(),
                                    line_number,
                                    type_name=type_name)
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
                            defining_scope = self.current_scope().get_defining_scope(var)
                            if defining_scope:
                                if defining_scope.types.get(var) == 'number' and not isinstance(evaluated_value, (int, float)):
                                    raise TypeError(
                                        f"Cannot assign non-numeric value {evaluated_value} to '{var}' at line {line_number}")
                                defining_scope.update(var, float(evaluated_value) if isinstance(
                                    evaluated_value, (int, float)) else evaluated_value, line_number)
                            else:
                                self.get_global_scope().define(var, evaluated_value,
                                                               inferred_type, constraints, False)
                        pending = list(
                            self.current_scope().pending_assignments.items())
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
                                    del self.current_scope(
                                    ).pending_assignments[key]
                                except Exception as e:
                                    pass
                        pending_vars = [
                            pending_var for pending_var in list(self.pending_assignments.keys())
                            if not pending_var.startswith('__line_')
                        ]
                        for pending_var in pending_vars:
                            self._attempt_resolve_pending_var(
                                pending_var, line_number)
                        i += 1
                        continue
                    else:
                        # Handle declarations without assignments (e.g., "for low, high as number")
                        for var, type_name, constraints, _ in var_list:
                            # Initialize with reasonable defaults based on variable names
                            if var == 'low':
                                default_value = 1
                            elif var == 'high':
                                # Try to get haystack length, default to 10 if not available
                                try:
                                    haystack = self.current_scope().get('haystack')
                                    if haystack and hasattr(haystack, '__len__'):
                                        default_value = len(haystack)
                                    else:
                                        default_value = 10
                                except:
                                    default_value = 10
                            else:
                                default_value = 0

                            self.current_scope().define(var, default_value, type_name,
                                                        constraints, is_uninitialized=False)
                        i += 1
                        continue

                # Handle FOR blocks or LET followers
                for var, _, _, _ in var_list:
                    defining_scope = self.current_scope().get_defining_scope(var)
                    if (defining_scope and not defining_scope.is_implicit_let(var) and
                            (not is_block and not (i + 1 < len(lines) and lines[i + 1][0].lower().startswith('let ')))):
                        raise ValueError(
                            f"Variable '{var}' already defined in scope at line {line_number}")

                # For comma-separated variables without assignments, define them in current scope
                if not is_block and len(var_list) > 1 and all(v[3] is None for v in var_list):
                    for var, type_name, constraints, _ in var_list:
                        self.current_scope().define(var, None, type_name, constraints, is_uninitialized=True)
                        self.current_scope().mark_implicit_let(var)

                if not is_block:
                    init_entries = [
                        (v, t, c) for v, t, c, _ in var_list
                        if c and 'init' in c
                    ]
                    if init_entries:
                        for var, type_name, constraints in init_entries:
                            init_expr = constraints.get('init')
                            try:
                                init_value = self.expr_evaluator.eval_or_eval_array(
                                    str(init_expr), self.current_scope().get_evaluation_scope(), line_number)
                            except Exception:
                                init_value = None
                            defining_scope = self.current_scope().get_defining_scope(var) or self.current_scope()
                            if var in defining_scope.variables:
                                defining_scope.update(
                                    var, init_value, line_number)
                            else:
                                defining_scope.define(
                                    var, init_value, type_name, constraints, is_uninitialized=False)
                    i += 1
                    continue

                self.push_scope(is_private=True, is_loop_scope=True)
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
                            self.get_global_scope().define(var, None, type_name,
                                                           constraints, is_uninitialized=True)
                    else:
                        self.get_global_scope().define(var, None, type_name,
                                                       constraints, is_uninitialized=True)

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
                    init_entry = next(
                        ((v, t, c) for v, t, c, _ in var_list if c and 'init' in c), None)
                    if init_entry:
                        init_expr = init_entry[2].get('init')
                        values = []
                        call_match = re.match(
                            r'^([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)$', str(init_expr))
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
                            self.pop_scope()
                        continue
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
                    self.pop_scope()
                else:
                    if len(var_list) == 1 and i + 1 < len(lines) and lines[i + 1][0].lower().startswith('let '):
                        for_block_lines = []
                        i += 1
                        for_block_lines.append(lines[i])
                        if i + 1 < len(lines) and not lines[i + 1][0].strip().lower() == "end":
                            for_block_lines.append(lines[i + 1])
                            i += 1
                        init_entries = [
                            (v, t, c) for v, t, c, _ in var_list
                            if c and 'init' in c
                        ]
                        if init_entries:
                            init_expr = init_entries[0][2].get('init')
                            try:
                                init_value = self.expr_evaluator.eval_or_eval_array(
                                    str(init_expr), self.current_scope().get_evaluation_scope(), line_number)
                            except Exception:
                                init_value = None
                            for var, type_name, constraints in init_entries:
                                defining_scope = self.current_scope().get_defining_scope(var) or self.current_scope()
                                if var in defining_scope.variables:
                                    defining_scope.update(
                                        var, init_value, line_number)
                                else:
                                    defining_scope.define(
                                        var, init_value, type_name, constraints, is_uninitialized=False)
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
                        self.pop_scope()
                    else:
                        init_entries = [(var, type_name, constraints) for var, type_name,
                                        constraints, _ in var_list if constraints and 'init' in constraints]
                        if init_entries:
                            for var, type_name, constraints in init_entries:
                                init_expr = constraints.get('init')
                                values = []
                                call_match = re.match(
                                    r'^([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)$', str(init_expr))
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
                                if not values:
                                    call_match = re.match(
                                        r'^([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)$', str(init_expr))
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
                                                    call_name, arg_values, instance_type=self.functions[call_name.lower()].get('member_of'))]
                                        except Exception:
                                            values = []
                                if not values:
                                    try:
                                        values = [self.expr_evaluator.eval_or_eval_array(
                                            str(init_expr), self.current_scope().get_evaluation_scope(), line_number)]
                                    except Exception:
                                        values = []
                                if values:
                                    val = values[-1]
                                    target_scope = self.current_scope().get_defining_scope(
                                        var) or self.current_scope()
                                    try:
                                        target_scope.update(
                                            var, val, line_number)
                                    except NameError:
                                        target_scope.define(
                                            var, val, type_name, constraints, is_uninitialized=False)
                        self.pop_scope()
                i += 1
            elif stripped_lower.startswith("if "):
                # Check if this is a complex IF statement (has ELSEIF/ELSE clauses)
                if_block = None
                for block in self.control_flow.if_blocks:
                    if block['start_line'] == line_number:
                        if_block = block
                        break

                if if_block and len(if_block.get('clauses', [])) > 1:
                    # Complex IF statement with ELSEIF/ELSE clauses
                    skip_lines = self.control_flow._process_if_statement_rich(
                        line, line_number, lines, i)
                else:
                    # Simple IF statement
                    skip_lines = self.control_flow._process_if_statement(
                        line, line_number, lines, i)
                i += skip_lines
                continue
            elif stripped_lower.startswith("let "):
                m = re.match(
                    r'^\s*let\s+(.+?)(?:\s+then\s*$|\s*$)', stripped, re.I)
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

                # First pass: Evaluate LET assignments, fail on forward references
                scope_dict = self.current_scope().get_evaluation_scope()
                let_var_order = [var.lower() for var, _, _, _ in var_list]
                for idx, (var, type_name, constraints, expr) in enumerate(var_list):
                    var_lower = var.lower()
                    remaining_let_vars = set(let_var_order[idx + 1:])
                    # Handle object field assignment on array elements (e.g., dck.cards(i).rank)
                    field_index_match = re.match(
                        r'^([\w_]+)\.([\w_]+)\(([^)]+)\)\.(\w+)$', var)
                    if field_index_match:
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
                        actual_array_field = public_keys.get(
                            array_field.lower(), array_field)
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
                        actual_field = element_keys.get(
                            field_name.lower(), field_name)
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
                        continue

                    # Handle array indexing assignment (e.g., D{i+1, 1}, D[A1], or D(k+1))
                    array_index_match = re.match(r'^([\w_]+)\{([^}]+)\}$', var)
                    cell_index_match = re.match(r'^([\w_]+)\[([^\]]+)\]$', var)
                    paren_index_match = re.match(r'^([\w_]+)\(([^)]+)\)$', var)
                    if array_index_match or cell_index_match or paren_index_match:
                        if array_index_match:
                            var_name, indices_str = array_index_match.groups()
                            indices = []
                            for index_expr in indices_str.split(','):
                                index_expr = index_expr.strip()
                                index_value = self.expr_evaluator.eval_expr(
                                    index_expr, scope_dict, line_number)
                                # Convert float indices to int (for array access)
                                if isinstance(index_value, float) and index_value.is_integer():
                                    index_value = int(index_value)
                                indices.append(
                                    index_value - 1 if isinstance(index_value, int) else index_value)
                        elif cell_index_match:
                            var_name, index_expr = cell_index_match.groups()
                            # Check if this is a numeric index (like D[1]) or a cell reference (like D[A1])
                            try:
                                # Try to evaluate as a numeric index first
                                index_value = self.expr_evaluator.eval_expr(
                                    index_expr, scope_dict, line_number)
                                if isinstance(index_value, float) and index_value.is_integer():
                                    index_value = int(index_value)
                                # Square bracket indexing is 1-based, so adjust to 0-based
                                indices = [index_value - 1]
                            except:
                                # If not a numeric index, treat as cell reference
                                indices = self.array_handler.cell_ref_to_indices(
                                    index_expr, line_number)
                        else:  # paren_index_match
                            var_name, index_expr = paren_index_match.groups()
                            # For parentheses indexing like D(k+1), evaluate the expression
                            index_value = self.expr_evaluator.eval_expr(
                                index_expr, scope_dict, line_number)
                            if isinstance(index_value, float) and index_value.is_integer():
                                index_value = int(index_value)
                            # Parentheses indexing is 1-based, so adjust to 0-based
                            indices = [index_value - 1]
                        value = self.expr_evaluator.eval_or_eval_array(
                            expr, scope_dict, line_number)
                        defining_scope = self.current_scope().get_defining_scope(var_name)
                        if defining_scope:
                            # Case-insensitive lookup for the array variable
                            actual_key = defining_scope._get_case_insensitive_key(
                                var_name, defining_scope.variables)
                            if actual_key:
                                arr = defining_scope.variables[actual_key]
                                constraints = defining_scope.constraints.get(
                                    actual_key, {})
                                if constraints and self._has_star_dim(constraints):
                                    if defining_scope.is_uninitialized(actual_key) or arr is None:
                                        raise ValueError(
                                            f"Array variable '{var_name}' with dim * must be initialized with PUSH or INIT before LET at line {line_number}")
                                # Check if array is wrapped in a dictionary (e.g., for grid dims)
                                if isinstance(arr, dict) and 'array' in arr:
                                    updated_array = self.array_handler.set_array_element(
                                        arr['array'], indices, value, line_number)
                                    arr['array'] = updated_array
                                    defining_scope.variables[actual_key] = arr
                                    scope_dict[actual_key] = arr
                                else:
                                    updated_array = self.array_handler.set_array_element(
                                        arr, indices, value, line_number)
                                    defining_scope.variables[actual_key] = updated_array
                                    scope_dict[actual_key] = updated_array
                                debug_array = updated_array.to_pylist() if hasattr(
                                    updated_array, 'to_pylist') else updated_array
                            else:
                                raise NameError(
                                    f"Array variable '{var_name}' not defined at line {line_number}")
                        else:
                            raise NameError(
                                f"Array variable '{var_name}' not defined at line {line_number}")
                        continue
                    defining_scope = self.current_scope().get_defining_scope(var)
                    if defining_scope:
                        # Preserve existing constraints if the variable already exists
                        if var in defining_scope.constraints and not constraints:
                            constraints = defining_scope.constraints[var]
                        if constraints:
                            defining_scope.constraints[var] = constraints
                        if expr is None and var in defining_scope.variables and defining_scope.variables[var] is not None:
                            continue
                    else:
                        if self.current_scope().is_shadowed(var):
                            print(
                                f"Warning: LET defines '{var}' which shadows a variable in an outer scope at line {line_number}")
                        self.current_scope().define(var, None, type_name, constraints, is_uninitialized=True)
                        self.current_scope().mark_implicit_let(var)
                    if expr is not None:
                        try:
                            evaluated_value = self.expr_evaluator.eval_or_eval_array(
                                expr, scope_dict, line_number)
                            if constraints.get('with'):
                                evaluated_value = self._apply_with_constraints(
                                    evaluated_value,
                                    constraints.get('with', {}),
                                    self.current_scope().get_full_scope(),
                                    line_number,
                                    type_name=type_name)
                            self.current_scope().update(var, evaluated_value, line_number)
                            scope_dict[var] = evaluated_value
                            pending_vars = [
                                pending_var for pending_var in list(self.pending_assignments.keys())
                                if not pending_var.startswith('__line_')
                            ]
                            for pending_var in pending_vars:
                                self._attempt_resolve_pending_var(
                                    pending_var, line_number)
                        except NameError as e:
                            missing = self.extract_missing_dependencies(e)
                            future_refs = {
                                dep for dep in missing if dep.lower() in remaining_let_vars}
                            if future_refs:
                                raise NameError(
                                    f"Forward LET reference to {future_refs} at line {line_number}")
                            if not missing:
                                raise
                            for dep in missing:
                                self.mark_dependency_missing(dep)
                            self.pending_assignments[var] = (
                                expr, line_number, set(missing), constraints)
                            continue
                    # Prevent double-processing of this variable in order tracking
                    remaining_let_vars.discard(var_lower)

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
                                    else:
                                        pass
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
                    block_pending = self.control_flow._process_block(
                        block_lines)
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
                                if op not in ('<', '>', '<=', '>='):
                                    continue
                                if isinstance(threshold, (list, dict, tuple)):
                                    continue
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
                                except (TypeError, ValueError):
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
            elif stripped_lower.startswith('for '):
                m = re.match(r'^\s*FOR\s+(.+?)(?:\s+DO\s*$|\s*$)',
                             stripped, re.I)
                if not m:
                    raise SyntaxError(
                        f"Invalid FOR syntax at line {line_number}")
                var_def = m.group(1).strip()
                has_block = stripped_lower.endswith('do')
                index_fallback = None
                index_match_line = re.search(
                    r'\bindex\s+([A-Za-z_][A-Za-z0-9_]*)\b', var_def, re.I)
                if index_match_line:
                    index_fallback = index_match_line.group(1)
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
                            elif clean.startswith('while ') and ' do' in clean:
                                depth += 1
                            elif clean.startswith('if ') and clean.endswith('then'):
                                depth += 1
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
                                r'^(.*)\s+index\s+([A-Za-z_][A-Za-z0-9_]*)\s*$', str(raw_init_expr), re.I)
                        except Exception:
                            init_match = None
                        if init_match and not index_name:
                            raw_init_expr = init_match.group(1)
                            index_name = init_match.group(2)
                        init_expr = raw_init_expr
                        try:
                            values = self._evaluate_push_expression(
                                str(init_expr), line_number)
                        except Exception:
                            try:
                                values = [self.expr_evaluator.eval_or_eval_array(
                                    str(init_expr), self.current_scope().get_evaluation_scope(), line_number)]
                            except Exception:
                                values = []
                        if not values:
                            call_match = re.match(
                                r'^([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)$', str(init_expr))
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
                                            call_name, arg_values, instance_type=self.functions[call_name.lower()].get('member_of'))]
                                except Exception:
                                    values = []
                        if not values:
                            try:
                                values = [self.expr_evaluator.eval_or_eval_array(
                                    str(init_expr), self.current_scope().get_evaluation_scope(), line_number)]
                            except Exception:
                                values = []
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
                        continue

                # If there is no DO block, handle INIT generator semantics immediately
                if not has_block:
                    init_handled = False
                    for var, type_name, constraints, expr in var_list:
                        if not constraints or 'init' not in constraints:
                            continue
                        init_expr = constraints.get('init')
                        try:
                            values = self._evaluate_push_expression(
                                str(init_expr), line_number)
                        except Exception:
                            try:
                                values = [self.expr_evaluator.eval_or_eval_array(
                                    str(init_expr), self.current_scope().get_evaluation_scope(), line_number)]
                            except Exception:
                                values = []
                        if not values:
                            call_match = re.match(
                                r'^([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)$', str(init_expr))
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
                                            call_name, arg_values, instance_type=self.functions[call_name.lower()].get('member_of'))]
                                except Exception:
                                    values = []
                        if not values:
                            try:
                                values = [self.expr_evaluator.eval_or_eval_array(
                                    str(init_expr), self.current_scope().get_evaluation_scope(), line_number)]
                            except Exception:
                                values = []
                        target_scope = self.current_scope().get_defining_scope(
                            var) or self.current_scope()
                        if values:
                            val = values[-1]
                            try:
                                target_scope.update(var, val, line_number)
                            except NameError:
                                target_scope.define(
                                    var, val, type_name, constraints, is_uninitialized=False)
                        init_handled = True
                    if init_handled:
                        i += 1
                        continue

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
                            public_fields = public_type_fields(
                                self.types_defined[type_name.lower()])
                            pa_type = pa.struct([(f, pa.string() if public_fields.get(
                                f) == 'text' else pa.float64()) for f in public_fields])
                            array = self.array_handler.create_array(
                                shape, None, pa_type, line_number)
                            for i in range(shape[0]):
                                for j in range(shape[1]):
                                    for k in range(shape[2]):
                                        obj = {f: with_constraints.get(
                                            f, None) for f in public_fields}
                                        obj['value'] = float(
                                            value) if value else 1.0
                                        array = self.array_handler.set_array_element(
                                            array, [i, j, k], obj, line_number)
                            self.current_scope().define(var, array, 'array', constraints, is_uninitialized=False)
                            i += 1
                            continue

            elif stripped.startswith(":") and "=" in stripped and not stripped_lower.startswith(("for ", "let ")):
                var_def, expr = map(str.strip, line[1:].split("=", 1))
                var, type_name, constraints, value = self._parse_variable_def(
                    var_def, line_number)
                deps = set(re.findall(r'\b[\w_]+\b', expr))
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
                i += 1
            else:
                if ':=' in line and '.grid' in line:
                    target, value = line.split(':=')
                    target = target.strip()
                    value = value.strip()
                    if re.match(r'^\[\^?[A-Za-z]+\d+\]$', target):
                        cell_ref = target[1:-1].strip()
                        try:
                            if cell_ref.startswith('^'):
                                # Handle array reference - remove ^ and validate the cell reference
                                array_cell_ref = cell_ref[1:].strip()
                                validate_cell_ref(array_cell_ref)
                                scope_value = self.current_scope().get_evaluation_scope()
                                is_grid_value = '.grid{' in value
                                evaluated_value = self.expr_evaluator.eval_or_eval_array(
                                    value, scope_value, line_number, is_grid_dim=is_grid_value)

                                # Check if this is a grid dictionary that should spill values
                                if isinstance(evaluated_value, dict):
                                    # Check if it's a grid dictionary with coordinate tuples as keys
                                    if all(isinstance(k, tuple) and len(k) == 2 for k in evaluated_value.keys()):
                                        # This is a grid dictionary, spill the values onto the actual grid
                                        for (row, col), val in evaluated_value.items():
                                            if isinstance(row, (int, float)) and isinstance(col, (int, float)):
                                                # Convert to cell reference format
                                                grid_row = int(row)
                                                grid_col = int(col)
                                                col_letter = num_to_col(
                                                    grid_col)
                                                cell_ref = f"{col_letter}{grid_row}"
                                                self.grid[cell_ref] = val
                                        # For [^A1] syntax, don't spill sequentially
                                        # Just set the target to 0 since grid{1,1} is undefined
                                        self.grid[array_cell_ref] = 0
                                    elif 'grid' in evaluated_value:
                                        # This is a type instance with a grid field
                                        grid_dict = evaluated_value['grid']
                                        if isinstance(grid_dict, dict):
                                            # Extract grid values and spill them onto the actual grid
                                            for (row, col), val in grid_dict.items():
                                                if isinstance(row, int) and isinstance(col, int):
                                                    # Convert to cell reference format
                                                    col_letter = num_to_col(
                                                        col)
                                                    cell_ref = f"{col_letter}{row}"
                                                    self.grid[cell_ref] = val
                                            # For [^A1] syntax, don't spill sequentially
                                            # Just set the target to 0 since grid{1,1} is undefined
                                            self.grid[array_cell_ref] = 0
                                        else:
                                            # Spill lists/arrays horizontally starting at the target
                                            self.array_handler._assign_horizontal_array(
                                                array_cell_ref, evaluated_value, value, line_number=line_number)
                                    else:
                                        # Spill lists/arrays horizontally starting at the target
                                        self.array_handler._assign_horizontal_array(
                                            array_cell_ref, evaluated_value, value, line_number=line_number)
                                else:
                                    # Spill lists/arrays horizontally starting at the target
                                    self.array_handler._assign_horizontal_array(
                                        array_cell_ref, evaluated_value, value, line_number=line_number)
                            else:
                                # Handle regular cell reference
                                validate_cell_ref(cell_ref)
                                scope_value = self.current_scope().get_evaluation_scope()
                                is_grid_value = '.grid{' in value
                                evaluated_value = self.expr_evaluator.eval_or_eval_array(
                                    value, scope_value, line_number, is_grid_dim=is_grid_value)
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
                    rhs_vars = None
                    # Special literal values like #N/A should not introduce dependencies
                    if rhs.lstrip('+-').startswith('#'):
                        rhs_vars = set()
                    if '$"' in rhs:
                        if rhs_vars is None:
                            rhs_vars = set()
                        placeholders = re.findall(r'\{\s*([^}]*?)\s*\}', rhs)
                        for ph in placeholders:
                            # Placeholders starting with '*' are formatting directives, not dependencies
                            clean_ph = ph.lstrip('{')
                            if clean_ph.strip().startswith('*'):
                                continue
                            rhs_vars.update(re.findall(
                                r'\b[\w_]+\b', clean_ph))
                    if rhs_vars is None:
                        rhs_vars = set(re.findall(
                            r'\b[\w_]+\b(?=\s*(?:[\[\{]|!\w+\s*\(|(?:\.\w+)?\s*$))', rhs))
                    # For field access like p.y, depend only on the base object 'p'
                    field_pairs = re.findall(r'\b([\w_]+)\.\s*([\w_]+)', rhs)
                    if field_pairs:
                        base_fields = {base for base, _ in field_pairs}
                        attr_fields = {attr for _, attr in field_pairs}
                        current = rhs_vars or set()
                        rhs_vars = (current | base_fields) - attr_fields
                    # Remove column letters extracted from cell references like [B{i}]
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
                            try:
                                self.array_handler.evaluate_line_with_assignment(
                                    line, line_number, scope.get_evaluation_scope())
                            except Exception as e:
                                missing = self.extract_missing_dependencies(e)
                                if missing:
                                    for dep in missing:
                                        self.mark_dependency_missing(dep)
                                    deps = (rhs_vars | target_vars) | missing
                                    self.pending_assignments[f"__line_{line_number}"] = (
                                        line, line_number, deps)
                                    i += 1
                                    continue
                                raise
                        else:
                            self.pending_assignments[f"__line_{line_number}"] = (
                                line, line_number, rhs_vars | target_vars)
                    i += 1
                elif re.match(r'^\[\s*\^?[A-Za-z]+\d+\s*\]\s*:\s*', line):
                    # Handle cell binding declarations and assignments
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
                    i += 1
                elif self._maybe_handle_subprocess_call(line, line_number):
                    i += 1
                else:
                    if not re.match(r'^\s*(input|define|output)\b', line, re.I):
                        pass
                    i += 1

        if guard_entries and not self._global_guards_pre_evaluated:
            guard_passed = self._evaluate_guard_conditions(guard_entries)
            if not guard_passed:
                self.global_guard_allows_execution = False
                self.grid.clear()
                self.output_values.clear()
                return self.grid

        self._resolve_pending_assignments()

        # Process any remaining deferred assignments
        self._process_deferred_assignments()

        # Print all output variables
        if not suppress_output:
            self._print_outputs()

        if return_output:
            return self.output_values
        return self.grid

    def _evaluate_push_expression(self, value_expr, line_number):
        """Evaluate a PUSH expression, expanding generator outputs into a sequence."""
        base_scope = self.current_scope().get_evaluation_scope()
        func_defs = getattr(self, 'functions', {}) or {}
        subprocess_defs = getattr(self, 'subprocesses', {}) or {}

        call_pattern = re.compile(r'([A-Za-z_][A-Za-z0-9_.]*)\s*\(')

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
                value_expr, base_scope, line_number)]

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
                    vals = self._evaluate_push_expression(a, line_number)
                except Exception:
                    vals = [self.expr_evaluator.eval_or_eval_array(
                        a, base_scope, line_number)]
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
                            call['name'], call_args, instance_type=member_of, collect_all=True) or {}
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
                if call['name'].lower() in func_defs:
                    outputs_map = self.call_function(
                        call['name'], eval_args, instance_type=member_of, collect_all=True) or {}
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

    def _handle_push_assignment(self, target, value_expr, line_number):
        if target.startswith('['):
            assignment_line = f"{target} := {value_expr}"
            self.array_handler.evaluate_line_with_assignment(
                assignment_line, line_number, self.current_scope().get_evaluation_scope())
            return
        if re.search(r'[\(\{]', target):
            self._assign_indexed_target(target, value_expr, line_number)
            return
        self._process_push_call(f"{target}.push({value_expr})", line_number)

    def _process_push_call(self, line, line_number):
        """Process a .push() method call"""

        # Handle multi-line expressions by removing newlines and extra whitespace
        clean_line = line.replace('\n', ' ').replace('  ', ' ').strip()

        # Handle .push() method calls (e.g., low.push(1), high.push(rows(haystack)))
        m = re.match(
            r'^\s*(\[[^\]]+\]|[\w_]+(?:\.[\w_]+)?(?:\{[^}]+\}|\[[^\]]+\]|\([^)]*\))?)\.push\s*\(\s*(.+?)\s*\)\s*$', clean_line, re.I)
        if m:
            var_name, value_expr = m.groups()
            var_name = var_name.strip()
            value_expr = value_expr.strip()


            # If pushing directly into a cell reference, translate to an assignment
            if var_name.startswith('[') and var_name.endswith(']'):
                assignment_line = f"{var_name} := {value_expr}"
                try:
                    self.array_handler.evaluate_line_with_assignment(
                        assignment_line, line_number, self.current_scope().get_evaluation_scope())
                    return
                except Exception as e:
                    raise

            # Handle object field pushes (e.g., Obj.field.push(val))
            field_match = re.match(r'^([\w_]+)\.([\w_]+)$', var_name)

            # Evaluate the expression to get the value(s)
            try:
                values = self._evaluate_push_expression(
                    value_expr, line_number)
                # Use the global scope to ensure we can access updated variable values
                global_scope = self.get_global_scope()

                indexed_var, indices = self._parse_indexed_target(
                    var_name, self.current_scope().get_evaluation_scope(), line_number)
                if indexed_var is not None:
                    defining_scope = self.current_scope().get_defining_scope(indexed_var)
                    if not defining_scope:
                        raise NameError(
                            f"Array variable '{indexed_var}' not defined at line {line_number}")
                    actual_key = defining_scope._get_case_insensitive_key(
                        indexed_var, defining_scope.variables) or indexed_var
                    constraints = defining_scope.constraints.get(
                        actual_key, {})
                    if not constraints or not constraints.get('dim'):
                        raise ValueError(
                            f"Array variable '{indexed_var}' has no dimensions defined at line {line_number}")
                    for value in values:
                        arr = defining_scope.variables.get(actual_key)
                        if isinstance(arr, dict) and 'array' in arr:
                            arr_value = arr['array']
                            if self._has_star_dim(constraints) and isinstance(arr_value, pa.Array):
                                arr_value = arr_value.to_pylist()
                            updated_array = self.array_handler.set_array_element(
                                arr_value, indices, value, line_number)
                            arr['array'] = updated_array
                            defining_scope.variables[actual_key] = arr
                        else:
                            arr_value = arr
                            if self._has_star_dim(constraints) and isinstance(arr_value, pa.Array):
                                arr_value = arr_value.to_pylist()
                            updated_array = self.array_handler.set_array_element(
                                arr_value, indices, value, line_number)
                            defining_scope.variables[actual_key] = updated_array
                        defining_scope.uninitialized.discard(actual_key)
                    return

                for value in values:

                    if field_match:
                        base_name, field_name = field_match.groups()
                        base_value = self.current_scope().get(base_name)
                        if not isinstance(base_value, dict):
                            raise ValueError(
                                f"'{base_name}' is not an object at line {line_number}")
                        public_keys = {k.lower(): k for k in base_value.keys()
                                       if not str(k).startswith('_')}
                        actual_field = public_keys.get(
                            field_name.lower(), field_name)
                        if hasattr(self, '_is_hidden_field') and self._is_hidden_field(base_value, actual_field):
                            if not getattr(self, '_allow_hidden_field_access', False):
                                raise PermissionError(
                                    f"Hidden field '{field_name}' is not accessible at line {line_number}")
                        immutable_fields = {f.lower()
                                            for f in base_value.get('_immutable_fields', set())}
                        if actual_field.lower() in immutable_fields:
                            raise ValueError(
                                f"Field '{field_name}' of '{base_name}' is immutable at line {line_number}")
                        existing = base_value.get(actual_field)
                        if isinstance(existing, list):
                            existing.append(value)
                        else:
                            base_value[actual_field] = value
                        self.current_scope().update(
                            base_name, base_value, line_number)
                        continue

                    # Update the variable with the new value
                    defining_scope = self.current_scope().get_defining_scope(var_name)
                    if defining_scope:
                        defining_scope.update(var_name, value, line_number)
                    else:
                        self.current_scope().define(
                            var_name, value, 'number', {}, is_uninitialized=False)

                    collecting_via_compiler = hasattr(
                        self, 'compiler') and self.compiler is not None
                    if (var_name.lower() in global_scope.output_variables) and not collecting_via_compiler:
                        self.output_values.setdefault(
                            var_name.lower(), []).append(value)

            except Exception as e:
                raise ValueError(
                    f"Failed to evaluate .push() expression at line {line_number}: {e}")
        else:
            raise SyntaxError(f"Invalid .push() syntax at line {line_number}")

    def _assign_indexed_target(self, target, value_expr, line_number):
        paren_match = re.match(r'^([\w_]+)\s*\(([^)]+)\)$', target)
        brace_match = re.match(r'^([\w_]+)\s*\{([^}]+)\}$', target)
        if not (paren_match or brace_match):
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
        actual_key = defining_scope._get_case_insensitive_key(
            var_name, defining_scope.variables) or var_name
        constraints = defining_scope.constraints.get(actual_key, {})
        if not constraints or not constraints.get('dim'):
            raise ValueError(
                f"Array variable '{var_name}' has no dimensions defined at line {line_number}")
        arr = defining_scope.variables.get(actual_key)
        if isinstance(arr, dict) and 'array' in arr:
            arr_value = arr['array']
            if self._has_star_dim(constraints) and isinstance(arr_value, pa.Array):
                arr_value = arr_value.to_pylist()
            updated_array = self.array_handler.set_array_element(
                arr_value, indices, value, line_number)
            arr['array'] = updated_array
            defining_scope.variables[actual_key] = arr
        else:
            arr_value = arr
            if self._has_star_dim(constraints) and isinstance(arr_value, pa.Array):
                arr_value = arr_value.to_pylist()
            updated_array = self.array_handler.set_array_element(
                arr_value, indices, value, line_number)
            defining_scope.variables[actual_key] = updated_array
        defining_scope.uninitialized.discard(actual_key)
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
                if output_var in output_values:
                    pass
                condition_result = output_var in output_values and output_values[output_var]
                if condition_result:
                    # Print all collected values for this output variable
                    for value in output_values[output_var]:
                        if output_var.lower() == 'output':
                            print(value)
                        else:
                            print(f"{output_var}: {value}")
                else:
                    # No fallback - if no values were pushed, don't print anything
                    pass
        except Exception as e:
            import traceback
            traceback.print_exc()

    def _materialize_inits(self):
        """Evaluate INIT constraints for uninitialized vars when dependencies are satisfied."""
        try:
            scope = self.current_scope()
        except Exception:
            return
        seen_scopes = set()
        while scope and id(scope) not in seen_scopes:
            seen_scopes.add(id(scope))
            for var_name, constraints in list(scope.constraints.items()):
                if not constraints or 'init' not in constraints:
                    continue
                if not scope.is_uninitialized(var_name):
                    continue
                init_expr = constraints.get('init')
                deps = self._extract_dependencies_from_expression(
                    str(init_expr))
                if any(self.has_unresolved_dependency(dep, scope=scope) for dep in deps):
                    continue
                try:
                    val = self.expr_evaluator.eval_or_eval_array(
                        str(init_expr), scope.get_full_scope(), None)
                    try:
                        val = copy.deepcopy(val)
                    except Exception:
                        pass
                    scope.update(var_name, val)
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
                    if any(size_spec is None for _, size_spec in resolved_dims):
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
                    pa_type = pa.float64() if var_type in (
                        'number', 'array') else pa.string()
                    default_value = 0 if var_type in (
                        'number', 'array') else ''
                    value = self.array_handler.create_array(
                        shape, default_value, pa_type, None)
                    self.dimensions[var_name] = resolved_dims
                    if actual_key in scope.constraints:
                        scope.constraints[actual_key]['dim'] = resolved_dims
                    elif var_name in scope.constraints:
                        scope.constraints[var_name]['dim'] = resolved_dims
                    scope.update(var_name, value)
                except Exception as exc:
                    pass
            scope = scope.parent if getattr(
                scope, 'parent', None) and not scope.is_private else None

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
                    self.has_unresolved_dependency(dep, scope=scope)
                    for dep in deps)
                if unresolved:
                    continue
                # Locate the index of the IF line in the cached preprocessed lines
                line_index = next(
                    (idx for idx, (_, ln) in enumerate(self.control_flow._preprocessed_lines)
                     if ln == line_number),
                    None)
                if line_index is None:
                    continue
                if_block = next(
                    (block for block in self.control_flow.if_blocks if block['start_line'] == line_number),
                    None)
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
                    self.has_unresolved_dependency(dep, scope=scope)
                    for dep in deps)
                if not unresolved:
                    # Process the assignment
                    self.array_handler.evaluate_line_with_assignment(
                        line_content, line_number, self.current_scope().get_evaluation_scope())
                    # Remove from pending assignments
                    if f"__line_{line_number}" in self.pending_assignments:
                        del self.pending_assignments[f"__line_{line_number}"]
                else:
                    pass
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
