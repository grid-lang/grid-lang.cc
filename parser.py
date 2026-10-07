"""
Parsing and preprocessing functionality for GridLang compiler.
Handles code preprocessing, variable parsing, and declaration processing.
"""
import re
from type_processor import GridLangTypeProcessor
from expression import ExpressionEvaluator
from grid_lang_common import _STATEMENT_KEYWORDS, _first_keyword


class GridLangParser:
    """Handles parsing and preprocessing of GridLang code."""

    def __init__(self, compiler=None):
        self.compiler = compiler

    @staticmethod
    def _literal_token_value(token):
        """Type an unquoted set-literal token the way the evaluator would.

        A set constraint is written as source text, so `{1,2}` arrives as the
        strings '1' and '2'. Comparing those against a number never matches,
        so numeric and boolean tokens are converted here and everything else
        is left as text.
        """
        if re.match(r'^[+-]?\d+$', token):
            return int(token)
        if re.match(r'^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$', token):
            return float(token)
        if token.lower() in ('true', 'false'):
            return token.lower() == 'true'
        return token

    def _parse_variable_def(self, def_str, line_number):
        # Allow leading ':' in global declarations
        def_str = def_str.lstrip(':').strip()
        constraints = {}
        expr = None
        type_name = None
        unit = None
        dims = None
        type_union = []

        direct_match = self._match_direct_assignment_patterns(
            def_str,
            line_number,
            constraints,
        )
        if direct_match is not None:
            return direct_match

        # Extract 'with' clause early to preserve parentheses (the keyword
        # splitter would otherwise break '=' inside the parentheses). For
        # 'new Type with (...)' constructions the WITH is folded back into
        # the expression so the evaluator applies construction, the WITH
        # values and any trailing builder chain in the correct order.
        with_content_str = None
        is_new_with = False
        with_match = self._find_with_clause_balanced(def_str)
        if with_match:
            span_start, open_idx, close_idx = with_match
            with_content_str = def_str[open_idx:close_idx + 1]
            is_new_with = bool(re.match(
                r'^.*\bnew\s+[\w.]+\s*$', def_str[:span_start], re.I | re.S))
            def_str = def_str[:span_start] + def_str[close_idx + 1:]

        # Split on keywords (as, of, dim, in, <=, >=, <, >, =) while ignoring quoted text
        parts = self._split_on_keywords(def_str)

        # Check for comma-separated variable lists (e.g., "low, high as number")
        if ',' in parts[0] and '{' not in parts[0] and '(' not in parts[0]:
            raw_names = [v.strip() for v in parts[0].split(',')]
            # Strip leading input/output keywords from each entry
            var_names = [re.sub(r'^(input|output)\s+', '', v, flags=re.I).strip()
                         for v in raw_names if v.strip()]
            # For now, return the first variable and let the caller handle the list
            # This will be expanded in the FOR loop processing
            var = var_names[0]
            # Store the full list in constraints for later processing
            constraints['var_list'] = var_names
        else:
            var = re.sub(r'^(input|output)\s+', '', parts[0], flags=re.I).strip()
        first_word = var.split()[0] if var.split() else ''
        if _first_keyword(first_word) in ('let','for','if','when','declare','input','output'):
            raise SyntaxError(
                f"'{first_word}' is an instruction and must be at the start "
                f"of a line, or after 'Then', 'Else', 'Do' or ':' at line "
                f"{line_number}")
        i = 1
        negated = False
        while i < len(parts):
            keyword = parts[i].lower()
            next_part = parts[i + 1].strip() if i + 1 < len(parts) else ''
            if keyword == 'and':
                i += 1
                continue
            if keyword == 'not':
                if next_part.lower() in ('null'):
                    constraints['not_null'] = True
                    i += 2
                    continue
                negated = True
                i += 1
                continue
            if keyword == 'null':
                constraints['null'] = True
                i += 1
                continue
            if keyword == 'as':
                if negated:
                    constraints['not_type'] = next_part.lower()
                    negated = False
                else:
                    # 'single' before the type is a shortcut for "dim none":
                    # the value must be a scalar (rank 0), never an array.
                    if re.match(r'^single\s+\S', next_part, re.I):
                        next_part = next_part[len('single'):].lstrip()
                        constraints['dim'] = '{}'
                    cleaned_part = re.sub(r'\s+or\s*$', '', next_part, flags=re.I).strip()
                    union_parts = [
                        part.strip().lower()
                        for part in re.split(r'\s+or\s+', cleaned_part, flags=re.I)
                        if part.strip()
                    ]
                    for _part in union_parts or [next_part.strip().lower()]:
                        # A type name is a single identifier token, optionally
                        # followed by a `Use <version>` clause (module imports,
                        # e.g. 'as ModuleVersion Use M.v'). Anything else is an
                        # unexpected token and rejects the declaration.
                        if not re.match(
                                r'^\$?[a-zA-Z_][\w.!*]*'
                                r'(?:\s+use\s+[\w.\-+]+|\s+key)?$',
                                _part, re.I):
                            raise SyntaxError(
                                f"Invalid type declaration for '{var}': "
                                f"unexpected token '{_part}' after type name "
                                f"at line {line_number}")
                    if union_parts:
                        type_union.extend(union_parts)
                    else:
                        type_name = next_part.lower()
            elif keyword == 'of':
                if negated:
                    constraints['not_unit'] = next_part
                    negated = False
                else:
                    unit = next_part
                    constraints['unit'] = unit
            elif keyword == 'dim':
                dims = next_part
                constraints['dim'] = dims
            elif keyword == 'in':
                # Handle both set-based and range-based constraints
                if next_part.startswith('{') and next_part.endswith('}'):
                    values = []
                    for v in next_part[1:-1].split(','):
                        v_clean = v.strip()
                        if not v_clean:
                            continue
                        if (v_clean.startswith('"') and v_clean.endswith('"')) or (
                                v_clean.startswith("'") and v_clean.endswith("'")):
                            # Quoted: text by definition.
                            values.append(v_clean[1:-1])
                            continue
                        values.append(self._literal_token_value(v_clean))
                    constraints['in'] = values
                elif ' to ' in next_part:
                    range_parts = next_part.split(' to ', 1)
                    if len(range_parts) != 2:
                        raise SyntaxError(f"Invalid range syntax: '{next_part}' at line {line_number}")
                    start_expr = range_parts[0].strip()
                    end_expr = range_parts[1].strip()
                    step_expr = None
                    if ' step ' in end_expr:
                        end_expr, step_expr = [p.strip() for p in end_expr.split(' step ', 1)]
                    range_constraint = {'start': start_expr, 'end': end_expr}
                    if step_expr:
                        range_constraint['step'] = step_expr
                    constraints['range'] = range_constraint
                else:
                    # Allow variable/expression-based in constraints (e.g., "in Options")
                    if not next_part:
                        raise SyntaxError(f"Invalid 'in' constraint syntax: '{next_part}' at line {line_number}")
                    constraints['in'] = next_part
            elif keyword in ('<=', '>=', '<', '>', '<>'):
                self._check_comparison_series(parts, i, line_number)
                if negated:
                    constraints[f'not_{keyword}'] = next_part
                    negated = False
                else:
                    constraints[keyword] = next_part
            elif keyword == 'init':
                constraints['init'] = next_part
                # Source line of the declaration: lazy INIT materialization
                # (Scope.get) needs it to report accurate errors.
                constraints['init_line'] = line_number
            elif keyword == 'index':
                constraints['index'] = next_part
            elif keyword == 'or':
                # 'or = <expr>' gives a null-default: null (#N/A) values read
                # as the evaluated default on access.
                if i + 1 < len(parts) and parts[i + 1] == '=':
                    constraints['default'] = (
                        parts[i + 2].strip() if i + 2 < len(parts) else '')
                    i += 3
                    continue
                i += 1
                continue
            elif keyword == '=':
                self._check_comparison_series(parts, i, line_number)
                if next_part.startswith('{'):
                    if '|' in next_part:
                        expr = next_part
                    elif ';' in next_part:
                        matrices = [m.strip()[1:-1] for m in next_part.split('|')]
                        matrix_data = []
                        for matrix in matrices:
                            rows = [r.strip().split(',') for r in matrix.split(';')]
                            parsed_rows = []
                            for row in rows:
                                parsed_row = []
                                for v in row:
                                    v_clean = v.strip()
                                    if (v_clean.startswith('"') and v_clean.endswith('"')) or (
                                            v_clean.startswith("'") and v_clean.endswith("'")):
                                        parsed_row.append(v_clean)
                                    else:
                                        try:
                                            parsed_row.append(float(v_clean))
                                        except ValueError:
                                            parsed_row.append(v_clean)
                                parsed_rows.append(parsed_row)
                            matrix_data.append(parsed_rows)
                        # A single ';'-separated literal is one matrix; the
                        # '|' case is handled above, so no multi-matrix form
                        # reaches this branch.
                        expr = matrix_data[0] if matrix_data else []
                    else:
                        values_str = next_part[1:-1].split(',')
                        values = []
                        for v in values_str:
                            v_clean = v.strip()
                            if (v_clean.startswith('"') and v_clean.endswith('"')) or (v_clean.startswith("'") and v_clean.endswith("'")):
                                val = v_clean
                            else:
                                try:
                                    val = float(v_clean)
                                except ValueError:
                                    val = v_clean
                            values.append(val)
                        expr = values
                else:
                    expr = next_part
            i += 2

        if is_new_with and expr is not None:
            # Fold the WITH clause back into the expression so the
            # evaluator constructs the object, applies the WITH values and
            # then any builder chain, in that order.
            base_part = str(expr).strip()
            chain_part = ''
            idx = base_part.find('->')
            if idx != -1:
                chain_part = base_part[idx:].strip()
                base_part = base_part[:idx].strip()
            expr = f"{base_part} with {with_content_str}"
            if chain_part:
                expr += ' ' + chain_part
        else:
            self._apply_with_clause(
                with_content_str,
                constraints,
                line_number,
            )

        self._apply_dimension_constraints(
            dims,
            constraints,
            line_number,
        )
        if type_union:
            unique_types = list(dict.fromkeys(type_union))
            if len(unique_types) == 1:
                type_name = unique_types[0]
            else:
                constraints['type_union'] = unique_types
                type_name = None

        self._merge_custom_type_constraints(type_name, constraints)

        return var, type_name, constraints, expr

    def _find_with_clause_balanced(self, def_str):
        """Locate a top-level WITH ( ... ) clause spanning balanced parens.

        Returns (span_start, open_idx, close_idx) covering the whole
        ' with (...)' text, or None. Stops at the first balanced close so
        trailing builder chains are never swallowed.
        """
        m = re.search(r'\s+with\s+(?=\()', def_str, re.I)
        if not m:
            return None
        start = m.end()
        depth = 0
        in_text = None
        for i in range(start, len(def_str)):
            ch = def_str[i]
            if in_text:
                if ch == in_text and (i == 0 or def_str[i - 1] != '\\'):
                    in_text = None
                continue
            if ch in ('"', "'"):
                in_text = ch
                continue
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return (m.start(), start, i)
        return None

    def _check_comparison_series(self, parts, i, line_number):
        """Require 'and' between comparison operators used in series.

        A comparison operator at parts[i] is followed by its value at
        parts[i + 1]. If another comparison operator directly follows the
        value, the series lacks the required 'and' separator.
        """
        comparison_ops = {'<=', '>=', '<>', '<', '>', '='}
        if (i + 2 < len(parts)
                and parts[i + 2].lower() in comparison_ops):
            raise SyntaxError(
                f"Comparison operators in a series must be separated by "
                f"'and' at line {line_number}")

    def _match_direct_assignment_patterns(self, def_str, line_number, constraints):
        field_match = re.match(r'^([\w_]+)\.(\w+)\s*=\s*(.+)$', def_str)
        if field_match:
            var, field, expr = field_match.groups()
            constraints['constant'] = expr
            return f"{var}.{field}", None, constraints, expr

        array_index_match = re.match(
            r'^([\w_]+)\{([^}]+)\}\s*=\s*(.+)$', def_str)
        if array_index_match:
            var_name, indices_str, expr = array_index_match.groups()
            var = f"{var_name}{{{indices_str}}}"
            return var, None, constraints, expr

        paren_index_match = re.match(
            r'^([\w_]+)\(([^)]+)\)\s*=\s*(.+)$', def_str)
        if paren_index_match:
            var_name, index_expr, expr = paren_index_match.groups()
            var = f"{var_name}({index_expr})"
            return var, None, constraints, expr

        return None

    @staticmethod
    def _split_with_assignment(part):
        """Split ``name = expr`` at the first top-level ``=`` that assigns.

        Comparison operators are skipped, so ``y >= 1.5`` is not read as an
        assignment to the field ``y >``.
        """
        in_quotes = None
        paren = bracket = brace = 0
        idx = 0
        while idx < len(part):
            char = part[idx]
            if in_quotes is not None:
                if char == '\\':
                    idx += 2
                    continue
                if char == in_quotes:
                    in_quotes = None
            elif char in '"\'':
                in_quotes = char
            elif char in '([{':
                if char == '(':
                    paren += 1
                elif char == '[':
                    bracket += 1
                else:
                    brace += 1
            elif char in ')]}':
                if char == ')':
                    paren -= 1
                elif char == ']':
                    bracket -= 1
                else:
                    brace -= 1
            elif char == '=' and paren == bracket == brace == 0:
                prev = part[idx - 1] if idx else ''
                nxt = part[idx + 1] if idx + 1 < len(part) else ''
                if prev in '<>!' or nxt == '=':
                    idx += 1
                    continue
                return part[:idx].strip(), part[idx + 1:].strip()
            idx += 1
        return None

    def _parse_with_grid_spec(self, part, line_number):
        """Parse a ``with (grid ...)`` entry into the grid dimension constraint.

        The entry is read with the shared declaration parser, so ``as``,
        ``not null``, ``dim``, ``or =`` and the data expression all follow the
        same grammar as a ``Let`` declaration.  The result keeps the shape the
        array layer already consumes: ``dims`` plus the optional ``grid_type``,
        ``not_null``, ``matrix_data``, ``value``, ``data_var`` and ``default``.
        """
        parsed = self._parse_with_field_spec(part, line_number)
        if parsed is None:
            raise SyntaxError(
                f"Expected 'dim' after grid definition in 'with ({part})' "
                f"at line {line_number}")
        _, grid_expr, cons = parsed
        grid_type = cons.pop('type', None)
        not_null = bool(cons.pop('not_null', False))
        init_source = cons.pop('init', None)
        raw_dims = cons.pop('dim', None)
        if not raw_dims:
            raise SyntaxError(
                f"Expected 'dim' after grid definition in 'with ({part})' "
                f"at line {line_number}")
        dims = []
        for _name, size in raw_dims:
            if isinstance(size, str):
                size = None if size.strip() == '*' else size.strip()
            dims.append((1, size))
        dim_constraint = {'dims': dims}
        if grid_type:
            dim_constraint['grid_type'] = grid_type
        if not_null:
            dim_constraint['not_null'] = True
        if init_source is not None:
            dim_constraint['data_var'] = init_source
        if grid_expr is not None:
            if re.match(r'^{.+?}(?:\s*\|\s*{.+?})*$', grid_expr):
                matrices = [m.strip()[1:-1] for m in grid_expr.split('|')]
                matrix_data = []
                for matrix in matrices:
                    rows = [r.strip().split(',') for r in matrix.split(';')]
                    matrix_data.append(
                        [[float(v.strip()) for v in row] for row in rows])
                dim_constraint['matrix_data'] = (
                    matrix_data if len(matrix_data) > 1 else matrix_data)
            elif re.match(r'^[\d.]+$', grid_expr):
                dim_constraint['value'] = float(grid_expr)
            elif re.match(r'^\w+$', grid_expr):
                dim_constraint['data_var'] = grid_expr
            else:
                raise SyntaxError(
                    f"Invalid grid data: '{grid_expr}' at line {line_number}")
        default = cons.pop('default', None)
        if default is not None:
            dim_constraint['default'] = default
        return dim_constraint

    def _parse_with_field_spec(self, part, line_number):
        """Parse one ``with (...)`` entry into (field, value, constraint spec).

        The entry is a variable declaration without its leading keyword, so the
        regular declaration parser is reused verbatim: ``y = 2``, ``y < 2``,
        ``y as number``, ``y in {1, 2}``, ``y of m`` and ``y not null`` all
        produce the same constraint dict a ``Let``/type-field declaration
        would.  ``value`` is the unevaluated ``=`` expression when present, so
        consumers keep evaluating it themselves exactly as before.
        """
        try:
            field, field_type, field_cons, field_expr = self._parse_variable_def(
                part, line_number)
        except SyntaxError:
            raise
        except Exception:
            return None
        field = (field or '').strip()
        if not field:
            return None
        spec = dict(field_cons)
        if field_type:
            spec['type'] = field_type
        value = None
        if field_expr is not None:
            if isinstance(field_expr, str):
                value = field_expr.strip()
            else:
                # A set literal is handed back already parsed; consumers expect
                # the source text, so recover it from the entry itself.
                split = self._split_with_assignment(part)
                if split is not None:
                    value = split[1]
        if value is None and not spec:
            return None
        return field, value, spec

    def _apply_with_clause(self, with_content_str, constraints, line_number):
        if not with_content_str:
            return

        with_inner_content = with_content_str[1:-1].strip()
        with_constraints = {}
        with_specs = {}
        dim_constraint = None
        wc_parts = []
        current = ""
        in_quotes = False
        paren_level, brace_level = 0, 0
        for char in with_inner_content + ',':
            if char == '"' and (len(current) == 0 or current[-1] != '\\'):
                in_quotes = not in_quotes
            elif not in_quotes:
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                elif char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
            if char == ',' and not in_quotes and paren_level == 0 and brace_level == 0:
                if current.strip():
                    wc_parts.append(current.strip())
                current = ""
            else:
                current += char

        for wc in wc_parts:
            wc = wc.strip()
            if re.match(r'grid\s+', wc, re.I):
                if dim_constraint is not None:
                    raise SyntaxError(
                        f"Multiple grid DIM statements not allowed in with clause at line {line_number}")
                dim_constraint = self._parse_with_grid_spec(wc, line_number)
            else:
                spec = self._parse_with_field_spec(wc, line_number)
                if spec is None:
                    raise SyntaxError(
                        f"Ambiguous 'with ({wc})' at line {line_number}: expected "
                        "'<field> <constraint>', e.g. 'y = 2.0' or 'y < 2'")
                field, value, field_spec = spec
                if value is not None:
                    if value.startswith('"') and value.endswith('"'):
                        # Keep the quoted form: a quoted value is a string
                        # literal, distinct from a bare identifier reference.
                        # Consumers finalize it (strip quotes) when applied.
                        with_constraints[field] = value
                    else:
                        try:
                            with_constraints[field] = self.compiler.expr_evaluator.eval_expr(
                                value, self.compiler.current_scope().get_full_scope(),
                                line_number)
                        except Exception:
                            with_constraints[field] = value.strip('"')
                    field_spec['constant'] = with_constraints[field]
                with_specs[field] = field_spec

        if with_constraints:
            constraints['with'] = with_constraints
        if with_specs:
            constraints['with_specs'] = with_specs
        if dim_constraint:
            constraints['dim'] = dim_constraint

    def _apply_dimension_constraints(self, dims, constraints, line_number):
        if dims:
            dim_expr = dims.strip()
            dim_expr = re.split(
                r'\s+or\s+',
                dim_expr,
                maxsplit=1,
                flags=re.I,
            )[0].strip()
            if dim_expr == '{}' or dim_expr.lower() == 'none':
                constraints['dim'] = '{}'
                return

            dim_parts = dim_expr[1:-1].split(',') if dim_expr.startswith('{') else [dim_expr]
            dims_list = []
            for part in dim_parts:
                part = part.strip()
                if not part:
                    continue
                if ':' in part:
                    name, size = map(str.strip, part.split(':'))
                    size_spec = self._parse_dim_size(size, line_number)
                    dims_list.append((name, size_spec))
                else:
                    size_spec = self._parse_dim_size(part, line_number)
                    dims_list.append((None, size_spec))
            constraints['dim'] = dims_list
            return

        if constraints.get('with', {}).get('dim'):
            dim_str = constraints['with']['dim']
            m = re.match(r'^\s*grid\s+dim\s*\{([^}]+)\}\s*$', dim_str, re.I)
            if not m:
                raise SyntaxError(
                    f"Invalid dimension syntax: '{dim_str}' at line {line_number}")
            dim_content = m.group(1).strip()
            dim_parts = [p.strip()
                         for p in dim_content.split(',') if p.strip()]
            dims_list = []
            for part in dim_parts:
                size_spec = self._parse_dim_size(part, line_number)
                dims_list.append((None, size_spec))
            constraints['dim'] = dims_list

    def _merge_custom_type_constraints(self, type_name, constraints):
        if not (type_name and self.compiler and hasattr(self.compiler, 'types_defined')):
            return
        type_def = self.compiler.types_defined.get(type_name.lower())
        if isinstance(type_def, dict):
            type_constraints = type_def.get('_constraints', {}) or {}
            for key, val in type_constraints.items():
                if key == 'dim' and isinstance(val, dict) and 'dims' in val:
                    # Type-level grid dims belong to the instance's 'grid'
                    # field, not to the variable itself, so never merge them
                    # onto the variable (that would reshape the instance).
                    continue
                constraints.setdefault(key, val)
            base_type = type_def.get('_base_type')
            if base_type and 'type' not in constraints:
                constraints['type'] = base_type

    def _split_on_keywords(self, text):
        """Split a variable definition on keywords, skipping quoted sections."""
        keywords = ['<=', '>=', '<>', '<', '>', '=', 'as', 'of', 'dim', 'in', 'init', 'index', 'and', 'not', 'null', 'or']
        # Soft keywords (constraint clauses and logical connectors) only split
        # the left-hand side of a declaration. Once the first '=' has been seen,
        # the remainder is an arbitrary RHS expression and must be preserved
        # verbatim (e.g. 'x of m = 2.54 of cm' keeps the RHS 'of cm' intact).
        soft_keywords = {'as', 'of', 'dim', 'init', 'index', 'and', 'not', 'null', 'in'}
        parts = []
        current = ""
        in_quote = False
        quote_char = None
        brace_depth = 0
        truth_open = False
        seen_equals = False
        seen_init = False
        i = 0
        lower_text = text.lower()

        while i < len(text):
            ch = text[i]
            if ch in ('"', "'"):
                if not in_quote:
                    in_quote = True
                    quote_char = ch
                elif quote_char == ch:
                    in_quote = False
                    quote_char = None
                current += ch
                i += 1
                continue

            if not in_quote and ch == '{':
                brace_depth += 1
                current += ch
                i += 1
                continue
            if not in_quote and ch == '}' and brace_depth > 0:
                brace_depth -= 1
                current += ch
                i += 1
                continue

            matched = None
            break_keyword_loop = False
            if not in_quote and brace_depth == 0 and not truth_open:
                # Builder chains use '->': keep the arrow atomic so the '>'
                # is not misread as the start of a comparison constraint.
                if ch == '-' and i + 1 < len(text) and text[i + 1] == '>':
                    current += '->'
                    i += 2
                    continue
                for kw in keywords:
                    if lower_text.startswith(kw, i):
                        # For word keywords, ensure boundaries
                        if kw.isalpha():
                            prev = text[i - 1] if i > 0 else ' '
                            nxt = text[i + len(kw)
                                       ] if i + len(kw) < len(text) else ' '
                            if prev.isalnum() or prev == '_' or nxt.isalnum() or nxt == '_':
                                continue
                        if (seen_equals or seen_init) and kw in soft_keywords:
                            # Inside the RHS or after init, constraint clauses / connectors
                            # are ordinary tokens (e.g. '500 of in' or '5 of in' after init): consume
                            # the keyword as part of the current token.
                            current += kw
                            i += len(kw)
                            break_keyword_loop = True
                            break
                        if kw == 'or':
                            # 'or' only separates the 'or = <default>'
                            # null-default clause; 'or' used as a union
                            # (type/dim) stays part of the current token.
                            j = i + len(kw)
                            while j < len(text) and text[j] == ' ':
                                j += 1
                            if j >= len(text) or text[j] != '=':
                                continue
                            matched = kw
                            break
                        matched = kw
                        break

            if break_keyword_loop:
                continue

            if matched:
                if current.strip():
                    parts.append(current.strip())
                parts.append(matched)
                current = ""
                truth_open = False
                if matched == '=':
                    seen_equals = True
                elif matched == 'init':
                    seen_init = True
                i += len(matched)
                # Skip following whitespace to mimic regex split behavior
                while i < len(text) and text[i].isspace():
                    i += 1
            else:
                current += ch
                if ch == '?' and not in_quote and brace_depth == 0:
                    # A '?' truth-test operator (e.g. "a or ? 9 = 9") makes the
                    # rest of the current token atomic: its '=' is equality,
                    # not a constraint separator.
                    truth_open = True
                i += 1

        if current.strip():
            parts.append(current.strip())
        return parts

    def _parse_dim_size(self, size_str, line_number=None):
        """
        Parse dimension size, handling both standard and grid_dim formats.
        :param size_str: Dimension string.
        :param line_number: Line number for errors.
        :param grid_dim: If True, expect grid_dim format (optional label, fixed int size).
        :return: For grid_dim: (label, int) or (None, int); else: (label, size_spec) or size_spec where size_spec is int/(start,end)/None.
        """
        size_str = size_str.strip()
        label = None
        if ':' in size_str:
            label, size_str = [s.strip() for s in size_str.split(':', 1)]
        size_str = size_str.strip()

        if size_str == '*':
            size_spec = None
        elif re.match(r'^-?\d+\s+to\s+\*$', size_str, re.I):
            start = int(size_str.split('to')[0].strip())
            size_spec = (start, None)
        elif re.match(r'^-?\d+\s+to\s+-?\d+$', size_str, re.I):
            m = re.match(r'^(-?\d+)\s+to\s+(-?\d+)$', size_str, re.I)
            start, end = map(int, m.groups())
            if start > end:
                raise SyntaxError(
                    f"Invalid range {start} to {end} at line {line_number}")
            size_spec = (start, end)
        else:
            try:
                size_spec = int(size_str)
            except ValueError:
                size_spec = size_str

        if label:
            return (label, size_spec)
        return size_spec
