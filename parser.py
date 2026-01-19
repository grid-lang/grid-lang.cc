"""
Parsing and preprocessing functionality for GridLang compiler.
Handles code preprocessing, variable parsing, and declaration processing.
"""
import re
from type_processor import GridLangTypeProcessor
from expression import ExpressionEvaluator


class GridLangParser:
    """Handles parsing and preprocessing of GridLang code."""

    def __init__(self, compiler=None):
        self.compiler = compiler

    def _parse_variable_def(self, def_str, line_number):
        # Allow leading ':' in global declarations
        def_str = def_str.lstrip(':').strip()
        constraints = {}
        expr = None
        type_name = None
        unit = None
        dims = None
        type_union = []

        # Handle field assignments (e.g., p.x = value)
        field_match = re.match(r'^([\w_]+)\.(\w+)\s*=\s*(.+)$', def_str)
        if field_match:
            var, field, expr_str = field_match.groups()
            expr = expr_str
            constraints['constant'] = expr
            return f"{var}.{field}", None, constraints, expr

        # Handle array indexing assignments (e.g., D{i+1, 1} = value)
        array_index_match = re.match(
            r'^([\w_]+)\{([^}]+)\}\s*=\s*(.+)$', def_str)
        if array_index_match:
            var_name, indices_str, expr_str = array_index_match.groups()
            expr = expr_str
            # Create a special variable name that includes the indices
            var = f"{var_name}{{{indices_str}}}"
            return var, None, constraints, expr

        # Handle parentheses indexing assignments (e.g., D(1) = value)
        paren_index_match = re.match(
            r'^([\w_]+)\(([^)]+)\)\s*=\s*(.+)$', def_str)
        if paren_index_match:
            var_name, index_expr, expr_str = paren_index_match.groups()
            expr = expr_str
            # Create a special variable name that includes the parentheses indices
            var = f"{var_name}({index_expr})"
            return var, None, constraints, expr

        # Extract 'with' clause early to preserve parentheses
        with_content_str = None
        with_match = re.search(r'\s+with\s+(\(.*\))', def_str, re.I)
        if with_match:
            with_content_str = with_match.group(1)
            def_str = def_str[:with_match.start()] + def_str[with_match.end():]

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
        i = 1
        negated = False
        while i < len(parts):
            keyword = parts[i].lower()
            next_part = parts[i + 1].strip() if i + 1 < len(parts) else ''
            if keyword == 'not':
                if next_part.lower() == 'null':
                    constraints['not_null'] = True
                    i += 2
                    continue
                negated = True
                i += 1
                continue
            if keyword == 'as':
                if negated:
                    constraints['not_type'] = next_part.lower()
                    negated = False
                else:
                    cleaned_part = re.sub(r'\s+or\s*$', '', next_part, flags=re.I).strip()
                    union_parts = [
                        part.strip().lower()
                        for part in re.split(r'\s+or\s+', cleaned_part, flags=re.I)
                        if part.strip()
                    ]
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
                            v_clean = v_clean[1:-1]
                        values.append(v_clean)
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
                    raise SyntaxError(f"Invalid 'in' constraint syntax: '{next_part}' at line {line_number}")
            elif keyword in ('<=', '>=', '<', '>', '<>'):
                if negated:
                    constraints[f'not_{keyword}'] = next_part
                    negated = False
                else:
                    constraints[keyword] = next_part
            elif keyword == 'init':
                constraints['init'] = next_part
            elif keyword == 'index':
                constraints['index'] = next_part
            elif keyword == '=':
                if next_part.startswith('{'):
                    if ';' in next_part or '|' in next_part:
                        matrices = [m.strip()[1:-1] for m in next_part.split('|')]
                        matrix_data = []
                        for matrix in matrices:
                            rows = [r.strip().split(',') for r in matrix.split(';')]
                            matrix_data.append([[float(v.strip()) for v in row] for row in rows])
                        expr = matrix_data
                    else:
                        values_str = next_part[1:-1].split(',')
                        values = []
                        for v in values_str:
                            v_clean = v.strip()
                            if (v_clean.startswith('"') and v_clean.endswith('"')) or (v_clean.startswith("'") and v_clean.endswith("'")):
                                val = v_clean[1:-1]
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

        # Process 'with' clause
        if with_content_str:
            with_inner_content = with_content_str[1:-1].strip()
            with_constraints = {}
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
                if wc.lower().startswith('grid dim'):
                    if dim_constraint is not None:
                        raise SyntaxError(
                            f"Multiple grid DIM statements not allowed in with clause at line {line_number}")
                    dim_match = re.match(
                        r'grid\s+dim\s*\{([^}]+)\}\s*(?:=\s*({.+?}(?:\s*\|\s*{.+?})*|[\d.]+|\w+))?', wc, re.I)
                    if not dim_match:
                        raise SyntaxError(
                            f"Invalid dimension syntax: '{wc}' at line {line_number}")
                    dim_str = dim_match.group(1)
                    grid_data = dim_match.group(2)
                    dim_parts = [p.strip() for p in dim_str.split(',')]
                    dim_constraint = {
                        'dims': [(1, int(p.strip())) for p in dim_parts]}
                    if grid_data:
                        if re.match(r'^{.+?}(?:\s*\|\s*{.+?})*$', grid_data):
                            matrices = [m.strip()[1:-1]
                                        for m in grid_data.split('|')]
                            matrix_data = []
                            for matrix in matrices:
                                rows = [r.strip().split(',')
                                        for r in matrix.split(';')]
                                matrix_data.append(
                                    [[float(v.strip()) for v in row] for row in rows])
                            # Normalize single-matrix to 2D list instead of [[[...]]]
                            dim_constraint['matrix_data'] = matrix_data if len(
                                matrix_data) > 1 else matrix_data
                        elif re.match(r'^[\d.]+$', grid_data):
                            dim_constraint['value'] = float(grid_data)
                        elif re.match(r'^\w+$', grid_data):
                            dim_constraint['data_var'] = grid_data
                        else:
                            raise SyntaxError(
                                f"Invalid grid data: '{grid_data}' at line {line_number}")
                elif '=' in wc:
                    k, v = [s.strip() for s in wc.split('=', 1)]
                    if v.startswith('"') and v.endswith('"'):
                        with_constraints[k] = v[1:-1]
                    else:
                        try:
                            with_constraints[k] = self.compiler.expr_evaluator.eval_expr(
                                v, self.compiler.current_scope().get_full_scope(), line_number)
                        except Exception:
                            v = v.strip('"')
                            with_constraints[k] = v
                else:
                    name = wc.strip()
                    if name:
                        with_constraints[name] = name
            if with_constraints:
                constraints['with'] = with_constraints
            if dim_constraint:
                constraints['dim'] = dim_constraint

        # Process dimensions
        if dims:
            dim_expr = dims.strip()
            # Allow "dim {} or dim 1" style declarations by taking the first clause.
            dim_expr = re.split(r'\s+or\s+', dim_expr, maxsplit=1, flags=re.I)[0].strip()
            if dim_expr == '{}':
                constraints['dim'] = '{}'
            else:
                dim_parts = dim_expr[1:-
                                     1].split(',') if dim_expr.startswith('{') else [dim_expr]
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
        elif constraints.get('with', {}).get('dim'):
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
        if type_union:
            unique_types = list(dict.fromkeys(type_union))
            if len(unique_types) == 1:
                type_name = unique_types[0]
            else:
                constraints['type_union'] = unique_types
                type_name = None

        # Merge in custom type constraints when available
        if type_name and self.compiler and hasattr(self.compiler, 'types_defined'):
            type_def = self.compiler.types_defined.get(type_name.lower())
            if isinstance(type_def, dict):
                type_constraints = type_def.get('_constraints', {}) or {}
                for key, val in type_constraints.items():
                    constraints.setdefault(key, val)
                base_type = type_def.get('_base_type')
                if base_type and 'type' not in constraints:
                    constraints['type'] = base_type

        return var, type_name, constraints, expr

    def _split_on_keywords(self, text):
        """Split a variable definition on keywords, skipping quoted sections."""
        keywords = ['<=', '>=', '<>', '<', '>', '=', 'as', 'of', 'dim', 'in', 'init', 'index', 'not']
        parts = []
        current = ""
        in_quote = False
        quote_char = None
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

            matched = None
            if not in_quote:
                for kw in keywords:
                    if lower_text.startswith(kw, i):
                        # For word keywords, ensure boundaries
                        if kw.isalpha():
                            prev = text[i - 1] if i > 0 else ' '
                            nxt = text[i + len(kw)
                                       ] if i + len(kw) < len(text) else ' '
                            if prev.isalnum() or prev == '_' or nxt.isalnum() or nxt == '_':
                                continue
                        matched = kw
                        break

            if matched:
                if current.strip():
                    parts.append(current.strip())
                parts.append(matched)
                current = ""
                i += len(matched)
                # Skip following whitespace to mimic regex split behavior
                while i < len(text) and text[i].isspace():
                    i += 1
            else:
                current += ch
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
        elif re.match(r'^\d+\s+to\s+\d+$', size_str, re.I):
            m = re.match(r'^(\d+)\s+to\s+(\d+)$', size_str, re.I)
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
