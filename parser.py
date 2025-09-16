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
        def_str = def_str.strip()
        constraints = {}
        expr = None
        type_name = None
        unit = None
        dims = None

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

        # Split on keywords (as, of, dim, in, <=, >=, <, >, =, with)
        parts = re.split(r'\s+(as|of|dim|in|<=|>=|<|>|=)\s+',
                         def_str, flags=re.I)

        # Check for comma-separated variable lists (e.g., "low, high as number")
        if ',' in parts[0]:
            var_names = [v.strip() for v in parts[0].split(',')]
            # For now, return the first variable and let the caller handle the list
            # This will be expanded in the FOR loop processing
            var = var_names[0]
            # Store the full list in constraints for later processing
            constraints['var_list'] = var_names
        else:
            var = parts[0].strip()
        i = 1
        while i < len(parts):
            keyword = parts[i].lower()
            next_part = parts[i + 1].strip() if i + 1 < len(parts) else ''
            if keyword == 'as':
                type_name = next_part.lower()
            elif keyword == 'of':
                unit = next_part
                constraints['unit'] = unit
            elif keyword == 'dim':
                dims = next_part
                constraints['dim'] = dims
            elif keyword == 'in':
                # Handle both set-based and range-based constraints
                if next_part.startswith('{') and next_part.endswith('}'):
                    # Set-based constraint: in {1, 2, 3}
                    values = [v.strip()
                              for v in next_part[1:-1].split(',') if v.strip()]
                    constraints['in'] = values
                elif ' to ' in next_part:
                    # Range-based constraint: in 1 to 10 or in 1 to I+1
                    range_parts = next_part.split(' to ', 1)
                    if len(range_parts) != 2:
                        raise SyntaxError(
                            f"Invalid range syntax: '{next_part}' at line {line_number}")
                    start_expr, end_expr = range_parts[0].strip(
                    ), range_parts[1].strip()
                    constraints['range'] = {
                        'start': start_expr, 'end': end_expr}
                else:
                    raise SyntaxError(
                        f"Invalid 'in' constraint syntax: '{next_part}' at line {line_number}")
            elif keyword in ('<=', '>=', '<', '>'):
                constraints[keyword] = next_part
            elif keyword == '=':
                if next_part.startswith('{'):
                    if ';' in next_part or '|' in next_part:
                        matrices = [m.strip()[1:-1]
                                    for m in next_part.split('|')]
                        matrix_data = []
                        for matrix in matrices:
                            rows = [r.strip().split(',')
                                    for r in matrix.split(';')]
                            matrix_data.append(
                                [[float(v.strip()) for v in row] for row in rows])
                        expr = matrix_data
                    else:
                        values_str = next_part[1:-1].split(',')
                        values = []
                        for v in values_str:
                            v_clean = v.strip()
                            if v_clean.startswith('"') and v_clean.endswith('"') or v_clean.startswith("'") and v_clean.endswith("'"):
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
            if with_constraints:
                constraints['with'] = with_constraints
            if dim_constraint:
                constraints['dim'] = dim_constraint

        # Process dimensions
        if dims:
            dim_parts = dims[1:-
                             1].split(',') if dims.startswith('{') else [dims]
            dims_list = []
            for part in dim_parts:
                part = part.strip()
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

        return var, type_name, constraints, expr

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
                raise SyntaxError(
                    f"Invalid dimension size: '{size_str}' at line {line_number}")

        if label:
            return (label, size_spec)
        return size_spec
