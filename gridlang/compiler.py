import re
import math
from gridlang.utils import export_to_csv, split_cell, col_to_num, num_to_col

class GridLangCompiler:
    def __init__(self):
        self.grid = {}
        self.variables = {}

    def run(self, code):
        self.grid.clear()
        self.variables.clear()
        processed_lines = []
        for line in code.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("'"):
                continue
            # Handle inline comments
            if "'" in line:
                line = line.split("'", 1)[0].strip()
            if line:
                processed_lines.append(line)
        
        for line in processed_lines:
            self._evaluate_line(line)
        return self.grid

    def export_to_csv(self, filename):
        export_to_csv(self.grid, filename)

    def _evaluate_line(self, line):
        if ':=' in line:
            cell, expr = map(str.strip, line.split(':=', 1))
            cell = cell[1:-1]
            cell = self._interpolate(cell)
            if cell.startswith('@'):
                base_cell = cell[1:]
                values = self._evaluate_array(expr)
                if self._is_vertical_array(expr):
                    for i, val in enumerate(values):
                        col, row = split_cell(base_cell)
                        target = col + str(int(row) + i)
                        self.grid[target] = val
                else:
                    for i, val in enumerate(values):
                        target = self._next_column(base_cell, i)
                        self.grid[target] = val
            elif ':' in cell:
                start, end = map(str.strip, cell.split(':'))
                expr = expr.strip()
                values = self._evaluate_array(expr)
                self._assign_range(start, end, values)
            else:
                self.grid[cell] = self._eval_expr(expr)
        elif ':=' not in line and ':' in line:
            parts = line.split(':', 1)
            cell_or_decl = parts[0].strip()
            rest = parts[1].strip()
            if '=' in rest:
                var, expr = map(str.strip, rest.split('=', 1))
                if expr.startswith('{') and expr.endswith('}'):  # Store arrays as evaluated lists
                    self.variables[var] = self._evaluate_array(expr)
                else:
                    value = self._eval_expr(expr)
                    self.variables[var] = value
                    if cell_or_decl:
                        cell = self._interpolate(cell_or_decl[1:-1])
                        self.grid[cell] = value
            else:
                raise SyntaxError(f"Invalid variable declaration: {line}")

    def _is_vertical_array(self, expr):
        return ';' in expr and not ',' in expr

    def _eval_expr(self, expr):
        expr = self._interpolate(expr)
        expr = re.sub(r'sum\[([^:\[\]]+):([^:\[\]]+)\]', lambda m: str(sum(self._get_range_values(m.group(1), m.group(2)))), expr)
        expr = re.sub(r'sum\(\[([^:\[\]]+):([^:\[\]]+)\]\)', lambda m: str(sum(self._get_range_values(m.group(1), m.group(2)))), expr)
        expr = re.sub(r'sum\{([^}]+)\}', lambda m: str(sum([self._eval_expr(x.strip()) for x in m.group(1).split(',')])), expr)

        for var in self.variables:
            if isinstance(self.variables[var], list):
                continue
            expr = re.sub(rf'\b{var}\b', str(self.variables[var]), expr)

        for cell in self.grid:
            expr = expr.replace(f'[{cell}]', str(self.grid[cell]))

        expr = expr.replace('SQRT', 'math.sqrt')
        expr = expr.replace('SUM', 'sum')

        try:
            return eval(expr, {"__builtins__": None, "math": math, "sum": sum})
        except Exception as e:
            print(f"Error evaluating expression: {expr} => {e}")
            return None

    def _evaluate_array(self, expr):
        expr = self._interpolate(expr)
        if expr.startswith('{') and expr.endswith('}'):  # array literal
            rows = expr[1:-1].split(';')
            values = []
            for row in rows:
                parts = row.split(',')
                for part in parts:
                    values.append(self._eval_expr(part.strip()))
            return values
        elif expr.startswith('[') and expr.endswith(']'):
            start, end = map(str.strip, expr[1:-1].split(':'))
            return self._get_range_values(start, end)
        elif expr in self.variables and isinstance(self.variables[expr], list):
            return self.variables[expr]
        elif expr in self.grid:
            val = self.grid[expr]
            return val if isinstance(val, list) else [val]
        return []

    def _get_range_values(self, start, end):
        start = self._interpolate(start)
        end = self._interpolate(end)
        start_col, start_row = split_cell(start)
        end_col, end_row = split_cell(end)
        values = []
        for r in range(start_row, end_row + 1):
            for c in range(col_to_num(start_col), col_to_num(end_col) + 1):
                cell = num_to_col(c) + str(r)
                if cell in self.grid:
                    values.append(self.grid[cell])
        return values

    def _assign_range(self, start, end, values):
        start = self._interpolate(start)
        end = self._interpolate(end)
        start_col, start_row = split_cell(start)
        end_col, end_row = split_cell(end)
        i = 0

        same_col = col_to_num(start_col) == col_to_num(end_col)
        same_row = start_row == end_row

        if same_col and not same_row:
            for r in range(start_row, end_row + 1):
                if i < len(values):
                    cell = start_col + str(r)
                    self.grid[cell] = values[i]
                    i += 1
        elif same_row and not same_col:
            for c in range(col_to_num(start_col), col_to_num(end_col) + 1):
                if i < len(values):
                    cell = num_to_col(c) + str(start_row)
                    self.grid[cell] = values[i]
                    i += 1
        else:
            for r in range(start_row, end_row + 1):
                for c in range(col_to_num(start_col), col_to_num(end_col) + 1):
                    if i < len(values):
                        cell = num_to_col(c) + str(r)
                        self.grid[cell] = values[i]
                        i += 1

    def _interpolate(self, expr):
        expr = re.sub(r'([A-Z])\s+&', r'\1&', expr)
        expr = re.sub(r'\s+', '', expr)
        pattern = re.compile(r'([A-Z]+)&\{([^}]+)\}')
        while True:
            match = pattern.search(expr)
            if not match:
                break
            col, inner = match.groups()
            val = str(int(self._eval_expr(inner)))
            expr = expr[:match.start()] + col + val + expr[match.end():]
        return expr

    def _next_column(self, cell, offset):
        col, row = split_cell(cell)
        new_col = num_to_col(col_to_num(col) + offset)
        return new_col + str(row)
