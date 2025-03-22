import re
import math
import csv

class GridLangCompiler:
    def __init__(self):
        self.grid = {}
        self.variables = {}

    def run(self, code):
        self.grid.clear()
        self.variables.clear()
        lines = [line.strip() for line in code.strip().splitlines() if line.strip() and not line.strip().startswith('#')]
        for line in lines:
            self._evaluate_line(line)
        return self.grid

    def export_to_csv(self, filename):
        max_col = 0
        max_row = 0
        cell_data = {}

        for cell, value in self.grid.items():
            try:
                col_str, row = self._split_cell(cell)
            except ValueError:
                continue
            col = self._col_to_num(col_str)
            max_col = max(max_col, col)
            max_row = max(max_row, row)
            cell_data[(row, col)] = value

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            for r in range(1, max_row + 1):
                row_data = []
                for c in range(1, max_col + 1):
                    val = cell_data.get((r, c), '')
                    row_data.append(val if val is not None else '')
                writer.writerow(row_data)

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
                        col, row = self._split_cell(base_cell)
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
        start_col, start_row = self._split_cell(start)
        end_col, end_row = self._split_cell(end)
        values = []
        for r in range(start_row, end_row + 1):
            for c in range(self._col_to_num(start_col), self._col_to_num(end_col) + 1):
                cell = self._num_to_col(c) + str(r)
                if cell in self.grid:
                    values.append(self.grid[cell])
        return values

    def _assign_range(self, start, end, values):
        start = self._interpolate(start)
        end = self._interpolate(end)
        start_col, start_row = self._split_cell(start)
        end_col, end_row = self._split_cell(end)
        i = 0

        same_col = self._col_to_num(start_col) == self._col_to_num(end_col)
        same_row = start_row == end_row

        if same_col and not same_row:
            for r in range(start_row, end_row + 1):
                if i < len(values):
                    cell = start_col + str(r)
                    self.grid[cell] = values[i]
                    i += 1
        elif same_row and not same_col:
            for c in range(self._col_to_num(start_col), self._col_to_num(end_col) + 1):
                if i < len(values):
                    cell = self._num_to_col(c) + str(start_row)
                    self.grid[cell] = values[i]
                    i += 1
        else:
            for r in range(start_row, end_row + 1):
                for c in range(self._col_to_num(start_col), self._col_to_num(end_col) + 1):
                    if i < len(values):
                        cell = self._num_to_col(c) + str(r)
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

    def _split_cell(self, ref):
        match = re.match(r'([A-Z]+)(\d+)', ref)
        if not match:
            raise ValueError(f"Invalid cell reference: {ref}")
        return match.group(1), int(match.group(2))

    def _col_to_num(self, col):
        num = 0
        for c in col:
            num = num * 26 + ord(c) - ord('A') + 1
        return num

    def _num_to_col(self, num):
        result = ''
        while num:
            num, rem = divmod(num - 1, 26)
            result = chr(65 + rem) + result
        return result

    def _next_column(self, cell, offset):
        col, row = self._split_cell(cell)
        new_col = self._num_to_col(self._col_to_num(col) + offset)
        return new_col + str(row)


# Example usage:
if __name__ == '__main__':
    compiler = GridLangCompiler()

    tests = [
        # Test 1
        """
        [A1] := 51
        [A2] := ([A1] + 15) / 1.2e3
        [A3] := sum[A1:A2]
        """,
        # Test 2
        """
        [A1] : a = 51
        [A2] : b = (a + 15) / 1.2e3
        [A3] : c = sum([A1:A2])
        """,
        # Test 3
        """
        : a = 51
        : b = (a + 15) / 1.2e3
        : c = sum{a, b}
        [A1] := a
        [A2] := b
        [A3] := c
        """,
        # Test 4
        """
        [AB2] := 33
        """,
        # Test 5
        """
        [A1] : a = 51
        [@B2] := {12, (15 + a) / 1.2e3, 1+[A1], 8}
        """,
        # Test 6
        """
        [A1] : a = 51
        : b = {12, (15 + a) / 1.2e3, 1+[A1], 8}
        [@B2] := b
        """,
        # Test 7
        """
        [A1] := 2*( SQRT(100) - 7 )
        """,
        # Test 8
        """
        : n = 2
        [A & {n}] := 33
        [A&{n + 1}] := -4.9
        """,
        # Test 9
        """
        [A1:C2] := {1, 2, 3; 10, 11, 12}
        [@A3] := [A1:C1]
        """,
        # Test 10
        """
        [@A1] := {1; 2; 3}
        [B1:B2] := [A2:A3]
        """,
        # Test 11
        """
        : n = 1
        [A1:A&{n+1}] := {1; 2}
        [B&{n}:B2] := {3; 4}
        [C&{2-n}:C&{n+1}] := {-5, -6}
        """
    ]

    for i, test_code in enumerate(tests, 1):
        print(f"Running Test {i}...")
        result = compiler.run(test_code)
        for cell in sorted(result):
            print(f"{cell} = {result[cell]}")
        compiler.export_to_csv(f"grid_output_test{i}.csv")
        print(f"Test {i} completed.\n")