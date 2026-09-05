"""
Type processing functionality for GridLang compiler.
Handles type definitions, type code execution, and type-related operations.
"""

import copy
import re
import numbers
from utils import (
    get_case_insensitive_key,
    is_wildcard_address,
    split_var_defs,
)


def split_builder_chain(expr):
    """Split a 'new ...' expression into (base_expr, chain_text).

    chain_text is the trailing '-> builder(args) ...' suffix (including the
    leading arrow), or None when the expression has no top-level chain.
    Everything inside quotes or nested delimiters is skipped.
    """
    in_quote = None
    paren = brace = bracket = 0
    for i in range(len(expr)):
        ch = expr[i]
        if in_quote:
            if ch == in_quote and (i == 0 or expr[i - 1] != '\\'):
                in_quote = None
            continue
        if ch in ('"', "'"):
            in_quote = ch
            continue
        if ch == '(':
            paren += 1
            continue
        if ch == ')':
            paren = max(paren - 1, 0)
            continue
        if ch == '{':
            brace += 1
            continue
        if ch == '}':
            brace = max(brace - 1, 0)
            continue
        if ch == '[':
            bracket += 1
            continue
        if ch == ']':
            bracket = max(bracket - 1, 0)
            continue
        if (ch == '-' and i + 1 < len(expr) and expr[i + 1] == '>'
                and paren == brace == bracket == 0):
            before = expr[i - 1] if i > 0 else ''
            after = expr[i + 2:].lstrip()
            if (not before or before.isspace() or before in '})'):
                if re.match(r'\$?[A-Za-z]', after):
                    return expr[:i].strip(), expr[i:]
    return expr, None


class GridLangTypeProcessor:
    """Handles type definitions and type-related processing."""

    def __init__(self, compiler=None):
        self.compiler = compiler

    def _new_type_def_state(self):
        """Create the accumulator used while parsing a type definition."""
        return {
            'fields': {},
            'executable_code': [],
            'inputs': [],
            'hidden_fields': set(),
            'field_constraints': {},
            'computed_fields': {},
            'init_fields': set(),
            'default_fields': set(),
        }

    def _parse_type_def(self, lines, line_number=None, type_name=None):
        """Parse type definition lines and extract fields and executable code."""
        state = self._new_type_def_state()
        for line in lines:
            self._parse_type_def_line(line, line_number, state)
        self._collect_type_computed_fields(state)
        return self._finalize_type_def_state(state)

    def _parse_type_def_line(self, line, line_number, state):
        """Route one type-definition line into inputs, fields, or executable code."""
        stripped = line.strip()
        if not stripped:
            return
        lowered = stripped.lower()
        if lowered.startswith('input '):
            state['inputs'].extend(
                self._parse_type_input_definition(line, stripped, line_number)
            )
            return

        field_line = self._extract_type_field_line(stripped, lowered)
        if field_line:
            self._record_type_field_definition(
                state, line, field_line, lowered, line_number
            )
            return

        # Executable code inside type definition (strip leading colon if present)
        state['executable_code'].append(line.lstrip(':').strip())

    def _parse_type_input_definition(self, line, stripped, line_number):
        """Parse a type constructor input declaration."""
        input_body = stripped[5:].strip()
        input_type = None
        input_default = None
        names_part = input_body
        m_as = re.search(r'\s+as\s+', input_body, re.I)
        if m_as:
            names_part = input_body[:m_as.start()].strip()
            remainder = input_body[m_as.end():].strip()
            m_default = re.search(r'\bor\s*=\s*(.+)$', remainder, re.I)
            if m_default:
                input_type = remainder[:m_default.start()].strip() or None
                input_default = m_default.group(1).strip()
            else:
                input_type = remainder.strip() or None
        else:
            m_default = re.search(r'\bor\s*=\s*(.+)$', input_body, re.I)
            if m_default:
                names_part = input_body[:m_default.start()].strip()
                input_default = m_default.group(1).strip()

        name_list = [n.strip() for n in names_part.split(',') if n.strip()]
        if not name_list:
            raise SyntaxError(
                f"Invalid input definition: '{line}' at line {line_number}")
        return [
            {
                'name': in_name,
                'type': input_type.lower() if input_type else None,
                'default': input_default
            }
            for in_name in name_list
        ]

    def _extract_type_field_line(self, stripped, lowered):
        """Return the declaration body for a type field line, if any.

        Public fields are declared with ``[addr] : var`` where the
        ``[addr]`` part is optional.  ``For``/``Let`` lines are executable
        code, never field declarations.
        """
        m = re.match(r'^\[\s*\^?[A-Za-z]+\d+\s*\]\s*:\s*(?!\=)(.+)$', stripped)
        if m:
            return m.group(1).strip()
        if stripped.startswith(':'):
            return stripped[1:].strip()
        return None

    def _split_type_field_initializer(self, field_line):
        """Separate a field declaration from its init/default expression.

        ``or =`` provides a *default* value applied only when the field has no
        value set yet (e.g. by a ``with`` clause); ``init`` pushes an active
        initial value into the field.
        """
        init_expr = None
        kind = None
        default_match = re.search(r'\bor\s*=\s*(.+)$', field_line, re.I)
        if default_match:
            init_expr = default_match.group(1).strip()
            kind = 'or_default'
            field_line = field_line[:default_match.start()].strip()
        init_match = re.search(r'\binit\b', field_line, re.I)
        if init_match:
            init_expr = field_line[init_match.end():].strip()
            kind = 'init'
            field_line = field_line[:init_match.start()].strip()
        if init_expr is None:
            # Typed inline initialization: ``: x as number = in_x`` assigns an
            # active initial value (equivalent to ``init in_x``), distinct from
            # the inert ``or =`` default.
            as_pos = re.search(r'\bas\s', field_line, re.I)
            eq_pos = field_line.find('=')
            while eq_pos > 0 and field_line[eq_pos - 1] in '<>=':
                eq_pos = field_line.find('=', eq_pos + 1)
            if as_pos and eq_pos > as_pos.end():
                init_expr = field_line[eq_pos + 1:].strip()
                kind = 'init'
                field_line = field_line[:eq_pos].strip()
        return field_line, init_expr, kind

    def _parse_type_field_constraints(self, field_line, line_number, type_name):
        """Parse field constraints with the compiler parser when available."""
        parsed_constraints = {}
        parsed_type = type_name
        if self.compiler and hasattr(self.compiler, '_parse_variable_def'):
            try:
                _, parsed_type, parsed_constraints, _ = self.compiler._parse_variable_def(
                    field_line, line_number)
                parsed_constraints = parsed_constraints or {}
            except Exception:
                parsed_constraints = {}
        if parsed_type and 'type' not in parsed_constraints and 'type_union' not in parsed_constraints:
            parsed_constraints['type'] = parsed_type.lower()
        parsed_constraints.pop('var_list', None)
        return parsed_constraints

    def _parse_type_field_definition(self, line, field_line, line_number):
        """Parse one field declaration line into normalized metadata."""
        field_line, init_expr, init_kind = self._split_type_field_initializer(
            field_line)
        match = re.match(
            r'^(\$?[A-Za-z][\w_]*(?:\s*,\s*\$?[A-Za-z][\w_.]*)*)', field_line)
        if not match:
            raise SyntaxError(
                f"Invalid field definition: '{line}' at line {line_number}")
        var_names = [v.strip() for v in match.group(1).split(',') if v.strip()]
        type_candidates = re.findall(r'\bas\s+([A-Za-z][\w_]*)', field_line, re.I)
        type_name = type_candidates[-1].lower() if type_candidates else None
        has_dim = re.search(r'\bdim\b', field_line, re.I)
        effective_type = 'array' if has_dim and not type_name else (type_name or 'unknown')
        parsed_cons = self._parse_type_field_constraints(
            field_line, line_number, type_name)
        # Keep `: x as number key` - key is part of type (`number key` ≡ `L` where `L as Keytype(number)`)
        if re.search(r'\bkey\b', field_line, re.I):
            parsed_cons = dict(parsed_cons) if parsed_cons else {}
            parsed_cons['key'] = True
        return {
            'field_line': field_line,
            'init_expr': init_expr,
            'init_kind': init_kind,
            'var_names': var_names,
            'effective_type': effective_type,
            'parsed_constraints': parsed_cons,
        }

    def _record_type_field_definition(self, state, line, field_line, lowered, line_number):
        """Apply one parsed field definition to the type-definition state."""
        parsed_field = self._parse_type_field_definition(
            line, field_line, line_number)
        for name in parsed_field['var_names']:
            clean_name = name[1:] if name.startswith('$') else name
            if name.startswith('$'):
                state['hidden_fields'].add(clean_name.lower())
            state['fields'][clean_name] = parsed_field['effective_type']
            if parsed_field['parsed_constraints']:
                state['field_constraints'][clean_name] = dict(
                    parsed_field['parsed_constraints'])
            if parsed_field['init_expr']:
                state['executable_code'].append(
                    f"{clean_name} = {parsed_field['init_expr']}")
                state['init_fields'].add(clean_name.lower())
                if parsed_field['init_kind'] == 'or_default':
                    state['default_fields'].add(clean_name.lower())
        # Allow constructor-style assignments (e.g., ": x = in_x") to execute.
        if (re.search(r'^\$?[A-Za-z][\w_.]*\s*=', parsed_field['field_line']) and
                'or =' not in lowered):
            state['executable_code'].append(parsed_field['field_line'])

    def _collect_type_computed_fields(self, state):
        """Capture computed fields for reactive recomputation."""
        for code_line in state['executable_code']:
            match = re.match(r'^\s*(\$?[A-Za-z][\w_.]*)\s*=\s*(.+)$', code_line)
            if not match:
                continue
            lhs = match.group(1).strip()
            rhs = match.group(2).strip()
            clean_lhs = lhs[1:] if lhs.startswith('$') else lhs
            if clean_lhs in state['fields']:
                state['computed_fields'][clean_lhs] = rhs

    def _finalize_type_def_state(self, state):
        """Build the public type-definition structure from parse state."""
        fields = state['fields']
        if state['executable_code']:
            fields['_executable_code'] = state['executable_code']
        if state['inputs']:
            fields['_inputs'] = state['inputs']
        if state['field_constraints']:
            fields['_field_constraints'] = state['field_constraints']
        if state['hidden_fields']:
            fields['_hidden_fields'] = state['hidden_fields']
        if state['computed_fields']:
            fields['_computed_fields'] = state['computed_fields']
        if state['init_fields']:
            fields['_init_fields'] = state['init_fields']
        if state['default_fields']:
            fields['_default_fields'] = state['default_fields']
        fields['_member_keys'] = {k.lower() for k in state['fields'].keys()
                                   if not str(k).startswith('_')}
        return fields

    def _execute_type_code(self, code_lines, var_name, value_dict, line_number, input_values=None):
        """Execute code that was defined inside a type definition"""

        # Create a temporary scope for execution
        self.compiler.push_scope()
        prev_hidden_access = getattr(
            self.compiler, '_allow_hidden_field_access', False)
        prev_hidden_member_calls = getattr(
            self.compiler, '_allow_hidden_member_calls', False)
        self.compiler._allow_hidden_field_access = True
        self.compiler._allow_hidden_member_calls = True

        # Add the type instance to the scope so code can reference it
        inferred_type = value_dict.get('_type_name') if isinstance(
            value_dict, dict) else None
        self.compiler.current_scope().define(
            var_name, value_dict, inferred_type or 'object')
        # Use lowercase "this"; case-insensitive lookup covers "This".
        if str(var_name).lower() != 'this':
            self.compiler.current_scope().define(
                'this', value_dict, inferred_type or 'object')
        input_values = input_values or {}
        for in_name, in_val in input_values.items():
            self.compiler.current_scope().define(
                in_name, in_val, None, {'input': True}, is_uninitialized=False)

        # Add a 'grid' field only if constructor code references it
        if 'grid' not in value_dict and any('grid' in line.lower() for line in code_lines):
            value_dict['grid'] = {}

        # Push instance grid as context so cell references inside the type body
        # resolve against the instance grid rather than the global grid.
        instance_grid = value_dict.get('grid') if isinstance(value_dict, dict) else None
        if isinstance(instance_grid, dict):
            if not hasattr(self.compiler, '_context_grid_stack'):
                self.compiler._context_grid_stack = []
            self.compiler._context_grid_stack.append(instance_grid)

        try:
            if isinstance(value_dict, dict) and value_dict.get('_with_conflict'):
                return
            type_def = {}
            if inferred_type:
                type_def = self.compiler.types_defined.get(
                    str(inferred_type).lower(), {}) or {}
            init_fields = set(type_def.get('_init_fields', set()))
            default_fields = set(type_def.get('_default_fields', set()))
            self._execute_type_block(
                code_lines, value_dict, input_values, line_number, init_fields,
                default_fields=default_fields)
        except Exception as e:
            raise
        finally:
            if isinstance(instance_grid, dict) and hasattr(self.compiler, '_context_grid_stack') and self.compiler._context_grid_stack:
                self.compiler._context_grid_stack.pop()
            self.compiler._allow_hidden_field_access = prev_hidden_access
            self.compiler._allow_hidden_member_calls = prev_hidden_member_calls
            self.compiler.pop_scope()
            if isinstance(value_dict, dict):
                # The WITH marker only matters while the constructor body runs.
                value_dict.pop('_with_applied_fields', None)

    def _execute_type_block(self, code_lines, value_dict, input_values, line_number, init_fields=None, default_fields=None):
        """Execute a list of type code lines within the current scope."""
        init_fields = init_fields or set()
        default_fields = default_fields or set()
        i = 0
        while i < len(code_lines):
            code_line = code_lines[i]
            stripped_line = code_line.strip()
            if not stripped_line:
                i += 1
                continue


            if (re.match(r'^(for|let)\b', stripped_line, re.I) and
                    '=' not in stripped_line and not re.search(r'\bdo\b', stripped_line, re.I)
                    and not re.search(r'\bthen\b', stripped_line, re.I)
                    and not re.search(r'\binit\b', stripped_line, re.I)):
                # Skip field declarations that slipped into executable code.
                i += 1
                continue

            if stripped_line.lower().startswith('push '):
                push_match = re.match(
                    r'^\s*push\s+(\$?[\w_]+(?:\([^)]+\)|\{[^}]+\})?)(?:\s*=\s*(.+))?\s*$',
                    stripped_line,
                    re.I,
                )
                if not push_match:
                    raise SyntaxError(
                        f"Invalid PUSH syntax at line {line_number}")
                target, value_expr = push_match.groups()
                if value_expr is None:
                    value_expr = target
                assign_line = f"{target} = {value_expr}"
                self._process_type_assignment(
                    assign_line, value_dict, input_values, line_number, init_fields,
                    default_fields=default_fields)
                i += 1
                continue
            if (re.match(r'^for\b', stripped_line, re.I)
                    and not re.search(r'\bdo\b', stripped_line, re.I)
                    and not re.search(r'\bthen\b', stripped_line, re.I)
                    and (re.search(r'=', stripped_line) or re.search(r'\binit\b', stripped_line, re.I))):
                for_body = stripped_line[4:].strip()
                var, type_name, constraints, expr = self.compiler._parse_variable_def(
                    for_body, line_number)
                init_expr = (constraints or {}).get('init')
                if expr is None and init_expr is not None:
                    expr = init_expr
                if expr is not None:
                    eval_scope = self._build_type_eval_scope(
                        value_dict, input_values)
                    value = self.compiler.expr_evaluator.eval_or_eval_array(
                        str(expr), eval_scope, line_number)
                    scope = self.compiler.current_scope()
                    inferred = type_name or self.compiler.array_handler.infer_type(
                        value, line_number)
                    if inferred == 'int':
                        inferred = 'number'
                    scope.define(var, value, inferred,
                                 constraints or {}, is_uninitialized=False)
                i += 1
                continue
            if re.match(r'^[A-Za-z][\w_.]*\s*\(.*\)\s*$', stripped_line):
                helper_name = stripped_line.split('(', 1)[0].strip()
                type_name = value_dict.get('_type_name') if isinstance(
                    value_dict, dict) else None
                helper_defs = {}
                if type_name:
                    type_def = self.compiler.types_defined.get(type_name.lower(), {})
                    helper_defs = type_def.get('_builders', {}) if isinstance(
                        type_def, dict) else {}
                if helper_name.lower() == 'super' or helper_name.lower() in helper_defs:
                    args_text = stripped_line[stripped_line.find('(') + 1: stripped_line.rfind(')')]
                    args = []
                    if args_text.strip():
                        args = [a.strip()
                                for a in re.split(r',(?![^{]*})', args_text) if a.strip()]
                    eval_scope = self._build_type_eval_scope(
                        value_dict, input_values)
                    arg_values = [self.compiler.expr_evaluator.eval_or_eval_array(
                        a, eval_scope, line_number) for a in args]
                    self._execute_builder(
                        type_name, helper_name, value_dict, line_number, arg_values)
                    i += 1
                    continue
            if stripped_line.startswith('[') and ':=' in stripped_line:
                # Assignment to grid: [B1] := 1
                self._process_grid_assignment(
                    stripped_line, 'this', value_dict, line_number)
                i += 1
                continue
            if re.match(r'^for\b', stripped_line, re.I) and re.search(r'\bdo\b', stripped_line, re.I):
                # For var = expr do — define var and run body once
                if '=' in stripped_line and not re.search(r'\bin\b', stripped_line, re.I):
                    i = self._process_type_for_assignment_block(
                        code_lines, i, value_dict, input_values, line_number)
                    continue
                i = self._process_type_for_loop(
                    code_lines, i, value_dict, input_values, line_number)
                continue
            if re.match(r'^if\b', stripped_line, re.I) and re.search(r'\bthen\b', stripped_line, re.I):
                i = self._process_type_if_block(
                    code_lines, i, value_dict, input_values, line_number)
                continue
            if re.match(r'^let\b', stripped_line, re.I):
                # Let ... then block inside type constructor
                if re.search(r'\bthen\b', stripped_line, re.I):
                    i = self._process_type_let_then_block(
                        code_lines, i, value_dict, input_values, line_number)
                    continue
                # Let statement: Let grid{a, b} = grid{a-1, b-1} + grid{a-1, b}
                self._process_type_let_statement(
                    stripped_line, 'this', value_dict, line_number)
                i += 1
                continue
            if stripped_line.startswith(':'):
                # Field declaration: : attr init "attr" or : attr = expr
                colon_line = stripped_line[1:].strip()
                init_m = re.match(r'^(\$?[\w_]+)\s+init\s+(.+)$', colon_line, re.I)
                if init_m:
                    colon_line = f"{init_m.group(1)} = {init_m.group(2)}"
                self._process_type_assignment(
                    colon_line, value_dict, input_values, line_number, init_fields,
                    default_fields=default_fields)
                i += 1
                continue
            if '=' in stripped_line:
                # Simple assignment inside constructor (e.g., x = in_x)
                self._process_type_assignment(
                    stripped_line, value_dict, input_values, line_number, init_fields,
                    default_fields=default_fields)
                i += 1
                continue
            if stripped_line.lower().startswith('end'):
                i += 1
                continue

            if stripped_line.startswith("'"):
                i += 1
                continue
            raise SyntaxError(
                f"Unrecognized statement in type constructor: '{stripped_line}' at line {line_number}")

    def _process_grid_assignment(self, line, var_name, value_dict, line_number):
        """Process grid assignment like [B1] := 1 or [A1.B1:A2.B2] := 7"""
        # Extract cell reference and value
        match = re.match(r'\[([^\]]+)\]\s*:=\s*(.+)$', line)
        if match:
            cell_ref, value_expr = match.groups()
            cell_ref = cell_ref.strip()
            # Strip leading ^ for spill prefix
            if cell_ref.startswith('^'):
                cell_ref = cell_ref[1:].strip()
            # Convert cell reference to grid coordinates
            value = self.compiler.expr_evaluator.eval_expr(
                value_expr, self.compiler.current_scope().get_full_scope(), line_number)
            if is_wildcard_address(cell_ref):
                self.compiler.array_handler._assign_wildcard_range(
                    cell_ref, value, expr_part=value_expr, line_number=line_number)
                return
            # Range address (e.g. A1.B1:A2.B2)
            if ':' in cell_ref:
                from utils import parse_address
                from math import prod as _prod
                range_parts = cell_ref.split(':')
                if len(range_parts) == 2:
                    start_addr, end_addr = range_parts[0].strip(), range_parts[1].strip()
                    try:
                        start_idx = tuple(i - 1 for i in parse_address(start_addr))
                        end_idx = tuple(i - 1 for i in parse_address(end_addr))
                        # Ensure value_dict has a grid
                        if 'grid' not in value_dict:
                            value_dict['grid'] = {}
                        inst_grid = value_dict['grid']
                        # Expand the range and write directly to the instance grid
                        starts = [min(a, b) for a, b in zip(start_idx, end_idx)]
                        shape = [abs(a - b) + 1 for a, b in zip(start_idx, end_idx)]
                        is_array = isinstance(value, list) or (isinstance(value, dict) and 'array' in value)
                        if is_array:
                            flat_vals = self.compiler.array_handler.flatten_array(value, line_number)
                        else:
                            flat_vals = [value]
                        if flat_vals:
                            for flat_i in range(_prod(shape)):
                                rem = flat_i
                                idxs = []
                                for dim_size in shape:
                                    idxs.append(rem % dim_size)
                                    rem //= dim_size
                                addr = tuple(starts[d] + idxs[d] for d in range(len(shape)))
                                inst_grid[addr] = flat_vals[flat_i % len(flat_vals)]
                        return
                    except (ValueError, KeyError):
                        pass
            # Single extended (N-D) address
            if '.' in cell_ref:
                from utils import parse_address
                try:
                    indices = parse_address(cell_ref)
                    zero_idx = tuple(i - 1 for i in indices)
                    if 'grid' not in value_dict:
                        value_dict['grid'] = {}
                    value_dict['grid'][zero_idx] = value
                    return
                except (ValueError, KeyError):
                    pass
            col = re.match(r'([A-Z]+)', cell_ref).group(1)
            row = int(re.match(r'[A-Z]+(\d+)', cell_ref).group(1))

            # Convert column letters to numbers (A=1, B=2, etc.)
            col_num = 0
            for char in col:
                col_num = col_num * 26 + (ord(char.upper()) - ord('A') + 1)

            # Store in grid (1-based cell ref -> 0-based sparse array key)
            if 'grid' not in value_dict:
                value_dict['grid'] = {}
            value_dict['grid'][(row - 1, col_num - 1)] = value


    def _process_type_for_loop(self, all_lines, start_index, value_dict, input_values, line_number):
        """Process for loop inside type definition."""
        loop_line = all_lines[start_index].strip()
        lower_line = loop_line.lower()
        if not re.search(r'\bdo\b', lower_line):
            # This is a field declaration, not a loop.
            return start_index + 1

        match = re.match(r'^\s*for\s+(.+?)\s+do\s*$', loop_line, re.I)
        if not match:
            raise SyntaxError(f"Unsupported loop syntax: {loop_line}")

        var_defs = match.group(1).strip()
        var_parts = split_var_defs(var_defs)
        loop_defs = []
        for var_part in var_parts:
            part = var_part.strip()
            part_match = re.match(
                r'^([\w_]+)\s+in\s+(.+?)(?:\s+step\s+(-?\d+))?(?:\s+index\s+([\w_]+))?$',
                part, re.I)
            if not part_match:
                raise SyntaxError(f"Unsupported loop syntax: {loop_line}")
            var_name, range_expr, step_str, index_var = part_match.groups()
            step = int(step_str) if step_str else None
            loop_defs.append({
                'var_name': var_name,
                'range_expr': range_expr.strip(),
                'step': step,
                'index_var': index_var
            })

        # Extract loop body
        depth = 1
        end_index = None
        i = start_index + 1
        while i < len(all_lines):
            line = all_lines[i].strip()
            if re.match(r'^\s*for\b', line, re.I) and re.search(r'\bdo\b', line, re.I):
                depth += 1
            elif re.match(r'^\s*end\b', line, re.I):
                depth -= 1
                if depth == 0:
                    end_index = i
                    break
            i += 1
        if end_index is None:
            raise SyntaxError(
                f"Unclosed FOR block starting at line {line_number}")

        loop_body = all_lines[start_index + 1:end_index]

        def _evaluate_loop_values(range_expr, step_value):
            eval_scope = self._build_type_eval_scope(value_dict, input_values)
            if ' to ' in range_expr:
                start_expr, end_expr = range_expr.split(' to ', 1)
                step_expr = None
                if ' step ' in end_expr:
                    end_expr, step_expr = [
                        part.strip() for part in end_expr.split(' step ', 1)]
                start_val = self.compiler.expr_evaluator.eval_expr(
                    start_expr.strip(), eval_scope, line_number)
                end_val = self.compiler.expr_evaluator.eval_expr(
                    end_expr.strip(), eval_scope, line_number)
                start_val = int(start_val)
                end_val = int(end_val)
                step_final = step_value
                if step_expr and step_final is None:
                    step_final = int(self.compiler.expr_evaluator.eval_expr(
                        step_expr, eval_scope, line_number))
                if step_final is None:
                    step_final = 1
                if step_final < 0:
                    return list(range(start_val, end_val - 1, step_final))
                return list(range(start_val, end_val + 1, step_final))
            if range_expr.startswith('{') and range_expr.endswith('}'):
                values = []
                inner = range_expr[1:-1].strip()
                if not inner:
                    return []
                parts = [p.strip() for p in inner.split(',') if p.strip()]
                for part in parts:
                    if (part.startswith('"') and part.endswith('"')) or (
                            part.startswith("'") and part.endswith("'")):
                        values.append(part[1:-1])
                        continue
                    try:
                        values.append(self.compiler.expr_evaluator.eval_expr(
                            part, eval_scope, line_number))
                    except Exception:
                        try:
                            values.append(float(part))
                        except ValueError:
                            values.append(part)
                return values

            result = self.compiler.expr_evaluator.eval_or_eval_array(
                range_expr, eval_scope, line_number)
            if isinstance(result, dict):
                return self.compiler.array_handler.flatten_array(
                    result, line_number)
            if isinstance(result, (list, tuple)):
                return list(result)
            if isinstance(result, numbers.Real):
                return [result]
            if result is None:
                return []
            return [result]

        def _execute_loop(level=0):
            if level >= len(loop_defs):
                self._execute_type_block(
                    loop_body, value_dict, input_values, line_number)
                return
            loop_def = loop_defs[level]
            values = _evaluate_loop_values(
                loop_def['range_expr'], loop_def['step'])
            for idx, val in enumerate(values, start=1):
                self.compiler.push_scope(is_private=False)
                self.compiler.current_scope().define(
                    loop_def['var_name'], val, 'number')
                if loop_def['index_var']:
                    self.compiler.current_scope().define(
                        loop_def['index_var'], idx, 'number')
                _execute_loop(level + 1)
                self.compiler.pop_scope()

        _execute_loop()
        return end_index + 1

    def _process_type_for_assignment_block(self, code_lines, i, value_dict, input_values, line_number):
        """Handle ``For var = expr do ... End`` inside type constructors.

        Defines *var* = *expr*, then executes the body lines once.
        """
        header = code_lines[i].strip()
        header = re.sub(r'\s+do\s*$', '', header, flags=re.I).strip()
        header = re.sub(r'^For\s+', '', header, count=1, flags=re.I)
        eval_scope = self._build_type_eval_scope(value_dict, input_values)
        var, type_name, constraints, expr = self.compiler._parse_variable_def(
            header, line_number)
        init_expr = (constraints or {}).get('init')
        if expr is None and init_expr is not None:
            expr = init_expr
        if expr is not None:
            value = self.compiler.expr_evaluator.eval_or_eval_array(
                str(expr), eval_scope, line_number)
            scope = self.compiler.current_scope()
            defining_scope = scope.get_defining_scope(var)
            inferred = type_name or self.compiler.array_handler.infer_type(
                value, line_number)
            if inferred == 'int':
                inferred = 'number'
            if defining_scope:
                defining_scope.update(var, value, line_number)
            else:
                scope.define(var, value, inferred,
                             constraints or {}, is_uninitialized=False)
        # Collect and execute body
        depth = 1
        scan_i = i + 1
        body = []
        while scan_i < len(code_lines) and depth > 0:
            line = code_lines[scan_i].strip()
            if re.match(r'^for\b', line, re.I) and re.search(r'\bdo\b', line, re.I):
                depth += 1
            elif re.match(r'^let\b', line, re.I) and re.search(r'\bthen\b', line, re.I):
                depth += 1
            elif re.match(r'^end\b', line, re.I):
                depth -= 1
                if depth == 0:
                    break
            body.append(code_lines[scan_i])
            scan_i += 1
        self.compiler.push_scope(is_loop_scope=True)
        self._execute_type_block(body, value_dict, input_values, line_number)
        self.compiler.pop_scope()
        return scan_i + 1

    def _process_type_let_statement(self, line, var_name, value_dict, line_number):
        """Process let statement inside type definition.

        Routes to the shared compiler utilities for all forms:
        ``Let grid{a,b} = expr``, ``Let grid![addr] = expr``,
        ``Let x as T = expr``, ``Let x not null or = 0``, etc.
        """
        body = re.sub(r'^Let\s+', '', line.strip(), count=1, flags=re.I)
        eval_scope = self._build_type_eval_scope(value_dict, {})

        # 1) Wildcard bang-assign: Let grid![C2.A] = 0.6
        #    Handle before _try_let_index_assignment because extended
        #    addresses like C2.A are not valid index targets.
        bang_match = re.match(
            r'^(\$?[\w_]+)!\[([^\]]+)\]\s*=\s*(.+)$', body)
        if bang_match:
            field_name, address, value_expr = bang_match.groups()
            if field_name.startswith('$'):
                field_name = field_name[1:]
            value = self.compiler.expr_evaluator.eval_expr(
                value_expr.strip(), eval_scope, line_number)
            self.compiler.array_handler._assign_wildcard_range(
                address, value, expr_part=value_expr.strip(),
                line_number=line_number)
            return

        # 2) Indexed write via shared utility: grid{a,b}=expr, x(1)=5
        if '=' in body:
            target, rhs = body.split('=', 1)
            target = target.strip()
            rhs = rhs.strip()
            if '{' in target or '(' in target:
                try:
                    if self.compiler._try_let_index_assignment(
                            target, rhs, eval_scope, line_number,
                            local_vars=value_dict):
                        return
                except (NameError, ValueError, SyntaxError):
                    raise
                except Exception:
                    pass

        # 3) General variable declaration/binding: Let x as T = expr, etc.
        var, type_name, constraints, expr = self.compiler._parse_variable_def(
            body, line_number)
        init_expr = (constraints or {}).get('init')
        if expr is None and init_expr is not None:
            expr = init_expr
            constraints.pop('init', None)
        self.compiler._process_let_binding(
            var, type_name, constraints, expr, line_number,
            scope_dict=eval_scope, shadow_keyword='LET')

    def _collect_type_block_lines(self, code_lines, start_i, track_if_depth=False):
        """Collect lines for a ``... then ... end`` block.

        Returns ``(block_lines, end_index)`` where *end_index* is the index
        past the closing ``end``.
        """
        block_lines = []
        depth = 1
        scan_i = start_i + 1
        while scan_i < len(code_lines) and depth > 0:
            next_line = code_lines[scan_i].strip()
            if next_line.lower() == 'end':
                depth -= 1
                if depth == 0:
                    break
            elif re.match(r'^for\b', next_line, re.I) and re.search(r'\bdo\b', next_line, re.I):
                depth += 1
            elif re.match(r'^let\b', next_line, re.I) and re.search(r'\bthen\b', next_line, re.I):
                depth += 1
            elif track_if_depth and re.match(r'^if\b', next_line, re.I) and re.search(r'\bthen\b', next_line, re.I):
                depth += 1
            block_lines.append(next_line)
            scan_i += 1
        return block_lines, scan_i

    def _process_type_let_then_block(self, code_lines, i, value_dict, input_values, line_number):
        """Handle ``Let cond then ... End`` blocks inside type constructors."""
        header = code_lines[i].strip()
        header = re.sub(r'\s+then\s*$', '', header, flags=re.I).strip()
        header = re.sub(r'^Let\s+', '', header, count=1, flags=re.I)
        eval_scope = self._build_type_eval_scope(value_dict, {})
        var, type_name, constraints, expr = self.compiler._parse_variable_def(
            header, line_number)
        block_lines, scan_i = self._collect_type_block_lines(code_lines, i)
        condition_passed = True
        if expr is not None:
            try:
                val = self.compiler.expr_evaluator.eval_or_eval_array(
                    str(expr), eval_scope, line_number)
                condition_passed = bool(val)
            except Exception:
                condition_passed = False
        if condition_passed:
            self._execute_type_block(
                block_lines, value_dict, input_values, line_number)
        return scan_i + 1

    def _process_type_if_block(self, code_lines, i, value_dict, input_values, line_number):
        """Handle ``If cond then ... [else ...] End`` blocks inside type constructors."""
        header = code_lines[i].strip()
        header = re.sub(r'\s+then\s*$', '', header, flags=re.I).strip()
        header = re.sub(r'^if\s+', '', header, count=1, flags=re.I)
        block_lines, scan_i = self._collect_type_block_lines(
            code_lines, i, track_if_depth=True)
        # Split at else
        if_block = block_lines
        else_block = []
        for idx, line in enumerate(block_lines):
            if line.strip().lower() == 'else':
                if_block = block_lines[:idx]
                else_block = block_lines[idx + 1:]
                break
        condition_passed = True
        try:
            condition_passed = self.compiler.control_flow._evaluate_if_condition(
                header, line_number)
        except Exception:
            condition_passed = False
        chosen = if_block if condition_passed else else_block
        if chosen:
            self._execute_type_block(
                chosen, value_dict, input_values, line_number)
        return scan_i + 1

    def _with_value_matches(self, a, b):
        """True when a WITH-applied value and a constructor-computed value agree."""
        if isinstance(a, bool) or isinstance(b, bool):
            return a == b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a == b
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(
                self._with_value_matches(x, y) for x, y in zip(a, b))
        if isinstance(a, dict) and isinstance(b, dict):
            pub_a = {k: v for k, v in a.items() if not str(k).startswith('_')}
            pub_b = {k: v for k, v in b.items() if not str(k).startswith('_')}
            return pub_a.keys() == pub_b.keys() and all(
                self._with_value_matches(pub_a[k], pub_b[k]) for k in pub_a)
        if a is None or b is None:
            return a is None and b is None
        return a == b

    def _process_type_assignment(self, line, value_dict, input_values, line_number, init_fields=None, default_fields=None):
        """Handle assignments inside type definitions (e.g., x = in_x)."""
        init_fields = init_fields or set()
        default_fields = default_fields or set()

        def _coerce_field_value(field_name, raw_value):
            if not isinstance(value_dict, dict):
                return raw_value
            type_name = value_dict.get('_type_name')
            if not type_name:
                return raw_value
            type_def = self.compiler.types_defined.get(type_name.lower(), {})
            field_map = {
                str(k).lower(): k for k in type_def.keys() if not str(k).startswith('_')
            }
            field_key = field_map.get(str(field_name).lower())
            if not field_key:
                return raw_value
            field_type = type_def.get(field_key)
            if not isinstance(field_type, str):
                return raw_value
            if field_type.lower() not in self.compiler.types_defined:
                return raw_value
            if isinstance(raw_value, dict):
                return raw_value
            if isinstance(raw_value, (list, tuple)):
                return self.compiler._instantiate_type(
                    field_type, list(raw_value), line_number, allow_default_if_empty=True)
            return raw_value

        def _strip_init_copy_immutability(raw_value):
            if isinstance(raw_value, dict):
                cleaned = {}
                for key, item in raw_value.items():
                    if key == '_immutable_fields':
                        continue
                    cleaned[key] = _strip_init_copy_immutability(item)
                return cleaned
            if isinstance(raw_value, list):
                return [_strip_init_copy_immutability(item) for item in raw_value]
            if isinstance(raw_value, tuple):
                return tuple(_strip_init_copy_immutability(item) for item in raw_value)
            return raw_value

        paren_match = re.match(r'^\s*(\$?[\w_]+)\s*\(([^)]+)\)\s*=\s*(.+)$', line)
        if paren_match:
            field_name, index_expr, value_expr = paren_match.groups()
            if field_name.startswith('$'):
                field_name = field_name[1:]
            actual_field = get_case_insensitive_key(
                value_dict, field_name) or field_name
            scope = self._build_type_eval_scope(value_dict, input_values)
            index_val = self.compiler.expr_evaluator.eval_expr(
                index_expr.strip(), scope, line_number)
            if isinstance(index_val, numbers.Real):
                index_val = int(round(index_val))
            value = self.compiler.expr_evaluator.eval_or_eval_array(
                value_expr.strip(), scope, line_number)
            value = _coerce_field_value(actual_field, value)
            arr = value_dict.get(actual_field)
            if arr is None or not isinstance(arr, list):
                arr = []
            if index_val is None or index_val < 1:
                raise ValueError(
                    f"Invalid index {index_val} for '{field_name}' at line {line_number}")
            while len(arr) < index_val:
                arr.append(None)
            arr[index_val - 1] = value
            value_dict[actual_field] = arr
            return

        brace_match = re.match(r'^\s*(\$?[\w_]+)\s*\{([^}]+)\}\s*=\s*(.+)$', line)
        if brace_match:
            field_name, indices_str, value_expr = brace_match.groups()
            if field_name.startswith('$'):
                field_name = field_name[1:]
            scope = self._build_type_eval_scope(value_dict, input_values)
            if ',' in indices_str:
                # Multi-dimensional: route to the shared index-assignment utility
                target = f"{field_name}{{{indices_str}}}"
                if self.compiler._try_let_index_assignment(
                        target, value_expr.strip(), scope, line_number,
                        local_vars=value_dict):
                    return
            actual_field = get_case_insensitive_key(
                value_dict, field_name) or field_name
            indices = [idx.strip() for idx in indices_str.split(',') if idx.strip()]
            index_val = self.compiler.expr_evaluator.eval_expr(
                indices[0], scope, line_number)
            if isinstance(index_val, numbers.Real):
                index_val = int(round(index_val))
            value = self.compiler.expr_evaluator.eval_or_eval_array(
                value_expr.strip(), scope, line_number)
            value = _coerce_field_value(actual_field, value)
            arr = value_dict.get(actual_field)
            if arr is None or not isinstance(arr, list):
                arr = []
            if index_val is None or index_val < 1:
                raise ValueError(
                    f"Invalid index {index_val} for '{field_name}' at line {line_number}")
            while len(arr) < index_val:
                arr.append(None)
            arr[index_val - 1] = value
            value_dict[actual_field] = arr
            return

        match = re.match(r'^\s*(\$?[\w_]+)\s*=\s*(.+)$', line)
        if not match:
            raise SyntaxError(
                f"Unsupported syntax in type constructor: '{line}' at line {line_number}")
        field_name, value_expr = match.groups()
        if field_name.startswith('$'):
            field_name = field_name[1:]
        actual_field = get_case_insensitive_key(
            value_dict, field_name) or field_name
        scope = self._build_type_eval_scope(value_dict, input_values)
        value = self.compiler.expr_evaluator.eval_or_eval_array(
            value_expr.strip(), scope, line_number)
        applied_fields = value_dict.get('_with_applied_fields')
        applied_fields = applied_fields if isinstance(applied_fields, (set, list, tuple)) else set()
        if (actual_field.lower() in applied_fields
                and actual_field in value_dict
                and value_dict[actual_field] is not None):
            if actual_field.lower() in default_fields:
                # ``or =`` provides a default only when no value has been set
                # for the variable yet (e.g. by a ``with`` clause). Keep the
                # applied value.
                return
            # A ``with`` clause value acts as a constraint on the field: the
            # constructor still computes its value, but a disagreement with the
            # applied value makes the whole constructed value a #VALUE error.
            if not self._with_value_matches(value, value_dict[actual_field]):
                value_dict['_with_conflict'] = True
                return
        if actual_field.lower() in init_fields:
            try:
                value = copy.deepcopy(value)
            except Exception:
                pass
            value = _strip_init_copy_immutability(value)
        value_dict[actual_field] = value
        if input_values:
            tokens = re.findall(r'\b[\w_]+\b', value_expr)
            input_names = {name.lower() for name in input_values.keys()}
            if any(tok.lower() in input_names for tok in tokens):
                immutable_fields = value_dict.setdefault(
                    '_immutable_fields', set())
                immutable_fields.add(actual_field.lower())

    def _build_type_eval_scope(self, value_dict, input_values):
        scope = self.compiler.current_scope().get_full_scope()
        scope = dict(scope) if isinstance(scope, dict) else {}
        if input_values:
            scope.update(input_values)
        if isinstance(value_dict, dict):
            for key, val in value_dict.items():
                if not str(key).startswith('_'):
                    scope[key] = val
            if 'grid' in value_dict:
                scope['grid'] = value_dict.get('grid', {})
        return scope

    def _execute_builder(self, type_name, builder_name, value_dict, line_number, arg_values):
        if not type_name:
            raise NameError(
                f"Builder '{builder_name}' has no type context at line {line_number}")
        type_def = self.compiler.types_defined.get(type_name.lower(), {})
        if builder_name.lower() == 'super':
            parent = type_def.get('_parent')
            if not parent:
                raise NameError(
                    f"Super() called but type '{type_name}' has no parent at line {line_number}")
            parent_obj = self.compiler._instantiate_type(
                parent, arg_values, line_number, allow_default_if_empty=True)
            for key, val in parent_obj.items():
                if key == '_type_name':
                    continue
                if key == '_hidden_fields':
                    hidden = set(value_dict.get('_hidden_fields', set()))
                    hidden.update(val or [])
                    value_dict['_hidden_fields'] = hidden
                    continue
                if key == '_immutable_fields':
                    imm = set(value_dict.get('_immutable_fields', set()))
                    imm.update(val or [])
                    value_dict['_immutable_fields'] = imm
                    continue
                if key == 'grid' and isinstance(val, dict):
                    grid_store = value_dict.get('grid')
                    if not isinstance(grid_store, dict):
                        grid_store = {}
                        value_dict['grid'] = grid_store
                    grid_store.update(val)
                    continue
                value_dict[key] = val
            return

        helpers = type_def.get('_builders', {}) or {}
        helper_def = helpers.get(builder_name.lower())
        if not helper_def:
            raise NameError(
                f"Builder '{builder_name}' not defined for type '{type_name}' at line {line_number}")
        input_defs = helper_def.get('input_defs', [])
        input_values = {}
        if input_defs:
            if len(arg_values) > len(input_defs):
                raise ValueError(
                    f"Too many arguments for builder '{builder_name}' at line {line_number}")
            for idx, entry in enumerate(input_defs):
                name = entry.get('name')
                if idx < len(arg_values):
                    input_values[name] = arg_values[idx]
                else:
                    default_expr = entry.get('constraints', {}).get('default') or entry.get('default')
                    if default_expr is None:
                        raise ValueError(
                            f"Missing argument '{name}' for builder '{builder_name}' at line {line_number}")
                    eval_scope = self._build_type_eval_scope(
                        value_dict, input_values)
                    input_values[name] = self.compiler.expr_evaluator.eval_or_eval_array(
                        str(default_expr), eval_scope, line_number)
        elif arg_values:
            raise ValueError(
                f"Builder '{builder_name}' does not take arguments at line {line_number}")

        code_lines = helper_def.get('code_lines') or helper_def.get('code', '').splitlines()
        member_keys = type_def.get('_member_keys', set())
        saved_non_member = {}
        for key in list(value_dict.keys()):
            if (not str(key).startswith('_')
                    and str(key).lower() not in member_keys
                    and key != 'grid'):
                saved_non_member[key] = value_dict.pop(key)
        saved_scopes = self.compiler.scopes[:]
        self.compiler.scopes = [self.compiler.scopes[0]]
        try:
            self._execute_type_code(
                code_lines, 'this', value_dict, line_number, input_values)
        finally:
            value_dict.update(saved_non_member)
            self.compiler.scopes = saved_scopes

    def _parse_builder_chain(self, chain_text, line_number):
        """Parse a '-> builder(args) -> ...' suffix into (name, args_text) pairs."""
        entries = []
        rest = chain_text.strip()
        while rest:
            rest = rest.lstrip()
            if not rest.startswith('->'):
                raise SyntaxError(
                    f"Unexpected tokens in builder chain: '{rest}' at line {line_number}")
            rest = rest[2:].lstrip()
            m = re.match(r'(\$?[A-Za-z][\w.]*)\s*\(', rest)
            if not m:
                raise SyntaxError(
                    f"Invalid builder call in chain at line {line_number}")
            builder_name = m.group(1).lstrip('$')
            open_pos = rest.index('(', m.start(1))
            in_quote = None
            depth = 0
            close_pos = None
            for i in range(open_pos, len(rest)):
                ch = rest[i]
                if in_quote:
                    if ch == in_quote and (i == 0 or rest[i - 1] != '\\'):
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
                        close_pos = i
                        break
            if close_pos is None:
                raise SyntaxError(
                    f"Unclosed builder call '{builder_name}' at line {line_number}")
            entries.append((builder_name, rest[open_pos + 1:close_pos]))
            rest = rest[close_pos + 1:].strip()
        return entries

    def _split_builder_args(self, args_text, line_number):
        args = []
        current = ""
        nest = []
        in_quote = None
        for ch in args_text + ',':
            if in_quote:
                current += ch
                if ch == in_quote and current[-2:-1] != '\\':
                    in_quote = None
                continue
            if ch in ('"', "'"):
                in_quote = ch
                current += ch
                continue
            if ch in ('(', '{', '['):
                nest.append(ch)
                current += ch
                continue
            if ch in (')', '}', ']'):
                if nest:
                    nest.pop()
                current += ch
                continue
            if ch == ',' and not nest:
                if current.strip():
                    args.append(current.strip())
                current = ""
                continue
            current += ch
        return args

    def _apply_builder_chain(self, instance, chain_text, scope, line_number):
        """Apply a '-> builder(args) ...' chain to a freshly-constructed instance.

        Builders only run during construction: either inside a type's own
        constructor, or in a chain that follows 'new' directly. They mutate
        the in-progress object and the final chain result is what gets bound.
        """
        if not isinstance(instance, dict):
            raise TypeError(
                f"Builder chains can only apply to object instances at line {line_number}")
        type_name = instance.get('_type_name')
        if not type_name:
            raise TypeError(
                f"Cannot determine the type for the builder chain at line {line_number}")
        type_def = self.compiler.types_defined.get(type_name.lower(), {})
        builders = type_def.get('_builders', {}) if isinstance(type_def, dict) else {}
        entries = self._parse_builder_chain(chain_text, line_number)
        for builder_name, args_text in entries:
            helper_entry = builders.get(builder_name.lower())
            if helper_entry is None:
                raise NameError(
                    f"Builder '{builder_name}' not defined for type '{type_name}' at line {line_number}")
            if helper_entry.get('hidden') and not getattr(
                    self.compiler, '_allow_hidden_member_calls', False):
                raise PermissionError(
                    f"Builder '{builder_name}' of type '{type_name}' is private and cannot be called here at line {line_number}")
            args = self._split_builder_args(args_text, line_number)
            eval_scope = scope if isinstance(scope, dict) else (
                self.compiler.current_scope().get_full_scope() if hasattr(
                    self.compiler, 'current_scope') else {})
            arg_values = [self.compiler.expr_evaluator.eval_or_eval_array(
                a, eval_scope, line_number) for a in args]
            self._execute_builder(
                type_name, builder_name, instance, line_number, arg_values)
        return instance
