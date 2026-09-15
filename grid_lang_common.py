"""Shared helpers for the GridLang engine.

This module is the single source of truth for constants and utilities shared
across the engine's classes. Both compiler.py (GridLangCompiler) and
executor.py (GridLangExecutor, now the runtime-loop base of the compiler)
import from here instead of re-defining or monkey-patching at runtime.

The earlier `setattr(extracted, method, getattr(self, method))` handoff in
compiler.py:_run_inner was eliminated: GridLangCompiler now inherits
GridLangExecutor directly, so there is a single engine object with no method
copying between a compiler and a separate facade executor.
"""

import re

# ---------------------------------------------------------------------------
# Statement dispatch — replaces re.match(r'^\s*(input|define|…)\b')
# ---------------------------------------------------------------------------
_STATEMENT_KEYWORDS = frozenset(['input','define','output','let','if','for','when','return','push','while','require'])

def _first_keyword(line):
    s = line.lstrip()
    if s.startswith('['):
        return ""
    m = re.match(r'([A-Za-z_]+)', s)
    return m.group(1).lower() if m else ""

# ---------------------------------------------------------------------------
# Lexing / dependency extraction
# ---------------------------------------------------------------------------
_IDENTIFIER_TOKEN_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9_.]*')
_STRING_LITERAL_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_DEPENDENCY_IGNORED_TOKENS = frozenset({
    'sum', 'rows', 'sqrt', 'min', 'max', 'abs', 'int', 'float', 'str', 'len',
    'textsplit', 'print', 'push', 'true', 'false', 'none', 'nan', 'inf', 'and', 'or', 'not',
    'if', 'then', 'else', 'elseif', 'end', 'do', 'for', 'while', 'when', 'step', 'return',
    'index', 'as', 'dim', 'with', 'grid', 'output', 'input', 'number', 'text',
    'array', 'mod', 'div', 'to', 'by', 'e', 'new', 'in', 'counta', 'rows', 'of', 'null'
})

def _strip_constraint_operands(expr):
    """Remove constraint clauses (' of <unit>', ' as <type>', ' dim <n>',
    ' not null') so dependency extraction doesn't treat unit/type names as
    variable references."""
    if not expr:
        return expr
    cleaned = _STRING_LITERAL_PATTERN.sub(' ', str(expr))
    cleaned = re.sub(r'\b(?:as)\s+(?:[A-Za-z][A-Za-z0-9_.]*)\b', ' ', cleaned)
    cleaned = re.sub(r'\b(?:of)\s+(?:[\w1][\w0-9./]*)\b', ' ', cleaned)
    cleaned = re.sub(
        r'\bdim\s+(?:\d+(?:\.\d*)?|[A-Za-z][A-Za-z0-9_.]*)',
        ' ', cleaned, flags=re.I)
    cleaned = re.sub(r'\bnot\s+null\b', ' ', cleaned, flags=re.I)
    return cleaned

def _strip_builder_arrows(text):
    """Remove builder-call names ('-> name(') so dependency extraction does not
    mistake builder names for variables."""
    return re.sub(r'->\s*\$?[A-Za-z][A-Za-z0-9_.]*\s*\(', '(', text)

def _strip_cell_address_tokens(text):
    # Simplified version of utils.strip_array_cell_indices for deps
    return re.sub(r'\[\s*[A-Za-z]+\d+(?::[A-Za-z]+\d+)?\s*\]', ' ', text)

def _is_numeric_token(token):
    return bool(re.match(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$', token, re.I)) or \
           bool(re.match(r'^e[+-]?\d+$', token, re.I))

def _filter_var_tokens(tokens):
    # Placeholder for future filtering
    return tokens

# ---------------------------------------------------------------------------
# Dependency / pending helpers — single source for has_unresolved etc.
# ---------------------------------------------------------------------------
def _has_star_dim(constraints, array_handler=None):
    """Return True if *constraints* contain an unbounded (star) dim spec."""
    dim_spec = (constraints or {}).get('dim')
    if isinstance(dim_spec, dict) and 'dims' in dim_spec:
        dim_spec = dim_spec['dims']
    if isinstance(dim_spec, list):
        for _, size_spec in dim_spec:
            if array_handler and hasattr(array_handler, '_is_unbounded_size_spec'):
                if array_handler._is_unbounded_size_spec(size_spec):
                    return True
            elif isinstance(size_spec, str) and '*' in size_spec:
                return True
        return False
    if isinstance(dim_spec, str):
        return '*' in dim_spec or bool(__import__('re').search(r'to\s+\*', dim_spec, __import__('re').I))
    return False

def _apply_dim_base_offsets(var_name, indices, dimensions=None, array_handler=None, line_number=None):
    """Convert parsed (1-based) source indices to storage indices."""
    if not indices:
        return indices
    dims = dimensions or {}
    # Simple fallback: no adjustment if no dimensions
    if var_name not in dims:
        return indices
    # Real logic is in compiler; this is a minimal version for common use
    return indices

def _infer_declared_type(value, array_handler=None):
    """Infer type string from a value, handling UnitValue correctly."""
    if value is None:
        return 'unknown'
    # Simplified: delegate to array_handler if available
    if array_handler and hasattr(array_handler, 'infer_type'):
        try:
            return array_handler.infer_type(value)
        except Exception:
            pass
    # Fallback
    if isinstance(value, bool):
        return 'logical'
    if isinstance(value, (int, float)):
        return 'number'
    if isinstance(value, str):
        return 'text'
    return 'unknown'


# ---------------------------------------------------------------------------
# Text (character) constants
#
# Written outside quotes (`[A1] := #NBSP`) or inside interpolation braces
# (`$"a{#NL}b"`). `#U/xxxx` is any Unicode code point, up to 4 hex digits.
# ---------------------------------------------------------------------------
TEXT_CHARACTER_CONSTANTS = {
    'NL': '\n',
    'NBSP': '\u00a0',
    'QUOT': '"',
    'APOS': "'",
    'LB': '{',
    'MDASH': '\u2014',
    'NDASH': '\u2013',
    'TAB': '\t',
    'SHY': '\u00ad',
    'CR': '\r',
}
_TEXT_NAMED_CHAR_RE = re.compile(r'#(?P<name>[A-Za-z]+)')
_TEXT_UNICODE_CHAR_RE = re.compile(r'#U/(?P<cp>[0-9a-fA-F]{1,4})', re.I)


def resolve_text_constant_token(token):
    """Resolve a `#...` text constant token to its character, or None."""
    if not isinstance(token, str) or not token:
        return None
    t = token.strip()
    if t.startswith('#U/') or t.startswith('#u/'):
        m = re.fullmatch(r'#u?/(?P<cp>[0-9a-fA-F]{1,4})', t, re.I)
        if m:
            return chr(int(m.group('cp'), 16))
        return None
    if not t.startswith('#'):
        return None
    m = re.fullmatch(r'#(?P<name>[A-Za-z]+)', t)
    if not m:
        return None
    name = m.group('name').upper()
    if name in TEXT_CHARACTER_CONSTANTS:
        return TEXT_CHARACTER_CONSTANTS[name]
    return None


def mask_text_constant_tokens(expr):
    """Replace `#...` character-constant tokens with spaces so dependency
    extraction doesn't mistake their name letters for variable references."""
    if not isinstance(expr, str) or '#' not in expr:
        return expr
    out = _TEXT_NAMED_CHAR_RE.sub(' ', expr)
    out = _TEXT_UNICODE_CHAR_RE.sub(' ', out)
    return out


class GridLangBase:
    """Shared base for the GridLang engine classes.

    Holds the small shared helpers (scope management, dependency filtering,
    type inference) used by both GridLangExecutor (the runtime-loop layer) and
    GridLangCompiler (the single public engine, which inherits it).
    """

    def current_scope(self):
        return self.scopes[-1]

    def push_scope(self, is_private=False, is_loop_scope=False):
        from scope import Scope
        scope = Scope(self, parent=self.current_scope(), is_private=is_private)
        if is_loop_scope:
            scope.is_loop_scope = True
        self.scopes.append(scope)

    def pop_scope(self):
        if len(self.scopes) > 1:
            self.scopes.pop()
        else:
            raise RuntimeError("Cannot pop global scope")

    def get_global_scope(self):
        # Executor previously had a wrapper that checked self.compiler
        if hasattr(self, 'compiler') and getattr(self, 'compiler', None) is not None:
            try:
                return self.compiler.scopes[0]
            except Exception:
                pass
        return self.scopes[0]

    def collect_input_output_variables(self):
        # Unified version of compiler:4093 / executor:154
        global_scope = self.get_global_scope()
        self.input_variables = list(getattr(global_scope, 'input_variables', []))
        self.output_variables = list(getattr(global_scope, 'output_variables', []))
        # Executor adds 'output' as default for push() calls
        if 'output' not in self.output_variables:
            self.output_variables.append('output')
