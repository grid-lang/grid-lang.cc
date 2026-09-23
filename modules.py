"""Module parsing and load-time policy for GridLang modules.

This module owns the module *policy*: how a module file's header, version and
API-version (``Version``) blocks are parsed, how a ``use`` / ``For B use``
statement is understood, and how export names are stripped of their version
tag. It imports no engine modules, so the compiler simply hands over source
lines and gets back structured metadata.

The shape here follows ``permissions.py``: the feature module keeps the pure
parsing/comparison logic and the engine wires it into execution.
"""

import os
import re


class ModuleImportError(SyntaxError):
    """Raised when a module cannot be loaded or its exports cannot be bound."""


_MODULE_HEADER_RE = re.compile(
    r'^\s*Module\s+([A-Za-z_][A-Za-z0-9_]*)(.*)$', re.I)
_VERSION_LINE_RE = re.compile(
    r'^\s*:\s*version\s*=\s*(.*)$', re.I)
_VERSION_BLOCK_RE = re.compile(
    r'^\s*Version\s+([A-Za-z_][A-Za-z0-9_]*)\s+exports\s+(.+)$', re.I)
_USE_RE = re.compile(
    r'^\s*(?:for\s+([A-Za-z_][A-Za-z0-9_]*)\s+)?'
    r'use\s+([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)'
    r'((?:\s+with\s*\(.*?\))?)\s*$', re.I)
_PIN_RE = re.compile(r'\s*version\s*([<>=]+)\s*(.+)$', re.I)
_RE_NUMBER = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$', re.I)


def parse_module_header(line):
    """Parse a ``Module <name> [runnable] [shared]`` header line.

    Returns a dict with 'name', 'runnable' and 'shared' keys, or None when
    the line is not a module header.
    """
    m = _MODULE_HEADER_RE.match(line)
    if not m:
        return None
    rest = m.group(2) or ''
    return {
        'name': m.group(1),
        'runnable': bool(re.search(r'\brunnable\b', rest, re.I)),
        'shared': bool(re.search(r'\bshared\b', rest, re.I)),
    }


def parse_module_version_line(line):
    """Parse a ``: version = <value>`` line.

    Returns a dict with 'kind' ('text' / 'number') and 'value' (unquoted
    string or float), or None when the line is not a version line.
    """
    m = _VERSION_LINE_RE.match(line)
    if not m:
        return None
    raw = (m.group(1) or '').strip()
    if not raw:
        return {'kind': 'text', 'value': ''}
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return {'kind': 'text', 'value': raw[1:-1]}
    if _RE_NUMBER.match(raw):
        try:
            return {'kind': 'number', 'value': float(raw)}
        except ValueError:
            pass
    return {'kind': 'text', 'value': raw}


def parse_version_block(line):
    """Parse a ``Version <tag> exports <name, ...>`` block line.

    Returns a dict with 'tag' and 'exports' (list of export names), or None.
    """
    m = _VERSION_BLOCK_RE.match(line)
    if not m:
        return None
    exports = [n.strip() for n in m.group(2).split(',') if n.strip()]
    return {'tag': m.group(1), 'exports': exports}


def strip_version_tag(name, tag):
    """Apply the §4 strip rule: a definition whose name ends with
    ``_<tag>`` is exported stripped (``bar_v1`` → ``bar`` under ``v1``).

    Returns the stripped export name and a bool telling whether stripping
    happened. Names not ending with the tag are returned unchanged.
    """
    if not tag:
        return name, False
    suffix = '_' + tag
    if name.lower().endswith(suffix.lower()):
        return name[:len(name) - len(suffix)], True
    return name, False


def parse_use_line(line):
    """Parse a ``use`` / ``For B use`` import statement.

    Returns a dict with keys:
      - namespace: bound namespace name or None (flat import),
      - module:    module name,
      - tag:       API version tag,
      - pin_op:    '=' / '>=' / ... or None,
      - pin_value: raw pin expression or None,
    or None when the line is not a use statement.
    """
    m = _USE_RE.match(line)
    if not m:
        return None
    namespace, module, tag, with_clause = m.groups()
    pin_op = None
    pin_value = None
    if with_clause:
        pm = _PIN_RE.search(with_clause)
        if not pm:
            raise ModuleImportError(
                f"Invalid version pin in '{line.strip()}'; "
                "expected 'with (version<op>value)'")
        pin_op = pm.group(1).strip()
        pin_value = pm.group(2).strip().rstrip(')').strip()
    return {
        'namespace': namespace,
        'module': module,
        'tag': tag,
        'pin_op': pin_op,
        'pin_value': pin_value,
    }


def parse_version_value(raw):
    """Parse a literal version value (quoted text or number) into a
    ``{kind, value}`` dict, mirroring :func:`parse_module_version_line`.
    """
    return parse_module_version_line(f': version = {raw}')


def version_pin_matches(pin, module_version, line_number=None):
    """Check that a module's declared ``: version`` satisfies a ``use`` pin.

    - Text versions: only ``=`` (strict equality) is meaningful.
    - Number versions: ``=`` and ``>=``.
    Raises ``ModuleImportError`` for unsupported operators/kinds.
    """
    if pin is None:
        return True
    pin_op = pin.get('pin_op')
    pin_entry = parse_version_value(pin.get('pin_value') or '')
    if module_version is None:
        raise ModuleImportError(
            f"Version pin 'version{pin_op}...' requested but the module "
            "declares no ': version'")
    kind = module_version.get('kind', 'text')
    if kind == 'text' and pin_op not in ('=',):
        raise ModuleImportError(
            f"Text module versions support only '=' pins, got 'version{pin_op}'")
    if kind == 'number' and pin_op not in ('=', '>='):
        raise ModuleImportError(
            f"Number module versions support only '=' and '>=' pins, got "
            f"'version{pin_op}'")
    if kind != pin_entry.get('kind'):
        # e.g. pin compares a text module version to a number literal: only
        # exact string equality could ever hold; treat mismatched kinds as no.
        if pin_op == '=':
            return False
        raise ModuleImportError(
            f"Version pin compares a {kind} module version to a "
            f"{pin_entry.get('kind')} literal")
    if kind == 'number':
        mod_val = float(module_version.get('value', 0))
        pin_val = float(pin_entry.get('value', 0))
        if pin_op == '=':
            return mod_val == pin_val
        return mod_val >= pin_val
    return module_version.get('value') == pin_entry.get('value')


def module_search_paths():
    """Directories searched for ``<name>.grid`` module files, in order."""
    paths = [os.getcwd()]
    grid_path = os.environ.get('GRID_PATH', '')
    if grid_path:
        paths.extend(p for p in grid_path.split(os.pathsep) if p)
    return paths


def resolve_module_source(name, registry=None):
    """Locate a module's source text by name.

    Checks the in-memory registry (``{name(lower): source}``) first, then
    each directory in :func:`module_search_paths` for ``<name>.grid``.
    Returns the source text or raises ``ModuleImportError``.
    """
    key = name.lower()
    if registry:
        src = registry.get(key)
        if src is not None:
            return src
    for directory in module_search_paths():
        candidate = os.path.join(directory, f"{name}.grid")
        if os.path.isfile(candidate):
            try:
                with open(candidate, 'r', encoding='utf-8') as fh:
                    return fh.read()
            except OSError as exc:
                raise ModuleImportError(
                    f"Cannot read module file '{candidate}': {exc}")
    raise ModuleImportError(f"Module '{name}' not found")