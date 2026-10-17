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
    r'^\s*(?:(?:for|let)\s+|:\s*)?'
    r'([A-Za-z_][A-Za-z0-9_]*\s+)?'
    r'(?:as\s+([A-Za-z_][A-Za-z0-9_]*)\s+)?'
    r'use\s+([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?'
    r'((?:\s+with\s*\(.*\))?)\s*$', re.I)
_USE_LEAD_RE = re.compile(r'^\s*(?:(?:for|let)\s+|:\s*)', re.I)
_RE_NUMBER = re.compile(
    r'^[+-]?(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)'
    r'(?:e[+-]?\d(?:_?\d)*)?$', re.I)


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


# Comparison operators a pin entry may carry; the with-clause parser puts
# them into the field constraint dict, a plain ``=`` assignment comes back
# as the unevaluated value text instead.
_PIN_OPS = ('>=', '<=', '<>', '<', '>', '=')


def parse_use_line(line, parser=None, line_number=None):
    """Parse a ``use`` import statement.

    The instruction lead and namespace stay in sync across ``For``, ``Let``
    and ``:``; all three accept ``<lead> <ns> [as ModuleVersion] use M.v``,
    and a bare ``use M.v`` performs a flat import. A lead without a namespace
    (``For use M.v``) is not an import statement. The API version tag is
    optional: a bare ``use M`` (no tag) imports a *runnable* module and
    exposes it as the module-run subprocess ``M`` (see docs §7). A
    versionless import may pin any top-level GV — the pin constrains which
    module build is loaded and binds nothing (docs §5).

    A ``with (...)`` clause may combine several GV pins, comma-separated;
    every pin must hold (docs §5). Pins are read with the shared
    with-constraint machinery (``parser`` supplies
    ``_split_with_clause_parts`` and ``_parse_with_field_spec``), so a pin
    entry is an ordinary ``<gv><op>value`` field spec.

    Returns a dict with keys:
      - namespace: bound namespace name or None (flat import),
      - module:    module name,
      - tag:       API version tag or None (versionless runnable import),
      - pins:      list of {'pin_name', 'pin_op', 'pin_value'} (empty list),
      - pin_name / pin_op / pin_value: the first pin, or None (mirrored
        for single-pin consumers),
    or None when the line is not a use statement.
    """
    m = _USE_RE.match(line)
    if not m:
        return None
    ns, as_name, module, tag, with_clause = m.groups()
    namespace = ns.strip() if ns else None
    if namespace is None and _USE_LEAD_RE.match(line):
        return None
    if as_name is not None and as_name.lower() != 'moduleversion':
        raise ModuleImportError(
            f"Invalid module import type in '{line.strip()}'; "
            "only 'as ModuleVersion' is supported")
    invalid = ModuleImportError(
        f"Invalid pin in '{line.strip()}'; "
        "expected 'with (<name><op>value)'")
    pins = []
    if with_clause:
        if parser is None:
            raise invalid
        span = parser._find_with_clause_balanced(line)
        if span is None:
            raise invalid
        parts = parser._split_with_clause_parts(line[span[1] + 1:span[2]])
        for part in parts:
            try:
                spec = parser._parse_with_field_spec(part, line_number)
            except SyntaxError:
                spec = None
            if spec is None:
                raise invalid
            field, value, field_spec = spec
            op, op_value = None, None
            if value is not None:
                op, op_value = '=', value
            else:
                for candidate in _PIN_OPS:
                    if candidate in field_spec:
                        op, op_value = candidate, str(field_spec[candidate])
                        break
            if op is None:
                raise invalid
            pins.append({
                'pin_name': field,
                'pin_op': op,
                'pin_value': op_value,
            })
        if not pins:
            raise invalid
    if namespace and tag is None:
        raise ModuleImportError(
            f"Versionless 'use {module}' in '{line.strip()}' cannot bind a "
            "namespace; use '<ns> use M.<tag>'")
    first = pins[0] if pins else {}
    return {
        'namespace': namespace,
        'module': module,
        'tag': tag,
        'pins': pins,
        'pin_name': first.get('pin_name'),
        'pin_op': first.get('pin_op'),
        'pin_value': first.get('pin_value'),
    }


def parse_version_value(raw):
    """Parse a literal version value (quoted text or number) into a
    ``{kind, value}`` dict, mirroring :func:`parse_module_version_line`.
    """
    return parse_module_version_line(f': version = {raw}')


def pin_target_from_module_vars(module_vars, name):
    """Resolve a pin target GV to a ``{kind, value}`` entry (or None).

    The target is any of the module's top-level GVs (its instance state,
    excluding Input/Output); values are the engine-evaluated ones, so text
    comes back unquoted and numbers as floats.
    """
    meta = (module_vars or {}).get((name or '').lower())
    if not meta or meta.get('uninitialized'):
        return None
    value = meta.get('value')
    if isinstance(value, str):
        return {'kind': 'text', 'value': value}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {'kind': 'number', 'value': float(value)}
    return None


def version_pin_matches(pin, module_version, line_number=None):
    """Check that a module GV satisfies a ``use`` pin.

    - Text values: only ``=`` (strict equality) is meaningful.
    - Number values: ``=`` and ``>=``.
    Raises ``ModuleImportError`` for unsupported operators/kinds.
    """
    if pin is None:
        return True
    pin_op = pin.get('pin_op')
    pin_name = pin['pin_name']
    pin_value = pin.get('pin_value') or ''
    pin_entry = parse_version_value(pin.get('pin_value') or '')
    if module_version is None:
        raise ModuleImportError(
            f"Pin '{pin_name}{pin_op}{pin_value}' requested but the module "
            f"defines no '{pin_name}'")
    kind = module_version.get('kind', 'text')
    if kind == 'text' and pin_op not in ('=',):
        raise ModuleImportError(
            f"Text pin targets support only '=' pins, "
            f"got '{pin_name}{pin_op}'")
    if kind == 'number' and pin_op not in ('=', '>='):
        raise ModuleImportError(
            f"Number pin targets support only '=' and '>=' pins, got "
            f"'{pin_name}{pin_op}'")
    if kind != pin_entry.get('kind'):
        # e.g. pin compares a text GV to a number literal: only exact string
        # equality could ever hold; treat mismatched kinds as no.
        if pin_op == '=':
            return False
        raise ModuleImportError(
            f"Pin '{pin_name}{pin_op}...' compares a {kind} module value "
            f"to a {pin_entry.get('kind')} literal")
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