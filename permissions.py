"""Capability permission resolution and UI: grant/deny YAML, the interactive
prompt, and the required-capability resolution stage.

This module owns the permission *policy*: how requested parameter expressions
are evaluated, how a chosen grant is merged over them, how parameters are
validated against resource field constraints, and whether a capability is
granted, denied (sticky #PERM), or prompted for. It imports no engine modules,
so the executor/a compiler simply hands over its ``Requirements`` entries and
host services (``expr_evaluator`` / ``array_handler`` / ``current_scope``).
"""

import os
import sys

from units import GrantError


# ---------------------------------------------------------------------------
# Minimal YAML support for the grant/list-required format (no external deps).
# ---------------------------------------------------------------------------

def format_yaml_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_yaml_scalar(v) for v in value) + "]"
    text = str(value)
    ambiguous = (
        text == ""
        or text.lower() in ("true", "false", "null", "none", "~", "yes", "no")
        or text[0] in "{}[],&*#!|>'\"%@`"
        or parse_yaml_scalar(text) != text
    )
    if ambiguous:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def parse_yaml_scalar(value):
    """Parse a YAML scalar of our subset (returns a distinct 'unparsed' kind so
    string round-tripping is reliable for the writer)."""
    v = value.strip()
    if v == "":
        return None
    low = v.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [parse_yaml_scalar(x) for x in inner.split(",")]
    try:
        num = float(v)
        return int(num) if num.is_integer() else num
    except ValueError:
        return v


def format_required_yaml(requirements):
    """Render the required-capability list as YAML suitable for --grant."""
    lines = ["# GridLang required capabilities",
             "# Edit `params` (or add 'denied: true') and pass this file to --grant.",
             "capabilities:"]
    for entry in requirements:
        lines.append(f"- name: {format_yaml_scalar(entry['name'])}")
        lines.append(f"  resource: {format_yaml_scalar(entry['resource'])}")
        params = entry.get('params_evaluated') or {}
        if params:
            lines.append("  params:")
            for field in sorted(params):
                lines.append(f"    {field}: {format_yaml_scalar(params[field])}")
        type_def = entry.get('type_def') or {}
        field_constraints = type_def.get('_field_constraints') or {}
        choices = {}
        for field in sorted(type_def.get('_member_keys', set()) -
                        type_def.get('_hidden_fields', set())):
            cons = field_constraints.get(str(field).lower()) or {}
            if cons.get('in'):
                choices[str(field)] = list(cons['in'])
        if choices:
            lines.append("  choices:")
            for field, values in choices.items():
                lines.append(f"    {field}: {format_yaml_scalar(values)}")
    return "\n".join(lines) + "\n"


def parse_grant_yaml(text, source_name):
    """Parse the grant YAML subset into entries:
    [{name, resource?, params?, denied?}]. Unknown fields raise GrantError."""
    entries = []
    current = None
    current_map = None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            if not stripped.startswith("- "):
                if stripped.lower() in ("capabilities:", "requirements:"):
                    continue
                raise GrantError(
                    f"{source_name}: expected a list entry starting with '- ', "
                    f"got '{stripped}'")
            body = stripped[2:].strip()
            key, sep, val = body.partition(":")
            current = {key.strip().lower(): parse_yaml_scalar(val)}
            entries.append(current)
            current_map = None
            if not sep:
                raise GrantError(
                    f"{source_name}: entry must have a 'name:' field, got '{body}'")
            if key.strip().lower() in ("params", "parameters"):
                current_map = "params"
            continue
        if current is None:
            raise GrantError(
                f"{source_name}: unexpected indented line '{stripped}'")
        if indent == 2:
            key, sep, val = stripped.partition(":")
            if not sep:
                raise GrantError(
                    f"{source_name}: expected 'key: value' at line '{stripped}'")
            k = key.strip().lower()
            if k in ("params", "parameters"):
                current_map = "params"
                if val.strip():
                    current.setdefault("params", {})
                continue
            if k in ("name", "resource", "choices", "denied", "granted"):
                current_map = None
                current[k] = parse_yaml_scalar(val)
                continue
            raise GrantError(
                f"{source_name}: unknown field '{key.strip()}' in capability "
                f"entry (allowed fields: name, resource, params, choices, denied)")
        elif indent == 4:
            if current_map != "params":
                raise GrantError(
                    f"{source_name}: unexpected indentation at '{stripped}'")
            key, sep, val = stripped.partition(":")
            if not sep:
                raise GrantError(
                    f"{source_name}: expected 'param: value' at '{stripped}'")
            params = current.setdefault("params", {})
            params[key.strip()] = parse_yaml_scalar(val)
        else:
            raise GrantError(
                f"{source_name}: unexpected indentation in '{line}'")
    return entries


def load_grants(spec):
    """Load a granted-capabilities mapping from a file or inline YAML list.

    Returns {name(lower): params-dict-or-None}; presence means granted, a
    params mapping means granted with those parameters, None means denied.
    """
    if os.path.exists(spec) and os.path.isfile(spec):
        try:
            with open(spec, "r") as fh:
                text = fh.read()
        except OSError as exc:
            raise GrantError(f"Could not read grant file '{spec}': {exc}")
        source_name = spec
    else:
        text = spec
        source_name = "--grant inline"
    lookahead = text.lstrip()
    if not (lookahead.startswith("- ") or lookahead.lower().startswith("-name")):
        before_name = lookahead.split("- ", 1)[0]
        raise GrantError(
            f"{source_name}: the grant list must be a YAML sequence of entries "
            f"starting with '- name: ...', got: {before_name.strip()[:60] or '(empty)'}")
    entries = parse_grant_yaml(text, source_name)
    grants = {}
    for entry in entries:
        name = entry.get("name")
        if name is None:
            raise GrantError(
                f"{source_name}: every grant entry needs a 'name' field")
        key = str(name).strip().lower()
        if not key:
            raise GrantError(f"{source_name}: grant entry has an empty 'name'")
        if key in grants:
            raise GrantError(
                f"{source_name}: duplicate grant entry for '{name}'")
        if entry.get("denied") is True or entry.get("granted") is False:
            grants[key] = None
        else:
            grants[key] = entry.get("params") or {}
    return grants


# ---------------------------------------------------------------------------
# Interactive grant prompt (pure formatting/parsing; the executor drives it).
# ---------------------------------------------------------------------------

def format_param_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple, dict)):
        return format_yaml_scalar(value)
    return str(value)


def render_require_prompt(entry):
    """Build the text describing one required capability for the prompt."""
    resource = entry['resource']
    type_def = entry['type_def']
    field_constraints = type_def.get('_field_constraints') or {}
    requested = entry.get('params_evaluated') or {}
    summary = []
    for field in sorted(type_def.get('_member_keys', set()) -
                    type_def.get('_hidden_fields', set())):
        key = str(field).lower()
        cons = field_constraints.get(key) or {}
        line = str(field)
        if 'in' in cons:
            line += f" (allowed: {', '.join(str(c) for c in cons['in'])})"
        if key in requested:
            line += f" = {format_param_value(requested[key])}"
        summary.append(line)
    if summary:
        return f"\n{entry['name']} requires {resource} with {', '.join(summary)}\n"
    return f"\n{entry['name']} requires {resource}\n"


def parse_grant_value(text):
    """Parse a scalar/array literal from a grant clause into a Python value."""
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    low = text.lower()
    if low == 'true':
        return True
    if low == 'false':
        return False
    if low in ('null', 'none'):
        return None
    if text.startswith('{') and text.endswith('}'):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [parse_grant_value(p) for p in _split_top_level(inner)]
    try:
        num = float(text)
        return int(num) if num.is_integer() else num
    except ValueError:
        return text


def _split_top_level(text):
    """Split text on top-level commas, ignoring quotes and nested brackets."""
    parts, depth, in_str, cur = [], 0, None, []
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str is not None:
            cur.append(ch)
            if ch == in_str and text[i - 1:i] != '\\':
                in_str = None
        elif ch in '\'"':
            in_str = ch
            cur.append(ch)
        elif ch in '{[(<':
            depth += 1
            cur.append(ch)
        elif ch in '}])>':
            depth -= 1
            cur.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append(''.join(cur))
    return parts


def parse_grant_clause(clause, line_number):
    """Parse `field = value, ...` text into a {field: value} grant mapping."""
    import re
    clause = clause.strip()
    if not clause:
        return {}
    result = {}
    for piece in _split_top_level(clause):
        fm = re.match(r'^\s*([\w_]+)\s*=\s*(.+?)\s*$', piece, re.S)
        if not fm:
            raise GrantError(
                f"Invalid parameter clause '{piece.strip()}' at line {line_number}")
        result[fm.group(1).strip()] = parse_grant_value(fm.group(2))
    return result


class RequirementResolver:
    """Resolve `Require` capability handles to granted values or sticky #PERM.

    The host is the engine object that owns a run (duck-typed): it must expose
    ``expr_evaluator``, ``array_handler`` and ``current_scope()``. Resolution
    mutates each entry's ``params_evaluated`` (used by ``--list-required``) and
    returns the mapping of capability variable -> bound value for the host to
    define in scope. Denied or ungranted capabilities always bind the sticky
    #PERM error; execution continues either way.
    """

    def __init__(self, host):
        self.host = host

    def to_native_param(self, value):
        """Collapse Grid values to plain Python scalars/lists for grant storage."""
        from units import UnitValue
        if isinstance(value, UnitValue):
            if value.error_code is not None:
                return value.error_code
            if isinstance(value.value, (list, tuple)):
                return [self.to_native_param(v) for v in value.value]
            return value.value
        if isinstance(value, (list, tuple)):
            return [self.to_native_param(v) for v in value]
        return value

    def evaluate_params(self, entry):
        """Resolve the requested `with (...)` parameter expressions to values."""
        scope = self.host.current_scope().get_evaluation_scope()
        evaled = {}
        for field, expr in entry['params'].items():
            try:
                val = self.host.expr_evaluator.eval_expr(
                    expr, scope, entry['line_number'])
                evaled[field] = self.to_native_param(val)
            except Exception:
                evaled[field] = expr  # keep the raw source text
        entry['params_evaluated'] = evaled

    def build_capability(self, entry, params):
        """Build the capability object a granted handle variable is bound to."""
        name, res = entry['name'], entry['resource']
        value = {
            '_type_name': res,
            '_capability': True,
            '_resource': res,
            '_params': dict(params),
            '_name': name,
        }
        if entry.get('resource_lower') == 'ticker':
            # A Ticker is a clock object. The engine counts ticks engine-side
            # (see executor `_advance_tickers`) but that counter is never
            # registered as a public Grid member: Grid observes elapsed ticks
            # only through a derived `tick.counter()`/`tick.timer()` handle
            # (reading <handle>.now / <handle>.remaining), never through
            # <cap>.value.
            pass
        hidden = entry['type_def'].get('_hidden_fields')
        if hidden:
            value['_hidden_fields'] = set(hidden)
        value.update(params)
        return value

    def merge_and_validate(self, entry, chosen, line_number):
        """Merge a chosen grant over the requested params, fill field defaults,
        then validate every parameter against the resource field constraints."""
        type_def = entry['type_def']
        field_constraints = type_def.get('_field_constraints') or {}
        member_keys = {str(f).lower() for f in type_def.get('_member_keys', set())}
        if not isinstance(chosen, dict):
            raise GrantError(
                f"Grant for '{entry['name']}' must map parameter names to values")
        hidden_fields = {str(f).lower() for f in
                         (type_def.get('_hidden_fields') or set())}
        unknown = [k for k in chosen if str(k).lower() not in member_keys 
                    or str(k).lower() in hidden_fields]
        if unknown:
            raise GrantError(
                f"Unknown parameter(s) {', '.join(sorted(unknown))} for resource "
                f"'{entry['resource']}' (valid: "
                f"{', '.join(sorted(member_keys - hidden_fields)) or 'none'})")
        requested = entry.get('params_evaluated') or {}
        params = {}
        for key in member_keys:
            if key in chosen:
                params[key] = self.to_native_param(chosen[key])
            elif key in requested:
                params[key] = requested[key]
        # Fill gaps from field defaults (``or = <expr>``).
        for key in member_keys:
            if key in params:
                continue
            cons = field_constraints.get(key) or {}
            default_expr = cons.get('default')
            if default_expr is None:
                continue
            try:
                val = self.host.expr_evaluator.eval_expr(
                    str(default_expr),
                    self.host.current_scope().get_evaluation_scope(),
                    line_number)
                params[key] = self.to_native_param(val)
            except Exception:
                pass
        # Validate each provided parameter against its declared constraints.
        for key, val in params.items():
            self.validate_param(
                key, val, field_constraints.get(key) or {},
                line_number, entry['resource'])
        return params

    def validate_param(self, field, value, cons, line_number, resource):
        """Validate one grant parameter against the resource field constraints."""
        from units import ConstraintError, is_error_value
        if not cons:
            return
        if is_error_value(value):
            return
        from scope import Scope
        constraints = dict(cons)
        constraints.pop('var_list', None)
        tmp = Scope(self.host)
        type_key = constraints.get('type')
        if type_key:
            constraints = dict(constraints)
            constraints['type'] = type_key.lower()
        inferred = type_key or self.host.array_handler.infer_type(
            value, line_number)
        tmp.define(field, value, inferred, constraints,
                   is_uninitialized=False, line_number=line_number)
        try:
            tmp._check_constraints(field, value, line_number)
        except ConstraintError as exc:
            raise GrantError(
                f"Invalid value for '{field}' of resource '{resource}': {exc}")
        except Exception as exc:
            raise GrantError(
                f"Invalid value for '{field}' of resource '{resource}': {exc}")

    def prompt_for(self, entry):
        """Interactively ask the user to grant or deny one required capability."""
        sys.stdout.write(render_require_prompt(entry))
        while True:
            answer = input(
                "Grant? (y/n, or 'edit' to change parameters): ").strip().lower()
            if answer in ('n', 'no'):
                return None
            if answer in ('y', 'yes', ''):
                return dict(entry.get('params_evaluated') or {})
            if answer == 'edit':
                try:
                    overrides = parse_grant_clause(
                        input("Parameters (field = value, ...): ").strip(),
                        entry['line_number'])
                except Exception as exc:
                    sys.stdout.write(f"  {exc}\n")
                    continue
                merged = dict(entry.get('params_evaluated') or {})
                merged.update({k.lower(): v for k, v in overrides.items()})
                return merged
            sys.stdout.write("  Please answer y/n/edit.\n")

    def resolve(self, requirements, grants, can_prompt=False):
        """Resolve requirement entries into capability bindings.

        Returns ``(grant_map, bindings)`` where ``grant_map`` is the complete
        {name(lower): params|None} mapping (including entries defaulted to deny
        or answered by the interactive prompt) and ``bindings`` is a list of
        ``(var_name, value, vtype, line_number)`` tuples for the engine to
        define in scope.
        """
        from units import PERM_ERROR, error_value
        grant_map = dict(grants or {})
        for entry in requirements:
            self.evaluate_params(entry)
        prompts = []
        for entry in requirements:
            key = entry['var_name'].lower()
            if key not in grant_map:
                if can_prompt:
                    prompts.append(entry)
                else:
                    # Default-deny when no interactive prompt and no grant file.
                    grant_map[key] = None
        for entry in prompts:
            grant_map[entry['var_name'].lower()] = self.prompt_for(entry)
        bindings = []
        for entry in requirements:
            key = entry['var_name'].lower()
            declaration = grant_map.get(key)
            line_number = entry['line_number']
            if declaration is None:
                value = error_value(PERM_ERROR)
                vtype = None
            else:
                params = self.merge_and_validate(entry, declaration, line_number)
                value = self.build_capability(entry, params)
                vtype = 'capability'
            bindings.append((entry['var_name'], value, vtype, line_number))
        return grant_map, bindings