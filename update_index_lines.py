#!/usr/bin/env python3
"""Re-sync the line numbers in index.md with the actual source files.

Usage:
    python update_index_lines.py            # rewrite index.md in place
    python update_index_lines.py --check    # exit 1 if index.md is stale (pre-commit)

For every `### <file.py (NNNN lines)>` section in index.md, this script
re-indexes `def`/`class`/module-level-constant locations in the referenced
source file and rewrites:

  - section headers:        `### file.py (NNNN lines)`
  - single refs:            `name` (NNNN)
  - two-name refs:          `a` / `b` (NNNN/MMMM)
  - "plus" refs:            `name` (NNNN+)
  - "tilde" refs:           `name` (~NNNN)
  - module-level refs:      `name` (module-level, NNNN)
  - wildcard spans:         `prefix_*` (NNNN–MMMM)   (first–last def matching prefix)
  - inline file refs:       file.py:NNNN   (preceding identifier, e.g. `run`)

Only the numbers are rewritten; surrounding text (including a leading
`class ` inside the backticks) is preserved verbatim.  A name is resolved
against the section's own file first (if it is defined several times there,
the definition closest to the currently documented line wins), then — if
unique — against every other indexed file.  Unresolvable references are left
untouched, so the script is safe to run at any time.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(ROOT, "index.md")

HEADER_RE = re.compile(r"^### `([A-Za-z_][\w.-]*\.py)` \(\d+ lines\)")
IDENT = r"[A-Za-z_]\w*"
DIGIT = r"\d+"

# Patterns in decreasing specificity.  Each identifier/number is a single
# capture group (identifiers may carry a non-captured 'class ' prefix).
WILDCARD_RE = re.compile(
    rf"`(?:class\s+)?({IDENT})\*` \(({DIGIT})[–-]({DIGIT})\)")
MODULE_LEVEL_RE = re.compile(
    rf"`(?:class\s+)?{IDENT}` \(module-level, ({DIGIT})\)")
TWO_NAME_RE = re.compile(
    rf"`(?:class\s+)?({IDENT})` / `(?:class\s+)?({IDENT})` "
    rf"\(({DIGIT})/({DIGIT})\)")
PLUS_RE = re.compile(rf"`(?:class\s+)?({IDENT})` \(({DIGIT})\+\)")
TILDE_RE = re.compile(rf"`(?:class\s+)?({IDENT})` \(~({DIGIT})\)")
SINGLE_RE = re.compile(rf"`(?:class\s+)?({IDENT})` \(({DIGIT})\)")
INLINE_RE = re.compile(rf"\b([A-Za-z_][\w.-]*\.py):({DIGIT})\b")
MODULE_CONST_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=")
WORD_RE = re.compile(r"[A-Za-z_]\w*")


def _swap_number(m, group, value):
    """Return the whole match with the number in `group` replaced."""
    start = m.start(group) - m.start(0)
    end = m.end(group) - m.start(0)
    return m.group(0)[:start] + str(value) + m.group(0)[end:]


def _scan_file(path):
    """Return (name_to_lines, line_to_name) for a Python source file."""
    name_to_lines = {}
    line_to_name = {}
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return name_to_lines, line_to_name
    for i, raw in enumerate(lines, start=1):
        m = re.match(r"^\s*(class|def)\s+(\w+)", raw)
        if m:
            name = m.group(2)
            name_to_lines.setdefault(name, []).append(i)
            line_to_name[i] = name
            continue
        # Module-level constant assignment (e.g. DEPENDENCY_IGNORED_TOKENS).
        if not raw[0].isspace():
            m2 = MODULE_CONST_RE.match(raw)
            if m2:
                name_to_lines.setdefault(m2.group(1), []).append(i)
                line_to_name[i] = m2.group(1)
    return name_to_lines, line_to_name


def main():
    check_only = "--check" in sys.argv
    try:
        with open(INDEX, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as err:
        print(f"error: cannot read {INDEX}: {err}", file=sys.stderr)
        return 2

    caches = {}
    current_file = None
    new_lines = []
    changes = []

    def _get(file_name):
        if file_name not in caches:
            caches[file_name] = _scan_file(os.path.join(ROOT, file_name))
        return caches[file_name]

    def _resolve(name, prefer_file, old_num=None):
        """Resolve name -> line.

        Prefer the section's own file; among several definitions of the same
        name there, take the one whose line is closest to the documented one.
        Falls back to a unique definition in any other indexed file.
        """
        def _pick(cands):
            if not cands:
                return None
            if len(cands) == 1:
                return cands[0]
            if old_num is None:
                return None
            return min(cands, key=lambda ln: abs(ln - old_num))

        if prefer_file:
            best = _pick(_get(prefer_file)[0].get(name))
            if best is not None:
                return best
        hits = []
        for other, (n2l, _) in caches.items():
            hits.extend(n2l.get(name, []))
        unique = sorted(set(hits))
        if len(unique) == 1:
            return unique[0]
        return _pick(hits) if old_num is not None else None

    def _record(orig, new):
        if new != orig:
            changes.append((orig.rstrip("\n"), new.rstrip("\n")))

    def _rewrite(line):
        orig = line

        # Wildcard spans: `prefix_*` (first–last) over matching defs.
        def _wildcard(m):
            prefix = m.group(1)
            if current_file:
                n2l, _ = _get(current_file)
                lines = [ln for n, lns in n2l.items()
                         if n.startswith(prefix) for ln in lns]
                if len(lines) >= 2:
                    base = m.start(0)
                    new = f"{min(lines)}–{max(lines)}"
                    return (m.group(0)[:m.start(2) - base] + new +
                            m.group(0)[m.end(3) - base:])
            return m.group(0)

        line = WILDCARD_RE.sub(_wildcard, line)

        # module-level refs.
        def _module_level(m):
            name = m.group(1)
            if current_file:
                n2l, _ = _get(current_file)
                if name in n2l:
                    return _swap_number(m, 2, n2l[name][0])
            return m.group(0)

        line = MODULE_LEVEL_RE.sub(_module_level, line)

        # two-name refs: `a` / `b` (N1/N2).
        def _two_name(m):
            name_a = m.group(1)
            name_b = m.group(2)
            line_a = _resolve(name_a, current_file, int(m.group(3)))
            line_b = _resolve(name_b, current_file, int(m.group(4)))
            base = m.start(0)
            text = m.group(0)
            new = (text[:m.start(3) - base]
                   + str(line_a if line_a is not None else m.group(3))
                   + text[m.end(3) - base:m.start(4) - base]
                   + str(line_b if line_b is not None else m.group(4))
                   + text[m.end(4) - base:])
            return new

        line = TWO_NAME_RE.sub(_two_name, line)

        # plus refs: `name` (NNN+).
        def _plus(m):
            name = m.group(1)
            line_no = _resolve(name, current_file, int(m.group(2)))
            if line_no is not None:
                return _swap_number(m, 2, line_no)
            return m.group(0)

        line = PLUS_RE.sub(_plus, line)

        # tilde refs: `name` (~NNN).
        def _tilde(m):
            name = m.group(1)
            line_no = _resolve(name, current_file, int(m.group(2)))
            if line_no is not None:
                return _swap_number(m, 2, line_no)
            return m.group(0)

        line = TILDE_RE.sub(_tilde, line)

        # single refs: `name` (NNN).
        def _single(m):
            name = m.group(1)
            line_no = _resolve(name, current_file, int(m.group(2)))
            if line_no is not None:
                return _swap_number(m, 2, line_no)
            return m.group(0)

        line = SINGLE_RE.sub(_single, line)

        # inline file refs: file.py:NNN — resolve the identifier just before
        # the reference inside that file.
        def _inline(m):
            file_name = m.group(1)
            n2l, _ = _get(file_name)
            before = line[:m.start()]
            for tok in reversed(list(WORD_RE.finditer(before))):
                name = tok.group(0)
                if name in n2l:
                    return _swap_number(m, 2, n2l[name][0])
            return m.group(0)

        line = INLINE_RE.sub(_inline, line)

        _record(orig, line)
        return line

    for line in lines:
        hdr = HEADER_RE.match(line)
        if hdr:
            file_name = hdr.group(1)
            current_file = file_name
            _get(file_name)
            try:
                with open(os.path.join(ROOT, file_name), encoding="utf-8") as fh:
                    count = sum(1 for _ in fh)
            except OSError:
                count = None
            if count is not None:
                new_header = f"### `{file_name}` ({count} lines)"
                new_line = HEADER_RE.sub(new_header, line)
                _record(line, new_line)
                line = new_line
        new_lines.append(_rewrite(line))

    if changes:
        if check_only:
            print(f"{len(changes)} stale line reference(s) in index.md:")
            for old, new in changes:
                print(f"  - {old}\n    + {new}")
            return 1
        with open(INDEX, "w", encoding="utf-8") as fh:
            fh.writelines(new_lines)
        print(f"Updated {len(changes)} line reference(s) in index.md.")
    else:
        print("index.md is up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
