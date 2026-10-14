# GridLang Codebase Index

This file is a developer map of the `grid-lang.cc` repository. It describes each
source file, what it owns, how the pieces connect, and the conventions a task
must respect. Read this before editing; don't rediscover the codebase.

## How to use this index (for agents new to the project)

1. **Start with Architecture / data flow** — it explains the single engine
   (`GridLangCompiler(GridLangExecutor)` with `self.compiler == self`) and the
   run pipeline. That one fact explains most of the codebase.
2. **Skim File-by-file only as needed** — `compiler.py`/`executor.py` are the
   engine, `expression.py`/`array_handler.py`/`scope.py` own semantics,
   `units.py` owns units, `test_runner.py` is the spec.
3. **Run `python3 test_runner.py` (525 tests) and `python3 main.py test_convert.grid`**
   before and after any change. Use `python3` only; temp files only in `.oc_tmp/`.
4. **Edit, then re-sync line numbers**: `python3 update_index_lines.py` (or
   `python3 update_index_lines.py --check` in CI). Only the numbers are
   rewritten — section text is yours to maintain.
5. **Case-insensitive everywhere, grid is `_GridStore` keyed by `(row,col)` tuples,
   arrays are flat/dict-form** — see "Language conventions" below.

## What the project is

GridLang is a **line-based interpreted language** for tabular data ("grids").
Source files end in `.grid`. The interpreter is pure Python with **no runtime
dependencies** (`pyarrow` was used historically for arrays but has been fully
removed). The CLI `grid` is produced from `main.py`. The README calls it a
"compiler" but there is **no tokenizer/parser/AST**: execution is a
statement-dispatch loop over source lines with regex-based parsing throughout.

**Unit system** (`units.py`, `test_convert.grid`): values can carry units
(`5 of m`, `"ox" of animal`, `2 of 1`), `Define X as UnitSource(Target)` +
`Convert` registers category (`"ox" of animal to "beef"`) or formula
(`x as number of cm to x/100`) mappings, `Input`/`Let`/`For`/`:`/`Push`/`Init`/`Output`
with `of <unit>` automatically convert via LHS `expected_unit`
(`ExpressionEvaluator._apply_expected_unit`) and central fallback
`Scope._unit_convert` (`units.apply_conversion`). Special unit `1` is
dimensionless: usable instead of unitless for `*`/`^`, `m/1 → m`, `m/m → 1`
for `/`/`\`/`mod`.

Language reference: `Documentation.md` (tutorial style). Install/usage docs:
`README.md`.

## Running things

```bash
# Run a program (args after filename are program inputs)
python main.py example.grid 42
python main.py example.grid --debug   # also exports <file>.csv

# Run the inline test suite (525 tests)
python test_runner.py                 # all tests
python test_runner.py 1 2 4           # subset by number
python test_runner.py 282 289         # unit tests

# Unit smoke test
python main.py test_convert.grid      # Butcher + SILength → dist_meter: 4012.7

# Installed CLI (both venv/ and .venv/ have it; py3.9)
.venv/bin/grid example.grid
```

`main.py` recognizes only `--debug` and `--` passthrough; anything else after
the grid filename becomes a program argument. (README's `-r` flag is NOT
implemented.) If no program args are given and stdin is a TTY, missing `Input`s
are prompted (`compiler.prompt_missing_inputs`).

## Architecture / data flow

1. `main.py` reads the file and constructs one `GridLangCompiler` (a fresh one
   per run).
2. `compiler.run(code, args)` (`compiler.py:848`) is the interpreter entry
   point. `GridLangCompiler` is a **single engine class** — it inherits its
   whole runtime loop from `GridLangExecutor` and executes directly on itself.
   There is **no separate executor object and no method/state copy handoff**:
   `_reset_state()` re-initialises `self`'s runtime state in place and then the
   inherited `run` pipeline runs on `self`.
3. The pipeline: `_run_setup` (which calls `_reset_state` and
   `_preprocess_code`/`_extract_functions`) → `_run_prepare_execution` (now
   materializes `UnitSource` constants and registers top-level `Convert`s) →
   `_run_main_loop` → `_resolve_pending_assignments` →
   `_process_deferred_assignments` → `_print_outputs`.
4. `_run_main_loop_impl_body` dispatches each line by statement kind: `For`,
   `Let`, `Push`, `When`, `Return`, grid assignment (`[A1] := ...`),
   declaration, or "misc". Big dispatch chains live in the executor layer
   (the `GridLangExecutor` base) and `control_flow.py`.
5. Unit flow: `_replace_of_unit_literals` rewrites `5 of m` / `"ox" of animal` / `2 of 1`
   → `gridlang_of_unit(...)` → `UnitValue`; `For`/`Let`/`Push` pass `expected_unit`
   from LHS `of T`, `Scope._unit_convert` falls back to `apply_conversion`.
6. Results: `Print x` (any scope) and `Return x` (inside functions and
    operations) append to `output_values['output']` (printed by
    `_print_outputs`); grid writes land in `self.grid` (a `_GridStore` dict
    keyed by index tuples like `(0,0)`). `--debug` → `compiler.export_to_csv`
    (`compiler.py`).

The single most important design fact: **there is one engine class.**
`GridLangCompiler(GridLangExecutor)`, where `GridLangExecutor` provides the
runtime loop and `GridLangCompiler` adds the persistent state, type/unit
machinery and public API (`run`, `call_function`, `call_subprocess`,
`set_input_values`, `export_to_csv`). `self.compiler == self` so executor-layer
code that references `self.compiler.*` simply resolves back to the engine. When
adding a feature, put a shared helper in `GridLangBase` (`grid_lang_common.py`)
or the executor layer, and compiler-only logic in `GridLangCompiler` — there is
no duplication to keep in sync across two live objects anymore.

## File-by-file

### `main.py` (133 lines) — CLI entry point
- `_parse_cli_args`: splits `--debug` / `--` / positional program args.
- `run_grid_program`: reads the `.grid` file, builds a `GridLangCompiler`, sets
  `prompt_missing_inputs`, calls `compiler.run`, and on `--debug` calls
  `compiler.export_to_csv`.
- `main()` is the console-script entry (`grid=main:main` in setup.py).

### `setup.py` — packaging
- Console script `grid=main:main`; declares the 10 top-level modules as
  `py_modules`; **no `install_requires`** (`pyarrow` was removed); LGPLv3.

### `units.py` (538 lines) — unit wrapper + conversion registry
- `UnitValue` + sticky `#UNIT` errors + `CONVERSIONS` dict `(src.lower(),dst.lower())->[entries]`
  (`register_conversion`/`lookup_conversions`/`has_conversion`/`apply_conversion`).
  `1` is special: `UnitValue._is_one`, `__mul__` treats `1` as unitless, `_divide` `m/1→m` `m/m→1`, `__pow__` allows exponent `1`.
- `UnitValue` overloads: `+`/`-` same unit or one side unitless; `*` with `1`; `/`/`\`/`mod` with `1`; `^` with `1`.

### `compiler.py` (7375 lines) — state + orchestration, public API
`class GridLangCompiler` is the engine's **state holder and public surface**. It
is the only class `main.py` constructs. It inherits the runtime loop from
`GridLangExecutor` and holds nearly all persistent state created in `__init__`:
- Callable-body validation: `_reject_output_reads` (an `Output` is a
  destination, never an input) and `_refresh_live_inputs` /
  `_maybe_publish_live_output` (live I/O, which must not notify dependents).
- Grid & scoping: `grid` (`_GridStore`), `scopes` (stack of `Scope`),
  `variables`, `types`, `dimensions`, `dim_names`, `dim_labels`.
- Publish/listen: `_listeners`, `_set_by`, `_propagating` (see "Conventions").
- Program constructs: `types_defined`, `functions`, `subprocesses`,
  `input_variables`, `output_variables`, `output_values`.
- UnitSource: `unit_sources` (`_target_unit`,`_constants`,`_orig_name`), `_top_level_converts`, `_UnitSourceNamespace`.
- Resolution machinery: `pending_assignments`, `deferred_lines`,
  `undefined_dependencies`, `dependency_graph`, `global_guard_entries`,
  `global_for_line_numbers`/`entries`, `handled_assignments`.
- Helper engines: `expr_evaluator` (ExpressionEvaluator), `array_handler`
  (ArrayHandler), `control_flow` (GridLangControlFlow), `type_processor`
  (GridLangTypeProcessor), `parser` (GridLangParser).

Public API (called from `main.py` / tests): `run`, `call_function`,
`call_subprocess`, `set_input_values`, `export_to_csv`.

Notable methods:
- `run` (848): engine entry — `_reset_state()` then delegates to the inherited
  runtime pipeline (no executor handoff).
- `current_scope`/`push_scope`/`pop_scope` (310/325/331).
- `_seed_grid_variable` (4378): predefines `grid` in the global scope as a
  `_GridStore` (sparse array keyed by 0-based `(row,col)` tuples; aliased as
  `self.grid`), so `grid{row, col}` works at top level. Skipped inside
  read-only function sub-compilers.
- UnitSource: `_parse_unit_source_header` (320) / `_finalize_unit_source` (334) / `_register_convert_line` (369) / `_infer_convert_target_unit` (262, evaluates RHS with stripped var, falls back to declared units) / `_materialize_unit_source_constants` (476) / `_register_top_level_converts` (470).
- `_extract_functions` (1110): pulls `Function`/`Subprocess` defs out of the
  main code and registers them.
- `_instantiate_type` (1669), `_evaluate_with_value` (2082), `_apply_with_clause`
  (1246): type/`with` object construction; now handles `:` field unit conversion.
- `call_subprocess` (2807): runs a sub-`GridLangCompiler` in isolation.
- `_process_grid_assignment` (5398), `_process_declarations_and_labels` (5981),
  `_collect_global_declarations` (6157): top-level statement handling; now handles `of 1` and `"ox" of animal`.
- `export_to_csv` (7162): `--debug` CSV export (grid as matrix, or outputs as
  one column when the grid is empty).
- `set_input_values` (7185): binds CLI/keyboard args to `Input`s; now evaluates `"5 of in"` before `update` so `Input a of m` converts.
- `_seed_globals` (1670): for sub-compilers; **skips redefining `grid`**.

Also defines `SubprocessResult` (51): result container exposing `grid`,
`variables`, `outputs`, and `_UnitSourceNamespace` (66) the UnitSource lookups.

### `executor.py` (7524 lines) — the runtime loop
`class GridLangExecutor` is the **base class that owns the interpreter's
dispatch loop and its per-run runtime state**. It is not instantiated directly
as a facade (the old compiler→executor copy handoff was removed); `run` is the
live entry. Key methods (the `compiler.py`/`grid_lang_common.py` layers call
`super()`/override these):
- `run` (3285): top-level sequence (acts on `self`; see Architecture).
- `_run_setup` (6484), `_run_prepare_execution` (6523), `_print_outputs`
  (4716), `_materialize_inits` (7251), `_process_deferred_assignments` (7396).
  `_run_prepare_execution` now calls `_materialize_unit_source_constants` + `_register_top_level_converts`.
- Main loop: `_run_main_loop` (3336) → `_run_main_loop_impl` (3884) →
  `_run_main_loop_impl_body` (3932). `_handle_main_loop_*` methods dispatch
  statement kinds: `Let` (1128), `For` (1752 fallback), grid assignment (3903),
  `When` blocks (3956), `Push` (4152), `Return` (4108), misc (3724).
- Dependency/guard machinery: `_build_dependency_network` (1180),
  `_determine_needed_lines`, `_evaluate_guard_conditions`,
  `_evaluate_global_guards_pre_execution`, `_execute_global_for_loops`,
  `_attempt_resolve_pending_var`, `_resolve_ready_pending_vars`.
- `Let` semantics: first pass `_process_let_first_pass` (2004), binding
  `_bind_declared_var` (1255, now `expected_unit`), standard assignment,
  second pass, generator values, `_materialize_inits`.
- `For`: `_execute_simple_for_assignment` (749, now `expected_unit`), `Push` via `target_unit`.
- `Push` semantics: `_handle_push_assignment` (6876), `_evaluate_push_expression`
  (4345, now `expected_unit`), `_handle_push_assignment_line`.
- `When` blocks: `_register_when_block` (277), `_process_when_triggers` (749),
  `_run_when_block` (765), `_handle_when_block`.
- Cell mirrors / spill listeners: `_cell_spill_dependencies` (every free
  variable in the RHS, plus the spill root) and
  `_pre_register_cell_spill_constraints` (skips block-body lines, whose
  listeners are installed when the block executes).
- Shared helpers (from `grid_lang_common.py`): `_strip_constraint_operands` and
  `_DEPENDENCY_IGNORED_TOKENS` (re-exported here as `DEPENDENCY_IGNORED_TOKENS` for
  back-compat; compiler.py imports the same `_DEPENDENCY_IGNORED_TOKENS`). They are
  now a single source of truth in `grid_lang_common.py`, shared by both layers.

### `grid_lang_common.py` (298 lines) — shared base + helpers
`class GridLangBase` (249) is the common ancestor of `GridLangExecutor` and thus
`GridLangCompiler`. It holds shared helpers used across layers: `_STATEMENT_KEYWORDS`,
`_first_keyword`, `_IDENTIFIER_TOKEN_PATTERN`, `_STRING_LITERAL_PATTERN`,
`_DEPENDENCY_IGNORED_TOKENS`, `_strip_constraint_operands` (113),
`_strip_builder_arrows`, `_strip_cell_address_tokens`, `_has_star_dim`,
`_apply_dim_base_offsets`, `_infer_declared_type`. Put methods/data needed by
both the compiler and executor layers here. (Note: `error_value`/`UNIVERSAL_ZERO`
live in `units.py`, not here.)

### `expression.py` (4176 lines) — expression evaluation
`class ExpressionEvaluator` evaluates RHS expressions, arrays, ranges, sums,
dimension selectors, interpolations, member/field access, and Python-fallback
evaluation.
- Entry points: `eval_or_eval_array` (116, now `expected_unit`), `eval_expr` (2557), and for
  assignments `_evaluate_array` (551).
- LHS-informed: `_apply_expected_unit` (79) + `_formula_eval` (97, strips LHS var) threaded via `expected_unit`.
- `eval_expr` is the big recursive dispatcher: array literals `{}`, pipes `|`,
  interpolated cell refs, paren/curly indexing, member calls, user function
  calls, object creation, field access, address-indexed access, then scalar
  constructs, then simple variables.
- Python fallback: `_evaluate_with_python_fallback` (2999) builds a scope and
  `eval()`s complex arithmetic (`_build_fallback_cell_scope` 2254,
  `_eval_python_fallback_result` 2501, `_get_eval_globals` 2937). Now includes
  `_replace_of_unit_literals` (4087) for `"ox" of animal`/`5 of in`/`2 of 1` → `gridlang_of_unit` and `Attribute` flat-key `SILength.inch` + case-insensitive `_resolve_fallback_name` (3586).
- Interpolation: `_process_interpolation` (3958). Operators:
  `_replace_operators` (4102).
- Grid indexing: `_replace_grid_indexing` (869) — rewrites bare `grid{...}`
  when the predefined `grid` variable is in scope; otherwise falls through to
  the generic array-access path (ranges, `*` selectors, `#N/A` for unset cells).
- Also `CaseInsensitiveDict` (29): case-insensitive dict used for
  eval scopes.

### `builtin_functions.py` (1125 lines) — builtin registry + predefined functions
Single source of truth for every predefined GridLang function (`SUM`/`MIN`/`MAX`/`ROWS`/`ABS`/`LEN`/`TEXTSPLIT`/`TRANSPOSE`/math, …). Both evaluation paths (`ExpressionEvaluator._build_python_fallback_scope` (`expression.py:3435`) and `_get_eval_globals` (`expression.py:4116`)) call `get_builtin_functions()` so a new builtin is instantly available everywhere.
- Registry (37): `BUILTINS`/`ALIASES`/`ARG_COUNTS`/`VECTORIZED` plus `KEYWORDS` (43, re-exported by `compiler.py`/`expression.py` for dependency stripping). `register_builtin` (66) and `register_vectorized_builtin` (96) are decorators (`@register_builtin("MIN", aliases=["Number.Min"], arg_count=1)`); non-vectorized by default, vectorized ones broadcast element-wise.
- Arity + helpers: `_check_arity` (228) / `_resolve_builtin` (248) / `_to_list` (129, normalizes sparse `dict`, `{'array':…}`, `list` → flat list).
- Builtins (143): `ROWS` (143) / `SUM` (155) / `LEN` (166, `Text.Len`) / `ABS` (174, `Number.Abs`) / `POWER` (179) / `INT` (184) / `MID` (189, `Text.Mid`) / `TEXTSPLIT` (199, `Text.Split`) / `COUNTA` (207) / `RANDARRAY` (223) / `SORTBY` (229) / `TRANSPOSE` (240) / `MIN` (291, `Number.Min`) / `MAX` (299, `Number.Max`) plus math shims (`sqrt`/`sin`/… → `Number.<name>`, 306) and `str`/`int`/`float`/`abs` (313). `MIN`/`MAX` flatten via `_to_list` before `min`/`max`.
- Public API (323): `_is_array_arg` (1039) / `_broadcast_builtin` (326, only for `VECTORIZED`; detects `list`/`{'array':…}`/sparse, checks equal lengths, applies scalar broadcast, otherwise returns `None` to fall back to normal call) / `get_builtin_functions` (373, builds `name→wrapped` dict; wrapper does `_check_arity`, then `_broadcast_builtin`, then `fn(*args)` with `TypeError→#TYPE/I`; injects `math`).

### `array_handler.py` (3066 lines) — grid/array/tensor operations
`class ArrayHandler` centralizes all array knowledge:
- Cell addressing & lookup: `resolve_cell_index` (65), `cell_ref_to_indices`
  (129), `lookup_cell` (1560), `get_range_values` (1253/1286),
  `_lookup_extended_address` (131367, `_write_extended_tensor` (1649).
- Assignment: `evaluate_line_with_assignment` (235),
  `_parse_assignment_target_details` (26257, `_perform_assignment_write` (537),
  `_assign_horizontal_array` (98984, `assign_range` (1336),
  `_assign_extended_range` (121221, `_assign_index_selector` (941),
  `_assign_dim_selector` (85837, `_update_bound_array_cell` (1136),
  `assign_implicit_intersection_range` (60597, implicit-intersection rewrite
  (`_rewrite_implicit_intersection` 597).
- Spilling helpers: `_resolve_spill_unset` (1542) replaces `None` sentinels
  in flat arrays before writing to grid (uses variable default or `#N/A`);
  `flatten_array` (1919) column-major flattens any array to 1D;
  `flatten_object_fields` (1860) flattens object fields for grid spills.
- Array construction/shape: `create_array` (1773, accepts `template=True` to
  fill with `None` sentinels), `create_object_array` (2189),
  `get_array_shape` (2087), `reshape_array` (2592), `infer_type` (1966, now handles `UnitValue("beef")→text`), 
  `fill_array` (2995), `flatten_object_fields` (1860), `flatten_array`
  (1443), `_nested_from_flat` (2007), `to_display_value` (1964).
- Constraints/dims: `set_labels` (2204), `check_dimension_constraints` (2298),
  `validate_array_element_types` (161694 — element-level base-type checking
  (`as number`/`as text` arrays reject mismatched scalars), `_dim_size` (2231).
- Grid-as-array: `get_grid_row` (2648), `get_grid_column` (2677); generic
  `get_array_element`/`set_array_element` handle `_GridStore` like any sparse array.

### `control_flow.py` (3467 lines) — blocks: For / If / Let / When
`class GridLangControlFlow` executes block constructs. Module-level regexes
(9–16) define `if...then`, `elseif...then`, `else`, `for...do`,
`when...do`, `end`.
- `process_for_statement` (203): For-loop handling (ranges, init, arrays).
- Block engine: `_process_block` (1866), `_extract_block_body` (533),
  `pre_scan_blocks` (2918), `_prepare_block_line` (563).
- If: `_process_if_statement` (1986) and the "new"/"rich" variants (2011,
  2113), `_parse_if_header` (2020), `_collect_if_blocks` (2104),
  `_execute_if_block_choice` (2199), `_process_if_elseif_else_block` (3062);
  condition evaluation helpers `_evaluate_if_*` (2446–2842).
- Let: `_process_let_statement_inline` (2297), field/index assignment helpers
  (1339, 1390). **One binder everywhere:** `_bind_let_var_in_block` delegates
  to `executor._bind_declared_var`, so a `Let` in a block body is the same
  instruction as a top-level `Let`. Equality bindings pin a `constant`
  constraint, so a conflicting rebind is `#VALUE`:
  ```
  Let x = 5 / Let x = 6   ->  #VALUE
  x = 5 / x = 6           ->  #VALUE
  ```
  A loop body still re-binds on every iteration, because
  `_bind_declared_var` records a loop-body declaration in
  `local_let_declarations` and the loop's copy-back to the parent scope skips
  exactly that set — the variable is dropped when the iteration ends, so
  `let a = (i + 3) * i \ 2 + 1` binds afresh each pass (Tests 119 and 141).
  This is scope, not a separate semantic: previously `_bind_declared_var`
  recorded loop-locality only in `implicit_let`, which the copy-back does not
  consult, so the variable leaked to the parent scope and its `constant` pin
  collided on the following iteration.
- `live` and `init` are declaration modifiers, not statements. They are valid
  only as suffixes on an `Input`/`Output`/`For` declaration line
  (`Input a live`, `Output o init 5 live`, `For rev as Text init ""`).
  `_reject_stray_pseudo_constraint` (executor.py) raises a `SyntaxError` on a
  standalone `a live` / `a init 5` line rather than letting it become an
  unevaluated expression or a silent no-op.
- **The `Let` keyword is optional** whenever a line is unambiguously a
  declaration: `_is_optional_let_line` rejects a leading instruction keyword,
  an `AND` connective, a `THEN` header, and the I/O-only modifiers `live` and
  `init`, then asks `_parse_variable_def` whether the line declares something
  (a value, a constraint, or a type). So every constraint kind works with the
  keyword dropped, not just assignment:
  ```
  Let x > 1                      x > 1
  Let v as number                v as number
  Let u of peso = 5              u of peso = 5
  Let arr dim 3 as logical = ..  arr dim 3 as logical = ..
  Let grid not null or = 0       grid not null or = 0
  ```
  A bare `a` declares nothing, so `Let a` stays mandatory. A comparison stays
  mandatory because its `=` belongs to `>=`/`<=`/`<>`/`!=`, not an assignment.
  A recognised declaration then dispatches to the ordinary `Let` binder
  (`_handle_block_optional_let_statement` in a block, and
  `_handle_main_loop_misc_statement` at top level), so bare and explicit `Let`
  are the same instruction with the same rebinding semantics.
- **Known gap: unrecognized operators.** `_replace_operators` translates `mod`,
  `\`, `^`, and the boolean literals for `eval()`, so a Python-only operator
  reaches `eval()` and may evaluate silently instead of erroring — `//` and
  `**` return `3.0` and `8.0` rather than failing. A denylist scan was tried and
  reverted: it hunts for bad tokens instead of failing to parse the line, so it
  could only ever approximate the real rule. The right shape is for the
  expression parser to attempt the line and raise when it cannot recognize a
  construct; that is still to be designed.
- `_handle_block_*` methods (343–960): per-statement handling inside blocks.
- `_scan_local_if_block` (3225): derives `If`/`Elseif`/`Else` clause structure
  from the lines at hand, for bodies that are not in the top-level pre-scan
  (constructor bodies, nested slices), so the shared clause-selection code
  runs unchanged.

### `scope.py` (1321 lines) — scope + variable semantics
- `class Scope` (49): variable storage with constraints.
  - `define` (441), `update` (539), `get` (708), `is_uninitialized` (734),
    `get_defining_scope` (749).
  - Inputs/outputs: `define_input` (762), `define_output` (590, now preserves `unit` from `Input` when `Output` shares name), `is_input`/`is_output` (518/527), `connect_pipe` (809), `push_value`
    (555), `_propagate_wave` (851) — the publish/listen ripple.
  - Unit handling: `_unit_convert` (104, now tries `apply_conversion` before `#UNIT`), `_has_pending_assignment` (119).
  - Constraints: `_re_evaluate_constraints` (904), `_check_constraints` (830, now unit-aware for `Let y of m = 5 of in`), `_validate_base_type` (975) — validates scalars AND, since the pyarrow
    removal, element-by-element base types of `dim` arrays (via
    `array_handler.validate_array_element_types`), `_expression_depends_on`
    (651). `_array_unset_value` (array_handler.py:1504) resolves `None`
    sentinels for unset template array cells: checks the variable's
    `constraints['default']` (from `or = <expr>`) and evaluates it; falls back
    to `error_value(NA_ERROR)` (`#N/A`).
  - Scoping: `is_shadowed` (865), `get_evaluation_scope` (873),
    `get_full_scope` (1311), `_coerce_custom_type_value` (164).
- `class _GridStore` (49): dict backing `compiler.grid`; every cell write
  calls `compiler._notify_cell_changed` (compiler.py:3892). (The old
  `_ListenerGrid`/`GridLiveView` classes were removed — `_GridStore` is the
  single grid store, keyed by 0-based index tuples.)
- `_ACTIVE_RUNNERS` (22): stack of executing compilers; used with the
  `_outer_scope_read_only` flag to reject writes from read-only function
  sub-compilers to outer scopes.

### `type_processor.py` (1501 lines) — `Define X as Type` handling
`class GridLangTypeProcessor`:
- Type-def parsing: `_parse_type_def` (90), `_parse_type_def_line` (99),
  `_extract_type_field_line` (9696, `_parse_type_field_constraints` (243),
  `_record_type_field_definition` (15157, `_collect_type_computed_fields`
  (178), `_finalize_type_def_state` (339).
- Executing type body code against an instance: `_execute_type_code` (368),
  `_execute_type_block` (532), `_process_grid_assignment` (784),
  `_process_type_let_statement` (859),
  `_process_type_assignment` (1039).
- `_build_type_eval_scope` (1265), `_execute_builder` (1286).
- **Constructor/builder bodies now share the main block engine.** `For`,
  `Let ... then`, and `If ... then/else` bodies are delegated to
  `control_flow._process_block` through `_process_type_for_block_via_engine`,
  `_process_type_let_then_via_engine`, and `_process_type_if_block_via_engine`.
  There is no type-specific For/Let/If interpreter: parsing, loop walking,
  clause selection and guard handling are all the engine's. Three
  consequences, all previously broken:
  - Instance **fields are readable inside the body**. `_push_type_fields_to_scope`
    mirrors `value_dict` into the scope, and `_sync_type_fields_from_scope`
    copies engine writes back, so `Print x` no longer raises `NameError` and
    `If x > 0 then` is no longer silently false. A field name now shadows a
    same-named outer/global variable inside the body — previously the body
    silently read the *global* even when the field was declared with a value.
  - A `Let` naming an instance field **rebinds that field** rather than
    colliding. `scope.define` honours `compiler._allow_scope_overwrite`,
    which is set only for the duration of `_execute_type_code`.
  - Bare `If <cond>` guards gate a constructor body, as elsewhere.
- `When <cond> do` inside a constructor body registers with
  `executor._register_when_block` instead of raising "Unrecognized statement
  in type constructor`. Whether it fires on later field writes is still
  unverified: a bare field name is not a reactive scope variable, and `When`
  is known to fire through the dotted path (`When p.x > 0`). Note that
  `When e > 0 do print $"..."` on one line is headerless, single-statement
  syntax with no `End` — see `test_when.grid`.
- Forward references inside a constructor body (`[A1] := y` before
  `Let y = 3`) fail exactly as they do in the main loop; see the pending
  all-bindings-first-pass work.

### `parser.py` (774 lines) — variable-definition parsing
`class GridLangParser`:
- `_parse_variable_def` (1616: the central parser for `: name [as type] [of
  unit] [dim ...] [constraints] = expr` / `Input`/`Output` lines. Returns
  (parsed_var, parsed_type, constraints, expression).
- Constraint handling: `_check_comparison_series` (323),
  `_match_direct_assignment_patterns` (22225, `_apply_with_clause` (498),
  `_apply_dimension_constraints` (33334, `_merge_custom_type_constraints` (610),
  `_split_on_keywords` (39390, now `seen_equals`/`seen_init` keep `of`/`as` in RHS `5 of in`/`Init 5 of in`), `_parse_dim_size` (740). The `or` keyword in
  `_split_on_keywords` extracts `or = <expr>` as `constraints['default']`;
  `not null` sets `constraints['nullable'] = True`. The default value is used
  by `_array_unset_value` when reading unset template array cells.

### `utils.py` (438 lines) — shared pure helpers
- Address math: `split_cell` (113), `col_to_num` (122), `num_to_col` (129),
  `offset_cell` (12125, `parse_address` (151, N-D dotted addresses like
  `A3.B4.8` → `[3,1,4,2,8]`), `indices_to_address` (188), `is_address` (158),
  `validate_cell_ref` (8585, `prod` (270).
- Case-insensitive dict access: `get_case_insensitive_key` (308),
  `get_case_insensitive_value` (24241.
- Object/type metadata filtering: `public_type_fields` (277),
  `object_public_keys` (21213, `public_object_view` (327) — strip `_hidden`
  fields and keys starting with `_`/`$`/`grid`.
- Text parsing: `iter_interpolation_placeholders` (18), `split_var_defs` (48).
- `format_display_value` (26265: display formatting with float-trimming and
  list/dict-form array support.

### `row_stream.py` (176 lines) — row-stream producers
Pure model layer for a producer that yields a *stream* of values per
statement execution (the `[addr] := <producer>` statement form). Holds the
producer/zip/cell-rule/index model with no engine state, so the executor can
apply it recursively without re-entering the main loop. Successive executions
push successive values; a **block** `For` loop skips a stream position whose
value is an error, an **inline** producer does not (it is a producer, not a
loop, and the cell is still written). `Push` and `Return` inside a producer
participate in the stream. Standalone producers push through.

### `modules.py` (237 lines) — `use` / module harvest
Module source loading, export harvesting and the import binding used by
`use`. Binds a harvested function/subprocess/type under its export name,
rebuilding dotted members for imported types.

### `net.py` (569 lines) — Net resource transport
Client/server socket transport behind the `Net` resource (LISTEN server
grants, request/response, `ReadAll`/`Reply`).

> **Known separation smell**: networking logic is split across *both*
> `net.py` and `builtin_functions.py`. `net.py` owns the transport, URI
> matching (`parse_address`, `_query_globe`, `_segments_match`) and the
> low-level routing, while `builtin_functions.py` owns the grant
> declaration and the `Net` builtin surface (`builtin_net_connect`,
> `builtin_net_server`, `_connection_handle`). The two overlap on
> connection/server concerns; consolidating them would make the resource
> layer a single, auditable unit.

### `test_runner.py` (4235 lines) — inline test suite
`class GridLangTestRunner` with `run_tests_independent(tests)` — runs 525 tests (Tests 282–293 unit tests, Tests 331–349 push/cell-mirror/subprocess semantics, Tests 350–357d resource/`Require`/grant semantics, Tests 385–405b module/`use`/instance-state/dotted-Require/ModuleVersion/runnable-module-run/unknown-call/init-never-fires-on-import semantics, Tests 452–465 live pipes + user-function vectorization/element-access, Tests 479–488 Net LISTEN/server routing + `ReadAll`/`Reply` semantics via the fake listener transport). At the bottom of the file it runs itself when executed directly:
`python test_runner.py [names...]`. Failing names are printed.

## Language conventions to remember when editing

- **Case-insensitive**: keywords, variable/field/type names, cell refs.
  Lookups go through `get_case_insensitive_key`.
- **Grid storage**: `compiler.grid` is a `_GridStore` (dict subclass) keyed by
  0-based index tuples like `(0,0)` for `A1` (row, col). Writes notify via
  `compiler._notify_cell_changed`. Ranges use `:`; `^` marks a range's top-left
  corner (e.g. `[^A3]`); `@` is implicit intersection (current row).
- **Addresses**: N-D dotted addresses (`[A3.D2.8]`) map to 1-based index
  lists; row/col pairs become letter+digit segments, trailing lone index stays
  a bare number.
- **Arrays/tensors**: no pyarrow — bounded arrays are **flat Python lists**
  (1D) or **dict-form** `{'array': flat, 'shape': [...], 'original_shape': [...]}`
  (N-D); unbounded (`dim *`) arrays are plain dicts keyed by 0-based index
  tuples. `|` concatenates dimensions, `;` starts rows, `_` continues rows.
  `dim`/`DIM` declares dimensions. Indexing is 1-based for the language,
  0-based inside array_handler internals. Arrays declared `as number`/`as
  text` are type-checked element-by-element (mismatched scalar → `#TYPE/I`,
  whole array rejected).
- **Template arrays**: `create_array(..., template=True)` fills the buffer with
  `None` sentinels instead of `0`/`""`. Reads of unset cells return `#N/A`
  (via `_array_unset_value` → `error_value(NA_ERROR)`), or a variable's
  declared default if `not null or = <expr>` is present. Python `None` is the
  sentinel; it is distinct from `UNIVERSAL_ZERO` (AST Constant `None`). Spill
  paths (`_resolve_spill_unset`) replace `None` in flat lists with the
  variable's default or `#N/A` before writing to grid.
- **Spilling semantics**: arrays always spill into grid cells — the `^` notation
  is NOT required for arrays. Spilling fills from fastest dimension (first
  declared dim = grid rows) to slowest (last dim = grid columns) in
  column-major order. An array of objects WITHOUT `^` places one object per
  cell (horizontally). With `^`, object fields are expanded: one object per
  row, fields across columns. A single object WITHOUT `^` writes as a single
  cell value; with `^` it spills fields. The core spilling function is
  `_assign_horizontal_array` (`array_handler.py:1214`);
  `_assign_extended_address` (`array_handler.py:1312`) handles dotted targets
  like `A1.B1.3`.
- **Variables**: `: x = expr` (client binding — deferred until deps resolve),
  `Let x init val`/`= val`, `For x init val`. `Push x = expr` updates x and
  propagates to dependents (publish/listen). `Input`/`Output` declare I/O.
  `Init` is `Let`/`For`/`:` with `Init <expr>` (lazy, evaluated on first read).
- **Undimmed rank >= 2 literals are sparse**: a multi-row/plane array literal
  with no declared `dim` is stored as a sparse `dim {*,*}`-style array — a dict
  keyed by 0-based index tuples (first declared dim = grid row) — regardless of
  raggedness. Only the **rank** matters, not the element counts, so ragged
  literals like `{1, 2; 3, 4, 5; 6}` are legal. Rank-1 flat literals stay plain
  lists. `_to_sparse_undimmed` (executor.py) applies this at every variable
  binding site (equality, `init`, deferred/pending resolutions, `[cell] : x =
  expr`); explicit `dim`, type or `with` declarations are left untouched.
- **Text (character) constants**: written outside quotes (`"a" & #NBSP & "b"`)
  or inside interpolation braces (`$"L1{#nl}L2"`): `#NL`, `#NBSP`, `#QUOT`,
  `#APOS`, `#LB`, `#MDASH`, `#NDASH`, `#TAB`, `#SHY`, `#CR`, and any code point
  as `#U/xxxx` (up to 4 hex digits). Case-insensitive (`#nbSp`, `#u/16f1`).
  `resolve_text_constant_token` / `mask_text_constant_tokens`
  (grid_lang_common.py) resolve them and scrub them from dependency extraction;
  interpolation pre-resolves braced forms to an inert sentinel so text running
  into a `{...}` (e.g. `L1{#nl}`) isn't misread as an array access, restoring
  the character in `_process_interpolation`.
- **Equality binding vs `Push` value**: `For x = expr`, `Let x = expr`,
  `: x = expr`, `[cell] := expr` and `[A1] : f = expr` are **equality
  bindings** — the variable becomes a *client* of its dependencies: it
  registers listeners on them (`_register_listeners`) and derives its value
  from `expr`, so **declaration order does not matter** (unresolved deps park
  the binding in `pending_assignments` until they resolve, then
  `_bind_declared_var` binds it). The binding is stored as
  `constraints['constant']`. `Push x = value` instead marks x as **set_by** the
  pushing statement: the value is stored and propagated to dependents
  (`_propagate_wave`) with no equality check and no order dependency.
- **Transient `#VALUE` wave**: a `Push` (or other write) onto a
  constant-bound variable whose value differs from the equality expression is
  **rejected before overwrite** (`constant_value_matches` in `_set_var_value`).
   The stored value is temporarily swapped for a `#VALUE` error and
   `_propagate_transient` (compiler.py) ripples the error through the
   listeners, then restores the original value. Derived clients re-evaluated
   during the wave store a **durable** `#VALUE`; cell mirrors (`[cell] := ...`)
   are recomputed during the wave too, so a mirror written before the rejected
   push absorbs the error and reads `#VALUE`, while one written afterwards
   sees the restored source (`_recompute_client`). `_in_cell_spill_recompute`
   stops mirrors from writing back while they recompute.
- **Binding vs publishing**: a *binding* is an ongoing constraint that keeps
  re-deriving (`Let x = e`, `: x = e`, `For x = e`, `[addr] := e`). A `Push` is
  **not** a binding — it is a stream update / message publish. It stores a
  value and notifies dependents; it declares nothing and honours no prior
  constraint except to reject a contradicting one (transient `#VALUE` above).
- **Binding scope decides survival**. What a binding writes determines how long
  it lives:
  - `Let` — a local variable, **dropped** when the block exits.
  - `For` — the scope is the body of the function/constructor/subprocess;
    **dropped** on exit.
  - `:` — in a constructor or subprocess this is a *field of the object*;
    otherwise it behaves like `For`.
  - `[addr]` — a field of the implicit `grid` object, i.e. it follows the `:`
    rule; `grid` is a default field.
  - A subprocess may **publish to** an external variable but may not **bind**
    it; other constructs may not even publish outward. A subprocess's object is
    the value the call returns, and that is forced to be published, so it is
    an inert copy — a `When` does not survive it. For a constructor the
    instance may stay live, so a `When` could fire after exit if its guard is
    still in scope (not yet implemented). A Builder sees object fields the way a
    subprocess sees external variables: it can publish, not bind.
- **Output is a destination, never an input**: `Output o` may be the target of
  `Push o = ...` (and of a call's output-parameter binding), but naming it
  anywhere else is a read and a **compile error** — `Return o`,
  `Print o`, `Let x = o` are all rejected by `_reject_output_reads`
  (`compiler.py`), which scans each function/subprocess/builder body at
  definition time. `Output`/`Input` declaration lines are exempt, as is a name
  inside a string literal or comment, and a dotted `p.o` field access is not
  the output `o`.
- **Live I/O does not notify**: a live input refreshing (`Input p as T live`,
  re-evaluated from the caller-side source before each statement) or a live
  output writing through to its caller binding republishes a value the callee
  has already observed. Neither cascades into re-running dependents —
  `_notify_var_changed` returns early while `_suppress_live_notify` is set
  (around `_refresh_live_inputs` and `_maybe_publish_live_output`). Only a
  normal assignment or `Push` notifies, so a mirror over a live input is
  evaluated once, not once per refresh.
- **Block execution order**: every `do`/`then` block runs **guards → bindings →
  ordered effects and nested blocks**, recursively. Bare `If <cond>` is a guard
  (no `then`, no `End`) gating its whole containing body; a dependency that is
  an error makes the guard false, and multiple in-block guards are AND-ed. An
  `If ... then ... End` is a structural block. Statements may be **reordered**,
  so binding source order is irrelevant; a name that is not yet bound parks its
  binding and is retried once the name resolves (see `_resolve_pending_assignments`
  and the `__if_line_*` deferred-IF retry). Because a block body only installs
  its cell mirrors (`_pre_register_cell_spill_constraints`) **when the block
  actually executes**, a body whose guard fails installs no binding at all — the
  pre-scan is deliberately skipped for block-body lines
  (`pre_scan_blocks` records `block_body_lines`).
- **A cell mirror binds every free variable in its RHS**, not one "root":
  `[C6] := a * b` re-evaluates when `a` *or* `b` is published to
  (`_cell_spill_dependencies` → `_register_cell_spill_listener`). The spill
  root (base of a dotted, called or indexed reference) is included too, so
  `[C1] := grid{a}` still reacts to `grid`.
- **`When` is event-driven off publish queues**: `_register_when_block`
  extracts the condition's dependencies and `_process_when_triggers` fires the
  block only when **every** dependency has a queued value, popping one per
  dependency and pairing them one-to-one (3 publishes on `x` plus 1 on `y`
  fires once, not three times). The condition is then evaluated; a false
  condition runs nothing. Registration order does not matter — a `Push` before
  the `When` is still queued when it registers. Chains work
  (`When x` → `Push y` → `When y`), and a granted `Ticker` drives its `When`
  through `_maybe_fire_tickers`. Re-triggering a `When` from its own body (or
  mutually, between two `When`s) **loops forever by design** — that is the
  `while true` of this language, not a bug.

- **Output channels**: `Print <expr>` is the **ambient console output**
  statement — legal in any scope. It is dispatched in
  `_handle_main_loop_quick_statements` (top level, executor.py) and in
  `control_flow._handle_block_tail_statements` (block bodies), and both handle
  via `_append_console_output` → `output_values['_console']`; function and
  subprocess sub-calls propagate their `_console` up to the caller
  (`call_function`/`call_subprocess`), and `_print_outputs` renders it. Users
  cannot declare variables beginning with `_`, so `_console` never collides
  with a user output variable. `Return <expr>`
  is the **call-return channel**, restricted to functions and operations:
  `_run_setup` rejects any `Return` left in the top-level program stream
  (whole-line forms) as a load error pointing at `Print`
  (`executor._validate_no_top_level_return`); inline block statements carrying
  a payload (Return or any single instruction) are first rewritten into real
  block form by `compiler._normalize_inline_blocks`, so their generated
  `Return` line is caught by the same check. Function/subprocess runner
  sub-compilers set `_is_operation_runner` to opt out.
- **Units**: `5 of m`, `"ox" of animal`, `2 of 1` → `UnitValue(value, unit)` via `gridlang_of_unit`; `of 1` is dimensionless. `Define B as UnitSource of Meat` (the `of K` form, reusing the unit syntax) or legacy `UnitSource(Meat)` / `Convert "ox" of animal to "beef"` (constant) or `Convert x as number of cm to x/100` (formula, LHS var stripped). `Convert` target for top-level is inferred by evaluating RHS with stripped var (`compiler.py:273`). `:` fields (`: f of m`) convert on `with` (`compiler.py:1228`). `Push`/`Input` preserve `of` unit. `1` handling: `m*1→m`, `1*1→1`, `m/1→m`, `m/m→1` (`units.py`).
- **Types**: `Define T as Type ... End T`, `new T with (field = v, ...)`.
  Types carry computed fields and constraints.
- **Resources / Require**: capabilities are **grants** — `Require name as
  Ticker` grants a *handle* (Timer/Counter) or the ticker clock itself; the
  grant is resolved by `permissions.RequirementResolver` (executor.py:4811)
  and binds `grant_map`/`bindings` in scope. Resources are **declarations
  only**: `register_resource` (builtin_functions.py:116) seeds the returned
  metadata into `RESOURCES`; `_seed_predefined_resources` (compiler.py:4397)
  copies it into `self.types_defined` once per engine, **surviving per-run
  resets** (`_reset_state`) like type definitions. Handle types (`Timer`,
  `Counter`) are declared with `register_handle` (builtin_functions.py:178),
  never grantable — a handle is only produced by calling a resource member
  (`Ticker.Timer(tick, n)`), and its state is advanced entirely engine-side
  (`_advance_handles`, executor.py).
  - **Engine-owned dotted creators are frozen**: names like `ticker.timer`,
    `ticker.counter`, `number.round` (dots over the pre-seeded `ticker`
    resource) and the predefined subprocesses `ticker.stop`/`ticker.start`
    live in `_DOTTED_ENGINE_CREATORS` / `_PREDEFINED_SUBPROCESSES`
    (compiler.py:79/89, derived from `BUILTINS`) and **cannot be redefined
    by a program**. `Define ticker.counter as Handle` is rejected at parse
    (compiler.py:3595-3620) *before* any truncation to the bare resource
    prefix, so `ticker` itself is never clobbered.
- **Net resource**: grant/handle semantics live in `builtin_functions.py` (declaration + connection/server helpers); `net.py` provides URI matching and transport utilities for a grant whose `methods` are exactly
  `{"LISTEN"}` is a **server** grant — mixing LISTEN with client verbs is
  refused at grant-merge time (permissions.py). The granted address is the
  bind target; no path template is needed for serving (`/**` is meaningless),
  routing lives entirely in `Route` templates. `web.server()` validates the
  address, checks LISTEN-only mode (`net.check_server_mode`) and returns a
  `net!server` handle carrying the granted headers as *pinned* response
  headers. `srv.Route("/path")` registers a template (`*`/`**` wildcards);
  `route.endpoint("doc")` pins the first wildcard (`/search/*` + `doc` =>
  `/search/doc`, appended when no wildcard exists). A bare
  `srv.Start()`/`srv.Stop()` statement is a side-effecting subprocess call
  (`executor._bare_handle_member_call`): engine handles own their members, so
  such calls execute instead of being no-op reads (same for `rq.Reply(...)`). `Start()` binds and hands inbound requests to the
  engine's queue (`executor._pump_server_requests`, drained each main-loop
  unit), `Serve()` blocks pumping until `Stop()`. Matching requests re-write
  the route handle with a `net!request` under `r.request` and fire the
  route's `When r do ... End`; `rq.Reply(status, body, headers)` answers the
  request and unmatched requests get an automatic 404. The inbound `net!request`
  is a readable object like the client-side `net!response`, not a text blob: the
  body is pulled off it with `rq.ReadAll()`/`rq.readall()` (`net!request.ReadAll`
  (builtin_functions.py:947), alias `readall`), mirroring `resp.ReadAll()`, so a
  future streaming operation acts on the handle instead of re-parsing text. The
  handle exposes a `started` logical field mirroring Start/Stop. A member call that mutates a
  handle notifies the base variable (`expression._try_eval_member_function_call`
  → `_notify_var_changed`), so equality-bound cells (`[C1] := srv.started`)
  re-derive live and always read the field's current state. Pinned grant headers
  are protected case-insensitively (RFC 7230) — an override of any naming is
  #PERM — and header names go on the wire with the case the author gave them.
  The real transport is `net.bind_server`/`_TCPServerListener`; tests replace
  it with a synchronous fake (test_runner `_FakeServerListener`).
- **Dependency extraction**: `_DEPENDENCY_IGNORED_TOKENS` / `_strip_constraint_operands`
  live in `grid_lang_common.py` and are re-exported by both layers — single
  source of truth. `_strip_constraint_operands` strips `of`/`as`/`dim`/`not null`/`of 1`/`"ox" of animal`
  clauses so they aren't mistaken for variable refs.
- **Functions/Subprocesses**: extracted from code, run in a fresh
  sub-`GridLangCompiler`; functions can read but not write the parent grid
  (`_outer_scope_read_only` flag + `_ACTIVE_RUNNERS` guard). A subprocess call
  has side effects, so execution order matters. Only `init`/`Push`/`Return`
  may bind a subprocess call's result — each deep-copies the returned
  `SubprocessResult` (read via `.grid`, e.g. `For myword init Sub("DIRG")`
  then `[A1] := myword.grid`, or `Push snap = Sub(7)` then `[^C1] :=
  snap.grid`). Equality bindings to a call (`For x = Sub(...)`,
  `Let x = Sub(...)`, `[cell] := Sub(...)`) are rejected at runtime via
  `_leading_subprocess_call` (executor.py) and the guarded check in
  `array_handler.evaluate_line_with_assignment` — "use INIT instead".
- **Live pipes**: `Input … live` / `Output … live` (compiler.py
  `_carve_live_suffix`, `_route_live_call`, `_refresh_live_inputs`,
  `_maybe_publish_live_output`) turn a function/subprocess parameter into a
  live channel. Arguments are matched **in declaration order** when the
  signature has any live slot: a live `Output` writes through to the caller's
  binding every time it is pushed (and its `init` seeds immediately), a live
  `Input` keeps the caller's expression and re-evaluates it in the caller's
  scope before each callee statement. Live outputs are excluded from the call
  value (non-live `Output`s form the return channel; `Return` still passes the
  call value). Signatures with no live slots keep the legacy argument routing.
  Write-through runs through the push funnel
  (`_handle_push_assignment`, executor.py) with a `_live_write_through`
  bypass of the read-only guard (scope.py); live inputs relax the
  input-update guard for pipe-fed names. This is what makes callbacks like
  `findmatches(liste, c, d, k(c, d))` work — see docs/modules.md §8.1.

## Scratch / auxiliary files

- `test_convert.grid` — **unit conversion smoke test** (`Butcher` category, `SILength` numeric, `dist + 500 of in` → `4012.7` via `Push`; `Convert x of in to x*SILength.inch` infers `m`).
- `example.grid` — tensor example (`For V as tensor with (name=..., grid DIM{...}=var)`,
  `V.grid{...}` reads). Good smoke test for array/dim features.
- `test.grid`, `testassign.grid`, `testbool.grid`, `testmdim.grid`,
  `test_constraints_cells.grid`, `foo.grid` — informal scratch programs used
  during development (not part of the runner). `testmdim.grid` is a useful
  array type-check repro: `b as number dim {2, 2, 2}` with string elements
  now outputs `#TYPE/I` (see array base-type checking, Tests 245–254).
- `Documentation_Tests/helloworld_basic.grid`, `helloworld_calc.grid` — empty
  placeholder files.
- `Documentation.md` — language tutorial (types, grid, arrays, constraints,
  variables, push). README.md — install/usage. `docs/modules.md` — the module
  design spec (use/Version/Export, Print/Return channels, closure rules).
  `LICENSE.md` — LGPLv3.
- `.opencode/summaries/previous-summary.md` — notes from an earlier working
  session (predefined `grid` variable work, Tests 191–200). Read it when
  resuming that thread; later sessions removed pyarrow (see commit
  `036c261` "Replace pyarrow with Python list and dict").
- `build/`, `venv/`, `.venv/`, `gridlang.egg-info/`, `gridlang/` (empty) —
  generated/env dirs, gitignored except `gridlang/`. The `grid` CLI is
  installed in `venv/` and `.venv/`.

PYEOF
