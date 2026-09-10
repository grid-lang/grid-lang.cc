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
3. **Run `python3 test_runner.py` (369 tests) and `python3 main.py test_convert.grid`**
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

# Run the inline test suite (369 tests)
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
2. `compiler.run(code, args)` (`compiler.py:707`) is the interpreter entry
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
6. Results: `Return x` appends to `output_values` (printed by
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

### `main.py` (70 lines) — CLI entry point
- `_parse_cli_args`: splits `--debug` / `--` / positional program args.
- `run_grid_program`: reads the `.grid` file, builds a `GridLangCompiler`, sets
  `prompt_missing_inputs`, calls `compiler.run`, and on `--debug` calls
  `compiler.export_to_csv`.
- `main()` is the console-script entry (`grid=main:main` in setup.py).

### `setup.py` — packaging
- Console script `grid=main:main`; declares the 10 top-level modules as
  `py_modules`; **no `install_requires`** (`pyarrow` was removed); LGPLv3.

### `units.py` (528 lines) — unit wrapper + conversion registry
- `UnitValue` + sticky `#UNIT` errors + `CONVERSIONS` dict `(src.lower(),dst.lower())->[entries]`
  (`register_conversion`/`lookup_conversions`/`has_conversion`/`apply_conversion`).
  `1` is special: `UnitValue._is_one`, `__mul__` treats `1` as unitless, `_divide` `m/1→m` `m/m→1`, `__pow__` allows exponent `1`.
- `UnitValue` overloads: `+`/`-` same unit or one side unitless; `*` with `1`; `/`/`\`/`mod` with `1`; `^` with `1`.

### `compiler.py` (4508 lines) — state + orchestration, public API
`class GridLangCompiler` is the engine's **state holder and public surface**. It
is the only class `main.py` constructs. It inherits the runtime loop from
`GridLangExecutor` and holds nearly all persistent state created in `__init__`:
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
- `run` (707): engine entry — `_reset_state()` then delegates to the inherited
  runtime pipeline (no executor handoff).
- `current_scope`/`push_scope`/`pop_scope` (310/325/331).
- `_seed_grid_variable` (2945): predefines `grid` in the global scope as a
  `_GridStore` (sparse array keyed by 0-based `(row,col)` tuples; aliased as
  `self.grid`), so `grid{row, col}` works at top level. Skipped inside
  read-only function sub-compilers.
- UnitSource: `_parse_unit_source_header` (182) / `_finalize_unit_source` (195) / `_register_convert_line` (230) / `_infer_convert_target_unit` (262, evaluates RHS with stripped var, falls back to declared units) / `_materialize_unit_source_constants` (337) / `_register_top_level_converts` (331).
- `_extract_functions` (740): pulls `Function`/`Subprocess` defs out of the
  main code and registers them.
- `_instantiate_type` (1148), `_evaluate_with_value` (1383), `_apply_with_clause`
  (1246): type/`with` object construction; now handles `:` field unit conversion.
- `call_subprocess` (1829): runs a sub-`GridLangCompiler` in isolation.
- `_process_grid_assignment` (3347), `_process_declarations_and_labels` (3359),
  `_collect_global_declarations` (3435): top-level statement handling; now handles `of 1` and `"ox" of animal`.
- `export_to_csv` (4301): `--debug` CSV export (grid as matrix, or outputs as
  one column when the grid is empty).
- `set_input_values` (4324): binds CLI/keyboard args to `Input`s; now evaluates `"5 of in"` before `update` so `Input a of m` converts.
- `_seed_globals` (1670): for sub-compilers; **skips redefining `grid`**.

Also defines `SubprocessResult` (32): result container exposing `grid`,
`variables`, `outputs`, and `_UnitSourceNamespace` (47) the UnitSource lookups.

### `executor.py` (5251 lines) — the runtime loop
`class GridLangExecutor` is the **base class that owns the interpreter's
dispatch loop and its per-run runtime state**. It is not instantiated directly
as a facade (the old compiler→executor copy handoff was removed); `run` is the
live entry. Key methods (the `compiler.py`/`grid_lang_common.py` layers call
`super()`/override these):
- `run` (2104): top-level sequence (acts on `self`; see Architecture).
- `_run_setup` (4476), `_run_prepare_execution` (4511), `_print_outputs`
  (4716), `_materialize_inits` (4992), `_process_deferred_assignments` (5131).
  `_run_prepare_execution` now calls `_materialize_unit_source_constants` + `_register_top_level_converts`.
- Main loop: `_run_main_loop` (2134) → `_run_main_loop_impl` (2671) →
  `_run_main_loop_impl_body` (2694). `_handle_main_loop_*` methods dispatch
  statement kinds: `Let` (1128), `For` (1752 fallback), grid assignment (3903),
  `When` blocks (3956), `Push` (4152), `Return` (4108), misc (3724).
- Dependency/guard machinery: `_build_dependency_network` (700),
  `_determine_needed_lines`, `_evaluate_guard_conditions`,
  `_evaluate_global_guards_pre_execution`, `_execute_global_for_loops`,
  `_attempt_resolve_pending_var`, `_resolve_ready_pending_vars`.
- `Let` semantics: first pass `_process_let_first_pass` (1224), binding
  `_bind_declared_var` (1255, now `expected_unit`), standard assignment,
  second pass, generator values, `_materialize_inits`.
- `For`: `_execute_simple_for_assignment` (749, now `expected_unit`), `Push` via `target_unit`.
- `Push` semantics: `_handle_push_assignment` (4726), `_evaluate_push_expression`
  (4345, now `expected_unit`), `_handle_push_assignment_line`.
- `When` blocks: `_register_when_block` (275), `_process_when_triggers` (394),
  `_run_when_block` (410).
- Shared helpers (from `grid_lang_common.py`): `_strip_constraint_operands` and
  `_DEPENDENCY_IGNORED_TOKENS` (re-exported here as `DEPENDENCY_IGNORED_TOKENS` for
  back-compat; compiler.py imports the same `_DEPENDENCY_IGNORED_TOKENS`). They are
  now a single source of truth in `grid_lang_common.py`, shared by both layers.

### `grid_lang_common.py` (164 lines) — shared base + helpers
`class GridLangBase` (124) is the common ancestor of `GridLangExecutor` and thus
`GridLangCompiler`. It holds shared helpers used across layers: `_STATEMENT_KEYWORDS`,
`_first_keyword`, `_IDENTIFIER_TOKEN_PATTERN`, `_STRING_LITERAL_PATTERN`,
`_DEPENDENCY_IGNORED_TOKENS`, `_strip_constraint_operands` (41),
`_strip_builder_arrows`, `_strip_cell_address_tokens`, `_has_star_dim`,
`_apply_dim_base_offsets`, `_infer_declared_type`. Put methods/data needed by
both the compiler and executor layers here. (Note: `error_value`/`UNIVERSAL_ZERO`
live in `units.py`, not here.)

### `expression.py` (3894 lines) — expression evaluation
`class ExpressionEvaluator` evaluates RHS expressions, arrays, ranges, sums,
dimension selectors, interpolations, member/field access, and Python-fallback
evaluation.
- Entry points: `eval_or_eval_array` (116, now `expected_unit`), `eval_expr` (2358), and for
  assignments `_evaluate_array` (546).
- LHS-informed: `_apply_expected_unit` (78) + `_formula_eval` (97, strips LHS var) threaded via `expected_unit`.
- `eval_expr` is the big recursive dispatcher: array literals `{}`, pipes `|`,
  interpolated cell refs, paren/curly indexing, member calls, user function
  calls, object creation, field access, address-indexed access, then scalar
  constructs, then simple variables.
- Python fallback: `_evaluate_with_python_fallback` (2793) builds a scope and
  `eval()`s complex arithmetic (`_build_fallback_cell_scope` 2254,
  `_eval_python_fallback_result` 2501, `_get_eval_globals` 2937). Now includes
  `_replace_of_unit_literals` (3824) for `"ox" of animal`/`5 of in`/`2 of 1` → `gridlang_of_unit` and `Attribute` flat-key `SILength.inch` + case-insensitive `_resolve_fallback_name` (3380).
- Interpolation: `_process_interpolation` (3696). Operators:
  `_replace_operators` (3839).
- Grid indexing: `_replace_grid_indexing` (863) — rewrites bare `grid{...}`
  when the predefined `grid` variable is in scope; otherwise falls through to
  the generic array-access path (ranges, `*` selectors, `#N/A` for unset cells).
- Also `CaseInsensitiveDict` (28): case-insensitive dict used for
  eval scopes.

### `builtin_functions.py` (402 lines) — builtin registry + predefined functions
Single source of truth for every predefined GridLang function (`SUM`/`MIN`/`MAX`/`ROWS`/`ABS`/`LEN`/`TEXTSPLIT`/`TRANSPOSE`/math, …). Both evaluation paths (`ExpressionEvaluator._build_python_fallback_scope` (`expression.py:3229`) and `_get_eval_globals` (`expression.py:3853`)) call `get_builtin_functions()` so a new builtin is instantly available everywhere.
- Registry (37): `BUILTINS`/`ALIASES`/`ARG_COUNTS`/`VECTORIZED` plus `KEYWORDS` (43, re-exported by `compiler.py`/`expression.py` for dependency stripping). `register_builtin` (50) and `register_vectorized_builtin` (80) are decorators (`@register_builtin("MIN", aliases=["Number.Min"], arg_count=1)`); non-vectorized by default, vectorized ones broadcast element-wise.
- Arity + helpers: `_check_arity` (100) / `_resolve_builtin` (120) / `_to_list` (129, normalizes sparse `dict`, `{'array':…}`, `list` → flat list).
- Builtins (143): `ROWS` (143) / `SUM` (155) / `LEN` (166, `Text.Len`) / `ABS` (174, `Number.Abs`) / `POWER` (179) / `INT` (184) / `MID` (189, `Text.Mid`) / `TEXTSPLIT` (199, `Text.Split`) / `COUNTA` (207) / `RANDARRAY` (223) / `SORTBY` (229) / `TRANSPOSE` (240) / `MIN` (291, `Number.Min`) / `MAX` (299, `Number.Max`) plus math shims (`sqrt`/`sin`/… → `Number.<name>`, 306) and `str`/`int`/`float`/`abs` (313). `MIN`/`MAX` flatten via `_to_list` before `min`/`max`.
- Public API (323): `_is_array_arg` (323) / `_broadcast_builtin` (326, only for `VECTORIZED`; detects `list`/`{'array':…}`/sparse, checks equal lengths, applies scalar broadcast, otherwise returns `None` to fall back to normal call) / `get_builtin_functions` (373, builds `name→wrapped` dict; wrapper does `_check_arity`, then `_broadcast_builtin`, then `fn(*args)` with `TypeError→#TYPE/I`; injects `math`).

### `array_handler.py` (3054 lines) — grid/array/tensor operations
`class ArrayHandler` centralizes all array knowledge:
- Cell addressing & lookup: `resolve_cell_index` (65), `cell_ref_to_indices`
  (129), `lookup_cell` (1563), `get_range_values` (1253/1286),
  `_lookup_extended_address` (131367, `_write_extended_tensor` (1648).
- Assignment: `evaluate_line_with_assignment` (296),
  `_parse_assignment_target_details` (26257, `_perform_assignment_write` (598),
  `_assign_horizontal_array` (98984, `assign_range` (1339),
  `_assign_extended_range` (121221, `_assign_index_selector` (944),
  `_assign_dim_selector` (85837, `_update_bound_array_cell` (1139),
  `assign_implicit_intersection_range` (60597, implicit-intersection rewrite
  (`_rewrite_implicit_intersection` 597).
- Spilling helpers: `_resolve_spill_unset` (1545) replaces `None` sentinels
  in flat arrays before writing to grid (uses variable default or `#N/A`);
  `flatten_array` (1918) column-major flattens any array to 1D;
  `flatten_object_fields` (1859) flattens object fields for grid spills.
- Array construction/shape: `create_array` (1773, accepts `template=True` to
  fill with `None` sentinels), `create_object_array` (2188),
  `get_array_shape` (2086), `reshape_array` (2580), `infer_type` (1966, now handles `UnitValue("beef")→text`), 
  `fill_array` (2983), `flatten_object_fields` (1859), `flatten_array`
  (1443), `_nested_from_flat` (2006), `to_display_value` (1963).
- Constraints/dims: `set_labels` (2203), `check_dimension_constraints` (2294),
  `validate_array_element_types` (161694 — element-level base-type checking
  (`as number`/`as text` arrays reject mismatched scalars), `_dim_size` (2230).
- Grid-as-array: `get_grid_row` (2636), `get_grid_column` (2665); generic
  `get_array_element`/`set_array_element` handle `_GridStore` like any sparse array.

### `control_flow.py` (2112 lines) — blocks: For / If / Let / When
`class GridLangControlFlow` executes block constructs. Module-level regexes
(9–16) define `if...then`, `elseif...then`, `else`, `for...do`, `while...do`,
`when...do`, `end`.
- `process_for_statement` (11118: For-loop handling (ranges, init, arrays).
- Block engine: `_process_block` (878), `_extract_block_body` (312),
  `pre_scan_blocks` (181833, `_prepare_block_line` (342).
- If: `_process_if_statement` (936) and the "new"/"rich" variants (2011,
  2113), `_parse_if_header` (969), `_collect_if_blocks` (1007),
  `_execute_if_block_choice` (111186, `_process_if_elseif_else_block` (1878);
  condition evaluation helpers `_evaluate_if_*` (1341–1696).
- Let: `_process_let_statement_inline` (1182), field/index assignment helpers
  (1339, 1390).
- `_handle_block_*` methods (343–960): per-statement handling inside blocks.

### `scope.py` (1307 lines) — scope + variable semantics
- `class Scope` (11119: variable storage with constraints.
  - `define` (498), `update` (591), `get` (741), `is_uninitialized` (762),
    `get_defining_scope` (777).
  - Inputs/outputs: `define_input` (790), `define_output` (590, now preserves `unit` from `Input` when `Output` shares name), `is_input`/`is_output` (518/527), `connect_pipe` (837), `push_value`
    (555), `_propagate_wave` (879) — the publish/listen ripple.
  - Unit handling: `_unit_convert` (104, now tries `apply_conversion` before `#UNIT`), `_has_pending_assignment` (155).
  - Constraints: `_re_evaluate_constraints` (932), `_check_constraints` (830, now unit-aware for `Let y of m = 5 of in`), `_validate_base_type` (1003) — validates scalars AND, since the pyarrow
    removal, element-by-element base types of `dim` arrays (via
    `array_handler.validate_array_element_types`), `_expression_depends_on`
    (651). `_array_unset_value` (array_handler.py:1507) resolves `None`
    sentinels for unset template array cells: checks the variable's
    `constraints['default']` (from `or = <expr>`) and evaluates it; falls back
    to `error_value(NA_ERROR)` (`#N/A`).
  - Scoping: `is_shadowed` (893), `get_evaluation_scope` (901),
    `get_full_scope` (1297), `_coerce_custom_type_value` (249).
- `class _GridStore` (37): dict backing `compiler.grid`; every cell write
  calls `compiler._notify_cell_changed` (compiler.py:2730). (The old
  `_ListenerGrid`/`GridLiveView` classes were removed — `_GridStore` is the
  single grid store, keyed by 0-based index tuples.)
- `_ACTIVE_RUNNERS` (22): stack of executing compilers; used with the
  `_outer_scope_read_only` flag to reject writes from read-only function
  sub-compilers to outer scopes.

### `type_processor.py` (1340 lines) — `Define X as Type` handling
`class GridLangTypeProcessor`:
- Type-def parsing: `_parse_type_def` (82), `_parse_type_def_line` (90),
  `_extract_type_field_line` (9696, `_parse_type_field_constraints` (194),
  `_record_type_field_definition` (15157, `_collect_type_computed_fields`
  (178), `_finalize_type_def_state` (281).
- Executing type body code against an instance: `_execute_type_code` (304),
  `_execute_type_block` (25252, `_process_grid_assignment` (529),
  `_process_type_for_loop` (38389, `_process_type_let_statement` (784),
  `_process_type_assignment` (58581.
- `_build_type_eval_scope` (70708, `_execute_builder` (1152).

### `parser.py` (648 lines) — variable-definition parsing
`class GridLangParser`:
- `_parse_variable_def` (1616: the central parser for `: name [as type] [of
  unit] [dim ...] [constraints] = expr` / `Input`/`Output` lines. Returns
  (parsed_var, parsed_type, constraints, expression).
- Constraint handling: `_check_comparison_series` (288),
  `_match_direct_assignment_patterns` (22225, `_apply_with_clause` (325),
  `_apply_dimension_constraints` (33334, `_merge_custom_type_constraints` (484),
  `_split_on_keywords` (39390, now `seen_equals`/`seen_init` keep `of`/`as` in RHS `5 of in`/`Init 5 of in`), `_parse_dim_size` (614). The `or` keyword in
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

### `test_runner.py` (1167 lines) — inline test suite
`class GridLangTestRunner` with `run_tests_independent(tests)` — runs 369 tests (Tests 282–293 unit tests, Tests 331–345 push/cell-mirror/subprocess semantics). At the bottom of the file (~840) it runs itself when executed directly:
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
  `_assign_horizontal_array` (`array_handler.py:1217`);
  `_assign_extended_address` (`array_handler.py:1315`) handles dotted targets
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
  listeners while `_transient_active` is set, then restores the original
  value. Derived clients re-evaluated during the wave store a **durable**
  `#VALUE`; cell mirrors (`[cell] := ...`) snapshot their cells and restore a
  previously non-error value after the wave (`_recompute_client`), so mirrors
  end at their last consistent value. `_in_cell_spill_recompute` stops mirrors
  from writing back while they recompute.
- **Units**: `5 of m`, `"ox" of animal`, `2 of 1` → `UnitValue(value, unit)` via `gridlang_of_unit`; `of 1` is dimensionless. `Define B as UnitSource(Meat)` / `Convert "ox" of animal to "beef"` (constant) or `Convert x as number of cm to x/100` (formula, LHS var stripped). `Convert` target for top-level is inferred by evaluating RHS with stripped var (`compiler.py:273`). `:` fields (`: f of m`) convert on `with` (`compiler.py:1228`). `Push`/`Input` preserve `of` unit. `1` handling: `m*1→m`, `1*1→1`, `m/1→m`, `m/m→1` (`units.py`).
- **Types**: `Define T as Type ... End T`, `new T with (field = v, ...)`.
  Types carry computed fields and constraints.
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
  variables, push). README.md — install/usage. `LICENSE.md` — LGPLv3.
- `.opencode/summaries/previous-summary.md` — notes from an earlier working
  session (predefined `grid` variable work, Tests 191–200). Read it when
  resuming that thread; later sessions removed pyarrow (see commit
  `036c261` "Replace pyarrow with Python list and dict").
- `build/`, `venv/`, `.venv/`, `gridlang.egg-info/`, `gridlang/` (empty) —
  generated/env dirs, gitignored except `gridlang/`. The `grid` CLI is
  installed in `venv/` and `.venv/`.

PYEOF
