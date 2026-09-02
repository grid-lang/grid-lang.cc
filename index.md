# GridLang Codebase Index

This file is a developer map of the `grid-lang.cc` repository. It describes each
source file, what it owns, how the pieces connect, and the conventions a task
must respect. Read this before editing; don't rediscover the codebase.

## What the project is

GridLang is a **line-based interpreted language** for tabular data ("grids").
Source files end in `.grid`. The interpreter is pure Python with **no runtime
dependencies** (`pyarrow` was used historically for arrays but has been fully
removed). The CLI `grid` is produced from `main.py`. The README calls it a
"compiler" but there is **no tokenizer/parser/AST**: execution is a
statement-dispatch loop over source lines with regex-based parsing throughout.

Recent addition (2026-09): **Unit system** (`units.py`, `test_convert.grid`).
Values can carry units (`5 of m`, `"ox" of animal`, `2 of 1`), `Define X as
UnitSource(Target)` + `Convert` registers category (`"ox" of animal to "beef"`)
or formula (`x as number of cm to x/100`) mappings, `Input`/`Let`/`For`/`:`/`Push`/`Init`/`Output`
with `of <unit>` automatically convert via LHS `expected_unit` (`ExpressionEvaluator._apply_expected_unit`)
and central fallback `Scope._unit_convert` (`units.apply_conversion`). Special
unit `1` is dimensionless: usable instead of unitless for `*`/`^`, `m/1 → m`,
`m/m → 1` for `/`/`\`/`mod`.

Language reference: `Documentation.md` (tutorial style). Install/usage docs:
`README.md`.

## Running things

```bash
# Run a program (args after filename are program inputs)
python main.py example.grid 42
python main.py example.grid --debug   # also exports <file>.csv

# Run the inline test suite (313 tests)
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
2. `compiler.run(code, args)` (`compiler.py:556`) creates a throwaway
   `GridLangExecutor`, **copies the compiler's state attributes AND every
   public method onto it**, and calls `executor.run()`. During a run the
   executor and compiler are effectively the same object; helper engines
   (`expr_evaluator`, `array_handler`, `control_flow`, `type_processor`,
   `parser`) were already constructed on the compiler and are shared.
3. `executor.run()` (`executor.py:1982`) is the interpreter entry point:
   `_run_setup` → `_run_prepare_execution` (now materializes `UnitSource`
   constants and registers top-level `Convert`s) → `_run_main_loop` →
   `_resolve_pending_assignments` → `_process_deferred_assignments` →
   `_print_outputs`.
4. `_run_main_loop_impl_body` dispatches each line by statement kind: `For`,
   `Let`, `Push`, `When`, `Return`, grid assignment (`[A1] := ...`),
   declaration, or "misc". Big dispatch chains in executor.py and
   control_flow.py.
5. Unit flow: `_replace_of_unit_literals` rewrites `5 of m` / `"ox" of animal` / `2 of 1`
   → `gridlang_of_unit(...)` → `UnitValue`; `For`/`Let`/`Push` pass `expected_unit`
   from LHS `of T`, `Scope._unit_convert` falls back to `apply_conversion`.
6. Results: `Return x` appends to `output_values` (printed by
   `_print_outputs`); grid writes land in `compiler.grid` (a dict keyed by
   cell refs like `'A1'`). `--debug` → `compiler.export_to_csv` (`compiler.py:3874`).

The single most important design fact: **`GridLangCompiler` (state holder) and
`GridLangExecutor` (loop) share one object during execution.** Many helpers
exist in BOTH files (dependency analysis, when-blocks, pending vars, push
handling) — check both before adding a feature so you extend the live path.

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

### `units.py` (526 lines) — unit wrapper + conversion registry
- `UnitValue` + sticky `#UNIT` errors + `CONVERSIONS` dict `(src.lower(),dst.lower())->[entries]`
  (`register_conversion`/`lookup_conversions`/`has_conversion`/`apply_conversion`).
  `1` is special: `UnitValue._is_one`, `__mul__` treats `1` as unitless, `_divide` `m/1→m` `m/m→1`, `__pow__` allows exponent `1`.
- `UnitValue` overloads: `+`/`-` same unit or one side unitless; `*` with `1`; `/`/`\`/`mod` with `1`; `^` with `1`.

### `compiler.py` (4088 lines) — state + orchestration
`class GridLangCompiler` is the **persistent brain** and holds nearly all state
created in `__init__`:
- Grid & scoping: `grid` (`_ListenerGrid`), `scopes` (stack of `Scope`),
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

Notable methods (all copied onto the executor during a run):
- `run` / `_run_inner` (323284345: the executor handoff described above.
- `current_scope`/`push_scope`/`pop_scope` (301/316/322).
- `_seed_grid_variable` (181811: predefines `grid` in the global scope as a
  `GridLiveView`, so `grid{row, col}` works at top level.
- UnitSource: `_parse_unit_source_header` / `_finalize_unit_source` / `_register_convert_line` / `_infer_convert_target_unit` (evaluates RHS with stripped var, falls back to declared units) / `_materialize_unit_source_constants` / `_register_top_level_converts`.
- `_extract_functions` (38389: pulls `Function`/`Subprocess` defs out of the
  main code and registers them.
- `_instantiate_type` (70702, `_evaluate_with_value` (1150), `_apply_with_clause`
  parsing (958+): type/`with` object construction; now handles `:` field unit conversion.
- `call_subprocess` (111128: runs a sub-`GridLangCompiler` in isolation.
- `_process_grid_assignment` (202080, `_process_declarations_and_labels` (3046),
  `_collect_global_declarations` (212168: top-level statement handling; now handles `of 1` and `"ox" of animal`.
- `export_to_csv` (282817: `--debug` CSV export (grid as matrix, or outputs as
  one column when the grid is empty).
- `set_input_values` (282843: binds CLI/keyboard args to `Input`s; now evaluates `"5 of in"` before `update` so `Input a of m` converts.
- `_seed_globals` (~1670): for sub-compilers; **skips redefining `grid`**.

Also defines `SubprocessResult` (48): result container exposing `grid`,
`variables`, `outputs`.

### `executor.py` (5030 lines) — the interpreter
`class GridLangExecutor` contains the main dispatch loop. This is where most
runtime behavior lives. Key methods:
- `run` (191918: top-level sequence (see Architecture).
- `_run_setup` (424220, `_run_prepare_execution` (4341), `_print_outputs`
  (4680), `_materialize_inits` (4771), `_process_deferred_assignments` (4910).
  `_run_prepare_execution` now calls `_materialize_unit_source_constants` + `_register_top_level_converts`.
- Main loop: `_run_main_loop` (2012) → `_run_main_loop_impl` (2539) →
  `_run_main_loop_impl_body` (242455. `_handle_main_loop_*` methods dispatch
  statement kinds: quick statements (1113), `Let` (1147/1494/1515), `For`
  (many: 1728 fallback, 1952 array/dim, 2060 simple, 2099 single-line, 2349
  consecutive shortcuts, 2544 declaration, 2980 range, 3315 nested, 3524
  prechecks, 3589 post-branches), grid assignment (3822), `When` blocks
  (3879), `Push` (3994–4176, now LHS-informed via `target_unit` and `_evaluate_push_expression`), `Return` (4021), misc (3649).
- Dependency/guard machinery: `_build_dependency_network` (626),
  `_determine_needed_lines` (91917, `_evaluate_guard_conditions` (992),
  `_evaluate_global_guards_pre_execution` (71717, `_execute_global_for_loops`
  (827), `_attempt_resolve_pending_var` (1012), `_resolve_ready_pending_vars`
  (1038).
- `Let` semantics: first pass `_process_let_first_pass` (1191), binding
  `_bind_declared_var` (131301, now `expected_unit`), standard assignment (1453), second pass
  (1483), generator values (1607), `_apply_init_values` (1718).
- `For`: `_execute_simple_for_assignment` (818, now `expected_unit`), `Push` via `target_unit`.
- `Push` semantics: `_handle_push_assignment` (4556), `_evaluate_push_expression`
  (4286, now `expected_unit`), `_process_push_call` (4656), `_assign_indexed_target` (4676),
  `_update_member_path_target` (444475.
- `When` blocks: `_register_when_block` (284), `_process_when_triggers` (320),
  `_run_when_block` (31317.
- Shared with compiler.py: `_strip_constraint_operands` (module-level, 26) and
  `DEPENDENCY_IGNORED_TOKENS` (1717 — duplicate of compiler's. Keep in sync.

### `expression.py` (3674 lines) — expression evaluation
`class ExpressionEvaluator` evaluates RHS expressions, arrays, ranges, sums,
dimension selectors, interpolations, member/field access, and Python-fallback
evaluation.
- Entry points: `eval_or_eval_array` (116, now `expected_unit`), `eval_expr` (2181), and for
  assignments `_evaluate_array` (503).
- LHS-informed: `_apply_expected_unit` (76) + `_formula_eval` (97, strips LHS var) threaded via `expected_unit`.
- `eval_expr` is the big recursive dispatcher: array literals `{}`, pipes `|`,
  interpolated cell refs, paren/curly indexing, member calls, user function
  calls, object creation, field access, address-indexed access, then scalar
  constructs, then simple variables.
- Python fallback: `_evaluate_with_python_fallback` (2577) builds a scope and
  `eval()`s complex arithmetic (`_build_fallback_cell_scope` 2254,
  `_eval_python_fallback_result` 2501, `_get_eval_globals` 2937). Now includes
  `_replace_of_unit_literals` for `"ox" of animal`/`5 of in`/`2 of 1` → `gridlang_of_unit` (3483) and `Attribute` flat-key `SILength.inch` + case-insensitive `_resolve_fallback_name` (3067).
- Interpolation: `_process_interpolation` (3347). Operators:
  `_replace_operators` (292923.
- Grid indexing: `_replace_grid_indexing` (820) — only still needed for legacy
  dict-based object grids; it early-returns for `GridLiveView` (which flows
  through the generic array path). `_eval_array_element` uses `base=1` for
  `GridLiveView`.
- Also `CaseInsensitiveDict` (26): case-insensitive dict used for
  eval scopes.

### `array_handler.py` (2976 lines) — grid/array/tensor operations
`class ArrayHandler` centralizes all array knowledge:
- Cell addressing & lookup: `resolve_cell_index` (30), `cell_ref_to_indices`
  (129), `lookup_cell` (1502), `get_range_values` (1253/1286),
  `_lookup_extended_address` (131367, `_write_extended_tensor` (1587).
- Assignment: `evaluate_line_with_assignment` (249),
  `_parse_assignment_target_details` (26257, `_perform_assignment_write` (544),
  `_assign_horizontal_array` (98984, `assign_range` (1278),
  `_assign_extended_range` (121221, `_assign_index_selector` (884),
  `_assign_dim_selector` (85837, `_update_bound_array_cell` (1079),
  `assign_implicit_intersection_range` (60597, implicit-intersection rewrite
  (`_rewrite_implicit_intersection` 597).
- Spilling helpers: `_resolve_spill_unset` (1484) replaces `None` sentinels
  in flat arrays before writing to grid (uses variable default or `#N/A`);
  `flatten_array` (1840) column-major flattens any array to 1D;
  `flatten_object_fields` (1798) flattens object fields for grid spills.
- Array construction/shape: `create_array` (1773, accepts `template=True` to
  fill with `None` sentinels), `create_object_array` (2110),
  `get_array_shape` (2008), `reshape_array` (2502), `infer_type` (1966, now handles `UnitValue("beef")→text`), 
  `fill_array` (2905), `flatten_object_fields` (1798), `flatten_array`
  (1443), `_nested_from_flat` (1928), `to_display_value` (1885).
- Constraints/dims: `set_labels` (2125), `check_dimension_constraints` (2216),
  `validate_array_element_types` (161694 — element-level base-type checking
  (`as number`/`as text` arrays reject mismatched scalars), `_dim_size` (2152).
- Grid-as-array: `get_grid_row` (2558), `get_grid_column` (2587), plus
  `GridLiveView` branches in `get_array_element`/`set_array_element`.

### `control_flow.py` (2108 lines) — blocks: For / If / Let / When
`class GridLangControlFlow` executes block constructs. Module-level regexes
(9–16) define `if...then`, `elseif...then`, `else`, `for...do`, `while...do`,
`when...do`, `end`.
- `process_for_statement` (11118: For-loop handling (ranges, init, arrays).
- Block engine: `_process_block` (874), `_extract_block_body` (308),
  `pre_scan_blocks` (181833, `_prepare_block_line` (338).
- If: `_process_if_statement` (932) and the "new"/"rich" variants (2011,
  2113), `_parse_if_header` (965), `_collect_if_blocks` (1003),
  `_execute_if_block_choice` (111186, `_process_if_elseif_else_block` (1874);
  condition evaluation helpers `_evaluate_if_*` (1337–1692).
- Let: `_process_let_statement_inline` (1178), field/index assignment helpers
  (1339, 1390).
- `_handle_block_*` methods (343–960): per-statement handling inside blocks.

### `scope.py` (1090 lines) — scope + variable semantics
- `class Scope` (11119: variable storage with constraints.
  - `define` (392), `update` (433), `get` (526), `is_uninitialized` (547),
    `get_defining_scope` (562).
  - Inputs/outputs: `define_input` (575), `define_output` (590, now preserves `unit` from `Input` when `Output` shares name), `is_input`/`is_output` (518/527), `connect_pipe` (622), `push_value`
    (555), `_propagate_wave` (664) — the publish/listen ripple.
  - Unit handling: `_unit_convert` (104, now tries `apply_conversion` before `#UNIT`), `_has_pending_assignment` (142).
  - Constraints: `_re_evaluate_constraints` (717), `_check_constraints` (830, now unit-aware for `Let y of m = 5 of in`), `_validate_base_type` (788) — validates scalars AND, since the pyarrow
    removal, element-by-element base types of `dim` arrays (via
    `array_handler.validate_array_element_types`), `_expression_depends_on`
    (651). `_array_unset_value` (array_handler.py:1446) resolves `None`
    sentinels for unset template array cells: checks the variable's
    `constraints['default']` (from `or = <expr>`) and evaluates it; falls back
    to `error_value(NA_ERROR)` (`#N/A`).
  - Scoping: `is_shadowed` (678), `get_evaluation_scope` (686),
    `get_full_scope` (1080), `_coerce_custom_type_value` (226).
- `class _ListenerGrid` (2121: dict backing `compiler.grid`; every cell write
  calls `compiler._notify_cell_changed`.
- `class GridLiveView` (4040: `(row, col)`-keyed live view of a grid
  (1-based tuples). Used for the predefined `grid` variable, and for per-type
  instance grids. `read_only` views exist for read-only function scopes.
- `_ACTIVE_RUNNERS` (1618: stack of executing compilers; used to reject writes
  from read-only function sub-compilers to outer scopes.

### `type_processor.py` (1251 lines) — `Define X as Type` handling
`class GridLangTypeProcessor`:
- Type-def parsing: `_parse_type_def` (81), `_parse_type_def_line` (95),
  `_extract_type_field_line` (9696, `_parse_type_field_constraints` (199),
  `_record_type_field_definition` (15157, `_collect_type_computed_fields`
  (178), `_finalize_type_def_state` (274).
- Executing type body code against an instance: `_execute_type_code` (295),
  `_execute_type_block` (25252, `_process_grid_assignment` (501),
  `_process_type_for_loop` (38389, `_process_type_let_statement` (756),
  `_process_type_assignment` (58581.
- `_build_type_eval_scope` (70708, `_execute_builder` (1063).

### `parser.py` (638 lines) — variable-definition parsing
`class GridLangParser`:
- `_parse_variable_def` (1616: the central parser for `: name [as type] [of
  unit] [dim ...] [constraints] = expr` / `Input`/`Output` lines. Returns
  (parsed_var, parsed_type, constraints, expression).
- Constraint handling: `_check_comparison_series` (278),
  `_match_direct_assignment_patterns` (22225, `_apply_with_clause` (315),
  `_apply_dimension_constraints` (33334, `_merge_custom_type_constraints` (474),
  `_split_on_keywords` (39390, now `seen_equals`/`seen_init` keep `of`/`as` in RHS `5 of in`/`Init 5 of in`), `_parse_dim_size` (604). The `or` keyword in
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

### `test_runner.py` (1004 lines) — inline test suite
`class GridLangTestRunner` with `run_tests_independent(tests)` — now 313 tests (was 263) including 11 new unit tests (`Test 282`–`Test 292` for `UnitSource` constant/numeric, `Output` addition, `Let`/`For`/`Push`/`Init`/`:` and `1`). At the bottom of the file (~830) it runs itself when executed directly:
`python test_runner.py [names...]`. Failing names are printed.

## Language conventions to remember when editing

- **Case-insensitive**: keywords, variable/field/type names, cell refs.
  Lookups go through `get_case_insensitive_key`.
- **Grid storage**: `compiler.grid` is a dict keyed by cell ref strings
  (`'A1'`), wrapped in `_ListenerGrid`. Ranges use `:`; `^` marks a range's
  top-left corner (e.g. `[^A3]`); `@` is implicit intersection (current row).
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
  `_assign_horizontal_array` (`array_handler.py:1157`);
  `_assign_extended_address` (`array_handler.py:1254`) handles dotted targets
  like `A1.B1.3`.
- **Variables**: `: x = expr` (client binding — deferred until deps resolve),
  `Let x init val`/`= val`, `For x init val`. `Push x = expr` updates x and
  propagates to dependents (publish/listen). `Input`/`Output` declare I/O.
  `Init` is `Let`/`For`/`:` with `Init <expr>` (lazy, evaluated on first read).
- **Units**: `5 of m`, `"ox" of animal`, `2 of 1` → `UnitValue(value, unit)` via `gridlang_of_unit`; `of 1` is dimensionless. `Define B as UnitSource(Meat)` / `Convert "ox" of animal to "beef"` (constant) or `Convert x as number of cm to x/100` (formula, LHS var stripped). `Convert` target for top-level is inferred by evaluating RHS with stripped var (`compiler.py:273`). `:` fields (`: f of m`) convert on `with` (`compiler.py:1228`). `Push`/`Input` preserve `of` unit. `1` handling: `m*1→m`, `1*1→1`, `m/1→m`, `m/m→1` (`units.py`).
- **Types**: `Define T as Type ... End T`, `new T with (field = v, ...)`.
  Types carry computed fields and constraints.
- **Dependency extraction**: `DEPENDENCY_IGNORED_TOKENS` in both compiler.py
  and executor.py (must stay in sync); `_strip_constraint_operands` strips
  `of/as/dim/not null`/`of 1`/`"ox" of animal` clauses so they aren't mistaken for variable refs.
- **Functions/Subprocesses**: extracted from code, run in a fresh
  sub-`GridLangCompiler`; functions can read but not write the parent grid
  (`read_only` GridLiveView + `_ACTIVE_RUNNERS` guard).

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
- `QWEN.md` — empty.
- `.opencode/summaries/previous-summary.md` — notes from an earlier working
  session (predefined `grid` variable work, Tests 191–200). Read it when
  resuming that thread; later sessions removed pyarrow (see commit
  `036c261` "Replace pyarrow with Python list and dict").
- `build/`, `venv/`, `.venv/`, `gridlang.egg-info/`, `gridlang/` (empty) —
  generated/env dirs, gitignored except `gridlang/`. The `grid` CLI is
  installed in `venv/` and `.venv/`.

PYEOF
