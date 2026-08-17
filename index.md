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

Language reference: `Documentation.md` (tutorial style). Install/usage docs:
`README.md`.

## Running things

```bash
# Run a program (args after filename are program inputs)
python main.py example.grid 42
python main.py example.grid --debug   # also exports <file>.csv

# Run the inline test suite (263 tests)
python test_runner.py                 # all tests
python test_runner.py 1 2 4           # subset by number

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
2. `compiler.run(code, args)` (`compiler.py:327`) creates a throwaway
   `GridLangExecutor`, **copies the compiler's state attributes AND every
   public method onto it**, and calls `executor.run()`. During a run the
   executor and compiler are effectively the same object; helper engines
   (`expr_evaluator`, `array_handler`, `control_flow`, `type_processor`,
   `parser`) were already constructed on the compiler and are shared.
3. `executor.run()` (`executor.py:2044`) is the interpreter entry point:
   `_run_setup` → `_run_prepare_execution` → `_run_main_loop` →
   `_resolve_pending_assignments` → `_process_deferred_assignments` →
   `_print_outputs`.
4. `_run_main_loop_impl_body` dispatches each line by statement kind: `For`,
   `Let`, `Push`, `When`, `Return`, grid assignment (`[A1] := ...`),
   declaration, or "misc". Big dispatch chains in executor.py and
   control_flow.py.
5. Results: `Return x` appends to `output_values` (printed by
   `_print_outputs`); grid writes land in `compiler.grid` (a dict keyed by
   cell refs like `'A1'`). `--debug` → `compiler.export_to_csv` (`compiler.py:2958`).

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

### `compiler.py` (3142 lines) — state + orchestration
`class GridLangCompiler` is the **persistent brain** and holds nearly all state
created in `__init__`:
- Grid & scoping: `grid` (`_ListenerGrid`), `scopes` (stack of `Scope`),
  `variables`, `types`, `dimensions`, `dim_names`, `dim_labels`.
- Publish/listen: `_listeners`, `_set_by`, `_propagating` (see "Conventions").
- Program constructs: `types_defined`, `functions`, `subprocesses`,
  `input_variables`, `output_variables`, `output_values`.
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
- `_extract_functions` (38389: pulls `Function`/`Subprocess` defs out of the
  main code and registers them.
- `_instantiate_type` (70702, `_evaluate_with_value` (852), `_apply_with_clause`
  parsing (958+): type/`with` object construction.
- `call_subprocess` (111128: runs a sub-`GridLangCompiler` in isolation.
- `_process_grid_assignment` (202080, `_process_declarations_and_labels` (2213),
  `_collect_global_declarations` (212168: top-level statement handling.
- `export_to_csv` (282817: `--debug` CSV export (grid as matrix, or outputs as
  one column when the grid is empty).
- `set_input_values` (282843: binds CLI/keyboard args to `Input`s with
  type/constraint coercion.
- `_seed_globals` (~1670): for sub-compilers; **skips redefining `grid`**.

Also defines `SubprocessResult` (46): result container exposing `grid`,
`variables`, `outputs`.

### `executor.py` (5134 lines) — the interpreter
`class GridLangExecutor` contains the main dispatch loop. This is where most
runtime behavior lives. Key methods:
- `run` (191918: top-level sequence (see Architecture).
- `_run_setup` (424220, `_run_prepare_execution` (4422), `_print_outputs`
  (4680), `_materialize_inits` (4879), `_process_deferred_assignments` (5014).
- Main loop: `_run_main_loop` (2074) → `_run_main_loop_impl` (2600) →
  `_run_main_loop_impl_body` (242455. `_handle_main_loop_*` methods dispatch
  statement kinds: quick statements (1113), `Let` (1147/1494/1515), `For`
  (many: 1728 fallback, 1952 array/dim, 2060 simple, 2099 single-line, 2349
  consecutive shortcuts, 2544 declaration, 2980 range, 3315 nested, 3524
  prechecks, 3589 post-branches), grid assignment (3822), `When` blocks
  (3879), `Push` (3994–4176), `Return` (4021), misc (3649).
- Dependency/guard machinery: `_build_dependency_network` (646),
  `_determine_needed_lines` (91917, `_evaluate_guard_conditions` (1011),
  `_evaluate_global_guards_pre_execution` (71717, `_execute_global_for_loops`
  (827), `_attempt_resolve_pending_var` (1031), `_resolve_ready_pending_vars`
  (1038).
- `Let` semantics: first pass `_process_let_first_pass` (1216), binding
  `_bind_declared_var` (131301, standard assignment (1453), second pass
  (1483), generator values (1607), `_apply_init_values` (1780).
- For-dim declarations: the regex at `_handle_for_array_and_dim_declarations`
  accepts optional `not null` before `as` and `or = <default>` after the
  dimension spec. `or = <expr>` is stored as `constraints['default']` so
  `_array_unset_value` can find it. The bounded-dim handler creates template
  arrays (`template=True`) when no `init`/standalone `=` is present.
- `Push` semantics: `_handle_push_assignment` (4631), `_evaluate_push_expression`
  (4286), `_process_push_call` (4697), `_assign_indexed_target` (4792),
  `_update_member_path_target` (444475.
- `When` blocks: `_register_when_block` (308), `_process_when_triggers` (340),
  `_run_when_block` (31317.
- Shared with compiler.py: `_strip_constraint_operands` (module-level, 26) and
  `DEPENDENCY_IGNORED_TOKENS` (1717 — duplicate of compiler's. Keep in sync.

### `expression.py` (3407 lines) — expression evaluation
`class ExpressionEvaluator` evaluates RHS expressions, arrays, ranges, sums,
dimension selectors, interpolations, member/field access, and Python-fallback
evaluation.
- Entry points: `eval_or_eval_array` (73), `eval_expr` (1993), and for
  assignments `_evaluate_array` (444).
- `eval_expr` is the big recursive dispatcher: array literals `{}`, pipes `|`,
  interpolated cell refs, paren/curly indexing, member calls, user function
  calls, object creation, field access, address-indexed access, then scalar
  constructs, then simple variables.
- Python fallback: `_evaluate_with_python_fallback` (2370) builds a scope and
  `eval()`s complex arithmetic (`_build_fallback_cell_scope` 2254,
  `_eval_python_fallback_result` 2501, `_get_eval_globals` 2937).
- Interpolation: `_process_interpolation` (3096). Operators:
  `_replace_operators` (292923.
- Grid indexing: `_replace_grid_indexing` (761) — only still needed for legacy
  dict-based object grids; it early-returns for `GridLiveView` (which flows
  through the generic array path). `_eval_array_element` uses `base=1` for
  `GridLiveView`.
- Also `CaseInsensitiveDict` (24): case-insensitive dict used for
  eval scopes.

### `array_handler.py` (2639 lines) — grid/array/tensor operations
`class ArrayHandler` centralizes all array knowledge:
- Cell addressing & lookup: `resolve_cell_index` (30), `cell_ref_to_indices`
  (129), `lookup_cell` (1412), `get_range_values` (1253/1286),
  `_lookup_extended_address` (131367, `_write_extended_tensor` (1488).
- Assignment: `evaluate_line_with_assignment` (205),
  `_parse_assignment_target_details` (26257, `_perform_assignment_write` (482),
  `_assign_horizontal_array` (98984, `assign_range` (1170),
  `_assign_extended_range` (121221, `_assign_index_selector` (777),
  `_assign_dim_selector` (85837, `_update_bound_array_cell` (972),
  `assign_implicit_intersection_range` (60597, implicit-intersection rewrite
  (`_rewrite_implicit_intersection` 597).
- Spilling helpers: `_resolve_spill_unset` (1394) replaces `None` sentinels
  in flat arrays before writing to grid (uses variable default or `#N/A`);
  `flatten_array` (1547) column-major flattens any array to 1D;
  `flatten_object_fields` (1505) flattens object fields for grid spills.
- Array construction/shape: `create_array` (1773, accepts `template=True` to
  fill with `None` sentinels), `create_object_array` (1804),
  `get_array_shape` (1702), `reshape_array` (2169), `infer_type` (1673),
  `fill_array` (2568), `flatten_object_fields` (1505), `flatten_array`
  (1443), `_nested_from_flat` (1635), `to_display_value` (1592).
- Constraints/dims: `set_labels` (1819), `check_dimension_constraints` (1883),
  `validate_array_element_types` (161694 — element-level base-type checking
  (`as number`/`as text` arrays reject mismatched scalars), `_dim_size` (1846).
- Grid-as-array: `get_grid_row` (2225), `get_grid_column` (2254), plus
  `GridLiveView` branches in `get_array_element`/`set_array_element`.

### `control_flow.py` (2201 lines) — blocks: For / If / Let / When
`class GridLangControlFlow` executes block constructs. Module-level regexes
(9–16) define `if...then`, `elseif...then`, `else`, `for...do`, `while...do`,
`when...do`, `end`.
- `process_for_statement` (11118: For-loop handling (ranges, init, arrays).
- Block engine: `_process_block` (960), `_extract_block_body` (308),
  `pre_scan_blocks` (181833, `_prepare_block_line` (338).
- If: `_process_if_statement` (1015) and the "new"/"rich" variants (2011,
  2113), `_parse_if_header` (1048), `_collect_if_blocks` (1086),
  `_execute_if_block_choice` (111186, `_process_if_elseif_else_block` (1967);
  condition evaluation helpers `_evaluate_if_*` (1430–1785).
- Let: `_process_let_statement_inline` (1271), field/index assignment helpers
  (1339, 1390).
- `_handle_block_*` methods (343–960): per-statement handling inside blocks.

### `scope.py` (959 lines) — scope + variable semantics
- `class Scope` (11119: variable storage with constraints.
  - `define` (341), `update` (382), `get` (475), `is_uninitialized` (496),
    `get_defining_scope` (511).
  - Inputs/outputs: `define_input` (524), `define_output` (539),
    `is_input`/`is_output` (518/527), `connect_pipe` (564), `push_value`
    (555), `_propagate_wave` (606) — the publish/listen ripple.
  - Constraints: `_re_evaluate_constraints` (659), `_check_constraints` (742),
    `_validate_base_type` (700) — validates scalars AND, since the pyarrow
    removal, element-by-element base types of `dim` arrays (via
    `array_handler.validate_array_element_types`), `_expression_depends_on`
    (651). `_array_unset_value` (array_handler.py:1356) resolves `None`
    sentinels for unset template array cells: checks the variable's
    `constraints['default']` (from `or = <expr>`) and evaluates it; falls back
    to `error_value(NA_ERROR)` (`#N/A`).
  - Scoping: `is_shadowed` (620), `get_evaluation_scope` (628),
    `get_full_scope` (952), `_coerce_custom_type_value` (176).
- `class _ListenerGrid` (2121: dict backing `compiler.grid`; every cell write
  calls `compiler._notify_cell_changed`.
- `class GridLiveView` (4040: `(row, col)`-keyed live view of a grid
  (1-based tuples). Used for the predefined `grid` variable, and for per-type
  instance grids. `read_only` views exist for read-only function scopes.
- `_ACTIVE_RUNNERS` (1618: stack of executing compilers; used to reject writes
  from read-only function sub-compilers to outer scopes.

### `type_processor.py` (790 lines) — `Define X as Type` handling
`class GridLangTypeProcessor`:
- Type-def parsing: `_parse_type_def` (30), `_parse_type_def_line` (38),
  `_extract_type_field_line` (9696, `_parse_type_field_constraints` (121),
  `_record_type_field_definition` (15157, `_collect_type_computed_fields`
  (178), `_finalize_type_def_state` (192).
- Executing type body code against an instance: `_execute_type_code` (209),
  `_execute_type_block` (25252, `_process_grid_assignment` (366),
  `_process_type_for_loop` (38389, `_process_type_let_statement` (520),
  `_process_type_assignment` (58581.
- `_build_type_eval_scope` (70708, `_execute_private_helper` (725).

### `parser.py` (531 lines) — variable-definition parsing
`class GridLangParser`:
- `_parse_variable_def` (1616: the central parser for `: name [as type] [of
  unit] [dim ...] [constraints] = expr` / `Input`/`Output` lines. Returns
  (parsed_var, parsed_type, constraints, expression).
- Constraint handling: `_check_comparison_series` (226),
  `_match_direct_assignment_patterns` (22225, `_apply_with_clause` (263),
  `_apply_dimension_constraints` (33334, `_merge_custom_type_constraints` (396),
  `_split_on_keywords` (39390, `_parse_dim_size` (497). The `or` keyword in
  `_split_on_keywords` extracts `or = <expr>` as `constraints['default']`;
  `not null` sets `constraints['nullable'] = True`. The default value is used
  by `_array_unset_value` when reading unset template array cells.

### `utils.py` (372 lines) — shared pure helpers
- Address math: `split_cell` (113), `col_to_num` (122), `num_to_col` (129),
  `offset_cell` (12125, `parse_address` (151, N-D dotted addresses like
  `A3.B4.8` → `[3,1,4,2,8]`), `indices_to_address` (186), `is_address` (156),
  `validate_cell_ref` (8585, `prod` (204).
- Case-insensitive dict access: `get_case_insensitive_key` (242),
  `get_case_insensitive_value` (24241.
- Object/type metadata filtering: `public_type_fields` (211),
  `object_public_keys` (21213, `public_object_view` (261) — strip `_hidden`
  fields and keys starting with `_`/`$`/`grid`.
- Text parsing: `iter_interpolation_placeholders` (18), `split_var_defs` (48).
- `format_display_value` (26265: display formatting with float-trimming and
  list/dict-form array support.

### `test_runner.py` (865 lines) — inline test suite
`class GridLangTestRunner` with `run_tests_independent(tests)` — a huge method
containing 263 hardcoded test cases (name, code, expected grid dict). At the
bottom of the file (~830) it runs itself when executed directly:
`python test_runner.py [names...]`. Failing names are printed. `test_runner.py.bak`
is a stale backup — ignore.

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
  is NOT required for arrays (only for objects). Spilling fills from fastest
  dimension (first declared dim = grid rows) to slowest (last dim = grid
  columns) in column-major order. An array of objects spills one object per row,
  fields across columns. The `^` notation on objects makes them spill along the
  fastest dimension (1D object array → along second-fastest dim). The core
  spilling function is `_assign_horizontal_array` (`array_handler.py:1050`);
  `_assign_extended_address` (`array_handler.py:1147`) handles dotted targets
  like `A1.B1.3`.
- **Variables**: `: x = expr` (client binding — deferred until deps resolve),
  `Let x init val`/`= val`, `For x init val`. `Push x = expr` updates x and
  propagates to dependents (publish/listen). `Input`/`Output` declare I/O.
- **Types**: `Define T as Type ... End T`, `new T with (field = v, ...)`.
  Types carry computed fields and constraints.
- **Dependency extraction**: `DEPENDENCY_IGNORED_TOKENS` in both compiler.py
  and executor.py (must stay in sync); `_strip_constraint_operands` strips
  `of/as/dim/not null` clauses so they aren't mistaken for variable refs.
- **Functions/Subprocesses**: extracted from code, run in a fresh
  sub-`GridLangCompiler`; functions can read but not write the parent grid
  (`read_only` GridLiveView + `_ACTIVE_RUNNERS` guard).

## Scratch / auxiliary files

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
