# GridLang Modules — Design Specification

Status: **design; prototype in progress**. This document captures the agreed
module semantics for GridLang. Pieces marked *(not yet implemented)* describe
behavior the engine does not support yet; the others describe the syntax as it
exists or as it is landing.

## 1. Concept

A module is **exactly one file**. There is no `End Module` marker — the file
*is* the module.

A module occupies a middle ground between a definition bundle and a
subprocess: it may publish **definitions** (types, functions, subprocesses,
unit categories, top-level variables) for reuse, and it may contain a **runnable
body** invoked in isolation like a subprocess. Whether a module is usable one
way, the other, or both is the module writer's decision, expressed in the
header and the export surface.

Two operations on a module are distinct:

- **Load** (`use`) — binds the module's definitions into the current program.
- **Run** (the module-run subprocess, `ModuleName(...)`) — executes the
  module's body in isolation when the module is `runnable`; importing a
  runnable module (with or without a version) exposes it as a subprocess
  named after the module (§7).

## 2. Module header

```grid
Module Mymodule [runnable] [shared]
```

- `runnable`: the module has a body that may be invoked by importing it and
  calling its module-run subprocess (`Mymodule(...)`, see §7). Absent, any body
  is inert and a versionless `use Mymodule` fails at the import line — a
  versioned `use Mymodule.v1` is required instead.
- `shared`: the module is instantiated **once per program** instead of once per
  importer. Importing the same `shared` module must always resolve to the same
  physical copy; a version pin selecting a different copy is a load error.
- A module with no `Version` blocks *(see §4)* and no `runnable` flag has
  neither surface nor body — it is definitionless.

## 3. Module version

```grid
: version = "rel12.3-5"     ' Text   → strict equality ( = ) only
: version = 12.3            ' Number → equals and ordered ( = , >= )
: version = <a date>        ' Date   → ordered ( = , >= )
```

The version is a **build pin**: it identifies a physical revision of the file.
Comparison power derives from the declared type:

- **Text**: only `=` is meaningful. Use it for label-like revisions.
- **Number / Date**: `=` and `>=`. An ordered bound is allowed because the
  value carries an ordering; `>=` picks the newest available copy satisfying
  the bound (see §8).

## 4. API versions and exports

API (export) versions are the **only importable surface** of a module. They
are strict, named views over the module's definitions, and they follow the
convention of versioned online services (an old API version keeps working
because its definitions keep existing).

```grid
Version v1 exports Foo, bar_v1, bam
Version v2 exports bar, bam, woosh
```

- A definition whose name ends with `<version-tag>` is exported **stripped**:
  `bar_v1` exports as `bar` under `Version v1`. Other names export as-is.
- Many `Version` blocks may coexist in one file — one implementation, many
  façades. The names that do not overlap are independent definitions.
- Exports may be **definitions** (function, subprocess, type, unit category)
  **or top-level variables** (see §6 for what binds on load).
- `Input` / `Output` variables are **never** exportable — they belong to the
  module-run interface only (§7).
- A module with no `Version` blocks has no importable API unless it is
  `runnable` (a versionless `use` then runs its body without binding exports).

*Implemented for definitions: exporting a **function, subprocess, or type**
binds its definition (see the Load-semantics and binding notes in §5).
**Top-level variables** export flat as read-only views of the module instance
(see §6 and §11). Unit categories export flat via the `of` form (see §13);
namespaced top-level-variable and namespaced unit-category exports are still
deferred.*

## 5. Import

API version is **not required for runnable modules**: a versionless
`use Mymodule` imports a runnable module as its module-run subprocess (§7) and
binds nothing else. Any module whose API is imported needs a `Version` tag;
there is no default/latest.

```grid
use Mymodule.v1                            ' flat: exports land in scope
use Mymodule.v1 with (version="rel12.3-5") ' module pin (module version)
use Mymodule.v1 with (version>=12.3)       ' ordered pin (Number/Date)
For B use Mymodule.v1                      ' namespaced: B.Foo, B.bar
For B use Mymodule.v1 with (version>=12)   ' namespaced + pin
use Mymodule                               ' runnable modules only (§7)
```

The instruction lead and namespace stay in sync across `For`, `Let` and `:`:
all three accept `B as ModuleVersion use <module>.<vN>`, and each may be
omitted — `B use <module>.<vN>` or a bare `use <module>.<vN>` (flat) work too.
`Let B as ModuleVersion Use Mymodule.v1` and `: B as ModuleVersion Use Mymodule.v1`
bind exactly like `For B use Mymodule.v1`. A versionless import is flat only:
a namespaced versionless `use` (e.g. `For B use Mymodule`) is a load error.

**`use` returns a `ModuleVersion`.** A namespaced import binds the namespace
name (e.g. `B`) to a `ModuleVersion` value at the top level. The value is
assignable (`Let alias as ModuleVersion = B`) and passable to functions whose
inputs are typed `ModuleVersion` (the function reads `m.module` for the bound
namespace name). Member/type access still goes through dotted names
(`B.Foo`, `alias.foo`), which resolve regardless of aliasing. Two caveats:
`Require` names the real namespaced type directly (`Require p as B.timer`) — an
alias is not resolved there — and namespaces whose name collides with the
`mod` modulo-operator keyword (e.g. `mod`) work in dotted form but not as a
bare operand.

- Flat bind injects the version's exported names into the current scope; a
  name collision is a load error.
- `For B use ...` binds the version as a namespace reachable by dotted paths
  (`B.Foo`), consistent with `For`'s client-binding semantics: order
  independent — you may reference `B.Foo` before the `For B use` line.
- Two versions of one module imported into the same program must use
  **namespaces**: flat imports overlap on every name both versions export, so
  a flat binding is a guaranteed collision — this is intentional and safe by
  construction.

*Implemented.* `use` is only legal at the top (global) level; a `use` inside a
block is a load error. Binding is order-independent for flat and namespaced
imports: you may reference an exported name (or `B.Name`) before its `use`
line. Load-time checks: unknown module/version-tag, an export not backed by a
definition, an unsatisfied pin, and a flat name collision are all `ModuleImportError`s.
Member functions of an exported type travel with the type — name only the type
in the `Version exports` list, not its members.

## 6. Load semantics

Loading a module **binds**; it never **runs**.

- The *equality family* — `: x = e`, `Let x = e`, `For ... = e` — binds and
  resolves through the dependency machinery as ordinary client bindings.
  These are self-contained values.
- The *Push family* — `Push`, `Init` (deferred push), `For ... init` — never
  fires on load. An exported variable that is only ever pushed reads `#N/A`
  unless it carries an `or = <default>`.
- Consistent contract: **exports intended to be consumed before the module is
  run must be equality-bound or carry a default.** Push-only exports are state
  slots — meaningful after a run has filled them.
- `Input` and `Output` declarations belong to the module-run interface and
  never bind on load.

*(Top-level variable exports, flat only, are implemented. Equality-bound
    (`: x = e`) exports instantiate on load as read-only views; `init`-seeded
    and push-only exports read `#N/A` until a run — or an exported subprocess
    push — fills them. Because `init` is a shortcut for `Push` (it never fires
    on load), a module that wants a stateful counter should export an
    initializer subprocess that pushes the seed value. Namespaced variable
    exports, and the run-result `SubprocessResult` value surface, are still
    deferred.)*

## 7. Run

A `runnable` module's body runs in isolation through the **module-run
subprocess**: *importing* the module — `use Mymodule` (no version) or
`use Mymodule.v1` — makes a subprocess named after the module available, and
calling it executes the module body exactly like a subprocess call:

```grid
use Mymodule                    ' runnable modules only; binds no exports
Mymodule(3, out)                ' runs the module body: Inputs are arguments,
                                ' Outputs flow back through the trailing bindings
```

- A versionless `use Mymodule` imports only the module-run subprocess; nothing
  is bound into scope. The module must declare `runnable` — otherwise the
  import fails at the `use` line (`use Mymodule.<tag>` is required instead).
- A versioned `use Mymodule.v1` of a `runnable` module binds the tag's exports
  *and* exposes the module-run subprocess, so the same program can use the
  module's API and run its body.
- The module body's `Input` declarations are the call parameters; its `Output`
  declarations are the returned interface, bound to the trailing call
  arguments. The run is **in isolation**: the body executes on a fresh
  compiler, like a subprocess call today, and re-derives the module's own
  definitions from its body text — so a versionless `use` can run a body whose
  `Input`/`Output` reference module-private types that were never bound into
  scope.
- A module body is **Print-only** section: `Return` is not allowed there. The
  body may use `Print` (console) and `Output` declarations (the run result);
  `Return` is reserved for functions and named subprocesses.

*Implemented. Importing a runnable module (with or without a version) registers
the module-run subprocess under the module's own name — the `Sub(module)`
primitive of the original §7 note remains deferred, and the module name itself
is the run handle today. A bare statement calling a name that is neither a
subprocess nor a function is now a `NameError` (no more silent no-ops).*

## 8. Channels: Print, Return, Output

There are three output channels.

| Channel      | Destination            | Legal in                                  |
|--------------|------------------------|-------------------------------------------|
| `Print x`    | ambient output (console)| any scope                               |
| `Return x`   | the caller (value)     | functions and operations (subprocesses)   |
| `Output` vars| the module-run interface | runnable modules' body / module head |

- `Print` pushes a value to the **ambient output**, which is the console
  today (conceptually redirectable later).
- `Return` is the **call-return channel** — the value passed back from a
  function or subprocess to its caller. It is legal only inside functions and
  operations.
- A top-level `Return` outside a function/operation is a **load error**
  pointing at `Print`. *Implemented: `_run_setup` rejects any `Return` left in
  the top-level program stream (`executor._validate_no_top_level_return`).
  Inline block statements carrying a single-instruction payload (`For … do
  return …`, `If … then return … [else …]`, `When … do return …`) are first
  rewritten into real block form by `_normalize_inline_blocks` (compiler.py),
  so their generated `Return` line is caught by the same whole-line check and
  the inline spelling behaves exactly like the multi-line one.*

## 9. Instances

- Each importer context instantiates each module once: **per (importer ×
  physical copy)**. Two `Use`s in one consumer resolve to the same instance;
  two different consumers get independent instances.
- A `shared` module is instantiated once per program; the first import's pin
  wins, and every later import is validated against that selected copy (a
  pin resolving to a different file is an error).
- **API versions are views over a copy**: importing both `M.v1` and `M.v2` of
  the same copy shares that copy's state — migration is state-continuous.
- Module instance state (variables and the module's private grid) lives per
  instance; `shared` is the explicit escape hatch for true program-wide
  services.

*Instance state is implemented per (importer × physical copy): the module's
top-level variables live in one instance scope shared by every `use` of the
module in the consuming program, and exported subprocesses mutate it. The
`shared` flag itself, and module-private grid state, are still deferred.*

## 10. Version resolution

- Text pin `version="x"`: the copy's version must equal `x`.
- Ordered pin `version>=n` (Number/Date): the newest available copy
  satisfying the bound is selected.
- When a module is **shared** and a copy is already selected, a later
  `version>=n` import is accepted **iff the already-selected copy satisfies
  the bound** — consistency over newest.

*(Not yet implemented.)*

## 11. Closure

Foreign modules are **closed**:

- Types defined in another module may not be extended: no member functions,
  no builders, no `new`-extension from outside the defining module.
- Foreign variables are read-only hands-on: a program cannot `Push`, assign,
  or write into a module's state directly.
- The **only mutation channel** is an exported *subprocess* whose body (authored
  by the module) performs the pushes. Exported subprocesses run against the
  imported module's live instance in the caller's program.
- No leak through conversions: standalone `Convert` rules are module-local
  and never cross the module boundary (see §13).

*(Type-extension and direct-write rules apply today's member rules. The
read-only view of exported variables and the exported-subprocess mutation
channel against the live instance are implemented; `shared` instances and the
module-private grid are not yet routed.)*

## 12. Requirements and capabilities

Modules have no power to acquire capabilities on their own: **module
requirements are never granted.** A `Require` inside a module does not reserve a
capability for the importing program and is never satisfied automatically.
Capabilities flow the other way — from the main program into the module:

- **Define and export a custom `Resource`.** The recommended pattern is for the
  module to declare a `Resource` type that carries the needed requirements and
  export it like any other definition. Consumption of that resource stays
  explicit and program-side.
- **Pass a handle.** Module-internal access to a resource arrives *only* through
  explicit parameters: an instance of a granted resource (or a bare handle)
  passed to the module's functions, subprocesses, type builders, etc.
- **Runnable modules:** a granted resource or a handle may be passed to the
  module when it runs (`Mymodule(...)`); the module never fetches one itself.
- **Full control stays with the main program.** Nothing in a module can reach a
  capability the main program has not explicitly handed over. If the module's
  requirements evolve, the main program must adapt (re-grant, pass a new handle)
  — the wiring is not automatic.

Because grants are always program-side, third-party code cannot bypass the main
program to obtain access.

*Inner `Require` statements of a user resource bind as namespaced, read-only
members of the granted instance (`p.ticks`), not as global names: the builtin
is parameterised per owning resource and nothing leaks into the importer's
global scope. A missing/denied grant still binds the member as the sticky
`#PERM` value, so the program continues and the taint propagates.*

*(Not yet implemented.)*

## 13. Units

```grid
Define X as UnitSource of K          ' reuses the "of" pattern (5 of m)
```

- `Convert` lines **inside** a `UnitSource` block are part of that named
  category and travel with it when the category is exported.
- A `Convert` **outside** a `UnitSource` is **module-local**: it names no
  symbol, cannot be exported, and affects only the module's own evaluation.
  Importers never see it.
- Multiple `UnitSource of K` for the same unit are allowed while the rules
  stay **unique**: an identical `(src,dst)` rule from another module is
  accepted idempotently; a conflicting rule for an occupied pair is a load
  error.

*Implemented for the `of` form:* `Define X as UnitSource of K` registers the
category during the loader preprocess, so it is harvestable and a flat export
binds (results are `UnitValue` objects). The parenthesized spelling
`Define X as UnitSource(K)` registers the category elsewhere (at run time) and
is not harvestable today. Namespaced unit-category exports are still deferred.

## 14. Delegation

Wrapping a foreign type through delegation is shorthand for a forwarding
member; it is the sanctioned way to reuse a foreign type's behavior without
extending it.

```grid
Define Circle as Type
  : center as Point
  : radius of m
End Circle
delegate Point.move, Point.area to center
```

- `delegate` is pure boilerplate removal for a hand-written member that
  forwards to the field's foreign member.
- Delegated members are explicit and per-name; a delegated name may not
  shadow a locally-defined member (load error).
- Forwarding resolves the **live** foreign member at call time; the API
  manifest records the delegated names plus the delegate module's version, so
  drift is caught by the version check.

*(Not yet implemented.)*

## 15. Versioning / manifest

- The **module version** is a build pin (§3). The **API version** is semantic
  and pinned at every `Use` (§5). There is no cross-version migration — vN
  keeps its definitions, consumers move when they choose.
- Versioning applies to importable (exported) modules; a runnable-only module
  is a process whose manifest is empty.
- Each `Version vN` block digests its own mapping: exported names → definitions
  (with signatures), committed top-level variable values, and the category
  rules it carries. The digest is checked at `Use`; a module-version mismatch
  refuses to load.

*(Not yet implemented.)*

## 16. Deferred / open

- `shared` modules combined with ordered (`>=`) pins: resolution is settled
  (§10), pending a real use case.
- Module resolution path: bare names resolve to `<name>.grid` across a search
  list (program directory, then `GRID_PATH`).
- Diagnostics and error text conventions for load-time failures.

## 17. Syntax glossary

```grid
Module <name> [runnable] [shared]
: version = <text | number | date>

Version <vN> exports <names...>

use <module>                                 ' runnable modules (§7); no exports bound
use <module>.<vN> [with (version=<pin>)]
[For | Let | :] <ns> as ModuleVersion use <module>.<vN> [with (version=<pin>)]

Print <expr>
Return <expr>            ' functions and operations only
<module>(...)            ' run the module body as the module-run subprocess (§7)

Define <X> as UnitSource of <K>
delegate <member...> to <field>
```