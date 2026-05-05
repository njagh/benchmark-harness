# CONVENTIONS.md

# CONVENTIONS.md

## Purpose

This document defines the engineering conventions for this repository, with special emphasis on **agentic coding**.

The goal is not just to produce working code. The goal is to produce code that is:

- correct
- easy to review
- easy to test
- easy to modify safely
- resilient to partial understanding
- robust when written iteratively by humans and coding agents

Agentic coding works best when the codebase is simple, explicit, modular, and heavily validated. These conventions are meant to reduce ambiguity, prevent fragile changes, and make it easy to verify that each change did what we intended.

---

## Core principles

### 1. Readability over cleverness
Prefer straightforward code over compact, clever, or highly abstract code.

- Use clear names.
- Use explicit control flow.
- Avoid surprising behavior.
- Avoid dense one-liners unless they are unquestionably clearer.

Code should be understandable by a new engineer or coding agent with minimal context.

---

### 2. Small, focused units
Keep files, functions, and classes small.

This is especially important for agentic coding:
- Smaller files are easier to inspect and reason about.
- Smaller functions are easier to test.
- Smaller changes reduce unintended side effects.
- Clear boundaries make it easier for an agent to modify one thing without breaking another.

Guidelines:
- A file should usually have a single clear responsibility.
- A function should usually do one thing.
- A class should represent one coherent concept.
- Split large files before they become hard to navigate.
- Prefer composition over giant utility files or god objects.

---

### 3. Explicit structure beats implicit coupling
Organize code so responsibilities are obvious.

Prefer:
- dedicated modules by concern
- clear interfaces between layers
- predictable naming
- explicit inputs and outputs

Avoid:
- hidden shared state
- cross-module side effects
- circular dependencies
- “misc”, “helpers”, or “utils” dumping grounds with no clear boundaries

---

### 4. Testability is a first-class requirement
Code is not complete until it is testable and tested.

Agentic workflows produce the best outcomes when every meaningful change can be validated quickly and locally.

Every non-trivial change should come with:
- tests for expected behavior
- checks for edge cases
- validation that existing behavior still works

Testing is not optional polish. It is the main defense against confident but incorrect edits.

---

### 5. Make the correct path the easy path
Code should guide future changes toward safety.

Prefer:
- typed interfaces
- assertions for invariants
- narrow public APIs
- deterministic behavior
- isolated side effects

The easier it is to do the right thing, the more reliable future agent and human changes will be.

---

## Repository organization

Structure the repository so the layout communicates intent.

A good structure typically separates:
- application logic
- domain logic
- infrastructure / adapters
- tests
- scripts / tooling
- docs

General expectations:
- Keep root directories clean.
- Put files where engineers would expect to find them.
- Co-locate related code.
- Avoid scattering one feature across too many unrelated locations.
- Avoid putting unrelated concepts in the same file.

### Naming
Use names that describe purpose, not history.

Prefer names like:
- `parser.py`
- `validation.py`
- `auth_service.py`
- `test_user_repository.py`

Avoid vague names like:
- `stuff.py`
- `misc.py`
- `temp.py`
- `new_new_final.py`

---

## File size and module boundaries

### Keep files small
Smaller files are easier for both humans and agents to load into context and modify safely.

Guidelines:
- Prefer adding a new focused module over continuously expanding a large one.
- Refactor large files before making them larger.
- Avoid files that contain multiple unrelated responsibilities.
- Avoid long files where important logic is buried among unrelated details.

### Keep modules cohesive
A module should have a clear reason to exist.

A good module:
- exposes a small, understandable surface area
- hides internal details
- contains closely related logic
- is easy to test independently

---

## Function and class design

### Functions
Functions should:
- have a clear purpose
- accept explicit inputs
- return explicit outputs
- avoid hidden mutations where possible
- stay reasonably short

Prefer:
- pure functions for core logic
- small helper functions with descriptive names
- early returns when they improve clarity

Avoid:
- long functions with mixed concerns
- functions that both compute and perform unrelated side effects
- boolean flags that drastically change behavior
- overly generic helpers used everywhere but understood nowhere

### Classes
Use classes when they clarify state or behavior, not by default.

A class should:
- model a real concept
- have a small public API
- maintain clear invariants
- avoid excessive internal branching

Avoid:
- large stateful classes that do many unrelated things
- classes used only as namespaces
- inheritance hierarchies where composition would be simpler

---

## State management

Be deliberate about state.

Prefer:
- local state over global state
- explicit dependency passing over implicit singletons
- immutable or minimally mutable data where practical
- clear ownership of state

Avoid:
- hidden state changes
- implicit initialization order
- mutation across distant parts of the codebase
- stateful behavior that makes tests order-dependent

State is one of the biggest sources of bugs in agentic coding because it is easy to overlook and hard to infer from partial context.

---

## Error handling

Handle errors explicitly and helpfully.

Prefer:
- failing loudly on impossible states
- raising specific errors
- adding context to failures
- validating assumptions at boundaries

Avoid:
- swallowing exceptions
- broad `except` without strong justification
- returning ambiguous sentinel values when errors should be explicit
- silent fallback behavior that hides real problems

When something goes wrong, the code should make diagnosis easier, not harder.

---

## Types, schemas, and contracts

Use types and schemas wherever available.

Prefer:
- type hints
- typed data structures
- validated inputs at boundaries
- explicit contracts between components

These are especially valuable in agentic coding because they reduce ambiguity and make incorrect edits easier to catch earlier.

Where applicable:
- validate external inputs
- encode invariants in types or constructors
- prefer structured data over loose dictionaries when the shape matters

---

## Comments and documentation

Write comments that explain **why**, not what the code obviously does.

Use comments for:
- non-obvious decisions
- constraints
- invariants
- performance tradeoffs
- domain-specific reasoning

Avoid comments that merely restate the code.

Documentation should help a future engineer or agent answer:
- What is this module responsible for?
- What assumptions does it make?
- What should not be changed casually?
- How is correctness validated?

For non-trivial modules, include a short top-level description.

---

## Logging and observability

Logs should support debugging without creating noise.

Prefer:
- structured, meaningful log messages
- logs at important boundaries
- logs for failures and critical state transitions
- stable messages that aid troubleshooting

Avoid:
- excessive debug spam in normal execution
- logs that duplicate obvious information
- logs that hide the actual failure cause

For important workflows, make it easy to inspect:
- inputs
- decisions
- outputs
- failures

---

## Dependency management

Prefer fewer dependencies.

Before adding a dependency, ask:
- Does the standard library already solve this?
- Is the dependency mature and well-maintained?
- Is the abstraction worth the cost?
- Will it simplify the codebase over time?

Avoid dependencies that:
- add heavy complexity for small convenience
- obscure core logic
- create lock-in without strong benefit

For agentic coding, fewer dependencies usually means less ambiguity and easier debugging.

---

## Configuration

Keep configuration explicit and separate from core logic.

Prefer:
- config files or clear config objects
- environment-driven settings for deployment concerns
- defaults that are safe and sensible

Avoid:
- hardcoded magic values spread through the codebase
- configuration hidden deep in implementation files
- environment access everywhere instead of at boundaries

---

## Testing standards

## Testing philosophy
Tests are required because they are the main mechanism for verifying agent-written changes.

Every meaningful code change should answer:
1. Does the code do what we think it does?
2. Did we preserve existing behavior?
3. Can we detect breakage quickly next time?

### Expectations
Write tests for:
- core behavior
- edge cases
- regression-prone bugs
- error paths
- invariants

Prefer:
- unit tests for focused logic
- integration tests for component interaction
- end-to-end tests for critical user flows where appropriate

### Test quality
Good tests are:
- deterministic
- isolated
- readable
- fast
- specific about failure

Avoid tests that are:
- brittle
- overly coupled to implementation details
- dependent on execution order
- dependent on network or external state unless explicitly intended
- so broad that failures are hard to diagnose

### Regression testing
When fixing a bug:
- write a test that reproduces the bug
- verify it fails before the fix when practical
- verify it passes after the fix
- keep the regression test unless there is a strong reason not to

### Validation for risky changes
For refactors, optimizations, or architectural changes:
- preserve behavior unless the change explicitly intends to alter it
- run existing tests
- add targeted tests around the changed area
- measure claims about performance rather than assuming them

Do not merge “cleanup” or “optimization” changes without validation.

---

## Agentic coding workflow

These rules are especially important when coding with LLMs or other agents.

### 1. Make small, reviewable changes
Prefer small diffs over sweeping rewrites.

A good agentic change:
- has a narrow scope
- is easy to explain
- is easy to test
- is easy to roll back

Avoid large changes that mix:
- refactors
- feature additions
- renames
- formatting-only edits
- dependency changes

Do one thing at a time.

---

### 2. Preserve structure unless there is a clear reason to change it
Agents should not casually reorganize the codebase.

Only restructure when it clearly improves:
- clarity
- maintainability
- testability
- separation of concerns

When restructuring:
- keep behavior stable
- move in small steps
- validate after each step

---

### 3. Do not invent requirements
Implement what is needed, not what might someday be useful.

Avoid:
- speculative abstractions
- premature extensibility
- generic frameworks for one concrete use case
- building a platform when a function would do

Agentic coding tends to overgeneralize. Resist that tendency.

---

### 4. Prefer explicit reasoning checkpoints
For non-trivial tasks, work in increments:
- understand the current behavior
- make one focused change
- test it
- interpret the result
- continue only when validated

This reduces compounding mistakes and makes failures easier to localize.

---

### 5. Respect existing conventions
Before introducing a new pattern, check whether the repository already has a good existing one.

Prefer consistency over novelty.

Match:
- naming
- file placement
- test style
- typing style
- error handling patterns
- logging patterns

A codebase is easier to maintain when it feels internally consistent.

---

### 6. Avoid silent scope creep
If a task is “fix X”, do not also:
- refactor unrelated code
- rename broadly
- update formatting everywhere
- change APIs without need
- add extra features unless required

Keep the diff aligned with the request.

---

### 7. Leave useful breadcrumbs
Because future work may be done by a different human or agent, leave behind clarity.

Useful breadcrumbs include:
- concise module docs
- tests that demonstrate intended behavior
- comments on invariants or non-obvious tradeoffs
- commit messages that explain why a change was made

---

## Refactoring rules

Refactoring is encouraged when it improves clarity and safety, but it must be disciplined.

A good refactor:
- reduces complexity
- improves naming
- isolates responsibilities
- preserves behavior
- is covered by tests

Before refactoring:
- understand current behavior
- identify the exact pain point
- add protection with tests when needed

After refactoring:
- verify behavior remained correct
- ensure the resulting structure is actually simpler

Do not refactor “because it feels cleaner” unless the improvement is real and demonstrable.

---

## Review checklist

Before considering a change complete, check:

### Correctness
- Does the code do the intended thing?
- Are edge cases handled?
- Are assumptions validated?

### Simplicity
- Is this the simplest reasonable design?
- Is any abstraction unnecessary?
- Can a new engineer understand it quickly?

### Structure
- Is the code in the right place?
- Are responsibilities clearly separated?
- Are file and function sizes still reasonable?

### Safety
- Are tests present and meaningful?
- Did we avoid hidden side effects?
- Is failure behavior clear?

### Maintainability
- Are names descriptive?
- Are comments useful?
- Will future edits be easier, not harder?

---

## Anti-patterns

Avoid the following unless there is a very strong and explicit reason:

- giant files with mixed responsibilities
- giant functions with long procedural flows
- hidden shared mutable state
- vague utility modules
- speculative abstractions
- excessive indirection
- dependency-heavy solutions to simple problems
- broad exception swallowing
- untested refactors
- implementation without validation
- comments that restate the code
- “temporary” code with no cleanup plan
- changing many unrelated things in one diff

---

## Definition of done

A task is done when:

- the implementation is correct
- the code follows repository conventions
- the structure remains clear
- tests cover the important behavior
- validation has been run
- the diff is reviewable
- the result is easier, or at least not harder, to maintain

Working code is necessary, but not sufficient.  
The final standard is **working, understandable, testable, maintainable code**.

---

## Default bias

When in doubt, choose the option that is:

1. simpler
2. clearer
3. smaller in scope
4. easier to test
5. easier to review
6. less likely to surprise the next human or agent

## Virtual Environment Requirement

All Python commands (including `python`, `pytest`, `mypy`, `ruff`) must be run from within the project's virtual environment.

**Activation Command:**
```bash
source ./.venv/bin/activate
```

**Usage Pattern:**
```bash
source ./.venv/bin/activate && <command>
```

**Examples:**
```bash
source ./.venv/bin/activate && pytest tests/
source ./.venv/bin/activate && mypy src/
source ./.venv/bin/activate && ruff check src/
source ./.venv/bin/activate && python scripts/smoke_test.py
```

**Important:** Never run Python commands directly without activating the virtual environment first.

