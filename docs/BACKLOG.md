# Refactor backlog

Everything outstanding from the repo-wide review, in the form it would take as
GitHub issues — one section per issue, so it can be split up when Issues is
enabled on the repo (Settings → General → Features → Issues).

Phases 1 and 2 are merged or in review. This file covers Phases 3–6 plus the
findings that turned up during them and do not belong to a phase.

**Full review with evidence:** https://claude.ai/code/artifact/2ba66572-add3-4642-8b92-2d5cb6c4057e

| | | |
|---|---|---|
| [Phase 3](#phase-3-close-the-silent-degradation-gaps) | silent degradation | results-affecting |
| [Phase 4](#phase-4-collapse-the-duplication) | ~700 lines of copy-paste | behaviour-preserving |
| [Phase 5](#phase-5-split-the-oversized-modules) | module splits, CSV schema | mostly moves |
| [Phase 6](#phase-6-make-the-operational-surface-reproducible) | container, deploy, logging | low risk |
| [`--task alfworld` cannot run](#--task-alfworld-cannot-run-the-environment-is-commented-out) | bug | **needs a decision** |
| [DyLAN/MacNet retry loops](#dylan-and-macnet-retry-loops-are-still-unbounded) | bug | deferred |
| [Intrinsic token accounting](#intrinsic-token-accounting-is-always-zero) | bug | silent |
| [`g-memory` offline](#g-memory-cannot-be-exercised-by-the-offline-test-suite) | test coverage | |
| [`max_trials` naming](#max_trials-names-two-different-budgets-in-dylan) | tech debt | |
| [Dependency pins](#single-source-the-dependency-pins) | tech debt | **needs a decision** |
| [Per-episode tokens](#per-episode-token-attribution) | tech debt | |
| [Test backlog](#test-backlog-groups-a-b-c-and-e) | tests | |

---

## Phase 3: close the silent-degradation gaps

`silent-failure` · **results-affecting by design**

The most expensive class of defect in a research codebase: the sweep completes,
the CSV fills in, and the number is not measuring what the flag says it measures.
Every published figure touched by one of these was produced under it.

**Flag these to whoever owns the current numbers before merging anything here.**

- [ ] **Restore FINCH clustering.** `cluster_tasks` wants the FINCH algorithm, but
  the pinned dependency was `finch-api` — an unrelated HR/payroll SDK that also
  exports a `Finch` symbol. So `Finch(X, distance='cosine')` raised, the
  surrounding `except Exception` printed to stdout, and `labels = np.zeros(...)`
  put every task in cluster 0. **Every `--mas_memory g-memory` result to date was
  produced with the task layer collapsed into a single cluster.** Phase 1
  corrected the pin to `finch-clust==0.2.3` (`baabfa0`); the call site is still
  wrong and is the work here — the real symbol is `FINCH` not `Finch`; it returns
  `(c, num_clust, requested_c)` so `_, _, labels` binds `labels` to
  `requested_c`, which is `None` unless `req_clust` is passed; and `c` is an
  `(n_samples × n_partitions)` matrix, so a partition level has to be chosen
  before it is a label vector.
- [ ] **Narrow the `except` at `GMemory.py:452`** so a clustering failure is
  logged as a warning through the recorder, never swallowed into a single-cluster
  fallback. The bare except is why the above went unnoticed for the life of the
  fork.
- [ ] **Fix the `autogen_mas` projector.** `_project_insights` was copied verbatim
  from `autogen.py` and still guards on
  `isinstance(self.meta_memory, GMemory)`, but `build_system` in that class
  assigns `meta_memory_solver` and `meta_memory_validator` and never
  `meta_memory`, which stays at the `MetaMAS` default of `None`.
  **`--use_projector` is accepted and silently ignored.** Best done as part of the
  merge in Phase 4.
- [ ] **Reconcile config with code.** `tasks/configs.yaml` names the MacNet block
  `graph:` while `build_task` looks up `CONFIG.get('macnet', {})`, so MacNet
  always falls back to in-code defaults — and they disagree with the file:
  `use_critic` defaults to `True` at `graph_mas.py:34` where the YAML says
  `False`. **MacNet runs with critics enabled despite the config saying
  otherwise.** Also delete the dead `memory_folder` and `max_steps` keys.
- [ ] **Decide on `temperature`.** `#temperature=temperature` is commented out in
  the `chat.completions.create` call, so `llm_config.temperature: 0.1` and every
  `ReasoningConfig(temperature=0)` are inert — runs execute at the backend
  default, and determinism claims tied to `--seed` do not hold for the sampler.
  Either pass it and accept the behaviour change, or delete it everywhere. Phase 2
  made this sharper: the parameter now resolves from `LLMSettings` and is threaded
  all the way to the call, where that one commented line still drops it. **One
  line from working, or one line from honest.** `num_comps`, the other sampling
  parameter, was removed in `737a8e5` on the same reasoning — it was sent to the
  API and could only cost money, since nothing reads past the first choice.
- [ ] **Repair or retire `intrinsicmemory-llm-structured-template`.** `summarize`
  generates a template and stores it in `agent_intrinsic_memory`, then passes
  `template_instructions=self.memory_template` — a field initialised to `""` and
  never assigned. The module's own debug line prints that it is empty. So that arm
  measures an empty template *and* clobbers the accumulated memory on first call.
  Phase 2 made the missing parameter visible by giving `summarize` a real
  signature (`2cc0d3c`) but deliberately did not change what it measures.
- [ ] **Split `predicate_map` per PDDL domain.**
  `tasks/envs/pddl_env/pddl_env.py` has 80 keys and 75 unique ones: `on`,
  `clear`, `holding`, `free` and `move` are each defined more than once, so
  domains overwrite each other's wording. Live effect on the active `TASK_NAMES`:
  tyreworld's `(free ?x - hub)` renders with gripper's `"{} is free. "` instead of
  `"Hub {} is free."`, and blockworld's `clear` gets hanoi's
  `"The {} is clear."`. Key it on `game_name`, then remove the `F601` per-file
  ignore from `pyproject.toml`.
- [ ] **Deduplicate `g-memory` in the Slurm sweeps.** It appears twice in the
  `--mas_memory` list in all four scripts. At ten seeds that is twenty redundant
  experiments per submission and two rows per configuration in
  `overall_results.csv`, which a naive group-by averages together. Check whether
  any published aggregate double-counted it.

### Done ahead of this phase

- ~~Decide the `trials` convention~~ — `2ead4a3`. `trials = i` was a zero-based
  loop index, so every mean-trials figure was one low.
- ~~Move the NLTK downloads out of import~~ — `ebf7597`.

---

## Phase 4: collapse the duplication

`tech-debt` · behaviour-preserving, wide diff

Roughly 700 lines of copy-paste, and the mechanism by which a fix in one copy left
the other broken — the `autogen_mas` projector bug (Phase 3) survived exactly
because `_project_insights` exists in five places and the fix landed in one copy's
context. Bodies confirmed byte-identical by hashing whitespace-stripped source.

- [ ] **Lift `_project_insights` and `_solver_stuck`** to `MetaMAS` or a
  `workflow/common.py`. `_project_insights` is identical across `autogen`,
  `autogen_mas` and `autogen_hotpot`, with a second identical variant in `dylan`
  and `graph_mas` — reconcile the two variants first. `_solver_stuck` is identical
  in `autogen` and `autogen_mas` (2 × 28 lines).
- [ ] **Replace the `GMemory` isinstance checks with a `SupportsProjection`
  protocol,** so the projector works for any memory implementing it rather than
  one concrete class. This is what makes the Phase 3 projector fix stay fixed.
- [ ] **Merge `autogen_mas` into `autogen`** behind a `use_validator` config flag.
  They are a near-verbatim fork, 336 vs 270 lines, and *both define a class called
  `AutoGen`*, aliased at import in the registry. Keep the existing tests green as
  the acceptance criterion — there are now 320 of them.
- [ ] **Turn the intrinsic-memory subclasses into data:** one
  `IntrinsicMASMemory` taking a prompt bundle, plus a `{name: bundle}` registry.
  Four files that differ by exactly one prompt constant become one, and a new task
  variant stops needing a new class. **Write test A1 first** — it is the
  regression net for this change, and right now nothing detects a copy-paste slip
  that runs PDDL's prompt on FEVER.
- [ ] **Delete `autogen_hotpot.py`** (232 lines). Not merely unused —
  non-functional: undefined name `solver` at `:176-177`, and three agents built
  and never used at `:116-118`. HotpotQA is not a supported task.
- [ ] **Delete `intrinsicmemory_deprecated.py`** (67 lines). Unreferenced, and
  imports `INTRINSICMEMORY`, which no longer exists in `prompt.py`. Also drop
  `IntrinsicMASMemory` from `__all__` if it is not meant to be selectable.
- [ ] **Collapse the Slurm scripts** to one parameterised script taking task and
  model as arguments. `fever`/`pddl`/`sciworld` differ by 3 lines out of 74.

### Done ahead of this phase

- ~~Lift the observer pattern to `MetaMAS`~~ — `afbf5f3`, because the shared retry
  helper reports through it. Four copies deleted; the fifth is in
  `autogen_hotpot.py`, which this phase deletes.
- ~~Remove the two `time.sleep(5)` calls~~ — `8c9d764`.

---

## Phase 5: split the oversized modules

`tech-debt` · low risk per step

Only worth doing once Phase 4 has removed the duplicate call sites these modules
serve. Mostly moves — do each as its own commit.

- [ ] **Split `GMemory.py`** into `gmemory/memory.py`, `task_layer.py` and
  `insights.py`. Lift the twice-nested `parse_numbered_list` (defined at `:316`
  and again at `:554`) to a shared parsing helper.
- [ ] **Split `prompt.py`** into one module per memory module, re-exported from a
  package `__init__` so imports stay stable.
- [ ] **Make dataset loading lazy** in `tasks/envs/__init__.py`. It currently
  reads all four task manifests at import — so `--task fever` still parses the
  ALFWorld, PDDL and SciWorld files, and every spawned worker repeats the work. A
  `{task: loader}` registry loads only what was asked for, with the handle closed.
- [ ] **Extract `results.py` from `run.py`,** owning one documented CSV schema
  written with `csv.DictWriter` and real headers. Retires the
  `result_fields[3:10]` positional slicing, which recovers columns by re-parsing a
  string that was just formatted.
  - [ ] **Add the task-count column** while doing it. `AggregateResults` now
    carries `episode_count`, so the denominator exists — it just is not written to
    the CSV yet, because that needs the schema work.
- [ ] **Extract `sweep.py` from `run.py`,** owning config expansion — including
  the dead `keys` computation at `:172`, whose result is never used.
- [ ] **Decide what the per-task CSV means.** `run_task` appends
  `recorder.average_results()` after every task, so
  `<task>-<memory>-results.csv` holds a *cumulative mean per row*, not a per-task
  result. Its column order also differs from the one `run_experiment` writes —
  `seed` is field 4 in one and field 11 in the other. Pick per-task rows or a
  running mean, and name the file for what it holds.
- [ ] **Unify the env hierarchy:** fold `mas.agents.Env` into `BaseEnv`, declare
  `max_trials` on the base (all four subclasses set it independently), and give
  `BaseEnv.__init__` a real body instead of `pass`.

~~`BaseRecorder` assigning `field(default_factory=dict)` outside a dataclass body~~
— removed in `78c914b` as a side effect of the recorder refactor.

Test **C5** golden-files the CSV schema and should land with the `results.py`
extraction.

---

## Phase 6: make the operational surface reproducible

`tech-debt` · low risk

Last, because it is the least entangled with the code — but it is what makes a
result someone else can reproduce.

- [ ] **Make the Dockerfile actually install.** It copies `requirements.txt` and
  never runs `pip install` — the install lines are commented out.
  `entrypoint.sh` then runs `source .venv/bin/activate`, so the image only works
  if a host-built `.venv` is picked up by `COPY . .`, from a macOS/arm64 host into
  an `nvcr.io/nvidia/pytorch` base. Use `uv sync --frozen` from the lockfile, on a
  base matching the target architecture, and drop the host-`.venv` assumption.
- [ ] **Extend `.dockerignore`** to `.venv`, `data/`, `logs/` and `.db*`. It
  currently excludes none of them.
- [ ] **Fix the deploy path mismatch.** `template/generate_templates.py:44` writes
  to a relative `deploy-templates/` with no `mkdir`, while `deploy.sh` reads
  `template/deploy-templates/*` — so the generator only works when run from inside
  `template/`, and the directory is absent from the repo.
- [ ] **Take the Azure identifiers out of the source.**
  `generate_templates.py` hardcodes a workspace subscription id and maps
  parameters to opaque `environmentVariable0…7` slots — give them names. There is
  also a tab mixed into the indentation at `:35`.
- [ ] **Route diagnostics through the recorder.** 41 `print()` calls in non-test
  first-party code bypass the logger, so a sweep's stderr interleaves prompts,
  results and errors from every worker. Phase 2 moved the LLM layer's error output
  to stderr and routed retry diagnostics through `notify_observers`; the rest is
  this.
- [ ] **Default `--num_workers` sensibly.** `tasks/run.py` defaults to
  `max(1, os.cpu_count() - 32)`, which is one specific machine — on any laptop
  that is `1`. Use a fraction of `cpu_count()`.
- [ ] **Fix the project metadata.** `pyproject.toml` still says
  `"Add your description here"` and names the project `gmemory`.
- [ ] **Update the README** once Phases 3–5 land: the flag table, the
  memory-module list and the `--mas_type` matrix all describe behaviour that will
  have changed.

---

## `--task alfworld` cannot run: the environment is commented out

`bug` `needs-decision` · found during Phase 2

`--task alfworld` is the argparse default, and it cannot start.

`tasks/envs/alfworld_env.py` has both the `alfworld` import and the
`self.main_env = get_environment(...)` assignment commented out, disabled in
`4f54cdd` — *"remove alfworld for now"*. But `AlfworldEnv.__init__` still calls
`self.reset()`, and `reset()` reads `self.main_env`:

```python
def reset(self):
    self.done = False
    self.env = self.main_env.init_env(batch_size=1)   # AttributeError
```

So constructing the environment raises `AttributeError` before any of the recorder
logic is reached.

**Why this matters beyond the error.** The review recorded *"the default task
cannot record a result"*; in fact the default task cannot get as far as recording
anything. The recorder repair in `d8e552d` is still correct and still needed — it
fixed a real arity violation and surfaced the same bug in `PDDLRecorder` — but it
does not make `--task alfworld` work.

**The decision.** Either:

1. **Restore the environment** — add `alfworld` back to the dependencies and
   uncomment, if ALFWorld results are still wanted; or
2. **Change the argparse default** to a task that runs, and say in the README that
   ALFWorld is unavailable.

Either is fine. What is not fine is the current state, where a reader following
the README gets an `AttributeError` with no explanation and the commit that caused
it says *"for now"*.

**Reproduce:** `uv run python tasks/run.py --mas_type autogen --mas_memory none`

---

## DyLAN and MacNet retry loops are still unbounded

`bug` `tech-debt` · found during Phase 2, deliberately deferred

Phase 2 replaced the hand-written agent retry loops with
`MetaMAS._call_agent_with_retries` (`afbf5f3`), but only in `autogen.py` and
`autogen_mas.py`. **DyLAN and MacNet still hold their own copies.** Deferred by
direction — only AutoGen is in use for the MAS.

`tasks/mas_workflow/dylan/dylan.py:219` and
`tasks/mas_workflow/macnet/graph_mas.py:~146`:

```python
tries = 0
while tries < 3:
    try:
        action = curr_neuron.execute(user_prompt, use_critic=self._use_critic)
        if action == '':
            continue          # <- jumps past the increment below
        action = env.process_action(action)
        break
    except Exception as e:
        print(f'Error during execution of node {curr_neuron.id}: {e}')
    tries += 1
```

The counter sits at the bottom of the body and is reached only on the exception
path. An empty response hits the `continue` and skips it, so the loop **never
terminates** for an agent that keeps returning `""`.

`GPTChat` now raises instead of returning `""` (`aaa8ebe`), so the most likely
trigger is gone — an API error is now an exception, which *does* reach the
increment. The loop is still unbounded for a model that genuinely answers with an
empty string, and `action` can still reach `AgentMessage(message=action)` unbound
if every attempt raises before it is assigned.

**The fix, when these come back into use.** Replace both loops with
`self._call_agent_with_retries(...)`. The helper takes the attempt as a callable,
so `curr_neuron.execute` and `curr_node.execute` slot straight in. Wrap it in the
same `except AgentCallFailed: break` the AutoGen workflows use (`8bcb42c`), and
charge the full budget as they do (`91e3a99`). Fold in the `max_trials` rename
below at the same time.

**What Phase 2 did give them**, so they are not stranded: both return
`EpisodeResult` (`7e8a06a` — before this, `--mas_type dylan` and `--mas_type
macnet` raised `ValueError: not enough values to unpack` on the first completed
task); inherit `add_observer`/`notify_observers` from `MetaMAS`; share the
`summarize` keyword contract (`2cc0d3c` — MacNet's `upstream_agent_ids=None` broke
all six intrinsic memory modules); share the `trials` convention (`2ead4a3`); and
are covered by the contract and smoke matrices. DyLAN also stopped ignoring the
configured embedding model (`7fcbf1d`).

---

## Intrinsic token accounting is always zero

`bug` `silent-failure` · found during Phase 1

`intrinsic_prompt_tokens` and `intrinsic_completion_tokens` are **always zero**,
while being written to every log line and to columns 10-11 of every result CSV.

`TokenTracker.record` only adds to them when called with `intrinsic=True`:

```python
def record(self, prompt_tokens, completion_tokens, intrinsic=False):
    self.prompt_tokens += prompt_tokens
    self.completion_tokens += completion_tokens
    if intrinsic:
        self.intrinsic_prompt_tokens += prompt_tokens
        self.intrinsic_completion_tokens += completion_tokens
```

`intrinsic=` appears exactly once across `mas/` and `tasks/` — the internal
forward inside `GPTChat.__call__`. **No memory module ever passes it.** So the
columns are populated, plausible, and measuring nothing.

**Why fix rather than delete.** The point of this fork is the intrinsic memory
family, and the natural question about it is what the memory *costs* in tokens
against the baseline. These columns are exactly where that answer should be, which
makes a silent zero worse than an absent column — a reader has no signal that the
number is not real.

**The fix.** `IntrinsicMASMemory.summarize` calls `self.llm_model(messages)` to
run the memory update. That call, and the template-generation call in
`intrinsicmemory_llm_structured_template.py`, should pass `intrinsic=True`. Audit
the other memory modules for LLM calls that are memory work rather than task work
— `ChatDevMASMemory.summarize`, `MemoryBankMASMemory.add_memory` and `GMemory`'s
insight merging are all candidates, and whether they count as "intrinsic" is a
definition worth writing down in the same commit.

**Test A2**: drive a memory update through a stubbed `GPTChat` with a real
`TokenTracker`, then assert `intrinsic_prompt_tokens > 0` and that it is less than
the run total. The plumbing is already tested from the other end —
`test_llm_layer.py::test_an_intrinsic_call_is_billed_to_the_intrinsic_columns_too`
confirms `intrinsic=True` works when someone passes it.

---

## `g-memory` cannot be exercised by the offline test suite

`tests` · found during Phase 2

`g-memory` is the one registered memory module the offline suite cannot drive, so
it is excluded from the 11 × 4 memory-by-workflow matrix (`2cc0d3c`). It is also
the module the upstream paper is named for.

It persists through `langchain_chroma`, which `tasks/tests/conftest.py` stubs with
a `MagicMock` so the package can be imported without the heavy dependency. A
`MagicMock` gets far enough to import and not far enough to run.

The exclusion is named rather than silent — `test_contracts.py` declares
`UNTESTABLE_OFFLINE = {"g-memory"}` and
`test_the_memory_matrix_covers_every_registered_module` fails if a module is added
to `module_map` without either being covered or listed there. So this cannot
quietly get worse.

**Options.**

1. **Add `chromadb` to the dev group.** It does not pull torch or the CUDA stack,
   so it is affordable — the CUDA-free dev environment is currently 26 packages
   and 89MB, and the constraint that matters is not pulling nvidia wheels.
   Cheapest, and gives real coverage.
2. **Put an interface in front of the vector store** and use an in-memory
   implementation in tests. More work, but it is the DIP fix `GMemory` wants
   anyway, and Phase 5 splits that file regardless.

**Do this before Phase 3/5 touch `GMemory.py`.** Phase 3 rewrites its FINCH
clustering call and narrows its bare `except`; Phase 5 splits the file into three.
Both are much safer with the module under test, and the FINCH work in particular
is a correctness change to code that has never once executed successfully.

---

## `max_trials` names two different budgets in DyLAN

`tech-debt` · found during Phase 2

Within twenty lines of `tasks/mas_workflow/dylan/dylan.py`:

```python
max_trials: int = task_config.get('max_trials', 3)   # :170  per-agent-call retries
...
for i in range(env.max_trials):                      # :204  episode budget
    ...
        while tries < max_trials:                    # :219  the retry one again
```

`env.max_trials` is how many trials the episode gets. The local `max_trials` is
how many times a single agent call may be retried.

The local one is also read from `task_config`, where `run.py` never puts it —
`max_trials` is an *experiment* config key, consumed by `build_task` as
`max_steps` and handed to the environment. So the local lookup always falls
through to the literal `3`.

**Severity:** confusion, not a live defect. `--max_trials` does reach the episode
budget by the intended route, and the retry budget genuinely is meant to be 3.
Worth fixing because the next person to read this cannot tell that from the code.

**Fix:** rename the retry budget — `max_retries`, or drop the local entirely —
when the DyLAN loop is folded into `MetaMAS._call_agent_with_retries`, which
already takes the budget as a named `max_tries` parameter. Land with the DyLAN
retry loop issue.

---

## Single-source the dependency pins

`needs-decision` `tech-debt` · carried forward from Phase 1

`requirements.txt` and `pyproject.toml` carry the same ~200 fully-pinned packages
— transitive dependencies and the entire CUDA/torch stack included — alongside a
`uv.lock`. Two hand-maintained copies of a `pip freeze`, which will drift.

They already did. Phase 1 had to make the same two corrections in both files in
lockstep (`baabfa0`):

- `attr==0.3.2` was pinned where `reasoning_modules.py:1` wanted `attrs` — and
  that import should have been `dataclasses.dataclass` regardless, which is what
  it is now;
- `finch-api==1.44.1` was an unrelated HR/payroll SDK, not the FINCH clustering
  algorithm — which is why `g-memory` clustering has never run.

Editing two files in lockstep to fix one wrong pin is precisely the failure mode.

**Why this is a decision, not a cleanup.** Which file is canonical depends on
whether the conda path in the README is still supported. If it is,
`requirements.txt` has to keep working for people not using `uv`. If it is not,
`pyproject.toml` plus `uv.lock` is the whole story and `requirements.txt` should
be generated or deleted.

**Options.**

1. **`pyproject.toml` canonical**, `requirements.txt` generated by
   `uv export --no-hashes` in CI and committed, or deleted outright. Cleanest if
   conda is gone.
2. **Keep both by hand**, and add a CI check that they agree. Cheap insurance if
   the conda path must stay.

Either way: the pins should name direct dependencies, not a frozen transitive
closure. Phase 1's `[dependency-groups] dev` group names 8 packages and installs
26, which is the shape to aim for.

Phase 6 needs this resolved before the Dockerfile can `uv sync --frozen`.

---

## Per-episode token attribution

`tech-debt` · raised in review of PR #5

There is no per-episode token accounting. One `TokenTracker` is shared by every
call in an experiment, `run.py` writes its cumulative totals after each task, and
nothing attributes tokens to the episode that spent them.

Two things want it:

- **Excluding a failed episode's tokens.** Raised in review. Not done, and I would
  argue against it on its own: tokens are a cost that really was incurred, not a
  measure of the system, so hiding real spend understates what a run cost. Trials
  and reward are measures and should not be flattered by a fault; tokens are a
  bill. But the option does not currently exist either way.
- **The always-zero intrinsic token columns** (see above). Reporting what the
  memory costs against the baseline needs per-call attribution to land somewhere
  per-episode, not just in a running total.

**Sketch:** snapshot the tracker at `task_begin` and diff at `task_end`, so
`EpisodeResult` or the recorder can carry a per-episode figure. Cheap, and it
makes both of the above possible. Land it with the intrinsic-token work, and with
Phase 5's CSV schema so the columns have somewhere honest to go.

---

## Test backlog: groups A, B, C and E

`tests`

Phases 1–2 took the suite from **115 tests** — reaching one workflow file and one
memory module — to **320**, reaching every workflow, recorder and environment and
eleven of twelve memory modules. Group D, the cross-registry contract tests, was
Phase 2's acceptance criterion and is done.

Each contract test in Phase 2 was verified by running it against the pre-fix code,
not only by watching it pass afterwards. **A test that has never been red is a test
of nothing** — keep that habit for these.

### A · The intrinsic memory family — 1 of 6 modules directly tested

This fork's core contribution (`a5a3643`). Only `intrinsicmemory-notemplate` has
direct tests; the other five appear only in the compatibility matrix, which
asserts they do not raise. Since the five subclasses differ by exactly one string,
**nothing currently detects a wrong one.**

- [ ] **A1 — each module gets its own prompt.** Parametrise over the registry: for
  every `intrinsicmemory-*` key, assert `memory_system_prompt` is the specific
  constant that module should carry, and that it is non-empty (the base uses
  `""`). Cheap, and the only thing standing between a copy-paste slip and a whole
  experiment arm silently running PDDL's prompt on FEVER. **Write this before
  Phase 4 collapses these classes into a registry.**
- [ ] **A2 — intrinsic LLM calls are billed as intrinsic.** See the
  intrinsic-token-accounting issue.
- [ ] **A3 — the `len(task_trajectory) > 5` gate.** Boundary test at 3 (the
  initial `'\n\n>'`), 5 and 6 characters. A bare magic number that decides whether
  the first memory update happens at all.
- [ ] **A4 — `save_task_context` honours its signature.** It is annotated
  `-> MASMessage`, returns `None`, and never calls `super()` — so the context is
  never persisted. **Write this as a failing test first**: it is a defect
  specification.
- [ ] **A5 — the LLM-structured template reaches the prompt.** See Phase 3; that
  arm currently measures an empty template.
- [ ] **A6 — memory does not leak between tasks.** Run two tasks against one
  memory instance and assert `agent_intrinsic_memory` is empty at the second
  task's start. The current reset works by rebuilding `GPTChat`, which is a side
  effect, not a contract.

### B · The validator workflow

`autogen_mas.py` is fork-owned (`3b5d156`) and the best-tested file here — but the
tests assert shape, not behaviour, which is how the projector bug lived directly
underneath them.

- [ ] **B1 — the projector actually projects.** With `use_projector=True` and a
  memory implementing projection, assert `project_insights` is called and per-role
  insight lists differ. Catches the dead branch; **write it failing** and fix it in
  Phase 3.
- [ ] **B4 — validator memory is written and read.** It is summarised and saved,
  but never retrieved into any prompt. Either assert it feeds something, or record
  that it is write-only. Right now the code does not say which was intended.
- [x] ~~B2 — an always-empty LLM response terminates~~ — `test_agent_retries.py`,
  verified to fail in 1.3s against the pre-fix loop.
- [x] ~~B3 — a persistently INVALID verdict is bounded~~ —
  `test_agent_retries.py`.

### C · The sweep runner

Multi-seed sweeps, `--num_workers`, `--db_dir`, per-seed memory isolation and
failure recording are all fork-owned and concurrency-sensitive. They *had* tests
until `38b21be` — titled "Add parallel code" — deleted them, five days after
`3159a59` added them.

- [ ] **C1 — restore `test_run_parallelism.py`.**
  `git show 3159a59:tasks/tests/test_run_parallelism.py` — both tests still apply:
  `test_append_local_result_is_serialized_across_threads` and
  `test_run_task_can_use_multiple_workers`.
- [ ] **C2 — per-seed memory directories are isolated.** Two seeds of one config
  must not share a persist dir. The comment at `run.py:214` claims this; nothing
  enforces it, and a regression would cross-contaminate memory between seeds —
  invisible in the results.
- [ ] **C3 — the Cartesian product is right.** `build_experiment_configs` with
  2 tasks × 3 memories × 3 seeds should yield 18 distinct configs. Also pins the
  dead `keys` computation at `:172`.
- [ ] **C5 — the CSV schema is stable.** Header present, column order fixed, one
  row per experiment. Golden-file it. Would have caught the two incompatible
  schemas and the positional `result_fields[3:10]` slicing between them. Land with
  Phase 5's `results.py`.
- [x] ~~C4 — one failing experiment does not kill the sweep~~ —
  `test_run_task.py` covers the task-level equivalent (`5e04c0a`), parametrised
  over all four tasks; the experiment level still deserves C1's restored coverage.

### D · Contract tests across the registries — done

All four written in Phase 2 and each verified against the pre-fix code: D1
recorders, D2 workflows, D3 environments, D4 memory × workflow.
`test_contracts.py` and `test_task_smoke.py`.

### E · Data rendering and the LLM layer

- [ ] **E1 — `predicate_map` has no colliding keys.**
  `assert len(keys) == len(set(keys))`, then per active domain that its predicates
  render with its own template — tyreworld's `free` as `"Hub {} is free."`. One
  line for the first half, and the `F601` ruff ignore comes out with it. Land with
  Phase 3.
- [ ] **E2 — sampling parameters reach the API.** Assert the temperature a caller
  sets arrives in the `chat.completions.create` kwargs — or delete the parameter
  everywhere. Either resolution is fine; silently dropping it is not. Land with
  Phase 3.
- [ ] **E4 — per-experiment loggers do not cross-contaminate.**
  `mas/logging_utils.py` was added specifically to stop a handler leak between
  experiments on a reused worker. Assert two loggers built in one process write
  only to their own file — easy to reintroduce and invisible until you read the
  logs.
- [x] ~~E3 — exhausted retries are distinguishable from an empty answer~~ —
  `test_llm_layer.py`.

### Suggested order

**A1 and C1–C2 next**, before Phase 4 touches the intrinsic modules or the sweep.
A2 and E1–E2 alongside their Phase 3 fixes. A4 and B1 written red, deliberately.
