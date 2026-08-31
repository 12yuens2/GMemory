# Refactor backlog

Everything outstanding from the repo-wide review, in the form it would take as
GitHub issues — one section per issue, so it can be split up when Issues is
enabled on the repo (Settings → General → Features → Issues).

Phases 1–3 are merged (PR #4, PR #5, PR #6). Phase 4 is on `refactor-p4` with
four of its five stages landed and one dropped by direction. This file covers
Phases 3–6 plus the findings that turned up during them and do not belong to a
phase.

**Full review with evidence:** https://claude.ai/code/artifact/2ba66572-add3-4642-8b92-2d5cb6c4057e

| | | |
|---|---|---|
| [Phase 3](#phase-3-close-the-silent-degradation-gaps) | silent degradation | 6 of 10 landed |
| [Phase 4](#phase-4-collapse-the-duplication) | ~700 lines of copy-paste | 4 of 5 landed |
| [Phase 5](#phase-5-split-the-oversized-modules) | module splits, CSV schema | mostly moves |
| [Phase 6](#phase-6-make-the-operational-surface-reproducible) | container, deploy, logging | low risk |
| [`--task alfworld` cannot run](#--task-alfworld-cannot-run-the-environment-is-commented-out) | bug | **needs a decision** |
| [DyLAN/MacNet retry loops](#dylan-and-macnet-retry-loops-are-still-unbounded) | bug | deferred |
| [Structured output vs a validator agent](#guarantee-the-action-format-instead-of-checking-it-with-a-second-agent) | research | **needs a decision** |
| [Intrinsic update prompts](#which-memory-update-prompt-should-each-intrinsic-variant-carry) | research | **needs a decision** |
| [~~`--mas_memory` default~~](#--mas_memory-defaulted-to-a-value-the-registry-does-not-have--closed) | bug | closed |
| [Intrinsic token accounting](#intrinsic-token-accounting-is-always-zero) | bug | silent |
| [G-Memory clustering](#g-memorys-clustering-does-not-run) | bug | **deferred** |
| [PDDL `predicate_map`](#predicate_map-collisions-between-pddl-domains) | bug | **deferred** |
| [~~Per-task trial budget~~](#per-task-max_steps-never-reaches-the-environment--closed) | bug | closed |
| [`g-memory` offline](#g-memory-cannot-be-exercised-by-the-offline-test-suite) | test coverage | |
| [`max_trials` naming](#max_trials-names-two-different-budgets-in-dylan) | tech debt | |
| [Dependency pins](#single-source-the-dependency-pins) | tech debt | **needs a decision** |
| [Per-episode tokens](#per-episode-token-attribution) | tech debt | |
| [Test backlog](#test-backlog-groups-a-b-c-and-e) | tests | |

---

## Phase 3: close the silent-degradation gaps

`silent-failure` · **results-affecting by design** · branch `refactor-p3`

The most expensive class of defect in a research codebase: the sweep completes,
the CSV fills in, and the number is not measuring what the flag says it measures.

Planned as ten items in four stages. **Six landed; four moved**, three of them by
direction and one because it turned out to belong with the work it was a
prerequisite for. Full write-up with evidence per item:
https://claude.ai/code/artifact/2ba66572-add3-4642-8b92-2d5cb6c4057e#phase-3

### Landed

| | Item | Commit |
|---|---|---|
| 1a | `from finch import Finch` named something the package does not export | `08453f6` |
| 4 | `--use_projector` decided by capability, not by naming `GMemory` | `01ee10e` |
| 5 | `temperature` sent, and dropped only if the endpoint refuses it | `379ff59` |
| 6 | each task's trial budget read from its own config entry | `96de17b` |
| 7 | the LLM-structured-template arm given its template | `364ac8a` |
| 9 | `g-memory` no longer requested twice per sweep | `e93687a` |
| 10 | dead `memory_folder` key deleted; MAS-block lookup written on the file | `e93687a` |

Tests: 115 → **395**. Every fix was run against the pre-fix code first.

**What changed about existing results.** Only two of these move a number:
`temperature` now reaches the sampler, so every arm changes; and the
`intrinsicmemory-llm-structured-template` arm now measures what it claims to,
having previously measured the no-template arm plus one wasted call per task.
`--use_projector` runs need re-running only if any were done — the flag was inert
under both workflows in use. FEVER's budget was raised to the 30 already in
effect, so no FEVER number moves.

### What the finch fix did and did not do

Fixing the import unblocks the CLI, which could not start for *any* value of
`--mas_memory`. It does not make `g-memory` work, and the clustering fix is
deferred by direction. Worth knowing what the deferral leaves: `cluster_tasks`
now reaches its call, `FINCH` returns `(c, num_clust, req_c)`, and `req_c` is
`None`, so the `zip` on the next line raises `TypeError` outside the `try`. So
`--mas_memory g-memory` fails loudly, per task, recorded in `failed_tasks.csv` by
`5e04c0a`. It does not resume producing single-cluster results, and the other
eleven memory modules work again.

### Two things the plan had wrong

Both found by writing the test before the fix.

- The template fix was recorded as one line — assign the generated template to
  `memory_template`. It is two: the module's own `memory_update_prompt` was
  commented out, so it inherited the base prompt, which has no
  `{template_instructions}` slot. The template would have been formatted into
  nothing. Neither change works without the other.
- The projector was recorded as a renamed-field bug in `autogen_mas`. It is
  wider: `autogen`'s guard is reachable but admits only `GMemory`, whose
  clustering has never run, so the projector has effectively never projected
  under either workflow. Pointing `autogen_mas` at the right field would have
  fixed the symptom and left the mechanism.

### Moved out of this phase

- **G-Memory's clustering** — see the standalone entry below. Deferred by
  direction. Takes stage 0 (`chromadb` into the dev group) with it, since that
  existed to make the clustering work observable.
- **`predicate_map` per PDDL domain** — see the standalone entry below. Deferred:
  the collision behaviour needs to be understood before it is changed.
- ~~**The other three copies of `_project_insights`**~~ — done in Phase 4's
  stage 2 (`f4445ab`). Lifting the method to `MetaMAS` migrated DyLAN and MacNet
  as a side effect; `autogen_hotpot` was deleted (`9f8cd07`).

### Done ahead of this phase

- ~~Decide the `trials` convention~~ — `2ead4a3`. `trials = i` was a zero-based
  loop index, so every mean-trials figure was one low.
- ~~Move the NLTK downloads out of import~~ — `ebf7597`.

---

## Phase 4: collapse the duplication

`tech-debt` · behaviour-preserving · branch `refactor-p4`

Roughly 700 lines of copy-paste, and the mechanism by which a fix in one copy
left the other broken — the `autogen_mas` projector bug (Phase 3) survived
exactly because `_project_insights` existed in five places and the fix landed in
one copy's context.

**Four of the five stages landed. Stage 5 was dropped by direction.** Net effect
on first-party code: **227 lines added, 886 removed**, with 5 new comment lines.
Tests went 395 → **417**.

| Stage | What | Commit |
|---|---|---|
| 0 | test A1/A1b as the regression net | `0fef126` |
| 1 | the two dead files, and their ruff ignores | `9f8cd07` |
| 2 | `_solver_stuck` and `_project_insights` lifted to `MetaMAS` | `f4445ab` |
| 3 | `autogen_mas` merged into `autogen` behind `--use_validator` | `07d6bb2` |
| 4 | intrinsic variants turned into three lines each | `f574ee6` |
| 5 | ~~collapse the Slurm scripts~~ | dropped |

### What changed about existing results

**Nothing.** Every stage is behaviour-preserving, and stage 3 — the only one with
a wide diff — was verified by driving the merged class and the deleted one side by
side over the same fakes and comparing every solver and validator prompt byte for
byte, not just the outcome:

- validator arm vs `autogen_mas` at `f4445ab`: valid first try, persistent
  INVALID, empty forever, recovers mid-episode, multi-trial — result, actions and
  all prompts identical;
- plain arm vs `autogen` at `f4445ab`: solves first try, empty forever, recovers,
  multi-trial, stuck repeating — identical.

The two validator prompts moved to `autogen_prompt.py` as named constants,
extracted from the source rather than retyped, and checked by rendering both
forms with the same values. The stray 16- and 20-space indentation inside those
f-strings is part of what the model receives, so it is preserved verbatim.

### What the CLI looks like now

```
--mas_type {autogen,dylan,macnet}        # autogen_mas is gone
--use_validator                          # the validator arm, any mas_type
--mas_memory {…12 unchanged keys…}
```

`autogen_mas` is rejected at parse time, naming the three that remain. Every
`MAS_MEMORY_MODULES` key is unchanged, so the Slurm scripts needed no edit and
result CSV paths keep their names.

### Two claims this section used to make, both wrong

Found by re-measuring against `main` rather than trusting the first review.

- **`_project_insights` had three variants, not two.** Phase 3's `01ee10e` moved
  `autogen` and `autogen_mas` onto `SupportsProjection` and left
  `autogen_hotpot`, `dylan` and `graph_mas` naming `GMemory`. `dylan` and
  `graph_mas` hashed identically to each other; `autogen` and `autogen_mas`
  differed only in which attribute the memory came from.
- **"Drop `IntrinsicMASMemory` from `__all__`"** conflated two classes of the same
  name. The exported one is the live base, which the variants extend; the dead one
  was in `intrinsicmemory_deprecated.py` and is deleted. No `__all__` change was
  needed.

### Three things worth knowing that the plan did not anticipate

- **DyLAN and MacNet have no distinct agent roles.** DyLAN's nine agents and
  MacNet's three all carry the profile `solver`, so per-role projection is
  vacuous for them however the projector is wired. `test_each_role_gets_its_own_
  insights` guarded itself with `len(projected) > 1` and so could not be widened;
  it is now two tests, one asserting every role receives insights projected for
  it (all four workflows) and one asserting roles that differ receive insights
  that differ (only `autogen` exercises it).
- **A `functools.partial` registry for stage 4 would have been silently broken.**
  `build_system` builds the validator's memory as `memory.__class__(...)`, which
  drops any constructor argument a registry supplied — the validator would have
  run on the base's empty system prompt. The prompts are class attributes,
  deliberately unannotated, and `test_a_rebuilt_memory_keeps_its_prompts` pins
  it.
- **The validator mechanism is on `MetaMAS`, so any workflow can use it.**
  `_reviewed_attempt` wraps a proposing agent in a reviewing one and hands the
  result to `_call_agent_with_retries`. Only AutoGen uses it: DyLAN and MacNet
  still hold their own hand-written retry loops, and folding those into
  `_call_agent_with_retries` is what would let them opt in — see the entry below.

### Stage 5 was dropped: the Slurm scripts stay as they are

Measured: `slurm/{fever,pddl,sciworld}_experiment.sh` are 74 lines each.
`fever` vs `sciworld` differ by **2** lines, `fever` vs `pddl` by **3**.

Dropped by direction, as experimental configuration rather than codebase — the
same reasoning that deleted the tests written for these scripts in Phase 3. A
parameterised script turns each submission from "copy the file, change the task"
into "get the flag order right at the call site", and by that same principle
nothing may test that it still works. If any of it is ever worth doing, it is the
27-line vLLM serve block that is verbatim in all three: `source` it from a shared
file and leave the three thin wrappers alone.

### Known consequence, deliberately not addressed

`working_dir` is `db_dir/model/task/mas_type/mas_memory`, so validator and plain
runs of the same task, memory and model now share a directory and append to the
same `results.csv`. **`--use_projector` has had exactly this property since it was
added**, so this is the existing design rather than a regression, and the CSV
schema was deliberately left untouched in this phase. The fix is a column, which
lands with Phase 5's `results.py` extraction.

### Also not done

- **`entrypoint.sh` cannot reach the validator arm.**
  `template/generate_templates.py` passed `autogen_mas` as
  `environmentVariable7`, which `entrypoint.sh` maps to `--mas_type`; that would
  now be rejected at parse time, so it passes `autogen`. This does change what
  that deploy path runs, from the validator arm to the plain one. Giving
  `entrypoint.sh` a way to pass `--use_validator` belongs with the Phase 6 deploy
  work, which has to touch that template anyway.
- **`README.md`'s `--mas_type` table still lists `autogen_mas`** and does not
  mention `--use_validator`. Untouched because that file carries uncommitted local
  edits; it is on the Phase 6 README item.

### Done ahead of this phase

- ~~Lift the observer pattern to `MetaMAS`~~ — `afbf5f3`. The fifth copy was in
  `autogen_hotpot.py`, deleted in stage 1.
- ~~Remove the two `time.sleep(5)` calls~~ — `8c9d764`.

---

## Which memory-update prompt should each intrinsic variant carry?

`needs-decision` · found while planning Phase 4's stage 4 · **open**

Not a defect, and not results-affecting today. It is a question about the
experiment that the code currently answers by accident.

**How the two prompts differ.** Each intrinsic module holds two:

- `memory_system_prompt` — the *role*. It tells the model what kind of thing it
  is writing: for `intrinsicmemory-pddl`, 3,309 characters of PDDL-specific
  memory structure; for `-fever` 4,562; for `-alfworld` 6,682; for `-notemplate`
  468 with no structure at all. **This is the independent variable of the
  experiment.**
- `memory_update_prompt` — the *instruction per turn*. It is the user-role
  message, formatted with five fields — `custom_message`,
  `template_instructions`, `task_description`, `task_trajectory`,
  `current_memory` — and it asks for the memory to be rewritten given the latest
  step. It is task-agnostic machinery.

So the current split is coherent: **the template varies, the update procedure is
held constant.** For an experiment whose question is *which kind of memory
template works best*, holding the update instruction fixed across arms is the
right design — it is the control.

**What is actually wrong** is only that four bundles define a
`memory_update_prompt` nothing reads:

| bundle | defines it | read | has `{custom_message}` |
|---|---|---|---|
| `INTRINSICMEMORYDEFAULT` | 373 chars | **yes, by all four** | yes |
| `INTRINSICMEMORYPDDL` | 290 | no | **no** |
| `INTRINSICMEMORYFEVER` | 290 | no | **no** |
| `INTRINSICMEMORYALFWORLD` | 290 | no | **no** |
| `INTRINSICMEMORY_NOTEMPLATE` | 331 | no | yes |
| `INTRINSICMEMORYLLMTEMPLATE` | 586 | **yes, it overrides** | yes |

The three task bundles' versions are identical to each other and are supersded
drafts: `DEFAULT` has the `{custom_message}` slot, the `OUTPUT ONLY THE UPDATED
MEMORY` instruction and a `## New Memory` trailer, none of which they have.
Wiring them up would *silently drop the solver message* that `a5a3643` added,
since their text has no slot for it.

**Recommendation: delete the four dead definitions**, leaving `DEFAULT` as the
single shared update prompt. No number moves, and the four arms then differ by
exactly the variable under test.

**The one case for a per-arm update prompt** is the reverse of the above: if a
task-specific *template* needs task-specific *filling instructions* to be used
properly, then holding the update prompt fixed under-serves the structured arms
and the comparison is unfair in the other direction. That is a real hypothesis,
but it makes the update prompt a second independent variable, and it should then
be varied deliberately for all four arms rather than inherited from three
identical drafts. `intrinsicmemory-llm-structured-template` already shows the
mechanism works — it overrides both, and Phase 3 had to un-comment that very line
(`364ac8a`) to make the arm measure anything.

Stage 4 left `update_prompt` as the class attribute where either decision lands:
one line per variant. `test_intrinsic_prompts.py` pins today's answer, so a
change to it is visible rather than silent.

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
  have changed. Already stale as of Phase 3 — the flag table gives `--mas_memory` a
  default of `none` and `--max_trials` a default of `30`; both are now
  required-or-absent, `--mas_type` too. `README.md` carries uncommitted local
  edits, so Phase 3 did not touch it.

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

**Reproduce:** `uv run python tasks/run.py --mas_type autogen --mas_memory empty`

This line used to read `--mas_memory none`, which was itself invalid — see the
entry below. So the documented smallest invocation was failing for two
independent reasons, and only one of them was ALFWorld.

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

**Phase 4 raised the value of doing it.** `MetaMAS._reviewed_attempt` (`07d6bb2`)
puts the validator's contract on the base class, so `--use_validator` would work
for DyLAN and MacNet too — but only once their loops go through
`_call_agent_with_retries`, which is what this entry is. Until then the flag is
accepted for any `--mas_type` and only `autogen` acts on it.

**What Phase 2 did give them**, so they are not stranded: both return
`EpisodeResult` (`7e8a06a` — before this, `--mas_type dylan` and `--mas_type
macnet` raised `ValueError: not enough values to unpack` on the first completed
task); inherit `add_observer`/`notify_observers` from `MetaMAS`; share the
`summarize` keyword contract (`2cc0d3c` — MacNet's `upstream_agent_ids=None` broke
all six intrinsic memory modules); share the `trials` convention (`2ead4a3`); and
are covered by the contract and smoke matrices. DyLAN also stopped ignoring the
configured embedding model (`7fcbf1d`).

---

## ~~`--mas_memory` defaulted to a value the registry does not have~~ — closed

`bug` · found while making the CLI read the registries · **closed in `8b62d40`**

`--mas_memory` defaulted to `['none']`, and `none` is not a registered module —
the registered name for no memory is `empty`:

```
python tasks/run.py --mas_type autogen
→ ValueError: Invalid MAS memory type 'none'. Allowed values: ['empty', ...]
```

The error is raised inside `build_mas`, which `run_experiment` catches per
experiment, so the default invocation **recorded a failed experiment, produced no
result rows, and exited cleanly**.

Not results-affecting: no published number can have come from a run that produced
none. What it does mean is that a bare `python tasks/run.py --mas_type autogen`
has never worked — the other default, `--task alfworld`, cannot start either.

`--mas_memory` is now **required with no default** (`eb6fd06`), because selecting
no memory module is not a meaningful experiment and so there is nothing to fall
back to. `--mas_type` was made required at the same time: it had the same defect in
a worse form — no default *and* not required, so omitting it resolved to `None`,
`build_experiment_configs` expanded that into one experiment, and `get_mas(None)`
raised inside the same per-experiment `try`. Both are validated against their
registries, so an unregistered value is rejected at parse time.

`--task` keeps `default=['alfworld']`. `alfworld` is registered, so it is a valid
selection; that it cannot be constructed is the separate open decision above, and
making the flag required would quietly resolve half of that decision. Found only because the CLI's choices were changed
to read the registries instead of repeating them, which is the general lesson: the
task list was written out four times, and the copies disagreed with each other in
ways nothing checked.

No test guards this. Once `choices` reads from a registry, argparse itself
rejects an unregistered value, so the defect is structurally impossible rather
than watched for — which is the better of the two, and why the CLI tests written
for it were deleted again on review.

---

## Guarantee the action format instead of checking it with a second agent

`needs-decision` · found while diffing `autogen` against `autogen_mas`

The validator agent exists to solve a problem the serving stack can solve for
free — and solving it that way turns a confounded comparison into a clean one.

**What the validator actually does.** Its system prompt is explicit:

```
ONLY EVALUATE THE FORMAT, NOT THE FACTUAL CORRECTNESS OF THE SOLUTION.
```

It answers `VALID`, or `INVALID: <brief explanation>`. So it is an output-format
checker implemented as an LLM call — one that can itself be wrong, which is
exactly why `_call_agent_with_retries` needed a fallback thunk (`afbf5f3`):
without it, a persistently mistaken validator would stall the episode rather than
let a disputed but well-formed action through.

**Structured output makes the format true by construction.** The endpoint is
`openai/gpt-oss-120b` behind vLLM, which supports guided decoding —
`guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` — as well as
`response_format` with a JSON schema. A malformed action stops being possible
rather than being detected after the fact. Feature-detect support the way
`temperature` now is: send it, and on a refusal fall back and remember. That
pattern already exists in `GPTChat._create` (`379ff59`).

**What it removes.**

- The validator response and the validator-memory update — roughly half the LLM
  calls per trial in `autogen_mas`, which currently does four where `autogen` does
  two.
- The second memory instance.
- The `INVALID` re-prompt path and its fallback thunk.
- The write-only validator memory. This also answers test-backlog **B4**: it *is*
  write-only. `meta_memory_validator` is constructed, given `init_task_context`,
  summarised on every attempt and saved at the end — and the `summarize` return
  value is discarded at the call site. Nothing calls `retrieve_memory` on it.

**The schema is a different shape per task,** and PDDL is the interesting case.

- **FEVER** is a clean grammar — `Search[x]`, `Lookup[x]`,
  `Finish[SUPPORTS|REFUTES|NOT ENOUGH INFO]` — so a regex covers it.
- **PDDL** already fuzzy-matches generated text against
  `env.action_space.all_ground_literals` in `_text_to_action`, so
  `guided_choice` over those literals would do more than fix formatting: it would
  make an invalid action impossible.
- **ALFWorld and ScienceWorld** are constrained vocabularies rather than grammars
  and need more thought.

**Why this is a decision and not an optimisation.** The two are not
interchangeable — the validator is a research condition, structured output is
infrastructure. As things stand, if `autogen_mas` beats `autogen` you cannot tell
whether a validator agent is useful or whether format errors were simply being
repaired.

Which is the argument for doing it: make guaranteed-valid output the baseline for
every arm, so format failures stop being a confound, and let the validator arm
test whether an LLM critic adds anything *beyond* a well-formed action. That is a
sharper question than the one currently being asked.

**Unblocked.** Phase 4's `use_validator` merge landed in `07d6bb2`, so there is
one code path to change rather than two. `MetaMAS._reviewed_attempt` is where the
validator's contract now lives, and it is what structured output would replace.

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

## ~~Per-task `max_steps` never reaches the environment~~ — closed

`bug` `silent-failure` · found while planning Phase 3 · **closed in `96de17b`**

`tasks/configs.yaml` gave every task a `max_steps` and nothing read it:
`build_task` took its budget from `--max_trials`, whose default was 30, and no
Slurm script passes the flag. FEVER, configured for 12 trials, ran with 30 — every
FEVER figure to date was produced at 2.5× the configured horizon.

Structural rather than a typo: `--task` is `nargs='+'` and `--max_trials` is one
scalar, so a sweep over two tasks could never have expressed two budgets. The
config is now the source, `--max_trials` an explicit override, and a task with no
configured budget raises rather than silently becoming 30.

FEVER's configured budget was raised to 30 by direction, so no existing number
moves — the 30 already in effect is now the one the config asks for, and a future
change to that value will take effect.

---

## G-Memory's clustering does not run

`bug` `silent-failure` · deferred out of Phase 3 by direction

Two defects behind the import that `08453f6` fixed. `cluster_tasks` at
`mas/memory/mas_memory/GMemory.py:425` cannot complete, so `--mas_memory g-memory`
fails every task that reaches `merge_insights`.

**1. `labels` is bound to `None`.** `FINCH` returns three values:

```
c          (n_samples × n_partitions) — one label vector per partition level
num_clust  the cluster count at each level
req_c      labels for req_clust, or None when req_clust is not passed
```

`_, _, labels = FINCH(X, distance='cosine')` binds `labels` to `req_c`, and
`req_clust` is never passed — so `labels` is `None` and
`zip(valid_nodes, labels)` on the next line raises `TypeError`, outside the
`try`.

Choosing a fix is a research decision, not a mechanical one: FINCH is
hierarchical and returns every level, so a level has to be picked. Either take
the finest partition, `labels = c[:, 0]`, which is the algorithm's own
first-order answer and needs no target count; or pass `req_clust=k` and read
`req_c`, fixing the cluster count by hand. `c[:, 0]` is the better default absent
a reason to want a specific `k`. Whichever is chosen, write the choice at the call
site — the number of task clusters is what the task layer *is*.

**2. The bare `except` is why nobody noticed.** `GMemory.py:449-453` catches
`Exception`, prints to stdout, and falls back to `labels = np.zeros(...)` — a
legitimate-looking clustering with one cluster containing everything. **Every
`--mas_memory g-memory` result predating `08453f6` was produced with the task
layer collapsed into a single cluster.** Narrow it, log through the recorder at
warning level, and let the rest raise: a memory module that cannot cluster should
fail the experiment, which `run.py` records per task (`5e04c0a`).

**Also in this function:** `self.task_storage._embedding_function` reaches into
Chroma's private attribute to embed each node. `TaskLayer` should hold the
embedding function it was given — the same fix `7fcbf1d` applied to DyLAN's
neurons.

**Do the offline-test entry below first.** This is a correctness change to code
that has never once executed successfully; writing it blind is guesswork. That
entry was Phase 3's stage 0 and moved here with this work.

---

## `predicate_map` collisions between PDDL domains

`bug` `silent-failure` · deferred out of Phase 3 — behaviour to be understood first

`predicate_map` at `tasks/envs/pddl_env/pddl_env.py:363-458` is one flat dict, 80
keys and 75 unique, already sectioned by domain in comments — and later sections
silently overwrite earlier ones. Against the four active domains in `TASK_NAMES`
(`barman`, `blockworld`, `gripper`, `tyreworld`):

| predicate | wins | loses | live effect |
|---|---|---|---|
| `clear` | hanoi `"The {} is clear."` | blockworld `"{} is clear."` | blockworld state text reads "The a is clear." |
| `free` | gripper `"{} is free. "` | tyreworld `"Hub {} is free."` | `(free ?x - hub)` loses the word "Hub" |
| `holding` | barman | blocks | trailing space only |
| `on` | tyreworld | blocks | none — the strings are identical |
| `move` | gripper | hanoi | none — hanoi is not in `TASK_NAMES` |

So two of the five collisions change the observation text the agent is prompted
with, in two of the four active domains. `predicate_map` is consulted from
`_literal_to_text`, which renders both the goal and the state for every step, so
the effect is on the prompt rather than on scoring — which is why fixing it means
re-running PDDL `blockworld` and `tyreworld` rather than re-tabulating them.

**The fix, when it is wanted:** `PREDICATE_MAPS: dict[str, dict[str, str]]` keyed
on `game_name`, following the comment sections already in the file, with
`_literal_to_text` selecting on `self.game_name` and keeping the existing
bare-predicate fallback for a predicate the domain does not list. The `F601`
per-file ignore comes out of `pyproject.toml` with it, which is the standing guard
against a reintroduction. Test **E1**.

**Why it is deferred:** whether a domain should inherit another's wording at all
is a question about the experiment, not about the code. Nothing here is urgent —
the wording has been stable for the life of the fork, so no result is
inconsistent with another.

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

This was Phase 3's stage 0 and moved out with the clustering work it existed to
make observable.

**Options.**

1. **Add `chromadb` to the dev group.** It does not pull torch or the CUDA stack,
   so it is affordable — the CUDA-free dev environment is currently 26 packages
   and 89MB, and the constraint that matters is not pulling nvidia wheels.
   Cheapest, and gives real coverage.
2. **Put an interface in front of the vector store** and use an in-memory
   implementation in tests. More work, but it is the DIP fix `GMemory` wants
   anyway, and Phase 5 splits that file regardless.

**Do this before the clustering entry above.** Phase 3 rewrites its FINCH
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

Phases 1–4 took the suite from **115 tests** — reaching one workflow file and one
memory module — to **417**, reaching every workflow, recorder and environment and
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

- [x] ~~**A1 — each module gets its own prompt.**~~ — `test_intrinsic_prompts.py`
  (`0fef126`), 27 tests, written before stage 4 collapsed the classes. It cannot
  be red against correct code, so it was verified by mutation: fever's constant
  swapped for PDDL's, notemplate's override deleted, llm-template's update prompt
  override cut. It also pins which update prompt each arm carries, and that the
  prompts survive the `memory.__class__(...)` rebuild `build_system` does.
- [ ] **A2 — intrinsic LLM calls are billed as intrinsic.** See the
  intrinsic-token-accounting issue.
- [ ] **A3 — the `len(task_trajectory) > 5` gate.** Boundary test at 3 (the
  initial `'\n\n>'`), 5 and 6 characters. A bare magic number that decides whether
  the first memory update happens at all.
- [ ] **A4 — `save_task_context` honours its signature.** It is annotated
  `-> MASMessage`, returns `None`, and never calls `super()` — so the context is
  never persisted. **Write this as a failing test first**: it is a defect
  specification.
- [x] ~~A5 — the LLM-structured template reaches the prompt~~ —
  `test_intrinsic_template.py`, three of eight red against the pre-fix code.
- [ ] **A6 — memory does not leak between tasks.** Run two tasks against one
  memory instance and assert `agent_intrinsic_memory` is empty at the second
  task's start. The current reset works by rebuilding `GPTChat`, which is a side
  effect, not a contract.

### B · The validator workflow

`autogen_mas.py` is fork-owned (`3b5d156`) and the best-tested file here — but the
tests assert shape, not behaviour, which is how the projector bug lived directly
underneath them.

- [x] ~~B1 — the projector actually projects~~ — `test_projector.py`, six of
  fourteen red against the pre-fix code, including for `autogen`. Widened in
  Phase 4 to every registered workflow, which turned six more red and established
  that DyLAN and MacNet had never projected either (`f4445ab`).
- [ ] **B4 — validator memory is written and read.** **Answered: it is
  write-only.** `meta_memory_validator` is constructed, given
  `init_task_context`, summarised on every attempt and saved at the end — and the
  `summarize` return value is discarded at the call site. Nothing calls
  `retrieve_memory` on it. So the test to write is the one that records that, and
  the item to act on is the structured-output entry above, which removes the call
  rather than finding it a reader. Phase 4 moved it behind `--use_validator`
  (`07d6bb2`) without changing that: the write-only memory exists only when the
  flag is set.
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

- [ ] **E1 — `predicate_map` has no colliding keys.** Deferred with its fix.
  `assert len(keys) == len(set(keys))`, then per active domain that its predicates
  render with its own template — tyreworld's `free` as `"Hub {} is free."`. One
  line for the first half, and the `F601` ruff ignore comes out with it. Land with
  Phase 3.
- [x] ~~E2 — sampling parameters reach the API~~ — `test_llm_temperature.py`,
  nine of eleven red against the pre-fix code. Also covers an endpoint that
  refuses the parameter.
- [ ] **E4 — per-experiment loggers do not cross-contaminate.**
  `mas/logging_utils.py` was added specifically to stop a handler leak between
  experiments on a reused worker. Assert two loggers built in one process write
  only to their own file — easy to reintroduce and invisible until you read the
  logs.
- [x] ~~E3 — exhausted retries are distinguishable from an empty answer~~ —
  `test_llm_layer.py`.

### Suggested order

**A1 and C1–C2 next**, before Phase 4 touches the intrinsic modules or the sweep.
A2 with the intrinsic-token work; E1 with the `predicate_map` fix whenever that is
taken up. A4 written red, deliberately.
