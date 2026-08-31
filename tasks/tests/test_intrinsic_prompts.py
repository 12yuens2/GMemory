"""Each registered intrinsic memory module carries the prompts it is named for.

The modules in this family differ from one another only by which prompt
constants they hold, and a wrong one raises nothing, logs nothing and passes the
compatibility matrix - it just runs a whole experiment arm against the wrong
memory instructions.

Both properties are asserted per registry key, so a module added to
`MAS_MEMORY_MODULES` without an entry here fails rather than going uncovered.
"""

import tempfile

import pytest

from mas.module_map import MAS_MEMORY_MODULES
from mas.memory.mas_memory.prompt import (
    INTRINSICMEMORYALFWORLD,
    INTRINSICMEMORYDEFAULT,
    INTRINSICMEMORYFEVER,
    INTRINSICMEMORYLLMTEMPLATE,
    INTRINSICMEMORYPDDL,
    INTRINSICMEMORY_NOTEMPLATE,
)

from tasks.tests.fakes import FakeEmbeddingFunc, FakeLLM

SYSTEM_PROMPT_OWNER = {
    'intrinsicmemory-pddl': INTRINSICMEMORYPDDL,
    'intrinsicmemory-fever': INTRINSICMEMORYFEVER,
    'intrinsicmemory-alfworld': INTRINSICMEMORYALFWORLD,
    'intrinsicmemory-notemplate': INTRINSICMEMORY_NOTEMPLATE,
    'intrinsicmemory-llm-structured-template': INTRINSICMEMORYLLMTEMPLATE,
}

# The four plain modules override only the system prompt, so they inherit the
# base's update prompt. Whether they should is an open question, not a defect:
# see docs/BACKLOG.md. This pins today's answer so the next change to it is visible.
UPDATE_PROMPT_OWNER = {
    'intrinsicmemory-pddl': INTRINSICMEMORYDEFAULT,
    'intrinsicmemory-fever': INTRINSICMEMORYDEFAULT,
    'intrinsicmemory-alfworld': INTRINSICMEMORYDEFAULT,
    'intrinsicmemory-notemplate': INTRINSICMEMORYDEFAULT,
    'intrinsicmemory-llm-structured-template': INTRINSICMEMORYLLMTEMPLATE,
}

INTRINSIC_KEYS = sorted(key for key in MAS_MEMORY_MODULES if key.startswith('intrinsicmemory'))


def build(key: str):
    return MAS_MEMORY_MODULES[key](
        namespace=key,
        global_config={'working_dir': tempfile.mkdtemp(), 'hop': 1},
        llm_model=FakeLLM(),
        embedding_func=FakeEmbeddingFunc(),
    )


def test_every_registered_intrinsic_module_has_an_expected_prompt():
    """A module added to the registry must be named here rather than go uncovered."""
    missing = set(INTRINSIC_KEYS) - set(SYSTEM_PROMPT_OWNER)
    assert not missing, f"registered but no expected system prompt: {sorted(missing)}"

    stale = set(SYSTEM_PROMPT_OWNER) - set(INTRINSIC_KEYS)
    assert not stale, f"expected a system prompt for unregistered modules: {sorted(stale)}"

    assert set(UPDATE_PROMPT_OWNER) == set(SYSTEM_PROMPT_OWNER), (
        "every module needs an expected update prompt as well as a system prompt"
    )


# ── A1 · the system prompt ────────────────────────────────────────────────────

@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_the_module_carries_its_own_system_prompt(key):
    expected = SYSTEM_PROMPT_OWNER[key].memory_system_prompt

    assert build(key).memory_system_prompt == expected, (
        f"{key} does not carry its own memory_system_prompt"
    )


@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_the_system_prompt_is_not_empty(key):
    """The base leaves it empty, so an unset override is indistinguishable from none."""
    assert build(key).memory_system_prompt, f"{key} has an empty memory_system_prompt"


def test_no_two_modules_share_a_system_prompt():
    """The modules are distinguished by this string and by nothing else."""
    prompts = {key: build(key).memory_system_prompt for key in INTRINSIC_KEYS}

    collisions = [
        (a, b) for a in prompts for b in prompts if a < b and prompts[a] == prompts[b]
    ]
    assert not collisions, f"modules sharing one system prompt: {collisions}"


# ── A1b · the update prompt ───────────────────────────────────────────────────

@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_the_module_carries_the_expected_update_prompt(key):
    expected = UPDATE_PROMPT_OWNER[key].memory_update_prompt

    assert build(key).memory_update_prompt == expected, (
        f"{key} does not carry the update prompt recorded for it"
    )


@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_the_update_prompt_accepts_every_field_summarize_formats(key):
    """`summarize` formats five fields in; a prompt missing a slot drops one silently."""
    prompt = build(key).memory_update_prompt

    formatted = prompt.format(
        custom_message='CUSTOM',
        template_instructions='TEMPLATE',
        task_description='DESCRIPTION',
        task_trajectory='TRAJECTORY',
        current_memory='MEMORY',
    )

    for field in ('DESCRIPTION', 'TRAJECTORY', 'MEMORY'):
        assert field in formatted, f"{key}'s update prompt drops {field}"

# ── the prompts survive the way build_system rebuilds a memory ────────────────

@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_a_rebuilt_memory_keeps_its_prompts(key):
    """`build_system` builds the validator's memory as `memory.__class__(...)`.

    Anything that carries a module's prompts as a constructor argument rather
    than on the class loses them here, silently.
    """
    original = build(key)

    rebuilt = original.__class__(
        namespace=original.namespace + '_validator',
        global_config=original.global_config,
        llm_model=original.llm_model,
        embedding_func=original.embedding_func,
    )

    assert rebuilt.memory_system_prompt == original.memory_system_prompt, (
        f"{key} lost its system prompt when rebuilt from its own class"
    )
    assert rebuilt.memory_update_prompt == original.memory_update_prompt, (
        f"{key} lost its update prompt when rebuilt from its own class"
    )
