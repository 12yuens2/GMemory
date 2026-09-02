"""The BabyAI environment: the wording an agent reads, and the actions it may write.

The simulator's observation is a symbolic array of the agent's view. Turning that
into sentences is what the agent's whole understanding of the gridworld rests on,
and it is plain functions over plain tuples, so it is asserted here directly.
Driving a level needs minigrid, which the dev environment does not install.
"""

import json
import re

import pytest

from tasks.envs import ENVS, TASKS_PATH, get_task
from tasks.envs.babyai_env import (
    ACTION_ALIASES,
    describe_view,
    name_object,
    parse_action,
    phrase_position,
    relative_position,
)
from tasks.prompts import babyai_prompt

# The view is 7x7 with the agent at the middle of the bottom row.
VIEW = (7, 7)
AGENT_CELL = (3, 6)

# The numbered list of actions the solver prompt offers, e.g. "(3) `go forward`".
OFFERED_ACTION = re.compile(r'^\(\d+\)\s+`([^`]+)`', re.MULTILINE)


# ── where a cell is, relative to the agent ────────────────────────────────────

def test_the_agent_is_the_origin_of_its_own_view():
    assert relative_position(*AGENT_CELL, *VIEW) == (0, 0)


@pytest.mark.parametrize(
    'cell, expected',
    [
        ((3, 5), (1, 0)),    # straight ahead
        ((3, 0), (6, 0)),    # the far end of the view
        ((4, 6), (0, 1)),    # abreast, to the right
        ((2, 6), (0, -1)),   # abreast, to the left
        ((5, 5), (1, 2)),    # the door the simulator put at view (5, 5)
        ((1, 2), (4, -2)),   # the key the simulator put at view (1, 2)
    ],
)
def test_a_cell_is_placed_forward_and_to_the_side(cell, expected):
    """Forward grows as the row index falls, and `right` is negative to the left.

    The two odd-looking cases are taken from the simulator: with the agent at
    world (4, 6) facing north, minigrid put a door at view (5, 5) one step ahead
    and two to the right, and a key at view (1, 2) four ahead and two to the left.
    """
    assert relative_position(*cell, *VIEW) == expected


def test_the_whole_view_lies_ahead_of_the_agent_and_never_behind():
    forwards = [
        relative_position(column, row, *VIEW)[0]
        for column in range(VIEW[0]) for row in range(VIEW[1])
    ]

    assert min(forwards) == 0, 'the agent cannot see behind itself'
    assert max(forwards) == VIEW[1] - 1


# ── how that position is worded ───────────────────────────────────────────────

@pytest.mark.parametrize(
    'forward, right, expected',
    [
        (1, 0, '1 step forward'),
        (2, 0, '2 steps forward'),
        (0, 1, '1 step to the right'),
        (0, 3, '3 steps to the right'),
        (0, -1, '1 step to the left'),
        (0, -2, '2 steps to the left'),
        (2, -1, '2 steps forward and 1 step to the left'),
        (1, 2, '1 step forward and 2 steps to the right'),
        (0, 0, 'right where you are standing'),
    ],
)
def test_a_position_is_worded_in_steps_and_a_side(forward, right, expected):
    """The agent has no coordinates, so this wording is all it has to plan with.

    A count of one takes `step` and anything else `steps`, and the sign of
    `right` picks the side rather than being read out.
    """
    assert phrase_position(forward, right) == expected


@pytest.mark.parametrize(
    'kind, colour, state, expected',
    [
        ('ball', 'red', None, 'a red ball'),
        ('key', 'green', None, 'a green key'),
        ('door', 'blue', 'closed', 'a closed blue door'),
        ('door', 'purple', 'locked', 'a locked purple door'),
        ('door', 'grey', 'open', 'an open grey door'),
        ('wall', 'grey', None, 'a grey wall'),
        ('box', None, None, 'a box'),
    ],
)
def test_a_thing_is_named_state_then_colour_then_kind(kind, colour, state, expected):
    """`an` before a vowel, and the state first, so `an open grey door` reads."""
    assert name_object(kind, colour, state) == expected


# ── the view as a whole ───────────────────────────────────────────────────────

def test_the_nearest_thing_is_described_first():
    """Nearest first, because the agent acts on what is in reach."""
    description = describe_view([
        ('a blue box', 4, 1),
        ('a red ball', 1, 0),
        ('a green key', 2, 1),
    ])

    assert description == (
        'You see a red ball 1 step forward, a green key 2 steps forward and 1 step to the right, '
        'a blue box 4 steps forward and 1 step to the right.'
    )


def test_the_wall_ahead_comes_last_however_near_it_is():
    """It is the terrain, not something to act on: it says how far `go forward` gets."""
    description = describe_view([('a red ball', 3, -1)], wall_ahead=1)

    assert description == 'You see a red ball 3 steps forward and 1 step to the left, a wall 1 step forward.'


def test_a_view_of_nothing_says_so_rather_than_trailing_off():
    assert describe_view([]) == 'You see nothing.'


def test_a_view_of_only_a_wall_still_reads_as_a_sentence():
    assert describe_view([], wall_ahead=2) == 'You see a wall 2 steps forward.'


# ── the seven actions ─────────────────────────────────────────────────────────

def test_the_action_names_are_the_seven_the_simulator_has():
    assert sorted(ACTION_ALIASES.values()) == sorted(
        ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
    )


def test_the_prompt_offers_exactly_the_actions_the_environment_accepts():
    """An action in the prompt but not the table is rejected every time the agent
    obeys the prompt; one in the table but not the prompt is never used.

    Read from the numbered list the prompt offers rather than from the prose,
    which mentions several of them again in passing.
    """
    offered = set(OFFERED_ACTION.findall(babyai_prompt.babyai_solver_system_prompt))

    assert offered == set(ACTION_ALIASES), (
        f'offered but not accepted: {sorted(offered - set(ACTION_ALIASES))}; '
        f'accepted but not offered: {sorted(set(ACTION_ALIASES) - offered)}'
    )


def test_the_few_shot_only_ever_uses_offered_actions():
    commands = [
        line[2:].strip()
        for shot in babyai_prompt.babyai_few_shots
        for line in shot.splitlines()
        if line.startswith('> ')
    ]
    unknown = [
        command for command in commands
        if command not in ACTION_ALIASES and not ENVS['babyai'].is_thought(command)
    ]

    assert commands, 'the few shot is what shows the format'
    assert not unknown, f'the few shot demonstrates actions the environment rejects: {unknown}'


@pytest.mark.parametrize(
    'raw, expected',
    [
        ('go forward', 'forward'),
        ('go forward.', 'forward'),
        ('Action: go forward', 'forward'),
        ('**go forward**', 'forward'),
        ('Go Forward', 'forward'),
        ('> turn left', 'left'),
        ('turn left!', 'left'),
    ],
)
def test_an_action_is_recognised_through_the_model_s_decoration(raw, expected):
    """An unrecognised action costs a trial, so punctuation must not cost one."""
    assert parse_action(ENVS['babyai'].process_action(raw)) == expected


@pytest.mark.parametrize(
    'raw, expected',
    [
        ('pick up the green key', 'pickup'),
        ('drop it next to the box', 'drop'),
        ('toggle the red door', 'toggle'),
    ],
)
def test_an_action_with_the_object_named_is_still_that_action(raw, expected):
    """None of the seven takes an argument, and the mission text names objects, so
    a model that writes `pick up the green key` plainly means `pick up`."""
    assert parse_action(ENVS['babyai'].process_action(raw)) == expected


@pytest.mark.parametrize('raw', ['fly to the ball', 'jump', 'go backward', ''])
def test_an_action_the_simulator_does_not_have_is_not_guessed_at(raw):
    """Rejecting it returns the seven to the agent; guessing would act on something
    it never asked for."""
    assert parse_action(raw) is None


def test_a_line_that_is_only_an_acknowledgement_comes_back_empty():
    assert ENVS['babyai'].process_action('OK.') == ''


@pytest.mark.parametrize(
    'line',
    [
        'think: the ball is to my left',
        'Think: the ball is to my left',
        'THINK: the ball is to my left',
        'Thought: the ball is to my left',
        'Thought 1: the ball is to my left',
        '> think: the ball is to my left',
    ],
)
def test_every_spelling_a_model_uses_for_a_thought_is_recognised(line):
    """A model asked for `think:` writes `Think:` too - the 0.5B model used to
    check this alternated between the two within one episode."""
    assert ENVS['babyai'].is_thought(line) is True


@pytest.mark.parametrize('line', ['turn left', 'pick up', 'done'])
def test_an_action_is_not_mistaken_for_a_thought(line):
    assert ENVS['babyai'].is_thought(line) is False


@pytest.mark.parametrize(
    'raw, expected',
    [
        ('1. turn left', 'left'),
        ('(1) turn left', 'left'),
        ('- turn left', 'left'),
        ('"turn left"', 'left'),
        ('\u201cturn left\u201d', 'left'),
        ('"turn left".', 'left'),
        ('```\nturn left\n```', 'left'),
        ('```text\nturn left\n```', 'left'),
        ('\n\nturn left', 'left'),
        ('OK.\nturn left', 'left'),
    ],
)
def test_an_action_is_found_through_a_list_marker_a_fence_or_quotes(raw, expected):
    """Each of these forms cost a trial before, and each is ordinary model output."""
    assert parse_action(ENVS['babyai'].process_action(raw)) == expected


def test_the_action_is_taken_before_an_invented_observation():
    """Models answer with the action and then guess what the view will become."""
    processed = ENVS['babyai'].process_action(
        'pick up\nYou see a grey ball 1 step forward and 1 step to the right.'
    )

    assert parse_action(processed) == 'pickup'


def test_the_action_is_preferred_when_a_thought_precedes_it():
    """A reply carrying both can only be done one way round, and taking the
    thought spends the trial achieving nothing."""
    processed = ENVS['babyai'].process_action('think: the ball is to my left\nturn left')

    assert parse_action(processed) == 'left'


def test_a_thought_followed_by_prose_stays_a_thought():
    """The seven actions are a closed set, so a line after a thought is only
    preferred over it when it is one of them. Prose is not, and a thought at
    least costs nothing.
    """
    processed = ENVS['babyai'].process_action(
        'think: the ball is to my left\nI will now proceed carefully'
    )

    assert ENVS['babyai'].is_thought(processed)


# ── what it refuses to do ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    'config', [{}, {'level': 'BabyAI-GoToRedBall-v0'}, {'seed': 3}]
)
def test_a_task_config_missing_the_level_or_the_seed_is_rejected(config):
    """The seed is half the task: without it the gridworld would differ per run."""
    with pytest.raises(ValueError, match='`level` and a `seed`'):
        ENVS['babyai'](env_config={}, max_trials=30).set_env(config)


# ── the manifest ──────────────────────────────────────────────────────────────

def test_the_loader_keeps_each_level_with_its_own_seed():
    """Reversing the two raises nothing: the tasks would just be other gridworlds."""
    with open(TASKS_PATH['babyai'], encoding='utf-8') as reader:
        source = [json.loads(line) for line in reader]

    tasks = get_task('babyai')

    assert [task['level'] for task in tasks] == [row['level'] for row in source]
    assert [task['seed'] for task in tasks] == [row['seed'] for row in source]
    assert all(task['env_name'] == 'babyai' for task in tasks)


def test_no_level_and_seed_pair_appears_twice():
    tasks = get_task('babyai')
    pairs = [(task['level'], task['seed']) for task in tasks]

    assert len(set(pairs)) == len(pairs), 'a repeated pair is the same gridworld twice'


def test_a_prefix_of_the_dataset_spans_the_levels_rather_than_one_of_them():
    """`max_tasks` takes the first n, so the manifest is ordered seed-major: a
    short run has to be a spread across the levels, not ten seeds of the easiest.
    """
    tasks = get_task('babyai')
    levels = {task['level'] for task in tasks}

    prefix = get_task('babyai', max_tasks=len(levels))

    assert {task['level'] for task in prefix} == levels, (
        'the first pass over the manifest does not cover every level'
    )
