"""The Jericho environment: scoring, and what it does before a rom is loaded.

`JerichoEnv` defers the interpreter to `set_env`, so the class constructs offline
and everything up to the point a rom is needed is exercised here. Driving an
actual game needs the rom suite, which is not in the repo.
"""

import json

import pytest

from tasks.envs import ENVS, TASKS_PATH, get_task
from tasks.envs.jericho_env import DEFAULT_ROM_DIR, progress
from tasks.prompts import jericho_prompt


def build(rom_dir: str = DEFAULT_ROM_DIR):
    return ENVS['jericho'](env_config={'rom_dir': rom_dir}, max_trials=30)


# ── the score, normalised to the range the agent can move through ─────────────

def test_progress_is_measured_from_the_opening_score():
    """`advent` opens on 36 of 350 and `detective` on 10 of 360.

    Measured from nothing, those two would credit the agent with progress it had
    not made before it took a single turn.
    """
    assert progress(36, 36, 350) == 0.0, 'advent would start a third of the way up its range'
    assert progress(10, 10, 360) == 0.0, 'detective would start already scoring'


def test_progress_reaches_one_only_at_the_maximum_score():
    assert progress(360, 10, 360) == 1.0
    assert progress(185, 10, 360) == 0.5, 'half the range from 10 to 360 is 185'


def test_progress_stays_within_zero_and_one():
    """A game whose score can fall must not report a negative rate, and no rate
    above 1 can be averaged with the others."""
    assert progress(-40, 10, 360) == 0.0
    assert progress(400, 10, 360) == 1.0


def test_a_game_whose_score_cannot_move_scores_zero_rather_than_dividing_by_zero():
    assert progress(0, 0, 0) == 0.0
    assert progress(5, 5, 5) == 0.0


# ── what it refuses to do ─────────────────────────────────────────────────────

@pytest.mark.parametrize('config', [{}, {'id': 'zork1'}, {'rom': 'zork1.z5'}])
def test_a_task_config_missing_the_game_or_the_rom_is_rejected(config):
    with pytest.raises(ValueError, match='`id` and a `rom`'):
        build().set_env(config)


def test_a_missing_rom_names_the_file_it_wanted():
    """The suite is downloaded separately, so this is the first thing to go wrong
    on a fresh checkout and the error has to say what to fetch."""
    with pytest.raises(FileNotFoundError, match='nowhere/detective.z5'):
        build(rom_dir='nowhere').set_env({'id': 'detective', 'rom': 'detective.z5'})


# ── getting the command out of a line of model output ─────────────────────────

@pytest.mark.parametrize(
    'raw, expected',
    [
        ('LOOK', 'LOOK'),
        ('TAKE BOOK', 'TAKE BOOK'),
        ('SMOKE', 'SMOKE'),
        ('BROKEN LOCK', 'BROKEN LOCK'),
    ],
)
def test_an_uppercase_command_survives_intact(raw, expected):
    """The older environments strip `OK` as a substring, which mangles these.

    Interactive fiction commands are conventionally uppercase - every walkthrough
    in the suite is - and `LOOK` is among the commonest commands in the genre. A
    parser given `LO` answers that it does not know the word.
    """
    assert ENVS['jericho'].process_action(raw) == expected


@pytest.mark.parametrize(
    'raw, expected',
    [
        ('> north', 'north'),
        ('Action: north', 'north'),
        ('ACTION: take paper', 'take paper'),
        ('**take paper**', 'take paper'),
        ('take paper.', 'take paper'),
        ('read leaflet!', 'read leaflet'),
        ('north\nsouth', 'north'),
        ('  north  ', 'north'),
    ],
)
def test_the_command_is_taken_out_of_a_decorated_line(raw, expected):
    """A trailing `!` matters: the parser folds it into the noun and then reports
    it does not know the word `leaflet!`."""
    assert ENVS['jericho'].process_action(raw) == expected


@pytest.mark.parametrize(
    'raw, expected',
    [
        ('1. north', 'north'),
        ('(1) north', 'north'),
        ('- north', 'north'),
        ('"north"', 'north'),
        ("'north'", 'north'),
        ('\u201cnorth\u201d', 'north'),
        ('"north".', 'north'),
        ('"north."', 'north'),
        ('```\nnorth\n```', 'north'),
        ('```text\nnorth\n```', 'north'),
        ('\n\nnorth', 'north'),
        ('OK.\nnorth', 'north'),
    ],
)
def test_a_command_is_found_through_a_list_marker_a_fence_or_quotes(raw, expected):
    """Each of these forms cost a trial before, and each is ordinary model output.

    Quotes and punctuation come off together because either can be outside the
    other - `"north".` and `"north."` both occur.
    """
    assert ENVS['jericho'].process_action(raw) == expected


def test_a_quote_that_is_part_of_the_command_is_kept():
    """Only quotes wrapping the whole line are decoration. Interactive fiction
    has commands that carry their own."""
    assert ENVS['jericho'].process_action('say "hello"') == 'say "hello"'


def test_the_command_is_taken_before_an_invented_observation():
    """Models answer with the command and then guess what the game will print.

    The first line is the command; the rest is invention and must not be sent.
    """
    assert ENVS['jericho'].process_action('take mailbox\nTaken.') == 'take mailbox'
    assert ENVS['jericho'].process_action(
        'east\n<< Temple >>\nYou are back in the Temple.'
    ) == 'east'


def test_the_action_is_preferred_when_a_thought_precedes_it():
    """A reply carrying both can only be done one way round, and taking the
    thought spends the trial achieving nothing.

    This is the failure #31 described for the ReAct environments, where a
    combined line was classified a thought and the action in it discarded.
    """
    assert ENVS['jericho'].process_action('think: the gun is worth taking\ntake gun') == 'take gun'


def test_a_thought_alone_is_still_a_thought():
    processed = ENVS['jericho'].process_action('think: I should look around first')

    assert ENVS['jericho'].is_thought(processed)


@pytest.mark.parametrize(
    'raw, expected',
    [
        ('Action 1: take paper', 'take paper'),
        ('Action: take paper', 'take paper'),
        ('ACTION 2: north', 'north'),
        ('Step 3: north', 'north'),
    ],
)
def test_a_numbered_label_comes_off_the_command(raw, expected):
    """A parser handed `Action 1: take paper` answers that it does not know the
    word `Action`, and the trial is gone."""
    assert ENVS['jericho'].process_action(raw) == expected


def test_a_command_after_a_thought_on_one_line_is_taken():
    """The shape issue #31 described, on a single line rather than two."""
    assert ENVS['jericho'].process_action(
        'think: I should look around. Action 1: look'
    ) == 'look'


def test_a_line_that_is_only_an_acknowledgement_comes_back_empty():
    """An empty action spends a retry rather than a trial, so the model gets
    another turn instead of the parser being handed `OK.`."""
    assert ENVS['jericho'].process_action('OK.') == ''
    assert ENVS['jericho'].process_action('OK') == ''


@pytest.mark.parametrize(
    'raw, expected',
    [
        ('look\x00', 'look'),
        ('\x00look', 'look'),
        ('take\x00 lamp', 'take lamp'),
        ('north\x00\x00', 'north'),
    ],
)
def test_a_nul_byte_never_survives_into_the_command(raw, expected):
    """The interpreter is a C library reached through `c_char_p`, and a NUL in
    the command aborts it: frotz reads the byte as the terminal's `Ctrl-@`,
    which its input path treats as a timeout sentinel. The process dies on a
    signal, so there is no exception for the runner to catch."""
    assert ENVS['jericho'].process_action(raw) == expected


def test_a_command_of_nothing_but_nul_bytes_comes_back_empty():
    """An empty action spends a retry rather than a trial."""
    assert ENVS['jericho'].process_action('\x00') == ''


# ── the reasoning marker the prompt and the environment have to agree on ──────

@pytest.mark.parametrize(
    'line',
    [
        'think: the paper is worth taking',
        'Think: the paper is worth taking',
        'THINK: the paper is worth taking',
        'Thought: the paper is worth taking',
        'Thought 1: the paper is worth taking',
        '> think: the paper is worth taking',
    ],
)
def test_every_spelling_a_model_uses_for_a_thought_is_recognised(line):
    """A model asked for `think:` writes `Think:` and `Thought 1:` too.

    An unrecognised thought is sent to the parser as a command, which costs a
    trial and tells the agent only that the game does not know the word.
    """
    assert ENVS['jericho'].is_thought(line) is True


@pytest.mark.parametrize('line', ['take paper', 'north', 'look', 'think'])
def test_a_command_is_not_mistaken_for_a_thought(line):
    """`think` without a colon is a command some games accept, so the colon is
    what distinguishes a reasoning step - otherwise a real action is swallowed."""
    assert ENVS['jericho'].is_thought(line) is False


def test_the_prompt_teaches_the_marker_the_environment_recognises():
    """The env classifies on `think:`; a prompt teaching any other spelling would
    have every reasoning step sent to the parser as a command."""
    assert 'think:' in jericho_prompt.jericho_solver_system_prompt
    assert jericho_prompt.jericho_few_shots, 'the few shot is what shows the format'
    assert all(
        ENVS['jericho'].is_thought(line)
        for shot in jericho_prompt.jericho_few_shots
        for line in shot.splitlines()
        if line.startswith('> think')
    )


# ── the manifest ──────────────────────────────────────────────────────────────

def test_the_loader_keeps_each_game_with_its_own_rom():
    """Reversing the two raises nothing: every task would ask for a missing rom."""
    with open(TASKS_PATH['jericho'], encoding='utf-8') as reader:
        source = [json.loads(line) for line in reader]

    tasks = get_task('jericho')

    assert [task['id'] for task in tasks] == [row['id'] for row in source]
    assert [task['rom'] for task in tasks] == [row['rom'] for row in source]
    assert all(task['env_name'] == 'jericho' for task in tasks)


def test_every_game_is_listed_once_and_names_a_z_machine_rom():
    tasks = get_task('jericho')
    ids = [task['id'] for task in tasks]

    assert len(set(ids)) == len(ids), f'duplicated games: {sorted({i for i in ids if ids.count(i) > 1})}'
    assert all(task['rom'].startswith(task['id'] + '.') for task in tasks), (
        'a rom whose name does not match its id would be scored under the wrong game'
    )
    assert all(task['rom'].rsplit('.', 1)[1].startswith('z') for task in tasks)
