"""FEVER and HotpotQA share a ReAct loop over Wikipedia but not an answer space.

Both present the dataset's task to the agent and score a Finish against the
dataset's answer, and the only thing separating them is which word the task is
presented under. Getting that wrong tells the agent a question is a claim, which
raises nothing - the episode runs to its trial budget answering the wrong kind
of thing.
"""

import json

import pytest

from tasks.envs import ENVS, TASKS_PATH, get_task
from tasks.envs.utils import WikipediaUnavailable
from tasks.envs.wiki_env import WikiReActEnv

WIKI_TASKS = {
    'fever': ('Claim', 'Telemundo is a English-language television network.', 'REFUTES'),
    'hotpotqa': ('Question', 'Were Pavel Urysohn and Leonid Levin known for the same '
                             'type of work?', 'yes'),
}


def build(task: str) -> WikiReActEnv:
    return ENVS[task](env_config=None, max_trials=30)


@pytest.mark.parametrize('task', sorted(WIKI_TASKS))
def test_the_task_is_presented_under_its_own_word(task):
    prefix, question, answer = WIKI_TASKS[task]
    env = build(task)

    task_main, task_description = env.set_env({'task': question, 'answer': answer})

    assert task_main == f'{prefix}: {question}'
    assert task_description == task_main


@pytest.mark.parametrize('task', sorted(WIKI_TASKS))
def test_a_finish_is_scored_against_the_datasets_answer(task):
    _, question, answer = WIKI_TASKS[task]
    env = build(task)
    env.set_env({'task': question, 'answer': answer})

    observation, reward, done = env.step(f'Finish[{answer}]')

    assert (reward, done) == (1, True), f'{task} scored its own answer {observation}'
    assert env.feedback()[:2] == (1, True)


@pytest.mark.parametrize('task', sorted(WIKI_TASKS))
def test_a_wrong_finish_ends_the_episode_unscored(task):
    _, question, answer = WIKI_TASKS[task]
    env = build(task)
    env.set_env({'task': question, 'answer': answer})

    _, reward, done = env.step('Finish[a different answer entirely]')

    assert (reward, done) == (0, True)
    assert env.feedback()[:2] == (0, False)


@pytest.mark.parametrize('task', sorted(WIKI_TASKS))
def test_a_reasoning_step_costs_a_trial_but_reaches_no_action(task):
    """One classifier per env: `step` and `_solver_stuck` must read the same one."""
    _, question, answer = WIKI_TASKS[task]
    env = build(task)
    env.set_env({'task': question, 'answer': answer})

    assert env.is_thought('Thought 1: I should search first.') is True
    assert env.step('Thought 1: I should search first.') == ('OK.', 0, False)


def test_a_hotpotqa_answer_matches_up_to_case_articles_and_punctuation():
    """FEVER's three labels never needed normalising; a free-text span does."""
    env = build('hotpotqa')
    env.set_env({'task': 'Which magazine was started first?', 'answer': "Arthur's Magazine"})

    assert env.step('Finish[the arthurs magazine]')[1] == 1


def test_the_hotpotqa_loader_asks_the_question_and_keeps_the_answer():
    """Reversing the two raises nothing: every task would hand the answer over."""
    with open(TASKS_PATH['hotpotqa'], encoding='utf-8') as reader:
        source = [json.loads(line) for line in reader][:5]

    tasks = get_task('hotpotqa', max_tasks=5)

    assert [task['task'] for task in tasks] == [row['question'] for row in source]
    assert [task['answer'] for task in tasks] == [row['answer'] for row in source]
    assert all(task['env_name'] == 'hotpotqa' for task in tasks)


# ── getting the action out of a ReAct line ────────────────────────────────────

@pytest.mark.parametrize(
    'raw, expected',
    [
        ('Search[Star Wars: A New Hope]', 'Search[Star Wars: A New Hope]'),
        ('Action 1: Search[Kill Bill: Volume 1]', 'Search[Kill Bill: Volume 1]'),
        ('Lookup[Chapter 3: the return]', 'Lookup[Chapter 3: the return]'),
    ],
)
def test_an_argument_carrying_a_colon_survives(raw, expected):
    """The `Action N:` prefix used to come off by splitting on the first colon,
    which took the argument with it.

    `Search[Star Wars: A New Hope]` became `Search[Star Wars`, which matches no
    action pattern, so the search never happened and the trial scored -1.
    Wikipedia titles carry colons constantly - films, albums, books with
    subtitles - so this was reachable by any agent searching for one.
    """
    assert ENVS['hotpotqa'].process_action(raw) == expected


@pytest.mark.parametrize('raw', ['Lookup[BOOK]', 'Search[OK Computer]', 'Finish[OK]'])
def test_an_argument_carrying_an_uppercase_ok_survives(raw):
    """`OK` was stripped as a substring, so `Lookup[BOOK]` became `Lookup[BO]`."""
    assert ENVS['hotpotqa'].process_action(raw) == raw


@pytest.mark.parametrize(
    'raw', ['Search[Thought experiment]', 'Lookup[thoughts]', 'Finish[thought]']
)
def test_an_action_that_merely_mentions_a_thought_is_still_an_action(raw):
    """`thought` was matched anywhere in the line, so searching for one was read
    as reasoning about one - and the search was discarded in silence.
    """
    env = ENVS['hotpotqa']
    processed = env.process_action(raw)

    assert env.is_thought(processed) is False, f'{raw!r} would never reach Wikipedia'
    assert processed == raw


def test_the_action_prefix_comes_off_but_the_verb_stays():
    assert ENVS['fever'].process_action('Action 2: Finish[SUPPORTS]') == 'Finish[SUPPORTS]'
    assert ENVS['fever'].process_action('Finish[SUPPORTS]') == 'Finish[SUPPORTS]'


def test_an_action_written_on_the_same_line_as_a_thought_is_taken():
    """The shape issue #31 described: one line carrying both, classified a
    thought, with the action in it discarded.
    """
    combined = 'Thought 1: I should search Telemundo. Action 1: Search[Telemundo]'

    assert ENVS['fever'].process_action(combined) == 'Search[Telemundo]'


def test_a_thought_alone_is_still_a_thought():
    env = ENVS['fever']

    for spelling in ('Thought 1: I need to search Telemundo.',
                     'thought: I need to search Telemundo.',
                     'THOUGHT 2: I need to search Telemundo.'):
        assert env.is_thought(env.process_action(spelling)) is True


def test_prose_after_a_thought_does_not_displace_it():
    """Only a line shaped like an action is preferred over a thought. A model that
    keeps talking has still only reasoned, and a thought costs nothing.
    """
    env = ENVS['fever']
    processed = env.process_action(
        'Thought 1: I should search Telemundo.\nThat seems like the right approach.'
    )

    assert env.is_thought(processed) is True


def test_an_unreachable_wikipedia_ends_the_task_rather_than_scoring_it_zero():
    """The environment must not turn this into an observation.

    A run that cannot reach Wikipedia would otherwise answer every Search with
    nothing found, score every episode zero, and read as a weak agent. Left to
    propagate, run_task records the task in failed_tasks.csv instead.
    """
    env = build('fever')
    env.set_env({'task': 'Telemundo is an English-language network.', 'answer': 'REFUTES'})

    def unreachable(argument):
        raise WikipediaUnavailable('Wikipedia could not be reached')

    env.explorer.search = unreachable

    with pytest.raises(WikipediaUnavailable):
        env.step('Search[Telemundo]')
