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
