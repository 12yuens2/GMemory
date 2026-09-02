from dataclasses import dataclass
from typing import NamedTuple
import os
import logging
import re
import time
from abc import ABC, abstractmethod

from mas.logging_utils import get_file_logger
from mas.mas import EpisodeResult

_LABEL = re.compile(r'^(?:action|command)\s*:\s*', re.IGNORECASE)
_LIST_MARKER = re.compile(r'^(?:[-+\u2022]|\(?\d+[.)])\s+')
_FENCE = re.compile(r'^\s*```')
# A leading marker is allowed because the few shots show `> think:` and an
# agent that copies them copies the marker too.
_THOUGHT = re.compile(r'^[\s>*\u2022`\-\'"]*(?:think|thought)\s*\d*\s*:', re.IGNORECASE)

# Quotes a model wraps a whole line in, straight and curly.
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ('\u201c', '\u201d'), ('\u2018', '\u2019'))


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0


def is_thought_line(line: str) -> bool:
    """Whether a line is a reasoning step rather than something to act on.

    Matches `think:`, `Think:`, `THINK:`, `Thought:` and `Thought 1:`, since a
    model asked for `think:` supplies any of them, and tolerates the `> ` marker
    the few shots are written with.

    The colon is required. `think` on its own is a verb some interactive fiction
    games accept as a command, and treating that as a reasoning step would
    swallow a real action.
    """
    return _THOUGHT.match(line) is not None


def _undecorate(line: str) -> str:
    """One line of model output with the formatting taken off.

    Returns empty for a line carrying nothing to act on - a code fence, or the
    bare acknowledgement `OK.`.
    """
    if _FENCE.match(line):
        return ''

    for decoration in ('<', '>', '*', '`', '#'):
        line = line.replace(decoration, '')

    line = _LIST_MARKER.sub('', line.strip())
    line = _LABEL.sub('', line.strip()).strip()
    if line.upper() in ('OK', 'OK.'):
        return ''

    # Punctuation and wrapping quotes come off together, since either can be
    # outside the other: `"north".` and `"north."` are both seen.
    previous = None
    while previous != line:
        previous = line
        line = line.strip(' .!?,;:')
        for opening, closing in _QUOTE_PAIRS:
            if len(line) > 1 and line.startswith(opening) and line.endswith(closing):
                line = line[1:-1]
                break

    return line.strip()


def clean_action_line(action: str, recognises=None) -> str:
    """The action out of what a model wrote, however it dressed it up.

    Every trial costs a turn of the episode's budget, so a reply that plainly
    says what to do should not be thrown away over a list marker or a pair of
    quotes.

    Where the model wrote a thought and then the action - which the prompts ask
    it not to, and which it does anyway - the action is preferred: the
    environment can only do one of the two, and taking the thought would spend
    the trial achieving nothing. `recognises` is how a caller with a fixed set of
    actions says which later lines qualify; without it, any line that is not a
    thought does.

    Note `OK.` is stripped only as a whole line. Substring-replacing it, which
    the older environments do, turns the interactive fiction command `LOOK` into
    `LO`.
    """
    lines = [line for line in (_undecorate(raw) for raw in action.splitlines()) if line]
    if not lines:
        return ''

    if not is_thought_line(lines[0]):
        return lines[0]

    for line in lines[1:]:
        if not is_thought_line(line) and (recognises is None or recognises(line)):
            return line

    return lines[0]

class BaseEnv(ABC):
    """The Env protocol, implemented over a config and a trial budget.

    `max_trials` is the budget for one episode; `get_env` always supplies it, so
    subclasses do not carry a default of their own.
    """

    def __init__(self, env_config: dict, max_trials: int):
        self.env_config: dict = env_config
        self.max_trials: int = max_trials

    @abstractmethod
    def set_env(self, task_config: dict) -> tuple[str, str]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, action: str) -> tuple[str, float, bool]:
        pass

    @classmethod
    @abstractmethod
    def process_action(cls, action: str) -> str:
        pass

    @classmethod
    @abstractmethod
    def is_thought(cls, action: str) -> bool:
        """Whether this action is a reasoning step rather than one to act on.

        The datasets spell it differently - `think:` for the embodied ones,
        `Thought N:` for the ReAct ones - and a workflow that guesses gets one
        family wrong silently, so each environment answers for its own.
        """
        pass

    @abstractmethod
    def feedback(self) -> tuple[float, bool, str]:
        pass


class AggregateResults(NamedTuple):
    """Means across the episodes recorded so far.

    Three different counts live near each other, so to fix the vocabulary:

    - a **task** is one problem from the dataset. One task produces one episode.
    - a **trial** is one turn inside an episode: one action sent to the
      environment. An episode's budget for these is `env.max_trials`.
    - **episode_count** is how many episodes went into these means. A task whose
      episode never ran is not among them.

    So `mean_trials` is the average number of turns an episode took, and
    `mean_done` is a success *rate* - which is why this is not an EpisodeResult,
    where `done` is a bool for one episode.

    `mean_trials` is averaged only over episodes that reported a trial count. An
    episode cut short by an agent that could not act reports None, since how many
    turns that task needed was never established.
    """

    mean_reward: float
    mean_done: float
    mean_trials: float
    episode_count: int


def aggregate(episodes: list[EpisodeResult]) -> AggregateResults:
    """Means over a list of episodes. Zeroes for an empty list."""
    measured_trials = [episode.trials for episode in episodes if episode.trials is not None]

    return AggregateResults(
        mean_reward=_mean([episode.reward for episode in episodes]),
        mean_done=_mean([episode.done for episode in episodes]),
        mean_trials=_mean(measured_trials),
        episode_count=len(episodes),
    )


@dataclass
class BaseRecorder:

    working_dir: str = None
    namespace: str = None
    task: str = None

    def __post_init__(self):

        self.file_path = os.path.join(self.working_dir, self.namespace + '.log')
        # log file path is unique per experiment, so using it as the logger name
        # guarantees a fresh logger even when the same namespace recurs on a
        # reused multiprocessing worker
        self.logger = get_file_logger(self.file_path, self.file_path, level=logging.DEBUG)

        self.current_task_id: int = None
        self.current_task_config: dict = {}

        self.episodes: list[EpisodeResult] = []

    def task_begin(self, task_id: int, task_config: dict) -> None:
        
        self.current_task_id = task_id
        self.current_task_config = task_config

    def task_end(self, episode: EpisodeResult) -> None:

        if self.current_task_id is None or self.current_task_config is None:
            raise RuntimeError('The task id or the task config should not be None.')

        self.episodes.append(episode)

    def average_results(self) -> AggregateResults:
        """Means over the episodes recorded so far, zero before the first one."""
        return aggregate(self.episodes)
        
    def dataset_begin(self) -> None:
        
        self.start_time = time.time()

        message: str = "=============== Task Begin ==============="
        self.log(message)

    def dataset_end(self) -> None:
        message: str = "=============== Task End ==============="

        end_time = time.time()  
        total_time = end_time - self.start_time
        time_message: str = f"Total execution time: {total_time:.2f} seconds"
        self.log(message)
        self.log(time_message)
    
    def log(self, message: str) -> None:    

        if hasattr(self, "logger") and self.logger:
            self.logger.info(message)
        else:
            print("Logger is not initialized.")
