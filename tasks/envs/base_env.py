from dataclasses import dataclass
from typing import NamedTuple
import os
import logging
import time
from abc import ABC, abstractmethod

from mas.logging_utils import get_file_logger
from mas.mas import EpisodeResult


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0

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
