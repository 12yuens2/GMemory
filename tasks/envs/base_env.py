from dataclasses import dataclass, field
import os
import logging
import time
from abc import ABC, abstractmethod

from mas.agents import Env
from mas.logging_utils import get_file_logger

class BaseEnv(Env, ABC):
     
    def __init__(self, env_config: dict, max_trials: int):
        pass
    
    @abstractmethod
    def set_env(self, task_config: dict) -> tuple[str, str]:
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
        self.logger = get_file_logger(self.file_path, self.file_path, level=logging.DEBUG, echo_console=True)

        self.current_task_id: int = None
        self.current_task_config: dict = field(default_factory=dict)

        # record total returns and rewards, and number of steps taken (trials)
        self.total_rewards = []
        self.total_dones = []
        self.total_trials = []

    def task_begin(self, task_id: int, task_config: dict) -> None:
        
        self.current_task_id = task_id
        self.current_task_config = task_config

    def task_end(self, reward: float, done: bool, trials: int) -> None:

        if self.current_task_id is None or self.current_task_config is None:
            raise RuntimeError('The task id or the task config should not be None.')

        self.total_rewards.append(reward)
        self.total_dones.append(done)
        self.total_trials.append(trials)

    def average_results(self):
        """ Returns a tuple (average rewards, average dones) """
        rewards = 0
        dones = 0
        trials = 0

        if self.total_rewards:
            rewards = sum(self.total_rewards) / len(self.total_rewards)
        if self.total_dones:
            dones = sum(self.total_dones) / len(self.total_dones)
        if self.total_trials:
            trials = sum(self.total_trials) / len(self.total_trials)

        return rewards, dones, trials
        
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
