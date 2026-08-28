from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple, Optional, Dict, Any

from .agents import Agent, Env
from .reasoning import ReasoningBase
from .memory import MASMemoryBase


class EpisodeResult(NamedTuple):
    """What every workflow's `schedule` returns for one task.

    A NamedTuple rather than a plain dataclass so `reward, done, trials = ...`
    keeps working at call sites outside this repo (analysis notebooks, Slurm
    wrappers) while the fields finally have names.

    `trials` counts the loop index the episode ended on, so an episode solved on
    the first step reports 0. That is the convention the AutoGen workflows
    already used; it is recorded here rather than changed, because the mean
    trials column of every existing result CSV was produced under it.
    """

    reward: float
    done: bool
    trials: int


@dataclass
class MetaMAS:
    """Unified management of members in the MAS system to achieve overall scheduling of the MAS.
    """
    agents_team: Dict[str, Agent] = field(default_factory=dict)  
    env: Optional[Env] = None  
    meta_memory: Optional[MASMemoryBase] = None  
    
    def hire(self, agents: Iterable[Agent]) -> None:
        for agent in agents:
            if agent.name not in self.agents_team:
                self.agents_team[agent.name] = agent  
            else:
                print(f"Agent {agent.name} is already in the team.")  

    def set_env(self, env: Env) -> None:
        self.env = env
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        return self.agents_team.get(agent_name)
    
    @abstractmethod
    def build_system(self, reasoning: ReasoningBase, mas_memory: MASMemoryBase) -> Any:
        pass
    
    @abstractmethod
    def schedule(self, task_config: dict) -> EpisodeResult:
        pass