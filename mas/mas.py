from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterable, NamedTuple, Optional, Dict, Any

from .agents import Agent, Env
from .reasoning import ReasoningBase
from .memory import MASMemoryBase


class AgentCallFailed(RuntimeError):
    """An agent could not produce a usable action within its retry budget.

    Raised instead of falling through with `action` unbound, which is what the
    hand-written retry loops did - the resulting NameError masked whatever the
    original failure was.
    """


class RetryAgentCall(Exception):
    """Raised by an attempt that wants another try, spending one from the budget.

    Used where a workflow rejects its own agent's answer - AutoGenMAS re-prompts
    the solver when the validator returns INVALID - so that path is counted like
    any other failed attempt rather than looping without limit.

    `fallback` is a thunk producing the action the attempt would not accept. If
    the budget runs out it is called and its action used, rather than failing the
    episode: an action a reviewer disliked can still be sent to the environment,
    whereas an empty one cannot. It is a thunk and not a string so that nothing
    is computed on the retry path, where the action is about to be discarded.
    """

    def __init__(self, reason: str, fallback: Optional[Callable[[], str]] = None):
        super().__init__(reason)
        self.fallback = fallback


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
    # Declared here because _call_agent_with_retries reports through it; the four
    # workflows each carried an identical copy of the two methods below.
    observers: list = field(default_factory=list)  
    
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

    def _call_agent_with_retries(
        self,
        attempt: Callable[[], str],
        description: str,
        max_tries: int = 3,
    ) -> str:
        """Call `attempt` until it yields a non-empty action, at most max_tries times.

        Every route through the loop spends one try. The four hand-written copies
        of this loop each incremented their counter at the bottom of the body and
        reached it only on the exception path: an empty response hit
        `if action == '': continue`, which jumped straight over the increment. So
        `while tries < 3` never terminated for an LLM that kept returning "" -
        and GPTChat returned "" on any non-rate-limit API error, the condition
        most likely to persist. A failing backend hung the experiment instead of
        failing it, with the ProcessPoolExecutor parent waiting on
        future.result() with no timeout.

        On exhaustion, the last action a RetryAgentCall offered as a fallback is
        used, so a persistently rejected but non-empty action still advances the
        episode. With no fallback there is nothing to act on, and AgentCallFailed
        reaches the experiment boundary run.py already records - rather than
        leaving the caller's `action` unbound and raising NameError, which masked
        whatever the original failure was.
        """
        last_error: Optional[BaseException] = None
        fallback: Optional[str] = None

        for _ in range(max_tries):
            try:
                action = attempt()
            except RetryAgentCall as rejected:
                last_error = rejected
                fallback = rejected.fallback or fallback
                continue
            except Exception as error:
                last_error = error
                self.notify_observers(f'Error during execution of {description}: {error}')
                continue

            if action:
                return action
            self.notify_observers(f'{description} returned an empty response; retrying')

        if fallback is not None:
            self.notify_observers(
                f'{description}: {max_tries} attempts were all rejected; '
                f'proceeding with the last action anyway'
            )
            return fallback()

        raise AgentCallFailed(
            f'{description} produced no usable action in {max_tries} attempts'
        ) from last_error

    def add_observer(self, observer) -> None:
        self.observers.append(observer)

    def notify_observers(self, message: str) -> None:
        for observer in self.observers:
            observer.log(message)