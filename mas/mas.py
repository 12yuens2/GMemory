from abc import abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Iterable, NamedTuple, Optional, Dict, Any

from .agents import Agent, Env
from .reasoning import ReasoningBase
from .memory import MASMemoryBase, SupportsProjection


class AgentCallFailed(RuntimeError):
    """An agent could not produce a usable action within its retry budget.

    Broader than a failed LLM call, though that is the usual cause. The endpoint
    failing, the model answering with an empty string every time, and
    env.process_action rejecting every answer all end here; `__cause__` carries
    which.

    Scope is one episode, not the experiment: the workflows catch it inside their
    trial loop so the task is scored as unsolved and the sweep continues.
    """


class RetryAgentCall(Exception):
    """Raised by an attempt that wants another try, spending one from the budget.

    For a workflow that rejects its own agent's answer, such as AutoGenMAS
    re-prompting its solver on an INVALID validator verdict.

    `fallback` is a thunk producing the action the attempt would not accept. If
    the budget runs out it is called and its action used rather than failing the
    episode: a disputed action can still be sent to the environment, an empty one
    cannot. A thunk rather than a string so nothing is computed on the retry path,
    where the action is discarded anyway.
    """

    def __init__(self, reason: str, fallback: Optional[Callable[[], str]] = None):
        super().__init__(reason)
        self.fallback = fallback


class EpisodeResult(NamedTuple):
    """What every workflow's `schedule` returns for one task.

    `trials` is the number of trials completed: an episode solved on its first
    step reports 1, one that exhausts a 30-trial budget reports 30. It is None when
    an agent could not act and the episode was cut short, because how many turns
    that task needed was never established - aggregates leave those out of the
    mean rather than counting them as fast or as slow. `reward` and `done` are
    still real: the task was not solved.

    A NamedTuple rather than a dataclass so `reward, done, trials = ...` keeps
    working at call sites outside this repo.
    """

    reward: float
    done: bool
    trials: Optional[int]


@dataclass
class MetaMAS:
    """Unified management of members in the MAS system to achieve overall scheduling of the MAS.
    """
    agents_team: Dict[str, Agent] = field(default_factory=dict)  
    env: Optional[Env] = None  
    meta_memory: Optional[MASMemoryBase] = None  
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

        Every route through the loop spends one try: an empty answer, a raised
        exception and a RetryAgentCall all count.

        On exhaustion, the last fallback a RetryAgentCall offered is used, so a
        persistently rejected but non-empty action still advances the episode.
        With no fallback there is nothing to act on and AgentCallFailed is raised.
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

    def _solver_stuck(self, current_action: str, action_history: list[str]) -> bool:
        """Whether the agent is going round in circles: the same action, two
        consecutive `think` steps, or three mutually near-identical actions."""
        identical = len(action_history) >= 2 and current_action == action_history[-1] and current_action == action_history[-2]

        double_think = len(action_history) >= 3 and "think" in current_action and "think" in action_history[-1] and "think" in action_history[-2]

        similar = False
        if len(action_history) >= 3:
            similarity_12 = SequenceMatcher(None, current_action, action_history[-1]).ratio()
            similarity_13 = SequenceMatcher(None, current_action, action_history[-2]).ratio()
            similarity_23 = SequenceMatcher(None, action_history[-1], action_history[-2]).ratio()

            threshold = 0.8
            similar = all(similarity > threshold for similarity in [similarity_12, similarity_13, similarity_23])

        return identical or double_think or similar

    @property
    def _projection_memory(self) -> Optional[MASMemoryBase]:
        """The memory whose insights get projected onto each role."""
        return self.meta_memory

    def _project_insights(self, insights: list[str]) -> dict[str, list[str]]:
        """One insight list per agent role, each capped at `_insights_topk`.

        Without a projector, or with a memory that cannot project, every role
        gets the same insights.
        """
        roles_rules: dict[str, list[str]] = {}
        roles = set([agent.profile for agent in self.agents_team.values()])

        memory = self._projection_memory
        if not self._use_projector or not isinstance(memory, SupportsProjection):
            for role in roles:
                roles_rules[role] = insights
        else:
            for role in roles:
                roles_rules[role] = memory.project_insights(insights, role)

        for role, role_insights in roles_rules.items():
            roles_rules[role] = role_insights[:self._insights_topk]
        return roles_rules

    def _reviewed_attempt(
        self,
        propose: Callable[[Optional[tuple[str, str]]], str],
        review: Callable[[str], str],
        process: Callable[[str], str],
    ) -> Callable[[], str]:
        """An attempt a reviewing agent may reject, for `_call_agent_with_retries`.

        `propose` is given the (action, verdict) a reviewer last rejected, or None
        on the first try, so the caller decides how to re-prompt. A verdict
        containing INVALID rejects, and offers the rejected action as the
        fallback: a disputed but well-formed action still advances the episode
        once the budget runs out, where an empty one could not.
        """
        rejected: Optional[tuple[str, str]] = None

        def attempt() -> str:
            nonlocal rejected

            action = propose(rejected)
            if not action:
                return action

            verdict = review(action)
            if "INVALID" in verdict:
                rejected = (action, verdict)
                raise RetryAgentCall(
                    'validator returned INVALID',
                    fallback=lambda disputed=action: process(disputed),
                )

            return process(action)

        return attempt

    def add_observer(self, observer) -> None:
        self.observers.append(observer)

    def notify_observers(self, message: str) -> None:
        for observer in self.observers:
            observer.log(message)