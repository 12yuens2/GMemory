from dataclasses import dataclass

from mas.agents import Agent
from mas.memory.common import MASMessage, AgentMessage
from mas.mas import MetaMAS
from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.memory import MASMemoryBase, GMemory
from mas.agents import Env

from ..autogen.autogen_prompt import solver_system_prompt
from ..format import format_task_prompt_with_insights, format_task_context


@dataclass
class SingleAgent(MetaMAS):

    def __post_init__(self):
        self.solver_name: str = 'solver'
        self.observers = []
        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=['\n'])

    def build_system(self, reasoning: ReasoningBase, mas_memory: MASMemoryBase, env: Env, config: dict):

        self._successful_topk: int = config.get('successful_topk', 1)
        self._failed_topk: int = config.get('failed_topk', 1)
        self._insights_topk: int = config.get('insights_topk', 3)
        self._threshold: float = config.get('threshold', 0)
        self._use_projector: bool = config.get('use_projector', False)
        self.notify_observers(f"Successful Topk   : {self._successful_topk}")
        self.notify_observers(f"Failed Topk       : {self._failed_topk}")
        self.notify_observers(f"Insights Topk     : {self._insights_topk}")
        self.notify_observers(f"Retrieve Threshold: {self._threshold}")
        self.notify_observers(f"Use Role Projector: {self._use_projector}")

        if 'stop_strs' in config:
            self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=config['stop_strs'])

        if not isinstance(reasoning, ReasoningBase):
            raise TypeError("reasoning module must be an instance of ReasoningBase")
        if not isinstance(mas_memory, MASMemoryBase):
            raise TypeError("mas_memory module must be an instance of MASMemoryBase")

        solver_agent = Agent(
            name=self.solver_name,
            role='solver',
            system_instruction=solver_system_prompt,
            reasoning_module=reasoning,
            memory_module=None
        )

        self.hire([solver_agent])
        self.set_env(env)
        self.meta_memory = mas_memory

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.log(message)

    def schedule(self, task_config: dict) -> tuple[float, bool]:
        if task_config.get('task_main') is None:
            raise ValueError("Missing required key `task_main` in task_config")
        if task_config.get('task_description') is None:
            raise ValueError("Missing required key `task_description` in task_config")

        task_main: str = task_config['task_main']
        task_description: str = task_config['task_description']
        few_shots: list[str] = task_config.get('few_shots', [])

        env: Env = self.env
        solver: Agent = self.get_agent(self.solver_name)
        env.reset()

        self.meta_memory.init_task_context(task_main, task_description)

        successful_trajectories, _, insights = self.meta_memory.retrieve_memory(
            query_task=task_main,
            successful_topk=self._successful_topk,
            failed_topk=self._failed_topk,
            insight_topk=self._insights_topk,
            threshold=self._threshold
        )
        successful_shots: list[str] = [
            format_task_context(t.task_description, t.task_trajectory, t.get_extra_field('key_steps'))
            for t in successful_trajectories
        ]
        raw_rules: list[str] = list(insights)
        roles_rules: dict[str, list[str]] = self._project_insights(raw_rules)

        for i in range(env.max_trials):
            user_prompt: str = format_task_prompt_with_insights(
                few_shots=few_shots,
                memory_few_shots=successful_shots,
                insights=roles_rules.get(solver.profile, raw_rules),
                task_description=self.meta_memory.summarize()
            )
            self.notify_observers(user_prompt)

            action: str = ''
            try:
                action = solver.response(user_prompt, self.reasoning_config)
                action = env.process_action(action)
            except Exception as e:
                print(f'Error during solver response: {e}')

            agent_message = AgentMessage(
                agent_name=solver.name,
                system_instruction=solver.system_instruction,
                user_instruction=user_prompt,
                message=action,
            )
            self.meta_memory.add_agent_node(agent_message, upstream_agent_ids=[])

            observation, reward, done = env.step(action)
            self.notify_observers(f'Act {i + 1}: {action}\nObs {i + 1}: {observation}')
            self.meta_memory.move_memory_state(action, observation, reward=reward)

            if done:
                break

        final_reward, final_done, final_feedback = self.env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)
        self.meta_memory.backward(final_done)

        return final_reward, final_done

    def _project_insights(self, insights: list[str]) -> dict[str, list[str]]:
        roles_rules: dict[str, list[str]] = {}
        roles = set(agent.profile for agent in self.agents_team.values())
        if not self._use_projector or not isinstance(self.meta_memory, GMemory):
            for role in roles:
                roles_rules[role] = insights
        else:
            for role in roles:
                roles_rules[role] = self.meta_memory.project_insights(insights, role)
        for role in roles_rules:
            roles_rules[role] = roles_rules[role][:self._insights_topk]
        return roles_rules