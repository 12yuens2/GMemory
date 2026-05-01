from dataclasses import dataclass

from mas.agents import Agent
from mas.memory.common import MASMessage, AgentMessage
from mas.mas import MetaMAS
from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.memory import MASMemoryBase, GMemory
from mas.agents import Env

from .coder_reviewer_prompt import CODER_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT
from ..format import format_task_prompt_with_insights, format_task_context


@dataclass
class CoderReviewer(MetaMAS):

    def __post_init__(self):
        self.coder_name: str = 'coder'
        self.reviewer_name: str = 'reviewer'
        self.observers = []
        # No stop strings: coder and reviewer both produce multi-line output
        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=[])

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

        if not isinstance(reasoning, ReasoningBase):
            raise TypeError("reasoning module must be an instance of ReasoningBase")
        if not isinstance(mas_memory, MASMemoryBase):
            raise TypeError("mas_memory module must be an instance of MASMemoryBase")

        coder = Agent(
            name=self.coder_name,
            role='coder',
            system_instruction=CODER_SYSTEM_PROMPT,
            reasoning_module=reasoning,
            memory_module=None
        )
        reviewer = Agent(
            name=self.reviewer_name,
            role='reviewer',
            system_instruction=REVIEWER_SYSTEM_PROMPT,
            reasoning_module=reasoning,
            memory_module=None
        )

        self.hire([coder, reviewer])
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
        coder: Agent = self.get_agent(self.coder_name)
        reviewer: Agent = self.get_agent(self.reviewer_name)
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

        # History of failed attempts: list of {trial, code, review}
        attempt_history: list[dict] = []

        for trial in range(env.max_trials):

            # --- Coder turn ---
            base_prompt: str = format_task_prompt_with_insights(
                few_shots=few_shots,
                memory_few_shots=successful_shots,
                insights=roles_rules.get(coder.profile, raw_rules),
                task_description=self.meta_memory.summarize()
            )
            coder_prompt: str = self._append_history_for_coder(base_prompt, attempt_history, trial, env.max_trials)
            self.notify_observers(coder_prompt)

            tries = 0
            code_action: str = ''
            while tries < 3:
                try:
                    code_action = coder.response(coder_prompt, self.reasoning_config)
                    if code_action:
                        code_action = env.process_action(code_action)
                        break
                except Exception as e:
                    print(f"Error in coder response: {e}")
                tries += 1
            if not code_action:
                code_action = 'Agent failed to produce response'

            # --- Reviewer turn ---
            reviewer_prompt: str = self._format_reviewer_prompt(
                task_description=self.meta_memory.summarize(),
                proposed_code=code_action,
                attempt_history=attempt_history,
                trial=trial
            )

            tries = 0
            review: str = ''
            while tries < 3:
                try:
                    review = reviewer.response(reviewer_prompt, self.reasoning_config)
                    if review:
                        break
                except Exception as e:
                    print(f"Error in reviewer response: {e}")
                tries += 1
            if not review:
                review = 'Agent failed to produce response'

            self.notify_observers(f"Trial {trial + 1} — Code:\n{code_action}")
            self.notify_observers(f"Trial {trial + 1} — Review:\n{review}")

            # Log both agent turns to memory graph
            coder_msg = AgentMessage(
                agent_name=coder.name,
                system_instruction=coder.system_instruction,
                user_instruction=coder_prompt,
                message=code_action,
            )
            reviewer_msg = AgentMessage(
                agent_name=reviewer.name,
                system_instruction=reviewer.system_instruction,
                user_instruction=reviewer_prompt,
                message=review,
            )
            self.meta_memory.add_agent_node(coder_msg, upstream_agent_ids=[])
            self.meta_memory.add_agent_node(reviewer_msg, upstream_agent_ids=[])

            # --- Execute code ---
            observation, reward, done = env.step(code_action)
            self.notify_observers(f"Obs {trial + 1}: {observation}")
            self.meta_memory.move_memory_state(code_action, observation, reward=reward)

            if done:
                break

            # Record failed attempt — no error details passed to agents
            attempt_history.append({
                'trial': trial + 1,
                'code': code_action,
                'review': review,
            })

        # Final feedback and memory update
        final_reward, final_done, final_feedback = env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)
        self.meta_memory.backward(final_done)

        return final_reward, final_done

    # -----------------------------------------------------------------------
    # Prompt helpers
    # -----------------------------------------------------------------------

    def _append_history_for_coder(
        self,
        base_prompt: str,
        attempt_history: list[dict],
        trial: int,
        max_trials: int
    ) -> str:
        if not attempt_history:
            return base_prompt
        history_lines = ["\n\n## Previous Failed Attempts"]
        for h in attempt_history:
            history_lines.append(f"\n### Attempt {h['trial']} — FAILED")
            history_lines.append(f"Code submitted:\n```python\n{h['code']}\n```")
            history_lines.append(f"Reviewer feedback: {h['review']}")
        history_lines.append(
            f"\nThis is attempt {trial + 1} of {max_trials}. "
            "Address the reviewer's concerns and try a different approach."
        )
        return base_prompt + "\n".join(history_lines)

    def _format_reviewer_prompt(
        self,
        task_description: str,
        proposed_code: str,
        attempt_history: list[dict],
        trial: int
    ) -> str:
        lines = [f"## Problem\n{task_description}"]
        if attempt_history:
            lines.append("## Previous Failed Attempts")
            for h in attempt_history:
                lines.append(f"### Attempt {h['trial']} — FAILED")
                lines.append(f"```python\n{h['code']}\n```")
        lines.append(f"## Proposed Code (Attempt {trial + 1})")
        lines.append(f"```python\n{proposed_code}\n```")
        lines.append(
            "Please review the code carefully and provide specific feedback on any logical errors, "
            "edge cases, or bugs. If it looks correct, say so."
        )
        return "\n\n".join(lines)

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
