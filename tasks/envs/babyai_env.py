"""BabyAI: gridworld instruction following, presented to the agent as text.

The simulator hands out a symbolic 7x7 array of whatever the agent can see. The
agent here reads and writes English, so the view is rendered as sentences and the
seven simulator actions are given names an LLM will produce.

The rendering is split from the simulator on purpose: `describe_view` and the two
functions it uses take plain tuples, so the wording an agent actually reads is
testable without minigrid installed.
"""

from dataclasses import dataclass

import gymnasium as gym
from minigrid.core.actions import Actions

from .base_env import BaseEnv, BaseRecorder, clean_action_line

# Our command vocabulary onto minigrid's action names. The values are strings
# rather than `Actions` members so a wrong one raises KeyError at the call.
ACTION_ALIASES = {
    'turn left': 'left',
    'turn right': 'right',
    'go forward': 'forward',
    'pick up': 'pickup',
    'drop': 'drop',
    'toggle': 'toggle',
    'done': 'done',
}

# Terrain rather than things to interact with. Walls are handled before this,
# summarised instead of listed: most of a view is wall, and only the one straight
# ahead is a decision.
SCENERY = ('empty', 'unseen', 'floor')


def parse_action(action: str) -> str:
    """The simulator's name for the action a line asks for, or None.

    Matched exactly where it can be, and otherwise by longest prefix, so
    `pick up the green key` is the `pick up` the agent plainly meant rather than
    a wasted trial. None of the seven takes an argument, so a trailing object
    phrase carries nothing to lose - and `done` is a no-op in BabyAI, where a
    mission ends by being accomplished, so matching it loosely ends nothing.
    """
    cleaned = action.strip().lower()
    if cleaned in ACTION_ALIASES:
        return ACTION_ALIASES[cleaned]

    for command in sorted(ACTION_ALIASES, key=len, reverse=True):
        if cleaned.startswith(command):
            return ACTION_ALIASES[command]

    return None


def relative_position(column: int, row: int, width: int, height: int) -> tuple[int, int]:
    """Steps forward and steps to the right of view cell (column, row).

    The agent stands at the middle of the bottom row of its own view and looks up
    it, so `forward` grows as `row` falls, and `right` is negative to the left.
    """
    return (height - 1) - row, column - (width // 2)


def phrase_position(forward: int, right: int) -> str:
    """`2 steps forward and 1 step to the left`, and the like."""

    def steps(count: int) -> str:
        return f'{count} step' if count == 1 else f'{count} steps'

    parts = []
    if forward:
        parts.append(f'{steps(abs(forward))} {"forward" if forward > 0 else "back"}')
    if right:
        parts.append(f'{steps(abs(right))} to the {"right" if right > 0 else "left"}')

    return ' and '.join(parts) if parts else 'right where you are standing'


def name_object(kind: str, colour: str = None, state: str = None) -> str:
    """The noun phrase for one visible thing, e.g. `a locked red door`."""
    words = [word for word in (state, colour, kind) if word]
    article = 'an' if words[0][0] in 'aeiou' else 'a'

    return f'{article} {" ".join(words)}'


def describe_view(sightings: list[tuple[str, int, int]], wall_ahead: int = None) -> str:
    """The view as one sentence: the things in it nearest first, then the wall.

    `sightings` are (noun phrase, forward, right) triples. `wall_ahead` is how
    many steps of clear floor lie straight in front, which is what makes
    `go forward` a wasted trial or not; it goes last however near it is, being
    the terrain rather than something to act on.
    """
    ordered = sorted(sightings, key=lambda sighting: (abs(sighting[1]) + abs(sighting[2]), sighting[0]))
    described = [
        f'{phrase} {phrase_position(forward, right)}'
        for phrase, forward, right in ordered
    ]

    if wall_ahead is not None:
        described.append(f'a wall {phrase_position(wall_ahead, 0)}')

    if not described:
        return 'You see nothing.'

    return f'You see {", ".join(described)}.'


class BabyAIEnv(BaseEnv):
    """One BabyAI level at one seed, read and driven in English.

    The level is per-task, so nothing is built until `set_env`.
    """

    def __init__(self, env_config: dict, max_trials: int):
        super().__init__(env_config, max_trials)
        self.env = None
        self.mission: str = ''
        self.done: bool = False

    def set_env(self, configs: dict) -> tuple[str, str]:
        level: str = configs.get('level')
        seed: int = configs.get('seed')

        if level is None or seed is None:
            raise ValueError('A babyai task config needs a `level` and a `seed`.')

        self.level, self.seed = level, seed
        if self.env is not None:
            self.env.close()  # one simulator per level, and 200 levels per experiment
        self.env = gym.make(level)

        observation = self.reset()
        return f'{level}___{seed}', f'Your task: {self.mission}\n{observation}'

    def reset(self) -> str:
        """Restart the level at its seed, and describe what the agent can see.

        The seed is the task's, so the same task config is the same gridworld on
        every run and across memory modules.
        """
        observation, _ = self.env.reset(seed=self.seed)
        # A level carries its own step limit, as low as 49, which would truncate
        # an episode before its trial budget was spent. `max_trials` is the
        # budget. Set after the reset, which is where the level computes it.
        self.env.unwrapped.max_steps = self.max_trials
        self.mission = observation['mission']
        self.done = False

        return self._describe()

    def step(self, action: str) -> tuple[str, float, bool]:
        action = self.process_action(action)

        if self.is_thought(action):
            return 'OK.', 0, False

        name = parse_action(action)
        if name is None:
            return (
                f'"{action}" is not an action here. Choose one of: '
                f'{", ".join(ACTION_ALIASES)}.',
                -1,
                False,
            )

        before = self._state()
        _, reward, terminated, truncated, _ = self.env.step(Actions[name])

        # BabyAI's reward is shaped - `1 - 0.9 * step_count / max_steps` on
        # success and 0 otherwise - so any reward at all means the mission was
        # accomplished. Terminating with none means the agent walked into lava.
        self.done = bool(terminated) and reward > 0
        description = self._describe()
        ended = bool(terminated or truncated)

        # A blocked `go forward`, a `pick up` over bare floor and a `toggle`
        # facing nothing all return the view unchanged, so without this the agent
        # gets no signal that its trial bought nothing.
        if not self.done and self._state() == before:
            return f'Nothing happens. {description}', -1, ended

        return description, (1 if self.done else 0), ended

    @staticmethod
    def is_thought(action: str) -> bool:
        return 'think:' in action

    @staticmethod
    def process_action(action: str) -> str:
        return clean_action_line(action)

    def feedback(self) -> tuple[float, bool, str]:
        """Pass or fail: a BabyAI mission has no partial credit."""
        message = (
            'You successfully finished this task!' if self.done
            else f'You failed the task: {self.mission}'
        )

        return (1.0 if self.done else 0.0), self.done, message

    def _describe(self) -> str:
        """What the agent can see, plus whatever it is carrying."""
        grid, visible = self.env.unwrapped.gen_obs_grid()

        sightings: list[tuple[str, int, int]] = []
        walls_ahead: list[int] = []

        for column in range(grid.width):
            for row in range(grid.height):
                thing = grid.get(column, row)
                if thing is None or not visible[column, row]:
                    continue

                forward, right = relative_position(column, row, grid.width, grid.height)
                if (forward, right) == (0, 0):
                    continue  # the agent's own cell, which holds whatever it carries
                if thing.type == 'wall':
                    if right == 0 and forward > 0:
                        walls_ahead.append(forward)
                elif thing.type not in SCENERY:
                    sightings.append((self._name(thing), forward, right))

        description = describe_view(sightings, min(walls_ahead) if walls_ahead else None)
        carrying = self.env.unwrapped.carrying

        if carrying is not None:
            description += f' You are carrying {self._name(carrying)}.'

        return description

    def _state(self) -> tuple:
        """Everything about the agent an action could change."""
        inner = self.env.unwrapped
        carrying = inner.carrying

        return (
            tuple(inner.agent_pos),
            inner.agent_dir,
            None if carrying is None else (carrying.type, carrying.color),
        )

    @staticmethod
    def _name(thing) -> str:
        state = None
        if thing.type == 'door':
            state = 'locked' if thing.is_locked else 'open' if thing.is_open else 'closed'

        return name_object(thing.type, getattr(thing, 'color', None), state)


@dataclass
class BabyAIRecorder(BaseRecorder):

    def __post_init__(self):
        super().__post_init__()
        self.task = 'babyai'

    def task_begin(self, task_id, task_config):
        super().task_begin(task_id, task_config)
        self.log(f'---------- Task: {task_id} ----------')

    def task_end(self, episode):
        super().task_end(episode)

        averages = self.average_results()
        self.log(
            f'reward: {episode.reward}, done: {episode.done}.\n'
            f'ave reward: {averages.mean_reward}, ave done: {averages.mean_done}'
        )
