"""Jericho: the interactive fiction suite, one game per task.

A game is played by typing English at a parser, so unlike the other embodied
datasets there is no action list to choose from - `get_valid_actions` exists but
is an oracle, and is deliberately not offered to the agent.

Scoring is the game's own score, normalised to the range it can actually move
through. Four of the games start above zero, so progress is measured from the
opening score rather than from nothing.
"""

from dataclasses import dataclass

from jericho import FrotzEnv

from mas.utils import repo_path

from .base_env import BaseEnv, BaseRecorder, is_thought_line

DEFAULT_ROM_DIR = 'data/jericho/roms'


def progress(score: int, start_score: int, max_score: int) -> float:
    """How far through the game's score range `score` is, in 0..1.

    Measured from `start_score` because `advent` opens on 36 of 350 and
    `detective` on 10 of 360: from nothing, those two would report progress the
    agent had not made. A game whose score cannot move scores 0.
    """
    room = max_score - start_score
    if room <= 0:
        return 0.0

    return max(0.0, min(1.0, (score - start_score) / room))


class JerichoEnv(BaseEnv):
    """One interactive fiction game, driven through the Frotz interpreter.

    The rom is per-task, so nothing is loaded until `set_env`.
    """

    def __init__(self, env_config: dict, max_trials: int):
        super().__init__(env_config, max_trials)
        self.rom_dir: str = (env_config or {}).get('rom_dir', DEFAULT_ROM_DIR)
        self.env = None
        self.done: bool = False
        self.moves: int = 0

    def set_env(self, configs: dict) -> tuple[str, str]:
        game: str = configs.get('id')
        rom: str = configs.get('rom')

        if game is None or rom is None:
            raise ValueError('A jericho task config needs an `id` and a `rom`.')

        rom_path = repo_path(self.rom_dir, rom)
        if not rom_path.exists():
            raise FileNotFoundError(
                f'No rom for {game} at {rom_path}. The suite is downloaded separately; '
                f'see data/data.md.'
            )

        self.game = game
        if self.env is not None:
            self.env.close()  # one interpreter per game, and 56 games per experiment
        self.env = FrotzEnv(str(rom_path))

        opening = self.reset()
        task_description = (
            f'{opening}\n\n'
            f'Your objective is to score as many of the {self.max_score} available '
            f'points as you can.'
        )

        return game, task_description

    def reset(self) -> str:
        """Restart the game, and return its opening text.

        Jericho 3 seeds the interpreter with the game's own walkthrough seed when
        none is given, so a task is the same game on every run.
        """
        opening, info = self.env.reset()

        self.start_score: int = self.env.get_score()
        self.max_score: int = self.env.get_max_score()
        self.moves = info.get('moves', 0)
        self.done = False

        return opening.strip()

    def step(self, action: str) -> tuple[str, float, bool]:
        action = self.process_action(action)

        if self.is_thought(action):
            return 'OK.', 0, False

        before = self.env.get_score()
        observation, _, done, info = self.env.step(action)

        moves = info.get('moves', self.moves)
        # The parser leaves the move count alone when it rejects the input
        # outright, which is this engine's `Nothing happens.` - the command was
        # never a move. A command it understood but that achieved nothing does
        # advance the counter.
        rejected = moves == self.moves
        self.moves = moves

        self.done = bool(self.env.victory())
        gained = self.env.get_score() - before
        observation = observation.strip()
        # Some games announce a score change themselves and some do not, so the
        # note is added only where the game left it out.
        if gained and 'score' not in observation.lower():
            observation += f'\n[Your score just went {"up" if gained > 0 else "down"} by {abs(gained)}.]'

        if self.done:
            reward = 1
        elif rejected:
            reward = -1
        else:
            reward = 0

        return observation, reward, bool(done or self.env.game_over() or self.done)

    @staticmethod
    def is_thought(action: str) -> bool:
        return is_thought_line(action, colon_required=True)

    def feedback(self) -> tuple[float, bool, str]:
        """The share of the game's score range reached, and whether it was won.

        Nearly every game in the suite is far longer than one episode's trial
        budget, so the score is the measure that separates the arms; `done` is
        reserved for an actual victory.
        """
        score = self.env.get_score()
        rate = progress(score, self.start_score, self.max_score)

        if self.done:
            message = 'You successfully finished this task!'
        else:
            message = (
                f'You did not finish {self.game}. You scored {score} of '
                f'{self.max_score} points.'
            )

        return rate, self.done, message


@dataclass
class JerichoRecorder(BaseRecorder):

    def __post_init__(self):
        super().__post_init__()
        self.task = 'jericho'

    def task_begin(self, task_id, task_config):
        super().task_begin(task_id, task_config)
        self.log(f'---------- Task: {task_id} ({task_config.get("id")}) ----------')

    def task_end(self, episode):
        super().task_end(episode)

        averages = self.average_results()
        self.log(
            f'reward: {episode.reward}, done: {episode.done}.\n'
            f'ave reward: {averages.mean_reward}, ave done: {averages.mean_done}'
        )
