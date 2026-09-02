from dataclasses import dataclass

from .wiki_env import WikiReActEnv, WikiReActRecorder


class FeverEnv(WikiReActEnv):
    """Fact verification: an answer is one of SUPPORTS, REFUTES, NOT ENOUGH INFO."""

    task_prefix = 'Claim'


@dataclass
class FeverRecorder(WikiReActRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = 'fever'
