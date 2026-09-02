from dataclasses import dataclass

from .wiki_env import WikiReActEnv, WikiReActRecorder


class HotpotQAEnv(WikiReActEnv):
    """Multi-hop question answering: an answer is a short free-text span."""

    task_prefix = 'Question'


@dataclass
class HotpotQARecorder(WikiReActRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = 'hotpotqa'
