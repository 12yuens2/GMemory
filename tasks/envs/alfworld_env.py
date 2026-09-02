from dataclasses import dataclass
from typing import Union, Any
#import alfworld
import re

#from alfworld.agents.environment import get_environment
from mas.mas import EpisodeResult

from .base_env import BaseEnv, BaseRecorder, aggregate, clean_action_line, is_thought_line

prefixes = {  # tasks: task_type
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

def get_env_name_from_gamefile(gamefile: str) -> Union[str, None]:

    for k in prefixes.keys():
        if k in gamefile:
            return k
    return None


class AlfworldEnv(BaseEnv):
    def __init__(self, env_config: dict[str, Any], max_trials: int):
        super().__init__(env_config, max_trials)
        # TODO: broken - main_env is commented out, so reset() raises AttributeError.
        #self.main_env = get_environment(self.env_config['env']['type'])(self.env_config, train_eval=self.env_config['split'])

        self.reset()
    
    def set_env(self, configs: dict) -> tuple[str, str]:  
        self.gamefile = configs['env_kwargs']['gamefile']
        self.env_name: str = configs['env_name']
        self.main_env.game_files = [self.gamefile]
        
        task = configs['task']
        
        self.reset()
        return self._parse_task_main(task), self._parse_task_description(task)

    def reset(self):

        self.done = False
        self.env = self.main_env.init_env(batch_size=1)
        self.env.reset()

    def step(self, action: str) -> tuple[str, float, bool]:

        action = self.process_action(action)
        observation, reward, done, info = self.env.step([action])
        def process_ob(ob):
            if ob.startswith('You arrive at loc '):
                ob = ob[ob.find('. ')+2:]    
            return ob
        
        observation = process_ob(observation[0])

        self.done = done[0]

        if self.is_thought(action):
            observation = 'OK.' 
            processed_reward = -1
        elif observation == 'Nothing happens.':
            processed_reward = -1
        else:
            processed_reward = 0 if info['won'][0] == False else 1
        
        return observation, processed_reward, self.done
    
    @staticmethod
    def is_thought(action: str) -> bool:
        return is_thought_line(action)

    def feedback(self) -> tuple[float, bool, str]:
        success = self.done
        reward = 1.0 if success else 0.0
        message = "You successfully finished this task!" if success else "You failed the task."
        
        return reward, success, message
    
    @staticmethod
    def process_action(action: str) -> str:
        return clean_action_line(action)
    
    def _parse_task_main(self, task: str):
        return self.env_name + '-' + re.search(r'Your task is to:\s*(.+)', task, re.DOTALL).group(1).strip()
    @staticmethod
    def _parse_task_description(task: str) -> str:
        return task.split('___')[0]
            

@dataclass
class AlfworldRecorder(BaseRecorder):   
    
    def __post_init__(self):
        
        super().__post_init__()
        self.task = 'alfworld'
        # Episodes grouped by ALFWorld task type, for the per-type breakdown in
        # the log. The overall aggregate comes from BaseRecorder.
        self.episodes_by_task_type: dict[str, list[EpisodeResult]] = {
            name: [] for name in prefixes
        }

    def task_begin(self, task_id, task_config):
        super().task_begin(task_id, task_config)
        
        message: str = f'---------- Task: {task_id} ----------'
        self.log(message)
    
    def task_end(self, episode: EpisodeResult):
        super().task_end(episode)

        gamefile: str = self.current_task_config['env_kwargs']['gamefile']
        env_name = get_env_name_from_gamefile(gamefile)
        if env_name is None:
            raise ValueError('Format of the task config is wrong.')

        self.episodes_by_task_type[env_name].append(episode)

        self.log(f'done: {episode.done}, ave done: {self.average_results().mean_done}')
        for name, episodes in self.episodes_by_task_type.items():
            if episodes:
                self.log(f'  {prefixes[name]}: {aggregate(episodes)}')
