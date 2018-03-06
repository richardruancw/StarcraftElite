import os
import sys
from pysc2.env import sc2_env
_path = os.path.dirname(os.path.abspath(__file__))
_path_utils = "/".join(_path.split('/')[:-1])+"/utils/"
_path_models = "/".join(_path.split('/')[:-1])+"/models/"
sys.path.insert(0, _path_utils)
sys.path.insert(0, _path_models)
from config import config
from simple_env import SimpleScEnvDiscrete
from pg import PG

steps = 20000
step_mul = 1
with sc2_env.SC2Env(map_name="DefeatZerglingsAndBanelings",
                    step_mul=1,
                    visualize=False,
                    game_steps_per_episode=steps * step_mul) as original_env:
    env = SimpleScEnvDiscrete(original_env)
    # train model
    model = PG(env, config)
    model.run()
