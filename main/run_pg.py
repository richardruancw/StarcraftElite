import os
import sys
from pysc2.env import sc2_env
_path = os.path.dirname(os.path.abspath(__file__))
_path_utils = "/".join(_path.split('/')[:-1])+"/utils/"
_path_models = "/".join(_path.split('/')[:-1])+"/models/"
sys.path.insert(0, _path_utils)
sys.path.insert(0, _path_models)
from config import config
from simple_env import SimpleScEnvCountinous
from pg import PG
from absl import flags

FLAGS = flags.FLAGS
FLAGS(sys.argv)
steps = 2000
step_mul = 10
with sc2_env.SC2Env(map_name="DefeatZerglingsAndBanelings",
                    step_mul=step_mul,
                    visualize=False,
                    game_steps_per_episode=steps * step_mul) as original_env:
    env = SimpleScEnvCountinous(original_env)
    # train model
    model = PG(env, config)
    model.run()
