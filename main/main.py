import os
import pickle
import sys
_path = os.path.dirname(os.path.abspath(__file__))
_path_utils = "/".join(_path.split('/')[:-1])+"/utils/"
_path_models = "/".join(_path.split('/')[:-1])+"/models/"
sys.path.insert(0, _path_utils)
sys.path.insert(0, _path_models)
from DDQN_model import DDQN
from simple_env import SimpleScEnvDiscrete

from collections import namedtuple

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env

from absl import flags
import numpy as np

import simple_run_loop

print "Successfully import models & utils"

flags.FLAGS(sys.argv)
    steps = 20000
    step_mul = 1
    with sc2_env.SC2Env(map_name="DefeatZerglingsAndBanelings",
                        step_mul=1,
                        visualize=True,
                        game_steps_per_episode=steps * step_mul) as env:
        simpleSC = SimpleScEnvDiscrete(env)
        dumb_agent = DumbAgent(simpleSC.num_actions)
        simple_run_loop.simple_run_loop(simpleSC, dumb_agent)