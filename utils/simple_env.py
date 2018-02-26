import sys
import time
from collections import namedtuple

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env

from absl import flags
import numpy as np

from utils import simple_run_loop

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECTED = features.SCREEN_FEATURES.selected.index

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id

# args for action funs
_SELECT_NOT_ADD = 0
_NOT_QUEUED = 0


class SimpleScEnvDiscrete:
    def __init__(self, env, split_base=30):
        self._env = env
        self.wrapped_actions = []
        self.split_base = split_base

        # maintain the timestep to get pysc2 env data from last step
        self.last_timestep = self._env.step([actions.FunctionCall(_NO_OP, [])])[0]

        # initialize functions
        self._populate_actions_funcs()
        self.num_actions = len(self.wrapped_actions)

        self.select_actions_id_range = None
        self.move_attack_id_range = None

        # key environment variables
        self.last = False

    def _operation_func_factory(self, op_type, direction):
        def f(env):
            selected = self.last_timestep.observation['screen'][_SELECTED]

            x_lim, y_lim = selected.shape
            select_y, select_x = (selected == 1).nonzero()
            if len(select_x) == 0 or len(select_y) == 0:
                return env.step([actions.FunctionCall(_NO_OP, [])])
            else:
                loc = [int(select_x.mean()), int(select_y.mean())]

            # print("Raw loc: {}".format(loc))
            # move to direction
            loc[0], loc[1] = loc[0] + direction[0], loc[1] + direction[1]

            # make sure destination are in bound
            loc[0] = min(max(0, loc[0]), x_lim - 1)
            loc[1] = min(max(0, loc[1]), y_lim - 1)

            # print("Move to: {}".format(loc))

            # env.step() returns a timestep object
            return env.step([actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], loc])])

        return f

    def _select_func_factory(self, lower_left, upper_right):
        def f(env):
            print("Choose from {} to {}".format(lower_left, upper_right))
            return env.step([actions.FunctionCall(_SELECT_RECT, [[_SELECT_NOT_ADD], lower_left, upper_right])])
        return f

    def _populate_actions_funcs(self):
        # 0 -> no_op
        self.wrapped_actions.append(lambda env: env.step([actions.FunctionCall(_NO_OP, [])]))

        # select split screen observation
        x_lim, y_lim = self.last_timestep.observation['screen'][_SELECTED].shape

        base_add = int(x_lim / self.split_base)
        for i in range(0, x_lim - 1, base_add):
            for j in range(0, y_lim - 1, base_add):
                lower_left = [i, j]
                upper_right = i + (base_add -1), j + (base_add -1)

                f = self._select_func_factory(lower_left, upper_right)

                self.wrapped_actions.append(f)

        print("Populated select actions with id up to: {}".format(len(self.wrapped_actions) - 1))
        self.select_actions_id_range = [1, len(self.wrapped_actions)]
        # move one step around the mean
        for op_tpye in [_MOVE_SCREEN, _ATTACK_SCREEN]:
            for dir in [[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]]:

                dir[0] *= 5
                dir[1] *= 5

                f = self._operation_func_factory(op_tpye, dir)
                self.wrapped_actions.append(f)

        self.move_attack_id_range = [self.select_actions_id_range[1], len(self.wrapped_actions)]
        print("Populated move/attact actions")

    def get_reward(self):
        self.compute_reward(self.last_timestep)

    def get_features(self):
        self.extract_features(self.last_timestep)

    def compute_reward(self, timestep):
        return timestep.reward

    def extract_features(self, timestep):
        return timestep.observation['screen']

    def update(self):
        self.last = self.last_timestep.last()

    def step(self, act_idx):
        self.last_timestep = self.wrapped_actions[act_idx](self._env)[0]
        self.update()
        feedback = namedtuple('feedback', ['features', 'reward'])
        return feedback(self.get_features(), self.get_reward())


class DumbAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def step(self, features):
        time.sleep(0.1)
        c = np.random.randint(0, self.num_actions)
        return c


def test():
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


if __name__ == "__main__":
    test()