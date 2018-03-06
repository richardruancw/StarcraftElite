import sys
import time
from collections import namedtuple

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env

from absl import flags
import numpy as np

import simple_run_loop

SCREEN_FEATURES_IDX = []
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
SCREEN_FEATURES_IDX.append(_PLAYER_RELATIVE)

_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
SCREEN_FEATURES_IDX.append(_UNIT_HIT_POINTS)

_UNIT_HIT_POINTS_RATIO = features.SCREEN_FEATURES.unit_hit_points_ratio.index
SCREEN_FEATURES_IDX.append(_UNIT_HIT_POINTS_RATIO)

_UNIT_DENSITY_AA = features.SCREEN_FEATURES.unit_density_aa.index
SCREEN_FEATURES_IDX.append(_UNIT_DENSITY_AA)

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
SCREEN_FEATURES_IDX.append(_UNIT_TYPE)

_SELECTED = features.SCREEN_FEATURES.selected.index

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id

# args for action funs
_SELECT_NOT_ADD = 0
_NOT_QUEUED = 0

#
MAX_UNIT_DENSITY_AA = 16
MAP_SIZE = 64
ACTION_DIM_COUNTINOUS = 13


class SimpleScEnv(object):
    def __init__(self, env):
        # key environment variables
        self.last = False
        self._env = env
        self.cum_score = 0
        self.last_timestep = None
        # maintain the timestep to get pysc2 env data from last step
        self.reset()
        self.no_op_action = lambda env: env.step([actions.FunctionCall(_NO_OP, [])])

    def reset(self):
        self.last_timestep = self._env.reset()[0]
        self.update()
        return self.get_features()

    def update(self):
        self.last = self.last_timestep.last()
        self.cum_score = self.last_timestep.observation['score_cumulative'][0]

    def get_reward(self):
        return self.compute_reward(self.last_timestep)

    def get_features(self):
        return self.extract_features(self.last_timestep)

    def compute_reward(self, timestep):
        return timestep.observation['score_cumulative'][0] - self.cum_score

    def extract_features(self, timestep):

        curr_features = timestep.observation['screen'].copy()
        curr_features = curr_features.astype(np.float32)
        # make the density between 0 and 1
        curr_features[_UNIT_DENSITY_AA, :, :] /= MAX_UNIT_DENSITY_AA
        # make the hits points ratio between 0 and 1
        curr_features[_UNIT_HIT_POINTS_RATIO, :, :] /= 255

        curr_features = curr_features[SCREEN_FEATURES_IDX, :, :]

        return curr_features

    def step(self):
        raise NotImplementedError("Subclasses should implement the step function!")


class SimpleScEnvCountinous(SimpleScEnv):
    def __init__(self, env):
        super(SimpleScEnvCountinous, self).__init__(env)

        self.action_dim = ACTION_DIM_COUNTINOUS
        self.observation_dim = [len(SCREEN_FEATURES_IDX), MAP_SIZE, MAP_SIZE]

    def _position_fixer(self, x):
        return int(max(0, min(x, MAP_SIZE - 1)))

    def step(self, move_action, attack_action, attack_prob):
        assert len(move_action) == len(attack_action), "The move and attack action should have the same dimension!"
        assert len(move_action) + len(attack_action) + 1 == self.action_dim, "The total input dimension doesn't equal to {} !".format(self.action_dim)
        assert (attack_prob >= 0) and (attack_prob <= 1), "The attack probability should be between 0 and 1!"

        # with attack probability to attack
        attack_flag = 1 if np.random.rand() < attack_prob else 0
        if attack_flag:
            op_type = _ATTACK_SCREEN
            agent_action = attack_action
        else:
            op_type = _MOVE_SCREEN
            agent_action = move_action

        # check bounds for position
        from_pos = [self._position_fixer(x) for x in agent_action[:2]]
        end_pos = [self._position_fixer(x) for x in agent_action[2:4]]
        target_pos = [self._position_fixer(x) for x in agent_action[4:6]]

        curr_reward = 0
        # select and move/attack if such actions are available
        if not self.last and _SELECT_RECT in self.last_timestep.observation["available_actions"]:
            self.last_timestep = self._env.step([actions.FunctionCall(_SELECT_RECT,
                                                                      [[_SELECT_NOT_ADD], from_pos, end_pos])])[0]
            curr_reward += self.get_reward()
            self.update()

            # then with attack_prob to attack
            if not self.last and op_type in self.last_timestep.observation["available_actions"]:
                self.last_timestep = self._env.step([actions.FunctionCall(op_type, [[_NOT_QUEUED], target_pos])])[0]
                curr_reward += self.get_reward()
                self.update()
        else:
            if not self.last:
                self.last_timestep = self.no_op_action(self._env)[0]
                curr_reward += self.get_reward()
                self.update()

        taken_actions = np.hstack([from_pos, end_pos, target_pos])

        feedback = namedtuple('feedback', ['features', 'reward', 'taken_actions', 'attack_flag'])
        return feedback(self.get_features(), curr_reward, taken_actions, attack_flag)


class SimpleScEnvDiscrete(SimpleScEnv):
    def __init__(self, env, split_base=30):
        # init from base
        super(SimpleScEnvDiscrete, self).__init__(env)

        # discrete specific data
        self.split_base = split_base
        self.num_actions = None
        self.select_actions_list = []
        self.move_attack_actions_list = []
        # initialize functions
        self._populate_actions_funcs()

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

            # env.step() returns a timestep object
            return env.step([actions.FunctionCall(op_type, [[_NOT_QUEUED], loc])])

        return f

    def _select_func_factory(self, lower_left, upper_right):
        def f(env):
            return env.step([actions.FunctionCall(_SELECT_RECT, [[_SELECT_NOT_ADD], lower_left, upper_right])])
        return f

    def _populate_actions_funcs(self):

        # select split screen observation
        x_lim, y_lim = self.last_timestep.observation['screen'][_SELECTED].shape

        base_add = int(x_lim / self.split_base)
        for i in range(0, x_lim - 1, base_add):
            for j in range(0, y_lim - 1, base_add):
                lower_left = [i, j]
                upper_right = [i + (base_add -1), j + (base_add -1)]

                f = self._select_func_factory(lower_left, upper_right)

                self.select_actions_list.append(f)

        # move one step around the mean
        for op_tpye in [_MOVE_SCREEN, _ATTACK_SCREEN]:
            for dir in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
                dir[0] *= 5
                dir[1] *= 5
                f = self._operation_func_factory(op_tpye, dir)
                self.move_attack_actions_list.append(f)
        print("Populated move/attack actions")

        # update num_action
        self.num_actions = 1 + len(self.select_actions_list) * len(self.move_attack_actions_list)

    def step(self, act_idx):
        assert((act_idx >= 0) and (act_idx < self.num_actions))
        curr_reward = 0
        if act_idx == 0:
            if not self.last:
                self.last_timestep = self.no_op_action(self._env)[0]
                curr_reward += self.get_reward()
                self.update()
        else:
            # decompose action to two steps
            idx = act_idx - 1
            select_idx = int(idx / len(self.move_attack_actions_list))
            move_attack_idx = idx % len(self.move_attack_actions_list)

            if not self.last and _SELECT_RECT in self.last_timestep.observation["available_actions"]:
                # take select action
                self.last_timestep = self.select_actions_list[select_idx](self._env)[0]
                curr_reward += self.get_reward()
                self.update()
                if not self.last and (_MOVE_SCREEN in self.last_timestep.observation["available_actions"] or _ATTACK_SCREEN in self.last_timestep.observation["available_actions"]):
                    # take move/attack action
                    self.last_timestep = self.move_attack_actions_list[move_attack_idx](self._env)[0]
                    curr_reward += self.get_reward()
                    self.update()

        feedback = namedtuple('feedback', ['features', 'reward'])
        return feedback(self.get_features(), curr_reward)


class DumbAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def step(self, features):
        # time.sleep(0.1)
        c = np.random.randint(0, self.num_actions)
        return c

class DumbContAgent:
    def __init__(self):
        pass

    def step(self, features):
        pos = MAP_SIZE * np.random.rand(6)
        attack_prob = np.random.rand()
        return pos, pos, attack_prob



def test_discrete():
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

def test_continous():
    flags.FLAGS(sys.argv)
    steps = 20000
    step_mul = 1
    with sc2_env.SC2Env(map_name="DefeatZerglingsAndBanelings",
                        step_mul=1,
                        visualize=True,
                        game_steps_per_episode=steps * step_mul) as env:
        simpleSC = SimpleScEnvCountinous(env)
        dumb_agent = DumbContAgent()
        simple_run_loop.simple_run_loop_continous(simpleSC, dumb_agent)

if __name__ == "__main__":
    # test_discrete()
    test_continous()
