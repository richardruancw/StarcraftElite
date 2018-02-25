"""A simple script agent to play the DefeatZerglingsAndBanelings map in the mini_maps"""

import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from utils import marine_actions


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECTED = features.SCREEN_FEATURES.selected.index

_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_SELECT_UNIT_ID = 1

_CONTROL_GROUP_SET = 1
_CONTROL_GROUP_RECALL = 0

_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

_NOT_QUEUED = [0]
_SELECT_ALL = [0]

class SimpleScriptAgent(base_agent.BaseAgent):
  """
  A dumb agent specifically for solving the DefeatZerglingsAndBanelings map.
  This agent will select all the marines and attack the zerglins in one group.
  """

  def step(self, obs):
    super(SimpleScriptAgent, self).step(obs)
    if _ATTACK_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      roach_y, roach_x = (player_relative == _PLAYER_HOSTILE).nonzero()
      if not roach_y.any():
        return actions.FunctionCall(_NO_OP, [])
      index = np.argmax(roach_y)
      target = [roach_x[index], roach_y[index]]
      return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
    elif _SELECT_ARMY in obs.observation["available_actions"]:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
    else:
        return actions.FunctionCall(_NO_OP, [])


class TacticScriptAgent(base_agent.BaseAgent):
  """
  A static agent specifically for solving the DefeatZerglingsAndBanelings map.
  It will spare the marines and then attack the zerglins to avoid the group damages
  from the banelings.
  """
  def step(self, obs):
    super(TacticScriptAgent, self).step(obs)
    # 1. Select marine!
    obs, screen, player = marine_actions.select_marine(self.env, [obs])

    player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]

    enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()

    # 2. Run away from nearby enemy
    closest, min_dist = None, None

    if (len(player) == 2):
      for p in zip(enemy_x, enemy_y):
        dist = np.linalg.norm(np.array(player) - np.array(p))
        if not min_dist or dist < min_dist:
          closest, min_dist = p, dist

    # 3. Sparse!
    friendly_y, friendly_x = (
            player_relative == _PLAYER_FRIENDLY).nonzero()

    closest_friend, min_dist_friend = None, None
    if (len(player) == 2):
      for p in zip(friendly_x, friendly_y):
        dist = np.linalg.norm(np.array(player) - np.array(p))
        if not min_dist_friend or dist < min_dist_friend:
          closest_friend, min_dist_friend = p, dist

    if (min_dist != None and min_dist <= 7):

      obs, new_action = marine_actions.act(self.env, obs, player, 2)

    elif (min_dist_friend != None and min_dist_friend <= 3):

      sparse_or_attack = np.random.randint(0, 2)

      obs, new_action = marine_actions.act(self.env, obs, player,
                                             sparse_or_attack)

    else:

      obs, new_action = marine_actions.act(self.env, obs, player, 1)

    return new_action[0]