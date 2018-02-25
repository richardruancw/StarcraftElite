# Understand Features From Observations 

The ``agent`` class has access to ``obs``, which provides information to support making decision 
(return the ``FunctionCall`` from ``actions`` lib).

``obs.observation`` is the dictionary for those information.

## How to play with it?

* ``obs.observation['screen']``

The "screen" gives info stored in ``SCREEN_FEATURES`` defined as below.


```python

SCREEN_FEATURES = ScreenFeatures(
    height_map=(256, FeatureType.SCALAR, colors.winter, False),
    visibility_map=(4, FeatureType.CATEGORICAL,
                    colors.VISIBILITY_PALETTE, False),
    creep=(2, FeatureType.CATEGORICAL, colors.CREEP_PALETTE, False),
    power=(2, FeatureType.CATEGORICAL, colors.POWER_PALETTE, False),
    player_id=(17, FeatureType.CATEGORICAL,
               colors.PLAYER_ABSOLUTE_PALETTE, False),
    player_relative=(5, FeatureType.CATEGORICAL,
                     colors.PLAYER_RELATIVE_PALETTE, False),
    unit_type=(1850, FeatureType.CATEGORICAL, colors.unit_type, False),
    selected=(2, FeatureType.CATEGORICAL, colors.SELECTED_PALETTE, False),
    unit_hit_points=(1600, FeatureType.SCALAR, colors.hot, True),
    unit_hit_points_ratio=(256, FeatureType.SCALAR, colors.hot, False),
    unit_energy=(1000, FeatureType.SCALAR, colors.hot, True),
    unit_energy_ratio=(256, FeatureType.SCALAR, colors.hot, False),
    unit_shields=(1000, FeatureType.SCALAR, colors.hot, True),
    unit_shields_ratio=(256, FeatureType.SCALAR, colors.hot, False),
    unit_density=(16, FeatureType.SCALAR, colors.hot, True),
    unit_density_aa=(256, FeatureType.SCALAR, colors.hot, False),
    effects=(16, FeatureType.CATEGORICAL, colors.effects, False),
)

```

Example

```python
from pysc2.lib import features

_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
```

``player_relative`` is a 2-d numpy array describes the locations of each
types of units in one screen. 

Other data can be found in similar way. 

* ``obs.observation['available_actions']``

A nd array of integer represents available actions. Used before applying some actions which
may not be feasible at all time.

```python
from pysc2.lib import actions

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

_NOT_QUEUED = [0]

if _MOVE_SCREEN in obs.observation["available_actions"]:
    target = [int(1), int(2)]
    return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target]) 
```

