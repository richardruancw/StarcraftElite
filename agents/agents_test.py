import sys

from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.env import run_loop
from absl import flags

from agents.marine_script import SimpleScriptAgent, TacticScriptAgent

step_mul = 10
steps = 20000

def main():
    flags.FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name="DefeatZerglingsAndBanelings",
                        step_mul=1,
                        visualize=True,
                        game_steps_per_episode=steps * step_mul) as env:
        # agent = DefeatZerglingsSimple()
        agent = TacticScriptAgent()
        agent.env = env
        run_loop.run_loop([agent], env, steps)

if __name__ == '__main__':
    main()


