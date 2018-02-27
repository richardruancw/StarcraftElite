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
import tensorflow as tf
import time

print "Successfully import models & utils"

def run_loop(env, agent, max_episodes = 30000, max_steps = 2000000):
    start_time = time.time()
    step = 0
    saver = tf.train.Saver()
    try:
        with tf.Session() as sess:
            # saver.restore(sess, "my_net/save_net.ckpt")
            sess.run(tf.global_variables_initializer())

            for episode in xrange(max_episodes):
                save_path = saver.save(sess, "my_net/save_net.ckpt")
                print("Save to path: ", save_path)

                observation = env.reset().reshape([-1])

                while True:
                    if step>max_steps:
                        break
                    action = agent.choose_action(sess, observation)
                    observation_, reward = env.step(action)
                    observation_ = observation_.reshape([-1])
                    agent.store_transition(observation, action, reward, observation_)
                    if (step > 200) and (step % 20 == 0):
                        agent.learn(sess)
                    observation = observation_
                    if env.last:
                        break
                    step += 1
                    print "Episode: "+str(episode)+"  Step: "+str(step)
            print('game over')
            save_path = saver.save(sess, "my_net/save_net.ckpt")
            print("Save to path: ", save_path)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, step, step / elapsed_time))

flags.FLAGS(sys.argv)
steps = 20000
step_mul = 1
with sc2_env.SC2Env(map_name="DefeatZerglingsAndBanelings",
                    step_mul=1,
                    visualize=True,
                    game_steps_per_episode=steps * step_mul) as env:
    simpleSC = SimpleScEnvDiscrete(env)
    DDQN_agent = DDQN(simpleSC.num_actions, 17*64*64,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=20000,
                      # output_graph=True
                      )
    run_loop(simpleSC, DDQN_agent, max_episodes = 300)
