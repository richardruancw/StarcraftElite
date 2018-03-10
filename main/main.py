import os
import pickle
import sys
_path = os.path.dirname(os.path.abspath(__file__))
_path_utils = "/".join(_path.split('/')[:-1])+"/utils/"
_path_models = "/".join(_path.split('/')[:-1])+"/models/"
_path_net = "/".join(_path.split('/')[:-1])+"/my_net/save_net.ckpt"
_path_log = "/".join(_path.split('/')[:-1])+"log.txt"
sys.path.insert(0, _path_utils)
sys.path.insert(0, _path_models)
from DDQN_model import DDQN
from simple_env_2 import SimpleScEnvDiscrete

from collections import namedtuple

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env

from absl import flags
import numpy as np
import tensorflow as tf
import time

print "Successfully import models & utils"

def run_loop(env, agent, max_episodes = 300, max_steps = 20000):
    start_time = time.time()
    step = 0
    saver = tf.train.Saver()
    with open(_path_log, 'w+') as f:
        f.write("Rewards\n")
    try:
        with tf.Session() as sess:
            # saver.restore(sess, _path_net)
            sess.run(tf.global_variables_initializer())

            for episode in xrange(max_episodes):
                save_path = saver.save(sess, _path_net)
                print("Save to path: ", save_path)

                observation = env.reset().reshape([-1])
                total_reward = 0.0

                while True:
                    if step>max_steps:
                        break
                    action = agent.choose_action(sess, observation)
                    observation_, reward = env.step(action)
                    total_reward+=reward
                    observation_ = observation_.reshape([-1])
                    agent.store_transition(observation, action, reward, observation_)
                    if (step > 1000) and (step % 50 == 0):
                        print "Episode: "+str(episode)+"  Step: "+str(step)+"  Reward: "+str(total_reward)
                        agent.learn(sess)
                    observation = observation_
                    if env.last:
                        break
                    step += 1
                    # print "Episode: "+str(episode)+"  Step: "+str(step)
                with open("log.txt", 'a+') as f:
                    f.write("Episode: "+str(episode)+"  Reward: "+str(total_reward)+"\n")
            print('game over')
            save_path = saver.save(sess, _path_net)
            print("Save to path: ", save_path)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, step, step / elapsed_time))

flags.FLAGS(sys.argv)
steps = 2000
step_mul = 20
with sc2_env.SC2Env(map_name="MoveToBeacon",
                    step_mul=1,
                    visualize=True,
                    game_steps_per_episode=steps * step_mul) as env:
    simpleSC = SimpleScEnvDiscrete(env, split_base = 1)

    DDQN_agent = DDQN(simpleSC.num_actions, 17*64*64,
                      learning_rate=0.01,
                      reward_decay=1.0,
                      e_greedy=0.8,
                      replace_target_iter=200,
                      memory_size=20000,
                      e_greedy_increment = 0.0001
                      # output_graph=True
                      )
    run_loop(simpleSC, DDQN_agent, max_episodes = 30000, max_steps = 20000000)
