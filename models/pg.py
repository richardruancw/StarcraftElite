import os
import sys
import numpy as np
import tensorflow as tf
_path = os.path.dirname(os.path.abspath(__file__))
_path_utils = "/".join(_path.split('/')[:-1])+"/utils/"
_path_models = "/".join(_path.split('/')[:-1])+"/models/"
_path_net_prefix = "/".join(_path.split('/')[:-1])+"/policy_gradient_net/"
sys.path.insert(0, _path_utils)
sys.path.insert(0, _path_models)
from general import get_logger, Progbar, export_plot
from LinearSchedule import LinearSchedule


def build_mlp(mlp_input, output_size, scope):
	with tf.variable_scope(scope):
		out = mlp_input
		out = tf.contrib.layers.conv2d(inputs=out, num_outputs=32, kernel_size=8, stride=4)
		out = tf.contrib.layers.conv2d(inputs=out, num_outputs=64, kernel_size=4, stride=2)
		out = tf.contrib.layers.conv2d(inputs=out, num_outputs=64, kernel_size=3, stride=1)
		out = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(out, scope=scope), 512, activation_fn=tf.nn.relu)
		out = tf.contrib.layers.fully_connected(out, output_size, activation_fn=None)
	return out


class PG(object):
	def __init__(self, env, config, logger=None):
		# directory for training outputs
		if not os.path.exists(config.output_path):
			os.makedirs(config.output_path)

		# store hyper-params
		self.config = config
		self.logger = logger
		if logger is None:
			self.logger = get_logger(config.log_path)

		self.env = env
		temp = self.env.observation_dim
		self.observation_dim = [temp[1], temp[2], temp[0]*self.config.history_mul]
		self.move_action_dim = int((self.env.action_dim - 1) / 2)
		self.attack_action_dim = int((self.env.action_dim - 1) / 2)

		self.lr = self.config.learning_rate
		self.scheduler = LinearSchedule(self.config.rand_begin, \
										self.config.rand_end,\
										self.config.rand_steps)

		self._path_net = "/".join(_path.split('/')[:-1]) \
						 + "/policy_gradient_net_" \
						 + str(self.config.history_mul) + "/policy_gradient.ckpt"

		# build model
		self.build()


	def add_placeholders_op(self):
		self.observation_placeholder = tf.placeholder(tf.float32, [None] + self.observation_dim)
		self.action_placeholder = tf.squeeze(tf.placeholder(tf.float32, [None, self.move_action_dim]))
		self.attack_flag_placeholder = tf.placeholder(tf.float32, [None])
		self.advantage_placeholder = tf.placeholder(tf.float32, [None])
		self.global_step = tf.Variable(0)
  
  
	def build_policy_network_op(self, scope = "policy_network"):
		move_action_means = build_mlp(self.observation_placeholder, self.move_action_dim, "move_network")
		move_log_std = tf.get_variable(name="move_log_std", shape=[self.move_action_dim], dtype=tf.float32)
		attack_action_means = build_mlp(self.observation_placeholder, self.attack_action_dim, "attack_network")
		attack_log_std = tf.get_variable(name="attack_log_std", shape=[self.attack_action_dim], dtype=tf.float32)
		self.sampled_move_action =  tf.squeeze(move_action_means, axis=0)+ \
		   tf.random_normal([self.move_action_dim])*tf.exp(move_log_std)
		self.sampled_attack_action =  tf.squeeze(attack_action_means, axis=0)+ \
		   tf.random_normal([self.attack_action_dim])*tf.exp(attack_log_std)
		attack_logit = build_mlp(self.observation_placeholder, 1, "logit_network")
		self.attack_prob = 1.0 / (1.0 + tf.exp(tf.minimum(attack_logit, 10)))

		move_mvn = tf.contrib.distributions.MultivariateNormalDiag(
			loc=move_action_means, scale_diag=tf.exp(move_log_std))
		attack_mvn = tf.contrib.distributions.MultivariateNormalDiag(
			loc=attack_action_means,
			scale_diag=tf.exp(attack_log_std))
		self.move_log_prob = tf.log(tf.maximum(move_mvn.prob(self.action_placeholder) * (1-self.attack_prob), 1e-6))
		self.attack_log_prob = tf.log(tf.maximum(move_mvn.prob(self.action_placeholder) * self.attack_prob, 1e-6))
		self.logprob = (1-self.attack_flag_placeholder) * self.move_log_prob + \
					   self.attack_flag_placeholder * self.attack_log_prob

	def add_loss_op(self):
		self.loss = -tf.reduce_mean(self.logprob*self.advantage_placeholder)
  
  
	def add_optimizer_op(self):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
  
  
	def add_baseline_op(self, scope = "baseline"):
		self.baseline = tf.squeeze(build_mlp(self.observation_placeholder, 1, scope), axis=1)
		self.baseline_target_placeholder = tf.placeholder(tf.float32, [None])
		baseline_loss = tf.losses.mean_squared_error(
		  	labels=self.baseline_target_placeholder, predictions=self.baseline)
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.update_baseline_op = optimizer.minimize(baseline_loss)

	def build(self):
		# add placeholders
		self.add_placeholders_op()
		# create policy net
		self.build_policy_network_op()
		# add square loss
		self.add_loss_op()
		# add optmizer for the main networks
		self.add_optimizer_op()
  
		if self.config.use_baseline:
			self.add_baseline_op()
  
	def initialize(self):
		# create tf session
		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		# tensorboard stuff
		self.add_summary()
		# initiliaze all variables
		if(self.config.restore):
			self.saver.restore(self.sess, self._path_net)
			msg = "[Saver] restore model from {}".format(self._path_net)
			self.logger.info(msg)
		else:
			init = tf.global_variables_initializer()
			self.sess.run(init)
			msg = "[Saver] initialize new model, no restore"
			self.logger.info(msg)
  
  
	def add_summary(self):
		# extra placeholders to log stuff from python
		self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
		self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
		self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

		self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

		# extra summaries from python -> placeholders
		tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
		tf.summary.scalar("Max Reward", self.max_reward_placeholder)
		tf.summary.scalar("Std Reward", self.std_reward_placeholder)
		tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

		# logging
		self.merged = tf.summary.merge_all()
		self.file_writer = tf.summary.FileWriter(self.config.output_path, self.sess.graph)

	def init_averages(self):
		self.avg_reward = 0.
		self.max_reward = 0.
		self.std_reward = 0.
		self.eval_reward = 0.
  

	def update_averages(self, rewards):
		self.avg_reward = np.mean(rewards)
		self.max_reward = np.max(rewards)
		self.std_reward = np.sqrt(np.var(rewards) / len(rewards))


	def record_summary(self):
		fd = {
		  self.avg_reward_placeholder: self.avg_reward,
		  self.max_reward_placeholder: self.max_reward,
		  self.std_reward_placeholder: self.std_reward,
		  self.eval_reward_placeholder: self.eval_reward,
		}
		summary = self.sess.run(self.merged, feed_dict=fd)
		# tensorboard stuff
		self.file_writer.add_summary(summary, self.global_step.eval(self.sess))
  
  
	def sample_path(self, env, episodes = None):
		episode = 0
		episode_rewards = []
		paths = []
		if(not episodes):
			episodes = self.config.batch_size
  
		while (episode < episodes):
			state = env.reset().transpose([1, 2, 0])
			states, actions, rewards, flags = [], [], [], []
			episode_reward = 0
			li_state = [state] * self.config.history_mul
  
			for step in range(self.config.max_ep_len):
				stacked_state = np.concatenate(li_state, axis=2)
				states.append(stacked_state)
				move_action, attack_action, attack_prob = self.sess.run(
					[self.sampled_move_action, self.sampled_attack_action, self.attack_prob],
					feed_dict={self.observation_placeholder: np.expand_dims(stacked_state, axis=0)})
				single_state, reward, action, flag = env.step(move_action, attack_action, attack_prob)
				single_state = single_state.transpose([1, 2, 0])
				li_state.append(single_state)
				li_state = li_state[(len(li_state)-self.config.history_mul):]
				actions.append(action)
				rewards.append(reward)
				flags.append(flag)
				episode_reward += reward
				if env.last:
					break
			episode_rewards.append(episode_reward)
			path = {"observation": np.array(states),
						"reward": np.array(rewards),
						"action": np.array(actions),
						"flags": np.array(flags)}
			paths.append(path)
			episode += 1

		return paths, episode_rewards
  
  
	def get_returns_MC(self, paths):
		all_returns = []
		for path in paths:
			rewards = path["reward"]
			n = len(rewards)
			r = 0
			path_returns = []
			for t in xrange(n-1, -1, -1):
				r *= self.config.gamma
				r += rewards[t]
				path_returns.insert(0, r)
			all_returns.append(path_returns)
		returns = np.concatenate(all_returns)
		return returns
  
  
	def calculate_advantage_MC(self, returns, observations):
		adv = returns
		if self.config.use_baseline:
			baseline_val = self.sess.run(
				[self.baseline], feed_dict={self.observation_placeholder: observations})[0]
			adv = returns - baseline_val
		if self.config.normalize_advantage:
			adv -= np.mean(adv)
			adv /= np.std(adv)
		return adv

	def calculate_advantage_TD(self, paths):
		all_advs = []
		all_returns = []
		for path in paths:
			rewards = path["reward"]
			observations = path["observation"]
			baseline_val = self.sess.run(
				[self.baseline], feed_dict={self.observation_placeholder: observations})[0]
			returns = rewards[:-1] + self.config.gamma * baseline_val[1:]
			adv = returns
			if(self.config.use_baseline):
				adv -= baseline_val[:-1]
			all_advs.append(adv)
			all_returns.append(returns)
		adv = np.concatenate(all_advs)
		returns = np.concatenate(all_returns)
		if self.config.normalize_advantage:
			adv -= np.mean(adv)
			adv /= np.std(adv)
		return adv, returns


	def update_baseline(self, returns, observations):
		self.sess.run([self.update_baseline_op], feed_dict={
		  self.baseline_target_placeholder:returns,
		  self.observation_placeholder:observations
		  })
  
  
	def train(self):
		last_record = 0

		self.init_averages()

		for t in range(self.config.num_batches):
			if(t % self.config.save_freq == 0):
				save_path = self.saver.save(self.sess, self._path_net)
				msg = "[Saver] save model to {}".format(save_path)
				self.logger.info(msg)

			# update the random exploration prob
			self.scheduler.update(t)
			self.env.rand_explore_prob = self.scheduler.epsilon

			# evaluate before training
			self.eval_reward = self.evaluate(num_episodes=int(self.config.eval_batch_size / 5))

			# collect a minibatch of samples
			paths, total_rewards = self.sample_path(self.env)

			# process the data based on mode
			if (self.config.mode == "MC"):
				observations = np.concatenate([path["observation"] for path in paths])
				actions = np.concatenate([path["action"] for path in paths])
				rewards = np.concatenate([path["reward"] for path in paths])
				flags = np.concatenate([path["flags"] for path in paths])
				# compute Q-val estimates (discounted future returns) for each time step
				returns = self.get_returns_MC(paths)
				advantages = self.calculate_advantage_MC(returns, observations)
			elif(self.config.mode == "TD"):
				observations = np.concatenate([path["observation"][:-1] for path in paths])
				actions = np.concatenate([path["action"][:-1] for path in paths])
				flags = np.concatenate([path["flags"][:-1] for path in paths])
				advantages, returns = self.calculate_advantage_TD(paths)
			else:
				raise("invalid mode!!!! should be MC or TD")

			# run training operations
			if self.config.use_baseline:
				self.update_baseline(returns, observations)
			self.sess.run(self.train_op, feed_dict={
						self.observation_placeholder: observations,
						self.action_placeholder: actions,
						self.attack_flag_placeholder: flags,
						self.advantage_placeholder: advantages})
  
			# tf stuff
			if (t % self.config.summary_freq == 0):
				self.update_averages(total_rewards)
				self.record_summary()

			# compute reward statistics for this batch and log
			avg_reward = np.mean(total_rewards)
			sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
			msg = "[Training] Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
			self.logger.info(msg)
  
			self.logger.info("- Training done.")


	def evaluate(self, env=None, num_episodes=1):
		if env==None:
			env = self.env
		env.set_greedy_mode()
		paths, rewards = self.sample_path(env, num_episodes)
		avg_reward = np.mean(rewards)
		sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
		msg = "[Evaluation] Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
		self.logger.info(msg)
		env.close_greedy_mode()
		return avg_reward


	def run(self):
		# initialize
		self.initialize()
		# model
		self.train()

