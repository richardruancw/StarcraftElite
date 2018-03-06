import os
import sys
import numpy as np
import tensorflow as tf
_path = os.path.dirname(os.path.abspath(__file__))
_path_utils = "/".join(_path.split('/')[:-1])+"/utils/"
_path_models = "/".join(_path.split('/')[:-1])+"/models/"
sys.path.insert(0, _path_utils)
sys.path.insert(0, _path_models)
from general import get_logger, Progbar, export_plot


def build_mlp(mlp_input, output_size, scope):
	with tf.variable_scope(scope) as scope:
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
		self.observation_dim = self.config.observation_dim
		self.action_dim = self.config.action_dim

		self.lr = self.config.learning_rate

		# build model
		self.build()
  
  
	def add_placeholders_op(self):
		self.observation_placeholder = tf.placeholder(tf.float32, [None] + self.observation_dim)
		self.action_placeholder = tf.squeeze(tf.placeholder(tf.float32, [None, self.action_dim]))
		self.advantage_placeholder = tf.placeholder(tf.float32, [None])
  
  
	def build_policy_network_op(self, scope = "policy_network"):
		action_means = build_mlp(self.observation_placeholder, self.action_dim, scope)
		log_std = tf.get_variable(name="log_std", shape=[self.action_dim], dtype=tf.float32)
		self.sampled_action =  tf.squeeze(action_means, axis=0)+ \
		   tf.random_normal([self.action_dim])*tf.exp(log_std)
		mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=action_means, scale_diag=tf.exp(log_std))
		self.logprob = tf.log(tf.maximum(mvn.prob(self.action_placeholder), 1e-6))

	def add_loss_op(self):
		self.loss = -tf.reduce_mean(self.logprob*self.advantage_placeholder)
  
  
	def add_optimizer_op(self):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.train_op = optimizer.minimize(self.loss)
  
  
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
		# tensorboard stuff
		self.add_summary()
		# initiliaze all variables
		init = tf.global_variables_initializer()
		self.sess.run(init)
  
  
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
  

	def update_averages(self, rewards, scores_eval):
		self.avg_reward = np.mean(rewards)
		self.max_reward = np.max(rewards)
		self.std_reward = np.sqrt(np.var(rewards) / len(rewards))
  
		if len(scores_eval) > 0:
			self.eval_reward = scores_eval[-1]
  
  
	def record_summary(self, t):
		fd = {
		  self.avg_reward_placeholder: self.avg_reward,
		  self.max_reward_placeholder: self.max_reward,
		  self.std_reward_placeholder: self.std_reward,
		  self.eval_reward_placeholder: self.eval_reward,
		}
		summary = self.sess.run(self.merged, feed_dict=fd)
		# tensorboard stuff
		self.file_writer.add_summary(summary, t)
  
  
	def sample_path(self, env, episodes = None):
		episode = 0
		episode_rewards = []
		paths = []
		if(not episodes):
			episodes = self.config.batch_size
  
		while (episode < episodes):
			state = env.reset()
			states, actions, rewards = [], [], []
			episode_reward = 0
  
			for step in range(self.config.max_ep_len):
				states.append(state)
				action = self.sess.run([self.sampled_action], feed_dict={self.observation_placeholder: state})[0]
				state, reward, done, action = env.step(action)
				actions.append(action)
				rewards.append(reward)
				episode_reward += reward
				if done:
					break
			episode_rewards.append(episode_reward)
			path = {"observation": np.array(states),
						  "reward": np.array(rewards),
						  "action": np.array(actions)}
			paths.append(path)
			episode += 1

		return paths, episode_rewards
  
  
	def get_returns(self, paths):
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
  
  
	def calculate_advantage(self, returns, observations):
		adv = returns
		if self.config.use_baseline:
			baseline_val = self.sess.run([self.baseline], feed_dict={self.observation_placeholder: observations})[0]
			adv = returns - baseline_val
		if self.config.normalize_advantage:
			adv -= np.mean(adv)
			adv /= np.std(adv)
		return adv
  
  
	def update_baseline(self, returns, observations):
		self.sess.run([self.update_baseline_op], feed_dict={
		  self.baseline_target_placeholder:returns,
		  self.observation_placeholder:observations
		  })
  
  
	def train(self):
		last_record = 0

		self.init_averages()
		scores_eval = [] # list of scores computed at iteration time

		for t in range(self.config.num_batches):
			# collect a minibatch of samples
			paths, total_rewards = self.sample_path(self.env)
			scores_eval = scores_eval + total_rewards
			observations = np.concatenate([path["observation"] for path in paths])
			actions = np.concatenate([path["action"] for path in paths])
			rewards = np.concatenate([path["reward"] for path in paths])
			# compute Q-val estimates (discounted future returns) for each time step
			returns = self.get_returns(paths)
			advantages = self.calculate_advantage(returns, observations)

			# run training operations
			if self.config.use_baseline:
				self.update_baseline(returns, observations)
			self.sess.run(self.train_op, feed_dict={
						self.observation_placeholder: observations,
						self.action_placeholder: actions,
						self.advantage_placeholder: advantages})
  
			# tf stuff
			if (t % self.config.summary_freq == 0):
				self.update_averages(total_rewards, scores_eval)
				self.record_summary(t)

			# compute reward statistics for this batch and log
			avg_reward = np.mean(total_rewards)
			sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
			msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
			self.logger.info(msg)
  
			if self.config.record and (last_record > self.config.record_freq):
				self.logger.info("Recording...")
				last_record =0
				self.record()
  
			self.logger.info("- Training done.")
			export_plot(scores_eval, "Score", self.config.plot_output)


	def evaluate(self, env=None, num_episodes=1):
		if env==None:
			env = self.env
		paths, rewards = self.sample_path(env, num_episodes)
		avg_reward = np.mean(rewards)
		sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
		msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
		self.logger.info(msg)
		return avg_reward


	def run(self):
		# initialize
		self.initialize()
		# record one game at the beginning
		if self.config.record:
			self.evaluate()
		# model
		self.train()
		# record one game at the end
		if self.config.evaluate:
			self.evaluate()

