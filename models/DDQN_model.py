import numpy as np
import tensorflow as tf

class DDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        # ------------------ build evaluate_net ------------------
        # print type(self.s)
        _input = tf.reshape(self.s, tf.pack([None, 17, 64, 64]))

        with tf.variable_scope('eval_net'):
            conv1 = tf.layers.conv2d(
                    inputs=_input,
                    filters=32,
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=64,
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[4, 4], strides=4)

            pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])

            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
            self.q_eval = tf.layers.dense(inputs=dropout, units=self.n_actions)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t_conv1 = tf.layers.conv2d(
                    inputs=_input,
                    filters=32,
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu)
            t_pool1 = tf.layers.max_pooling2d(inputs=t_conv1, pool_size=[2, 2], strides=2)
            t_conv2 = tf.layers.conv2d(
                    inputs=t_pool1,
                    filters=64,
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu)
            t_pool2 = tf.layers.max_pooling2d(inputs=t_conv2, pool_size=[4, 4], strides=4)

            t_pool2_flat = tf.reshape(t_pool2, [-1, 8 * 8 * 64])

            t_dense = tf.layers.dense(inputs=t_pool2_flat, units=1024, activation=tf.nn.relu)
            t_dropout = tf.layers.dropout(inputs=t_dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
            self.q_next = tf.layers.dense(inputs=t_dropout, units=self.n_actions)

        with tf.variable_scope('q_target'):
            self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_t')
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_target_wrt_a = tf.gather_nd(params=self.q_target, indices=a_indices)
        with tf.variable_scope('q_eval'):
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target_wrt_a, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, sess, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self, sess):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            sess.run(self.target_replace_op)

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        temp_q_eval_pri = sess.run(self.q_eval, {self.s: batch_memory[:, -self.n_features:]})
        temp_q_eval = sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})
        temp_q_next = sess.run(self.q_next, {self.s_: batch_memory[:, -self.n_features:]})

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        max_act4next = np.argmax(temp_q_eval_pri, axis=1)
        selected_q_next = temp_q_next[batch_index, max_act4next]
        temp_q_eval[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, cost = sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.s_: batch_memory[:, -self.n_features:],
                self.q_target: temp_q_eval
            })

        # print "Cost = "+str(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
