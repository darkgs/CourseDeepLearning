
import tensorflow as tf
import cv2 
import gym
from gym import spaces

import numpy as np
import os
import argparse
import random

class Environment(object):
	def __init__(self, args):
		# Make an environment instance of CartPole-v0.
		self._env = gym.make(args.env_name)
		self._initialized = False

		# Uncomment to show the screenshot of the environment (rendering game screens)
		# env.render() 

	def random_action(self):
		# Return a random action.
		return self._env.action_space.sample()

	def get_action_space_count(self):
		return self._env.action_space.n

	def get_state_shape(self):
		return self._env.observation_space.shape

	def render_worker(self):
		# If display in your option is true, do rendering. Otherwise, do not.
		pass

	def new_episode(self):
		# Sart a new episode and return the first state of the new episode.
		self._initialized = True
		state = self._env.reset()

		return state

	def act(self, action):
		assert(self._initialized)
		assert(self._env.action_space.contains(action))
		# Perform an action which is given by input argument and return the results of acting.
		# "step" function performs agent's actions given current state of the environment and returns several values.
		# Input: action (numerical data)
		#        - env.action_space.sample(): select a random action among possible actions.
		# Output: next_state (numerical data, next state of the environment after performing given action)
		#         reward (numerical data, reward of given action given current state)
		#         terminal (boolean data, True means the agent is done in the environment)
		next_state, reward, terminal, info = self._env.step(action)
		return next_state, reward, terminal


class ReplayMemory(object):
	def __init__(self, args):
		self._args = args
		self._memory = []
		
	def add(self, s, a, r, t, s_):
		# Add current_state, action, reward, terminal, (next_state which can be added by your choice). 
		self._memory.append([s, a, r, t, s_])
		if len(self._memory) > self._args.replay_memory_size:
			self._memory = self._memory[-self._args.replay_memory_size:]

	def mini_batch(self):
		# Return a mini_batch whose data are selected according to your sampling method. (such as uniform-random sampling in DQN papers)
		assert(len(self._memory) >= self._args.re_valid_count)

		datas = []

		while(len(datas) < self._args.re_batch_size):
			datas.append(self._memory[random.randrange(len(self._memory))])

		return np.array(datas)

	def count(self):
		return len(self._memory)


class DQN(object):
	def __init__(self, args, sess, env):
		self.sess = sess
		self._env = env
		self._args = args

		self.prediction_Q = self.build_network('pred')
		self.target_Q = self.build_network('target')
		self.optimizer = self.build_optimizer()

		self._copy_op = []
		pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
		target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
		
		for pred_var, target_var in zip(pred_vars, target_vars):
			self._copy_op.append(target_var.assign(pred_var.value()))

	def build_network(self, name):
		# Make your a deep neural network
		with tf.variable_scope(name):
			state_dim = self._env.get_state_shape()[0]
			action_dim = 1
			drop_out_rate = 0.7

			t_input_s = tf.placeholder(tf.float32, shape=[None, state_dim])
			t_input_a = tf.placeholder(tf.float32, shape=[None, action_dim])
			t_is_train = tf.placeholder_with_default(True, shape=())

			flow = tf.concat([t_input_s, t_input_a], axis=1)
			flow = tf.layers.dense(flow, 16, activation=tf.nn.relu)
			flow = tf.layers.dense(flow, 48, activation=tf.nn.relu)
			flow = tf.layers.batch_normalization(flow)
			flow = tf.layers.dense(flow, 64, activation=tf.nn.relu)
			flow = tf.layers.dense(flow, 72, activation=tf.nn.relu)
			flow = tf.layers.batch_normalization(flow)
			flow = tf.layers.dense(flow, 32, activation=tf.nn.relu)
			flow = tf.layers.dense(flow, 8, activation=tf.nn.relu)
			flow = tf.layers.batch_normalization(flow)
			flow = tf.layers.dropout(flow, rate=drop_out_rate, training=t_is_train)
			flow = flow + flow * (1.0 / (1.0-drop_out_rate) - 1.0) * tf.to_float(t_is_train)
			t_reward = tf.layers.dense(flow, 1)

		return (t_input_s, t_input_a, t_is_train, t_reward)

	def build_optimizer(self):
		# Make your optimizer 
		action_count = self._env.get_action_space_count()
		gamma = self._args.gamma
		lr = self._args.learning_rate

		t_input_r = tf.placeholder(tf.float32, shape=[None, 1])
		t_input_tq = tf.placeholder(tf.float32, shape=[None, 1])
		_, _, _, t_input_q = self.prediction_Q
		t_input_t = tf.placeholder(tf.float32, shape=[None, 1])

		t_loss = tf.reduce_mean(tf.square(t_input_r + gamma * tf.reduce_max(t_input_tq, axis=1) * t_input_t - t_input_q), axis=1)
		t_optimizer = tf.train.AdamOptimizer(lr).minimize(t_loss) 

		return (t_input_r, t_input_tq, t_input_t, t_loss, t_optimizer)

	def train_network(self, samples):
		# Train the prediction_Q network using a mini-batch sampled from the replay memory
		#self._memory.append([s, a, r, t, s_])
		t_s = np.stack(samples[:,0])
		t_a = np.expand_dims(np.array(samples[:,1]), axis=1)
		t_r = np.expand_dims(np.array(samples[:,2]), axis=1)
		t_t = np.expand_dims(np.array(samples[:,3]), axis=1)
		t_t = 1.0 - np.ones(t_t.shape, dtype=np.float32) * t_t
		t_s_ = np.stack(samples[:,4])

		_, t_max_target_rewards = self.predict_Q(t_s_, self.target_Q)
		t_tq = np.expand_dims(t_max_target_rewards.eval(), axis=1)

		t_input_s, t_input_a, t_is_train, t_reward = self.prediction_Q
		t_input_op_r, t_input_op_tq, t_input_op_t, t_loss, t_optimizer = self.optimizer

		feed_dict = {
			t_input_s: t_s,
			t_input_a: t_a,
			t_input_op_r: t_r,
			t_input_op_tq: t_tq,
			t_input_op_t: t_t,
		}

		loss, _ = self.sess.run([t_loss, t_optimizer], feed_dict=feed_dict)

		return np.mean(loss)

	def update_target_network(self):
		self.sess.run(self._copy_op)
		
	def predict_Q(self, t_s, Q_net):
		t_input_s, t_input_a, t_is_train, t_reward = Q_net

		batch_size = t_s.shape[0]

		predicts = []
		for a in range(self._env.get_action_space_count()):
			t_a = np.expand_dims(np.ones([batch_size], dtype=np.float32) * a, axis=1)
			feed_dict = {
				t_input_s: t_s,
				t_input_a: t_a,
				t_is_train: False,
			}
			reward, = self.sess.run([t_reward], feed_dict=feed_dict)
			predicts.append(tf.squeeze(reward, axis=1))

		predicts = tf.stack(predicts, axis=1)

		return tf.argmax(predicts, axis=1), tf.reduce_max(predicts, axis=1)

	def predict_action(self, s):
		t_s = np.array([s])
		t_a, t_r = self.predict_Q(t_s, self.prediction_Q)
		return t_a.eval()[0]


class Agent(object):
	def __init__(self, args, sess, memory, dqn, env):
		self.saver = tf.train.Saver()

		self._args = args
		self._sess = sess
		self._memory = memory
		self._dqn = dqn
		self._env = env

		self._train_count = 0

	def get_epsilon(self):
		return max(1.0 - self._train_count / 1000000.0, 0.1)

	def select_action(self, state):
		# Select an action according ε-greedy. You need to use a random-number generating function and add a library if necessary.

		epsilon = self.get_epsilon()

		if random.random() < epsilon:
			return self._env.random_action()

		return self._dqn.predict_action(state)

	def train(self):
		# Train your agent which has the neural nets.
		# Several hyper-parameters are determined by your choice (Options class in the below cell)
		# Keep epsilon-greedy action selection in your mind

		s = self._env.new_episode()

		losses = []
		steps = []
		for step in range(self._args.max_step_per_episode):
			a = self.select_action(s)
			s_, r, t = self._env.act(a)

			# Backup
			self._memory.add(s, a, r, t, s_)

			# Train DQN
			if self._memory.count() >= self._args.re_valid_count:
				loss = self._dqn.train_network(self._memory.mini_batch())
				losses.append(loss)
				self._train_count += self._args.re_batch_size

			# Update target Q-Network
			if (step+1) % self._args.target_q_update_step == 0:
				self._dqn.update_target_network()

			# The end of this episode
			if t:
				steps.append(step)
				break

			# next state
			s = s_

		return np.mean(losses), np.mean(steps)

	def play(self):
		# Test your agent 
		# When performing test, you can show the environment's screen by rendering,
		total_reward = 0.0
		s = self._env.new_episode()
		t = False
		for _ in range(self._args.max_step_per_episode):
			if t:	break
			a = self._dqn.predict_action(s)
			s, r, t = self._env.act(a)
			total_reward += r

		return total_reward

	def save(self):
		checkpoint_dir = 'cartpole'
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)
		self.saver.save(self._sess, os.path.join(checkpoint_dir, 'trained_agent'))

	def load(self):
		checkpoint_dir = 'cartpole'
		self.saver.restore(self._sess, os.path.join(checkpoint_dir, 'trained_agent'))


def main():
	scale = 5

	parser = argparse.ArgumentParser(description="CartPole")
	parser.add_argument('--env-name', default='CartPole-v0', type=str, help="Environment")
	parser.add_argument('--re-batch-size', default=64, type=int, help="ReplayMemory minibatch size")
	parser.add_argument('--re-valid-count', default=8, type=int, help="Minimum memory size to generate minibatch")
	parser.add_argument('--replay-memory-size', default=10000*scale, type=int, help="Capacity of the replaymemeory")
	parser.add_argument('--max-step-per-episode', default=200*scale, type=int, help="Maximum steps for an episode")
	parser.add_argument('--target-q-update-step', default=1*scale, type=int, help="Update period for target Q-net")
	parser.add_argument('--gamma', default=0.9, type=float, help="Decay factor of rewards for the future")
	parser.add_argument('--learning-rate', default=1e-3, type=float, help="Learning rate of optimizer")

	args = parser.parse_args()

	# tf config
	config = tf.ConfigProto()
	# If you use a GPU, uncomment
	config.log_device_placement = False
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
		env = Environment(args)
		memory = ReplayMemory(args)
		dqn = DQN(args, sess, env)
		myAgent = Agent(args, sess, memory, dqn, env)
		tf.global_variables_initializer().run()
#		myAgent.load()
		steps = []
		for epi in range(1000):
			loss, step = myAgent.train()
			steps.append(step)

			if epi % 10 == 0:
				play_rewards = []
				for _ in range(5):
					play_reward = myAgent.play()
					play_rewards.append(play_reward)
				myAgent.save()
				print('epi {} - model saved! loss({:.6f}) train_steps({:.2f}) epsiolon({:.4f} and play reward is {:.2f}'.format(epi, loss, np.mean(steps), myAgent.get_epsilon(), np.mean(play_rewards)))
				steps = []


if __name__ == '__main__':
	main()
