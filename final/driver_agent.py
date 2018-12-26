# coding: utf-8
import gym
import tensorflow as tf
import numpy as np
import math

import argparse
import random

import os

# If you want, add hyperparameters

class Actor(object):
    def __init__(self, args, sess, state_dim):
        self._tau = args.actor_tau
        self._lr = args.actor_lr
        self._action_dim = 3

        self._sess = sess
        self._state_dim = state_dim

        self._pred_net = self.build_network('actor_pred')
        self._target_net = self.build_network('actor_target')

        self._pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_pred')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_target')

        # generate target net initialization ops
        self._init_target_ops = []
        for pred_var, target_var in zip(self._pred_vars, target_vars):
            self._init_target_ops.append(
                    tf.assign(target_var, pred_var.value(), validate_shape=True))
        
        # generate target net update ops
        self._update_target_ops = []

        for pred_var, target_var in zip(self._pred_vars, target_vars):
            self._update_target_ops.append(
                    tf.assign(target_var, self._tau * pred_var.value() + \
                        (1.0 - self._tau) * target_var.value(), validate_shape=True))

        # optimizer
        self._optimizer = self.build_optimizer()

    def build_optimizer(self):
        t_input_state, t_pred_action = self._pred_net
        with tf.variable_scope('optimizer'):
            t_action_gradient = tf.placeholder(tf.float64,[None, self._action_dim])
            params_grad = tf.gradients(t_pred_action, self._pred_vars, -t_action_gradient)

            grads = zip(params_grad, self._pred_vars)
            optimizer = tf.train.AdamOptimizer(self._lr).apply_gradients(grads)

        return (t_input_state, t_action_gradient, optimizer)

    def train(self, states, action_grad):
        t_input_state, t_action_gradient, optimizer = self._optimizer

        feed_dict = {
            t_input_state: states,
            t_action_gradient: action_grad,
        }

        self._sess.run([optimizer], feed_dict=feed_dict)

    def build_network(self, name):
        with tf.variable_scope(name):
            t_states = tf.placeholder(tf.float64, shape=[None, self._state_dim])

            flow = t_states
            flow = tf.layers.dense(flow, 300, activation=tf.nn.relu)
            flow = tf.layers.dense(flow, 400, activation=tf.nn.relu)

            steer = tf.layers.dense(flow, 1, activation=tf.nn.tanh, \
                    kernel_initializer=tf.random_uniform_initializer(-1e-4,1e-4))
            acc = tf.layers.dense(flow, 1, activation=tf.nn.sigmoid, \
                    kernel_initializer=tf.random_uniform_initializer(-1e-4,1e-4))
            brake = tf.layers.dense(flow, 1, activation=tf.nn.sigmoid, \
                    kernel_initializer=tf.random_uniform_initializer(-1e-4,1e-4))

            t_action= tf.concat([steer, acc, brake], axis=1)

        return (t_states, t_action)

    def init_target_network(self):
        self._sess.run(self._init_target_ops)

    def update_target_network(self):
        self._sess.run(self._update_target_ops)

    def action(self, states):
        t_states, t_action = self._pred_net

        feed_dict = {
            t_states: states,
        }

        pred_action, = self._sess.run([t_action], feed_dict=feed_dict)

        return pred_action

    def target_action(self, states):
        t_states, t_action = self._target_net

        feed_dict = {
            t_states: states,
        }

        pred_action, = self._sess.run([t_action], feed_dict=feed_dict)

        return pred_action


class Critic(object):
    def __init__(self, args, sess, state_dim, action_dim):
        self._tau = args.critic_tau
        self._lr = args.critic_lr

        self._sess = sess
        self._state_dim = state_dim
        self._action_dim = action_dim

        self._pred_net = self.build_network('critic_pred')
        self._target_net = self.build_network('critic_target')

        self._pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_pred')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_target')

        # generate target net initialization ops
        self._init_target_ops = []
        for pred_var, target_var in zip(self._pred_vars, target_vars):
            self._init_target_ops.append(
                    tf.assign(target_var, pred_var.value(), validate_shape=True))
        
        # generate target net update ops
        self._update_target_ops = []
        for pred_var, target_var in zip(self._pred_vars, target_vars):
            self._update_target_ops.append(
                    tf.assign(target_var, self._tau * pred_var.value() + \
                        (1.0 - self._tau) * target_var.value(), validate_shape=True))

        # optimizer
        self._optimizer = self.build_optimizer()

    def build_optimizer(self):
        _, _, t_pred_action, _ = self._pred_net
        with tf.variable_scope('optimizer'):
            t_input_target_q = tf.placeholder(tf.float64, [None, self._action_dim])

            t_acc = tf.metrics.accuracy(labels=t_input_target_q, predictions=t_pred_action)
            t_loss = tf.reduce_mean(tf.squared_difference(t_input_target_q, t_pred_action))
            t_optimizer = tf.train.AdamOptimizer(self._lr).minimize(t_loss)

        return (t_input_target_q, t_loss, t_optimizer)

    def build_network(self, name):
        with tf.variable_scope(name):
            t_states = tf.placeholder(tf.float64, shape=[None, self._state_dim])
            t_action = tf.placeholder(tf.float64, shape=[None, self._action_dim])

            flow_s = t_states
            flow_s = tf.layers.dense(flow_s, 300, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            flow_s = tf.layers.dense(flow_s, 600, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

            flow_a = t_action
            flow_a = tf.layers.dense(flow_a, 600)

            flow = tf.concat([flow_s, flow_a], axis=1)
            flow = tf.layers.dense(flow, 600, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
            flow = tf.layers.dense(flow, self._action_dim, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

            t_output = flow
            t_action_grad = tf.gradients(t_output, t_action)

        return (t_states, t_action, t_output, t_action_grad)

    def init_target_network(self):
        self._sess.run(self._init_target_ops)

    def update_target_network(self):
        self._sess.run(self._update_target_ops)

    def predict(self, states, actions):
        t_states, t_action, t_output, _ = self._pred_net

        feed_dict = {
            t_states: states,
            t_action: actions,
        }

        predicts, = self._sess.run([t_output], feed_dict=feed_dict)

        return predicts

    def gradient(self, states, actions):
        t_states, t_action, _, t_action_grad = self._pred_net
        feed_dict = {
            t_states: states,
            t_action: actions,
        }

        grad, = self._sess.run([t_action_grad], feed_dict=feed_dict)
        return grad

    def target_predict(self, states, actions):
        t_states, t_action, t_output, _ = self._target_net

        feed_dict = {
            t_states: states,
            t_action: actions,
        }

        predicts, = self._sess.run([t_output], feed_dict=feed_dict)

        return predicts

    def train(self, states, actions, target_q):
        t_states, t_action, _, _ = self._pred_net
        t_input_target_q, t_loss, t_optimizer = self._optimizer

        feed_dict = {
            t_states: states,
            t_action: actions,
            t_input_target_q: target_q,
        }

        loss, _ = self._sess.run([t_loss, t_optimizer], feed_dict=feed_dict)

        return np.mean(loss)


class ReplayMemory(object):
    def __init__(self, args):
        self._args = args
        self._memory = []

    def add(self, s, a, r, s_, t):
        self._memory.append([s,a,r,s_,t])

        if len(self._memory) > self._args.replay_memory_size:
            self._memory = self._memory[-self._args.replay_memory_size:]

    def mini_batch(self):
        assert self.count() > 0
        datas = []

        while len(datas) < self._args.batch_size:
            item = self._memory[random.randrange(len(self._memory))]
            datas.append(item)

        return np.array(datas)

    def count(self):
        return len(self._memory)


class DriverAgent:
    def __init__(self, env_name, state_dim, action_dim):
        # Parse arguments
        parser = argparse.ArgumentParser(description="Torcs Projects")
        parser.add_argument('--actor-tau', default=0.001, type=float, help="Actor Tau")
        parser.add_argument('--actor-lr', default=0.0001, type=float, help="Actor Learning rate")

        parser.add_argument('--critic-tau', default=0.001, type=float, help="Critic Tau")
        parser.add_argument('--critic-lr', default=0.001, type=float, help="Critic Learning rate")

        parser.add_argument('--gamma', default=0.10, type=float, help="How long to see the future")
        
        parser.add_argument('--replay-memory-size', default=1000000, type=int, help="ReMem size")
        parser.add_argument('--batch-size', default=32, type=int, help="Batch Size of training")

        args = parser.parse_args()

        #
        self.name = 'DriverAgent' # name for uploading results
        self.env_name = env_name

        #
        self._args = args

        # Randomly initialize actor network and critic network
        # with both their target networks
        self._state_dim = state_dim
        self._action_dim = action_dim

        assert self._action_dim == 3
    
        # open new session of Tensorflow
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.log_device_placement = False
        config.gpu_options.allow_growth = True

        self._sess = tf.Session(config=config)
        
        # actor & critic
        self._actor = Actor(args=args, sess=self._sess, state_dim=self._state_dim)
        self._critic = Critic(args=args, sess=self._sess, state_dim=state_dim, \
                action_dim=self._action_dim)

        self._sess.run(tf.global_variables_initializer())

        self._r_mem = ReplayMemory(args)

        # loading networks. modify as you want 
        self._saver = tf.train.Saver()
        self._checkpoint_dir = 'torcs_model'

        if os.path.exists(self._checkpoint_dir):
            self.load()
        else:
            self._actor.init_target_network()
            self._critic.init_target_network()

#        checkpoint = tf.train.get_checkpoint_state("path_to_save/")
#        if checkpoint and checkpoint.model_checkpoint_path:
#            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
#            print("Successfully loaded:", checkpoint.model_checkpoint_path)
#        else:
#            print("Could not find old network weights")

    def __del__(self):
        self._sess.close()

    def train(self, state, action, reward, next_state, done):
        self._r_mem.add(state, action, reward, next_state, done)

        if self._r_mem.count() < 16:
            return

        #
        samples = self._r_mem.mini_batch()
        b_state = np.stack(samples[:,0])
        b_action = np.stack(samples[:,1])
        b_reward = np.array(samples[:,2])
        b_next_state = np.stack(samples[:,3])
        b_done = np.array(samples[:,4])

        #
        target_q = self._critic.target_predict(b_next_state, \
                self._actor.target_action(b_next_state))

        gamma = self._args.gamma

        b_y = []
        for i in range(samples.shape[0]):
            if b_done[i] == True:
                b_y.append(b_reward[i] + np.zeros(target_q[i].shape))
            else:
                b_y.append(b_reward[i] + gamma * target_q[i])
        b_y = np.stack(b_y)

        loss = self._critic.train(b_state, b_action, b_y)
        a_for_grad = self._actor.action(b_state)

        grad = self._critic.gradient(b_state, a_for_grad)[0]
        self._actor.train(b_state, grad)

        self._actor.update_target_network()
        self._critic.update_target_network()
            
    def saveNetwork(self, new_path=None):
        # save your own network
        if new_path != None:
            checkpoint_dir = new_path
        else:
            checkpoint_dir = self._checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self._saver.save(self._sess, os.path.join(checkpoint_dir, 'trained_agent'))
        print('model saved!')

    def load(self):
        self._saver.restore(self._sess, os.path.join(self._checkpoint_dir, 'trained_agent'))

    def action(self,state):
        # return an action by state.
        state = np.expand_dims(state, axis=0)
        action = self._actor.action(state)
        action = np.squeeze(action, axis=0)

#        action[0] = np.clip(action[0], -1.0 , 1.0)
#        action[1] = np.clip(action[1], 0.0 , 1.0)
#        action[2] = np.clip(action[2], 0.0 , 1.0)

        print(action)

        action[0] = np.clip(action[0], -0.3 , 0.3)
        action[1] = np.clip(action[1], 0.5 , 0.7)
        action[2] = np.clip(action[2], 0.0 , 0.2)

        print(action)
    
        return action

    def noise_action(self, state, epsilon):
        # return an action according to the current policy and exploration noise
        action = self.action(state)

        def noise(x, mu, theta, sigma):
            return theta * (mu - x) + sigma * np.random.randn(1)

        prev_action = np.copy(action)

        action[0] += max(epsilon, 0) * noise(action[0],  0.0 , 0.60, 0.30)
        action[1] += max(epsilon, 0) * noise(action[1],  0.5 , 1.00, 0.10)
        action[2] += max(epsilon, 0) * noise(action[2], -0.1 , 1.00, 0.05)

#        action[0] = np.clip(action[0], -1.0 , 1.0)
#        action[1] = np.clip(action[1], 0.0 , 1.0)
#        action[2] = np.clip(action[2], 0.0 , 1.0)

        print(action)

        action[0] = np.clip(action[0], -0.3 , 0.3)
        action[1] = np.clip(action[1], 0.5 , 0.7)
        action[2] = np.clip(action[2], 0.0 , 0.2)

        print(action)

        return action


