import numpy as np
import tensorflow as tf
from ops import *
import os
from env import *
from experience_replay import *


class Painter():
    def __init__(self):
        self.img_dim = 32
        self.num_colors = 1
        self.num_actions = 3
        self.tau = 0.001 #how much to update target network from actor network

        # actor network
        self.actor_state = tf.placeholder(tf.float32, [None, self.img_dim, self.img_dim, self.num_colors])
        self.actor_out = self.actor(self.actor_state, "actor_real")
        self.actor_params = [var for var in tf.trainable_variables() if 'actor_real' in var.name]

        # target actor network
        self.actor_target_state = tf.placeholder(tf.float32, [None, self.img_dim, self.img_dim, self.num_colors])
        self.actor_target_out = self.actor(self.actor_target_state, "actor_target")
        self.actor_target_params = [var for var in tf.trainable_variables() if 'actor_target' in var.name]


        # critic network
        self.critic_state = tf.placeholder(tf.float32, [None, self.img_dim, self.img_dim, self.num_colors])
        self.critic_action = tf.placeholder(tf.float32, [None, self.num_actions])
        self.critic_out = self.critic(self.critic_state, self.critic_action, "critic_real")
        self.critic_params = [var for var in tf.trainable_variables() if 'critic_real' in var.name]

        # critic target network
        self.critic_target_state = tf.placeholder(tf.float32, [None, self.img_dim, self.img_dim, self.num_colors])
        self.critic_target_action = tf.placeholder(tf.float32, [None, self.num_actions])
        self.critic_target_out = self.critic(self.critic_target_state, self.critic_target_action, "critic_target")
        self.critic_target_params = [var for var in tf.trainable_variables() if 'critic_target' in var.name]

        # how does the Q network change as the action chosen changes?
        # can be seen as: how should the action chosen increase/decrease for better reward?
        self.critic_gradient = tf.gradients(self.critic_out, self.critic_action)

        # update actions according to self.action_gradient
        # self.action_gradient is a measure of which direction we want the action to move
        # if action_gradient is positiive, we want the action to be a higher value. etc
        self.action_gradient = tf.placeholder(tf.float32, [None, self.num_actions])
        self.actor_gradients = tf.gradients(self.actor_out, self.actor_params, -self.action_gradient)
        grads_zip = zip(self.actor_gradients, self.actor_params)
        self.optimize_actor = tf.train.AdamOptimizer(0.001).apply_gradients(grads_zip)

        # slowly update the target actor network -> actor network
        # also target critic network -> critic network
        self.update_actor_target = [self.actor_target_params[i].assign(tf.mul(self.actor_params[i], self.tau) + tf.mul(self.actor_target_params[i], 1. - self.tau)) for i in range(len(self.actor_params))]
        self.update_critic_target = [self.critic_target_params[i].assign(tf.mul(self.critic_params[i], self.tau) + tf.mul(self.critic_target_params[i], 1. - self.tau)) for i in range(len(self.critic_params))]

        # update the critic towards new values
        self.new_critic_values = tf.placeholder(tf.float32, [None, 1])
        self.critic_loss = tf.nn.l2_loss(self.new_critic_values - self.critic_out)
        self.optimize_critic = tf.train.AdamOptimizer(0.001).minimize(self.critic_loss)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    # given s, return an action
    def actor(self, state, scope):
        with tf.variable_scope(scope):
            h1 = lrelu(conv2d(state, 1, 16, "p_h1")) # 32x32x1 -> 16x16x16
            h2 = lrelu(conv2d(h1, 16, 32, "p_h2")) # 16x16x16 -> 8x8x32
            h3 = lrelu(conv2d(h2, 32, 64, "p_h3")) # 8x8x32 -> 4x4x64
            flattened = tf.reshape(h3, [-1, 4*4*64])

            h4 = tf.nn.relu(dense(flattened, 4*4*64, 300, "p_h4"))
            w_mean = dense(h4, 300, self.num_actions, "w_mean")
        return tf.sigmoid(w_mean)

    # Q(s,a) = what value?
    def critic(self, state, actions, scope):
        with tf.variable_scope(scope):
            flatstate = tf.reshape(state, [-1, self.img_dim*self.img_dim*self.num_colors])
            s1 = tf.nn.relu(dense(flatstate, self.img_dim*self.img_dim*self.num_colors, 300, "s1"))
            s2 = tf.nn.relu(dense(s1, 300, 300, "s2"))
            a1 = tf.nn.relu(dense(actions, self.num_actions, 300, "a1"))
            combined1 = tf.concat(1,[s2,a1])
            combined2 = tf.nn.relu(dense(combined1, 600, 300, "combined2"))
            out = dense(combined2, 300, 1, "out")
        return out

    def train(self):
        self.env = Canvas()
        self.buffer = ReplayBuffer(3000)

        # train for 10k episodes
        for i in xrange(10000):
            print("Episode : " + str(i) + " Replay Buffer " + str(self.buffer.size()))
            self.env.reset()

            totalreward = 0

            # 50 steps in an episode
            for s in xrange(50):
                current_state = self.env.get_state()

                actions = self.sess.run(self.actor_out, feed_dict={self.actor_state: np.expand_dims(current_state, 0)})[0]
                actions += np.random.randn(3) / 10.0
                actions = np.clip(actions, 0, 1)

                next_state, reward = self.env.step(actions)
                totalreward += reward
                self.buffer.add(current_state, actions, reward, (s == 49), next_state)

                if self.buffer.count > 32:
                    # Now it's time to learn!
                    states, actions, rewards, dones, new_states = self.buffer.sample_batch(32)
                    updated_values = np.zeros(32)

                    # What actions would we take from the new states
                    new_actions = self.sess.run(self.actor_target_out, feed_dict={self.actor_target_state: new_states})
                    # What is the Q value of the new states (taking acttions from policy)
                    target_q_values = self.sess.run(self.critic_target_out, feed_dict={self.critic_target_state: new_states, self.critic_target_action: new_actions})

                    for k in range(32):
                        if dones[k]:
                            updated_values[k] = rewards[k]
                        else:
                            updated_values[k] = rewards[k] + 0.99*target_q_values[k]
                    updated_values = np.expand_dims(updated_values, 1)

                    # update the critic towards [reward + next state value]
                    critic_loss, _ = self.sess.run([self.critic_loss, self.optimize_critic], feed_dict={self.critic_state: states, self.critic_action: actions, self.new_critic_values: updated_values})
                    # print "critic loss: %d" % critic_loss

                    # update the actor network to increase reward
                    actions_chosen = self.sess.run(self.actor_out, feed_dict={self.actor_state: states})
                    # action_gradients = how should i change the action i took (increase/decrease) to get more reward?
                    action_gradients = self.sess.run(self.critic_gradient, feed_dict={self.critic_state: states, self.critic_action: actions_chosen})[0]
                    # optmize = how can i adjust my policy parameters so my action changes as specified in action_gradients
                    self.sess.run(self.optimize_actor, feed_dict={self.actor_state: states, self.action_gradient: action_gradients})

                    # slowly bring targets up to date
                    self.sess.run(self.update_actor_target)
                    self.sess.run(self.update_critic_target)

            print "Total reward was %d" % totalreward







paint = Painter()
paint.train()
