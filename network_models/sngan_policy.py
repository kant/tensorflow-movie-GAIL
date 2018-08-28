import numpy as np
import tensorflow as tf
from network_models.layers import *


class SNGANPolicy:
    '''SNGAN Encoder Decoder'''
    def __init__(self, name, obs_shape, batch_size, decode=True):

        with tf.variable_scope(name):
            # placeholder for input state
            self.obs = tf.placeholder(dtype=tf.float32, shape=[batch_size]+obs_shape, name='obs')
            self.obs = tf.squeeze(self.obs, [-1])
            self.obs = tf.transpose(self.obs, [0, 2, 3, 1])
            input_c = tf.shape(self.obs)[-1]
            self.leaky = leaky

            # policy network
            with tf.variable_scope('policy_net'):
                with tf.variable_scope('encoder'):
                    with tf.variable_scope('block_1')
                        x = leaky_relu(conv(self.obs,
                            [5, 5, input_c, 64], [1, 2, 2, 1]))
                    with tf.variable_scope('block_2'):
                        x = leaky_relu(instanceNorm(conv(x,
                            [5, 5, 64, 128], [1, 2, 2, 1])))
                    with tf.variable_scope('block_3'):
                        x = leaky_relu(instanceNorm(conv(x,
                            [5, 5, 128, 256], [1, 2, 2, 1])))
                    with tf.variable_scope('block_4'):
                        x = leaky_relu(instanceNorm(conv(x,
                            [5, 5, 256, 512], [1, 2, 2, 1])))

            if decode:
                with tf.variable_scope('decoder')
                    with tf.variable_scope(name_or_scope="block_1"):
                        x = tf.nn.relu(instanceNorm(deconv(x,
                            [5, 5, 256, 512], [1, 2, 2, 1], [batchsize, 8, 8, 256])))
                    with tf.variable_scope(name_or_scope="block_2"):
                        x = tf.nn.relu(instanceNorm(deconv(x,
                            [5, 5, 128, 256], [1, 2, 2, 1], [batchsize, 16, 16, 128]))
                    with tf.variable_scope(name_or_scope="block_3"):
                        x = tf.nn.relu(instanceNorm(deconv(x,
                            [5, 5, 64, 128], [1, 2, 2, 1], [batchsize, 32, 32, 64])))
                    with tf.variable_scope(name_or_scope="block_4"):
                        mu = tf.nn.sigmoid(instanceNorm(deconv(x,
                            [5, 5, 64, 128], [1, 2, 2, 1], [batchsize, 64, 64, 1])))
                        self.mu = tf.layers.flatten(mu, name='mu')

                        sigma = tf.nn.softplus(instanceNorm(deconv(x,
                            [5, 5, 64, 128], [1, 2, 2, 1], [batchsize, 64, 64, 1])))
                        sigma = tf.layers.flatten(mu)
                        self.sigma = tf.clip_by_value(sigma, 1e-10, 10.0, name='sigma')

                # sample operation
                samples = self.mu + self.sigma * \
                        tf.random_normal([tf.shape(self.mu)[0], tf.shape(self.mu)[1]])

                # calclate prob density
                probs = tf.exp(- 0.5 * \
                        (tf.square((samples - self.mu) / self.sigma))) / \
                        (tf.sqrt(2 * np.pi) * self.sigma)

                self.sample_op = samples
                self.probs_op = probs

                # test operation
                self.test_op = tf.shape(self.sample_op)

            # value network
            with tf.variable_scope('value_net'):
                with tf.variable_scope('encoder'):
                    with tf.variable_scope('block_1')
                        x = leaky_relu(conv(self.obs,
                            [5, 5, input_c, 64], [1, 2, 2, 1]))
                    with tf.variable_scope('block_2'):
                        x = leaky_relu(instanceNorm(conv(x,
                            [5, 5, 64, 128], [1, 2, 2, 1])))
                    with tf.variable_scope('block_3'):
                        x = leaky_relu(instanceNorm(conv(x,
                            [5, 5, 128, 256], [1, 2, 2, 1])))
                    with tf.variable_scope('block_4'):
                        x = leaky_relu(instanceNorm(conv(x,
                            [5, 5, 256, 512], [1, 2, 2, 1])))
                    with tf.variable_scope('block_5'):
                        x = tf.layers.flatten(x, name='flatten')
                        self.v_preds_op = fully_connected(x, 1)

            # get network scope name
            self.scope = tf.get_variable_scope().name

    def act(self, obs):
        '''
        action function
        get sampled stochastic action and predicted value
        obs: stacked state image
        '''
        return tf.get_default_session().run(
                [self.sample_op, self.v_preds_op],
                feed_dict={self.obs: obs})

    def get_mu_sigma(self, obs):
        return tf.get_default_session().run(
                [self.mu, self.sigma],
                feed_dict={self.obs: obs})

    def get_variables(self):
        '''get param function'''
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        '''get trainable param function'''
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def test_run(self, obs):
        '''train operatiion function'''
        return tf.get_default_session().run(
                self.test_op,
                feed_dict={self.obs: obs})
