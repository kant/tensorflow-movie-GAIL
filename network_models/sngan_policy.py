import numpy as np
import tensorflow as tf
<<<<<<< HEAD
from network_models.layers import *


class SNGANPolicy:
    '''SNGAN Encoder Decoder'''
=======
from network_models.layers import conv, fully_connected, spectral_norm, instanceNorm, leaky_relu, deconv


class SNGANPolicy:
    '''encoder decoder policy network'''
>>>>>>> c79cfc48f93b70a6c24e29d063cb881ff88f5fde
    def __init__(self, name, obs_shape, batch_size, decode=True):

        with tf.variable_scope(name):
            # placeholder for input state
            self.obs = tf.placeholder(dtype=tf.float32, shape=[batch_size]+obs_shape, name='obs')
<<<<<<< HEAD
            self.obs = tf.squeeze(self.obs, [-1])
            self.obs = tf.transpose(self.obs, [0, 2, 3, 1])
            input_c = tf.shape(self.obs)[-1]
            self.leaky = leaky
=======
            obs = tf.squeeze(self.obs, [-1])
            obs = tf.transpose(obs, [0, 2, 3, 1])
            img_H = obs_shape[1]
            img_W = obs_shape[2]
            input_channel = obs_shape[0]
>>>>>>> c79cfc48f93b70a6c24e29d063cb881ff88f5fde

            # policy network
            with tf.variable_scope('policy_net'):
                with tf.variable_scope('encoder'):
<<<<<<< HEAD
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
=======
                    with tf.variable_scope('block_1'):
                        x = leaky_relu(conv(obs, [5, 5, input_channel, 64], [1, 2, 2, 1]))
                    with tf.variable_scope('block_2'):
                        x = leaky_relu(instanceNorm(conv(x, [5, 5, 64, 128], [1, 2, 2, 1])))
                    with tf.variable_scope('block_3'):
                        x = leaky_relu(instanceNorm(conv(x, [5, 5, 128, 256], [1, 2, 2, 1])))
                    with tf.variable_scope('block_4'):
                        x = leaky_relu(instanceNorm(conv(x, [5, 5, 256, 512], [1, 2, 2, 1])))

            if decode:
                with tf.variable_scope('decoder'):
                    with tf.variable_scope('block_1'):
                        x = tf.nn.relu(instanceNorm(deconv(x, [5, 5, 256, 512], [1, 2, 2, 1], [batch_size, 8, 8, 256])))
                    with tf.variable_scope('block_2'):
                        x = tf.nn.relu(instanceNorm(deconv(x, [5, 5, 128, 256], [1, 2, 2, 1], [batch_size, 16, 16, 128])))
                    with tf.variable_scope('block_3'):
                        x = tf.nn.relu(instanceNorm(deconv(x, [5, 5, 64, 128], [1, 2, 2, 1], [batch_size, 32, 32, 64])))
                    with tf.variable_scope('block_4_mu'):
                        mu = tf.nn.relu(instanceNorm(deconv(x, [5, 5, 1, 64], [1, 2, 2, 1], [batch_size, img_H, img_W, 1])))
                        mu = tf.layers.flatten(tf.sigmoid(mu), name='mu')
                    with tf.variable_scope('block_4_sigma'):
                        sigma = tf.nn.relu(instanceNorm(deconv(x, [5, 5, 1, 64], [1, 2, 2, 1], [batch_size, img_H, img_W, 1])))
                        sigma = tf.layers.flatten(tf.nn.softplus(sigma), name='sigma')
                        sigma = tf.clip_by_value(sigma, 1e-10, 1000.0)

                    # sample operation
                    samples = mu + sigma * \
                            tf.random_normal([tf.shape(mu)[0], tf.shape(mu)[1]])

                    # calclate prob density
                    probs = tf.exp(- 0.5 * (tf.square((samples - mu) / sigma))) / \
                    (tf.sqrt(2 * np.pi) * sigma)

                    self.sample_op = samples
                    self.probs_op = probs
                    self.mu = mu
                    self.sigma = sigma

                    # test operation
                    self.test_op = tf.shape(self.sample_op)

            # value network
            with tf.variable_scope('value_net'):
                with tf.variable_scope('block_1'):
                    x = leaky_relu(conv(obs, [5, 5, input_channel, 64], [1, 2, 2, 1]))
                with tf.variable_scope('block_2'):
                    x = leaky_relu(instanceNorm(conv(x, [5, 5, 64, 128], [1, 2, 2, 1])))
                with tf.variable_scope('block_3'):
                    x = leaky_relu(instanceNorm(conv(x, [5, 5, 128, 256], [1, 2, 2, 1])))
                with tf.variable_scope('block_4'):
                    x = leaky_relu(instanceNorm(conv(x, [5, 5, 256, 512], [1, 2, 2, 1])))
                with tf.variable_scope('block_5'):
                    x = tf.layers.flatten(x, name='flatten')
                    self.v_preds_op = fully_connected(x, 1)
>>>>>>> c79cfc48f93b70a6c24e29d063cb881ff88f5fde

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

<<<<<<< HEAD
    def get_mu_sigma(self, obs):
        return tf.get_default_session().run(
                [self.mu, self.sigma],
=======
    def inference(self, obs):
        '''inference function'''
        return tf.get_default_session().run(
                self.mu,
>>>>>>> c79cfc48f93b70a6c24e29d063cb881ff88f5fde
                feed_dict={self.obs: obs})

    def get_variables(self):
        '''get param function'''
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        '''get trainable param function'''
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def test_run(self, obs):
<<<<<<< HEAD
        '''train operatiion function'''
=======
        '''train operation実行関数'''
>>>>>>> c79cfc48f93b70a6c24e29d063cb881ff88f5fde
        return tf.get_default_session().run(
                self.test_op,
                feed_dict={self.obs: obs})
