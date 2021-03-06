import numpy as np
import tensorflow as tf


class DCGANPolicy:
<<<<<<< HEAD:network_models/dcgan_policy.py
    '''DCGAN Encoder Decoder including 3D conv'''

=======
    '''encoder decoder policy network'''
>>>>>>> c79cfc48f93b70a6c24e29d063cb881ff88f5fde:network_models/dcgan_policy.py
    def __init__(self, name, obs_shape, batch_size, decode=True, leaky=True):

        with tf.variable_scope(name):
            # placeholder for input state
            self.obs = tf.placeholder(dtype=tf.float32, shape=[batch_size]+obs_shape, name='obs')
            self.leaky = leaky

            # policy network
            with tf.variable_scope('policy_net'):
                with tf.variable_scope('enc'):
                    # 3x64x64x1 -> 3x32x32x64
                    with tf.variable_scope('enc_1'):
                        x = tf.layers.conv3d(
                                inputs=self.obs,
                                filters=64,
                                kernel_size=(3,5,5),
                                strides=(1,2,2),
                                padding='same',
                                activation=None,
                                name='conv')
                        if self.leaky:
                            x = tf.nn.leaky_relu(x, alpha=0.2, name='activation')
                        else:
                            x = tf.nn.relu(x, name='activation')

                    # 3x32x32x64 -> 3x16x16x128
                    with tf.variable_scope('enc_2'):
                        x = tf.layers.conv3d(
                                x,
                                filters=128,
                                kernel_size=(3,5,5),
                                strides=(1,2,2),
                                padding='same',
                                activation=None,
                                name='conv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        if self.leaky:
                            x = tf.nn.leaky_relu(x, alpha=0.2, name='activation')
                        else:
                            x = tf.nn.relu(x, name='activation')

                    # 3x16x16x128 -> 3x8x8x256
                    with tf.variable_scope('enc_3'):
                        x = tf.layers.conv3d(
                                x,
                                filters=256,
                                kernel_size=(3,5,5),
                                strides=(1,2,2),
                                padding='same',
                                activation=None,
                                name='conv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        if self.leaky:
                            x = tf.nn.leaky_relu(x, alpha=0.2, name='activation')
                        else:
                            x = tf.nn.relu(x, name='activation')

                    # 3x8x8x256 -> 3x4x4x512
                    with tf.variable_scope('enc_4'):
                        x = tf.layers.conv3d(
                                x,
                                filters=512,
                                kernel_size=(3,5,5),
                                strides=(1,2,2),
                                padding='same',
                                activation=None,
                                name='conv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        if self.leaky:
                            x = tf.nn.leaky_relu(x, alpha=0.2, name='activation')
                        else:
                            x = tf.nn.relu(x, name='activation')

                    # 3x4x4x512 -> 2x2x512
                    with tf.variable_scope('enc_5'):
                        x = tf.layers.conv3d(
                                x,
                                filters=512,
                                kernel_size=(3,5,5),
                                strides=(3,2,2),
                                padding='same',
                                activation=None,
                                name='conv')
                        if self.leaky:
                            x = tf.nn.leaky_relu(x, alpha=0.2, name='activation')
                        else:
                            x = tf.nn.relu(x, name='activation')
                        self.enc_feature = tf.reshape(x, shape=[batch_size,2,2,512])

            if decode:
                with tf.variable_scope('dec'):
                    # 2x2x512 -> 4x4x512
                    with tf.variable_scope('dec_1'):
                        x = tf.layers.conv2d_transpose(
                                self.enc_feature,
                                filters=512,
                                kernel_size=(5,5),
                                strides=2,
                                padding='same',
                                activation=None,
                                name='deconv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        if self.leaky:
                            x = tf.nn.leaky_relu(x, alpha=0.2, name='activation')
                        else:
                            x = tf.nn.relu(x, name='activation')

                    # 4x4x512 -> 8x8x256
                    with tf.variable_scope('dec_2'):
                        x = tf.layers.conv2d_transpose(
                                x,
                                filters=256,
                                kernel_size=(5,5),
                                strides=2,
                                padding='same',
                                activation=None,
                                name='deconv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        if self.leaky:
                            x = tf.nn.leaky_relu(x, alpha=0.2, name='activation')
                        else:
                            x = tf.nn.relu(x, name='activation')

                    # 8x8x256 -> 16x16x128
                    with tf.variable_scope('dec_3'):
                        x = tf.layers.conv2d_transpose(
                                x,
                                filters=128,
                                kernel_size=(5,5),
                                strides=2,
                                padding='same',
                                activation=None,
                                name='deconv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        if self.leaky:
                            x = tf.nn.leaky_relu(x, alpha=0.2, name='activation')
                        else:
                            x = tf.nn.relu(x, name='activation')

                    # 16x16x128 -> 32x32x64
                    with tf.variable_scope('dec_4'):
                        x = tf.layers.conv2d_transpose(
                                x,
                                filters=64,
                                kernel_size=(5,5),
                                strides=2,
                                padding='same',
                                activation=None,
                                name='deconv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        if self.leaky:
                            x = tf.nn.leaky_relu(x, alpha=0.2, name='activation')
                        else:
                            x = tf.nn.relu(x, name='activation')

                    # 32x32x64 -> 64x64x1
                    with tf.variable_scope('dec_5'):
                        # inference action mu
                        mu = tf.layers.conv2d_transpose(
                                x,
                                filters=1,
                                kernel_size=(5,5),
                                strides=2,
                                padding='same',
                                activation=None,
                                name='deconv_mu')
                        self.mu = tf.layers.flatten(tf.nn.sigmoid(mu), name='mu')

<<<<<<< HEAD:network_models/dcgan_policy.py
                        # inference action sigma
                        sigma = tf.layers.conv2d_transpose(
                                x,
                                filters=1,
                                kernel_size=(5,5),
                                strides=2,
                                padding='same',
                                activation=None,
                                name='deconv_sigma')
                        self.sigma = tf.clip_by_value(
                                tf.nn.softplus(tf.layers.flatten(sigma)),
                                1e-10,
                                1.0,
                                name='sigma')
=======
                            # inference action sigma
                            sigma = tf.layers.conv2d_transpose(
                                    x,
                                    filters=1,
                                    kernel_size=(5,5),
                                    strides=2,
                                    padding='same',
                                    activation=None,
                                    name='deconv_sigma')
                            sigma = tf.clip_by_value(
                                    tf.nn.softplus(tf.layers.flatten(sigma)),
                                    1e-10,
                                    1000.0,
                                    name='sigma')
>>>>>>> c79cfc48f93b70a6c24e29d063cb881ff88f5fde:network_models/dcgan_policy.py

                    # sample operation
                    samples = self.mu + self.sigma * \
                            tf.random_normal([tf.shape(self.mu)[0], tf.shape(self.mu)[1]])

                    # calclate prob density
                    probs = tf.exp(- 0.5 * \
                            (tf.square((samples - self.mu) / self.sigma))) / \
                            (tf.sqrt(2 * np.pi) * self.sigma)

<<<<<<< HEAD:network_models/dcgan_policy.py
                    self.sample_op = samples
                    self.probs_op = probs
=======
                            self.sample_op = samples
                            self.probs_op = probs
                            self.mu = mu
                            self.sigma = sigma
>>>>>>> c79cfc48f93b70a6c24e29d063cb881ff88f5fde:network_models/dcgan_policy.py

                    # test operation
                    self.test_op = tf.shape(self.sample_op)

            # value network
            with tf.variable_scope('value_net'):
                with tf.variable_scope('enc'):
                    # 3x64x64x1 -> 3x16x16x64
                    with tf.variable_scope('enc_1'):
                        x = tf.layers.conv3d(
                                inputs=self.obs, filters=64,
                                kernel_size=(3,5,5),
                                strides=(1,4,4),
                                padding='same',
                                activation=None,
                                name='conv')
                        if self.leaky:
                            x = tf.nn.leaky_relu(x, alpha=0.2, name='activation')
                        else:
                            x = tf.nn.relu(x, name='activation')

                    # 3x16x16x64 -> 3x8x8x128
                    with tf.variable_scope('enc_2'):
                        x = tf.layers.conv3d(
                                x,
                                filters=128,
                                kernel_size=(3,5,5),
                                strides=(1,2,2),
                                padding='same',
                                activation=None,
                                name='conv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        if self.leaky:
                            x = tf.nn.leaky_relu(x, alpha=0.2, name='activation')
                        else:
                            x = tf.nn.relu(x, name='activation')

                    # 3x8x8x128 -> 3x4x4x256
                    with tf.variable_scope('enc_3'):
                        x = tf.layers.conv3d(
                                x,
                                filters=256,
                                kernel_size=(3,5,5),
                                strides=(1,2,2), padding='same',
                                activation=None,
                                name='conv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        if self.leaky:
                            x = tf.nn.leaky_relu(x, alpha=0.2, name='activation')
                        else:
                            x = tf.nn.relu(x, name='activation')

                    # 3x4x4x256 -> 3x2x2x512
                    with tf.variable_scope('enc_4'):
                        x = tf.layers.conv3d(
                                x,
                                filters=512,
                                kernel_size=(3,5,5),
                                strides=(1,2,2),
                                padding='same',
                                activation=None,
                                name='conv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        if self.leaky:
                            x = tf.nn.leaky_relu(x, alpha=0.2, name='activation')
                        else:
                            x = tf.nn.relu(x, name='activation')

                    # 3x2x2x512 -> 1x1x1x1
                    with tf.variable_scope('enc_5'):
                        x = tf.layers.conv3d(
                                x,
                                filters=1,
                                kernel_size=(3,5,5),
                                strides=(3,2,2),
                                padding='same',
                                activation=None,
                                name='conv')
                        self.v_preds_op = tf.reshape(x, shape=[batch_size,1], name='v_preds')

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
    
    def inference(self, obs):
        '''inference function'''
        return tf.get_default_session().run(
                self.mu,
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
        '''train operation function'''
        return tf.get_default_session().run(
                self.test_op,
                feed_dict={self.obs: obs})
