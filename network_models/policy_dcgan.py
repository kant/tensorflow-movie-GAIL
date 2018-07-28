import tensorflow as tf


class Policy_dcgan:
    '''encoder decoder policy network'''
    def __init__(self, name, obs_shape, decode=True):

        with tf.variable_scope(name):
            # placeholder for input state
            self.obs = tf.placeholder(dtype=tf.float32, shape=obs_shape, name='obs')

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
                        x = tf.nn.relu(x, name='relu')

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
                        x = tf.nn.relu(x, name='relu')

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
                        x = tf.nn.relu(x, name='relu')

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
                        x = tf.nn.relu(x, name='relu')

                    # 3x4x4x512 -> 1x2x2x512
                    with tf.variable_scope('enc_5'):
                        x = tf.layers.conv3d(
                                x,
                                filters=512,
                                kernel_size=(3,5,5),
                                strides=(3,2,2),
                                padding='same',
                                activation=None,
                                name='conv')
                        self.enc_feature = tf.reshape(
                                tf.nn.relu(x),
                                shape=(obs_shape[0],2,2,512))

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
                            x = tf.nn.relu(x, name='relu')

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
                            x = tf.nn.relu(x, name='relu')

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
                            x = tf.nn.relu(x, name='relu')

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
                            x = tf.nn.relu(x, name='relu')

                        # 32x32x64 -> 64x64x1
                        with tf.variable_scope('dec_5'):
                            # inference action mean
                            mu = tf.layers.conv2d_transpose(
                                    x,
                                    filters=1,
                                    kernel_size=(5,5),
                                    strides=2,
                                    padding='same',
                                    activation=tf.nn.sigmoid,
                                    name='deconv_mu')
                            mu = tf.nn.tanh(mu, name='mu')

                            # inference action std
                            sigma = tf.layers.conv2d_transpose(
                                    x,
                                    filters=1,
                                    kernel_size=(5,5),
                                    strides=2,
                                    padding='same',
                                    activation=tf.nn.sigmoid,
                                    name='deconv_sigma')
                            sigma = tf.nn.softplus(sigma, name='sigma')

                            # action space distribution
                            dist = tf.distributions.Normal(loc=mu, scale=sigma, name='dist')

                            # sampling operarion
                            sample = dist.sample(1)
                            self.sample_act_op = tf.reshape(
                                    sample,
                                    shape=(
                                        obs_shape[0],
                                        1,
                                        obs_shape[2],
                                        obs_shape[3],
                                        obs_shape[4]),
                                    name='sample_act_op')

                            # get action prob operation
                            self.act_probs = dist.prob(sample, name='act_probs')


            # value network
            with tf.variable_scope('value_net'):
                with tf.variable_scope('enc'):
                    # 3x64x64x1 -> 3x16x16x64
                    with tf.variable_scope('enc_1'):
                        x = tf.layers.conv3d(
                                inputs=self.obs,
                                filters=64,
                                kernel_size=(3,5,5),
                                strides=(1,4,4),
                                padding='same',
                                activation=None,
                                name='conv')
                        x = tf.nn.relu(x, name='relu')

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
                        x = tf.nn.relu(x, name='relu')

                    # 3x8x8x128 -> 3x4x4x256
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
                        x = tf.nn.relu(x, name='relu')

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
                        x = tf.nn.relu(x, name='relu')

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
                        x = tf.reshape(x, shape=(obs_shape[0], 1))
                        self.v_preds = tf.nn.sigmoid(x, name='v_preds')

            # get network scope name
            self.scope = tf.get_variable_scope().name

    def act(self, obs):
        '''
        action function
        get sampled stochastic action and predicted value
        obs: stacked state image
        '''
        return tf.get_default_session().run(
                [self.sample_act_op, self.v_preds],
                feed_dict={self.obs: obs})

    def get_action_probs(self, obs):
        '''
        get action prob conditioned from state
        obs: stacked state image
        '''
        return tf.get_default_session().run(
                self.act_prob_op,
                feed_dict={self.obs: obs})

    def get_variables(self):
        '''get param function'''
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        '''get trainable param function'''
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
