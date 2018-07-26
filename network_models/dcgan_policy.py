import tensorflow as tf


class Policy_dcgan:
    '''dcgan実装によるpolicy networkクラス'''
    def __init__(self, name, obs_shape=(64,64,3), decode=True):

        with tf.variable_scope(name):
            # placeholder for input state
            self.obs = tf.placeholder(dtype=tf.float32, shape=obs_shape, name='obs')

            # policy network
            with tf.variable_scope('policy_net'):
                with tf.variable_scope('enc'):
                    with tf.variable_scope('enc_1'):
                        x = tf.layers.conv3d(
                                inputs=self.obs,
                                filters=64,
                                kernel_size=(5,5),
                                stride=2,
                                activation=None,
                                name='conv')
                        x = tf.nn.relu(x, name='relu')

                    with tf.variable_scope('enc_2'):
                        x = tf.layers.conv3d(
                                x,
                                filters=128,
                                kernel_size=(5,5),
                                stride=2,
                                activation=None,
                                name='conv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        x = tf.nn.relu(x, name='relu')

                    with tf.variable_scope('enc_3'):
                        x = tf.layers.conv3d(
                                x,
                                filters=256,
                                kernel_size=(5,5),
                                stride=2,
                                activation=None,
                                name='conv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        x = tf.nn.relu(x, name='relu')

                    with tf.variable_scope('enc_4'):
                        x = tf.layers.conv3d(
                                x,
                                filters=512,
                                kernel_size=(5,5),
                                stride=2,
                                activation=None,
                                name='conv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        x = tf.nn.relu(x, name='relu')

                    with tf.variable_scope('enc_5'):
                        x = tf.layers.conv3d(
                                x,
                                filters=512,
                                kernel_size=(5,5),
                                stride=2,
                                activation=None,
                                name='conv')
                        self.enc_feature = tf.nn.relu(x, name='relu')
                if decode:
                    with tf.variable_scope('dec'):
                        with tf.variable_scope('dec_1'):
                            x = tf.layers.conv2d_transpose(
                                    self.enc_feature,
                                    filters=512,
                                    kernel_size=(5,5),
                                    stride=2,
                                    activation=None,
                                    name='deconv')
                            x = tf.layers.batch_normalization(x, name='BN')
                            x = tf.nn.relu(x, name='relu')

                        with tf.variable_scope('dec_2'):
                            x = tf.layers.conv2d_transpose(
                                    x,
                                    filters=256,
                                    kernel_size=(5,5),
                                    stride=2,
                                    activation=None,
                                    name='deconv')
                            x = tf.layers.batch_normalization(x, name='BN')
                            x = tf.nn.relu(x, name='relu')

                        with tf.variable_scope('dec_3'):
                            x = tf.layers.conv2d_transpose(
                                    x,
                                    filters=128,
                                    kernel_size=(5,5),
                                    stride=2,
                                    activation=None,
                                    name='deconv')
                            x = tf.layers.batch_normalization(x, name='BN')
                            x = tf.nn.relu(x, name='relu')

                        with tf.variable_scope('dec_4'):
                            x = tf.layers.conv2d_transpose(
                                    x,
                                    filters=64,
                                    kernel_size=(5,5),
                                    stride=2,
                                    activation=None,
                                    name='deconv')
                            x = tf.layers.batch_normalization(x, name='BN')
                            x = tf.nn.relu(x, name='relu')

                        with tf.variable_scope('dec_5'):
                            # inference action mean
                            mu = tf.layers.conv2d_transpose(
                                    x,
                                    filters=1,
                                    kernel_size=(5,5),
                                    stride=2,
                                    activation=tf.nn.sigmoid,
                                    name='deconv_mu')
                            mu = tf.nn.tanh(tf.layers.flatten(mu), name='mu')

                            # inference action std
                            sigma = tf.layers.conv2d_transpose(
                                    x,
                                    filters=1,
                                    kernel_size=(5,5),
                                    stride=2,
                                    activation=tf.nn.sigmoid,
                                    name='deconv_sigma')
                            sigma = tf.nn.softplus(tf.layer.flatten(sigma), name='sigma')

                            # action space distribution
                            dist = tf.distributions.Normal(loc=mu, scale=sigma, name='dist')

                            # sampling operarion
                            sample = tf.squeeze(dist.sample(1), axis=0)
                            self.sample_act_op = tf.reshape(sample, shape=(obs_shape[0], obs_shape[1]), name='sample_act_op')

                            # get action prob operation
                            self.act_prob_op = dist.prob(self.obs, name='act_prob_op')


            # value network
            with tf.variable_scope('value_net'):
                with tf.variable_scope('enc'):
                    with tf.variable_scope('enc_1'):
                        x = tf.layers.conv3d(
                                inputs=self.obs,
                                filters=64,
                                kernel_size=(5,5),
                                stride=2,
                                activation=None,
                                name='conv')
                        x = tf.nn.relu(x, name='relu')

                    with tf.variable_scope('enc_2'):
                        x = tf.layers.conv3d(
                                x,
                                filters=128,
                                kernel_size=(5,5),
                                stride=2,
                                activation=None,
                                name='conv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        x = tf.nn.relu(x, name='relu')

                    with tf.variable_scope('enc_3'):
                        x = tf.layers.conv3d(
                                x,
                                filters=256,
                                kernel_size=(5,5),
                                stride=2,
                                activation=None,
                                name='conv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        x = tf.nn.relu(x, name='relu')

                    with tf.variable_scope('enc_4'):
                        x = tf.layers.conv3d(
                                x,
                                filters=512,
                                kernel_size=(5,5),
                                stride=2,
                                activation=None,
                                name='conv')
                        x = tf.layers.batch_normalization(x, name='BN')
                        x = tf.nn.relu(x, name='relu')

                    with tf.variable_scope('enc_5'):
                        x = tf.layers.conv3d(
                                x,
                                filters=1,
                                kernel_size=(5,5),
                                stride=2,
                                activation=None,
                                name='conv')
                        self.v_preds = tf.nn.sigmoid(x, name='v_preds')

            # get network scope name
            self.scope = tf.get_variable_scope().name

    def act(self, obs):
        '''
        action function
        get sampled stochastic action and predicted value
        '''
        return tf.get_default_session().run(
                [self.sample_act_op, self.v_preds]
                feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        '''
        obs: state image
        get action prob conditioned from state
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
