import tensorflow as tf


class Policy_dcgan:
    '''encoder decoder policy network'''
    # relu or leaky_relu
    nonlinear = tf.nn.leaky_relu
    #nonlinear = tf.nn.relu
    def __init__(self, name, obs_shape, decode=True):

        with tf.variable_scope(name):
            # placeholder for input state
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None]+obs_shape, name='obs')

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
                        x = nonlinear(x, name='nonlin')
                        #x = tf.nn.relu(x, name='relu')

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
                        x = nonlinear(x, name='nonlin')
                        #x = tf.nn.relu(x, name='relu')

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
                        x = nonlinear(x, name='nonlin')
                        #x = tf.nn.relu(x, name='relu')

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
                        x = nonlinear(x, name='nonlin')
                        #x = tf.nn.relu(x, name='relu')

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
                        x = nonlinear(x, name='nonlin')
                        x = tf.nn.relu(x, name='relu')
                        self.enc_feature = tf.reshape(x, shape=[-1,2,2,512])

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
                            x = nonlinear(x, name='nonlin')
                            #x = tf.nn.relu(x, name='relu')

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
                            x = nonlinear(x, name='nonlin')
                            #x = tf.nn.relu(x, name='relu')

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
                            x = nonlinear(x, name='nonlin')
                            #x = tf.nn.relu(x, name='relu')

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
                            x = nonlinear(x, name='nonlin')
                            #x = tf.nn.relu(x, name='relu')

                        # 32x32x64 -> 64x64x1
                        with tf.variable_scope('dec_5'):
                            # inference action mean
                            mu = tf.layers.conv2d_transpose(
                                    x,
                                    filters=1,
                                    kernel_size=(5,5),
                                    strides=2,
                                    padding='same',
                                    activation=None,
                                    name='deconv_mu')
                            #mu = tf.nn.tanh(tf.layers.flatten(mu), name='mu')
                            mu = tf.layers.flatten(mu, name='mu')

                            # inference action std
                            sigma = tf.layers.conv2d_transpose(
                                    x,
                                    filters=1,
                                    kernel_size=(5,5),
                                    strides=2,
                                    padding='same',
                                    activation=None,
                                    name='deconv_sigma')
                            #sigma = tf.nn.softplus(tf.layers.flatten(sigma), name='sigma')
                            sigma = tf.layers.flatten(tf.clip_by_value(sigma, 1e-10, 10.0), name='sigma')

                            # action space distribution
                            dist = tf.contrib.distributions.MultivariateNormalDiag(
                                    loc=tf.zeros(shape=tf.shape(mu)),
                                    scale_diag=tf.ones(shape=tf.shape(sigma)),
                                    allow_nan_stats=False,
                                    name='dist')
                            '''
                            dist = tf.distributions.Normal(
                                    loc=mu,
                                    scale=sigma,
                                    name='dist'
                                    )
                            '''


                            if tf.__version__ == '1.4.0':
                                dist = tf.contrib.distributions.Independent(dist, reduce_batch_ndims=0)
                            else:
                                dist = tf.contrib.distributions.Independent(dist, reinterpreted_batch_ndims=0)

                            print('distribution batch shape: ', dist.batch_shape)
                            print('distribution event shape: ', dist.event_shape)

                            # sampling seed
                            #seed = dist.sample()
                            #self.sample_act_op = mu + seed * sigma
                            self.sample_act_op = dist.sample()

                            # prob operation
                            self.act_probs_op = dist.prob(self.sample_act_op, name='act_probs_op')

                            # test operation
                            self.test_op = self.act_probs_op

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
                        x = nonlinear(x, name='nonlin')
                        #x = tf.nn.relu(x, name='relu')

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
                        x = nonlinear(x, name='nonlin')
                        #x = tf.nn.relu(x, name='relu')

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
                        x = nonlinear(x, name='nonlin')
                        #x = tf.nn.relu(x, name='relu')

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
                        x = nonlinear(x, name='nonlin')
                        #x = tf.nn.relu(x, name='relu')

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
                        self.v_preds = tf.reshape(x, shape=[-1,1], name='v_preds')

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

    def get_variables(self):
        '''get param function'''
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        '''get trainable param function'''
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def test_run(self, obs):
        '''train operation実行関数'''
        return tf.get_default_session().run(
                self.test_op,
                feed_dict={self.obs: obs})
