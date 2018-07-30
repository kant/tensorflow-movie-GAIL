import tensorflow as tf


class Discriminator:
    def __init__(self, obs_shape, lr):
        """
        visual forecasting by imitation learning class
        obs_shape: stacked state image shape
        """

        with tf.variable_scope('discriminator'):
            # get vaiable scope name
            self.scope = tf.get_variable_scope().name

            # expert state placeholder
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None]+obs_shape)
            # expert action placeholder
            self.expert_s_next = tf.placeholder(dtype=tf.float32, shape=[None]+obs_shape)
            # add noise to expert action
            #self.expert_s_next += tf.random_normal(tf.shape(self.expert_s_next), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            # concatenate state and action to input discriminator
            expert_policy = tf.concat([self.expert_s, self.expert_s_next], axis=1)

            # agent state placeholder
            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None]+obs_shape)
            # agent action placeholder
            self.agent_s_next = tf.placeholder(dtype=tf.float32, shape=[None]+obs_shape)
            # add noise to agent action
            #self.agent_s_next += tf.random_normal(tf.shape(self.agent_s_next), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            # concatenate state and action to input discriminator
            agent_policy = tf.concat([self.agent_s, self.agent_s_next], axis=1)

            with tf.variable_scope('network') as network_scope:
                # state-action of expert
                expert_prob = self.construct_network(input=expert_policy)
                # share parameter of same scope with expert and agent
                network_scope.reuse_variables()
                # state-action of agent
                agent_prob = self.construct_network(input=agent_policy)

            with tf.variable_scope('loss'):
                # maximiz D(s,a), because expert rewards are bigger than agent rewards
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(expert_prob, 0.01, 1)))
                tf.summary.scalar('loss_expert', loss_expert)

                # maximize 1-D(s,a), because agent rewards are smoller than expert rewards
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - agent_prob, 0.01, 1)))
                tf.summary.scalar('loss_agent', loss_agent)

                # inverse sign because tensorflow minimize loss
                loss = loss_expert + loss_agent
                loss = -loss
                # add discriminator loss to summary
                tf.summary.scalar('discriminator', loss)

            # optimize operation
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
            self.train_op = optimizer.minimize(loss)

            # 全てのsummaryを取得するoperation
            self.merged = tf.summary.merge_all()

            # fix discriminator and get d_reward
            self.rewards = tf.log(tf.clip_by_value(agent_prob, 1e-10, 1))

    def construct_network(self, input):
        '''
        input: expertかactionのstate-action
        discriminatorのbuild関数
        '''

        with tf.variable_scope('block_1'):
            # 6x64x64x1 -> 6x16x16x64
            x = tf.layers.conv3d(
                    inputs=input,
                    filters=64,
                    kernel_size=(6,5,5),
                    strides=(1,4,4),
                    padding='same',
                    activation=None,
                    name='conv')
            x = tf.nn.relu(x, name='relu')

        with tf.variable_scope('block_2'):
            # 6x16x16x64 -> 6x8x8x128
            x = tf.layers.conv3d(
                    x,
                    filters=128,
                    kernel_size=(6,5,5),
                    strides=(1,2,2),
                    padding='same',
                    activation=None,
                    name='conv')
            x = tf.layers.batch_normalization(x, name='BN')
            x = tf.nn.relu(x, name='relu')

        with tf.variable_scope('block_3'):
            # 6x8x8x128 -> 6x4x4x256
            x = tf.layers.conv3d(
                    x,
                    filters=256,
                    kernel_size=(6,5,5),
                    strides=(1,2,2),
                    padding='same',
                    activation=None,
                    name='conv')
            x = tf.layers.batch_normalization(x, name='BN')
            x = tf.nn.relu(x, name='relu')

        with tf.variable_scope('block_4'):
            # 6x4x4x256 -> 6x2x2x512
            x = tf.layers.conv3d(
                    x,
                    filters=512,
                    kernel_size=(6,5,5),
                    strides=(1,2,2),
                    padding='same',
                    activation=None,
                    name='conv')
            x = tf.layers.batch_normalization(x, name='BN')
            x = tf.nn.relu(x, name='relu')

        with tf.variable_scope('block_5'):
            # 6x2x2x512 -> 1x1x1x1
            x = tf.layers.conv3d(
                    x,
                    filters=1,
                    kernel_size=(6,5,5),
                    strides=(6,2,2),
                    padding='same',
                    activation=None,
                    name='conv')

            x = tf.layers.flatten(x, name='flatten')
            # sigmoid activation 0~1
            prob = tf.sigmoid(x, name='prob')

        return prob

    def train(self, expert_s, expert_s_next, agent_s, agent_s_next):
        '''
        train discriminator function
        expert_s, expert_s_next: state-action of expert
        agent_s, agent_s_next: state-action of expert
        '''
        return tf.get_default_session().run(
                self.train_op,
                feed_dict={
                    self.expert_s: expert_s,
                    self.expert_s_next: expert_s_next,
                    self.agent_s: agent_s,
                    self.agent_s_next: agent_s_next})

    def get_summary(self, expert_s, expert_s_next, agent_s, agent_s_next):
        '''summary operation実行関数'''
        return tf.get_default_session().run(
                self.merged,
                feed_dict={
                    self.expert_s: expert_s,
                    self.expert_s_next: expert_s_next,
                    self.agent_s: agent_s,
                    self.agent_s_next: agent_s_next})

    def get_rewards(self, agent_s, agent_s_next):
        '''
        fix D and get rewards
        agent_s: agent state
        agent_s_next: agent action
        '''
        return tf.get_default_session().run(
                self.rewards,
                feed_dict={
                    self.agent_s: agent_s,
                    self.agent_s_next: agent_s_next})

    def get_trainable_variables(self):
        '''get trainable paremeter function'''
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
