import tensorflow as tf
from network_models.layers import conv, fully_connected, spectral_norm, instanceNorm, leaky_relu


class SNGANDiscriminator:
    '''Generative Advesarial Imitation Learning'''
    def __init__(self, obs_shape, batch_size):
        """
        visual forecasting by imitation learning class
        obs_shape: stacked state image shape
        """

        with tf.variable_scope('discriminator'):
            # get vaiable scope name
            self.lr = tf.placeholder(dtype=tf.float32, name='learningrate')
            self.scope = tf.get_variable_scope().name

            # expert state placeholder
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[batch_size]+obs_shape)
            # expert action placeholder
            self.expert_s_next = tf.placeholder(dtype=tf.float32, shape=[batch_size]+obs_shape)
            # concatenate state and action to input discriminator
            expert_policy = tf.concat([self.expert_s, self.expert_s_next], axis=1)
            expert_policy = tf.squeeze(expert_policy, [-1])
            expert_policy = tf.transpose(expert_policy, [0, 2, 3, 1])

            # agent state placeholder
            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[batch_size]+obs_shape)
            # agent action placeholder
            self.agent_s_next = tf.placeholder(dtype=tf.float32, shape=[batch_size]+obs_shape)
            # concatenate state and action to input discriminator
            agent_policy = tf.concat([self.agent_s, self.agent_s_next], axis=1)
            agent_policy = tf.squeeze(agent_policy, [-1])
            agent_policy = tf.transpose(agent_policy, [0, 2, 3, 1])

            # channel of input
            self.input_channel = obs_shape[0] * 2

            with tf.variable_scope('network') as network_scope:
                # state-action of expert
                expert_prob = self.construct_network(input=expert_policy, is_sn=True)
                # share parameter of same scope with expert and agent
                network_scope.reuse_variables()
                # state-action of agent
                agent_prob = self.construct_network(input=agent_policy, is_sn=True)

            with tf.variable_scope('loss'):
                # maximiz D(s,a), because expert rewards are bigger than agent rewards
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(expert_prob, 0.01, 1)))
                tf.summary.scalar('loss_expert', loss_expert)

                # maximize 1-D(s,a), because agent rewards are smoller than expert rewards
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - agent_prob, 0.01, 1)))
                tf.summary.scalar('loss_agent', loss_agent)

                # inverse sign because tensorflow minimize loss
                loss = loss_expert + loss_agent
                self.loss = - loss
                # add discriminator loss to summary
                tf.summary.scalar('discriminator', self.loss)

            # optimize operation
            #optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-5)
            self.train_op = optimizer.minimize(self.loss)

            # 全てのsummaryを取得するoperation
            self.merged = tf.summary.merge_all()

            # fix discriminator and get d_reward
            self.rewards = tf.log(tf.clip_by_value(agent_prob, 1e-10, 1))

    def construct_network(self, input, is_sn=True):
        '''SNGAN'''
        with tf.variable_scope('block_1'):
            x = leaky_relu(conv(input, [5, 5, self.input_channel, 64], [1, 2, 2, 1], is_sn))
        with tf.variable_scope('block_2'):
            x = leaky_relu(instanceNorm(conv(x, [5, 5, 64, 128], [1, 2, 2, 1], is_sn)))
        with tf.variable_scope('block_3'):
            x = leaky_relu(instanceNorm(conv(x, [5, 5, 128, 256], [1, 2, 2, 1], is_sn)))
        with tf.variable_scope('block_4'):
            x = leaky_relu(instanceNorm(conv(x, [5, 5, 256, 512], [1, 2, 2, 1], is_sn)))
        with tf.variable_scope('block_5'):
            x = tf.layers.flatten(x, name='flatten')
            x = fully_connected(x, 1, is_sn)
            # sigmoid activation 0~1
            prob = tf.sigmoid(x, name='prob')
        return prob

    def train(self, expert_s, expert_s_next, agent_s, agent_s_next, lr):
        '''
        train discriminator function
        expert_s, expert_s_next: state-action of expert
        agent_s, agent_s_next: state-action of expert
        '''
        return tf.get_default_session().run(
                [self.train_op, self.loss],
                feed_dict={
                    self.lr: lr,
                    self.expert_s: expert_s,
                    self.expert_s_next: expert_s_next,
                    self.agent_s: agent_s,
                    self.agent_s_next: agent_s_next})

    def get_summary(self, expert_s, expert_s_next, agent_s, agent_s_next, lr):
        '''summary operation実行関数'''
        return tf.get_default_session().run(
                self.merged,
                feed_dict={
                    self.lr: lr,
                    self.expert_s: expert_s,
                    self.expert_s_next: expert_s_next,
                    self.agent_s: agent_s,
                    self.agent_s_next: agent_s_next})

    def get_rewards(self, agent_s, agent_s_next):
        ''' fix D and get rewards
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
