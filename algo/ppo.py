import tensorflow as tf
import copy


class PPOTrain:
    '''PPO trainer class'''
    def __init__(
            self,
            Policy,
            Old_Policy,
            obs_shape,
            gamma=0.95,
            clip_value=0.2,
            c_vf=0.2,
            c_entropy=0.01,
            c_l1=1.0,
            obs_size=64,
            vf_clip=''):
        # Policy network
        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma
        # get trainable parameters
        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        # assign parameters operation
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # prepare placeholder
        with tf.variable_scope('train_inp'):
            self.lr = tf.placeholder(dtype=tf.float32,
                    name='learning_rate')
            self.rewards = tf.placeholder(dtype=tf.float32,
                    shape=[None, 1], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32,
                    shape=[None, 1], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32,
                    shape=[None, 1], name='gaes')
            self.expart_act = tf.placeholder(dtype=tf.float32,
                    shape=[None, obs_size * obs_size])

        # get distribution probs
        probs = self.Policy.probs_op
        probs_old = self.Old_Policy.probs_op
        # get value
        v_preds = self.Policy.v_preds_op
        sample_act = self.Policy.sample_op

        with tf.variable_scope('loss'):
            # 更新後の方策と更新前の方策のKL距離の制約
            # ratio = tf.div(probs, probs_old)
            # 収益期待値->最大化
            ratios = tf.exp(
                    tf.log(tf.clip_by_value(probs, 1e-10, 1.0)) - \
                            tf.log(tf.clip_by_value(probs_old, 1e-10, 1.0)))
            # clipping ratios
            clipped_ratios = tf.clip_by_value(
                    ratios,
                    clip_value_min=1 - clip_value,
                    clip_value_max=1 + clip_value)
            # clipping前とclipping後のlossで小さい方を使う
            loss_clip = tf.minimum(
                    tf.reduce_mean(tf.multiply(self.gaes, ratios)),
                    tf.reduce_mean(tf.multiply(self.gaes, clipped_ratios)))
            self.loss_clip = tf.reduce_mean(loss_clip)
            # add clipping loss to summary
            tf.summary.scalar('loss_clip', self.loss_clip)

            # 状態価値の分散を大きくしないための制約項->最小化
            loss_vf = tf.squared_difference(self.rewards + \
                    self.gamma * self.v_preds_next, v_preds)
            self.loss_vf = tf.reduce_mean(loss_vf)
            if vf_clip:
                self.loss_vf = tf.clip_by_value(self.loss_vf, 1e-10, 1000)
            tf.summary.scalar('value_difference', self.loss_vf)

            # 探索を促すためのentropy制約項->最大化
            # 方策のentropyが小さくなりすぎるのを防ぐ
            entropy = -tf.reduce_mean(probs * \
                    tf.log(tf.clip_by_value(probs, 1e-10, 1.0)), axis=1)
            self.entropy = tf.reduce_mean(entropy, axis=0)
            tf.summary.scalar('entropy', self.entropy)

            # L1 loss
            loss_l1 = tf.reduce_sum(
                    tf.abs(sample_act - self.expart_act),
                    axis=1)
            self.loss_l1 = tf.reduce_mean(loss_l1)
            tf.summary.scalar('loss_l1', self.loss_l1)

            # 以下の式を最小化
            loss = self.loss_clip - c_vf * self.loss_vf + \
                    c_entropy * self.entropy - c_l1 * self.loss_l1
            self.loss = -loss
            # tensorflowのoptimizerは最小最適化を行うため
            tf.summary.scalar('total', self.loss)

        # optimizer
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-5)
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)

        # 勾配の取得
        self.gradients = optimizer.compute_gradients(self.loss, var_list=pi_trainable)

        # train operation
        self.train_op = optimizer.minimize(self.loss, var_list=pi_trainable)

        # 全てのsummaryを取得するoperation
        self.merged = tf.summary.merge_all()

    def train(self, obs , gaes, rewards, v_preds_next, expert_act, lr):
        '''train operation実行関数'''
        return tf.get_default_session().run(
                [self.train_op, self.loss, self.loss_clip, self.loss_vf, self.entropy, self.loss_l1],
                feed_dict={
                    self.Policy.obs: obs,
                    self.Old_Policy.obs: obs,
                    self.lr: lr,
                    self.rewards: rewards,
                    self.v_preds_next: v_preds_next,
                    self.gaes: gaes,
                    self.expart_act: expert_act})

    def get_summary(self, obs , gaes, rewards, v_preds_next, expert_act, lr):
        '''summary operation実行関数'''
        return tf.get_default_session().run(
                self.merged,
                feed_dict={
                    self.Policy.obs: obs,
                    self.Old_Policy.obs: obs,
                    self.lr: lr,
                    self.rewards: rewards,
                    self.v_preds_next: v_preds_next,
                    self.gaes: gaes,
                    self.expart_act: expert_act})

    def assign_policy_parameters(self):
        '''PolicyのパラメータをOld_Policyに代入'''
        return tf.get_default_session().run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        '''
        GAE: generative advantage estimator
        Advantage関数で何ステップ先まで考慮すべきかを1つの式で表現
        rewards: 即時報酬系列
        v_preds: 状態価値
        v_preds_next: 次の状態の状態価値
        '''
        # advantage関数
        # 現時点で予測している状態価値と実際に行動してみたあとの状態価値との差
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        # is T-1, where T is time step which run policy
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = deltas[t] + self.gamma * gaes[t + 1]
        return gaes

    def get_grad(self, obs , gaes, rewards, v_preds_next, expert_act, lr):
        '''
        勾配計算関数
        obs: 状態
        gaes: generative advantage estimator
        rewards: 即時報酬系列
        v_preds_next: 次の状態価値関数
        '''
        return tf.get_default_session().run(
                self.gradients,
                feed_dict={
                    self.Policy.obs: obs,
                    self.Old_Policy.obs: obs,
                    self.lr: lr,
                    self.rewards: rewards,
                    self.v_preds_next: v_preds_next,
                    self.gaes: gaes,
                    self.expart_act: expert_act})
