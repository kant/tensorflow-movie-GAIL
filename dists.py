import numpy as np
import tensorflow as tf

class Dist(object):
    """
    A particular probability distribution
    """
    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return - self.neglogp(x)


class DiagGaussianDist(Dist):
    '''微分可能なガウス分布クラス'''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.logstd = tf.log(std)

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
                + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
                + tf.reduce_sum(self.logstd, axis=-1)

    def prob(self, x):
        return tf.exp(self.logp(x))

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))
