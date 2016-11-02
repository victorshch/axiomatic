import random
import numpy as np
import settings


class FeatureExtractionStage(object):
    def __init__(self, config):
        """
        @param self:
        @param config: config for feature extraction stage, should be dict (e.g. {'sample_length': 20, 'ratio': 0.2})
        @return: True if 1-dim array x belongs to cluster_id cluster, False otherwise
        (return an array of bool if x has 2 dimensions)
        """
        self.sample_length = config.get('sample_length', settings.DEFAULT_SAMPLE_LENGTH)
        self.ratio = config.get('ratio', settings.DEFAULT_RATIO)
        self.features = config.get('features', settings.DEFAULT_FEATURES)

    def make_samples(self, X):
        """
        Sample time series, every sample would have length == self.sample_length, number of samples
        for each row would be proportional to self.ratio * number of different possible samles
        of length self.sample_length
        @param self:
        @param X: numpy.array containing 1-dim time series
        @return: numpy.array with samples of initial time series
        """
        n_samples = int(self.ratio * (X.shape[1] - self.sample_length))
        samples = []

        for i in xrange(X.shape[0]):
            start_counts = random.sample(range(X.shape[1] - self.sample_length), n_samples)
            samples.extend([X[i][j:j + self.sample_length] for j in start_counts])

        return np.array(samples)

    def make_features(self, X):
        """
        Compute features for every sample
        @param self:
        @param X: numpy.array containing 1-dim time series
        @return: numpy.array with computed features
        """
        sample_features = []
        for sample in X:
            computed_features = []
            for feature in self.features:
                f = feature(sample)
                if isinstance(f, list):
                    computed_features.extend(f)
                else:
                    computed_features.append(f)
            sample_features.append(computed_features)
        return np.array(sample_features)

    def train(self, X, artifacts=None):
        """
        Combines make_samples and make_features steps
        @param self:
        @param X: numpy.array containing 1-dim time series
        @param artifacts: additional parameters from previous steps
        @return: numpy.array with computed features
        """
        X_sample = self.make_samples(X)
        return self.make_features(X_sample)
