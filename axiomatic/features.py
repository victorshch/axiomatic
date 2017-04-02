import numpy as np
from scipy import stats


class Maximum(object):
    def __call__(self, sample):
        """
        Compute maximum feature for an array of samples
        @param self:
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing maximum value for each sample
        """
        return np.max(sample, 1).reshape(-1, 1)


class Minimum(object):
    def __call__(self, sample):
        """
        Computes minimum feature for an array of samples
        @param self:
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing minimum value for each sample
        """
        return np.min(sample, 1).reshape(-1, 1)


class Mean(object):
    def __call__(self, sample):
        """
        Computes mean feature for an array of samples
        @param self:
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing mean value for each sample
        """
        return np.mean(sample, 1).reshape(-1, 1)


class StdDeviation(object):
    def __call__(self, sample):
        """
        Computes standard deviation feature for an array of samples
        @param self:
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing std deviation for each sample
        """
        return np.std(sample, 1).reshape(-1, 1)


class LinearRegressionCoef(object):
    def __call__(self, sample):
        """
        Computes linear regression coefficient for an array of samples.
        see https://en.wikipedia.org/wiki/Simple_linear_regression
        @param self:
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing linear regression coefficient for each sample
        """
        
        n = sample.shape[1]
        
        x = np.arange(1.0, n + 1)
        x_mean = (n + 1) / 2.0
        x_variation = n * (n * n - 1) / 12.0
        
        y_mean = np.mean(sample, 1).reshape(-1, 1)
        
        value = (x - x_mean) * (sample - y_mean)
        
        slope = np.sum(value, 1) / x_variation
        
        return slope.reshape(-1, 1)


class FFTCoef(object):
    def __init__(self, n_coef):
        """
        @param self:
        @param n_coef: number of fft coefficients
        """
        self.n_coef = n_coef

    def __call__(self, sample):
        """
        Computes first self.n_coef fft coefficients for an array of samples
        @param self:
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x self.n_coef numpy array containing fft coefficients for each sample
        """
        fft_coef = np.abs(np.fft.rfft(sample, axis=1))
        return fft_coef[:, :self.n_coef]
