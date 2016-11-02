import numpy as np
from scipy import stats


class Maximum(object):
    def __call__(self, sample):
        """
        Compute maximum feature
        @param self:
        @param sample: 1-dim numpy.array
        @return: maximum value from sample
        """
        return np.max(sample)


class Minimum(object):
    def __call__(self, sample):
        """
        Computes minimum feature
        @param self:
        @param sample: 1-dim numpy.array
        @return: minimum value from sample
        """
        return np.min(sample)


class Mean(object):
    def __call__(self, sample):
        """
        Computes mean feature
        @param self:
        @param sample: 1-dim numpy.array
        @return: mean value for values from sample
        """
        return np.mean(sample)


class StdDeviation(object):
    def __call__(self, sample):
        """
        Computes standard deviation feature
        @param self:
        @param sample: 1-dim numpy.array
        @return: std deviation for values from sample
        """
        return np.std(sample)


class LinearRegressionCoef(object):
    def __call__(self, sample):
        """
        Computes linear regression coefficient
        @param self:
        @param sample: 1-dim numpy.array
        @return: linear regression coefficient
        """
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(sample)), sample)
        return slope


class FFTCoef(object):
    def __init__(self, n_coef):
        """
        @param self:
        @param n_coef: number of fft coefficients
        """
        self.n_coef = n_coef

    def __call__(self, sample):
        """
        Computes first self.n_coef fft coefficients
        @param self:
        @param sample: 1-dim numpy.array
        @return: list with self.n_coef fft coefficients
        """
        fft_coef = np.abs(np.fft.rfft(sample))
        return list(fft_coef[:self.n_coef])
