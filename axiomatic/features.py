import numpy as np
from statsmodels.tsa.ar_model import AR


class Maximum(object):
    def __call__(self, sample):
        """
        Compute maximum feature for an array of samples
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing maximum value for each sample
        """
        return np.max(sample, axis=1).reshape(-1, 1)


class Minimum(object):
    def __call__(self, sample):
        """
        Computes minimum feature for an array of samples
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing minimum value for each sample
        """
        return np.min(sample, axis=1).reshape(-1, 1)


class Mean(object):
    def __call__(self, sample):
        """
        Computes mean feature for an array of samples
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing mean value for each sample
        """
        return np.mean(sample, axis=1).reshape(-1, 1)


class StdDeviation(object):
    def __call__(self, sample):
        """
        Computes standard deviation feature for an array of samples
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing std deviation for each sample
        """
        return np.std(sample, axis=1).reshape(-1, 1)


class Kurtosis(object):
    def __call__(self, sample):
        """
        See https://en.wikipedia.org/wiki/Kurtosis
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing kurtosis for each sample
        """
        n = sample.shape[1]

        value = np.sum((sample - np.mean(sample, axis=1).reshape(-1, 1))**4, axis=1).reshape(-1, 1) / n
        sigma = np.std(sample, axis=1).reshape(-1, 1)

        return value / sigma**4


class Skewness(object):
    def __call__(self, sample):
        """
        See https://en.wikipedia.org/wiki/Skewness
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing skewness for each sample
        """
        n = sample.shape[1]

        value = np.sum((sample - np.mean(sample, axis=1).reshape(-1, 1))**3, axis=1).reshape(-1, 1) / n
        sigma = np.std(sample, axis=1).reshape(-1, 1)

        return value / sigma**3


class LinearRegressionCoef(object):
    def __call__(self, sample):
        """
        Computes linear regression coefficient for an array of samples.
        see https://en.wikipedia.org/wiki/Simple_linear_regression
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing linear regression coefficient for each sample
        """
        n = sample.shape[1]

        x = np.arange(1.0, n + 1)
        x_mean = (n + 1) / 2.0
        x_variation = n * (n * n - 1) / 12.0

        y_mean = np.mean(sample, axis=1).reshape(-1, 1)

        value = (x - x_mean) * (sample - y_mean)

        slope = np.sum(value, axis=1) / x_variation

        return slope.reshape(-1, 1)


class ARCoef(object):
    def __init__(self, n_coef, use_constant=True):
        """
        @param n_coef: number of AR coefficients
        @param use_constant: use constant in AR model
        """
        self.n_coef = n_coef
        self.use_constant = use_constant

    def __call__(self, sample):
        """
        Computes self.n_coef AR coefficients for an array of samples
        See https://en.wikipedia.org/wiki/Autoregressive_model
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x self.n_coef numpy array containing AR coefficients for each sample
        """

        m = sample.shape[0]
        trend = 'c' if self.use_constant else 'nc'
        maxlag = self.n_coef - 1 if self.use_constant else self.n_coef
        features = []

        for i in xrange(m):
            model = AR(sample[i])
            results = model.fit(maxlag, trend=trend)
            features.append(results.params)

        return np.array(features)


class FFTCoef(object):
    def __init__(self, n_coef):
        """
        @param n_coef: number of fft coefficients
        """
        self.n_coef = n_coef

    def __call__(self, sample):
        """
        Computes first self.n_coef fft coefficients for an array of samples
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x self.n_coef numpy array containing fft coefficients for each sample
        """
        fft_coef = np.abs(np.fft.rfft(sample, axis=1))
        return fft_coef[:, :self.n_coef]
