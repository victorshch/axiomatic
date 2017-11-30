# -*- coding: utf-8 -*-

import numpy as np
from statsmodels.tsa.ar_model import AR
import math

def generate_binary(sample):
  now = sample - np.mean(sample, axis=1).reshape(-1, 1)
  VP = np.max(now, axis=1).reshape(-1, 1)
  VN = np.min(now, axis=1).reshape(-1, 1)
  PC = np.sum(np.logical_and(0 < now, now < VP * 0.1), axis=1)
  NC = np.sum(np.logical_and(0.1 * VN < now, now < 0), axis=1)
  n = sample.shape[1]

  TD = np.logical_and(PC + NC >= 0.4 * n, PC < NC).reshape(-1, 1) * 0.2 * VP + np.logical_and(PC + NC >= -0.4 * n, PC > NC).reshape(-1, 1) * 0.2 * VN
  return sample >= TD

class Count2(object):
    def __init__(self, n):
        now = np.zeros(n)
        now[0] = 1

        for i in range(1, n):
            now[i] = (14 * now[i - 1] - 7 * now[i - 2]) / 8
        self.mul = np.hstack(tuple(np.concatenate((now[i :], np.full(i, 0))).reshape(-1, 1) for i in range(n))).T

    def __call__(self, sample):
        sample = np.nan_to_num(sample)
        sample = (sample - np.hstack((np.full((len(sample), 2), 0), sample[:, : -2]))) / 16
        now = np.abs(np.dot(sample, self.mul))
        return np.sum(now.mean(axis=1).reshape(-1, 1) <= now, axis=1).reshape(-1, 1)
        '''res = np.zeros(len(sample))

        for i in range(len(sample)):
            now = np.zeros(len(sample[i]))

            for j in range(len(sample[i])):
                if j >= 2:
                    now[j] = (14 * now[j - 1] - 7 * now[j - 2] + (sample[i][j] - sample[i][j - 2]) / 2) / 8
                else:
                    now[j] = sample[i][j]
            now = np.abs(now)
            res[i] = np.sum(np.logical_and(np.mean(now) <= now, now <= np.max(now)))
        return res.reshape(-1, 1)'''

class Leakage(object):
    def __call__(self, sample):
        T = (np.sum(np.abs(sample), axis=1) * np.sum(np.abs(np.hstack((sample[:, 0].reshape(-1, 1), sample[:, 1:] - sample[:, :-1]))), axis=1) ** (-1) * 2 * math.pi).astype(int)
        res = np.zeros(len(sample))

        for i in range(len(T)):
            if T[i] // 2 >= len(sample[i]):
                res[i] = 1
            else:
                res[i] = np.sum(np.abs(sample[i][T[i] // 2 :] + sample[i][: -(T[i] // 2)])) * np.sum(np.abs(sample[i][T[i] // 2 :]) + np.abs(sample[i][: -(T[i] // 2)])) ** (-1)
        return res.reshape(-1, 1)

class BinaryCovariance(object):
    def __call__(self, sample):
        sample = generate_binary(sample)
        return np.var(sample, axis=1).reshape(-1, 1)

class BinaryFrequency(object):
    def __call__(self, sample):
        sample = generate_binary(sample)
        return np.sum(np.abs(sample[:, 1 :] - sample[:, : -1]), axis=1).reshape(-1, 1)

class AreaBinary(object):
    def __call__(self, sample):
        sample = generate_binary(sample)
        return np.maximum(np.sum(sample, axis=1), np.sum(np.full(sample.shape, 1) - sample, axis=1)).reshape(-1, 1)

class Complexity(object):
    def __call__(self, sample):
        sample = generate_binary(sample)
        res = np.zeros(len(sample))

        for i in range(len(sample)):
            j = 1
            k = 2
            c = 1
            
            while k < len(sample[i]):
                while k < len(sample[i]) and sample[i][j : k] in sample[i][: k - 1]:
                    k += 1
                j = k
                k = j + 1
                c += 1
            res[i] = c * math.log(len(sample[i])) / len(sample[i])
        return res.reshape(-1, 1)

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

def p3(v):
    return v*v*v

def p4(v):
    return v*v*v*v

class Kurtosis(object):
    def __call__(self, sample):
        """
        See https://en.wikipedia.org/wiki/Kurtosis
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing kurtosis for each sample
        """
        n = sample.shape[1]

        value = np.sum(p4(sample - np.mean(sample, axis=1).reshape(-1, 1)), axis=1).reshape(-1, 1) / n
        sigma = np.std(sample, axis=1).reshape(-1, 1)

        return value / p4(sigma)


class Skewness(object):
    def __call__(self, sample):
        """
        See https://en.wikipedia.org/wiki/Skewness
        @param sample: m x n numpy array, m -- number of samples, n -- length of each sample
        @return: m x 1 numpy array containing skewness for each sample
        """
        n = sample.shape[1]

        value = np.sum(p3(sample - np.mean(sample, axis=1).reshape(-1, 1)), axis=1).reshape(-1, 1) / n
        sigma = np.std(sample, axis=1).reshape(-1, 1)

        return value / p3(sigma)


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
