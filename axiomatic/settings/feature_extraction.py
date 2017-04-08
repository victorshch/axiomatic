from ..features import (
    Maximum,
    Minimum,
    Mean,
    StdDeviation,
    Kurtosis,
    Skewness,
    LinearRegressionCoef,
)

DEFAULT_SAMPLE_LENGTH = 20
DEFAULT_RATIO = 0.3
DEFAULT_FEATURES = [Maximum(), Minimum(), Mean(), StdDeviation(), Kurtosis(), Skewness(), LinearRegressionCoef()]
