# -*- coding: utf-8 -*-

import numpy as np
from axiomatic.utils import dtw_distances


class TimeSeriesClassifier(object):
    def __init__(self, axiom_system, model_dict, config=None):
        """
        @param axiom_system: instance of axiomatic.base.AxiomSystem
        @param model_dict: models for every possible class of time series
        @param config: config for time series classifier
        """

        self.axiom_system = axiom_system
        self.model_dict = model_dict

        self.classification_threshold = 0
        if config is not None:
            self.classification_threshold = config.get('classification_threshold', 0)

    def _ts_predict(self, ts):
        """
        Compute distance to every model from model_dict for a single ts
        @param ts: time series
        @return dict with class name as key and distance as value
        """
        marked_ts = self.axiom_system.perform_marking(ts)

        class_distances = {}

        for class_name, class_model in self.model_dict.iteritems():
            class_distances[class_name] = np.min(dtw_distances(class_model, marked_ts, lambda a, b: a != b))

        return class_distances

    def predict(self, ts_list):
        """
        Predict the class of time series
        @param ts_list: list of time series
        @return predicted class for every time series
        """
        predictions = []

        for ts in ts_list:
            single_pred = self._ts_predict(ts)
            predicted_class = min(single_pred.iteritems(), lambda tup: tup[1])
            predictions.append(predicted_class)

        return predictions
