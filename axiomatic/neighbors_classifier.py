# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from utils import transform_dataset
from dtw_on_steroids import dtw_distances


def index_of(lst, predicate):
    for i, elem in enumerate(lst):
        if predicate(elem):
            return i
    return -1


def last_index_of(lst, predicate):
    i = index_of(reversed(lst), predicate)
    if i == -1:
        return -1
    return len(lst) - i - 1


class TimeSeriesKNearestNeighborsClassifier(object):
    def __init__(self, axiom_system, model_dict, config={}):
        self.axiom_system = axiom_system
        self.model_dict = model_dict

        self.n_neighbors = config.get('n_neighbors', 5)
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric='precomputed')
        self.encoder = LabelEncoder()

        models, tmp_labels = transform_dataset(self.model_dict)

        labels = []
        self.model_markings = []

        for i, model in enumerate(models):
            marking = self.axiom_system.perform_marking(model)
            first_axiom_index = index_of(marking, lambda x: x >= 0)
            if first_axiom_index > 0:
                last_axiom_index = last_index_of(marking, lambda x: x >= 0)
                stripped_model = marking[first_axiom_index:last_axiom_index + 1]
                self.model_markings.append(stripped_model)
                labels.append(tmp_labels[i])

        X_train = self.precompute_distances(self.model_markings, self.model_markings)
        y_train = self.encoder.fit_transform(labels)

        self.model.fit(X_train, y_train)

    @staticmethod
    def precompute_distances(models, markings):
        dst = np.zeros((len(markings), len(models)), dtype=float)
        for i, marking in enumerate(markings):
            for j, model in enumerate(models):
                dst[i, j] = np.min(dtw_distances(model, marking))
        return dst

    def predict(self, ts_list):
        markings = [self.axiom_system.perform_marking(ts) for ts in ts_list]
        X_test = self.precompute_distances(self.model_markings, markings)
        y_pred = self.model.predict(X_test)
        return self.encoder.inverse_transform(y_pred)
