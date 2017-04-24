# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from utils import transform_dataset
from dtw_on_steroids import dtw_distances


class CustomKNearestNeighborsClassifier(object):
    def __init__(self, axiom_system, model_dict, config={}):
        self.axiom_system = axiom_system
        self.model_dict = model_dict

        self.n_neighbors = config.get('n_neighbors', 5)
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric='precomputed')
        self.encoder = LabelEncoder()

        models, labels = transform_dataset(self.model_dict)
        self.model_markings = [self.axiom_system.perform_marking(model) for model in models]
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
