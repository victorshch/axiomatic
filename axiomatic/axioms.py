import numpy as np


class ClusteringAxiom(object):
    def __init__(self, model, cluster_id):
        """
        @param self:
        @param model: sklearn.cluster.KMeans object, containing information about cluster centers
        @param cluster_id: number of the cluster for this axiom
        """
        self.model = model
        self.cluster_id = cluster_id

    def is_satisfied(self, x):
        """
        Check whether axiom is satisfied.
        @param self:
        @param x: 1-dim or 2-dim numpy.array, features computed for time series samples
        @return: True if 1-dim array x belongs to cluster_id cluster, False otherwise
        (return an array of bool if x has 2 dimensions)
        """
        return self.model.predict(x) == self.cluster_id


class DummyAxiom(object):
    def is_satisfied(self, x):
        """
        Check whether axiom is satisfied.
        @param self:
        @param x: 1-dim or 2-dim numpy.array
        @return: True of bool numpy.array, containing True values
        """
        if x.ndim == 1:
            return True
        else:
            return np.full((x.shape[0]), True, dtype=bool)
