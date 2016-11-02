from sklearn.cluster import KMeans
from axioms import DummyAxiom, ClusteringAxiom


class KMeansClusteringAxiomStage(object):
    def __init__(self, config):
        """
        @param self:
        @param config: config for clustering stage, should be dict (e.g. {'clustering_params': {'n_clusters': 15,
        'init': 'k-means++', 'n_init': 10}})
        """
        self.model = KMeans(**config.get('clustering_params', {}))

    def train(self, X, artifacts=None):
        """
        Train clustering stage, generate ClusteringAxiom axioms
        @param self:
        @param X: 2-dimensional numpy.array, where every row corresponds to features computed for samples of time series
        @param artifacts: additional parameters from previous steps
        @return: list of axioms
        """
        self.model.fit(X)
        axioms = []
        for cluster in self.model.n_clusters:
            axioms.append(ClusteringAxiom(self.model, cluster))
        axioms.append(DummyAxiom())
        return axioms
