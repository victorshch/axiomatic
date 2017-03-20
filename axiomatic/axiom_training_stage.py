# coding=UTF-8
from scipy import optimize
from axiomatic.base import AbstractAxiom, Axiom, DummyAxiom
from scipy.ndimage import maximum_filter
import numpy as np
import random
from sklearn.cluster import KMeans

import settings

class DummyAxiomTrainingStage(object):
    """
    This training stage creates dummy axioms for every abnormal behavior class
    """
    def __init__(self, dummy_axiom_count = 10):
        self.dummy_axiom_count = dummy_axiom_count
        pass
    
    def train(self, data_set, artifacts):
        artifacts['axioms'] = {}
        for cl in data_set['train'].keys():
            if cl == 'normal': continue
            artifacts['axioms'][cl] = [DummyAxiom()] * self.dummy_axiom_count
        
        return artifacts

class FrequencyECTrainingStage(object):
    def __init__(self, config):
        self.num_part = config['num_part']
        self.max_window = config['max_window']
        self.num_axioms = config['num_axioms']
        self.axiom_list = config['axiom_list']

    def train(self, data_set, artifacts):
        normal = data_set["normal"]
        maxdim = data_set["normal"][0].shape[1]
        artifacts = dict()
        artifacts["axioms"] = dict()

        for name in data_set:
            if name != "normal":
                class_axioms = []

                for dim in range(maxdim):
                    dim_axioms = []

                    for axiom in self.axiom_list:
                        for sign in [-1, 1]:
                            abstract_axiom = AbstractAxiom(sign, dim, axiom)
                            rranges = (slice(0, len(normal[0]), 1), slice(0, self.max_window, 1))
                            rranges = rranges + abstract_axiom.bounds(data_set[name] + normal, self.num_part)

                            resbrute = optimize.brute(abstract_axiom.static_run, rranges,
                                args=(data_set[name], normal), full_output=True, finish=optimize.fmin)

                            res = (resbrute[3] == maximum_filter(resbrute[3], size=3)).ravel()
                            
                            axioms = list(zip(res, resbrute[3].ravel(), *[param.ravel() for param in resbrute[2]]))
                            axioms = [[param[0], param[1], int(param[2]), int(param[3])] + param[4:] for param in axioms]
                            axioms.sort(reverse=True)

                            high = sorted(res, reverse=True).index(False) if False in res else len(res)

                            axioms = axioms[0: min(self.num_axioms, high)]

                            axioms = [(params, sign, dim) for params in axioms]
                            dim_axioms += axioms
                    
                    dim_axioms.sort(reverse=True)
                    dim_axioms = dim_axioms[0: self.num_axioms]
                    print(dim_axioms)
                    dim_axioms = [AbstractAxiom(sign, dim, axiom, params[2:]) for params, sign, dim in dim_axioms]
                    
                    class_axioms += dim_axioms
                artifacts["axioms"][name] = class_axioms
        return artifacts


class FrequencyAxiomTrainingStage:
    def __init__(self, config):
        self.num_axioms = config['num_axioms']
        self.num_step_axioms = config['num_step_axioms']
        self.max_depth = config['max_depth']

    def train(self, data_set, artifacts):
        axioms = artifacts["axioms"]
        artifacts["full_axioms"] = dict()
        normal = data_set["normal"]

        for name in axioms:
            now = [Axiom(x) for x in axioms[name]]
            nums = [x.run_all(data_set[name], normal) for x in now]
            
            for i in range(self.max_depth):
                add = []

                for pos1 in range(len(now)):
                    for pos2 in range(pos1 + 1, len(now)):
                        axiom_or = now[pos1].logical_or(now[pos2])
                        axiom_and = now[pos1].logical_and(now[pos2])
                        num_or = axiom_or.run_all(data_set[name], normal)
                        num_and = axiom_and.run_all(data_set[name], normal)

                        if max(num_or, num_and) > min(nums[pos1], nums[pos2]):
                            add.append(axiom_or if num_or > num_and else axiom_and)
                now += add
                nums = [x.run_all(data_set[name], normal) for x in now]
                res = sorted(list(zip(nums, now)), key=lambda x: x[0])[0: self.num_step_axioms]

                now = [x[1] for x in res]
                nums = [x[0] for x in res]

            res = sorted(list(zip(nums, now)), key=lambda x: x[0])[0: self.num_axioms]
            res = [x[1] for x in res]
            artifacts["full_axioms"][name] = res
            return artifacts

class ClusteringAxiom(object):
    def __init__(self, model, feature_extractor, dim, cluster_id):
        """
        @param model: sklearn.cluster.KMeans object, containing information about cluster centers
        @param dim: dimension for which axiom is constructed
        @param cluster_id: number of the cluster for this axiom
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.dim = dim
        self.cluster_id = cluster_id

    def run(self, ts):
        """
        Check whether axiom is satisfied for some dimension of time series.
        @param ts: pd.DataFrame time series
        @return: 2-dim bool np.array where True corresponds to positions where axiom is satisfied
        """
        ans = np.full(ts.shape, False, dtype=bool)  # axiom can be satisfied only for specific dimension

        # specific dimension of time series
        dim_ts = ts.values[self.dim]

        sample_length = self.feature_extractor.sample_length
        features = self.feature_extractor.features

        # axiom can be satisfied only in central points of time series: from first_part position to len(ts) - 1 - last_part position
        first_part = sample_length / 2
        last_part = sample_length - first_part

        for i in range(first_part, len(dim_ts) - last_part):
            sample_ts = dim_ts[i - first_part, i - first_part + sample_length]

            computed_features_for_sample = []
            for feature in features:
                f = feature(sample_ts)
                if isinstance(f, list):
                    computed_features_for_sample.extend(f)
                else:
                    computed_features_for_sample.append(f)

            x = np.array(computed_features_for_sample)
            ans[self.dim][i] = (self.model.predict(x) == self.cluster_id)

        return ans


class DummyAxiom(object):
    def run(self, ts):
        """
        Check whether axiom is satisfied.
        @param ts: pd.DataFrame containing time series
        @return: bool numpy.array, containing True values on positions where axiom is satisfied
        """
        return np.full(ts.shape, True, dtype=bool)  # dummy axiom is satisfied on every position in all dimensions


class FeatureExtractionStage(object):
    def __init__(self, config):
        """
        @param config: config for feature extraction stage, should be dict (e.g. {'sample_length': 20, 'ratio': 0.2})
        """
        self.sample_length = config.get('sample_length', settings.DEFAULT_SAMPLE_LENGTH)
        self.ratio = config.get('ratio', settings.DEFAULT_RATIO)
        self.features = config.get('features', settings.DEFAULT_FEATURES)

    def make_samples(self, ts_list):
        """
        Sample time series, every sample would have length == self.sample_length, number of samples
        for each ts would be proportional to self.ratio * number of different possible samples
        of length self.sample_length
        @param ts_list: list of 1-dim numpy.array with time series
        @return: samples of initial time series
        """
        ts_len = len(ts_list[0])

        n_samples = int(self.ratio * (ts_len - self.sample_length))
        samples = []

        for ts in ts_list:
            start_counts = random.sample(range(ts_len - self.sample_length), n_samples)
            samples.extend([ts[j:j + self.sample_length] for j in start_counts])

        return samples

    def make_features(self, sample_list):
        """
        Compute features for every sample
        @param sample_list: list of numpy.array containing 1-dim time series
        @return: 2-dim numpy.array with computed features for every sample
        """
        sample_features = []
        for sample in sample_list:
            computed_features = []
            for feature in self.features:
                f = feature(sample)
                if isinstance(f, list):
                    computed_features.extend(f)
                else:
                    computed_features.append(f)
            sample_features.append(computed_features)
        return np.array(sample_features)

    def prepare_features(self, ts_list):
        """
        Combines make_samples and make_features steps
        @param ts_list: list of 1-dim numpy.array with time series
        @return: 2-dim numpy.array with computed features
        """
        samples = self.make_samples(ts_list)
        return self.make_features(samples)


class KMeansClusteringAxiomStage(object):
    def __init__(self, config):
        """
        @param config: config for clustering stage, should be dict (e.g. {'clustering_params': {'n_clusters': 15,
        'init': 'k-means++', 'n_init': 10}, 'feature_extraction_params': {'sample_length': 20, 'ratio': 0.2}}})
        """

        # config for clustering stage
        self.config = config

        # number of dimensions in time series
        self.n_dimensions = None

        # stores clustering models for each dimension
        self.clustering_models = []

        # stores feature extractor
        self.feature_extractor = FeatureExtractionStage(config.get('feature_extraction_params', {}))

    @staticmethod
    def get_all_time_series(training_set):
        all_time_series = []
        for class_name, ts_list in training_set.iteritems():
            all_time_series.extend(ts_list)

        return all_time_series

    @staticmethod
    def extract_dimension(ts_list, dim):
        """
        Extract specific dimension if time series
        @param ts_list: list of pd.DataFrame time series
        @param dim: dimension
        @return: list of 1-dim time series of specific dimension
        """

        ts_for_dim = []
        for ts in ts_list:
            ts_for_dim.append(ts.values[dim])

        return ts_for_dim

    def train_clustering_model_for_dim(self, dim_ts_list, dim):
        """
        Train clustering model for specific dimension
        @param dim_ts_list: list of time series for specific dimension
        @param dim: dimension for which model is being constructed
        @return: axioms for specific dimension
        """
        # make features for clustering
        X = self.feature_extractor.prepare_features(dim_ts_list)

        # train clustering model
        self.clustering_models[dim].fit(X)

        # generate axioms for this dimension
        axioms = []
        for cluster in self.clustering_models[dim].n_clusters:
            axioms.append(ClusteringAxiom(self.clustering_models[dim], self.feature_extractor, dim, cluster))

        return axioms

    def train(self, dataset, artifacts=None):
        """
        Train clustering stage, generate ClusteringAxiom axioms
        @param dataset: dataset in specific format
        @param artifacts: additional parameters from previous steps
        """

        all_time_series = self.get_all_time_series(dataset['train'])
        self.n_dimensions = all_time_series[0].shape[0]
        self.clustering_models = [KMeans(**self.config.get('clustering_params', {})) for i in range(self.n_dimensions)]

        all_axioms = [DummyAxiom()]

        for dim in range(self.n_dimensions):
            # list of specific dimension for all time series used for training
            dim_time_series = self.extract_dimension(all_time_series, dim)
            dim_axioms = self.train_clustering_model_for_dim(dim_time_series, dim)
            all_axioms.extend(dim_axioms)

        artifacts['axioms'] = all_axioms
        return artifacts
