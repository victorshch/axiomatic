# -*- coding: utf-8 -*-

import random
import numpy as np
from scipy import optimize
from scipy.ndimage import maximum_filter
from sklearn.cluster import KMeans

from axiomatic import settings
from axiomatic.base import AbstractAxiom, Axiom, DummyAxiom, form_matrix
from axiomatic.utils import time_series_embedding


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
            artifacts['axioms'][cl] = [DummyAxiom() for i in xrange(self.dummy_axiom_count)]
        
        return artifacts


class FrequencyECTrainingStage(object):
    def __init__(self, config):
        self.num_part = config['num_part']
        self.left_window = config['left_window']
        self.right_window = config['right_window']
        self.num_axioms = config['num_axioms']
        self.axiom_list = config['axiom_list']
        self.enable_cache = config['enable_cache']

    def form_dict(self, data_set):
        res = dict()
        maxdim = data_set["train"]["normal"][0].shape[1]

        for name in data_set["train"]:
            res[name] = [[[] if not self.enable_cache else form_matrix(ts[dim], self.left_window, self.right_window) for dim in ts.columns] for ts in data_set["train"][name]]
        return res

    def train(self, data_set, artifacts):
        normal = data_set["train"]["normal"]
        maxdim = data_set["train"]["normal"][0].shape[1]
        artifacts = dict()
        artifacts["axioms"] = dict()

        cache = self.form_dict(data_set)
        artifacts["cache"] = cache

        for name in data_set["train"]:
            if name != "normal":
                class_axioms = []

                for dim in range(maxdim):
                    dim_axioms = []

                    for axiom in self.axiom_list:
                        for sign in [-1, 1]:
                            abstract_axiom = AbstractAxiom(sign, dim, axiom)
                            rranges = abstract_axiom.bounds(data_set["train"][name] + normal, self.num_part)

                            resbrute = optimize.brute(abstract_axiom.static_run, rranges,
                                args=(data_set["train"][name], normal, self.left_window, self.right_window, cache[name], cache["normal"]), full_output=True, finish=None)
                            
                            res = (resbrute[3] == maximum_filter(resbrute[3], size=3)).ravel()
                            
                            axioms = list(zip(res, resbrute[3].ravel(), *[param.ravel() for param in resbrute[2]]))
                            axioms = [[param[0], param[1], self.left_window, self.right_window] + list(param[2:]) for param in axioms]
                            axioms.sort(reverse=True)

                            high = sorted(res, reverse=True).index(False) if False in res else len(res)

                            axioms = axioms[0: min(self.num_axioms, high)]

                            axioms = [(params, sign, axiom) for params in axioms]
                            dim_axioms += axioms
                    
                    dim_axioms.sort(key=lambda x: x[0], reverse=True)
                    dim_axioms = dim_axioms[0: self.num_axioms]
                    dim_axioms = [AbstractAxiom(sign, dim, axiom, params[2:]) for params, sign, axiom in dim_axioms]
                    
                    class_axioms += dim_axioms
                artifacts["axioms"][name] = class_axioms
        return artifacts


class FrequencyAxiomTrainingStage:
    def __init__(self, config):
        self.num_axioms = config['num_axioms']
        self.num_step_axioms = config['num_step_axioms']
        self.max_depth = config['max_depth']

    def train(self, data_set, artifacts):
        cache = artifacts["cache"]
        axioms = artifacts["axioms"]
        result = dict()
        normal = data_set["train"]["normal"]

        for name in axioms:
            now = [Axiom(x) for x in axioms[name]]
            nums = [x.run_all(data_set["train"][name], normal, cache[name], cache["normal"]) for x in now]

            for i in range(self.max_depth):
                add = []

                for pos1 in range(len(now)):
                    for pos2 in range(pos1 + 1, len(now)):
                        axiom_or = now[pos1].logical_or(now[pos2])
                        axiom_and = now[pos1].logical_and(now[pos2])
                        num_or = axiom_or.run_all(data_set["train"][name], normal, cache[name], cache["normal"])
                        num_and = axiom_and.run_all(data_set["train"][name], normal, cache[name], cache["normal"])

                        if max(num_or, num_and) > min(nums[pos1], nums[pos2]):
                            add.append(axiom_or if num_or > num_and else axiom_and)
                now += add
                nums = [x.run_all(data_set["train"][name], normal, cache[name], cache["normal"]) for x in now]
                res = sorted(list(zip(nums, now)), key=lambda x: x[0])[0: self.num_step_axioms]

                now = [x[1] for x in res]
                nums = [x[0] for x in res]

            res = sorted(list(zip(nums, now)), key=lambda x: x[0])[0: self.num_axioms]
            res = [x[1] for x in res]
            result[name] = res

        artifacts["axioms"] = result
        artifacts.pop("cache")
        return artifacts


class ClusteringAxiom(object):
    def __init__(self, model, feature_extractor, dim, cluster_id):
        """
        @param model: sklearn.cluster.KMeans object, containing information about cluster centers
        @param feature_extractor: instance of FeatureExtractionStage
        @param dim: dimension for which axiom is constructed
        @param cluster_id: number of the cluster for this axiom
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.dim = dim
        self.cluster_id = cluster_id

    def run(self, ts, cache=None):
        """
        Check whether axiom is satisfied for some dimension of time series.
        @param ts: pd.DataFrame time series
        @param cache: cache for predictions
        @return: 2-dim bool np.array where True corresponds to positions where axiom is satisfied
        """
        cluster_ids = np.full(ts.shape[0], -1, dtype=int)  # cluster ids for every point of ts
        
        # we store clustering of ts at cache[self.model][self.dim]
        if (cache is not None) and (self.model in cache) and (self.dim in cache[self.model]):
            cluster_ids = cache[self.model][self.dim]
        else:
            # specific dimension of time series as pd.Series object
            dim_ts = ts[ts.columns[self.dim]]

            sample_length = self.feature_extractor.sample_length
            features = self.feature_extractor.features

            # axiom can be satisfied only in central points of time series:
            # from first_part position to len(ts) - 1 - last_part position
            left_nei = sample_length / 2
            right_nei = sample_length - left_nei - 1
            
            dim_ts_embedding = time_series_embedding(dim_ts, left_nei, right_nei)
                    
            feature_values_list = [feature(dim_ts_embedding.values) for feature in features]
            feature_values = np.hstack(feature_values_list)
            
            cluster_ids[left_nei: -right_nei] = self.model.predict(feature_values[left_nei: -right_nei, :])
            
            if isinstance(cache, dict):
                if self.model not in cache:
                    cache[self.model] = { self.dim: cluster_ids }
                else:
                    cache[self.model][self.dim] = cluster_ids

        return cluster_ids == self.cluster_id


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
        ts_len = min(len(ts) for ts in ts_list)

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
        sample_array = np.array(sample_list)
        feature_values_list = [feature(sample_array) for feature in self.features]
        return np.hstack(feature_values_list)

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
            ts_for_dim.append(ts.values[:, dim])

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
        for cluster in range(self.clustering_models[dim].n_clusters):
            axioms.append(ClusteringAxiom(self.clustering_models[dim], self.feature_extractor, dim, cluster))

        return axioms

    def train(self, dataset, artifacts={}):
        """
        Train clustering stage, generate ClusteringAxiom axioms
        @param dataset: dataset in specific format
        @param artifacts: additional parameters from previous steps
        """

        all_time_series = self.get_all_time_series(dataset['train'])
        self.n_dimensions = all_time_series[0].shape[1]
        self.clustering_models = [KMeans(**self.config.get('clustering_params', {})) for i in range(self.n_dimensions)]

        all_axioms = []

        for dim in range(self.n_dimensions):
            # list of specific dimension for all time series used for training
            dim_time_series = self.extract_dimension(all_time_series, dim)
            dim_axioms = self.train_clustering_model_for_dim(dim_time_series, dim)
            all_axioms.extend(dim_axioms)

        artifacts['axioms'] = {}
        artifacts['axioms']['_clusters'] = all_axioms
        return artifacts


class TrainingStage(object):
    def __init__(self, stages):
        self.stages = stages

    def train(self, data_set, artifacts):
        axioms = []

        for stage, config in self.stages:
            axioms += stage(config).train(data_set, artifacts.deep_copy())["axioms"]
        artifacts["axioms"] = axioms
        return artifacts
