import pickle
import time
import argparse

from axiomatic.base import TrainingPipeline
from axiomatic.axiom_training_stage import KMeansClusteringAxiomStage
from axiomatic.genetic_classifier_training_stage import GeneticClassifierTrainingStage
from axiomatic.objective_function import Accuracy
from axiomatic.neighbors_classifier import TimeSeriesKNearestNeighborsClassifier
from axiomatic.features import (
    Maximum,
    Minimum,
    Mean,
    StdDeviation,
    LinearRegressionCoef,
    Kurtosis,
    Skewness,
)


default_config = {
    'clustering_config': {
        'clustering_params': {'n_clusters': 15, 'init': 'k-means++', 'n_init': 10},
        'feature_extraction_params': {
            'sample_length': 10,
            'ratio': 0.3,
            'features': [Maximum(), Minimum(), Mean(), StdDeviation(), Kurtosis(), Skewness(), LinearRegressionCoef()],
        },
    },
    'genetic_config': {
        'population_size': 50,
        'iteration_without_improvement': 10,
        'initial_as_size': 6,
        'n_models_for_class': 10,
        'objective_function': Accuracy(),
        'recognizer': TimeSeriesKNearestNeighborsClassifier,
        'recognizer_config': {'n_neighbors': 3},
        'iteration_count': 50,
    },
    'classifier_config': {
        'n_neighbors': 3,
    },
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify time series.')
    parser.add_argument('--name', type=str)

    args = parser.parse_args()

    with open('datasets/{name}_dataset.pickle'.format(name=args.name), 'rb') as f_dataset:
        dataset = pickle.load(f_dataset)

        before_training = time.time()
        print 'Training started', before_training

        # learn model
        clustering_axiom_stage = KMeansClusteringAxiomStage(default_config['clustering_config'])

        # stage for constructing axiom system and abnormal models
        classifier_training_stage = GeneticClassifierTrainingStage(config=default_config['genetic_config'])

        training_pipeline = TrainingPipeline([clustering_axiom_stage, classifier_training_stage])
        artifacts = training_pipeline.train(dataset)

        classifier = TimeSeriesKNearestNeighborsClassifier(
            artifacts['axiom_system'],
            artifacts['abn_models'],
            default_config['classifier_config'],
        )

        # count accuracy, note that we count not actual accuracy, but the ratio of wrong classified objects
        obj_fn = Accuracy()
        before_classifying = time.time()
        print 'Classifiying started', before_classifying

        obj_fn_value = obj_fn.calculate(classifier, dataset['test'])

        print "Accuracy: ", 1 - obj_fn_value
        after_classifying = time.time()

        print 'Classifying ended', after_classifying
        print
