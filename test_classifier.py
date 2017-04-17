import pickle

from axiomatic.base import TrainingPipeline
from axiomatic.axiom_training_stage import KMeansClusteringAxiomStage
from axiomatic.genetic_recognizer_training_stage import GeneticRecognizerTrainingStage
from axiomatic.objective_function import Accuracy
from axiomatic.ts_classifier import TimeSeriesClassifier


with open('datasets/debug_classifier_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

clustering_axiom_stage = KMeansClusteringAxiomStage(
    {
        'clustering_params': {'n_clusters': 15, 'init': 'k-means++', 'n_init': 10},
        'feature_extraction_params': {'sample_length': 20, 'ratio': 0.2},
    }
)

# stage for constructing axiom system and abnormal models
classifier_training_stage = GeneticRecognizerTrainingStage(
    config={
        'population_size': 50,
        'initial_as_size': 8,
        'objective_function': Accuracy(),
        'recognizer': TimeSeriesClassifier,
        'recognizer_config': {},
    }
)

training_pipeline = TrainingPipeline([clustering_axiom_stage, classifier_training_stage])

artifacts = training_pipeline.train(dataset)

print "Artifacts after training: ", artifacts

classifier = TimeSeriesClassifier(artifacts['axiom_system'], artifacts['abn_models'])

# count accuracy
obj_fn = Accuracy()
obj_fn_value = obj_fn.calculate(classifier, dataset['test'])

print "Accuracy: ", obj_fn_value
