import pickle

from axiomatic.base import TrainingPipeline
from axiomatic.axiom_training_stage import KMeansClusteringAxiomStage
from axiomatic.genetic_classifier_training_stage import GeneticClassifierTrainingStage
from axiomatic.objective_function import Accuracy
from axiomatic.neighbors_classifier import CustomKNearestNeighborsClassifier


with open('datasets/classification_1d_test_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

clustering_axiom_stage = KMeansClusteringAxiomStage(
    {
        'clustering_params': {'n_clusters': 15, 'init': 'k-means++', 'n_init': 10},
        'feature_extraction_params': {'sample_length': 15, 'ratio': 0.3},
    }
)

# stage for constructing axiom system and abnormal models
classifier_training_stage = GeneticClassifierTrainingStage(
    config={
        'population_size': 10,
        'initial_as_size': 6,
        'objective_function': Accuracy(),
        'recognizer': CustomKNearestNeighborsClassifier,
        'recognizer_config': {'n_neighbors': 2},
    }
)

training_pipeline = TrainingPipeline([clustering_axiom_stage, classifier_training_stage])

artifacts = training_pipeline.train(dataset)

print "Artifacts after training: ", artifacts

classifier = CustomKNearestNeighborsClassifier(artifacts['axiom_system'], artifacts['abn_models'], {'n_neighbors': 2})

# count accuracy
obj_fn = Accuracy()
obj_fn_value = obj_fn.calculate(classifier, dataset['test'])

print "Accuracy: ", obj_fn_value
