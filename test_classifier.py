import pickle

from axiomatic.base import TrainingPipeline
from axiomatic.axiom_training_stage import KMeansClusteringAxiomStage
from axiomatic.recognizer_training_stage import ClassifierRecognizerTrainingStage
from axiomatic.objective_function import Accuracy
from axiomatic.abnormal_behavior_recognizer import Classifier


with open('datasets/debug_classifier_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

clustering_axiom_stage = KMeansClusteringAxiomStage({'clustering_params': {'n_clusters': 15,
        'init': 'k-means++', 'n_init': 10}, 'feature_extraction_params': {'sample_length': 20, 'ratio': 0.2}})

# stage for constructing axiom system and abnormal models
# abn_models_type == 'brute_force' for exhaustive search of models with fixed length
# abn_models_type == 'genetic' for constructing abnormal models using genetic algorithm
classifier_recognizer_stage = ClassifierRecognizerTrainingStage(abn_models_type='brute_force')

training_pipeline = TrainingPipeline([clustering_axiom_stage, classifier_recognizer_stage])

artifacts = training_pipeline.train(dataset)

print "Artifacts after training: ", artifacts

classifier = Classifier(artifacts['axiom_system'], artifacts['abn_models'])

# make list of time series and list of class labels
ts_list, class_labels = transform_dataset(dataset['test'])
predictions = classifier.predict(ts_list)

# count accuracy
obj_fn = Accuracy()
obj_fn_value = obj_fn.calculate(class_labels, predictions)

print "Accuracy: ", obj_fn_value