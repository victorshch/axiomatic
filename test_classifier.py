import pickle
from sklearn.metrics import accuracy_score

from axiomatic.base import TrainingPipeline
from axiomatic.utils import transform_dataset
from axiomatic.axiom_training_stage import KMeansClusteringAxiomStage
from axiomatic.classifier_training_stage import ClassifierTrainingStage
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
classifier_training_stage = ClassifierTrainingStage()

training_pipeline = TrainingPipeline([clustering_axiom_stage, classifier_training_stage])

artifacts = training_pipeline.train(dataset)

print "Artifacts after training: ", artifacts

classifier = TimeSeriesClassifier(artifacts['axiom_system'], artifacts['abn_models'])

# make list of time series and list of class labels
ts_list, class_labels = transform_dataset(dataset['test'])
predictions = classifier.predict(ts_list)

# count accuracy
obj_fn_value = accuracy_score(class_labels, predictions)

print "Accuracy: ", obj_fn_value
