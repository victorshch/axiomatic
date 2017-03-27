import pandas as pd
import numpy as np
import pickle

from axiomatic.base import AxiomSystem, MinMaxAxiom, IntegralAxiom, TrainingPipeline
from axiomatic.axiom_training_stage import KMeansClusteringAxiomStage
from axiomatic.recognizer_training_stage import DummyRecognizerTrainingStage
from axiomatic.objective_function import ObjectiveFunction
from axiomatic.abnormal_behavior_recognizer import AbnormalBehaviorRecognizer


with open('datasets/debug_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

clustering_axiom_stage = KMeansClusteringAxiomStage({'clustering_params': {'n_clusters': 15,
        'init': 'k-means++', 'n_init': 10}, 'feature_extraction_params': {'sample_length': 20, 'ratio': 0.2}})
dummy_recognizer_stage = DummyRecognizerTrainingStage()

training_pipeline = TrainingPipeline([clustering_axiom_stage, dummy_recognizer_stage])

artifacts = training_pipeline.train(dataset)

print "Artifacts after training: ", artifacts

recognizer = AbnormalBehaviorRecognizer(artifacts['axiom_system'], artifacts['abn_models'],
                                        dict(bound=0.1, maxdelta=0.5))

obj_fn = ObjectiveFunction(1, 20)

obj_fn_value = obj_fn.calculate(recognizer, dataset['test'])

print "Recognizer objective function: ", obj_fn_value
