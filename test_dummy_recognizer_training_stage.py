import pandas as pd
import numpy as np
import pickle

from axiomatic.base import AxiomSystem, TrainingPipeline
from axiomatic.axiom_training_stage import DummyAxiomTrainingStage
from axiomatic.recognizer_training_stage import DummyRecognizerTrainingStage
from axiomatic.objective_function import ObjectiveFunction
from axiomatic.abnormal_behavior_recognizer import AbnormalBehaviorRecognizer

with open('datasets/debug_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

dummy_axiom_stage = DummyAxiomTrainingStage()
dummy_recognizer_stage = DummyRecognizerTrainingStage()

training_pipeline = TrainingPipeline([dummy_axiom_stage, dummy_recognizer_stage])

artifacts = training_pipeline.train(dataset)

print "Artifacts after training: ", artifacts

recognizer = AbnormalBehaviorRecognizer(artifacts['axiom_system'], artifacts['abn_models'],
                                        dict(bound=0.1,maxdelta=0.5))

obj_fn = ObjectiveFunction(1, 20)

obj_fn_value = obj_fn.calculate(recognizer, dataset['test'])

print "Recognizer objective function: ", obj_fn_value