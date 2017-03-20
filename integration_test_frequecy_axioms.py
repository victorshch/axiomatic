import pandas as pd
import numpy as np
import pickle

from axiomatic.base import AxiomSystem, MinMaxAxiom, IntegralAxiom, TrainingPipeline
from axiomatic.axiom_training_stage import FrequencyECTrainingStage, FrequencyAxiomTrainingStage
from axiomatic.recognizer_training_stage import DummyRecognizerTrainingStage
from axiomatic.objective_function import ObjectiveFunction

with open('datasets/debug_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

frequency_ec_stage = FrequencyECTrainingStage({'num_part': 5, 'max_window': 5, 'num_axioms': 10, 'axiom_list': [MinMaxAxiom, IntegralAxiom]})
frequency_axiom_stage = FrequencyAxiomTrainingStage({'num_axioms': 10, 'max_depth': 5, 'num_step_axioms': 10})
dummy_recognizer_stage = DummyRecognizerTrainingStage()

training_pipeline = TrainingPipeline([frequency_ec_stage, frequency_axiom_stage, dummy_recognizer_stage])

artifacts = training_pipeline.train(dataset)

print "Artifacts after training: ", artifacts

recognizer = AbnormalBehaviorRecognizer(artifacts['axiom_system'], artifacts['abn_models'],
                                        dict(bound=0.1,maxdelta=0.5))

obj_fn = ObjectiveFunction(1, 20)

obj_fn_value = obj_fn.calculate(recognizer, dataset['test'])

print "Recognizer objective function: ", obj_fn_value

