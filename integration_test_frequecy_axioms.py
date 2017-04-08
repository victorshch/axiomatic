# coding=UTF-8

import pandas as pd
import numpy as np
import pickle

from axiomatic.base import AxiomSystem, MinMaxAxiom, MaxAxiom, MinAxiom, ChangeAxiom, IntegralAxiom
from axiomatic.base import RelativeChangeAxiom, FirstDiffAxiom, SecondDiffAxiom, TrainingPipeline
from axiomatic.axiom_training_stage import FrequencyECTrainingStage, FrequencyAxiomTrainingStage
from axiomatic.recognizer_training_stage import DummyRecognizerTrainingStage
from axiomatic.objective_function import ObjectiveFunction
from axiomatic.abnormal_behavior_recognizer import AbnormalBehaviorRecognizer

with open('datasets/debug_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

axiom_list = [MinMaxAxiom, MaxAxiom, MinAxiom, ChangeAxiom, IntegralAxiom, RelativeChangeAxiom, FirstDiffAxiom, SecondDiffAxiom]
frequency_ec_stage = FrequencyECTrainingStage({'num_part': 5, 'left_window': 2, 'right_window': 2, 'num_axioms': 10, 'axiom_list': axiom_list, 'enable_cache': True})
frequency_axiom_stage = FrequencyAxiomTrainingStage({'num_axioms': 10, 'max_depth': 5, 'num_step_axioms': 10})

dummy_recognizer_stage = DummyRecognizerTrainingStage()

training_pipeline = TrainingPipeline([frequency_ec_stage, frequency_axiom_stage, dummy_recognizer_stage])

artifacts = training_pipeline.train(dataset)

print("Artifacts after training: ", artifacts)

recognizer = AbnormalBehaviorRecognizer(artifacts['axiom_system'], artifacts['abn_models'],
                                        dict(bound=0.1,maxdelta=0.5))

obj_fn = ObjectiveFunction(1, 20)

obj_fn_value = obj_fn.calculate(recognizer, dataset['test'])

print("Recognizer objective function: ", obj_fn_value)
