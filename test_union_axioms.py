import pandas as pd
import numpy as np
import pickle

from axiomatic.base import AxiomSystem, TrainingPipeline
from axiomatic.axiom_training_stage import *
from axiomatic.genetic_recognizer_training_stage import GeneticRecognizerTrainingStage
from axiomatic.elementary_conditions import *
from axiomatic.objective_function import ObjectiveFunction
from axiomatic.abnormal_behavior_recognizer import AbnormalBehaviorRecognizer

with open('dataset_1_1.pickle', 'rb') as f:
    dataset = pickle.load(f)

axiom_list = [MinMaxAxiom, MaxAxiom, MinAxiom, ChangeAxiom, IntegralAxiom, RelativeChangeAxiom, FirstDiffAxiom, SecondDiffAxiom]
frequency_ec_stage = FrequencyECTrainingStage({'num_part': 50, 'left_window': 5, 'right_window': 5, 'num_axioms': 20, 'axiom_list': axiom_list, 'enable_cache': True})
frequency_axiom_stage = FrequencyAxiomTrainingStage({'num_axioms': 20, 'max_depth': 10, 'num_step_axioms': 5})

clustering_axiom_stage = KMeansClusteringAxiomStage({'clustering_params': {'n_clusters': 5,
        'init': 'k-means++', 'n_init': 10}, 'feature_extraction_params': {'sample_length': 20, 'ratio': 0.2}})

genetic_recognizer_stage = GeneticRecognizerTrainingStage(dict(n_jobs=1, population_size=30, iteration_count=30, use_question_mark=False, num_axioms_weight=0.1))

frequency_stage = TrainingPipeline([frequency_ec_stage, frequency_axiom_stage])
axiom_union_stage = AxiomUnionStage([frequency_stage, clustering_axiom_stage])
training_pipeline = TrainingPipeline([clustering_axiom_stage, genetic_recognizer_stage])

artifacts = training_pipeline.train(dataset, dict())

print "Artifacts after training: ", artifacts

recognizer = AbnormalBehaviorRecognizer(artifacts['axiom_system'], artifacts['abn_models'],
                                        dict(bound=0.1,maxdelta=0.5))

obj_fn = ObjectiveFunction(1, 20)

obj_fn_value = obj_fn.calculate(recognizer, dataset['test'])

print "Recognizer objective function: ", obj_fn_value
