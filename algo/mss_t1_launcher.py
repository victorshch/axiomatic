from mne.preprocessing.ecg import qrs_detector

from datasets.Arrhythmia.get import load
from datetime import datetime
import sys
import numpy as np
import pandas as pd
from axiomatic.features import *

from axiomatic.base import AxiomSystem, TrainingPipeline
from axiomatic.axiom_training_stage import *
from axiomatic.genetic_recognizer_training_stage import GeneticRecognizerTrainingStage
from axiomatic.elementary_conditions import *
from axiomatic.objective_function import ScoreObjectiveFunction as ObjectiveFunction
from axiomatic.abnormal_behavior_recognizer import AbnormalBehaviorRecognizer
import random


class MssT1Launcher:
  def run(self, feature_list, axiom_list):
    true = load('./datasets/Arrhythmia/training', 'all', 'true', resample = False, ecg_only = True)
    false = load('./datasets/Arrhythmia/training', 'all', 'false', resample = False, ecg_only = True)

    FEATURES = {'Leakage' : Leakage, 'BinaryCovariance' : BinaryCovariance, 'BinaryFrequency' : BinaryFrequency, 'AreaBinary' : AreaBinary, 'Complexity' : Complexity, 'Kurtosis' : Kurtosis, 'Count2' : Count2, 'LinearRegressionCoef' : LinearRegressionCoef, 'Skewness' : Skewness, 'Count1' : Count1, 'Count3' : Count3, 'IntegralRatioHigh' : IntegralRatioHigh, 'IntegralRatioLow' : IntegralRatioLow}
    axioms = {'MinMaxAxiom': MinMaxAxiom, 'MaxAxiom': MaxAxiom, 'MinAxiom': MinAxiom, 'ChangeAxiom': ChangeAxiom, 'IntegralAxiom': IntegralAxiom, 'RelativeChangeAxiom': RelativeChangeAxiom, 'FirstDiffAxiom': FirstDiffAxiom, 'SecondDiffAxiom': SecondDiffAxiom}
    
    features = [FEATURES[f] for f in feature_list]
    axiom_list = [axioms[a] for a in axiom_list]

    for pack in tqdm([true, false]):
      for i in tqdm(range(len(pack))):
        now1 = pack[i]['ECG1'].as_matrix().astype('float64')
        now2 = pack[i]['ECG2'].as_matrix().astype('float64')
        peaks = qrs_detector(250, now1)
        print(len(peaks))
        
        table = dict()

        for feature in features:
          table['ecg1_' + feature.__name__] = []
          table['ecg2_' + feature.__name__] = []

        for left, right in zip(peaks[: -1], peaks[1 :]):
          if right - left <= 1:
            continue

          for feature in features:
            feat = feature() if feature.__name__[: -1] != "Count" else feature(right - left, 'nontable')
            table['ecg1_' + feature.__name__].append(feat(np.array([now1[left : right]]))[0][0])
            table['ecg2_' + feature.__name__].append(feat(np.array([now2[left : right]]))[0][0])
        pack[i] = pd.DataFrame(table)

    dataset = {'train' : {'abnormal' : true[: len(true) // 2], 'normal' : false[: len(false) // 2]},
            'test' : {'abnormal' : true[len(true) // 2 : ], 'normal' : false[len(false) // 2 : ]}}

    frequency_ec_stage = FrequencyECTrainingStage({'num_part': 50, 'left_window': 8, 'right_window': 8, 'num_axioms': 20, 'axiom_list': axiom_list, 'enable_cache': True})
    frequency_axiom_stage = FrequencyAxiomTrainingStage({'num_axioms': 20, 'max_depth': 10, 'num_step_axioms': 5})

    genetic_recognizer_stage = GeneticRecognizerTrainingStage(dict(n_jobs=1, population_size=100, iteration_count=10, use_question_mark=False, num_axioms_weight=0.0))

    frequency_stage = TrainingPipeline([frequency_ec_stage, frequency_axiom_stage])
    training_pipeline = TrainingPipeline([frequency_stage, genetic_recognizer_stage])

    artifacts = training_pipeline.train(dataset, dict())

    print "Artifacts after training: ", artifacts

    recognizer = AbnormalBehaviorRecognizer(artifacts['axiom_system'], artifacts['abn_models'],
                                            dict(bound=0.1,maxdelta=0.5))

    obj_fn = ObjectiveFunction()
    test_result = obj_fn.calculate(recognizer, dataset['test'])
    print test_result
    return recognizer
