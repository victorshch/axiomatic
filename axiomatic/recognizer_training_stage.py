# -*- coding: utf-8 -*-

import numpy as np

from axiomatic.base import AxiomSystem


class DummyRecognizerTrainingStage(object):
    """
    This training stage selects axioms and abnormal behavior models randomly.
    It can be used for testing
    """
    def __init__(self, axiom_count=5, abn_model_length=5):
        self.axiom_count = axiom_count
        self.abn_model_length = abn_model_length
    
    def train(self, data_set, artifacts):
        """
        Perform training using data_set and axioms in artifacts generated for each class. Axioms
        are taken as artifacts['axioms'][<class_name>].
        Abnormal behavior models are created as a random segment from the marking of a random 
        emergency ts from data_set['test']
        """
        axiom_list = [a for key in artifacts['axioms'].keys() for a in artifacts['axioms'][key]]
        axiom_system = AxiomSystem(np.random.choice(axiom_list, self.axiom_count, replace=False))
        test_data_set = data_set['test']
        model_dict = {}
        for cl, ts_list in test_data_set.items():
            if cl == 'normal': continue
            ts_no = np.random.randint(0, len(ts_list) - 1)
            marking = axiom_system.perform_marking(ts_list[ts_no])
            abn_model_start = np.random.randint(0, len(marking) - self.abn_model_length)
            model_dict[cl] = marking[abn_model_start:abn_model_start + self.abn_model_length]
        artifacts['axiom_system'] = axiom_system
        artifacts['abn_models'] = model_dict
        return artifacts
