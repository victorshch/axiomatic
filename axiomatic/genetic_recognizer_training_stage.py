# -*- coding: utf-8 -*-

import copy
import random
import numpy as np
from joblib import Parallel, delayed

from base import AxiomSystem, QuestionMarkSymbol
from abnormal_behavior_recognizer import AbnormalBehaviorRecognizer
from objective_function import ObjectiveFunction


def index_of(lst, predicate_or_num):
    """
    Return index of element from lst for which predicate_or_num(elem) == True
    if predicate_or_num is callable or elem == predicate_or_num otherwise.
    """
    if not callable(predicate_or_num):
        def f(x):
            return x == predicate_or_num
    else:
        f = predicate_or_num
    for i, elem in enumerate(lst):
        if f(elem):
            return i
    # for debug
    return -3


def last_index_of(lst, func):
    i = index_of(lst[::-1], func)
    if i < 0:
        return i
    return len(lst) - i - 1


class Specimen(object):
    def __init__(self, axiom_list, abn_models):
        self.axiom_list = axiom_list
        self.abn_models = abn_models
        self.objective = None
        self.fitness = None

    def __copy__(self):
        """
        Custom copy for Specimen.
        We need to deepcopy only abnormal models.
        """
        copied_specimen = Specimen(
            axiom_list=copy.copy(self.axiom_list),
            abn_models=copy.deepcopy(self.abn_models)
        )

        return copied_specimen

    def add_axiom(self, new_axiom, priority=-1):
        """
        Update all abn_models
        @param new_axiom:
        @param priority: priority of the axiom
        @return: index of new axiom
        """
        if priority == -1:
            pos = np.random.randint(len(self.axiom_list))
        else:
            pos = priority
        
        self.axiom_list.insert(pos, new_axiom)
        
        for abn_model in self.abn_models.itervalues():
            for i in xrange(len(abn_model)):
                if abn_model[i] >= pos:
                    abn_model[i] += 1
        
        return pos
    
    def check_add_axiom(self, new_axiom, priority=-1):
        for i in xrange(len(self.axiom_list)):
            if new_axiom == self.axiom_list[i]:
                return i

        return self.add_axiom(new_axiom, priority)
    
    def remove_axiom(self, axiom_index_to_remove):
        if len(self.axiom_list) <= 1:
            return
        
        for abn_model in self.abn_models.itervalues():
            for i in xrange(len(abn_model)):
                if abn_model[i] == axiom_index_to_remove:
                    abn_model[i] = -1
                elif abn_model[i] > axiom_index_to_remove:
                    abn_model[i] -= 1
        
        del self.axiom_list[axiom_index_to_remove]
    
    def check_remove_axiom(self, axiom_index_to_remove):
        for abn_model in self.abn_models.itervalues():
            for axiom_index in abn_model:
                if axiom_index == axiom_index_to_remove:
                    return
        self.remove_axiom(axiom_index_to_remove)
    
    def make_list_with_axioms(self, cl):
        marking = self.abn_models[cl]
        return [self.axiom_list[i] if i >= 0 else i for i in marking]

    def make_marking(self, axiom_list_with_numbers):
        return [(index_of(self.axiom_list, a) if not isinstance(a, int) else a) for a in axiom_list_with_numbers]
    
    def __repr__(self):
        return "Specimen(axioms:" + repr(self.axiom_list) + ", abn_models" + repr(self.abn_models) + ")"


class MutationActions(object):
    ReplaceWithOwn = 0
    ReplaceWithForeign = 1
    AddOwn = 2
    AddForeign = 3
    Remove = 4
    AddQuestionMarkSymbol = 5
    ActionCount = 6


class Mutation(object):
    def __init__(self, use_question_mark=True):
        self.use_question_mark = use_question_mark
    
    def mutate(self, specimen, axioms):
        """
        Inplace mutation
        """
        # see copy method in Specimen
        # specimen.objective = None
        # specimen.abn_models = copy.deepcopy(specimen.abn_models)
        # specimen.axiom_list = copy.copy(specimen.axiom_list)

        model_to_mutate = random.choice(specimen.abn_models.itervalues())

        last_action = MutationActions.AddQuestionMarkSymbol if self.use_question_mark else MutationActions.Remove
        action = np.random.randint(last_action + 1)
        
        if action == MutationActions.ReplaceWithOwn:
            pos = np.random.randint(len(model_to_mutate))
            model_to_mutate[pos] = np.random.randint(len(specimen.axiom_list))
        elif action == MutationActions.ReplaceWithForeign:
            new_axiom = np.random.choice(axioms)
            new_axiom_index = specimen.add_axiom(new_axiom)
            pos = np.random.randint(len(model_to_mutate))
            model_to_mutate[pos] = new_axiom_index
        elif action == MutationActions.AddOwn:
            pos = np.random.randint(len(model_to_mutate))
            new_axiom_index = np.random.randint(len(specimen.axiom_list))
            model_to_mutate.insert(pos, new_axiom_index)
        elif action == MutationActions.AddForeign:
            pos = np.random.randint(len(model_to_mutate))
            new_axiom = np.random.choice(axioms)
            new_axiom_index = specimen.add_axiom(new_axiom)
            model_to_mutate.insert(pos, new_axiom_index)
        elif action == MutationActions.Remove:
            if len(model_to_mutate) > 1:
                pos = np.random.randint(len(model_to_mutate))
                axiom_index_to_remove = model_to_mutate[pos]
                del model_to_mutate[pos]
                specimen.check_remove_axiom(axiom_index_to_remove)
        elif action == MutationActions.AddQuestionMarkSymbol:
            pos = np.random.randint(len(model_to_mutate))
            model_to_mutate.insert(pos, QuestionMarkSymbol)
        else:
            print "Specimen.mutate() unknown action", action
            
        return specimen


class Crossover(object):
    @staticmethod
    def one_point_crossover(l1, l2):
        """
        @param l1: first abnormal model
        @param l2: second abnormal model
        """
        p1 = np.random.randint(len(l1) + 1)
        
        # we need to choose min and max points to exclude empty models
        min_point_2 = 0 if p1 < len(l1) else 1
        max_point_2 = len(l2) if p1 > 0 else len(l2) - 1
        
        if min_point_2 == max_point_2:
            p2 = max_point_2
        else:
            p2 = np.random.randint(min_point_2, max_point_2 + 1)
        
        result = (l1[:p1], l2[p2:]), (l2[:p2], l1[p1:])
        # print "One-point crossover for ", l1, l2, "points:", p1, p2, " result:", result

        return result
    
    def crossover(self, specimen1, specimen2):
        class_to_mutate = np.random.choice(specimen1.abn_models.iterkeys())
        # print "Class to mutate: ", class_to_mutate
        # print "Parents: ", s1, s2
        
        offspring1 = copy.copy(specimen1)
        offspring2 = copy.copy(specimen2)

        # see __copy__ for specimen
        # offspring1.abn_models = copy.deepcopy(offspring1.abn_models)
        # offspring1.axiom_list = copy.copy(offspring1.axiom_list)
        # offspring2.abn_models = copy.deepcopy(offspring2.abn_models)
        # offspring2.axiom_list = copy.copy(offspring2.axiom_list)
        
        m1_axioms = offspring1.make_list_with_axioms(class_to_mutate)
        m2_axioms = offspring2.make_list_with_axioms(class_to_mutate)
        
        offspring1_m, offspring2_m = self.one_point_crossover(m1_axioms, m2_axioms)
        
        for a in offspring1_m[1]:
            if not isinstance(a, int):
                offspring1.check_add_axiom(a, len(offspring1.axiom_list))
        for a in offspring2_m[1]: 
            if not isinstance(a, int):
                offspring2.check_add_axiom(a, len(offspring2.axiom_list))
        
        offspring1.abn_models[class_to_mutate] = offspring1.make_marking(offspring1_m[0] + offspring1_m[1])
        offspring2.abn_models[class_to_mutate] = offspring2.make_marking(offspring2_m[0] + offspring2_m[1])
        
        # for a in offspring1_m[0]: offspring1.check_remove_axiom(a)
        # for a in offspring2_m[1]: offspring2.check_remove_axiom(a)
        # print "Offsprings: ", offspring1, offspring2
        
        return offspring1, offspring2


def _calculate_objective_for_specimen(self, specimen, data_set):
    # DEBUG
    # err1 = np.random.randint(1, 20)
    # err2 = np.random.randint(1, 20)
    # specimen.objective = (err1 + 20*err2, err1, err2)
    # return
    if specimen.objective is not None:
        return specimen

    objective = self.objective_function.calculate(
        self.recognizer(AxiomSystem(specimen.axiom_list), specimen.abn_models, self.recognizer_config),
        data_set,
    )

    # different cases
    if isinstance(objective, tuple):
        specimen.objective = (objective[0] + self.num_axioms_weight * len(specimen.axiom_list), objective[1], objective[2])
    else:
        specimen.objective = objective + self.num_axioms_weight * len(specimen.axiom_list)


class GeneticRecognizerTrainingStage(object):
    def __init__(self, config={}):
        self.iteration_count = config.get('iteration_count', 10)
        self.population_size = config.get('population_size', 100)
        self.elitism = config.get('elitism', 0.05)
        self.selective_pressure = config.get('selective_pressure', 1.1)
        self.initial_as_size = config.get('initial_as_size', 5)
        self.initial_model_length = config.get('initial_model_length', 10)
        self.use_question_mark = config.get('use_question_mark', True)
        self.n_jobs = config.get('n_jobs', 2)
        # regularization
        self.num_axioms_weight = config.get('num_axioms_weight', 0.0)

        self.objective_function = config.get('objective_function', ObjectiveFunction(1, 20))
        self.recognizer = config.get('recognizer', AbnormalBehaviorRecognizer)
        self.recognizer_config = config.get('recognizer_config', dict(bound=0.1, maxdelta=0.5))

    def generate_initial_population(self, axioms, data_set):
        population = []
        for i in xrange(self.population_size):
            axiom_list = list(np.random.choice(axioms, size=self.initial_as_size, replace=False))
            axiom_system = AxiomSystem(axiom_list)
            abn_models = {}
            for cl in data_set:
                if cl == 'normal':
                    continue
                cl_ts = random.choice(data_set[cl])
                marking = axiom_system.perform_marking(cl_ts)

                first_axiom_index = index_of(marking, lambda x: x > 0)
                if first_axiom_index < 0:
                    marking = [np.random.randint(len(axiom_list)) for i in xrange(self.initial_model_length)]
                else:
                    last_axiom_index = last_index_of(marking, lambda x: x > 0)
                    marking = marking[first_axiom_index:last_axiom_index + 1]

                abn_models[cl] = marking
            population.append(Specimen(axiom_list, abn_models))
        return population
    
    def calculate_objective_for_population(self, population, data_set):
        # TODO parallelize
        # for more elegant solution see http://qingkaikong.blogspot.ru/2016/12/python-parallel-method-in-class.html
        result = Parallel(n_jobs=self.n_jobs)(delayed(_calculate_objective_for_specimen)(self, specimen, data_set) for specimen in population)
        # for i, specimen in enumerate(population):
        #     if specimen.objective is not None: continue
        #     specimen.objective = self.objective_function.calculate(
        #         self.recognizer(AxiomSystem(specimen.axiom_list), specimen.abn_models, self.recognizer_config),
        #         data_set,
        #     )[0]
        return result

    def linear_rank_fitness(self, population):
        a = 2.0 * (self.selective_pressure - 1) / (len(population) - 1)
        for i in xrange(len(population)):
            population[i].fitness = self.selective_pressure - a * i        

    def roulette(self, population):
        # TODO rewrite using cool random.choice from numpy
        # def weighted_random_choice(chromosomes):
        #     max = sum(chromosome.fitness for chromosome in chromosomes)
        #
        #     pick = np.random.uniform(0, max)
        #     current = 0
        #     for chromosome in chromosomes:
        #         current += chromosome.fitness
        #         if current > pick:
        #             return chromosome
        #
        # return [weighted_random_choice(population) for i in xrange(self.population_size)]

        probabilities = np.array([specimen.fitness for specimen in population], dtype=float)
        probabilities /= np.sum(probabilities)

        return list(np.random.choice(
            population,
            size=self.population_size,
            replace=False,
            p=probabilities,
        ))

    def select(self, population):
        new_population = []
        population.sort(key=lambda s: s.objective[0])

        print "Best objective function", population[0].objective
        print "Worst objective function", population[-1].objective

        elite_count = int(round(self.elitism * self.population_size))
        new_population.extend(population[:elite_count])
        population = population[elite_count:]
        self.linear_rank_fitness(population)
        new_population.extend(self.roulette(population))
        return new_population

    def train(self, data_set, artifacts):
        axioms = [a for cl in artifacts['axioms'] for a in artifacts['axioms'][cl]]
        # print "Initial axioms: ", axioms
        train_data_set = data_set['train']
        val_data_set = data_set['validate']
                
        # 1. generate initial population
        # 2. mutation
        # 3. crossover
        # 4. selection
        
        population = self.generate_initial_population(axioms, train_data_set)
        
        print "Calculating objective function for initial population..."
        
        self.calculate_objective_for_population(population, val_data_set)
        
        mutation = Mutation(self.use_question_mark)
        crossover = Crossover()
        
        for i in xrange(self.iteration_count):
            print "Iteration ", i + 1, "of", self.iteration_count
            mutated = [mutation.mutate(copy.copy(specimen), axioms) for specimen in population]
            population.extend(mutated)
            offspring = []
            print "Crossing..."
            for s1_index, s2_index in zip(np.random.randint(0, 2 * self.population_size, self.population_size),
                                          np.random.randint(0, 2 * self.population_size, self.population_size)):
                new_specimen1, new_specimen2 = crossover.crossover(population[s1_index], population[s2_index])
                offspring.append(new_specimen1)
                offspring.append(new_specimen2)
            population.extend(offspring)
            print "Calculating objective function for extended population..."
            self.calculate_objective_for_population(population, val_data_set)
            
            print "Selecting..."
            population = self.select(population)
            # TODO stopping criterion

        artifacts['axiom_system'] = AxiomSystem(population[0].axiom_list)
        artifacts['abn_models'] = population[0].abn_models
        return artifacts
