# -*- coding: utf-8 -*-

import copy
import random
import warnings

import numpy as np

from base import AxiomSystem
from objective_function import Accuracy
from neighbors_classifier import TimeSeriesKNearestNeighborsClassifier


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
        @param new_axiom: axiom to add
        @param priority: priority of the axiom
        @return: index of new axiom
        """
        if priority == -1:
            pos = np.random.randint(len(self.axiom_list) + 1)
        else:
            pos = priority

        self.axiom_list.insert(pos, new_axiom)

    def remove_axiom(self, axiom_index_to_remove):
        if len(self.axiom_list) <= 1:
            return

        del self.axiom_list[axiom_index_to_remove]

    def change_axiom_priority(self, axiom_index_to_change, new_priority):
        axiom_to_move = self.axiom_list[axiom_index_to_change]
        del self.axiom_list[axiom_index_to_change]
        self.axiom_list.insert(new_priority, axiom_to_move)

    def switch_axioms(self, first_axiom, second_axiom):
        tmp_axiom = self.axiom_list[first_axiom]
        self.axiom_list[first_axiom] = self.axiom_list[second_axiom]
        self.axiom_list[second_axiom] = tmp_axiom

    def replace_axiom(self, axiom_index_to_replace, new_axiom):
        self.axiom_list[axiom_index_to_replace] = new_axiom

    def add_model_for_class(self, model, class_to_change):
        self.abn_models[class_to_change].append(model)

    def remove_model_for_class(self, class_to_change):
        n_abn_models = len(self.abn_models[class_to_change])
        if n_abn_models > 1:
            model_to_delete = np.random.randint(n_abn_models)
            del self.abn_models[class_to_change][model_to_delete]

    def __repr__(self):
        return "Specimen(axioms:" + repr(self.axiom_list) + ", abn_models" + repr(self.abn_models) + ")"


class MutationActions(object):
    AddAxiom = 0                  # add axiom not present in axiom system
    RemoveAxiom = 1               # remove axiom from axiom system
    ChangeAxiomPriority = 2       # change the priority of axiom in axiom system
    SwitchAxioms = 3              # switch two axioms in axiom system
    ReplaceWithForeignAxiom = 4   # replace axiom in axiom system with random axiom

    AddNewAbnormalModel = 5       # add new abnormal model for class
    RemoveAbnormalModel = 6       # remove random abnormal for class

    ActionCount = 7


class Mutation(object):
    @staticmethod
    def mutate(specimen, axioms, data_set):
        """
        Inplace mutation
        """
        action = np.random.randint(MutationActions.ActionCount)
        present_axioms = set(specimen.axiom_list)
        foreign_axioms = list(set(axioms).difference(set(present_axioms)))

        if action == MutationActions.AddAxiom:
            priority = np.random.randint(len(specimen.axiom_list) + 1)
            axiom_to_add = random.choice(foreign_axioms)
            specimen.add_axiom(axiom_to_add, priority)
        elif action == MutationActions.RemoveAxiom:
            if len(specimen.axiom_list) > 1:
                axiom_index_to_remove = np.random.randint(len(specimen.axiom_list))
                specimen.remove_axiom(axiom_index_to_remove)
        elif action == MutationActions.ChangeAxiomPriority:
            axiom_index_to_change = np.random.randint(len(specimen.axiom_list))
            new_priority = np.random.randint(len(specimen.axiom_list))
            specimen.change_axiom_priority(axiom_index_to_change, new_priority)
        elif action == MutationActions.SwitchAxioms:
            first_axiom = np.random.randint(len(specimen.axiom_list))
            second_axiom = np.random.randint(len(specimen.axiom_list))
            specimen.switch_axioms(first_axiom, second_axiom)
        elif action == MutationActions.ReplaceWithForeignAxiom:
            axiom_index_to_replace = np.random.randint(len(specimen.axiom_list))
            new_axiom = random.choice(foreign_axioms)
            specimen.replace_axiom(axiom_index_to_replace, new_axiom)
        elif action == MutationActions.AddNewAbnormalModel:
            class_to_change = np.random.choice(specimen.abn_models.keys())
            cl_ts = random.choice(data_set[class_to_change])
            marking = AxiomSystem(specimen.axiom_list).perform_marking(cl_ts)
            if np.any(marking >= 0):
                specimen.add_model_for_class(cl_ts, class_to_change)
        elif action == MutationActions.RemoveAbnormalModel:
            class_to_change = np.random.choice(specimen.abn_models.keys())
            specimen.remove_model_for_class(class_to_change)
        else:
            print "Specimen.mutate() unknown action", action

        return specimen


class Crossover(object):
    def split_models(self, models, n):
        random.shuffle(models)
        return models[:n], models[n:]

    def crossover(self, specimen1, specimen2):
        class_to_crossover = np.random.choice(specimen1.abn_models.keys())

        offspring1 = copy.copy(specimen1)
        offspring2 = copy.copy(specimen2)

        abn_models_1 = offspring1.abn_models[class_to_crossover]
        abn_models_2 = offspring2.abn_models[class_to_crossover]

        n_models_to_switch_1 = np.random.randint(1, len(abn_models_1) + 1)
        n_models_to_switch_2 = np.random.randint(1, len(abn_models_2) + 1)

        models_to_switch_1, models_to_leave_1 = self.split_models(abn_models_1, n_models_to_switch_1)
        models_to_switch_2, models_to_leave_2 = self.split_models(abn_models_2, n_models_to_switch_2)

        offspring1.abn_models[class_to_crossover] = models_to_switch_2 + models_to_leave_1
        offspring2.abn_models[class_to_crossover] = models_to_switch_1 + models_to_leave_2

        return offspring1, offspring2


class GeneticClassifierTrainingStage(object):
    def __init__(self, config={}):
        self.iteration_count = config.get('iteration_count', 50)
        self.iteration_without_improvement = config.get('iteration_without_improvement', 5)
        self.population_size = config.get('population_size', 50)
        self.elitism = config.get('elitism', 0.05)
        self.selective_pressure = config.get('selective_pressure', 1.1)
        self.initial_as_size = config.get('initial_as_size', 10)
        self.n_models_for_class = config.get('n_models_for_class', 7)

        # regularization
        self.num_axioms_weight = config.get('num_axioms_weight', 0.0)

        self.objective_function = config.get('objective_function', Accuracy())
        self.recognizer = config.get('recognizer', TimeSeriesKNearestNeighborsClassifier)
        self.recognizer_config = config.get('recognizer_config', {'n_neighbors': 5})

        self.objectives = []

    def generate_initial_population(self, axioms, data_set):
        population = []

        i = 0
        while i < self.population_size:
            axiom_list = list(np.random.choice(axioms, size=self.initial_as_size, replace=False))
            axiom_system = AxiomSystem(axiom_list)
            abn_models = {}

            for cl in data_set:
                abn_models[cl] = []

                for j in xrange(self.n_models_for_class):
                    cl_ts = random.choice(data_set[cl])
                    marking = axiom_system.perform_marking(cl_ts)

                    if np.any(marking >= 0):
                        abn_models[cl].append(cl_ts)
                    else:
                        warnings.warn('Bad marking ' + repr(marking))

            if all(len(models) > 0 for models in abn_models.itervalues()):
                population.append(Specimen(axiom_list, abn_models))
                i += 1

        return population

    def calculate_objective_for_population(self, population, data_set):
        for i, specimen in enumerate(population):
            if specimen.objective is not None:
                continue

            n_models = 0
            for cl, models in specimen.abn_models.iteritems():
                n_models += len(models)

            if n_models < self.recognizer_config['n_neighbors']:
                objective = 1.0
            else:
                objective = self.objective_function.calculate(
                    self.recognizer(AxiomSystem(specimen.axiom_list), specimen.abn_models, self.recognizer_config),
                    data_set,
                )

            population[i].objective = objective

    def linear_rank_fitness(self, population):
        a = 2.0 * (self.selective_pressure - 1) / (len(population) - 1)
        for i in xrange(len(population)):
            population[i].fitness = self.selective_pressure - a * i

    def roulette(self, population):
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
        if len(population) <= 0:
            return new_population

        population.sort(key=lambda s: s.objective)

        print "Best objective function", population[0].objective
        print "Worst objective function", population[-1].objective

        self.objectives.append(population[0].objective)

        elite_count = int(round(self.elitism * self.population_size))
        new_population.extend(population[:elite_count])
        population = population[elite_count:]
        self.linear_rank_fitness(population)
        new_population.extend(self.roulette(population))
        return new_population

    def train(self, data_set, artifacts):
        axioms = [a for cl in artifacts['axioms'] for a in artifacts['axioms'][cl]]

        train_data_set = data_set['train']
        val_data_set = data_set['validate']

        # 1. generate initial population
        # 2. mutation
        # 3. crossover
        # 4. selection

        population = self.generate_initial_population(axioms, train_data_set)

        print "Calculating objective function for initial population..."

        self.calculate_objective_for_population(population, val_data_set)

        mutation = Mutation()
        crossover = Crossover()

        n_iter_without_improvement = 0

        cur_objective = population[0].objective

        for i in xrange(self.iteration_count):
            print "Iteration ", i + 1, "of", self.iteration_count
            mutated = [mutation.mutate(copy.copy(specimen), axioms, train_data_set) for specimen in population]
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

            # stopping criterion
            if self.objectives[-1] < 1e-5 and i > 4:
                break

            if self.objectives[-1] < cur_objective:
                n_iter_without_improvement = 0
            else:
                n_iter_without_improvement += 1
            cur_objective = self.objectives[-1]

            if n_iter_without_improvement > self.iteration_without_improvement:
                print "Iters without improvement", n_iter_without_improvement
                print "Breaking"
                break

        artifacts['axiom_system'] = AxiomSystem(population[0].axiom_list)
        artifacts['abn_models'] = population[0].abn_models
        return artifacts
