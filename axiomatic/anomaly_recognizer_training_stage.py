from base import AxiomSystem
import numpy as np
import random, copy, time

class AxiomsMarking(object):
    '''
    Represents a axioms marking.
    Marking = solution
    '''
    def __init__(self):
        self.axioms = []
        self.obj_f = None
        self.fitness_f = None
        self.num = 0
        self.normal_data_anomaly_score = None
        self.anomaly_data_anomaly_score = None

    def __computeObjF(self, value=0):
        self.obj_f = value

    def Update(self, value=0, normal_score=0, anomaly_score=0):
        '''
        Updates objective function.
        Call it after every changing in axioms
        '''
        self.__computeObjF(value)
        self.normal_data_anomaly_score = normal_score
        self.anomaly_data_anomaly_score = anomaly_score

    def GenerateRandom(self, axioms_set):
        '''
        Generates random solution.
        :param axioms_set: all generated axioms.
        '''
        n = random.randint(1, len(axioms_set))
        self.axioms = random.sample(axioms_set, n) # type(self.axioms)=list
        #self.Update()
   
class GeneticAlgorithm(object):
    def __init__(self, config=dict()):
        print('...init started')
        """
        @param config: config for genetic algorithm, should be dict (e.g. 
        {'genetic_algorithms_params': {'n_individuals': 15, 'mutation_prob_init': 0.7, 'crossover_prob_init': 0.7}, 
         'stop_criterion': {'maxIter': 1000, 'maxIterWithoutChange': 30, 'minObjF': -1},
         'objective_function_algorithm': 'knn',
         'axioms_set': artifacts['axioms']['_clusters'],
         'train_data': dataset['train'], 'test_data': dataset['test']})
         
        type(axioms_set)=list
        type(test_data)=dict
        """

        # config for genetic algorithm
        self.config = config
        self.currentSolution = None
        self.currentBestSolution = None
        self.currentIter = 0
        self.currentIterWithoutChange = 0
        self.population = []
        self.mutPercent = 1.0
        self.crossPercent = 1.0
        self.mutate_prob_matrix = [self.config['genetic_algorithms_params']['mutation_prob_init'] for i in range(self.config['genetic_algorithms_params']['n_individuals'])]
        self.crossover_prob_matrix = [self.config['genetic_algorithms_params']['crossover_prob_init'] for i in range(self.config['genetic_algorithms_params']['n_individuals'])]
        self.selective_pressure = 1.5
        self.elitism = 0.05
        #self.mutProb = 0.7 #[random.uniform(0.1, 0.9) for j in range(0, self.algconf.popNum)]
        #self.crossProb = 0.7 #[random.uniform(0.1, 0.9) for j in range(0, self.algconf.popNum)]
        print('\t init done...')
        
    def Step(self):
        print('...step started')
        self._select()
        self._crossover()
        self._mutate()
        self._evalPopulation()
        self.currentIter += 1
        print('\t step done...')
        
    def Run(self):
        print('...run started')
        self.Clear()
        
        print('...create population started')
        for i in range(self.config['genetic_algorithms_params']['n_individuals']):
            s = AxiomsMarking()
            s.GenerateRandom(self.config['axioms_set'])
            value, normal_score, anomaly_score = self._knn_objective_function(self._ts_transform(s.axioms))
            s.Update(value, normal_score, anomaly_score)
            self.population.append(s)
        self.population.sort(key = lambda x: x.obj_f, reverse = False)
        print('\t create population done...')
        
        print('pop:', len(self.population))
        
        self.currentBestSolution = copy.deepcopy(self.population[0])
        
        while not self._checkStopCondition():
            self.Step()
            print('Iteration =', self.currentIter ,'; Best score =', '%.10f' % self.currentBestSolution.obj_f, self.currentBestSolution)
            
        print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
        print("Best solution (found on", self.currentIter, "iteration):", self.currentBestSolution)
        print("--------------------------------------")
        #Algorithm.time = time.time() - Algorithm.time
        #self.stat.AddExecution(Execution(copy.deepcopy(self.currentSolution), self.currentIter, Algorithm.time, Algorithm.timecounts, Algorithm.simcounts))
        print('\t run done...')
        
    def Clear(self):
        print('...clear started')
        self.currentSolution = None
        self.currentBestSolution = None
        self.currentIter = 0
        self.currentIterWithoutChange = 0
        self.population = []
        print('\t clear done...')
        
    def _ts_transform(self, axioms):
        print('...ts_transform started')
        axiom_system = AxiomSystem(axioms)
        #vfunc = np.vectorize(lambda ts: axiom_system.perform_marking(ts))
        
        
        train_normal_data_marked = []
        for ts in self.config['train_data']['normal']:
            ts_marking = axiom_system.perform_marking(ts)
            train_normal_data_marked.append(ts_marking)
        
        #train_normal_data_marked = vfunc(self.config['train_data']['normal'])
    
        
        test_normal_data_marked = []
        for ts in self.config['test_data']['normal']:
            ts_marking = axiom_system.perform_marking(ts)
            test_normal_data_marked.append(ts_marking)
        
        #test_normal_data_marked = self.config['test_data']['normal'].apply(lambda ts: axiom_system.perform_marking(ts))
        
        
        test_anomaly_data_marked = []
        for ts in self.config['test_data']['anomaly']:
            ts_marking = axiom_system.perform_marking(ts)
            test_anomaly_data_marked.append(ts_marking)
        
        #test_anomaly_data_marked = self.config['test_data']['anomaly'].apply(lambda ts: axiom_system.perform_marking(ts))
        
        data = {'train': train_normal_data_marked, 'normal': test_normal_data_marked, 'anomaly': test_anomaly_data_marked}
        
        print('\t ts_transform done...')
        return data
        
    def _LCS(self, s1, s2):
        #print('...LCS started')
        m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
        longest, x_longest = 0, 0
        for x in range(1, 1 + len(s1)):
            for y in range(1, 1 + len(s2)):
                if s1[x - 1] == s2[y - 1]:
                    m[x][y] = m[x - 1][y - 1] + 1
                    if m[x][y] > longest:
                        longest = m[x][y]
                        x_longest = x
                else:
                    m[x][y] = 0
        
        #print('\t LCS done...')
        return len(s1[x_longest - longest: x_longest])
        
    def _knn_objective_function(self, data, k=5):
        print('...knn_obj_f started')
        normal_data_anomaly_score = 0
        for ts_test in data['normal']:
            anomaly_scores = []
            for ts_train in data['train']:
                anomaly_score = 1 - self._LCS(ts_test, ts_train) / float(len(ts_test))
                anomaly_scores.append(anomaly_score)
            anomaly_scores.sort(reverse = True)
            normal_data_anomaly_score += anomaly_scores[k-1]
                
        anomaly_data_anomaly_score = 0
        for ts_test in data['anomaly']:
            anomaly_scores = []
            for ts_train in data['train']:
                anomaly_score = 1 - self._LCS(ts_test, ts_train) / float(len(ts_test))
                anomaly_scores.append(anomaly_score)
            anomaly_scores.sort(reverse = True)
            anomaly_data_anomaly_score += anomaly_scores[k-1]
        
        print('\t knn_obj_f done...')
        return normal_data_anomaly_score - anomaly_data_anomaly_score, normal_data_anomaly_score, anomaly_data_anomaly_score
        
    def adaptate_probability(self, prob, x, y):
        prob = prob * x / float(y)
        
        if prob < 0.1:
            prob = 0.1
        elif prob > 0.9:
            prob = 0.9
            
        return prob
        
    def _mutate(self):
        print('...mutate started')
        '''
        if len(self.population) > 0:
            pop_i = random.randint(0, len(self.population)-1)
        else:
            pop_i = None
        mutation_type = random.randint(0,4)
        '''
        
        sp = c = int((1.0-self.mutPercent) * len(self.population))
        #if self.corrMode == 10:
        #    old_popul = copy.deepcopy(self.population)
        for s in self.population[sp:]:   
            if random.random() <= self.mutate_prob_matrix[c]:
            #if random.random() <= self.config['genetic_algorithms_params']['mutation_prob_init']: #mutProb[c]:
                mutation_type = random.randint(0,4)
                
                #old = copy.deepcopy(s)
                #delta_old = np.std([s_.obj_f for s_ in self.population]) / (np.mean([s_.obj_f for s_ in self.population])+0.00001)
                f_old = s.anomaly_data_anomaly_score / float(s.normal_data_anomaly_score+0.001)
                
                if mutation_type == 0:
                    # _mutate_0: random shuffle
                    print('_mutate_0')
                    random.shuffle(s.axioms)
                elif mutation_type == 1:       
                    # _mutate_1: swap two random elements
                    print('_mutate_1')
                    if len(s.axioms) > 1:
                        i, j = random.sample(range(len(s.axioms)), 2)
                        s.axioms[i], s.axioms[j] = s.axioms[j], s.axioms[i]
                elif mutation_type == 2:
                    # _mutate_2: insert random element in random place  
                    print('_mutate_2')
                    diff_set = set(self.config['axioms_set']).difference(set(s.axioms))
                    if diff_set != set():
                        item = random.choice(list(diff_set))
                        s.axioms.insert(random.randrange(len(s.axioms)+1), item)
                elif mutation_type == 3:
                    # _mutate_3: delete random element
                    print('_mutate_3')
                    if len(s.axioms) > 1:
                        i = random.randint(0, len(s.axioms)-1)
                        s.axioms.pop(i)
                else:
                    # _mutate_4: random change random element
                    print('_mutate_4')
                    diff_set = set(self.config['axioms_set']).difference(set(s.axioms))
                    if diff_set != set():
                        i = random.randint(0, len(s.axioms)-1)
                        s.axioms[i] = random.choice(list(diff_set))
                
                value, normal_score, anomaly_score = self._knn_objective_function(self._ts_transform(s.axioms))
                s.Update(value, normal_score, anomaly_score)
                
                ##############################################################################
                
                #delta_new = np.std([s_.obj_f for s_ in self.population]) / (np.mean([s_.obj_f for s_ in self.population])+0.00001)
                #self.mutate_prob_matrix[c] *= delta_old / delta_new
                
                f_new = s.anomaly_data_anomaly_score / float(s.normal_data_anomaly_score+0.001)
                self.mutate_prob_matrix[c] = self.adaptate_probability(self.mutate_prob_matrix[c], f_old, f_new)
                
                '''
                if self.corrMode == 10:
                    self.corrPar1 = count_variance(old_popul)
                    self.corrPar2 = count_variance(self.population)
                
                if self.corrMode == 5:
                    self.corrPar2 = count_avg(self.population)
                    adaptate(self.mutProb[c], s, self.corrPar2, max([j.rel for j in self.population]), self.corrPar1)
                else:
                    adaptate(self.mutProb[c], s, old, self.corrMode, self.corrPar1, self.corrPar2)
                '''
                ##############################################################################
                
            else:
                print('no mutation')
            
            c += 1
        
        print('\t mutate done...')
        
    def _crossover(self):
        print('...crossover started')
        
        if len(self.population) < 2:
            print('\t crossover done...')
            return
            
        print('pop:', len(self.population))
            
        new_pop = []
        notCrossNum = int((1.0 - self.crossPercent) * len(self.population))
        #if self.corrMode == 10:
        #    old_popul = self.population
        for i in range(notCrossNum):
            new_pop.append(copy.deepcopy(self.population[i]))
        for i in range(int(len(self.population)/2)):
            p1, p2 = random.sample(range(len(self.population)), 2)
            parents = [self.population[p1], self.population[p2]]
            rr = random.random()
            
            #if rr <= self.config['genetic_algorithms_params']['crossover_prob_init']:
            if rr <= self.crossover_prob_matrix[p1] and rr <= self.crossover_prob_matrix[p2]: # adaptive
                # parents = random.sample(self.population,  2)
                # k = random.randint(1,Module.conf.modNum-1)
                
                #old0 = copy.deepcopy(parents[0]) # adaptive
                #old1 = copy.deepcopy(parents[1]) # adaptive
                #delta_old = np.std([s_.obj_f for s_ in self.population]) / (np.mean([s_.obj_f for s_ in self.population])+0.00001)
                
                f_old_1 = self.population[p1].anomaly_data_anomaly_score / float(self.population[p1].normal_data_anomaly_score+0.001)
                f_old_2 = self.population[p2].anomaly_data_anomaly_score / float(self.population[p2].normal_data_anomaly_score+0.001)
                
                min_l, max_l = min(len(parents[0].axioms), len(parents[1].axioms)), max(len(parents[0].axioms), len(parents[1].axioms))
                k = random.randrange(min_l, max_l + 1)
                
                union_set = set(parents[0].axioms).union(set(parents[1].axioms))
                child1 = random.sample(list(union_set), k)
                child2 = random.sample(list(union_set), k)
                
                parents[0].axioms = child1
                parents[1].axioms = child2
                
                value, normal_score, anomaly_score = self._knn_objective_function(self._ts_transform(parents[0].axioms))
                parents[0].Update(value, normal_score, anomaly_score)
                value, normal_score, anomaly_score = self._knn_objective_function(self._ts_transform(parents[1].axioms))
                parents[1].Update(value, normal_score, anomaly_score)
                
                #######################################################################################
                
                #delta_new = np.std([s_.obj_f for s_ in self.population]) / (np.mean([s_.obj_f for s_ in self.population])+0.00001)
                #self.crossover_prob_matrix[p1] *= delta_new / delta_old
                #self.crossover_prob_matrix[p2] *= delta_new / delta_old
                
                f_new_1 = self.population[p1].anomaly_data_anomaly_score / float(self.population[p1].normal_data_anomaly_score+0.001)
                f_new_2 = self.population[p2].anomaly_data_anomaly_score / float(self.population[p2].normal_data_anomaly_score+0.001)
                self.crossover_prob_matrix[p1] = self.adaptate_probability(self.crossover_prob_matrix[p1], f_new_1, f_old_1)
                self.crossover_prob_matrix[p2] = self.adaptate_probability(self.crossover_prob_matrix[p2], f_new_2, f_old_2)
                
                '''
                if self.corrMode == 10:
                    self.corrPar2 = count_variance(old_popul)
                    self.corrPar1 = count_variance(self.population)
                    
                if self.corrMode == 5:
                    self.corrPar2 = count_avg(self.population)
                    adaptate(self.crossProb[p1], parents[0], self.corrPar2, max([j.rel for j in self.population]), self.corrPar1)
                    self.corrPar2 = count_avg(self.population)
                    adaptate(self.crossProb[p2], parents[1], self.corrPar2, max([j.rel for j in self.population]), self.corrPar1)
                else:
                    adaptate(self.crossProb[p1], parents[0], old1, self.corrMode, self.corrPar1, self.corrPar2)
                    adaptate(self.crossProb[p2], parents[1], old1, self.corrMode, self.corrPar1, self.corrPar2)
                
                #adaptate(self.crossProb[p1], parents[0], old0, self.corrMode, self.corrPar1, self.corrPar2) 
                    
                #adaptate(self.crossProb[p2], parents[1], old1, self.corrMode, self.corrPar1, self.corrPar2)
                
                '''
                #######################################################################################
                
        self.population.sort(key = lambda x: x.obj_f, reverse = False)
        new_pop += self.population[:len(self.population) - notCrossNum]
        self.population = new_pop
        self.population.sort(key = lambda x: x.obj_f, reverse = False)
        
        print('\t crossover done...')
        
    def _linearRankFitness(self, population):
        a = 2.0 * (self.selective_pressure - 1) / (len(population) - 1)
        for i in range(len(population)):
            population[i].fitness_f = self.selective_pressure - a * i    
        #return population

    def _weighted_random_choice(self, population):
        max = sum(axioms.fitness_f for axioms in population)
        pick = np.random.uniform(0, max)
        current = 0
        for axioms in population:
            current += axioms.fitness_f
            if current > pick:
                return axioms
            
    def _roulette(self, population):
        #return [self._weighted_random_choice(population) for i in range(self.config['genetic_algorithms_params']['n_individuals'])]
        return [self._weighted_random_choice(population) for i in range(len(population))]

    def _select(self):
        print('...select started')
        if len(self.population) < 2:
            print('\t select done...')
            return
        newPopulation = []
        self.population.sort(key = lambda x: x.obj_f)
        eliteCount = int(round(self.elitism * self.config['genetic_algorithms_params']['n_individuals']))
        newPopulation.extend(self.population[:eliteCount])
        self.population = self.population[eliteCount:]
        self._linearRankFitness(self.population)
        newPopulation.extend(self._roulette(self.population))
        self.population = newPopulation
        self.population.sort(key = lambda x: x.obj_f)
        print('\t select done...')
    
    def _evalPopulation(self):
        print('...evalPopulation started')
        self.population.sort(key = lambda x: x.obj_f)
        if self.population[0].obj_f < self.currentBestSolution.obj_f:
            self.currentBestSolution = copy.deepcopy(self.population[0])
            self.currentIterWithoutChange = 0
        else:
            self.currentIterWithoutChange += 1
        print('\t evalPopulation done...')
        
    def _checkStopCondition(self):
        if self.config['stop_criterion']['maxIter'] != -1:
            if self.currentBestSolution != None and self.currentIter >= self.config['stop_criterion']['maxIter']:
                #self.currentSolution.Update()
                return True

        if self.config['stop_criterion']['maxIterWithoutChange'] != -1:
            if self.currentBestSolution != None and self.currentIterWithoutChange >= self.config['stop_criterion']['maxIterWithoutChange']:
                #self.currentSolution.Update()
                return True
                
        if self.config['stop_criterion']['minObjF'] != -1:
            if self.currentBestSolution != None and self.currentBestSolution.obj_f >= self.config['stop_criterion']['minObjF']:
                #self.currentSolution.Update()
                return True
        
        return False
