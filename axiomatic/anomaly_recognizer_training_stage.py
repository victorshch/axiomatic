class Solution(object):
    def __init__(self):
        pass
   
class Axioms_marking(object):
    '''
    Represents a system.
    '''
    constraints = []
    def __init__(self):
        self.axioms = []
        self.obj_f = 0
        self.num = 0

    def __eq__(self, other):
        if other == None:
            return False
        for m1, m2 in zip(self.modules, other.modules):
            if not (m1 == m2):
                return False
        return True

    def distance(self, other):
        '''
        :param other: other system.
        :returns: number of different modules for self and other.
        '''
        res = 0
        for m in self.modules:
            if not (m == other.modules[m.num]):
                res += 1
        return res

    def __computeObjF(self, value=0):
        self.obj_f = value

    def Update(self, value=0):
        '''
        Updates reliability, cost and times.
        Call it after every changing in modules!!!
        '''
        self.__computeObjF(value)

    def GenerateRandom(self, axioms_set):
        '''
        Generates random solution.
        :param checkConstraints: if generated solution must satisfy constraints.
        '''
        n = random.randint(1, len(axioms_set)+1)
        self.axioms = random.sample(axioms_set, n)
        self.Update()
   
class GeneticAlgorithm(object):
    def __init__(self, parameters):
        self.currentSolution = None
        self.currentIter = 0
        self.population = []
        self.popNum = popNum
        self.maxIter = 0
        self.iterWithoutChange = 0
        self.mutProb = 0.7 #[random.uniform(0.1, 0.9) for j in range(0, self.algconf.popNum)]
        self.crossProb = 0.7 #[random.uniform(0.1, 0.9) for j in range(0, self.algconf.popNum)]
        
    def Step(self):
        self._select()
        self._crossover()
        self._mutate()
        self._evalPopulation()
        
    def Run(self):
        self.Clear()
        #Algorithm.timecounts = 0
        #Algorithm.simcounts = 0
        #Algorithm.time = time.time()
        for i in range(self.popNum):
            s = Axioms_marking()
            s.GenerateRandom(axioms_set)
            self.population.append(s)
        self.population.sort(key=lambda x: x.obj_f, reverse = True)
        while not self._checkStopCondition():
            self.Step()
            print('Iteration =', self.currentIter ,'; Best score =', '%.10f' % self.currentSolution.rel, self.currentSolution)
            
        print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
        print("Best solution (found on", self.currentIter, "iteration):", self.currentSolution)
        print("--------------------------------------")
        Algorithm.time = time.time() - Algorithm.time
        self.stat.AddExecution(Execution(copy.deepcopy(self.currentSolution), self.currentIter, Algorithm.time, Algorithm.timecounts, Algorithm.simcounts))

    def Clear(self):
        self.population = []
        self.iterWithoutChange = 0
        self.candidate = None 
        self.mutProb = 0.7 #[random.uniform(0.1, 0.9) for j in range(0, self.algconf.popNum)]
        self.crossProb = 0.7 #[random.uniform(0.1, 0.9) for j in range(0, self.algconf.popNum)]
        
    def _mutate(self):
        pop_i = random.randint(0,len(self.population))
        mutation_type = random.randint(0,4)
        
        
        sp = c = int((1.0-self.mutPercent) * len(self.population))
        if self.corrMode == 10:
            old_popul = copy.deepcopy(self.population)
        for s in self.population[sp:]:   
            if random.random() <= self.mutProb[c]:
                mutation_type = random.randint(0,5)
                
                old = copy.deepcopy(s)
                
                if mutation_type == 0:
                    # _mutate_0
                    s = random.shuffle(s)
                elif mutation_type == 1:       
                    # _mutate_1
                    i, j = random.sample(range(0, s)), 2)
                    s[i], s[j] = s[j], s[i]
                elif mutation_type == 2:
                    # _mutate_2
                    item = random.choice(set(axioms_set).difference(set(s)))
                    s.insert(random.randrange(len(s)+1), item)
                elif mutation_type == 3:
                    # _mutate_3
                    i = random.randint(0, len(s))
                    s.pop(i)
                else:
                    # _mutate_4
                    i = random.randint(0, len(s))
                    s[i] = random.choice(set(axioms_set).difference(set(s)))
                
                s.obj_f = 
                
                s.Update()
                
                ##############################################################################
                
                if self.corrMode == 10:
                    self.corrPar1 = count_variance(old_popul)
                    self.corrPar2 = count_variance(self.population)
                
                if self.corrMode == 5:
                    self.corrPar2 = count_avg(self.population)
                    adaptate(self.mutProb[c], s, self.corrPar2, max([j.rel for j in self.population]), self.corrPar1)
                else:
                    adaptate(self.mutProb[c], s, old, self.corrMode, self.corrPar1, self.corrPar2)
                
                ##############################################################################
                
            c += 1
        
    def _crossover(self):
        pass
        
    def genEvent(dict):
    '''Generates random event from dictionary.
    :param dict: dictionaty event --> probability.
    Event must be hashable.
    :returns: Event.
    '''
        points = [0]
        cur = 0
        for event in dict.keys():
            cur += dict[event]
            points.append(cur)
        i = random.random()
        for p in range(1,len(points)):
            if points[p-1] <= i < points[p]:
                return dict.keys()[p-1]
        
    def _select(self):
        #self.population.sort(key = lambda x: x.obj_f, reverse=True)
        
        probabilities = []
        sum = 0.0 
        for s in self.population:
            val = s.obj_f
            sum += val
            probabilities.append(val)
        for p in range(len(self.population)):
            probabilities[p] = probabilities[p]/sum
        nums = range(0, len(self.population))
        events = dict(zip(nums, probabilities))
        new_pop = []
        for i in nums:
            new_pop.append(self.population[genEvent(events)])
        self.population = new_pop
        self.population.sort(key=lambda x: x.obj_f, reverse = True)
    
    def _evalPopulation(self):
        pass
        
    def _checkStopCondition(self):
        pass
    