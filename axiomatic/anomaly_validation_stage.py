from base import AxiomSystem
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
import random, copy, time
from math import sqrt     
   
class KNN(object):
    def __init__(self, config=dict()):
        print('...init started')
        """
        @param config: config for knn, should be dict (e.g. 
        {'knn_params': {'k': 5, 'threshold': 0.7}, 
         'axioms': axiom_system,
         'train_data': dataset['train'], 'test_data_normal': dataset['test_n'], 'test_data_anomaly': dataset['test_an']})
         
        type(axioms_set)=list
        type(test_data)=dict
        """

        # config for genetic algorithm
        self.k = config['knn_params']['k']
        self.threshold = config['knn_params']['threshold']
        self.axioms = config['axioms']
        self.train_n = config['train_data']
        self.test_n = config['test_data_normal']
        self.test_an = config['test_data_anomaly']
        self.lift = -1
        print('\t init done...')
        
    def Clear(self):
        print('...clear started')
        self.k = 5
        self.threshold = 0.7
        self.lift = -1
        print('\t clear done...')
        
    def _ts_transform(self):
        print('...ts_transform started')
        axiom_system = AxiomSystem(self.axioms)
        
        train_normal_data_marked = pd.DataFrame(self.train_n).ix[:,0].apply(axiom_system.perform_marking)
        test_normal_data_marked = pd.DataFrame(self.test_n).ix[:,0].apply(axiom_system.perform_marking)
        test_anomaly_data_marked = pd.DataFrame(self.test_an).ix[:,0].apply(axiom_system.perform_marking)
        
        '''
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
        '''
        
        data = {'train': train_normal_data_marked, 'normal': test_normal_data_marked, 'anomaly': test_anomaly_data_marked}
        
        print('\t ts_transform done...')
        return data
        
    def _LCS(self, s1, s2):
        #print('...LCS started')
        '''
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
        '''
        
        s = SequenceMatcher(None, s1, s2, autojunk=False)
        #t = s.find_longest_match(0, len(s1), 0, len(s2))
        l = sum(t[2] for t in s.get_matching_blocks())
        
        #print('\t LCS done...')
        #return len(s1[x_longest - longest: x_longest])
        #return t[2]
        return l
        
    def _fastLCS(self, s1, s2):
        #print('...fastLCS started')
        
        
        #print('\t fastLCS done...')
        return len(s1[x_longest - longest: x_longest])
        
    def _knn_objective_function(self, data):
        print('...knn_obj_f started')
        normal_data_anomaly_score = []
        for ts_test in data['normal']:
            anomaly_scores = []
            for ts_train in data['train']:
                anomaly_score = 1 - self._LCS(ts_test, ts_train) / float(sqrt(len(ts_test)*len(ts_train)))
                anomaly_scores.append(anomaly_score)
            anomaly_scores.sort(reverse = False)
            normal_data_anomaly_score.append(anomaly_scores[self.k-1])
                
        anomaly_data_anomaly_score = []
        for ts_test in data['anomaly']:
            anomaly_scores = []
            for ts_train in data['train']:
                anomaly_score = 1 - self._LCS(ts_test, ts_train) / float(sqrt(len(ts_test)*len(ts_train)))
                anomaly_scores.append(anomaly_score)
            anomaly_scores.sort(reverse = False)
            anomaly_data_anomaly_score.append(anomaly_scores[self.k-1])
        
        print('\t knn_obj_f done...')
        return normal_data_anomaly_score, anomaly_data_anomaly_score
        
    def run(self):
        # compute comulative lift
        print('run started\n')
        
        marked_data = self._ts_transform()
        normal_data_anomaly_scores, anomaly_data_anomaly_scores = self._knn_objective_function(marked_data)
        
        true_normal_count = len(normal_data_anomaly_scores)
        print('true_normal_count = '+str(true_normal_count)+'\n')
        normal_data_anomaly_scores = pd.DataFrame(normal_data_anomaly_scores, columns=['value'])
        normal_data_anomaly_scores['label'] = 'normal'
        
        true_anomaly_count = len(anomaly_data_anomaly_scores)
        print('true_anomaly_count = '+str(true_anomaly_count)+'\n')
        anomaly_data_anomaly_scores = pd.DataFrame(anomaly_data_anomaly_scores, columns=['value'])
        anomaly_data_anomaly_scores['label'] = 'anomaly'
        
        all_anomaly_scores = pd.concat([normal_data_anomaly_scores, anomaly_data_anomaly_scores], axis=0)
        del normal_data_anomaly_scores, anomaly_data_anomaly_scores

        all_anomaly_scores = all_anomaly_scores.reset_index().drop('index', axis=1)
        all_anomaly_scores.sort_values('value', ascending=False, inplace=True, axis=0)
        print(all_anomaly_scores)
        slice_ = all_anomaly_scores[:true_anomaly_count].label
        pred_anomaly_count = slice_[slice_ == 'anomaly'].count()
        pred_normal_count = true_normal_count - true_anomaly_count + pred_anomaly_count
        
        self.lift = pred_anomaly_count / float(true_anomaly_count)
        e1_n = true_anomaly_count - pred_anomaly_count
        e2_n = true_normal_count - pred_normal_count
        
        print('lift: '+str(self.lift)+'\n')
        print('e1 number: '+str(e1_n)+'\n')
        print('e2 number: '+str(e2_n)+'\n')

        print('\t done\n')
        
class KnnDtw(object):    
    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step
        
    def _dtw_distance(self, ts_a, ts_b, d = lambda x,y: abs(x-y)):
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxint * np.ones((M, N))

        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in xrange(1, M):
            cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

        for j in xrange(1, N):
            cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

        for i in xrange(1, M):
            for j in xrange(max(1, i - self.max_warping_window),
                            min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        return cost[-1, -1]
    
    def _dtw_multidim(self, s1, s2):
        s = 0
        for i in s1.columns:
            distance = self._dtw_distance(s1[i], s2[i])
            s += distance
        
        return s