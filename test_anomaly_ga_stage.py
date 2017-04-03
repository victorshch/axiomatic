# coding: utf-8

print('...import modules started')

import axiomatic
from axiomatic import axiom_training_stage
from axiomatic.axiom_training_stage import KMeansClusteringAxiomStage
from axiomatic import anomaly_recognizer_training_stage
from axiomatic.anomaly_recognizer_training_stage import GeneticAlgorithm

import numpy as np
import pandas as pd
import random

print('\t done...')

# ### Test

# #### Prepare data

print('...prepare data started')

ds = pd.read_csv('train.csv', sep=',', header=None)

ds.columns = ['class_']+list(ds.columns[1:])

ds.drop('class_', inplace=True, axis=1)

ds = ds.sample(100, random_state=42)

df_list = [] # list of time series, type(time series)=pd.DataFrame
for index, row in ds.iterrows():
    df_list.append(pd.DataFrame(row))

df_list_stage1 = df_list[:34] # train time series to create axoims set
df_list_stage2 = df_list[34:66] # train time series to create axioms marking
df_list_stage3 = df_list[66:] # train time series to learn anomaly recognizer

df_list_stage2_normal = df_list_stage2[:16]
df_list_stage2_anomaly = df_list_stage2[16:]

df_list_stage3_normal = df_list_stage3[:17]
df_list_stage3_anomaly = df_list_stage3[17:]

#del ds, df_list

# #### Create multy-layer dataset

train_dict = {'normal': df_list_stage1}
test_dict = {'normal': df_list_stage2_normal, 'anomaly': df_list_stage2_anomaly}
validate_dict = {'normal': df_list_stage3_normal, 'anomaly': df_list_stage3_anomaly}
dataset = {'train': train_dict, 'test': test_dict, 'validate': validate_dict}

print('\t done...')

# #### Stage 1. Create axioms set

print('...create axioms set started')

config = {'clustering_params': {'n_clusters': 15, 'init': 'k-means++', 'n_init': 10}, 
          'feature_extraction_params': {'sample_length': 20, 'ratio': 0.2}}
model = KMeansClusteringAxiomStage(config)

artifacts = model.train(dataset)

#print(type(artifacts['axioms']['_clusters']))
#print(artifacts['axioms']['_clusters'])

#ts = np.array(df_list_stage1[2].values.reshape(-1,1))
#is_work = artifacts['axioms']['_clusters'][5].run(df_list_stage1[2])
#print(is_work)

print('\t done...')

# #### Stage 2. Create axioms marking

print('...create ga config started')

config = {'genetic_algorithms_params': {'n_individuals': 15, 'mutation_prob_init': 0.7, 'crossover_prob_init': 0.7}, 
         'stop_criterion': {'maxIter': 10, 'maxIterWithoutChange': 5, 'minObjF': -1},
         'objective_function_algorithm': 'knn',
         'axioms_set': artifacts['axioms']['_clusters'],
         'train_data': dataset['train'], 'test_data': dataset['test']}
ga_model = GeneticAlgorithm(config)

print('\t done...')

print('...ga run started')

ga_model.Run()

print('\t done...')
raw_input('Ok')