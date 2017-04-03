
# coding: utf-8

# In[80]:

import axiomatic
from axiomatic import axiom_training_stage
from axiomatic.axiom_training_stage import KMeansClusteringAxiomStage
from axiomatic import anomaly_recognizer_training_stage
from axiomatic.anomaly_recognizer_training_stage import GeneticAlgorithm


# In[51]:

import numpy as np
import pandas as pd
import random


# ### Test

# #### Prepare data

# In[5]:

ds = pd.read_csv('train.csv', sep=',', header=None)


# In[6]:

ds.columns = ['class_']+list(ds.columns[1:])


# In[7]:

#ds.head()


# In[8]:

ds.drop('class_', inplace=True, axis=1)


# In[9]:

#ds.head()


# In[10]:

ds = ds.sample(100, random_state=42)


# In[11]:

print(ds.shape)
print(ds.head())


# In[12]:

df_list = [] # list of time series, type(time series)=pd.DataFrame
for index, row in ds.iterrows():
    df_list.append(pd.DataFrame(row))


# In[13]:

print(len(df_list))


# In[14]:

df_list_stage1 = df_list[:34] # train time series to create axoims set
df_list_stage2 = df_list[34:66] # train time series to create axioms marking
df_list_stage3 = df_list[66:] # train time series to learn anomaly recognizer


# In[15]:

df_list_stage2_normal = df_list_stage2[:16]
df_list_stage2_anomaly = df_list_stage2[16:]

df_list_stage3_normal = df_list_stage3[:17]
df_list_stage3_anomaly = df_list_stage3[17:]


# In[16]:

#del ds, df_list


# #### Create multy-layer dataset

# In[17]:

train_dict = {'normal': df_list_stage1}
test_dict = {'normal': df_list_stage2_normal, 'anomaly': df_list_stage2_anomaly}
validate_dict = {'normal': df_list_stage3_normal, 'anomaly': df_list_stage3_anomaly}
dataset = {'train': train_dict, 'test': test_dict, 'validate': validate_dict}


# #### Stage 1. Create axioms set

# In[18]:

config = {'clustering_params': {'n_clusters': 15, 'init': 'k-means++', 'n_init': 10}, 
          'feature_extraction_params': {'sample_length': 20, 'ratio': 0.2}}
model = KMeansClusteringAxiomStage(config)


# In[19]:

artifacts = model.train(dataset)


# In[45]:

print(type(artifacts['axioms']['_clusters']))
print(artifacts['axioms']['_clusters'])


# In[21]:

#ts = np.array(df_list_stage1[2].values.reshape(-1,1))
#is_work = artifacts['axioms']['_clusters'][5].run(df_list_stage1[2])


# In[22]:

#is_work


# #### Stage 2. Create axioms marking

# In[81]:

config = {'genetic_algorithms_params': {'n_individuals': 15, 'mutation_prob_init': 0.7, 'crossover_prob_init': 0.7}, 
         'stop_criterion': {'maxIter': 10, 'maxIterWithoutChange': 5, 'minObjF': None},
         'objective_function_algorithm': 'knn',
         'axioms_set': artifacts['axioms']['_clusters'],
         'train_data': dataset['train'], 'test_data': dataset['test']}
ga_model = GeneticAlgorithm(config)


# In[82]:

ga_model.Run()

