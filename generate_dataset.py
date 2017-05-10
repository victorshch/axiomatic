from datasets.synthetic2.gen import generate_dataset
from datasets.synthetic2.config import dataset_config
import pickle

dataset_config['dimension_count'] = int(input('Number of dimensions (1 or 2): '))
number_of_classes = int(input('Number of classes (1 or 2): '))

if number_of_classes == 1:
    dataset_config['classes'].pop('bad2')

dataset = generate_dataset(**dataset_config)
pickle.dump(dataset, open('dataset_{0}_{1}.pickle'.format(dataset_config['dimension_count'], number_of_classes), 'wb'))
