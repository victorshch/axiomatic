from datasets.synthetic2.gen import generate_dataset
from datasets.synthetic2.config import dataset_config
import pickle, sys

dataset_config['dimension_count'] = int(sys.argv[1])
number_of_classes = int(sys.argv[2])

if number_of_classes == 1:
    dataset_config['classes'].pop('bad2')

dataset_config['noise']['amp_distortion'] = float(sys.argv[3])
dataset_config['noise']['time_distortion'] = float(sys.argv[4])

dataset = generate_dataset(**dataset_config)
pickle.dump(dataset, open('dataset.pickle', 'wb'))
