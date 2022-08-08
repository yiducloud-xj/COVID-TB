import pickle
import json

with open('/users/qiaoweixu/desktop/covid-19-joint-extraction/processors/dataset_labels.json', 'rb') as f:
    dataset_labels = pickle.load(f)

print(dataset_labels)
train_inputs = dataset_labels['train_inputs']
test_inputs = dataset_labels['test_inputs']
valid_inputs = dataset_labels['valid_inputs']
train_labels = dataset_labels['train_labels']
test_labels = dataset_labels['test_labels']
valid_labels = dataset_labels['valid_labels']