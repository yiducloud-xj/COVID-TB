import pickle
import json
import pandas as pd
import os
import torch

from transformers import BertConfig, BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
device_ids = [0, 1, 2, 3, 4, 5, 6, 7] 
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

cur_path = os.getcwd()
train_path = "./datasets/ECR_COVID_19/train.txt"
test_path = "./covid/datasets/ECR_COVID_19/test.txt"
valid_path = "./covid/datasets/ECR_COVID_19/valid.txt"
BATCH_SIZE=16


def load_datasets(datasets):
    """
    convert string to dict list and DataFrame.
    inputs: train.txt, test.txt, valid.txt (abslote path)
    outputs: data_list[dict0, dict1,...]
             data_df: type: DataFrame
                      forexample: (The first five pieces of data in the training set.)
                         doc_id  ...                                             events
                     0    7720  ...  [{'type': [24, 26, 'HospitalVisit'], 'tuple': ...
                     1    6818  ...  [{'type': [46, 48, 'Onset'], 'tuple': [[41, 46...
                     2    6128  ...  [{'type': [58, 60, 'Onset'], 'tuple': [[52, 57...
                     3    7831  ...  [{'type': [38, 40, 'Onset'], 'tuple': [[28, 38...
                     4    7967  ...  [{'type': [43, 45, 'Event'], 'tuple': [[37, 42...

                     [5 rows x 6 columns]
    """
    with open(datasets, 'r') as f:
        data_file = f.readlines()

    data_list = []
    for lines in data_file:
        l = lines.strip()
        data_dict = json.loads(l)
        data_list.append(data_dict)

    data_df = pd.DataFrame(data_list)

    return data_list, data_df


train_list, train_df = load_datasets(train_path)
test_list, test_df = load_datasets(test_path)
valid_list, valid_df = load_datasets(valid_path)






def create_time_labels(data_entities):
    """
    创建时间的位置标签
    input: data_df['entities']
    output: time_pos_dict = {'time_start_pos': [[...], ...], 'time_end_pos': [[...], ...]}
    """
    if not isinstance(data_entities, list):
        data_entities = list(data_entities)
    # print(len(data_entities))

    time_start_pos_list = []
    # time_end_pos_list = []
    for idx, patient_entity in enumerate(data_entities):
        time_start_pos = [0] * 512
        # time_end_pos = [0] * 512
        # 每个人
        for i, entity in enumerate(patient_entity):
            if entity[2] == 'Date' and entity[1]+1 < 511:
                # bert 起始位置为['CLS']
                for pos_i in range(entity[0]+1, entity[1]+2):
                    time_start_pos[pos_i] = 1
                # time_end_pos[entity[1]+1] = 1
        time_start_pos_list.append(time_start_pos.copy())
        # time_end_pos_list.append(time_end_pos.copy())
    
    time_pos_dict = {}
    time_pos_dict['time_start_pos'] = time_start_pos_list

    return time_pos_dict


train_time_pos_dict = create_time_labels(train_df['entities'])
test_time_pos_dict = create_time_labels(test_df['entities'])
valid_time_pos_dict = create_time_labels(valid_df['entities'])
# print(train_time_pos_dict.keys())
# print(train_time_pos_dict['time_start_pos'][0])
# print(train_time_pos_dict['time_end_pos'][0])






def create_patient_labels(data_entities):
    """
    创建患者信息的位置标签
    input: data_df['entities']
    output: patient_pos_dict = {'patient_start_pos': [[...], ...], 'patient_end_pos': [[...], ...]}
    """
    if not isinstance(data_entities, list):
        data_entities = list(data_entities)

    info2idx = {
        'LocalID': 1,
        'Age': 2,
        'Gender': 3,
        'ResidencePlace': 4,
        'SuspectedPatientContact': 5,
        'InfectionOriginContact': 6
    }

    patient_start_pos_list = []

    for _, patient_info in enumerate(data_entities):
        # 每个患者的信息
        patient_start_pos = [0] * 512
        for _, info in enumerate(patient_info):
            if info[1]+1 < 511:
                # bert 起始位置为['CLS']
                for patient_i in range(info[0]+1, info[1]+2):
                    patient_start_pos[patient_i] = info2idx[info[2]]

        patient_start_pos_list.append(patient_start_pos.copy())

    patient_pos_dict = {}
    patient_pos_dict['patient_start_pos'] = patient_start_pos_list

    return patient_pos_dict



train_patient_pos_dict = create_patient_labels(train_df['patient'])
test_patient_pos_dict = create_patient_labels(test_df['patient'])
valid_patient_pos_dict = create_patient_labels(valid_df['patient'])
# print(train_patient_pos_dict['patient_start_pos'][0])




def create_relation_labels(data_entities):
    """
    创建realtion信息的位置标签
    input: data_df['entities']
    output: relation_pos_dict = {'relation_start_pos': [[...], ...], 'relation_end_pos': [[...], ...]}
    """
    if not isinstance(data_entities, list):
        data_entities = list(data_entities)

    info2idx = {
        'SocialRelation': 1,
        'LocalID': 2,
        'Name': 3,
    }

    relation_start_pos_list = []
    # relation_end_pos_list = []

    for _, relation_info in enumerate(data_entities):
        # 每个患者的信息
        relation_start_pos = [0] * 512
        # 该患者存在关系信息
        if len(relation_info) != 0:
            for _, info in enumerate(relation_info):
                relation_type = info['type']
                relation_tuples = info['tuple']
                if relation_type[1]+1 < 511:
                    # bert 起始位置为['CLS']
                    for relation_type_i in range(relation_type[0]+1, relation_type[1]+2):
                        relation_start_pos[relation_type_i] = info2idx[relation_type[2]]

                for _, relation_tuple in enumerate(relation_tuples):
                    if relation_tuple[1]+1 < 511:
                        for relation_tuple_i in range(relation_tuple[0]+1, relation_tuple[1]+2):
                            relation_start_pos[relation_tuple_i] = info2idx[relation_tuple[2]]

        relation_start_pos_list.append(relation_start_pos.copy())

    relation_pos_dict = {}
    relation_pos_dict['relation_start_pos'] = relation_start_pos_list

    return relation_pos_dict



train_relation_pos_dict = create_relation_labels(train_df['relations'])
test_relation_pos_dict = create_relation_labels(test_df['relations'])
valid_relation_pos_dict = create_relation_labels(valid_df['relations'])
# print(train_relation_pos_dict['relation_start_pos'][0])






def create_event_labels(data_entities):
    """
    创建event信息的位置标签
    input: data_df['entities']
    output: event_pos_dict = {'event_start_pos': [[...], ...], 'event_end_pos': [[...], ...]}
    """
    if not isinstance(data_entities, list):
        data_entities = list(data_entities)

    info2idx = {
        'Event': 1,
        'Onset': 2,
        'HospitalVisit': 3,
        'DiagnosisConfirmed': 4,
        'Inpatient': 5,
        'Discharge': 6,
        'Death' : 7,
        'Observed' : 8,
        'Date' : 9,
        'Symptom' : 10,
        'LabTest' : 11,
        'ImagingExamination' : 12,
        'Location' : 13,
        'Spot' : 14,
        'Vehicle' : 15
    }

    event_start_pos_list = []


    for _, event_info in enumerate(data_entities):
        # 每个患者的信息
        event_start_pos = [0] * 512
        # 该患者存在事件信息
        if len(event_info) != 0:
            for _, info in enumerate(event_info):
                event_type = info['type']
                event_tuples = info['tuple']
                if event_type[1]+1 < 511:
                    # bert 起始位置为['CLS']
                    for event_type_i in range(event_type[0]+1, event_type[1]+2):
                        event_start_pos[event_type_i] = info2idx[event_type[2]]

                for _, event_tuple in enumerate(event_tuples):
                    if event_tuple[1]+1 < 511:
                        for event_tuple_i in range(event_tuple[0]+1, event_tuple[1]+2):
                            event_start_pos[event_tuple_i] = info2idx[event_tuple[2]]

        event_start_pos_list.append(event_start_pos.copy())

    event_pos_dict = {}
    event_pos_dict['event_start_pos'] = event_start_pos_list

    return event_pos_dict



train_event_pos_dict = create_event_labels(train_df['events'])
test_event_pos_dict = create_event_labels(test_df['events'])
valid_event_pos_dict = create_event_labels(valid_df['events'])





"""

def create_relation_labels(data_entities):
    if not isinstance(data_entities, list):
        data_entities = list(data_entities)


    socialrelation_start_pos_list = []
    socialrelation_end_pos_list = []

    localid_start_pos_list = []
    localid_end_pos_list = []

    name_start_pos_list = []
    name_end_pos_list = []

    for _, relations in enumerate(data_entities):
        # 每一句文本，每一个患者
        # type
        socialrelation_start_pos = [0] * 512
        socialrelation_end_pos = [0] * 512

        # tuple
        localid_start_pos = [0] * 512
        localid_end_pos = [0] * 512

        name_start_pos = [0] * 512
        name_end_pos = [0] * 512
        for _, relation in enumerate(relations):
            # type
            # print(relation)
            relation_type = relation['type']
                # print(relation_type)
            if relation_type[2] == 'SocialRelation' and relation_type[1]+1 < 511:
                socialrelation_start_pos[relation_type[0]+1] = 1
                socialrelation_end_pos[relation_type[1]+1] = 1
            # tuple
            for relation_tuple in relation['tuple']:
                if relation_tuple[2] == 'LocalID' and relation_tuple[1]+1 < 511:
                    localid_start_pos[relation_tuple[0]+1] = 1
                    localid_end_pos[relation_tuple[1]+1] = 1
                if relation_tuple[2] == 'Name' and relation_tuple[1]+1 < 511:
                    name_start_pos[relation_tuple[0]+1] = 1
                    name_end_pos[relation_tuple[1]+1] = 1

        socialrelation_start_pos_list.append(socialrelation_start_pos.copy())
        socialrelation_end_pos_list.append(socialrelation_end_pos.copy())

        localid_start_pos_list.append(localid_start_pos.copy())
        localid_end_pos_list.append(localid_end_pos.copy())

        name_start_pos_list.append(name_start_pos.copy())
        name_end_pos_list.append(name_end_pos.copy())


    relation_pos_dict = {}
    relation_pos_dict['socialrelation_start_pos'] = socialrelation_start_pos_list
    relation_pos_dict['socialrelation_end_pos'] = socialrelation_end_pos_list
    relation_pos_dict['localid_start_pos'] = localid_start_pos_list
    relation_pos_dict['localid_end_pos'] = localid_end_pos_list
    relation_pos_dict['name_start_pos'] = name_start_pos_list
    relation_pos_dict['name_end_pos'] = name_end_pos_list
    
    return relation_pos_dict


train_relation_pos_dict = create_relation_labels(train_df['relations'])
test_relation_pos_dict = create_relation_labels(test_df['relations'])
valid_relation_pos_dict = create_relation_labels(valid_df['relations'])
# print(train_relation_pos_dict['localid_start_pos'][0])





def create_event_labels(data_entities):
    if not isinstance(data_entities, list):
        data_entities = list(data_entities)
    
    event_start_pos_list = []
    event_end_pos_list = []

    onset_start_pos_list = []
    onset_end_pos_list = []

    hospitalvisit_start_pos_list = []
    hospitalvisit_end_pos_list = []

    diagnosisconfirmed_start_pos_list = []
    diagnosisconfirmed_end_pos_list = []

    inpatient_start_pos_list = []
    inpatient_end_pos_list = []

    discharge_start_pos_list = []
    discharge_end_pos_list = []
        
    death_start_pos_list = []
    death_end_pos_list = []

    observed_start_pos_list = []
    observed_end_pos_list = []

        # tuple
    date_start_pos_list = []
    date_end_pos_list = []

    symptom_start_pos_list = []
    symptom_end_pos_list = []
        
    labtest_start_pos_list = []
    labtest_end_pos_list = []

    imagingexamination_start_pos_list = []
    imagingexamination_end_pos_list = []

    location_start_pos_list = []
    location_end_pos_list = []

    spot_start_pos_list = []
    spot_end_pos_list = []

    vehicle_start_pos_list = []
    vehicle_end_pos_list = []



    for _, events in enumerate(data_entities):
        # 每一句文本，每一个患者
        # type
        event_start_pos = [0] * 512
        event_end_pos = [0] * 512

        onset_start_pos = [0] * 512
        onset_end_pos = [0] * 512

        hospitalvisit_start_pos = [0] * 512
        hospitalvisit_end_pos = [0] * 512

        diagnosisconfirmed_start_pos = [0] * 512
        diagnosisconfirmed_end_pos = [0] * 512

        inpatient_start_pos = [0] * 512
        inpatient_end_pos = [0] * 512

        discharge_start_pos = [0] * 512
        discharge_end_pos = [0] * 512
        
        death_start_pos = [0] * 512
        death_end_pos = [0] * 512

        observed_start_pos = [0] * 512
        observed_end_pos = [0] * 512

        # tuple
        date_start_pos = [0] * 512
        date_end_pos = [0] * 512

        symptom_start_pos = [0] * 512
        symptom_end_pos = [0] * 512
        
        labtest_start_pos = [0] * 512
        labtest_end_pos = [0] * 512

        imagingexamination_start_pos = [0] * 512
        imagingexamination_end_pos = [0] * 512

        location_start_pos = [0] * 512
        location_end_pos = [0] * 512

        spot_start_pos = [0] * 512
        spot_end_pos = [0] * 512

        vehicle_start_pos = [0] * 512
        vehicle_end_pos = [0] * 512

        for _, event in enumerate(events):
            # type
            # print(relation)
            event_type = event['type']
                # print(relation_type)
            if event_type[2] == 'Event' and event_type[1]+1 < 511:
                event_start_pos[event_type[0]+1] = 1
                event_end_pos[event_type[1]+1] = 1
            if event_type[2] == 'Onset' and event_type[1]+1 < 511:
                onset_start_pos[event_type[0]+1] = 1
                onset_end_pos[event_type[1]+1] = 1
            if event_type[2] == 'HospitalVisit' and event_type[1]+1 < 511:
                hospitalvisit_start_pos[event_type[0]+1] = 1
                hospitalvisit_end_pos[event_type[1]+1] = 1
            if event_type[2] == 'DiagnosisConfirmed' and event_type[1]+1 < 511:
                diagnosisconfirmed_start_pos[event_type[0]+1] = 1
                diagnosisconfirmed_end_pos[event_type[1]+1] = 1
            if event_type[2] == 'Inpatient' and event_type[1]+1 < 511:
                inpatient_start_pos[event_type[0]+1] = 1
                inpatient_end_pos[event_type[1]+1] = 1
            if event_type[2] == 'Discharge' and event_type[1]+1 < 511:
                discharge_start_pos[event_type[0]+1] = 1
                discharge_end_pos[event_type[1]+1] = 1
            if event_type[2] == 'Death' and event_type[1]+1 < 511:
                death_start_pos[event_type[0]+1] = 1
                death_end_pos[event_type[1]+1] = 1
            if event_type[2] == 'Observed' and event_type[1]+1 < 511:
                observed_start_pos[event_type[0]+1] = 1
                observed_end_pos[event_type[1]+1] = 1
            # tuple
            for event_tuple in event['tuple']:
                if event_tuple[2] == 'Date' and event_tuple[1]+1 < 511:
                    date_start_pos[event_tuple[0]+1] = 1
                    date_end_pos[event_tuple[1]+1] = 1
                if  event_tuple[2] == 'Symptom' and event_tuple[1]+1 < 511:
                    symptom_start_pos[event_tuple[0]+1] = 1
                    symptom_end_pos[event_tuple[1]+1] = 1

                if  event_tuple[2] == 'LabTest' and event_tuple[1]+1 < 511:
                    labtest_start_pos[event_tuple[0]+1] = 1
                    labtest_end_pos[event_tuple[1]+1] = 1

                if  event_tuple[2] == 'ImagingExamination' and event_tuple[1]+1 < 511:
                    imagingexamination_start_pos[event_tuple[0]+1] = 1
                    imagingexamination_end_pos[event_tuple[1]+1] = 1

                if  event_tuple[2] == 'Location' and event_tuple[1]+1 < 511:
                    location_start_pos[event_tuple[0]+1] = 1
                    location_end_pos[event_tuple[1]+1] = 1

                if  event_tuple[2] == 'Spot' and event_tuple[1]+1 < 511:
                    spot_start_pos[event_tuple[0]+1] = 1
                    spot_end_pos[event_tuple[1]+1] = 1

                if  event_tuple[2] == 'Vehicle' and event_tuple[1]+1 < 511:
                    vehicle_start_pos[event_tuple[0]+1] = 1
                    vehicle_end_pos[event_tuple[1]+1] = 1

        event_start_pos_list.append(event_start_pos.copy())
        event_end_pos_list.append(event_end_pos.copy())

        onset_start_pos_list.append(onset_start_pos.copy())
        onset_end_pos_list.append(onset_end_pos.copy())

        hospitalvisit_start_pos_list.append(hospitalvisit_start_pos.copy())
        hospitalvisit_end_pos_list.append(hospitalvisit_end_pos.copy())

        diagnosisconfirmed_start_pos_list.append(diagnosisconfirmed_start_pos.copy())
        diagnosisconfirmed_end_pos_list.append(diagnosisconfirmed_end_pos.copy())

        inpatient_start_pos_list.append(inpatient_start_pos.copy())
        inpatient_end_pos_list.append(inpatient_end_pos.copy())

        discharge_start_pos_list.append(discharge_start_pos.copy())
        discharge_end_pos_list.append(discharge_end_pos.copy())

        death_start_pos_list.append(death_start_pos.copy())
        death_end_pos_list.append(death_end_pos.copy())

        observed_start_pos_list.append(observed_start_pos.copy())
        observed_end_pos_list.append(observed_end_pos.copy())

        date_start_pos_list.append(date_start_pos.copy())
        date_end_pos_list.append(date_end_pos.copy())

        symptom_start_pos_list.append(symptom_start_pos.copy())
        symptom_end_pos_list.append(symptom_end_pos.copy())

        labtest_start_pos_list.append(labtest_start_pos.copy())
        labtest_end_pos_list.append(labtest_end_pos.copy())

        imagingexamination_start_pos_list.append(imagingexamination_start_pos.copy())
        imagingexamination_end_pos_list.append(imagingexamination_end_pos.copy())

        location_start_pos_list.append(location_start_pos.copy())
        location_end_pos_list.append(location_end_pos.copy())

        spot_start_pos_list.append(spot_start_pos.copy())
        spot_end_pos_list.append(spot_end_pos.copy())

        vehicle_start_pos_list.append(vehicle_start_pos.copy())
        vehicle_end_pos_list.append(vehicle_end_pos.copy())


    event_pos_dict = {}
    event_pos_dict['event_start_pos'] = event_start_pos_list
    event_pos_dict['event_end_pos'] = event_end_pos_list

    event_pos_dict['onset_start_pos'] = onset_start_pos_list
    event_pos_dict['onset_end_pos'] = onset_end_pos_list

    event_pos_dict['hospitalvisit_start_pos'] = hospitalvisit_start_pos_list
    event_pos_dict['hospitalvisit_end_pos'] = hospitalvisit_end_pos_list

    event_pos_dict['diagnosisconfirmed_start_pos'] = diagnosisconfirmed_start_pos_list
    event_pos_dict['diagnosisconfirmed_end_pos'] = diagnosisconfirmed_end_pos_list

    event_pos_dict['inpatient_start_pos'] = inpatient_start_pos_list
    event_pos_dict['inpatient_end_pos'] = inpatient_end_pos_list
    
    event_pos_dict['discharge_start_pos'] = discharge_start_pos_list
    event_pos_dict['discharge_end_pos'] = discharge_end_pos_list
    
    event_pos_dict['death_start_pos'] = death_start_pos_list
    event_pos_dict['death_end_pos'] = death_end_pos_list
    
    event_pos_dict['observed_start_pos'] = observed_start_pos_list
    event_pos_dict['observed_end_pos'] = observed_end_pos_list
    
    event_pos_dict['date_start_pos'] = date_start_pos_list
    event_pos_dict['date_end_pos'] = date_end_pos_list
    
    event_pos_dict['symptom_start_pos'] = symptom_start_pos_list
    event_pos_dict['symptom_end_pos'] = symptom_end_pos_list
    
    event_pos_dict['labtest_start_pos'] = labtest_start_pos_list
    event_pos_dict['labtest_end_pos'] = labtest_end_pos_list
    
    event_pos_dict['imagingexamination_start_pos'] = imagingexamination_start_pos_list
    event_pos_dict['imagingexamination_end_pos'] = imagingexamination_end_pos_list
    
    event_pos_dict['location_start_pos'] = location_start_pos_list
    event_pos_dict['location_end_pos'] = location_end_pos_list
    
    event_pos_dict['spot_start_pos'] = spot_start_pos_list
    event_pos_dict['spot_end_pos'] = spot_end_pos_list
    
    event_pos_dict['vehicle_start_pos'] = vehicle_start_pos_list
    event_pos_dict['vehicle_end_pos'] = vehicle_end_pos_list
    
    return event_pos_dict




train_event_pos_dict = create_event_labels(train_df['events'])
test_event_pos_dict = create_event_labels(test_df['events'])
valid_event_pos_dict = create_event_labels(valid_df['events'])
# print(train_event_pos_dict['date_start_pos'][0])

"""



"""
def creat_labels(time_pos_dict, patient_pos_dict, relation_pos_dict, event_pos_dict):

    # time
    time_start_pos_list = time_pos_dict['time_start_pos']
    time_end_pos_list = time_pos_dict['time_end_pos']


    # patient
    patient_start_pos_list = patient_pos_dict['patient_start_pos']
    patient_end_pos_list = patient_pos_dict['patient_end_pos']


    # relations
    socialrelation_start_pos_list = relation_pos_dict['socialrelation_start_pos']
    socialrelation_end_pos_list = relation_pos_dict['socialrelation_end_pos']

    localid_start_pos_list = relation_pos_dict['localid_start_pos']
    localid_end_pos_list = relation_pos_dict['localid_end_pos']

    name_start_pos_list = relation_pos_dict['name_start_pos']
    name_end_pos_list = relation_pos_dict['name_end_pos']


    # events
    event_start_pos_list = event_pos_dict['event_start_pos']
    event_end_pos_list = event_pos_dict['event_end_pos']

    onset_start_pos_list = event_pos_dict['onset_start_pos']
    onset_end_pos_list = event_pos_dict['onset_end_pos']

    hospitalvisit_start_pos_list = event_pos_dict['hospitalvisit_start_pos']
    hospitalvisit_end_pos_list = event_pos_dict['hospitalvisit_end_pos']

    diagnosisconfirmed_start_pos_list = event_pos_dict['diagnosisconfirmed_start_pos']
    diagnosisconfirmed_end_pos_list = event_pos_dict['diagnosisconfirmed_end_pos']

    inpatient_start_pos_list = event_pos_dict['inpatient_start_pos']
    inpatient_end_pos_list = event_pos_dict['inpatient_end_pos']
    
    discharge_start_pos_list = event_pos_dict['discharge_start_pos']
    discharge_end_pos_list = event_pos_dict['discharge_end_pos']
    
    death_start_pos_list = event_pos_dict['death_start_pos']
    death_end_pos_list = event_pos_dict['death_end_pos']
    
    observed_start_pos_list = event_pos_dict['observed_start_pos']
    observed_end_pos_list = event_pos_dict['observed_end_pos']
    
    date_start_pos_list = event_pos_dict['date_start_pos']
    date_end_pos_list = event_pos_dict['date_end_pos']
    
    symptom_start_pos_list = event_pos_dict['symptom_start_pos']
    symptom_end_pos_list = event_pos_dict['symptom_end_pos']
    
    labtest_start_pos_list = event_pos_dict['labtest_start_pos']
    labtest_end_pos_list = event_pos_dict['labtest_end_pos']
    
    imagingexamination_start_pos_list = event_pos_dict['imagingexamination_start_pos']
    imagingexamination_end_pos_list = event_pos_dict['imagingexamination_end_pos']
    
    location_start_pos_list = event_pos_dict['location_start_pos']
    location_end_pos_list = event_pos_dict['location_end_pos']
    
    spot_start_pos_list = event_pos_dict['spot_start_pos']
    spot_end_pos_list = event_pos_dict['spot_end_pos']
    
    vehicle_start_pos_list = event_pos_dict['vehicle_start_pos']
    vehicle_end_pos_list = event_pos_dict['vehicle_end_pos']

    labels  = []
    for (time_start_pos, 
    time_end_pos, 

    patient_start_pos, 
    patient_end_pos, 
    
    socialrelation_start_pos, 
    socialrelation_end_pos, 
    
    localid_start_pos,
    localid_end_pos,

    name_start_pos,
    name_end_pos,


    # events
    event_start_pos,
    event_end_pos,

    onset_start_pos,
    onset_end_pos,

    hospitalvisit_start_pos,
    hospitalvisit_end_pos,

    diagnosisconfirmed_start_pos,
    diagnosisconfirmed_end_pos,

    inpatient_start_pos,
    inpatient_end_pos,
    
    discharge_start_pos,
    discharge_end_pos,
    
    death_start_pos,
    death_end_pos,
    
    observed_start_pos,
    observed_end_pos,
    
    date_start_pos,
    date_end_pos,
    
    symptom_start_pos,
    symptom_end_pos,
    
    labtest_start_pos,
    labtest_end_pos,
    
    imagingexamination_start_pos,
    imagingexamination_end_pos,
    
    location_start_pos,
    location_end_pos,
    
    spot_start_pos,
    spot_end_pos,
    
    vehicle_start_pos,
    vehicle_end_pos) in zip(time_start_pos_list, 
    time_end_pos_list, 
    
    patient_start_pos_list, 
    patient_end_pos_list, 
    
    socialrelation_start_pos_list, 
    socialrelation_end_pos_list, 
    
    localid_start_pos_list,
    localid_end_pos_list,

    name_start_pos_list,
    name_end_pos_list,


    # events
    event_start_pos_list,
    event_end_pos_list,

    onset_start_pos_list,
    onset_end_pos_list,

    hospitalvisit_start_pos_list,
    hospitalvisit_end_pos_list,

    diagnosisconfirmed_start_pos_list,
    diagnosisconfirmed_end_pos_list,

    inpatient_start_pos_list,
    inpatient_end_pos_list,
    
    discharge_start_pos_list,
    discharge_end_pos_list,
    
    death_start_pos_list,
    death_end_pos_list,
    
    observed_start_pos_list,
    observed_end_pos_list,
    
    date_start_pos_list,
    date_end_pos_list,
    
    symptom_start_pos_list,
    symptom_end_pos_list,
    
    labtest_start_pos_list,
    labtest_end_pos_list,
    
    imagingexamination_start_pos_list,
    imagingexamination_end_pos_list,
    
    location_start_pos_list,
    location_end_pos_list,
    
    spot_start_pos_list,
    spot_end_pos_list,
    
    vehicle_start_pos_list,
    vehicle_end_pos_list):

    # 添加到标签集
        labels.append({'time_start_pos' : torch.LongTensor(time_start_pos), 
    'time_end_pos' : torch.LongTensor(time_end_pos), 

    'patient_start_pos' : torch.LongTensor(patient_start_pos), 
    'patient_end_pos' : torch.LongTensor(patient_end_pos), 
    
    'socialrelation_start_pos':torch.LongTensor(socialrelation_start_pos), 
    'socialrelation_end_pos':torch.LongTensor(socialrelation_end_pos), 
    
    'localid_start_pos':torch.LongTensor(localid_start_pos),
    'localid_end_pos':torch.LongTensor(localid_end_pos),

    'name_start_pos':torch.LongTensor(name_start_pos),
    'name_end_pos':torch.LongTensor(name_end_pos),


    # events
    'event_start_pos':torch.LongTensor(event_start_pos),
    'event_end_pos':torch.LongTensor(event_end_pos),

    'onset_start_pos':torch.LongTensor(onset_start_pos),
    'onset_end_pos':torch.LongTensor(onset_end_pos),

    'hospitalvisit_start_pos':torch.LongTensor(hospitalvisit_start_pos),
    'hospitalvisit_end_pos':torch.LongTensor(hospitalvisit_end_pos),

    'diagnosisconfirmed_start_pos':torch.LongTensor(diagnosisconfirmed_start_pos),
    'diagnosisconfirmed_end_pos':torch.LongTensor(diagnosisconfirmed_end_pos),

    'inpatient_start_pos':torch.LongTensor(inpatient_start_pos),
    'inpatient_end_pos':torch.LongTensor(inpatient_end_pos),
    
    'discharge_start_pos':torch.LongTensor(discharge_start_pos),
    'discharge_end_pos':torch.LongTensor(discharge_end_pos),
    
    'death_start_pos':torch.LongTensor(death_start_pos),
    'death_end_pos':torch.LongTensor(death_end_pos),
    
    'observed_start_pos':torch.LongTensor(observed_start_pos),
    'observed_end_pos':torch.LongTensor(observed_end_pos),
    
    'date_start_pos':torch.LongTensor(date_start_pos),
    'date_end_pos':torch.LongTensor(date_end_pos),
    
    'symptom_start_pos':torch.LongTensor(symptom_start_pos),
    'symptom_end_pos':torch.LongTensor(symptom_end_pos),
    
    'labtest_start_pos':torch.LongTensor(labtest_start_pos),
    'labtest_end_pos':torch.LongTensor(labtest_end_pos),
    
    'imagingexamination_start_pos':torch.LongTensor(imagingexamination_start_pos),
    'imagingexamination_end_pos':torch.LongTensor(imagingexamination_end_pos),
    
    'location_start_pos':torch.LongTensor(location_start_pos),
    'location_end_pos':torch.LongTensor(location_end_pos),
    
    'spot_start_pos':torch.LongTensor(spot_start_pos),
    'spot_end_pos':torch.LongTensor(spot_end_pos),
    
    'vehicle_start_pos':torch.LongTensor(vehicle_start_pos),
    'vehicle_end_pos':torch.LongTensor(vehicle_end_pos)})


    return labels




train_labels = creat_labels(train_time_pos_dict, train_patient_pos_dict, train_relation_pos_dict, train_event_pos_dict)
test_labels = creat_labels(test_time_pos_dict, test_patient_pos_dict, test_relation_pos_dict, test_event_pos_dict)
valid_labels = creat_labels(valid_time_pos_dict, valid_patient_pos_dict, valid_relation_pos_dict, valid_event_pos_dict)
# print(train_labels[0].keys())
"""


def creat_labels(time_pos_dict, patient_pos_dict, relation_pos_dict, event_pos_dict):

    # time
    time_start_pos_list = time_pos_dict['time_start_pos']


    # patient
    patient_start_pos_list = patient_pos_dict['patient_start_pos']

    # relation
    relation_start_pos_list = relation_pos_dict['relation_start_pos']

    # event
    event_start_pos_list = event_pos_dict['event_start_pos']

    labels  = []
    for (time_start_pos,
    patient_start_pos,
    relation_start_pos,
    event_start_pos, 
    ) in zip(time_start_pos_list,
    patient_start_pos_list,
    relation_start_pos_list,
    event_start_pos_list
    ):

    # 添加到标签集
        labels.append({'time_start_pos' : torch.LongTensor(time_start_pos).to(device), 

    'patient_start_pos' : torch.LongTensor(patient_start_pos).to(device), 

    'relation_start_pos' : torch.LongTensor(relation_start_pos).to(device), 

    'event_start_pos':torch.LongTensor(event_start_pos).to(device)})

    return labels




train_labels = creat_labels(train_time_pos_dict, train_patient_pos_dict, train_relation_pos_dict, train_event_pos_dict)
test_labels = creat_labels(test_time_pos_dict, test_patient_pos_dict, test_relation_pos_dict, test_event_pos_dict)
valid_labels = creat_labels(valid_time_pos_dict, valid_patient_pos_dict, valid_relation_pos_dict, valid_event_pos_dict)










def create_inputs(inputs):
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']

    datas = []
    for (input_ids, token_type_ids, attention_mask) in zip(input_ids, token_type_ids, attention_mask):
        datas.append({'input_ids':input_ids.to(device), 'token_type_ids':token_type_ids.to(device), 'attention_mask':attention_mask.to(device)})

    return datas










train_texts = list(train_df['text'])
test_texts = list(test_df['text'])
valid_texts = list(valid_df['text'])
# print(train_texts[0])


config = BertConfig.from_pretrained('/Users/jiaoxiaokang/Desktop/covid/bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('/Users/jiaoxiaokang/Desktop/covid/bert-base-chinese')
train_inputss = tokenizer(train_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
test_inputss = tokenizer(test_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
valid_inputss = tokenizer(valid_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
# print(train_inputs.keys())


# print(train_df['text'])
# print(train_df.head())
# print(train_df['patient'])

train_inputs = create_inputs(train_inputss)
test_inputs = create_inputs(test_inputss)
valid_inputs = create_inputs(valid_inputss)


# save dataset and labels
dataset_labels = {'train_inputs':train_inputs,
                'test_inputs':test_inputs,
                'valid_inputs':valid_inputs,
                'train_labels':train_labels,
                'test_labels':test_labels,
                'valid_labels':valid_labels}

"""
with open('/users/qiaoweixu/desktop/covid-19-joint-extraction/processors/dataset_labels.json', 'wb') as f:
    pickle.dump(dataset_labels, f)
"""







"""
def setdata(train_df):
    ans = set()
    for i, pa in enumerate(train_df['patient']):
        for pa_entities in pa:
            ans.add(pa_entities[2])

    print(ans)

    ans = set()
    ans_tuple = set()
    for i, pa in enumerate(train_df['events']):
        for pa_entities in pa:
            # print(pa_entities)
            ans.add(pa_entities['type'][2])
            for paa in pa_entities['tuple']:
                ans_tuple.add(paa[2])

    print("********************************************************************************")
    print(ans)
    print("********************************************************************************")
    print(ans_tuple)

    ans = set()
    ans_tuple = set()
    for i, pa in enumerate(train_df['relations']):
        for pa_entities in pa:
            # print(pa_entities)
            ans.add(pa_entities['type'][2])
            for paa in pa_entities['tuple']:
                ans_tuple.add(paa[2])

    print("********************************************************************************")
    print(ans)
    print("********************************************************************************")
    print(ans_tuple)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>train>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
setdata(train_df)
print('\n')
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>test>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
setdata(test_df)
print('\n')
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>valid>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
setdata(valid_df)

"""


class MyDataset(Dataset):
    def __init__(self, inputs, labels):
        self.data = inputs
        self.lalebs = labels

    def __getitem__(self, index):
        return self.data[index], self.lalebs[index]

    def __len__(self):
        return len(self.data)


train_inputs = dataset_labels['train_inputs']
test_inputs = dataset_labels['test_inputs']
valid_inputs = dataset_labels['valid_inputs']
train_labels = dataset_labels['train_labels']
test_labels = dataset_labels['test_labels']
valid_labels = dataset_labels['valid_labels']
train_dataset = MyDataset(train_inputs, train_labels)
test_dataset = MyDataset(test_inputs, test_labels)
valid_dataset = MyDataset(valid_inputs, valid_labels)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
valid_loader = DataLoader(dataset=valid_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)





