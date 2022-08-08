import time
import torchmetrics
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
# from torch.utils import data
from transformers import BertConfig, AdamW
import os

from datasets.ECR_COVID_19.load_datasets import *
from models.je_models import *
from tools.tools import *
from metrics.mertics import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
device_ids = [0, 1, 2, 3, 4, 5, 6, 7] 
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

setup_seed(20)


MODEL_PATH = '/home/fidtqh3/COVID_19_joint_extraction_2/prev_trained_model/bert-base-chinese'
OUTPUT_PATH = '/home/fidtqh3/COVID_19_joint_extraction_2/outputs/checkpoint_linear_ignore0.pth.tar'
config = BertConfig.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, num_labels=2)
config.soft_label = False
MAX_SEQ_LEN = 512
VEC_SIZE = 768
BATCH_SIZE = 16
EPOCH = 10
LEANING_RATE = 3e-5
EPS = 1e-12

a_1 = 0.25
a_2 = 0.25
a_3 = 0.25
a_4 = 0.25

idx2patient = {
        1: 'LocalID',
        2: 'Age',
        3: 'Gender',
        4: 'ResidencePlace',
        5: 'SuspectedPatientContact',
        6: 'InfectionOriginContact'
    }

idx2relation = {
        1: 'SocialRelation',
        2: 'LocalID',
        3: 'Name',
    }

idx2event = {
        1: 'Event',
        2: 'Onset',
        3: 'HospitalVisit',
        4: 'DiagnosisConfirmed',
        5: 'Inpatient',
        6: 'Discharge',
        7: 'Death',
        8: 'Observed',
        9: 'Date',
        10: 'Symptom',
        11: 'LabTest',
        12: 'ImagingExamination',
        13: 'Location',
        14: 'Spot',
        15: 'Vehicle'
    }



# load data and labels
train_inputs = dataset_labels['train_inputs']
test_inputs = dataset_labels['test_inputs']
valid_inputs = dataset_labels['valid_inputs']
train_labels = dataset_labels['train_labels']
test_labels = dataset_labels['test_labels']
valid_labels = dataset_labels['valid_labels']
# print(train_inputs[0])
# print(train_labels[0])
# print(len(train_inputs))
# print(len(test_inputs))
# print(len(valid_inputs))





class MyDataset(Dataset):
    def __init__(self, inputs, labels):
        self.data = inputs
        self.lalebs = labels
    
    def __getitem__(self, index):
        return self.data[index], self.lalebs[index]
        
    def __len__(self):
        return len(self.data)



train_dataset = MyDataset(train_inputs, train_labels)
test_dataset = MyDataset(test_inputs, test_labels)
valid_dataset = MyDataset(valid_inputs, valid_labels)



train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
valid_loader = DataLoader(dataset=valid_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)


model = JointExtract(config=config, batch_size=BATCH_SIZE, model_path=MODEL_PATH, seq_len=MAX_SEQ_LEN, vec_size=VEC_SIZE)
if torch.cuda.is_available():
    model.to(device)
    # model = nn.DataParallel(model, device_ids=device_ids)





def train(train_loader, train_model):
    optim = AdamW(train_model.parameters(), lr=LEANING_RATE, eps=EPS)
    start_epoch = 0
    if os.path.exists(OUTPUT_PATH):
        checkpoint = torch.load(OUTPUT_PATH)
        start_epoch = checkpoint['epoch']
        train_model.load_state_dict(checkpoint['model_state_dict'])
    for epoch in range(start_epoch, EPOCH):
        print("epoch: {}".format(epoch))
        for idx, (tra_data, tra_labels) in enumerate(train_loader):
            optim.zero_grad() 

            # l1_h_0, l1_c_0 = train_model.module.init_hidden_state()
            # l2_h_0, l2_c_0 = train_model.module.init_hidden_state()

            l1_h_0, l1_c_0 = train_model.init_hidden_state()
            l2_h_0, l2_c_0 = train_model.init_hidden_state()

            # bert_loss, patient_loss, event_loss, relation_loss, patient_pos, event_pos, relation_pos = model(**tra_data, **tra_labels)
            bert_loss, patient_loss, relation_loss, event_loss, patient_pos, relation_pos, event_pos = train_model(**tra_data, **tra_labels, l1_h_0=l1_h_0, l1_c_0=l1_c_0, l2_h_0=l2_h_0, l2_c_0=l2_c_0)
            # bert_loss, patient_loss, patient_pos= model(**tra_data, **tra_labels)
            print("idx : {} \t loss backward ...".format(idx))
            # loss = 2*a_1*bert_loss + 2*a_2*patient_loss 
            # loss = a_1*bert_loss + a_2*patient_loss + a_3*event_loss + a_4*relation_loss
            loss = a_1*bert_loss + a_2*patient_loss + a_3*relation_loss + a_4*event_loss

            print("total loss:{} \t bert loss:{} \t patient loss:{} \t relation loss:{} \t event loss:{}".format(loss.item(), bert_loss.item(), patient_loss.item(), relation_loss.item(), event_loss.item()))

              
            loss.backward()
            optim.step()
            print("next iterator")
    state = {
        'epoch' : epoch + 1,
        'model_state_dict' : train_model.state_dict()
    }

    torch.save(state, OUTPUT_PATH)

    # return train_model
            


def predict():
    pass


def test(test_loader, model):
    with torch.no_grad():
        """
        test_precision_patient = torchmetrics.Precision(average='weighted', num_classes=7, mdmc_average="global")
        test_recall_patient = torchmetrics.Recall(average='weighted', num_classes=7, mdmc_average="global")
        test_f1score_patient = torchmetrics.F1Score(average='weighted', num_classes=7, mdmc_average="global")

        test_precision_relation = torchmetrics.Precision(average='weighted',num_classes=4, mdmc_average="global")
        test_recall_relation = torchmetrics.Recall(average='weighted',num_classes=4, mdmc_average="global")
        test_f1score_relation = torchmetrics.F1Score(average='weighted',num_classes=4, mdmc_average="global")

        test_precision_event = torchmetrics.Precision(average='weighted', num_classes=16, mdmc_average="global")
        test_recall_event = torchmetrics.Recall(average='weighted', num_classes=16, mdmc_average="global")
        test_f1score_event = torchmetrics.F1Score(average='weighted', num_classes=16, mdmc_average="global")
        """

        patient_score = SeqScore(idx2patient)
        relation_score = SeqScore(idx2relation)
        event_score = SeqScore(idx2event)
        

        checkpoint = torch.load(OUTPUT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model.to(device)
            # model = nn.DataParallel(model, device_ids=device_ids)
        for test_data, test_labels in test_loader:
            t1 = time.time()
            l1_h_0, l1_c_0 = model.init_hidden_state()
            l2_h_0, l2_c_0 = model.init_hidden_state()

            _, _, _, _, patient_pos, relation_pos, event_pos = model(**test_data, **test_labels, l1_h_0=l1_h_0, l1_c_0=l1_c_0, l2_h_0=l2_h_0, l2_c_0=l2_c_0)
            t2 = time.time()
            tt = t2 - t1
            ans = 60.0 * float(BATCH_SIZE) / tt
            print("cpu level:{}".format(ans))


            
            patient_start_pos = patient_pos[0].argmax(dim=-1).tolist()
            patient_start_pos_labels = test_labels['patient_start_pos'].tolist()

            relation_start_pos = relation_pos[0].argmax(dim=-1).tolist()
            relation_start_pos_labels = test_labels['relation_start_pos'].tolist()

            event_start_pos = event_pos[0].argmax(dim=-1).tolist()
            # print('event:{}'.format(event_start_pos))
            event_start_pos_labels = test_labels['event_start_pos'].tolist()


            patient_score.update(patient_start_pos_labels, patient_start_pos)
            relation_score.update(relation_start_pos_labels, relation_start_pos)
            event_score.update(event_start_pos_labels, event_start_pos)
            """
            
            
            
            
            # patient metrics
            patient_start_pos = patient_pos[0].argmax(dim=-1)
            # patient_end_pos = patient_pos[1].argmax(dim=-1)

            patient_start_pos_labels = test_labels['patient_start_pos']
            # patient_end_pos_labels = test_labels['patient_end_pos']

            # print("patient_start_pos:{}".format(patient_start_pos))
            # print("patient_start_pos_labels:{}".format(patient_start_pos_labels))

            # print("patient_start_pos:{}".format(patient_start_pos.shape))
            # print("patient_start_pos_labels{}".format(patient_start_pos_labels.shape))
            # print("patient_start_pos:{}".format(patient_start_pos))
            # print("patient_start_pos_labels:{}".format(patient_start_pos_labels))
            patient_precision = test_precision_patient(patient_start_pos, patient_start_pos_labels)
            # patient_precision = test_precision_patient(patient_end_pos, patient_end_pos_labels)

            patient_recall = test_recall_patient(patient_start_pos, patient_start_pos_labels)
            # patient_recall = test_recall_patient(patient_end_pos, patient_end_pos_labels)

            patient_f1score = test_f1score_patient(patient_start_pos, patient_start_pos_labels)
            # patient_f1score = test_f1score_patient(patient_end_pos, patient_end_pos_labels)    

            # relation metrics
            relation_start_pos = relation_pos[0].argmax(dim=-1)
            # relation_end_pos = relation_pos[1].argmax(dim=-1)

            relation_start_pos_labels = test_labels['relation_start_pos']
            # relation_end_pos_labels = test_labels['relation_end_pos']

            relation_precision = test_precision_relation(relation_start_pos, relation_start_pos_labels)
            # relation_precision = test_precision_relation(relation_end_pos, relation_end_pos_labels)

            relation_recall = test_recall_relation(relation_start_pos, relation_start_pos_labels)
            # relation_recall = test_recall_relation(relation_end_pos, relation_end_pos_labels)

            relation_f1score = test_f1score_relation(relation_start_pos, relation_start_pos_labels)
            # relation_f1score = test_f1score_relation(relation_end_pos, relation_end_pos_labels)    

            # event metrics
            event_start_pos = event_pos[0].argmax(dim=-1)
            # event_end_pos = event_pos[1].argmax(dim=-1)

            event_start_pos_labels = test_labels['event_start_pos']
            # event_end_pos_labels = test_labels['event_end_pos']

            event_precision = test_precision_event(event_start_pos, event_start_pos_labels)
            # event_precision = test_precision_event(event_end_pos, event_end_pos_labels)

            event_recall = test_recall_event(event_start_pos, event_start_pos_labels)
            # event_recall = test_recall_event(event_end_pos, event_end_pos_labels)

            event_f1score = test_f1score_event(event_start_pos, event_start_pos_labels)
            # event_f1score = test_f1score_event(event_end_pos, event_end_pos_labels)    
            

        """
        patient_socre_dict, patient_info = patient_score.result()
        relation_socre_dict, relation_info = relation_score.result()
        event_socre_dict, event_info = event_score.result()
        """


        # patient metrics compute
        total_precision_patient = test_precision_patient.compute()
        total_recall_patient = test_recall_patient.compute()
        total_f1score_patient = test_f1score_patient.compute()

        # relation metrics compute
        total_precision_relation = test_precision_relation.compute()
        total_recall_relation = test_recall_relation.compute()
        total_f1score_relation = test_f1score_relation.compute()

        # event metrics compute
        total_precision_event = test_precision_event.compute()
        total_recall_event = test_recall_event.compute()
        total_f1score_event = test_f1score_event.compute()
        
        """
        print("Test patient precision: {} \t recall : {} \t f1score : {}".format(patient_socre_dict['acc'], patient_socre_dict['recall'], patient_socre_dict['f1']))
        print("Test relation precision: {} \t recall : {} \t f1score : {}".format(relation_socre_dict['acc'], relation_socre_dict['recall'], relation_socre_dict['f1']))
        print("Test event precision: {} \t recall : {} \t f1score : {}".format(event_socre_dict['acc'], event_socre_dict['recall'], event_socre_dict['f1']))

        print('patient:{}'.format(patient_info))
        print('relation:{}'.format(relation_info))
        print('event:{}'.format(event_info))
        """

        print("Test patient precision: {} \t recall : {} \t f1score : {}".format(total_precision_patient, total_recall_patient, total_f1score_patient))
        print("Test relation precision: {} \t recall : {} \t f1score : {}".format(total_precision_relation, total_recall_relation, total_f1score_relation))
        print("Test event precision: {} \t recall : {} \t f1score : {}".format(total_precision_event, total_recall_event, total_f1score_event))
        """

        patient_score.reset()
        relation_score.reset()
        event_score.reset()

        """
        # patient metrics reset
        total_precision_patient = test_precision_patient.reset()
        total_recall_patient = test_recall_patient.reset()
        total_f1score_patient = test_f1score_patient.reset()

        # relation metrics reset
        total_precision_relation = test_precision_relation.reset()
        total_recall_relation = test_recall_relation.reset()
        total_f1score_relation = test_f1score_relation.reset()

        # event metrics reset
        total_precision_event = test_precision_event.reset()
        total_recall_event = test_recall_event.reset()
        total_f1score_event = test_f1score_event.reset()
        """
        





def main(train_loader, test_loader, valid_loader, model):
    # model.train()
    # train(train_loader, model)

    test_model = model
    if os.path.exists(OUTPUT_PATH):
        checkpoint = torch.load(OUTPUT_PATH)
        start_epoch = checkpoint['epoch'] - 1
        print('epoch: {}'.format(start_epoch))
        test_model.load_state_dict(checkpoint['model_state_dict'])

    test_model.eval()
    test(test_loader, test_model)
    


if __name__ == "__main__":
    main(train_loader, test_loader, valid_loader, model)