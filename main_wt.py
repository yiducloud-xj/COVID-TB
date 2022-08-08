
import torchmetrics
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
# from torch.utils import data
from transformers import BertConfig, AdamW
import os

from datasets.ECR_COVID_19.load_datasets import *
from models.je_models_wt import *
from tools.tools import *
from sklearn.metrics import classification_report

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
device_ids = [0, 1, 2, 3, 4, 5, 6, 7] 
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

setup_seed(20)


MODEL_PATH = '/home/fidtqh3/COVID_19_joint_extraction_2/prev_trained_model/bert-base-chinese'
OUTPUT_PATH = '/home/fidtqh3/COVID_19_joint_extraction_2/outputs/checkpoint_linear_wt.pth.tar'
config = BertConfig.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, num_labels=2)
config.soft_label = False
MAX_SEQ_LEN = 512
VEC_SIZE = 768
BATCH_SIZE = 16
EPOCH = 1
LEANING_RATE = 3e-5
EPS = 1e-12

a_1 = 0.25
a_2 = 0.25
a_3 = 0.25
a_4 = 0.25





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


#第一维度是空实体的label
target_names=["None_entities"]
for k in idx2event:
    target_names.append(idx2event[k])






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
        # 空实体自成一类
        test_precision_event = torchmetrics.Precision(average='weighted', num_classes=16, mdmc_average="global")
        test_recall_event = torchmetrics.Recall(average='weighted', num_classes=16, mdmc_average="global")
        test_f1score_event = torchmetrics.F1Score(average='weighted', num_classes=16, mdmc_average="global")

        patient_score = SeqScore(idx2patient)
        relation_score = SeqScore(idx2relation)
        event_score = SeqScore(idx2event)

        checkpoint = torch.load(OUTPUT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model.to(device)
            # model = nn.DataParallel(model, device_ids=device_ids)

        real_all = []
        pred_all = []
        for test_data, test_labels in test_loader:
            t1 = time.time()
            l1_h_0, l1_c_0 = model.init_hidden_state()
            l2_h_0, l2_c_0 = model.init_hidden_state()

            _, _, _, _, patient_pos, relation_pos, event_pos = model(**test_data, **test_labels, l1_h_0=l1_h_0,
                                                                     l1_c_0=l1_c_0, l2_h_0=l2_h_0, l2_c_0=l2_c_0)
            t2 = time.time()
            tt = t2 - t1
            ans = 60.0 * float(BATCH_SIZE) / tt
            print("cpu level:{}".format(ans))

            # event metrics
            event_start_pos = event_pos.argmax(dim=-1)

            event_start_pos_labels = test_labels['event_start_pos']

            pred = event_start_pos.flatten().tolist()
            real = event_start_pos_labels.flatten().tolist()
            real_all += real
            pred_all += pred
            event_precision = test_precision_event(event_start_pos, event_start_pos_labels)

            event_recall = test_recall_event(event_start_pos, event_start_pos_labels)

            event_f1score = test_f1score_event(event_start_pos, event_start_pos_labels)

        # event metrics compute
        total_precision_event = test_precision_event.compute()
        total_recall_event = test_recall_event.compute()
        total_f1score_event = test_f1score_event.compute()

        print(
            "Test event precision: {} \t recall : {} \t f1score : {}".format(total_precision_event, total_recall_event,
                                                                             total_f1score_event))

        # 分实体进行各自类别的准召率F1计算。
        print(classification_report(real_all, pred_all, target_names=target_names))

        # event metrics reset
        total_precision_event = test_precision_event.reset()
        total_recall_event = test_recall_event.reset()
        total_f1score_event = test_f1score_event.reset()




def main(train_loader, test_loader, valid_loader, model):
    # model.train()
    # train(train_loader, model)

    test_model = model
    if os.path.exists(OUTPUT_PATH):
        checkpoint = torch.load(OUTPUT_PATH)
        start_epoch = checkpoint['epoch'] - 1
        test_model.load_state_dict(checkpoint['model_state_dict'])

    test_model.eval()
    test(test_loader, test_model)
    


if __name__ == "__main__":
    main(train_loader, test_loader, valid_loader, model)