# import torch
from collections import Counter

def get_entities(seq, id2label, label_entities):
    chunks = []
    chunk = [-1, -1, -1]

    for i in reversed(range(len(label_entities))):
        if label_entities[i] != 0:
            if i != 512:
                seq_len = i
            else:
                seq_len = 511

    seq_len = len(seq)-1

    idx = 1
    right = 0
    while idx < seq_len and right < seq_len:
        # print("idx: {}".format(idx))
        # print('right: {}'.format(right))
        tag_num = seq[idx]
        # print('tag_num{}'.format(tag_num))
        # print(id2label)
        if tag_num == 0:
            idx += 1
            continue

        tag = id2label[tag_num]

        if idx == 510:
            break
            chunk = [tag, 510, 510]
            
            """
            chunks.append(chunk)
            chunk = [-1, -1, -1]
            break
            """

        if right == 510:
            chunk = [tag, idx, 510]
            if idx == 510:
                break
            else:
                chunks.append(chunk)
                chunk = [-1, -1, -1]
                break


        for right in range(idx+1, seq_len):
            if seq[right] == tag_num:
                if right == len(seq) - 1:
                    if idx == right:
                        continue
                    else:
                        chunk = [tag, idx, right]
                        chunks.append(chunk)
                        chunk = [-1, -1, -1]
                        idx = right
                        break
                else:
                    continue
            else:
                if idx == right-1:
                    continue
                else:
                    chunk = [tag, idx, right-1]
                    chunks.append(chunk)
                    chunk = [-1, -1, -1]
                    idx = right
                    break

    return chunks






def findRight(pred_entities, label_entities):
    ans = []
    for pred_entity in pred_entities:
        for label_entity in label_entities:
            # if pred_entity[0] == label_entity[0]:
            # if (pred_entity not in ans) and (pred_entity[1] <= label_entity[1] and pred_entity[2] >= label_entity[1]) or (pred_entity[1] >= label_entity[1] and pred_entity[1] <= label_entity[2]) or (pred_entity[1] <= label_entity[1] and pred_entity[2] >= label_entity[2]) or (pred_entity[1] >= label_entity[1] and pred_entity[2] <= label_entity[2]):
            if (pred_entity not in ans) and (pred_entity[1] <= label_entity[1] and pred_entity[2] >= label_entity[1]) or (pred_entity[1] >= label_entity[1] and pred_entity[1] <= label_entity[2]):
                ans.append(pred_entity)
                break
    return ans











class SeqScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2*recall*precision) / (recall + precision)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {'acc': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}

        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        score_dic ={'acc': precision, 'recall': recall, 'f1': f1}
        print("origin:{}".format(self.origins))
        print("founds:{}".format(self.founds))
        print("rights:{}".format(self.rights))
        return score_dic, class_info


    def update(self, label_paths, pred_paths):
        # step = 1
        for lable_path, pred_path in zip(label_paths, pred_paths):
            # print('step: {}'.format(step))
            # step += 1
            label_entities = get_entities(lable_path, self.id2label, lable_path)
            pred_entities = get_entities(pred_path, self.id2label, lable_path)
            self.origins.extend(label_entities)
            self.founds.extend(pred_entities)
            self.rights.extend([pred_entity for pred_entity in pred_entities if pred_entity in label_entities])
            # self.rights.extend(findRight(pred_entities, label_entities))

        # print('origins:{}'.format(self.origins))
        # print('founds:{}'.format(self.founds))

