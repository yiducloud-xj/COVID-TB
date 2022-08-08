import math
import os
from transformers import BertModel, BertTokenizer, BertPreTrainedModel
from transformers import BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.liners import StartLogits, EndLogits

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


class PositionEmbedding(nn.Module):
    def __init__(self, batch_size, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.expand(batch_size, max_len, d_model).clone()
        self.register_buffer('pe', pe)

    def forward(self, x, cur_batch_size):
        # print(x.shape)
        # print(self.pe[:, :x.size(1), :].shape)
        x = x + self.pe[:cur_batch_size, :x.size(1), :]
        return self.dropout(x)


"""
cur_path = os.getcwd()
cur_path, _ = os.path.split(cur_path)
model_path = os.path.join(cur_path, '/prev_trained_model/bert-base-chinese')
config = BertConfig.from_pretrained(model_path, num_labels=1)
"""


class BertForJE(BertPreTrainedModel):
    def __init__(self, config, model_path) -> None:
        super(BertForJE, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.soft_label = config.soft_label

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_path, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = StartLogits(hidden_size=config.hidden_size, num_labels=config.num_labels)
        self.end_fc = EndLogits(hidden_size=config.hidden_size, num_labels=config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                start_pos=None, end_pos=None, labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        output = outputs[0]

        # print(output.shape)
        # print(input_ids.shape)

        output = self.dropout(output)

        start_logits = self.start_fc(output)
        """
        if start_pos is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_pos.unsqueeze(2), 1)
            else:
                label_logits = start_pos.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits)
            if not self.soft_label:
                label_logits = label_logits.argmax(dim=-1).float().unsqueeze(2)
        end_logits = self.end_fc(output, label_logits)
        end_logits = F.sigmoid(end_logits)
        outputs = (start_logits, end_logits,) + (outputs[0],) + outputs[2:]
        """
        outputs = (start_logits,) + (outputs[0],) + outputs[2:]

        if start_pos is not None:
            # loss_fct = nn.CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            # end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.contiguous().view(-1) == 1
            active_start_logits = start_logits[active_loss]
            # active_end_logits = end_logits[active_loss]

            active_start_labels = start_pos.contiguous().view(-1, 1).float()[active_loss]
            # active_end_labels = end_pos.contiguous().view(-1, 1).float()[active_loss]

            # print(active_start_logits)
            # print(active_start_logits.shape)
            # print(active_start_labels)
            # print(active_start_labels.shape)
            start_loss = self.loss_fct(active_start_logits, active_start_labels.squeeze(1).long())
            # end_loss = self.loss_fct(active_end_logits, active_end_labels.squeeze(1).long())
            # total_loss = (start_loss + end_loss) / 2
            # outputs = (total_loss,) + outputs
            outputs = (start_loss,) + outputs
        return outputs


class PoolingLayer(nn.Module):
    """
    pooling
    x = (batch_size, seq_len, vec_size)
    return y = (batch_size, 1, vec_size)
    """

    def __init__(self, reduce='mean') -> None:
        super().__init__()
        self.reduce = reduce

    def forward(self, x):
        if self.reduce == 'mean':
            y = torch.mean(x, dim=0, keepdim=True)

            # print(y.shape)
            # print(x.shape)

        if self.reduce == 'max':
            y, _ = torch.max(x, dim=0, keepdim=True)

        return y


class AttentionLayer(nn.Module):
    """
    q: the output of _pool2d (batch_size, 1, vec_size) * Wq + Bq
    k: bertoutput (batch_size, seq_len, vec_size) * Wk + Bk
    v: bertoutput (batch_size, seq_len, vec_size) * Wv + Bv
    """

    def __init__(self, batch_size, seq_len, vec_size) -> None:
        super().__init__()
        # self.batch_size = bert_output_size[0]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vec_size = vec_size
        self.linearq = nn.Linear(self.vec_size, self.vec_size)
        self.lineark = nn.Linear(self.vec_size, self.vec_size)
        self.linearv = nn.Linear(self.vec_size, self.vec_size)
        """
        self.Wq = nn.Parameter(torch.empty(self.batch_size, self.vec_size, self.vec_size))
        self.Bq = nn.Parameter(torch.empty(self.batch_size, 1, self.vec_size))
        self.Wk = nn.Parameter(torch.empty(self.batch_size, self.vec_size, self.vec_size))
        self.Bk = nn.Parameter(torch.empty(self.batch_size, self.vec_size, self.vec_size))
        self.Wv = nn.Parameter(torch.empty(self.batch_size, self.vec_size, self.vec_size))
        self.Bv = nn.Parameter(torch.empty(self.batch_size, self.vec_size, self.vec_size))
        """

    def forward(self, pool_output, bert_output):
        """
        q = torch.matmul(pool_output, self.Wq) + self.Bq
        k = torch.matmul(bert_output, self.Wk) + self.Bk
        v = torch.matmul(bert_output, self.Wv) + self.Bv
        """
        q = self.linearq(pool_output)
        k = self.lineark(bert_output)
        v = self.linearv(bert_output)
        # a (batch_size, seq_len, 1) = q * k.T
        a = torch.matmul(q, k.permute(0, 2, 1))
        a = torch.div(a, math.sqrt(self.seq_len))

        # output (batch_size, 1, vec_size)
        output = torch.matmul(a, v)

        return output


class PoolAttentionForJE(nn.Module):
    def __init__(self, batch_size, seq_len, vec_size, pa_reduce='mean') -> None:
        super().__init__()
        """
        self.bert_output = bert_output
        self.start_pos = start_pos
        self.end_pos = end_pos
        """
        self.positionembedding1 = PositionEmbedding(batch_size=1, d_model=vec_size)
        self.positionembedding2 = PositionEmbedding(batch_size=batch_size, d_model=vec_size)
        self.poolinglayer = PoolingLayer(reduce=pa_reduce)
        self.attentionlayer = AttentionLayer(batch_size, seq_len, vec_size)

    def _create_time_seq(self, bert_output, start_pos):
        """
        input: [O, O, T, O, O, T, O]
        output:[[O, O], [T, O, O], [T, O]]
        """
        # print("start_pos_ndim:{}".format(start_pos.ndim))
        if start_pos.ndim == 2:
            start_pos_list = start_pos
        else:
            start_pos_list = start_pos.unsqueeze(1)
        ans = list()
        for idx in range(start_pos_list.shape[0]):
            if idx == 0 and start_pos_list[0][0] >= 0.5:
                ans.append(0)
                continue
            if start_pos_list[idx - 1][0].item() < 0.5 and start_pos_list[idx][0].item() >= 0.5:
                ans.append(idx)

        n = len(ans)
        start_seq = list()
        if n == 0:
            start_seq.append(bert_output)
        elif n == 1:
            start_seq.append(bert_output[0, :ans[0], :])
            start_seq.append(bert_output[0, ans[0]:, :])
        else:
            start_seq.append(bert_output[0, :ans[0], :])
            pre = ans[0]
            for i in range(1, n):
                start_seq.append(bert_output[0, pre:ans[i], :])
                pre = ans[i]
            start_seq.append(bert_output[0, ans[n - 1]:, :])

        return start_seq, ans

    def _cat_qvec(self, allqvec, start_pos):
        pass

    def forward(self, bert_output, startpos, endpos=None, position_embedding=True):
        batch_size = bert_output.shape[0]

        # print(bert_output.shape)

        if position_embedding:
            bert_output = self.positionembedding2(bert_output, batch_size)

        all_qvec = list()
        # 记录每句话有几个time段
        num_time_seq = list()
        for i in range(batch_size):
            # for every seq
            # print(bert_output.shape)
            # print(bert_output[i, ...].shape)
            bert_seq = bert_output[i, ...].unsqueeze(0)
            if position_embedding:
                bert_seq = self.positionembedding1(bert_seq, 1)

            start_pos = startpos[i, ...]

            time_start_list, start_pos_list = self._create_time_seq(bert_seq, start_pos)
            num_time_seq.append(len(time_start_list))

            # print(start_pos_list)

            start_pos_list.insert(0, 0)
            if len(start_pos_list) == 1:
                diff_val_list = [bert_output.shape[1]]
            else:
                diff_val_list = [start_pos_list[i + 1] - start_pos_list[i] for i in range(len(start_pos_list) - 1)]
                diff_val_list.append(bert_output.shape[1] - start_pos_list[-1])

            # print(diff_val_list)

            # 记录每个时间段向量查询结果
            all_time_qvec = list()
            for idx, time_start_seq in enumerate(time_start_list):
                pool_output = self.poolinglayer(time_start_seq)
                attention_output = self.attentionlayer(pool_output, bert_seq)
                output_shape = attention_output.shape

                # print("time_start_seq: {}".format(time_start_seq.shape))
                # print("pool_output {}".format(pool_output.shape))
                # print("attention_output: {}".format(attention_output.shape))
                # print("output_shape: {}".format(output_shape))

                # 复制每个时间段的查询值到时间段的长度
                all_time_qvec.append(
                    attention_output.expand(output_shape[0], diff_val_list[idx], output_shape[2]).contiguous())
            seq_vec = torch.cat(all_time_qvec, dim=1)
            all_qvec.append(seq_vec)

        # all_attn_vec (batch_size, seq_len, vec_size)
        all_attn_vec = torch.cat(all_qvec, dim=0)

        # print("all_attn_vec shape:{}".format(all_attn_vec.shape))
        # print("bert_output shape:{}".format(bert_output.shape))
        # print("qvec shape:{}".format(len(all_qvec)))

        # all_seq_vec (batch_size, seq_len, 2*vec_size)
        all_seq_vec = torch.cat((all_attn_vec, bert_output), dim=2)

        return all_seq_vec, num_time_seq


class GlobalEncoder(nn.Module):
    """
    计算globalscore
    """

    def __init__(self, batch_size, seq_len, vec_size, num_layers=2, batch_first=True, bilstm=True,
                 drop_prob=0.5) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vec_size = vec_size
        self.num_layers = num_layers
        self.bilstm = bilstm

        self.bilstm = nn.LSTM(input_size=vec_size,
                              hidden_size=vec_size,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              bidirectional=bilstm)
        if bilstm:
            self.liner1 = nn.Linear(2 * num_layers * vec_size, vec_size)
        else:
            self.liner1 = nn.Linear(num_layers * vec_size, vec_size)

        self.dropout = nn.Dropout(drop_prob)
        self.liner2 = nn.Linear(vec_size, seq_len)
        # self.liner3 = nn.Linear(vec_size, seq_len)
        # self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, l2_h_0, l2_c_0):
        # h_0 = h_0.permute(1,0,2)
        # c_0 = c_0.permute(1,0,2)
        _, (h_n, _) = self.bilstm(x, (l2_h_0, l2_c_0))
        h = h_n.permute(1, 0, 2)
        h = h.contiguous().view(self.batch_size, -1)

        h_output = self.liner1(F.relu(h))
        h_output = self.dropout(h_output)
        h_start = self.liner2(F.relu(h_output))
        # h_end = self.liner3(F.relu(h_output))
        # start_global_score = self.sigmoid(h_start)
        start_global_score = self.tanh(h_start)
        # end_global_score = self.sigmoid(h_end)

        # global_score (batch_size, seq_len, 1)
        # return start_global_score.unsqueeze(2), end_global_score.unsqueeze(2)
        return start_global_score.unsqueeze(2)


class SelfAttnLayer(nn.Module):
    def __init__(self, batch_size, seq_len, cated_vec_size) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        # 拼接后的特征长度
        self.vec_size = cated_vec_size
        self.linearq = nn.Linear(self.vec_size, self.vec_size)
        self.lineark = nn.Linear(self.vec_size, self.vec_size)
        self.linearv = nn.Linear(self.vec_size, self.vec_size)
        """
        self.Wq = nn.Parameter(torch.empty(self.batch_size, self.vec_size, self.vec_size))
        self.Bq = nn.Parameter(torch.empty(self.batch_size, self.vec_size, self.vec_size))
        self.Wk = nn.Parameter(torch.empty(self.batch_size, self.vec_size, self.vec_size))
        self.Bk = nn.Parameter(torch.empty(self.batch_size, self.vec_size, self.vec_size))
        self.Wv = nn.Parameter(torch.empty(self.batch_size, self.vec_size, self.vec_size))
        self.Bv = nn.Parameter(torch.empty(self.batch_size, self.vec_size, self.vec_size))
        """

    def forward(self, x):
        """
        q = torch.matmul(x, self.Wq) + self.Bq
        k = torch.matmul(x, self.Wk) + self.Bk
        v = torch.matmul(x, self.Wv) + self.Bv
        """
        q = self.linearq(x)
        k = self.lineark(x)
        v = self.linearv(x)

        a = torch.div(torch.matmul(q, k.permute(0, 2, 1)), math.sqrt(self.seq_len))
        a = torch.matmul(a, v)

        return a


class Extract(nn.Module):
    def __init__(self, batch_size, seq_len, cated_vec_size, num_labels, num_layers=2, batch_first=True, bilstm=True,
                 soft_label=False) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.soft_label = soft_label
        # self.training = training
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.vec_size = cated_vec_size

        self.attn = SelfAttnLayer(batch_size, seq_len, cated_vec_size)
        self.bilstm = nn.LSTM(input_size=cated_vec_size,
                              hidden_size=cated_vec_size,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              bidirectional=bilstm)

        self.globalencoder = GlobalEncoder(batch_size, seq_len, cated_vec_size)

        self.startlogits = StartLogits(2 * cated_vec_size, num_labels)
        if soft_label:
            self.endlogits = EndLogits(2 * cated_vec_size + num_labels, num_labels)
        else:
            self.endlogits = EndLogits(2 * cated_vec_size, num_labels)

    def forward(self, x, start_pos=None, end_pos=None, attention_mask=None):
        start_global_score, end_global_score = self.globalencoder(x)
        start_global_score.expand(x.shape[0], x.shape[1], self.num_labels)
        end_global_score.expand(x.shape[0], x.shape[1], self.num_labels)

        x_a = self.attn(x)
        if self.bilstm:
            h_0 = torch.randn(2 * self.num_layers, self.batch_size, self.vec_size)
            c_0 = torch.randn(2 * self.num_layers, self.batch_size, self.vec_size)
        else:
            h_0 = torch.randn(self.num_layers, self.batch_size, self.vec_size)
            c_0 = torch.randn(self.num_layers, self.batch_size, self.vec_size)
        x_lstm, _ = self.bilstm(x_a, (h_0, c_0))

        start_logits = torch.mul(self.startlogits(x_lstm), start_global_score)

        if start_pos is not None and self.training:
            if self.soft_label:
                batch_size = x.size(0)
                seq_len = x.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(x.device)
                label_logits.scatter_(2, start_pos.unsqueeze(2), 1)
            else:
                label_logits = start_pos.unsqueeze(2).float()
        else:
            label_logits = F.sigmoid(start_logits, -1)
            if not self.soft_label:
                label_logits = label_logits.ge(0.5).float().unsqueeze(2)
        end_logits = torch.mul(self.endlogits(x_lstm, label_logits), end_global_score)

        outputs = (start_logits, end_logits,)

        if start_pos is not None and end_pos is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.contiguous().view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_pos.contiguous().view(-1, self.num_labels).float()[active_loss]
            active_end_labels = end_pos.contiguous().view(-1, self.num_labels).float()[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        print("extract_output")

        return outputs


class PatientExtract(nn.Module):
    def __init__(self, batch_size, seq_len, hidden_size, num_labels, num_layers=2, batch_first=True, bilstm=True,
                 soft_label=False) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.soft_label = soft_label
        # self.training = training

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.vec_size = hidden_size

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)

        self.attn = SelfAttnLayer(batch_size, seq_len, hidden_size)
        self.bilstm = nn.LSTM(input_size=hidden_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              bidirectional=bilstm)

        self.globalencoder = GlobalEncoder(batch_size, seq_len, hidden_size)

        self.startlogits = StartLogits(2 * hidden_size, num_labels)
        self.linear = nn.Linear(num_labels, num_labels)
        self.softmax = nn.Softmax(dim=-1)
        """
        if soft_label:
            self.endlogits = EndLogits(2*hidden_size+num_labels, num_labels)
        else:
            self.endlogits = EndLogits(2*hidden_size, num_labels)
        """

    def forward(self, x, start_pos=None, end_pos=None, attention_mask=None, l1_h_0=None, l1_c_0=None, l2_h_0=None,
                l2_c_0=None):
        # h_0, c_0 = self.globalencoder.init_hidden_state()
        # start_global_score, end_global_score = self.globalencoder(x, l1_h_0, l1_c_0)
        start_global_score = self.globalencoder(x, l1_h_0, l1_c_0)

        start_global_score.expand(x.shape[0], x.shape[1], self.num_labels)
        # end_global_score.expand(x.shape[0], x.shape[1], self.num_labels)

        # print("start_global_score:{}".format(start_global_score))
        # print("end_global_score:{}".format(end_global_score))

        x_a = self.attn(x)

        x_lstm, _ = self.bilstm(x_a, (l2_h_0, l2_c_0))

        # print("x_lstm.shape {}".format(x_lstm.shape))
        # print("start_global_score {}".format(start_global_score.shape))

        start_logits_lstm = self.startlogits(x_lstm)
        start_logits_lstm_shape = start_logits_lstm.shape
        start_logits = torch.mul(start_logits_lstm,
                                 start_global_score.expand(start_logits_lstm_shape[0], start_logits_lstm_shape[1],
                                                           start_logits_lstm_shape[2]))

        start_logits = self.linear(start_logits)
        start_logits = self.softmax(start_logits)
        # print("start_logits_lstm:{}".format(start_logits_lstm))
        # print("start_globalscore:{}".format(start_global_score))

        """
        if start_pos is not None and self.training:
            if self.soft_label:
                batch_size = x.size(0)
                seq_len = x.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(x.device)
                label_logits.scatter_(2, start_pos.unsqueeze(2), 1)
            else:
                label_logits = start_pos.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits_lstm = self.endlogits(x_lstm, label_logits)
        end_logits = torch.mul(end_logits_lstm, end_global_score)

        outputs = (start_logits, end_logits,)
        """
        outputs = (start_logits,)

        if start_pos is not None:
            # loss_fct = nn.CrossEntropyLoss()

            # start_logits = torch.argmax(start_logits, dim=-1)
            # end_logits = torch.argmax(end_logits, dim=-1)
            # print("start_logits {}".format(start_logits.shape))

            start_logits = start_logits.view(-1, self.num_labels)
            # end_logits = end_logits.view(-1,self.num_labels)

            # print("end_logits_shape:{}".format(end_logits.shape))
            # print("end_logits:{}".format(end_logits))

            active_loss = attention_mask.contiguous().view(-1) == 1
            active_start_logits = start_logits[active_loss]
            # active_end_logits = end_logits[active_loss]

            active_start_labels = start_pos.contiguous().view(-1, 1).float()[active_loss]
            # active_end_labels = end_pos.contiguous().view(-1,1).float()[active_loss]

            # print(active_start_logits)
            # print(active_start_labels)
            # print("start_logits:{}".format(start_logits.shape))
            # print("activate_start_logits: {}".format(active_start_logits.shape))
            # print("activate_start_labels {}".format(active_start_labels.shape))

            start_loss = self.loss_fct(active_start_logits.float(), active_start_labels.squeeze(1).long())
            # end_loss = self.loss_fct(active_end_logits.float(), active_end_labels.squeeze(1).long())
            # total_loss = (start_loss + end_loss) / 2
            # outputs = (total_loss,) + outputs
            outputs = (start_loss,) + outputs

        return outputs


class EventExtract(nn.Module):
    def __init__(self, batch_size, seq_len, hidden_size, num_labels, soft_label=False) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.soft_label = soft_label
        # self.training = training

        # event type
        self.event = Extract(batch_size, seq_len, hidden_size, num_labels, soft_label=soft_label)
        self.onset = Extract(batch_size, seq_len, hidden_size, num_labels, soft_label=soft_label)
        self.hospitalvisit = Extract(batch_size, seq_len, hidden_size, num_labels, soft_label=soft_label)
        self.diagnosisconfirmed = Extract(batch_size, seq_len, hidden_size, num_labels, soft_label=soft_label)
        self.inpatient = Extract(batch_size, seq_len, hidden_size, num_labels, soft_label=soft_label)
        self.discharge = Extract(batch_size, seq_len, hidden_size, num_labels, soft_label=soft_label)
        self.death = Extract(batch_size, seq_len, hidden_size, num_labels, soft_label=soft_label)
        self.observed = Extract(batch_size, seq_len, hidden_size, num_labels, soft_label=soft_label)

        # event tuple
        self.date = Extract(batch_size, seq_len, hidden_size + 2 * 8, num_labels, soft_label=soft_label)
        self.symptom = Extract(batch_size, seq_len, hidden_size + 2 * 8, num_labels, soft_label=soft_label)
        self.labtest = Extract(batch_size, seq_len, hidden_size + 2 * 8, num_labels, soft_label=soft_label)
        self.imagingexamination = Extract(batch_size, seq_len, hidden_size + 2 * 8, num_labels, soft_label=soft_label)
        self.location = Extract(batch_size, seq_len, hidden_size + 2 * 8, num_labels, soft_label=soft_label)
        self.spot = Extract(batch_size, seq_len, hidden_size + 2 * 8, num_labels, soft_label=soft_label)
        self.vehicle = Extract(batch_size, seq_len, hidden_size + 2 * 8, num_labels, soft_label=soft_label)

    def forward(self, x, event_start_pos=None, event_end_pos=None,
                onset_start_pos=None, onset_end_pos=None,
                hospitalvisit_start_pos=None, hospitalvisit_end_pos=None,
                diagnosisconfirmed_start_pos=None, diagnosisconfirmed_end_pos=None,
                inpatient_start_pos=None, inpatient_end_pos=None,
                discharge_start_pos=None, discharge_end_pos=None,
                death_start_pos=None, death_end_pos=None,
                observed_start_pos=None, observed_end_pos=None,
                date_start_pos=None, date_end_pos=None,
                symptom_start_pos=None, symptom_end_pos=None,
                labtest_start_pos=None, labtest_end_pos=None,
                imagingexamination_start_pos=None, imagingexamination_end_pos=None,
                location_start_pos=None, location_end_pos=None,
                spot_start_pos=None, spot_end_pos=None,
                vehicle_start_pos=None, vehicle_end_pos=None,
                attention_mask=None):
        # event type
        event_output = self.event(x, event_start_pos, event_end_pos, attention_mask)
        onset_output = self.event(x, onset_start_pos, onset_end_pos, attention_mask)
        hospitalvisit_output = self.hospitalvisit(x, hospitalvisit_start_pos, hospitalvisit_end_pos, attention_mask)
        diagnosisconfirmed_output = self.diagnosisconfirmed(x, diagnosisconfirmed_start_pos, diagnosisconfirmed_end_pos,
                                                            attention_mask)
        inpatient_output = self.inpatient(x, inpatient_start_pos, inpatient_end_pos, attention_mask)
        discharge_output = self.discharge(x, discharge_start_pos, discharge_end_pos, attention_mask)
        death_output = self.death(x, death_start_pos, death_end_pos, attention_mask)
        observed_output = self.observed(x, observed_start_pos, observed_end_pos, attention_mask)

        event_type_loss = event_output[0] + onset_output[0] + hospitalvisit_output[0] + diagnosisconfirmed_output[0] + \
                          inpatient_output[0] + discharge_output[0] + death_output[0] + observed_output[0]
        start_cat = torch.cat([event_output[1], onset_output[1], hospitalvisit_output[1], diagnosisconfirmed_output[1],
                               inpatient_output[1], discharge_output[1], death_output[1], observed_output[1]], dim=-1)
        end_cat = torch.cat([event_output[2], onset_output[2], hospitalvisit_output[2], diagnosisconfirmed_output[2],
                             inpatient_output[2], discharge_output[2], death_output[2], observed_output[2]], dim=-1)

        # event tuple
        date_output = self.date(torch.cat([x, start_cat, end_cat], dim=-1), date_start_pos, date_end_pos,
                                attention_mask)
        symptom_output = self.symptom(torch.cat([x, start_cat, end_cat], dim=-1), symptom_start_pos, symptom_end_pos,
                                      attention_mask)
        labtest_output = self.labtest(torch.cat([x, start_cat, end_cat], dim=-1), labtest_start_pos, labtest_end_pos,
                                      attention_mask)
        imagingexamination_output = self.imagingexamination(torch.cat([x, start_cat, end_cat], dim=-1),
                                                            imagingexamination_start_pos, imagingexamination_end_pos,
                                                            attention_mask)
        location_output = self.location(torch.cat([x, start_cat, end_cat], dim=-1), location_start_pos,
                                        location_end_pos, attention_mask)
        spot_output = self.spot(torch.cat([x, start_cat, end_cat], dim=-1), spot_start_pos, spot_end_pos,
                                attention_mask)
        vehicle_output = self.vehicle(torch.cat([x, start_cat, end_cat], dim=-1), vehicle_start_pos, vehicle_end_pos,
                                      attention_mask)

        event_tuple_loss = date_output[0] + symptom_output[0] + labtest_output[0] + imagingexamination_output[0] + \
                           location_output[0] + spot_output[0] + location_output[0] + vehicle_output[0]
        loss = (event_type_loss + event_tuple_loss) / 15


        #第一个event实体只做占位使用，最终将被替换为1-所有output的prob，作为非实体位置label的prob，在argmax函数时作用生成label0。
        return (loss, event_output[1:],event_output[1:], onset_output[1:], hospitalvisit_output[1:], diagnosisconfirmed_output[1:],
                inpatient_output[1:], discharge_output[1:], death_output[1:], observed_output[1:], date_output[1:],
                symptom_output[1:], labtest_output[1:], imagingexamination_output[1:], location_output[1:],
                spot_output[1:], vehicle_output[1:])


class RelationExtract(nn.Module):
    def __init__(self, batch_size, seq_len, hidden_size, num_labels, soft_label=False) -> None:
        super().__init__()

        # relation type
        self.socialtelation = Extract(batch_size, seq_len, hidden_size, num_labels, soft_label=soft_label)

        # relation tuple
        self.localid = Extract(batch_size, seq_len, hidden_size + 2, num_labels, soft_label=soft_label)
        self.name = Extract(batch_size, seq_len, hidden_size + 2, num_labels, soft_label=soft_label)

    def forward(self, x, socialtelation_start_pos=None, socialtelation_end_pos=None,
                localid_start_pos=None, localid_end_pos=None,
                name_start_pos=None, name_end_pos=None,
                attention_mask=None):
        # relation type
        socialtelation_output = self.socialtelation(x, socialtelation_start_pos, socialtelation_end_pos, attention_mask)
        relation_type_loss = socialtelation_output[0]

        # relation tuple
        localid_output = self.localid(torch.cat([x, socialtelation_output[1], socialtelation_output[2]], dim=-1),
                                      localid_start_pos, localid_end_pos, attention_mask)
        name_output = self.name(torch.cat([x, socialtelation_output[1], socialtelation_output[2]], dim=-1),
                                name_start_pos, name_end_pos, attention_mask)
        relation_tuple_loss = localid_output[0] + name_output[0]

        loss = (relation_type_loss + relation_tuple_loss) / 3

        return (loss, socialtelation_output[1:], localid_output[1:], name_output[1:])


class JointExtract(nn.Module):
    def __init__(self, config, batch_size, model_path, seq_len, vec_size, cated_vec_size=None, num_layers=2,
                 batch_first=True, bilstm=True) -> None:
        super().__init__()
        self.bilstm = bilstm
        self.batch_size = batch_size
        self.vec_size = vec_size
        self.num_layers = num_layers
        self.config = config
        self.model_path = model_path

        self.bert = BertForJE(config=self.config, model_path=self.model_path)
        self.poolattention = PoolAttentionForJE(batch_size, seq_len, vec_size)
        # self.attn = SelfAttnLayer(batch_size, seq_len, cated_vec_size)
        # self.globalencoder = GlobalEncoder(batch_size, seq_len, vec_size)

        self.patientextract = PatientExtract(batch_size, seq_len, 2 * vec_size, num_labels=7)

        self.relationextract = PatientExtract(batch_size, seq_len, 2 * vec_size, num_labels=4)
        self.eventextract = PatientExtract(batch_size, seq_len, 2 * vec_size, num_labels=16)
        self.linear1 = nn.Linear(7, 7)
        self.linear2 = nn.Linear(4, 4)
        self.linear3 = nn.Linear(16, 16)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.softmax3 = nn.Softmax(dim=-1)

        """
        self.eventextract = EventExtract(batch_size, seq_len, 2*vec_size, num_labels=1)
        self.relationextract = RelationExtract(batch_size, seq_len, 2*vec_size, num_labels=1)
        """
        """
        self.bilstm = nn.LSTM(input_size=cated_vec_size,
                            hidden_size=cated_vec_size,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            bidirectional=bilstm)
        """

    """
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                time_start_pos=None, time_end_pos=None,
                patient_start_pos=None, patient_end_pos=None,
                event_start_pos=None, event_end_pos=None,
                onset_start_pos=None, onset_end_pos=None,
                hospitalvisit_start_pos=None, hospitalvisit_end_pos=None,
                diagnosisconfirmed_start_pos=None, diagnosisconfirmed_end_pos=None,
                inpatient_start_pos=None, inpatient_end_pos=None,
                discharge_start_pos=None, discharge_end_pos=None,
                death_start_pos=None, death_end_pos=None,
                observed_start_pos=None, observed_end_pos=None,
                date_start_pos=None, date_end_pos=None,
                symptom_start_pos=None, symptom_end_pos=None,
                labtest_start_pos=None, labtest_end_pos=None,
                imagingexamination_start_pos=None, imagingexamination_end_pos=None,
                location_start_pos=None, location_end_pos=None,
                spot_start_pos=None, spot_end_pos=None,
                vehicle_start_pos=None, vehicle_end_pos=None,
                socialrelation_start_pos=None, socialrelation_end_pos=None,
                localid_start_pos=None, localid_end_pos=None,
                name_start_pos=None, name_end_pos=None):
    """

    def init_hidden_state(self):
        if self.bilstm:
            h_0 = torch.randn(self.batch_size, 2 * self.num_layers, 2 * self.vec_size).to(device)
            c_0 = torch.randn(self.batch_size, 2 * self.num_layers, 2 * self.vec_size).to(device)
        else:
            h_0 = torch.randn(self.batch_size, 2 * self.num_layers, self.vec_size).to(device)
            c_0 = torch.randn(self.batch_size, 2 * self.num_layers, self.vec_size).to(device)

        return h_0, c_0

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                time_start_pos=None,
                patient_start_pos=None,
                relation_start_pos=None,
                event_start_pos=None,
                l1_h_0=None, l1_c_0=None,
                l2_h_0=None, l2_c_0=None):

        l1_h_0 = l1_h_0.permute(1, 0, 2).contiguous()
        l1_c_0 = l1_c_0.permute(1, 0, 2).contiguous()
        l2_h_0 = l2_h_0.permute(1, 0, 2).contiguous()
        l2_c_0 = l2_c_0.permute(1, 0, 2).contiguous()

        bert_outputs = self.bert(input_ids, attention_mask, token_type_ids,
                                 time_start_pos)

        bert_loss = bert_outputs[0]
        bert_start_logits = bert_outputs[1]
        # bert_end_logits = bert_outputs[2]
        # bert_feature_output = bert_outputs[3]
        bert_feature_output = bert_outputs[2]
        # print("bert_feature_output:{}, shape:{}".format(bert_feature_output, bert_feature_output.shape))

        if time_start_pos is not None and self.training:
            all_seq_vec, _ = self.poolattention(bert_feature_output, time_start_pos)
        else:
            all_seq_vec, _ = self.poolattention(bert_feature_output, bert_start_logits.ge(0.5).float())

        patient_outputs = self.patientextract(all_seq_vec, patient_start_pos, attention_mask=attention_mask,
                                              l1_h_0=l1_h_0, l1_c_0=l1_c_0, l2_h_0=l2_h_0, l2_c_0=l2_c_0)
        patient_loss = patient_outputs[0]
        patient_pos = patient_outputs[1:]

        #print("patient_output")
        """
        event_outputs = self.eventextract(all_seq_vec, event_start_pos, event_end_pos,
                onset_start_pos, onset_end_pos,
                hospitalvisit_start_pos, hospitalvisit_end_pos,
                diagnosisconfirmed_start_pos, diagnosisconfirmed_end_pos,
                inpatient_start_pos, inpatient_end_pos,
                discharge_start_pos, discharge_end_pos,
                death_start_pos, death_end_pos,
                observed_start_pos, observed_end_pos,
                date_start_pos, date_end_pos,
                symptom_start_pos, symptom_end_pos,
                labtest_start_pos, labtest_end_pos,
                imagingexamination_start_pos, imagingexamination_end_pos,
                location_start_pos, location_end_pos,
                spot_start_pos, spot_end_pos,
                vehicle_start_pos, vehicle_end_pos,
                attention_mask=attention_mask)
        event_loss = event_outputs[0]
        event_pos = event_outputs[1:]


        print("event_output")

        """
        """
        relation_outputs = self.relationextract(all_seq_vec, socialrelation_start_pos, socialrelation_end_pos,
                localid_start_pos, localid_end_pos,
                name_start_pos, name_end_pos,
                attention_mask=attention_mask)
        relation_loss = relation_outputs[0]
        relation_pos = relation_outputs[1:]
        """
        relation_outputs = self.relationextract(all_seq_vec, relation_start_pos, attention_mask=attention_mask,
                                                l1_h_0=l1_h_0, l1_c_0=l1_c_0, l2_h_0=l2_h_0, l2_c_0=l2_c_0)
        relation_loss = relation_outputs[0]
        relation_pos = relation_outputs[1:]
        #print("relation_output")

        event_outputs = self.eventextract(all_seq_vec, event_start_pos, attention_mask=attention_mask, l1_h_0=l1_h_0,
                                          l1_c_0=l1_c_0, l2_h_0=l2_h_0, l2_c_0=l2_c_0)
        event_loss = event_outputs[0]
        event_pos = event_outputs[1:]
        print("event_output")

        tmp_pos=event_pos[0].clone().detach()
        #patient_p in range(event_pos[0].shape[0]):
        for i in range(tmp_pos.shape[0]):
            for j in range(tmp_pos.shape[1]):
                tmp_pos[i][j][1] = 1 - sum(tmp_pos[i][j][1:tmp_pos.shape[2]])

        # return (bert_loss, patient_loss, patient_pos)
        # return (bert_loss, patient_loss, event_loss, relation_loss, patient_pos, event_pos, relation_pos)
        return (bert_loss, patient_loss, relation_loss, event_loss, patient_pos, relation_pos, tmp_pos)
