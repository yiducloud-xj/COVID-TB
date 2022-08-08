import torch
import torch.nn as nn
import torch.nn.functional as F


class StartLogits(nn.Module):
    def __init__(self, hidden_size, num_labels) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.act_fc = nn.ReLU()
        self.liner = nn.Linear(self.hidden_size, self.num_labels)


    def forward(self, hidden_states):
        output = self.liner(self.act_fc(hidden_states))

        return output



class EndLogits(nn.Module):
    def __init__(self, hidden_size, num_labels) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.liner1 = nn.Linear(hidden_size+1, hidden_size)
        self.act_fc = nn.Tanh()
        self.laynorm = nn.LayerNorm(hidden_size)
        self.liner2 = nn.Linear(hidden_size, num_labels)

    
    def forward(self, hidden_states, start_pos=None):
        x = torch.cat([hidden_states, start_pos], dim=-1)
        x = self.liner1(x)
        x = self.act_fc(x)
        x = self.laynorm(x)
        x = self.liner2(x)

        return x