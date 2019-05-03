import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class RNNClassifier(nn.Module):
    def __init__(self, pre_trained):
        super().__init__()
        
        self.embedding = nn.Embedding(10004, 300, padding_idx=0)
        self.embedding.weight.data.copy_(pre_trained)
        self.embedding.weight.requires_grad=False

        self.rnn = nn.GRU(300, 512, num_layers=1, bidirectional=True)
        self.fc1 = nn.Linear(512*2, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # use pack_sequence for the best performance
        x = rnn_utils.pack_sequence(x)
        #x = rnn_utils.pad_sequence(x).cuda()
        embedded = rnn_utils.PackedSequence(self.embedding(x.data), x.batch_sizes)
        
        #embedded = self.embedding(x)
    
        _, hidden = self.rnn(embedded)
        
        #hidden, _ = rnn_utils.pad_packed_sequence(hidden)
        
        hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        out = self.fc1(hidden)
        return out.squeeze()