import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        # self.init_weights()
        self.apply(self.init_weights)
        
    # def init_weights(self):
    def init_weights(self, m):  
        # same initialization as keras. Adapted from the initialization developed 
        # by JUN KODA (https://www.kaggle.com/junkoda) in this notebook
        # https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization
        # for name, params in self.named_parameters():
        for name, params in m.named_parameters():
            # print(name, params)
            if "weight_ih" in name: 
                nn.init.xavier_normal_(params)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(params)
            elif 'bias_ih' in name:
                params.data.fill_(0)
                # Set forget-gate bias to 1
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                params.data.fill_(0)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)   # [batch_size, seq_len] -> [batch_size, 1, seq_len]
        x = x.transpose(2,1)        # [batch_size, 1, seq_len] -> [batch_size, seq_len, 1]
        output, _ = self.lstm(x)    # [batch_size, seq_len, hidden_size]
        output = output[:, -1]      # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size] output last sequence 
        output = self.fc(output)
        return torch.squeeze(output, 1)
