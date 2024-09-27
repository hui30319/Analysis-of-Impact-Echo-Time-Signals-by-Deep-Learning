import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, num_hiddens, dropout_list):
        super().__init__()
        self.mlp = LinearNet(in_features, num_hiddens, dropout_list)
        self.dropout = nn.Dropout(p=dropout_list[-1], inplace=False)
        self.linear = nn.Linear(num_hiddens[-1], 1)

    def forward(self, x):
        out = self.mlp(x)
        out = self.dropout(out)
        out = self.linear(out)
        return torch.squeeze(out, 1)

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_hiddens, dropout_list):
        super(LinearNet, self).__init__()
        layers = []
        for i in range(len(num_hiddens)):
            in_channels = num_inputs if i == 0 else num_hiddens[i-1]
            out_channels = num_hiddens[i]
            dropout = dropout_list[i]
            layers += [LinearBlock(in_channels, out_channels, dropout)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.dropout(x)
        out = self.linear(out)
        out = self.relu(out)
        return out

# model = MLP(500, [500,500,500], [0.1,0.2,0.2,0.])
# x = torch.randn(32, 500)
# print(model)
# output = model(x)
# print(f"Output shape: {output.shape}")  