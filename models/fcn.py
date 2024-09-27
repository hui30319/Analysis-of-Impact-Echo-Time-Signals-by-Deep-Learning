import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, num_channels, kernel_sizes):
        super().__init__()
        self.fcn = ConvNet(1, num_channels, kernel_sizes=kernel_sizes)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)    
        self.linear = nn.Linear(num_channels[-1], 1)
        # self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv1d):
    #             nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1.)
    #             nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.fcn(x)
        x = self.avgpool(x)
        x = torch.squeeze(x, -1)
        x = self.linear(x) 
        return torch.squeeze(x, 1)
    
class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes):
        super(ConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            kernel_size = kernel_sizes[i]
            layers += [ConvBlock(in_channels, out_channels, kernel_size)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class ConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        
        self.init_weights()

    def init_weights(self):
        # nn.init.kaiming_uniform_(self.conv.weight)
        nn.init.kaiming_normal_(self.conv.weight)
        self.bn.bias.data.fill_(1e-3)
        self.bn.weight.data.fill_(1.)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
        