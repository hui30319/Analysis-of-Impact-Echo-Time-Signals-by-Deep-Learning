import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TCN(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super().__init__()
        self.tcn = TemporalConvNet(1, num_channels, kernel_size=kernel_size)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)    
        self.linear = nn.Linear(num_channels[-1], 1)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.tcn(x)

        x = self.avgpool(x)
        x = torch.squeeze(x, -1)
        x = self.linear(x) 
        return torch.squeeze(x, 1)
    
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)

        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(out + res)
        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()