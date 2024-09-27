import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, num_channels, kernel_sizes):
        super().__init__()
        self.resnet = ResConvNet(1, num_channels, kernel_sizes=kernel_sizes)
        
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
        x = self.resnet(x)
        x = self.avgpool(x)
        x = torch.squeeze(x, -1)
        x = self.linear(x) 
        return torch.squeeze(x, 1)
    
class ResConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes):
        super(ResConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [ResBlock(in_channels, out_channels, kernel_sizes)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class ResBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes):
        super(ResBlock, self).__init__()
        # self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_sizes[0], padding=kernel_sizes[0]//2, bias=False)
        # self.bn1 = nn.BatchNorm1d(n_outputs)
        # self.relu1 = nn.ReLU()

        # self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_sizes[1], padding=kernel_sizes[1]//2, bias=False)
        # self.bn2 = nn.BatchNorm1d(n_outputs)
        # self.relu2 = nn.ReLU()

        # self.conv3 = nn.Conv1d(n_outputs, n_outputs, kernel_sizes[2], padding=kernel_sizes[2]//2, bias=False)
        # self.bn3 = nn.BatchNorm1d(n_outputs)

        # self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=False) if n_inputs != n_outputs else None
        # self.bn = nn.BatchNorm1d(n_outputs)
        # self.relu = nn.ReLU()

        self.convblock1 = ConvBlock(n_inputs, n_outputs, kernel_sizes[0], relu=True)
        self.convblock2 = ConvBlock(n_outputs, n_outputs, kernel_sizes[1], relu=True)
        self.convblock3 = ConvBlock(n_outputs, n_outputs, kernel_sizes[2], relu=False)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=False) if n_inputs != n_outputs else None
        self.bn = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # nn.init.kaiming_uniform_(self.conv1.weight)
        # self.bn1.weight.data.fill_(1.)
        # self.bn1.bias.data.fill_(1e-3)
        
        # nn.init.kaiming_uniform_(self.conv2.weight)
        # self.bn2.weight.data.fill_(1.)
        # self.bn2.bias.data.fill_(1e-3)
        
        # self.bn3.weight.data.fill_(1.)
        # self.bn3.bias.data.fill_(1e-3)
        
        self.bn.weight.data.fill_(1.)
        self.bn.bias.data.fill_(1e-3)

    def forward(self, x):
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu1(out)
        
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu2(out)

        # out = self.conv3(out)
        # out = self.bn3(out)
        out = self.convblock1(x)
        out = self.convblock2(out)
        out = self.convblock3(out)
        res = self.bn(x) if self.downsample is None else self.bn(self.downsample(x))
        out = self.relu(out + res)
        return out

class ConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, relu):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU() if relu else None
        
        self.init_weights()

    def init_weights(self):
        if self.relu:
            # nn.init.kaiming_uniform_(self.conv.weight)
            nn.init.kaiming_normal_(self.conv.weight)
        self.bn.bias.data.fill_(1e-3)
        self.bn.weight.data.fill_(1.)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.relu:
            out = self.relu(out)
        return out