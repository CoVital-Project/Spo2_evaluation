import torch.nn as nn
import math

## 1D variant of VGG model take X dimensional time series inputs
class VGG(nn.Module):

    def __init__(self, features, num_classes, arch='vgg'):
        super(VGG, self).__init__()
        self.arch = arch
        self.features = features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(512 * 6),
            self.Flatten(),
            nn.Linear(512 * 6, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
            nn.Sigmoid(),
        )
        self._initialize_weights()
    class Flatten(nn.Module):
        def forward(self, input):
            return input.squeeze()
    def forward(self, x):
        x = self.features(x.reshape(x.shape[0],-1, x.shape[2]))
        x = x.view(1, x.size(0), -1)
        x = self.classifier(x)
        return x*100

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 6
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg13bn(**kwargs):
    model = VGG(make_layers(cfg['B'], batch_norm=True), arch='vgg13bn', **kwargs)
    return model

def vgg16bn(**kwargs):
    model = VGG(make_layers(cfg['D'], batch_norm=True), arch='vgg16bn', **kwargs)
    return model

def vgg19bn(**kwargs):
    model = VGG(make_layers(cfg['E'], batch_norm=True), arch='vgg19bn', **kwargs)
    return model
