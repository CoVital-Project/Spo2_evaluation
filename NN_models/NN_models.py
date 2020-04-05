# Code adapted from: https://github.com/bhpfelix/PyTorch-Time-Series-Classification-Benchmarks/
# Added mobilenet_v2
import shutil, os, csv, itertools, glob
import sys
sys.path.append("./data_loader")
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from data_loader import Spo2Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
import pandas as pd
import pickle as pk

cuda = torch.cuda.is_available()
# Utils

def load_pickle(filename):
    try:
        p = open(filename, 'r')
    except IOError:
        print("Pickle file cannot be opened.")
        return None
    try:
        picklelicious = pk.load(p)
    except ValueError:
        print('load_pickle failed once, trying again')
        p.close()
        p = open(filename, 'r')
        picklelicious = pk.load(p)

    p.close()
    return picklelicious

def save_pickle(data_object, filename):
    pickle_file = open(filename, 'w')
    pk.dump(data_object, pickle_file)
    pickle_file.close()
    
def read_data(filename):
    print("Loading Data...")
    df = pd.read_csv(filename, header=None)
    data = df.values
    return data

def read_line(csvfile, line):
    with open(csvfile, 'r') as f:
        data = next(itertools.islice(csv.reader(f), line, None))
    return data
#####################################################################################################################
#  Classifiers: Code directly modified based on the PyTorch Model Zoo Implementations
#####################################################################################################################

## 1D variant of VGG model take 200 dimensional fixed time series inputs
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


## 1D Variant of ResNet taking in 200 dimensional fixed time series inputs
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # print('out', out.size(), 'res', residual.size(), self.downsample)
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, arch):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1]) #, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2]) #, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3]) #, stride=2)
        self.avgpool = nn.AvgPool1d(7, stride=1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(22528),
            self.Flatten(),
            nn.Linear(22528 , num_classes),
            nn.Sigmoid()
        )
        
        self.arch = arch

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    class Flatten(nn.Module):
        def forward(self, input):
            return input.squeeze()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x.reshape(x.shape[0],-1, x.shape[2]))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(1, x.size(0), -1)
        # print(x.size())
        x = self.fc(x)

        return x*100

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], arch='resnet18', **kwargs)

    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], arch='resnet34', **kwargs)

    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], arch='resnet50', **kwargs)

    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], arch='resnet101', **kwargs)

    return model

def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], arch='resnet152', **kwargs)
    
    return model
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=2,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
            nn.Sigmoid(),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x*100

    def forward(self, x):
        return self._forward_impl(x)

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    model.arch = 'mobilenet2'
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


#####################################################################################################################
#  Trainer and Core Experiment Scripts
#####################################################################################################################

def run_trainer(experiment_path, model_path, model, train_loader, test_loader, get_acc, resume, batch_size, num_epoch):

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    def save_checkpoint(state, is_best, filename=model_path+'checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, model_path+'model_best.pth.tar')
    def get_last_checkpoint(model_path):
        fs = sorted([f for f in os.listdir(model_path) if 'Epoch' in f], key=lambda k: int(k.split()[1]))
        return model_path+fs[-1] if len(fs) > 0 else None
    
    start_epoch = 0
    best_res = 100000
    lrcurve = []
    conf_mats = []
    resume_state = get_last_checkpoint(model_path) if resume else None
    if resume_state and os.path.isfile(resume_state):
        print("=> loading checkpoint '{}'".format(resume_state))
        checkpoint = torch.load(resume_state)
        start_epoch = checkpoint['epoch']+1
        best_res = checkpoint['val_acc']
        lrcurve = checkpoint['lrcurve']
        conf_mats = checkpoint['conf_mats']
        model.load_state_dict(checkpoint['state_dict'])
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr = 5e-4)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_state, checkpoint['epoch']))
    else:
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr = 5e-4)

    criterion = nn.L1Loss()
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5) # optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    def train(epoch):
        model.train()
        total, total_correct = 0., 0.
        loss_list = []
        for batch_idx, (data,target,_) in enumerate(train_loader):
            #data, target = Variable(data.float()), Variable(target.long())
            if cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data.reshape(data.shape[0],3,-1,2))
            loss = criterion(output[:,0], target[:,0])
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data.item())
            metric, correct, num_instance, conf_mat = get_acc(output[:,0], target[:,0])
            '''if batch_idx % 2 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Metric: {:.2f}%'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item(), metric))'''
        print('Average Loss: {:.6f} Metric: {:.2f}%'.format(
            np.average(loss_list), np.average(metric)))
        return metric

    def test():
        model.eval()
        test_loss = 0.
        total, total_correct = [], []
        preds = []
        for data,target,_ in test_loader:
            #data, target = Variable(data.float(), volatile=True), Variable(target.long())
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data.reshape(data.shape[0],3,-1,2))
            test_loss += criterion(output, target).data.item() # sum up batch loss
            
            total_correct.extend(target[:,0])
            total.extend(output[:,0])
            preds.extend(output[:,0])
        metric, correct, num_instance, conf_mat = get_acc(torch.stack(total), torch.stack(total_correct))
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Metric: ({:.2f}%)\n'.format(
            test_loss,
            metric))

        return torch.stack(preds), metric

    for epoch in range(start_epoch, num_epoch):
        is_best = False

        train_acc = train(epoch)
        preds, val_acc = test()
        
        #print("Training Confmat: ")
        #print(train_conf)
        #print("Testing Confmat: ")
        #print(val_conf)
        #print("Number of Predictions Made: ")
        #print(preds.shape)
        
        lrcurve.append((train_acc, val_acc))
        # scheduler.step(val_loss)

        if val_acc < best_res:
            best_res = val_acc
            is_best = True
            save_checkpoint({
                    'epoch': epoch,
                    'arch': model.arch,
                    'state_dict': model.cpu().state_dict(),
                    'train_acc':train_acc,
                    'val_acc': val_acc,
                    'optimizer' : optimizer.state_dict(),
                    'lrcurve':lrcurve,
                    'test_predictions':preds,
                }, is_best,
                model_path+"Epoch %d Acc %.4f.pt"%(epoch, val_acc))

        if cuda:
            model.cuda()
            
    return lrcurve


def run_experiment(experiment_path, data_path, model_root, models, norm, get_acc, resume=False, num_epoch=10):
    
    exp_result = {}
    for batch_size, model in models:
        print("Running %s" % model.arch)
        
        print('Loading Data..')
        train_data = Spo2Dataset('nemcova_data/train')
        test_data = Spo2Dataset('nemcova_data/test')

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=spo2_collate_fn)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=spo2_collate_fn)
    
        model_path = os.path.join(
            experiment_path, 
            model_root,
            'norm' if norm else 'nonorm',
            model.arch) + '/'
        lrcurve = run_trainer(experiment_path, model_path, model, train_loader, test_loader, get_acc, resume, batch_size, num_epoch)
        exp_result[model.arch] = {'lrcurve':lrcurve}
        
    return exp_result
#####################################################################################################################
#  Dummy Experiment for Testing Purpose
#####################################################################################################################

def get_models():
    return [
        #(4, resnet18(num_classes=2)),
        #(6, vgg13bn(num_classes=2)),
        (4, mobilenet_v2(False, True))
    ]

def get_acc(output, target):
    # takes in two tensors to compute accuracy
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct = pred.eq(target.data.view_as(pred)).cpu().sum()
    conf_mat = confusion_matrix(pred.cpu().numpy(), target.data.cpu().numpy(), labels=range(15))
    return pred.cpu().numpy(), correct, target.size(0), conf_mat

def get_mae(output, target):
    # takes in two tensors to compute Mean Absolute Error
    mae = mean_absolute_error(target.cpu().detach().numpy(), output.cpu().detach().numpy())
    return mae, output.cpu().detach().numpy(), target.size(0), None

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


experiments = {
        'experiment_path':'roll', 
        'data_path':'roll/roll',
        'model_root':'model', 
        'models':get_models(),
        'norm':False, 
        'get_acc': get_mae,
        'resume':False,  
        'num_epoch':1000
    }
run_experiment(**experiments)