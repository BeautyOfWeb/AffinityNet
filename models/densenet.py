import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_feature_input, growth_rate, bn_size, dropout_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_feature_input))
        self.add_module('relu.1', nn.ReLU(inplace = True))
        self.add_module('conv.1', nn.Conv2d(num_feature_input, bn_size * growth_rate, 1, 1, bias = False))
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu.2', nn.ReLU(inplace = True))
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate, 3, 1, padding = 1, bias = False))
        self.dropout_rate = dropout_rate
    def forward(self, input):
        new_features = super(_DenseLayer, self).forward(input)
        if self.dropout_rate > 0:
            new_features = F.dropout2d(new_features, p = self.dropout_rate, training = self.training)
        return torch.cat([input, new_features], 1) # pylint: disable=maybe-no-member


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size, dropout_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i + 1), 
                            _DenseLayer(num_input_features + growth_rate * i, 
                                        growth_rate, bn_size, dropout_rate))


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, pool_param = (2, 2, 0)):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, 1, 1, bias = False))
        self.add_module('relu', nn.ReLU(inplace = True))
        kernel_size, stride, padding = pool_param
        self.add_module('pool', nn.AvgPool2d(kernel_size, stride, padding))


class DenseNet(nn.Module):
    def __init__(self, input_param, block_layers, num_classes, growth_rate, bn_size, dropout_rate, transition_pool_param = (2, 2, 0)):
        super(DenseNet, self).__init__()
        num_input_features, num_out_features = input_param
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_input_features, num_out_features, 7, 2, padding = 3, bias = False)),
            ('norm0', nn.BatchNorm2d(num_out_features)),
            ('relu0', nn.ReLU(inplace = True)),
            ('pool0', nn.MaxPool2d(3, 2, padding = 1))
        ]))
        for i, num_layers in enumerate(block_layers):
            self.features.add_module('denseblock%d' % (i + 1), _DenseBlock(num_layers, num_out_features, 
                                                                          growth_rate, bn_size, dropout_rate))
            num_out_features += growth_rate * num_layers
            if i != len(block_layers) - 1:
                self.features.add_module('transition%d' % (i + 1), _Transition(num_out_features, num_out_features // 2, transition_pool_param))
                num_out_features = num_out_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_out_features))
        self.classifier = nn.Linear(num_out_features, num_classes)
    
    def forward(self, input):
        features = self.features(input)
        N, C, H, W = features.size()
        out = F.relu(features, inplace = True)
        out = F.avg_pool2d(out, kernel_size = (H, W)).view(N, C)
        out = self.classifier(out)
        return out


#import torch.utils.model_zoo as model_zoo
#model = DenseNet((3, 64), (6, 12, 24, 16), 1000, 32, 4, 0)
#model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/densenet121-a639ec97.pth'))
#x = Variable(torch.ones(1, 3, 224, 224))
#model(x).size()