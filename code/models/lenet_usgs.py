import torch.nn as nn
import math
import torch

class PyramidLeNet(nn.Module):
    """Pyramid-shaped LeNets"""
    def __init__(self, settings):
        super(PyramidLeNet, self).__init__()
        self.settings = settings
        self.features = self._build_conv_layers()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size()[0], -1)
        return out

    def _build_conv_layers(self):
        """[16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M']
        """
        conv_layers = []
        in_channels = 1
        negative_slope = 5e-2
        self.kernel_size = 21
        for x in self.settings['ConvLayers']:
            if x == 'M':
                conv_layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
                # self.kernel_size = self.kernel_size - 4
                self.kernel_size = math.ceil(self.kernel_size*0.5)
            else:
                conv_layers += [nn.Conv1d(in_channels, x,
                                    kernel_size=self.kernel_size, stride=1,
                                    padding=math.floor(self.kernel_size*0.5))]
                # conv_layers += [nn.LeakyReLU(negative_slope=negative_slope)]
                conv_layers += [nn.BatchNorm1d(x)]
                conv_layers += [nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*conv_layers)


class LeNet(nn.Module):
    def __init__(self, settings):
        super(LeNet, self).__init__()
        self.settings = settings
        self.features = PyramidLeNet(settings=settings)
        self.num_classes = settings['class_num']
        self.embedding_dim = settings['linear_dim']  #8256  16512
        self.head = self._make_head()
        self.loss = nn.CrossEntropyLoss()

        self.liner_1 = nn.Linear(self.embedding_dim, 2048)
        self.pooling = nn.MaxPool1d(kernel_size = 2,stride = 2)
        self.bn_1 = nn.BatchNorm1d(2048)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.liner_2 = nn.Linear(1024, self.num_classes)
        self.bn_2 = nn.BatchNorm1d(self.num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x, y=None):
        if y != None:
            x = torch.cat((x,y),0)
        out = self.features(x)      
        # out = out.view(out.size()[0], -1)
        # out = self.head(out)
        out = self.liner_1(out)
        out = self.bn_1(out)
        out = self.tanh(out)
        out = torch.unsqueeze(out, 1)
        out = self.pooling(out)
        out = out.view(out.size()[0], -1)
        out = self.dropout(out)
        out = self.liner_2(out)
        out = self.bn_2(out)
        out = self.softmax(out)
       
        return out

    def _make_head(self):
        head = []
        head.append(nn.Linear(self.embedding_dim, 2048))
        head.append(nn.BatchNorm1d(2048))
        head.append(nn.Tanh())
        return nn.Sequential(*head)
        
