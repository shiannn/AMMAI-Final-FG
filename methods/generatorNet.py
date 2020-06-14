import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class GeneratorNet(nn.Module):
    def __init__(self, num_classes=5, dim=512, norm=True, scale=True):
        super(GeneratorNet, self).__init__()
        self.dim = dim
        self.generator = Generator(dim = self.dim)
        self.add_info = AddInfo(dim = self.dim)
        self.fc = nn.Linear(self.dim, num_classes, bias=False)
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, B1=None, B2=None, A=None, classifier=False):
        #print('B1.shape', B1.shape)
        #print('B2.shape', B2.shape)
        #print('A.shape', A.shape)
        add_info = self.add_info(A, B1, B2)
        A_rebuild = self.generator(add_info)
        A_rebuild = self.l2_norm(A_rebuild)
        score = self.fc(A_rebuild*self.s)
        return A_rebuild, score
   
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        norm = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(norm)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

class AddInfo(nn.Module):
    def __init__(self, dim=512):
        super(AddInfo, self).__init__()
        self.dim = dim
        self.fc = nn.Linear(self.dim, self.dim*2)
        self.leakyRelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, A, B1, B2):
        A = self.fc(A)
        A = self.leakyRelu1(A)
        B1 = self.fc(B1)
        B1 = self.leakyRelu1(B1)
        B2 = self.fc(B2)
        B2 = self.leakyRelu1(B2)

        #print('addinfo B1.shape', B1.shape)
        #print('addinfo B2.shape', B2.shape)
        #print('addinfo A.shape', A.shape)
        out = A+(B1-B2)
        # print(torch.abs(B1-B2))
        out = self.dropout(out)

        return out

class Generator(nn.Module):
    def __init__(self, dim=512):
        super(Generator, self).__init__()
        self.dim = dim
        self.fc = nn.Linear(self.dim*2, self.dim)
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.fc(x)
        out = self.leakyRelu(out)
        out = self.dropout(out)

        return out