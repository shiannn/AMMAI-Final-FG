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
        origin_feature = A
        ways_nums = A.shape[0]
        shot_nums = A.shape[1]
        gen_nums = B1.shape[0]
        # clone B1 B2
        B1 = B1.expand(ways_nums, B1.shape[0], B1.shape[1])
        B2 = B2.expand(ways_nums, B2.shape[0], B2.shape[1])
        #print('after expand B1.shape', B1.shape)
        #print('after expand B2.shape', B2.shape)
        B1 = B1.repeat(1,shot_nums,1)
        B2 = B2.repeat(1,shot_nums,1)
        A = A.repeat_interleave(gen_nums, dim=1)
        #print('after repeat B1.shape', B1.shape)
        #print('after repeat B2.shape', B2.shape)
        #print('after repeat A.shape', A.shape)
        add_info = self.add_info(A, B1, B2)
        #print('after add_info', add_info.shape)
        A_rebuild = self.generator(add_info)
        #print('after A_rebuild', A_rebuild.shape)
        A_rebuild = self.l2_norm_3D(A_rebuild)
        #print('A_rebuild.shape',A_rebuild.shape)
        score = self.fc(A_rebuild*self.s)
        #print('score.shape', score.shape)

        return A_rebuild, score
   
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def l2_norm_3D(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        norm = torch.sum(buffer, 2).add_(1e-10)
        norm = torch.sqrt(norm)

        #print('norm.shape', norm.shape)
        _output = torch.div(input, norm.view(norm.shape[0], norm.shape[1], 1).expand_as(input))
        output = _output.view(input_size)

        return output

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