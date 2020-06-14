# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from methods.generatorNet import GeneratorNet

import utils

class DTN_ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(DTN_ProtoNet, self).__init__( model_func,  n_way, n_support)
        self.loss_fn  = nn.CrossEntropyLoss()
        self.model_G = GeneratorNet(num_classes=5, dim=512, norm=True, scale=True)

    def train_loop(self, epoch, train_loader, gen_loader, optimizer ):
        print_freq = 1

        avg_loss=0
        for idx, (x,_ ) in enumerate(train_loader):
            gen_x, gen_label = next(iter(gen_loader))
            #print(x.shape)
            #print(gen_x.shape)
            #print(gen_label.shape)
            #print(gen_label)
            generated_support_1 = gen_x[:,0,:,:,:]
            generated_support_2 = gen_x[:,1,:,:,:]
            ### [5, 21, 3, 244, 244] n_support 5, n_query 16 => 21
            ### from all 64 classes random pick 5 classes for training
            self.gen_num = generated_support_1.size(0)
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            z_support, z_query  = self.parse_feature(x,is_feature=False)
            #print(z_support.shape)
            # 5, 5, 512  => n_way 5 n_support 5 dimension 512
            #print(z_query.shape)
            # 5, 16, 512 => n_way 5 n_query 16 dimension 512
            gen_support_1 = self.forward(generated_support_1.cuda())
            gen_support_2 = self.forward(generated_support_2.cuda())
            self.dim = gen_support_2.shape[1]
            #print(gen_support_1.shape)
            # 6, 512 => 6 classes of generation 1
            #print(gen_support_2.shape)
            # 6, 512 => 6 classes of generation 2

            weight = torch.zeros((self.n_way, self.dim), requires_grad=True).cuda()
            for i in range(self.n_way):
                weight_point = torch.zeros((self.n_support)*(self.gen_num+1), self.dim)
                for j in range(self.n_support):
                    gen_feature, _ = self.model_G(gen_support_1, gen_support_2, z_support[i,j])
                    features = torch.cat((gen_feature, z_support[i,j].unsqueeze(0)), 0)
                    weight_point[j*(self.gen_num+1):(j+1)*(self.gen_num+1)] = features
                    #print(gen_feature.shape)
                weight[i] = torch.mean(weight_point, 0)
            
            #print(weight)
            weight = self.model_G.l2_norm(weight)
            predict = torch.matmul(z_query, torch.transpose(weight,0,1))*self.model_G.s
            predict = predict.view(-1, predict.shape[2])
            #print(weight)
            #print(predict.shape)

            y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
            y_query = Variable(y_query.cuda())
            loss = self.loss_fn(predict, y_query)
            #print(loss)
            #exit(0)
            
            optimizer.zero_grad()
            #loss = self.set_forward_loss( x )
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

            if idx % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, idx, len(train_loader), avg_loss/float(idx+1)))


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )


        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
        loss = self.loss_fn(scores, y_query)

        return loss

def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
