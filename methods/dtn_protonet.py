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

    def correct(self, x, generated_support_1, generated_support_2):       
        scores  = self.set_forward(x, generated_support_1, generated_support_2)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)
    
    def test_loop(self, test_loader, val_gen_loader, record = None):
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        for idx, ZIP in enumerate(zip(test_loader, val_gen_loader)):
            test_data, gen_data = ZIP
            x, _ = test_data
            gen_x, gen_label = gen_data

            generated_support_1 = gen_x[:,0,:,:,:]
            generated_support_2 = gen_x[:,1,:,:,:]

            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this = self.correct(x, generated_support_1, generated_support_2)
            acc_all.append(correct_this/ count_this*100  )

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        return acc_mean


    def train_loop(self, epoch, train_loader, gen_loader, optimizer ):
        print_freq = 1

        avg_loss=0
        for idx, ZIP in enumerate(zip(train_loader, gen_loader)):
            #gen_x, gen_label = next(iter())
            train_data, gen_data = ZIP
            x, _ = train_data
            gen_x, gen_label = gen_data
            
            generated_support_1 = gen_x[:,0,:,:,:]
            generated_support_2 = gen_x[:,1,:,:,:]
            
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            
            optimizer.zero_grad()
            loss = self.set_forward_loss(x, generated_support_1, generated_support_2)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

            if idx % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, idx, len(train_loader), avg_loss/float(idx+1)))
            
            #break


    def set_forward(self,x, generated_support_1, generated_support_2, is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature=False)
        # 5, 5, 512  => n_way 5 n_support 5 dimension 512
        # 5, 16, 512 => n_way 5 n_query 16 dimension 512
        self.gen_num = generated_support_1.size(0)
        gen_support_1 = self.forward(generated_support_1.cuda())
        gen_support_2 = self.forward(generated_support_2.cuda())
        self.dim = gen_support_2.shape[1]
        # 6, 512 => 6 classes of generation 1
        # 6, 512 => 6 classes of generation 2

        gen_feature, _ = self.model_G(gen_support_1, gen_support_2, z_support)
        features = torch.cat((gen_feature, z_support), 1)
        weight = torch.mean(features, 1)
        """
        weight = torch.zeros((self.n_way, self.dim), requires_grad=True).cuda()
        for i in range(self.n_way):
            weight_point = torch.zeros((self.n_support)*(self.gen_num+1), self.dim)
            for j in range(self.n_support):
                gen_feature, _ = self.model_G(gen_support_1, gen_support_2, z_support[i,j])
                features = torch.cat((gen_feature, z_support[i,j].unsqueeze(0)), 0)
                weight_point[j*(self.gen_num+1):(j+1)*(self.gen_num+1)] = features
                #print(gen_feature.shape)
            weight[i] = torch.mean(weight_point, 0)
        """
        
        #print(weight)
        weight = self.model_G.l2_norm(weight)
        predict = torch.matmul(z_query, torch.transpose(weight,0,1))*self.model_G.s
        predict = predict.view(-1, predict.shape[2])

        """
        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )


        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        """
        scores = predict

        return scores


    def set_forward_loss(self, x, generated_support_1, generated_support_2):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x, generated_support_1, generated_support_2)
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
