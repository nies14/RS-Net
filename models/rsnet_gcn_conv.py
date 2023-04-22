from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RsNetGraphConv(nn.Module):
    """
    High-order graph convolution layer 
    """

    def __init__(self, in_features, out_features, adj, bias=True,):
        super(RsNetGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(8, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.W2 = nn.Parameter(torch.zeros(size=(4, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.M = nn.Parameter(torch.zeros(size=(4,adj.size(0), out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)
  
        self.adj_0 = torch.eye(adj.size(0), dtype=torch.float) # declare self-connections
        self.adj2_0 = nn.Parameter(torch.ones_like(self.adj_0))        
        nn.init.constant_(self.adj2_0, 1e-6)

        self.adj_1 = adj # one_hop neighbors
        self.adj2_1 = nn.Parameter(torch.ones_like(self.adj_1))        
        nn.init.constant_(self.adj2_1, 1e-6)
        
        self.adj_2 = torch.matmul(self.adj_1, adj) # two_hop neighbors
        self.adj2_2 = nn.Parameter(torch.ones_like(self.adj_2))        
        nn.init.constant_(self.adj2_2, 1e-6)

        self.adj_3 = torch.matmul(self.adj_2, adj) # three_hop neighbors
        self.adj2_3 = nn.Parameter(torch.ones_like(self.adj_3))        
        nn.init.constant_(self.adj2_3, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features*4, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
            
            self.bias_2 = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias_2.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input1, input2):

        I = torch.eye(self.adj_1.size(0), dtype=torch.float).to(input1.device)

        adj = self.adj_0.to(input1.device) + self.adj2_0.to(input1.device)
        A_0 = (adj.T + adj)/2
        h0 = torch.matmul(input1,self.W[0])
        h1 = torch.matmul(input1,self.W[1])
        x1 = torch.matmul(input2,self.W2[0])
        output_0 = torch.matmul(A_0*I,self.M[0]*h0) + torch.matmul(A_0*(1-I),self.M[0]*h1) + x1

        adj = self.adj_1.to(input1.device) + self.adj2_1.to(input1.device)
        A_1 = (adj.T + adj)/2
        h2 = torch.matmul(input1,self.W[2])
        h3 = torch.matmul(input1,self.W[3])
        x2 = torch.matmul(input2,self.W2[1])
        output_1 = torch.matmul(A_1*I,self.M[1]*h2) + torch.matmul(A_1*(1-I),self.M[1]*h3) + x2

        adj = self.adj_2.to(input1.device) + self.adj2_2.to(input1.device)
        A_2 = (adj.T + adj)/2
        h4 = torch.matmul(input1,self.W[4])
        h5 = torch.matmul(input1,self.W[5])
        x3 = torch.matmul(input2,self.W2[2])
        output_2 = torch.matmul(A_2*I,self.M[2]*h4) + torch.matmul(A_2*(1-I),self.M[2]*h5) + x3

        adj = self.adj_3.to(input1.device) + self.adj2_3.to(input1.device)
        A_3 = (adj.T + adj)/2
        h6 = torch.matmul(input1,self.W[6])
        h7 = torch.matmul(input1,self.W[7])
        x4 = torch.matmul(input2,self.W2[3])
        output_3 = torch.matmul(A_3*I,self.M[3]*h6) + torch.matmul(A_3*(1-I),self.M[3]*h7) + x4
        
        if self.out_features != 3:  
            output = torch.cat([output_0, output_1, output_2, output_3], dim = 2)####################################################cat
        else: 
            output = output_0 + output_1 + output_2 + output_3
            return output + self.bias_2.view(1,1,-1)
        if self.bias is not None:
            return output + self.bias.view(1,1,-1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
