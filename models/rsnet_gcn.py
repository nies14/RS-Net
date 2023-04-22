from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.functional as F
from models.rsnet_gcn_conv import RsNetGraphConv
from models.graph_non_local import GraphNonLocal
from nets.non_local_embedded_gaussian import NONLocalBlock2D
from functools import reduce


class _GraphConvI(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConvI, self).__init__()

        self.gconv = RsNetGraphConv(input_dim, output_dim, adj)
        self.ln = nn.LayerNorm(output_dim*4)
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x, x0):

        x = self.gconv(x, x0)
        x = self.ln(x)
        
        if self.dropout is not None:
            x = self.dropout(x)

        return x

class _GraphConvII(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConvII, self).__init__()

        self.gconv = RsNetGraphConv(input_dim, output_dim, adj)
        self.act = nn.GELU()
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x, x0):

        x = self.gconv(x, x0)
        if self.dropout is not None:
            x = self.dropout(self.act(x))
        
        x = self.act(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConvI(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConvII(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv3 = _GraphConvI(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv4 = _GraphConvII(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv5 = _GraphConvI(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv6 = _GraphConvII(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv7 = _GraphConvI(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv8 = _GraphConvII(adj, hid_dim*4, output_dim, p_dropout)

    def forward(self, x,x_0):
        initial = x_0
        residual = x
        out = self.gconv1(x,initial)
        out = self.gconv2(out,initial)
        out = out+residual
        residual = out
        out = self.gconv3(out,initial)
        out = self.gconv4(out,initial)
        out = out+residual
        residual = out
        out = self.gconv5(out,initial)
        out = self.gconv6(out,initial)
        out = out+residual
        residual = out
        out = self.gconv7(out,initial)
        out = self.gconv8(out,initial)
        out = out+residual

        return  out


class Model(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4,nodes_group=None, p_dropout=None):
        super(Model, self).__init__()

        self.gconv_input = _GraphConvII(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)
        self.gconv_layers = _ResGraphConv(adj, hid_dim*4, hid_dim, hid_dim, p_dropout=p_dropout)
        self.gconv_output = RsNetGraphConv(hid_dim*4, coords_dim[1], adj)
        self.non_local = NONLocalBlock2D(in_channels=hid_dim*4, sub_sample=False)
    def forward(self, x): 

        x = x.squeeze() 
        x = x.permute(0,2,1)
        out = self.gconv_input(x,x)
        x_0=out
        out = self.gconv_layers(out,x_0)
        out = out.unsqueeze(2)
        out = out.permute(0,3,2,1)
        out = self.non_local(out)
        out = out.permute(0,3,1,2)
        out = out.squeeze()
        out = self.gconv_output(out,x_0)
        out = out.permute(0,2,1)
        out = out.unsqueeze(2)
        out = out.unsqueeze(4)
        return out