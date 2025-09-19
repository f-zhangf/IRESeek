import os
import torch
import torchtext
import numpy as np
from torch import nn,Tensor
from torch.utils import data
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder,TransformerEncoderLayer
import sys
import torch.nn as nn
import math
import time
import dgl
import scipy.sparse as sp
import dgl.function as fn
from dgl.nn.pytorch import GraphConv,MaxPooling
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=None, hidden_dim=None, dropout=0.1):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or 4*in_dim
        self.layers = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
          )
        
    def forward(self, x):
        return self.layers(x)

# Sends a message of node feature h
# Equivalent to => return {'m': edges.src['h']}
# msg = fn.copy_src(src='h', out='m')
#
# def reduce(nodes):
#     accum = torch.mean(nodes.mailbox['m'], 1)
#     return {'h': accum}

# def msg_func(edges):
#     return {'m': torch.mul(edges.data['feat'], edges.src['h'])}
msg_func = fn.u_mul_e('h', 'feat', 'm')
reduce_mean = fn.mean('m', 'h')
reduce_sum = fn.sum('m', 'h')
reduce_max = fn.max('m', 'h')
###########提取结构特征模块###################
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x) 
class ResConv2dSimple(nn.Module):
    def __init__(self, 
                 in_c, 
                 out_c,
                 kernel_size=7,
                 use_se = False,
                ):  
        super().__init__()
        if use_se:
            self.conv = nn.Sequential(
                # b c w h
                nn.Conv2d(in_c,
                          out_c, 
                          kernel_size=kernel_size, 
                          padding="same", 
                          bias=False),
                # b w h c#
                nn.BatchNorm2d(out_c),
                SE_Block(out_c),
                nn.GELU(),
                # b c e 
            )
            
        else:
            self.conv = nn.Sequential(
                # b c w h
                nn.Conv2d(in_c,
                          out_c, 
                          kernel_size=kernel_size, 
                          padding="same", 
                          bias=False),
                # b w h c#
                nn.BatchNorm2d(out_c),
                nn.GELU(),
                # b c e 
            )
        
        if in_c == out_c:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_c,
                          out_c, 
                          kernel_size=1, 
                          bias=False)
            )

    def forward(self, x):
        # b e s 
        h = self.conv(x)
        x = self.res(x) + h
        return x
class ConvReadoutLayer(nn.Module):
    """Conv Readout layer."""
    def __init__(self, mode="mean", **kwargs):
        super().__init__()
        self.mode = mode

    def forward(self, g, feat):
        start, output = 0, 0
        first_flag = 0

        for batch_num in g.batch_num_nodes():
            if first_flag == 0:
                output = torch.transpose(feat[start:start + batch_num], 1, 0).unsqueeze(0)
                first_flag = 1
            else:
              output = torch.cat([output, torch.transpose(feat[start:start + batch_num], 1, 0).unsqueeze(0)], dim=0)
            start += batch_num
        return output
class MAXPoolLayer(nn.Module):
    """MAXPool layer."""
    def __init__(self, kernel_size, stride = 1, padding=0, **kwargs):
        super().__init__()

        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding

        self.pooling = nn.MaxPool2d(kernel_size, stride, padding=padding)

    def forward(self, inputs):
        output = self.pooling(inputs)
        return output
class NodeApplyModule(nn.Module):
    # Update node feature h_v with (Wh_v+b)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}
class ConvLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, in_channel, out_channel, kernel_size, activation, batch_norm, padding="same", residual=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.residual = residual
        self.padding = padding
        
        if in_channel != out_channel:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm2d(out_channel)
        self.activation = activation
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)

    def forward(self, feature):
        h_in = feature  # to be used for residual connection

        h = self.conv(feature)

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                                                         self.in_channel,
                                                                         self.out_channel, self.residual)
class GCNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, residual=False, dgl_builtin=True,zero = False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual
        self.dgl_builtin = dgl_builtin

        if in_dim != out_dim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if self.dgl_builtin == False:
            self.apply_mod = NodeApplyModule(in_dim, out_dim)
        else:
            self.conv = GraphConv(in_dim, out_dim,allow_zero_in_degree=zero)

    def forward(self, g, feature):
        h_in = feature  # to be used for residual connection

        if self.dgl_builtin == False:
            g.ndata['h'] = feature
            # g.update_all(msg, reduce)
            g.update_all(message_func=msg_func, reduce_func=reduce_mean)
            g.apply_nodes(func=self.apply_mod)
            h = g.ndata['h']  # result of graph convolution
        else:
            h = self.conv(g, feature)

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        h = self.dropout(h)
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                                                         self.in_channels,
                                                                         self.out_channels, self.residual)
class GCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        self.device = net_params['device']
        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.type = net_params['type']
        self.sequence = None
        self.sequence_att = None
        self.pool = True
        seq_size = 174
        if self.type == "GCN":
            #conv_kernel = 3 #使用池化层在使用
            if self.pool:
                conv_kernel = 3
                pooling_kernel = [5, 4]
                pooling_padding,pooling_stride = [2,0],1
                #GCN 使用maxpool
                gcn_kernel = 3
                gcn_pooling_padding =1
                gcn_pooling_stride =1
            else:
                conv_kernel = [5, 4]
                conv_padding, conv_stride = [2, 0], 1

            self.embedding_h = nn.Linear(in_dim, hidden_dim)
            self.in_feat_dropout = nn.Dropout(in_feat_dropout)
            self.layers_gcn = nn.ModuleList()
            
            self.layers_gcn.append(GCNLayer(hidden_dim, hidden_dim, F.relu, dropout, self.batch_norm, self.residual))
            for _ in range(self.n_layers * 2 - 2):
                self.layers_gcn.append(GCNLayer(hidden_dim, hidden_dim, F.relu, dropout, self.batch_norm, self.residual))
            self.layers_gcn.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual))
            '''
            for _ in range(self.n_layers):
                self.layers_gcn.append(GCNLayer(hidden_dim, hidden_dim, F.relu, dropout, self.batch_norm, self.residual))
            '''
            if self.pool:
                self.layers_gcn_pooling = nn.MaxPool1d(gcn_kernel, gcn_pooling_stride, padding=gcn_pooling_padding)
        
            #CNN提取权重模块
            self.conv_readout_layer = ConvReadoutLayer(self.readout)
            if self.pool:
                self.layers_cnn =ConvLayer(1, 32, conv_kernel, F.leaky_relu, self.batch_norm, residual=False)
                self.layers_cnn_pool =MAXPoolLayer(pooling_kernel,pooling_stride,pooling_padding)
            else:
                self.layers_cnn = ConvLayer(1, 32, conv_kernel, F.relu, self.batch_norm,padding =conv_padding ,residual=False)
                
            self.batchnorm_weight = nn.BatchNorm1d(seq_size) #序列长度

    def forward(self, g, h):
        if self.type == "GCN":
            h1 = self.embedding_h(h)
            h1 = self.in_feat_dropout(h1)
            
            for i in range(self.n_layers):
                h1 = self.layers_gcn[2*i](g, h1)
                h1 = self.layers_gcn[2*i + 1](g, h1) #h1 [bxl,d]
            '''
            for i in range(self.n_layers):
                h1 = self.layers_gcn[i](g, h1)
            '''
            #cnn权重序列
            h2 = self._graph2feature(g)#[b,1,174,4]
            self.sequence = h2
            h2 = h2.to(self.device)
            h2 = self.layers_cnn(h2)#[b,32,174,1]
            if self.pool:
                h2 = self.layers_cnn_pool(h2)
            cnn_node_weight = torch.mean(h2, dim=1).squeeze(-1)#[b,174]
            cnn_node_weight = torch.sigmoid(self.batchnorm_weight(cnn_node_weight))#[b,174]
            #权重指导二级结构
            hg = self.conv_readout_layer(g, h1)#[b,32,174]
            if self.pool:
                hg = hg.permute(0,2,1)
                hg = self.layers_gcn_pooling(hg)
                hg = hg.permute(0,2,1).unsqueeze(-1)
            else:
                hg = hg.unsqueeze(-1)#[b,32,174,1]
            #hg = torch.mul(hg, cnn_node_weight.unsqueeze(1).unsqueeze(-1))#[b,32,174,1]x[b,1,174,1]
            hg = hg.permute(0,2,1,3).squeeze(-1) #[b,174,32]
            return hg
    def _graph2feature(self, g):
        torch.set_printoptions(profile="full")
        feat = g.ndata['feat']
        start, first_flag = 0, 0
        for batch_num in g.batch_num_nodes():
            #print("batch_num_nodes: ",g.batch_num_nodes())
            #print("batch_num: ",batch_num)
            #sys.exit()  
            if first_flag == 0:
                output = torch.transpose(feat[start:start + batch_num], 1, 0).unsqueeze(0)
                #print("first output: ",output)
                #sys.exit()
                first_flag = 1
            else:
                output = torch.cat([output, torch.transpose(feat[start:start + batch_num], 1, 0).unsqueeze(0)], dim=0)
                #print("sec output: ",output)
                #sys.exit()
            start += batch_num
        output = torch.transpose(output, 1, 2)
        output = output.unsqueeze(1)
        #print("third output: ",output.shape)
        return output

class GCNNetSS(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        self.device = net_params['device']
        self.n_layers = 1
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.type = net_params['type']
        self.sequence = None
        self.sequence_att = None
        self.pool = True
        seq_size = 174
        if self.type == "GCN":
            #conv_kernel = 3 #使用池化层在使用
            if self.pool:
                conv_kernel = 3
                pooling_kernel = [5, 4]
                pooling_padding,pooling_stride = [2,0],1
                #GCN 使用maxpool
                gcn_kernel = 3
                gcn_pooling_padding =1
                gcn_pooling_stride =1
            else:
                conv_kernel = [5, 4]
                conv_padding, conv_stride = [2, 0], 1

            self.embedding_h = nn.Linear(in_dim, hidden_dim)
            self.in_feat_dropout = nn.Dropout(in_feat_dropout)
            self.layers_gcn = nn.ModuleList()
            self.layers_gcn.append(GCNLayer(hidden_dim, hidden_dim, F.relu, dropout, self.batch_norm, self.residual,zero =True))
            for _ in range(self.n_layers * 2 - 2):
                self.layers_gcn.append(GCNLayer(hidden_dim, hidden_dim, F.relu, dropout, self.batch_norm, self.residual,zero=True))
            self.layers_gcn.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual,zero = True))
            
            if self.pool:
                self.layers_gcn_pooling = nn.MaxPool1d(gcn_kernel, gcn_pooling_stride, padding=gcn_pooling_padding)
                
            self.conv_readout_layer = ConvReadoutLayer(self.readout)

    def forward(self, g, h):
        if self.type == "GCN":
            h1 = self.embedding_h(h)
            h1 = self.in_feat_dropout(h1)
            for i in range(self.n_layers):
                h1 = self.layers_gcn[2*i](g, h1)
                h1 = self.layers_gcn[2*i + 1](g, h1) #h1 [bxl,d]
            hg = self.conv_readout_layer(g, h1)#[b,32,174]
            if self.pool:
                hg = hg.permute(0,2,1)
                hg = self.layers_gcn_pooling(hg)
            else:
                hg = hg.permute(0,2,1)
            return hg           





class RNA_CNN(nn.Module):
    def __init__(self, 
                 input_dim=4,
                 feature_dim=32,
                 seq_len=174,
                 head = 8,
                 T_num_layers =4):
        super(RNA_CNN, self).__init__()
        #print(f"nhead is {head} T_num_layers {T_num_layers} RNN")
        # 输入形状: (b, 174, 4)
        
        # 第一层卷积 (保持长度不变)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, 
                              kernel_size=5, padding=2, padding_mode='replicate')
        self.bn1 = nn.BatchNorm1d(64)
        
        # 第二层卷积 (保持长度不变)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn2 = nn.BatchNorm1d(64)
        
        # 第三层卷积 (保持长度不变)
        self.conv3 = nn.Conv1d(64, feature_dim, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn3 = nn.BatchNorm1d(feature_dim)

        encoder_layers = TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=head,
            batch_first=True
        )
        self.transformer = TransformerEncoder(
            encoder_layer= encoder_layers,
            num_layers =T_num_layers
        )
        
        # 输出形状: (b, 174, 32)
        
    def forward(self, x):
        # 输入x形状: (b, 174, 4)
        x = x.permute(0, 2, 1)  # 转为(b, 4, 174)以适应Conv1d
        
        # 第一层
        x = F.relu(self.bn1(self.conv1(x)))  # (b, 64, 174)
        
        # 第二层
        x = F.relu(self.bn2(self.conv2(x)))  # (b, 64, 174)
        
        # 第三层
        x = F.relu(self.bn3(self.conv3(x)))  # (b, 32, 174)
        
        # 转回(b, 174, 32)的输出形状
        x = x.permute(0, 2, 1)
        x =self.transformer(x)
        return x

class IreeSeek_cov_model(nn.Module):
    def  __init__(self,
                  net_params):
        super().__init__()
        seq_size = 174
        input_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        self.fea_info = net_params['fea_info']
        class_out_dim = 2 if net_params["use_softmax"] else 1
        self.bitransformer = RNA_CNN(input_dim,hidden_dim,seq_size)
        self.gcn_net_ss = GCNNetSS(net_params)
        self.gcn_net_bpe =GCNNet(net_params)
        self.gcn_net_bpp = GCNNet(net_params)

        if self.fea_info == 1: #seq
            feat_in_dim = seq_size*out_dim * 1
        elif self.fea_info == 2 or self.fea_info == 6: #seq bpe #seq bpp
            feat_in_dim = seq_size*out_dim * 2
        elif self.fea_info == 3 or self.fea_info== 4: #3:seq bpe ss;4:seq bpe bpp
            feat_in_dim = seq_size*out_dim * 3
        elif self.fea_info == 5: #seq ss bpe bpp
            feat_in_dim = seq_size*out_dim * 4
                         
        self.MLP_layer = MLP(in_dim=feat_in_dim,out_dim=class_out_dim, hidden_dim=128)
        # self.MLP_layer = MLP(in_dim=feat_in_dim, out_dim=1, hidden_dim=None) # TODO
    def forward(self,h,bpe_g =None,ss_g= None,bpp_g=None):
        assert bpe_g is not None , "Error: bpe_g cannot be None or empty!"
        seq = self._graph2seq(bpe_g) #[b,174,4]
        #print(seq[:10])
        seq_sf = self.bitransformer(seq)
        bpe_gf = self.gcn_net_bpe(bpe_g,h)
        if ss_g != None:
            ss_gf = self.gcn_net_ss(ss_g,h)
        if bpp_g != None:
            bpp_gf = self.gcn_net_bpp(bpp_g,h)
    
        if self.fea_info == 1: #seq
            seq_sf = torch.flatten(seq_sf, start_dim=1)
            final = seq_sf
        elif self.fea_info == 2:#seq bpe
            seq_sf = torch.flatten(seq_sf, start_dim=1)
            bpe_gf = torch.flatten(bpe_gf, start_dim=1)
            final = torch.cat([seq_sf,bpe_gf], dim=1)
        elif self.fea_info == 3:#3:seq bpe ss;
            seq_sf = torch.flatten(seq_sf, start_dim=1)
            bpe_gf = torch.flatten(bpe_gf, start_dim=1)
            ss_gf = torch.flatten(ss_gf, start_dim=1)
            final = torch.cat([seq_sf,bpe_gf,ss_gf], dim=1)
        elif self.fea_info == 4:#seq,bpe,bpp
             seq_sf = torch.flatten(seq_sf, start_dim=1)
             bpe_gf = torch.flatten(bpe_gf, start_dim=1)
             bpp_gf = torch.flatten(bpp_gf, start_dim=1)
             final = torch.cat([seq_sf,bpe_gf,bpp_gf], dim=1)
             #print('final shape:',final.shape)
             #return final
        elif self.fea_info == 5: #seq,ss,bpe,bpp
            seq_sf = torch.flatten(seq_sf, start_dim=1)
            bpe_gf = torch.flatten(bpe_gf, start_dim=1)
            bpp_gf = torch.flatten(bpp_gf, start_dim=1)
            ss_gf = torch.flatten(ss_gf, start_dim=1)
            final = torch.cat([seq_sf,bpe_gf,bpp_gf,ss_gf], dim=1)
        elif self.fea_info == 6: #seq,bpp
             seq_sf = torch.flatten(seq_sf, start_dim=1)
             bpp_gf = torch.flatten(bpp_gf, start_dim=1)
             final = torch.cat([seq_sf,bpp_gf], dim=1)

        pred = self.MLP_layer(final)
        #print(final.shape,final[:10,:10])
        #print(pred[:10])
        return pred        
        
    def _graph2seq(self, g):
        torch.set_printoptions(profile="full")
        feat = g.ndata['feat'] # num x feat
        start, first_flag = 0, 0
        for batch_num in g.batch_num_nodes(): 
            if first_flag == 0:
                output = torch.transpose(feat[start:start + batch_num], 1, 0).unsqueeze(0)
                first_flag = 1
            else:
                output = torch.cat([output, torch.transpose(feat[start:start + batch_num], 1, 0).unsqueeze(0)], dim=0)
            start += batch_num
        output = torch.transpose(output, 1, 2)
        return output

