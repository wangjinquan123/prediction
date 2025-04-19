'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_dim, dims, out_dim, activation=F.relu):
        super(MLP, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(in_dim, dims[0]),
            nn.ReLU(inplace=True)
        )
        if len(dims) > 0:
            self.layers = nn.ModuleList([
                    nn.Sequential(
                    nn.Linear(dims[i], dims[i]),
                    nn.ReLU(inplace=True)
                    ) for i in range(len(dims))
                ])
        else:
            self.layers = None
        self.out_layer = nn.Linear(dims[-1], out_dim)
        self.activation = activation

    def forward(self, x):
        x = self.in_layer(x)
        if self.layers is not None:
            for layer in self.layers:
                x = layer(x)
        x = self.out_layer(x)
        return x


class TopoFuser(nn.Module):
    def __init__(self, device,input_dim, dim, drop=0.1):
        super(TopoFuser, self).__init__()
        self.src_mlp = MLP(input_dim, [dim], dim// 2)
        self.tgt_mlp = MLP(input_dim, [dim], dim // 2)
    
    def forward(self, src_feat, tgt_feat, prev_occ_feat=None):
        """
        src_feat, tgt_feat :[b, len_src, d], [b, len_tgt, d]
        prev_occ_feat: [b, len_src, len_tgt, d]
        return occ_feat[b, len_src, len_tgt, d]
        """

        src_feat = self.src_mlp(src_feat)
        tgt_feat = self.tgt_mlp(tgt_feat)
        # broadcast the source and target feature:
        len_src, len_tgt = src_feat.shape[1], tgt_feat.shape[1]
        # [b, len_src, len_tgt, d//2]
        
        src = src_feat.unsqueeze(2).repeat(1, 1, len_tgt, 1)
        tgt = tgt_feat.unsqueeze(1).repeat(1, len_src, 1, 1)
        agt_inter_feat = torch.cat([src, tgt], dim=-1)

        if prev_occ_feat is not None:
            agt_inter_feat = agt_inter_feat + prev_occ_feat

        return agt_inter_feat

class TopoDecoder(nn.Module):
    def __init__(self, device, dim, drop=0.1, multi_step=1):
        super(TopoDecoder, self).__init__()
        self.decoder = MLP(dim, [dim], multi_step)
    
    def forward(self, occ_feat):
        # [b, a+m, a+m, 1]
        out = self.decoder(occ_feat)
        # return out[..., 0]
        return out





