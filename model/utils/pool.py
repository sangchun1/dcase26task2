import os
import math
import torch
import random
import numpy as np
import pandas as pd

import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

from scipy.stats import hmean
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import loralib as lora

def seed_everything(seed: int): #for deterministic result; currently wav2vec2 model and torch.use_deterministic_algorithms is incompatible
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
        
class Classic_Attention(nn.Module):
    def __init__(self, input_dim, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim,embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))
         
    def forward(self,inputs):
        lin_out = self.lin_proj(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = F.tanh(lin_out.bmm(v_view).squeeze(-1))
        attention_weights_normalized = F.softmax(attention_weights,1)
        return attention_weights_normalized

class attentive_statistics_pooling(nn.Module):
    def __init__(self, input_dim, embed_dim, ds_rate, output_dim, dropout, use_lora=False):
        super().__init__()
        self.attention = Classic_Attention(input_dim, embed_dim)

        self.k = ds_rate
        self.output_dim = output_dim
        self.dropout = dropout

        # MLP Layers with optional Batch Normalization and Activation
        layers = []
        in_dim = int(embed_dim * 2)
        mid_dim = int(in_dim * self.k)
        layers.append(nn.BatchNorm1d(in_dim))

        # First Block
        layers.append(lora.Linear(in_dim, mid_dim, r=64) if use_lora else nn.Linear(in_dim, mid_dim))
        layers.append(nn.BatchNorm1d(mid_dim))
        layers.append(nn.ReLU())
        self.block1 = nn.Sequential(*layers)

        # Clear layers list for next block
        layers = []

        # Second Block
        layers.append(lora.Linear(mid_dim, mid_dim, r=64) if use_lora else nn.Linear(mid_dim, mid_dim))
        layers.append(nn.BatchNorm1d(mid_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))
        self.block2 = nn.Sequential(*layers)

        # Projection Layer
        self.proj = nn.Linear(mid_dim, self.output_dim)
        
    def weighted_sd(self, inputs, attention_weights, mean):
        el_mat_prod = torch.mul(inputs, attention_weights.unsqueeze(2).expand(-1, -1, inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs, el_mat_prod)
        variance = torch.sum(hadmard_prod, 1) - torch.mul(mean, mean)
        return variance
        
    def stat_attn_pool(self,inputs,attention_weights):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1, -1, inputs.shape[-1]))
        mean = torch.mean(el_mat_prod, 1)
        variance = self.weighted_sd(inputs,attention_weights,mean)
        std = torch.sqrt(torch.clamp(variance, min=0) + 1e-12)
        stat_pooling = torch.cat((mean,std), 1)
        return stat_pooling
        
    def forward(self, inputs):
        attn_weights = self.attention(inputs) 
        stat_pool_out = self.stat_attn_pool(inputs, attn_weights)

        # First Block
        output = self.block1(stat_pool_out)

        # Second Block
        output = self.block2(output)

        # Projection Layer
        output = self.proj(output)
        
        return output