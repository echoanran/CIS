import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SubjectAttentionLayer(nn.Module):
    def __init__(self, input_channel, d_model, output_channel):
        super(SubjectAttentionLayer, self).__init__()
        
        self.input_channel = input_channel
        self.d_model = d_model
        self.output_channel = output_channel

        # subject attention
        self.kmap = nn.Linear(self.input_channel, self.d_model)
        self.qmap = nn.Linear(self.input_channel, self.d_model)
        self.xmap = nn.Linear(self.input_channel, self.output_channel)
        self.smap = nn.Linear(self.input_channel, self.output_channel)
   
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, x, subject_infos):

        K_subject, V_subject = subject_infos
        
        K_subject = self.kmap(K_subject)
        Q_subject = self.qmap(x)

        subject_embedding, _ = self.attention(Q_subject, K_subject, V_subject)
        x = self.xmap(x) + self.smap(subject_embedding)

        return x
