# -*- coding: utf-8 -*-
import torch.nn as nn
import torch as tr
from models.positional_encoding import PositionalEncoding

class Self_attention(nn.Module):
  def __init__(self, num_of_features, num_of_heads=1,layers=1,  dropOut=.1, feedForward=1, usePosition = False):
    super().__init__()
    tr.manual_seed(num_of_features)
    tr.cuda.manual_seed(num_of_features)

    self.usePosition = usePosition
    self.positional_encoding = PositionalEncoding(num_of_features, dropout=dropOut)
    models = []
    for i in range(layers):
        models.append(nn.TransformerEncoderLayer(d_model=num_of_features, nhead=num_of_heads, dim_feedforward=feedForward, dropout=dropOut))
    
    #self.transformer = nn.TransformerEncoderLayer(d_model=num_of_features, nhead=num_of_heads, dim_feedforward=feedForward, dropout=dropOut)
    
    self.transformer = nn.Sequential(*models)

    
    self.isBi = 1
  def forward(self, X):
    if self.usePosition:
        X = self.positional_encoding(X)
    X = self.transformer(X)
    return X

