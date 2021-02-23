# -*- coding: utf-8 -*-
import torch.nn as nn
from models.rnn import myRNN
from models.sa import Self_attention

class Combination(nn.Module):
  def __init__(self, num_of_features, num_of_heads=1,layers=1,  dropOut=.1, feedForward=1, usePosition = False, bi=False, configuration="N", order="rnn", rnnType="lstm"):
    super().__init__()
    
    
    self.isBi = 1 if bi==False else 2
    
    if order == "rnn":
        self.combime = nn.Sequential(
            myRNN(rnnType, num_of_features, bi, configuration, layers=layers),
            Self_attention(num_of_features, num_of_heads=num_of_heads, dropOut=dropOut, layers=layers, usePosition=usePosition)
        )
    elif order == "sa":
        self.combime = nn.Sequential(
            Self_attention(num_of_features, num_of_heads=num_of_heads, dropOut=dropOut, layers=layers, usePosition=usePosition),
            myRNN(rnnType, num_of_features, bi, configuration, layers=layers)
        )
  def forward(self, X):
    X = self.combime(X)
    return X