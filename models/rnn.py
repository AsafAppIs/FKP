# -*- coding: utf-8 -*-
import torch.nn as nn
from models.none import None_


class myRNN(nn.Module):
  def __init__(self, rnnType, num_of_features, bi=False, configuration="N", layers=1):
    super().__init__()
    if rnnType == "lstm":
        self.rnn = nn.LSTM(num_of_features, num_of_features, bidirectional=bi, num_layers=layers)
    elif rnnType == "gru":
        self.rnn = nn.GRU(num_of_features, num_of_features, bidirectional=bi, num_layers=layers)
    self.isBi = 1 if bi==False else 2
    
    if configuration == "N":
        self.filter = None_()
    elif configuration == "D":
        self.filter = nn.Dropout(.5)
    elif configuration == "B":
        self.filter = nn.BatchNorm1d(50)

  def forward(self, X):
    X = X.permute(1,0,2)
    X, _ = self.rnn(X)
    

    X = X.permute(1,0,2)
    X = self.filter(X)

    return X
