# -*- coding: utf-8 -*-
import torch as tr
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
  def __init__(self, input_size):
    super().__init__()

    self.input_size = input_size
    self.matrix = nn.Linear(input_size, input_size)
    self.vector = nn.Linear(input_size, 1)

    self.device = 'cuda' if tr.cuda.is_available() else 'cpu'

  def forward(self, x):
      result = self.matrix(x)
      result = self.vector(result)
      result = F.softmax(result, dim=1)
      x = x * result
      x = tr.sum(x, dim=1)
      return x
