# -*- coding: utf-8 -*-
import torch.nn as nn


class None_(nn.Module):
  def __init__(self):
    super().__init__()


  def forward(self, x):
      return x
