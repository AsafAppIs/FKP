# -*- coding: utf-8 -*-
import torch as tr
import torch.nn as nn


class Squeeze(nn.Module):
    def forward(self, X):
        X = tr.squeeze(X, len(X.shape) - 1)
        return X
