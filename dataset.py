# -*- coding: utf-8 -*-
import torch as tr
from torch.utils.data import Dataset
import numpy as np
#maybe will fit to anything 
class FeatureDataSet(Dataset):
  def __init__(self, samples, labels, meta_labels, mix=False):
    if mix:
        indices = list(range(len(samples)))
        np.random.shuffle(indices)
        samples= np.array([samples[i] for i in indices])
        labels = np.array([labels[i] for i in indices])
        meta_labels = np.array([meta_labels[i] for i in indices])
    device = 'cuda' if tr.cuda.is_available() else 'cpu'
    self.X = tr.tensor(samples).type(tr.float)
    #self.X = self.X.to(device)
    self.Y = tr.tensor(labels).type(tr.float)
    #self.Y = self.Y.to(device)
    self.Y = tr.unsqueeze(self.Y, 1)
    self.Z = tr.tensor(meta_labels).type(tr.uint8)
    #self.Z = self.Z.to(device)
  
  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return (self.X[i], self.Y[i], self.Z[i])

  def mix(self):
      indices = list(range(len(self.X)))
      np.random.shuffle(indices)
      print(self.X[0])
      self.X = tr.tensor([self.X[i] for i in indices])
      self.Y = tr.tensor([self.Y[i] for i in indices])
      self.Z = tr.tensor([self.Z[i] for i in indices])