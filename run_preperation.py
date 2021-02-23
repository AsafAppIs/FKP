# -*- coding: utf-8 -*-
import torch as tr
import numpy as np
from dataset import FeatureDataSet
from  import_data import load_data, import_model_data
import torch.nn as nn
from models.sa import Self_attention
from models.rnn import myRNN
from models.KiNET import KiNET
from models.combination import Combination
import configurations




def down_sampling(X, Y, Z):
  num_of_zeros = sum([1 if Y[i] == 0 else 0 for i in range(len(Y))])
  num_of_ones = len(Y) - num_of_zeros

  min_label = min(num_of_ones, num_of_zeros)

  indices = []
  num_of_zeros = 0
  num_of_ones = 0

  for i in range(len(Y)):
    if Y[i] == 1 and num_of_ones < min_label:
      num_of_ones += 1
      indices.append(i)
    elif Y[i] == 0 and num_of_zeros < min_label:
      num_of_zeros += 1
      indices.append(i)

  filter_indices = [i for i in range(len(Y)) if (i not in indices)]

  X = np.delete(X, filter_indices, axis=0)
  Y = np.delete(Y, filter_indices, axis=0)
  Z = np.delete(Z, filter_indices, axis=0)

  return X, Y, Z

#under the assemption that there is 3 times more one labels
def over_sampling(X, Y, Z):
    zero_indices = [i for i in range(len(Y)) if Y[i] == 0]
    zero_X = [X[i] for i in zero_indices]
    zero_Y = [Y[i] for i in zero_indices]
    zero_Z = [Z[i] for i in zero_indices]
    
    for i in range(2):
        X = np.concatenate((X, zero_X), axis=0)
        Y = np.concatenate((Y, zero_Y), axis=0)
        Z = np.concatenate((Z, zero_Z), axis=0)
    
    #mixing the data
    indices = list(range(len(Z)))
    np.random.shuffle(indices)
    X = np.array([X[i] for i in indices])
    Y = np.array([Y[i] for i in indices])
    Z = np.array([Z[i] for i in indices])
    
    return X, Y, Z


def concat_array(data_array):
  for i in range(1, len(data_array)):
      data_array[0] = np.concatenate((data_array[0], data_array[i]), axis=0)
  data_array = data_array[0]
  return data_array

def pseudo_random(random_seed):
  tr.manual_seed(random_seed)
  tr.cuda.manual_seed(random_seed)
  np.random.seed(random_seed)


def trial_data_split(X, Y, Z, proporions=[0.7, 0.9]):
  indices = list(range(len(X)))
  split_train = int(np.floor(proporions[0] * len(X)))
  split_val = int(np.floor(proporions[1] * len(X)))
  np.random.shuffle(indices)
  train_set_x = np.array([X[i] for i in indices[:split_train]])
  train_set_y = np.array([Y[i] for i in indices[:split_train]])
  train_set_z = np.array([Z[i] for i in indices[:split_train]])

  val_set_x = np.array([X[i] for i in indices[split_train:split_val]])
  val_set_y = np.array([Y[i] for i in indices[split_train:split_val]])
  val_set_z = np.array([Z[i] for i in indices[split_train:split_val]])

  test_set_x = np.array([X[i] for i in indices[split_val:]])
  test_set_y = np.array([Y[i] for i in indices[split_val:]])
  test_set_z = np.array([Z[i] for i in indices[split_val:]])
  print(len(train_set_x))
  print(len(val_set_y))
  print(len(test_set_x))
  train_dataset = FeatureDataSet(train_set_x, train_set_y, train_set_z)
  validation_dataset = FeatureDataSet(val_set_x, val_set_y, val_set_z)
  test_dataset = FeatureDataSet(test_set_x, test_set_y, test_set_z)
  return train_dataset, validation_dataset, test_dataset

# this function is kind of Unnecessary, mosly here for readability...
def subjects_data_split(X, Y, Z, proporions=[0.7, 0.9]):
  indices = list(range(len(X)))
  split_train = int(np.floor(proporions[0] * len(X)))
  split_val = int(np.floor(proporions[1] * len(X)))
  np.random.shuffle(indices)

  train_set_x = concat_array(np.array([X[i] for i in indices[:split_train]]))
  train_set_y = concat_array(np.array([Y[i] for i in indices[:split_train]]))
  train_set_z = concat_array(np.array([Z[i] for i in indices[:split_train]]))
  val_set_x = concat_array(np.array([X[i] for i in indices[split_train:split_val]]))
  val_set_y = concat_array(np.array([Y[i] for i in indices[split_train:split_val]]))
  val_set_z = concat_array(np.array([Z[i] for i in indices[split_train:split_val]]))

  test_set_x = concat_array(np.array([X[i] for i in indices[split_val:]]))
  test_set_y = concat_array(np.array([Y[i] for i in indices[split_val:]]))
  test_set_z = concat_array(np.array([Z[i] for i in indices[split_val:]]))
  print(len(train_set_x))
  print(len(val_set_y))
  print(len(test_set_x))

  train_dataset = FeatureDataSet(train_set_x, train_set_y, train_set_z)  
  validation_dataset = FeatureDataSet(val_set_x, val_set_y, val_set_z)
  test_dataset = FeatureDataSet(test_set_x, test_set_y, test_set_z)
  return train_dataset, validation_dataset, test_dataset


def data_preperation(num_of_subject, proprtions=[0.7, 0.9], cross_subject=True, down=False): 
  pseudo_random(configurations.random_seed)
  X, Y, Z = load_data(import_model_data, num_of_subject, label_type=1, concat=cross_subject)
  
  # equalize the number of different labels 
  if down:
    X, Y, Z = down_sampling(X, Y, Z)
  else:
    X, Y, Z = over_sampling(X, Y, Z)

  if cross_subject:
    train_dataset, validation_dataset, test_dataset = trial_data_split(X, Y, Z, proprtions)
  else:
    train_dataset, validation_dataset, test_dataset = subjects_data_split(X, Y, Z, proprtions)
  return train_dataset, validation_dataset, test_dataset


def create_model(type, first_layer_channels=10, second_layer_channels=10, third_layer_channels=configurations.num_of_filters, num_of_heads=1, cnnConfig="DBBNNN", rnnType="lstm", bi=False, rnnConfig="N", middleLayers=1, dropout=.1, sa_dropout=0.1, usePosition=False, order="rnn", attention=False):
  pseudo_random(configurations.random_seed)
  if type == "rnn":
    mid_layer = myRNN(rnnType, third_layer_channels, bi, rnnConfig, layers=middleLayers)
  elif type == "sa":
    mid_layer = Self_attention(third_layer_channels, num_of_heads=num_of_heads, dropOut=sa_dropout, layers=middleLayers, usePosition=usePosition)
  elif type == "both":
    mid_layer = Combination(third_layer_channels, num_of_heads=num_of_heads, dropOut=dropout, layers=middleLayers, usePosition=usePosition, bi=bi, configuration=rnnConfig, rnnType=rnnType, order=order)
  model = KiNET(mid_layer, first_layer_channels=first_layer_channels, 
                second_layer_channels=second_layer_channels, 
                third_layer_channels=third_layer_channels, 
                cnnConfig=cnnConfig, attention=attention,dropout=dropout)
  return model


def import_model(name):
  model = tr.load(configurations.model_path + name + ".pth")
  return model

