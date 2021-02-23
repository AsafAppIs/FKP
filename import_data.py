# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import configurations
from dataset import FeatureDataSet

def csv_to_trials(trial):
  dataframes = np.array(np.array_split(trial, 123))
  dataframes = dataframes.reshape((123,4,3))
  return dataframes



def formating(X, fun):
  X = np.array([fun(trial) for trial in X])
  return X



def import_model_data(num=0, splited=False, name=""):
  is_splited = ["data/subject" + str(num), "splited model data/" + name]
  path = configurations.general_path + is_splited[splited] + ".csv"
  df = pd.read_csv(path, header=None)
  data = np.array(df)
  
  X = data[:,:-2]
  X = formating(X, csv_to_trials)
  X = np.expand_dims(X, axis=1)

  Y = data[:, -2:]
  return X, Y



def not_zero(num):
  if num:
    return 1
  return 0



def load_data(import_function, num_of_subjects=0, fake=False, only_fake=False, fake_data=[], label_type=0, manipulation_generalization=0, concat=True):
  if num_of_subjects and only_fake:
    raise Exception("Can't have both \"only_fake\" and  \"num_of_subjects\" ")
  if label_type == 0 and manipulation_generalization != 0:
    raise Exception("There is no different types of SoA labels")
  if not fake and only_fake:
    raise Exception("Can't have both \"only_fake\" without  \"fake\" ")
  np.random.seed(configurations.random_seed)

  if num_of_subjects == -1:
    num_of_subjects = configurations.number_of_subjects
    
  indices = list(range(1, num_of_subjects + 1))
  np.random.shuffle(indices)
  indices = indices[:num_of_subjects]
  X = []
  Y = []
  Z = []
  count = 0
  # load real data
  for i in indices:
    x, y = import_function(i)
    count += len(x)
    z = [[not_zero(label[1 - label_type]), int(label[1])] for label in y]
    if fake:
      y = [0 for label in y]
    elif label_type == 0:
      y = [label[0] for label in y]
    elif label_type == 1:
      if manipulation_generalization == 0:
        y = [not_zero(label[1]) for label in y]
      else:
        manipulation_generalization.append(0)
        filter = [i for i in range(len(y)) if y[i,1] not in manipulation_generalization]
        x = np.delete(x, filter)
        y = np.delete(y, filter)
        y = [label[1] for label in y]

    X.append(x)
    Y.append(y)
    Z.append(z)

    # load fake data
  if fake:
    for i, index in enumerate(fake_data):
      x, y = import_function(index, True)
      if only_fake:
        y = [i for label in y]
      else:
        y = [1 for label in y]

      X.append(x)
      Y.append(y)
  # concatenate the samples and labels lists
  if concat:
    for i in range(1, len(X)):
      X[0] = np.concatenate((X[0], X[i]), axis=0)
      Y[0] = np.concatenate((Y[0], Y[i]), axis=0)
      Z[0] = np.concatenate((Z[0], Z[i]), axis=0)
    X = X[0]
    Y = Y[0]
    Z = Z[0]

  return X, Y, Z


def import_augemntations_data(indices=[0], concat=True):
  print(indices)
  X = []
  Y = []
  for idx in indices:
    path = configurations.augmentation_path + "augmentation" + str(idx) + ".csv"
    df = pd.read_csv(path, header=None)
    data = np.array(df)
    x = data[:,:-2]
    x = formating(x, csv_to_trials)
    x = np.expand_dims(x, axis=1)
    y = data[:, -2:]

    X.append(x)
    Y.append(y)
  if concat:
      for i in range(1, len(X)):
        X[0] = np.concatenate((X[0], X[i]), axis=0)
        Y[0] = np.concatenate((Y[0], Y[i]), axis=0)
      X = X[0]
      Y = Y[0]

  return X, Y



def import_split_data(manipulation_generalization=0, indices=[0], mix=False, concat=True):
  # load train data
  #x, y = import_model_data(splited=True, name="train")
  x, y = import_augemntations_data(indices, concat)
  if concat:
      z = [[int(label[0]), int(label[1])] for label in y] 
      y = [not_zero(label[1]) for label in y]
      train_dataset = FeatureDataSet(x, y, z, mix)
  else:
      train_dataset = []
      for i in range(len(x)):
          z = [[int(label[0]), int(label[1])] for label in y[i]]
          temp_y = [not_zero(label[1]) for label in y[i]]
          train_dataset.append(FeatureDataSet(x[i], temp_y, z, mix))

  # load validation data
  x, y = import_model_data(splited=True, name="val")
  z = [[int(label[0]), int(label[1])] for label in y] 
  y = [not_zero(label[1]) for label in y]
  validation_dataset = FeatureDataSet(x, y, z)

  # load validation data
  x, y = import_model_data(splited=True, name="test")
  z = [[int(label[0]), int(label[1])] for label in y] 
  y = [not_zero(label[1]) for label in y]
  test_dataset = FeatureDataSet(x, y, z)

  return train_dataset , validation_dataset, test_dataset

