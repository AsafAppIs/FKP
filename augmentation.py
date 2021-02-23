# -*- coding: utf-8 -*-
import random
from math import sin, cos, radians
import numpy as np
from import_data import import_split_data
from export_data import export_cnn_model_data
import configurations
from math import sqrt


def rotationMatrix(yaw, pitch, roll):
  # convert degrees to radians
  yaw = radians(yaw)
  pitch = radians(pitch)
  roll = radians(roll)

  # define  the rotation matrix
  mat = np.zeros((3,3))

  mat[0,0] = cos(yaw)*cos(pitch)
  mat[0,1] = cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll)
  mat[0,2] = cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll)

  mat[1,0] = sin(yaw)*cos(pitch)
  mat[1,1] = sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll)
  mat[1,2] = sin(yaw)*sin(pitch)*cos(roll) - cos(yaw)*sin(roll)

  mat[2,0] = -sin(pitch)
  mat[2,1] = cos(pitch)*sin(roll)
  mat[2,2] = cos(pitch)*cos(roll)

  return mat

def posOrNeg(num):
  if random.randint(0,1) == 1:
    return num
  return -num

def augmentate(train_dataset, yaw, pitch, roll):
    #train_dataset, _, _ = import_split_data()
    data = np.copy(train_dataset.X)
    lables= np.copy(train_dataset.Z)
    matrix = rotationMatrix(yaw, pitch, roll)
    augmented_trial= np.dot(data,matrix)  
    return augmented_trial.squeeze(), lables

def createAugmentation(numOfAug=10, low=10, high=20):
    train_dataset, _, _ = import_split_data()
    for i in range(1, numOfAug+1):
      new_path = configurations.augmentation_path + "augmentation" + str(i) + ".csv"
      yaw = posOrNeg(random.uniform(low,high))
      pitch = posOrNeg(random.uniform(low,high))
      roll = posOrNeg(random.uniform(low,high))
      print(f"creating {new_path} with parameters: yaw={yaw}, pitch={pitch}, roll={roll}")
      X, Y = augmentate(train_dataset, yaw, pitch, roll)
      export_cnn_model_data(new_path, X, Y)


def augmentationMagnitude(yaw, pitch, roll):
    magnitude = sqrt(yaw ** 2 + pitch ** 2 + roll ** 2)
    if magnitude > 6 or magnitude < sqrt(3):
        return False
    return True

def gridCreateAugmentation():
    train_dataset, _, _ = import_split_data()
    count = 1
    for yaw in range(-3,4,2):
        for pitch in range(-5,6,2):
            for roll in range(-5,6,2):
                if not augmentationMagnitude(yaw, pitch, roll):
                    continue
                new_path = configurations.augmentation_path + "augmentation" + str(count) + ".csv"
                print(f"creating {new_path} with parameters: yaw={yaw}, pitch={pitch}, roll={roll}")
                X, Y = augmentate(train_dataset, yaw, pitch, roll)
                export_cnn_model_data(new_path, X, Y)
                count += 1
                
