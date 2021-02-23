# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import configurations

def format_func_y(value, tick_number):
  return f"{value*100:.1f}%"

def format_func_x(value, tick_number):
  return f"{int(value*5)}"

def to_float(x):
  if isinstance(x, float):
    return x
  x = x.split('(')
  if len(x) == 1:
    return float(x[0])
  x = x[1]
  x = x.split(',')[0]
  return float(x)



class Counter():
  def __init__(self):
    self.correct_predictions = 0.0
    self.total_predictions = 0.0

    self.total_predictions_second_label = 0.0
    self.correct_predictions_second_label = 0.0

    self.total_predictions_array = np.zeros((10, 2))
    self.correct_predictions_array = np.zeros((10, 2))

    

  def add(self, label, l_type):
    self.total_predictions += 1
    self.total_predictions_second_label += label
    self.total_predictions_array[l_type, label] += 1

  def update(self, yhat, y_val, z_val):

    predictions = [1 if x > 0.5 else 0 for x in yhat]
    for i, prediction in enumerate(predictions):
      self.add(z_val[i, 0], z_val[i, 1])
      if prediction == y_val[i]:
        # count general correct predictions
        self.correct_predictions += 1

        # count specific second label correct predictions
        # if the second label of the trial is 1 so we will count it
        self.correct_predictions_second_label += z_val[i, 0]

        # count specific type correct predictions
        # z_val[i, 1] is the type of the specific trial, so we want to add one to the coresponding cell in the array
        self.correct_predictions_array[z_val[i, 1], z_val[i, 0]] += 1

  def clean(self):
    self.correct_predictions = 0.0
    self.total_predictions = 0.0

    self.total_predictions_second_label = 0.0
    self.correct_predictions_second_label = 0.0

    self.total_predictions_array = np.zeros((10, 2))
    self.correct_predictions_array = np.zeros((10, 2))


  def total_accuracy(self):
    return self.correct_predictions / self.total_predictions

  def second_label_zero_accuracy(self):
    return (self.correct_predictions - self.correct_predictions_second_label) / (self.total_predictions - self.total_predictions_second_label)

  def second_label_one_accuracy(self):
    return self.correct_predictions_second_label / self.total_predictions_second_label

  def print_stats(self):
    print(f"total predictions: {self.total_predictions}, from them {self.correct_predictions}")
    print(f"total SoA predictions: {self.total_predictions_second_label}, from them {self.correct_predictions_second_label}")
    print(f"total not SoA predictions: {self.total_predictions - self.total_predictions_second_label}, from them {self.correct_predictions - self.correct_predictions_second_label}")
    for i in range(10):
      print(f"total {i} type predictions: {sum(self.total_predictions_array[i])}, from them {sum(self.correct_predictions_array[i])}")
      print(f"total {i} without SoA type predictions: {self.total_predictions_array[i, 0]}, from them {self.correct_predictions_array[i, 0]}")
      print(f"total {i} with SoA type predictions: {self.total_predictions_array[i, 1]}, from them {self.correct_predictions_array[i, 1]}")


  def type_accuracy(self):
    accuracy_array = np.zeros((4,3))

    # no manipulation calculations
    accuracy_array[0,0] = np.sum(self.correct_predictions_array[0]) / np.sum(self.total_predictions_array[0])
    accuracy_array[0,1] = self.correct_predictions_array[0, 0] / self.total_predictions_array[0, 0]
    accuracy_array[0,2] = self.correct_predictions_array[0, 1] / self.total_predictions_array[0, 1]
    for i in range(1, 4):
      accuracy_array[i,0] = np.sum(self.correct_predictions_array[i*3-2: i*3+1]) / np.sum(self.total_predictions_array[i*3-2: i*3+1])
      accuracy_array[i,1] = np.sum(self.correct_predictions_array[i*3-2: i*3+1, 0]) / np.sum(self.total_predictions_array[i*3-2: i*3+1, 0])
      accuracy_array[i,2] = np.sum(self.correct_predictions_array[i*3-2: i*3+1, 1]) / np.sum(self.total_predictions_array[i*3-2: i*3+1, 1])


    return accuracy_array


  def deep_type_accuracy(self):
    accuracy_array = np.zeros((10,3))

    for i in range(0, 10):
      accuracy_array[i,0] = np.sum(self.correct_predictions_array[i]) / np.sum(self.total_predictions_array[i])
      accuracy_array[i,1] = self.correct_predictions_array[i, 0] / self.total_predictions_array[i, 0]
      accuracy_array[i,2] = self.correct_predictions_array[i, 1] / self.total_predictions_array[i, 1]
    return accuracy_array



class Evaluator():
  def __init__(self):
    self.counter = Counter()

    self.train_loss = []
    self.validation_loss = []

    self.total_percenage = []
    self.train_total_percenage = []
    self.second_one_label_percenage = []
    self.second_zero_label_percenage = []

    self.types_percenage = []
    for i in range(4):
      self.types_percenage.append([])
      for _ in range(3):
        self.types_percenage[i].append([])

    self.deep_types_percenage = []
    for i in range(10):
      self.deep_types_percenage.append([])
      for _ in range(3):
        self.deep_types_percenage[i].append([])
    
    self.max=0
    self.maxId=0
    
  def mean_of_last_n(self, num):
    counter = np.sum(self.total_percenage[-num:])
    avg = counter / num
    return avg




  def is_new_best(self):
    return self.max_percentage() == len(self.total_percenage) - 1

  def min_validation_loss(self):
    return self.validation_loss.index(min(self.validation_loss))

  def max_percentage_value(self):
    return max(self.total_percenage)


  def max_percentage(self):
    return self.total_percenage.index(max(self.total_percenage))

  
  def min_train_loss(self):
      return min(self.train_loss)

  
  def interesting(self):
      if self.max_percentage_value() > 0.795 or self.min_train_loss() < 0.15:
          return True
      return False


  def update_loss(self, train_loss, validation_loss, train_rate):
    self.train_loss.append(train_loss)
    self.validation_loss.append(validation_loss)
    self.train_total_percenage.append(train_rate)
    self.total_percenage.append(self.counter.total_accuracy())
    self.second_zero_label_percenage.append(self.counter.second_label_zero_accuracy().item())
    self.second_one_label_percenage.append(self.counter.second_label_one_accuracy().item())
    

    type_accuracy = self.counter.type_accuracy()
    for i in range(4):
      for j in range(3):
        self.types_percenage[i][j].append(type_accuracy[i, j])

    type_accuracy = self.counter.deep_type_accuracy()
    for i in range(10):
      for j in range(3):
        self.deep_types_percenage[i][j].append(type_accuracy[i, j])

    # clean the counter for the next epoch
    #self.counter.print_stats()
    self.counter.clean()


  def update_counter(self, yhat, y_val, z_val):
    self.counter.update(yhat, y_val, z_val)
  
  def print_best_stats(self):
    best_index = self.max_percentage()
    print(f"best results, epoch number: {best_index}")
    self.print_partial_statistics(full=True, index=best_index)

  def print_partial_statistics(self, full=False, index=-1):

    max_accuracy = self.max_percentage()
    min_validation = self.min_validation_loss()
    print("==================================================")
    print(f"mean training loss: {self.train_loss[index]}")
    print(f"mean validation loss: {self.validation_loss[index]}")
    print()
    print(f"train accuracy is {self.train_total_percenage[index] * 100:.3f}%")
    print(f"validation accuracy is {self.total_percenage[index] * 100:.3f}%")
    print()
    if full:
      self.print_full_statistics(index)
    if index == -1:
      print(f"the best epoch in term of accuracy so far: {max_accuracy + 1} with {self.validation_loss[max_accuracy]} loss and accuracy of {self.total_percenage[max_accuracy] * 100:.3f}%")
      print(f"the best epoch in term of validation error so far: {min_validation + 1} with {self.validation_loss[min_validation]} loss and accuracy of {self.total_percenage[min_validation] * 100:.3f}%")
    print("==================================================")
    print()

  def print_full_statistics(self, index=-1):
    print("===============================")
    print("deep statstic analysis:")
    print("cross labels stats")
    print(f"label 0 accuracy: {self.second_zero_label_percenage[index] * 100:.3f}%")
    print(f"label 1 accuracy: {self.second_one_label_percenage[index] * 100:.3f}%")
    print()
    print("_________")
    print("cross types stats:")
    for i in range(4):
      print(f"type {i} accuracy:")
      print(f"total: {self.types_percenage[i][0][index] * 100:.3f}%")
      print(f"without SoA: {self.types_percenage[i][1][index] * 100:.3f}%")
      print(f"with SoA: {self.types_percenage[i][2][index] * 100:.3f}%")
      print()
    print("_________")
    print("deep cross types stats:")
    for i in range(10):
      print(f"type {i} accuracy:")
      print(f"total: {self.deep_types_percenage[i][0][index] * 100:.3f}%")
      print(f"without SoA: {self.deep_types_percenage[i][1][index] * 100:.3f}%")
      print(f"with SoA: {self.deep_types_percenage[i][2][index] * 100:.3f}%")
      print()
    
    print("===============================")
    print()


  def plot_stats(self):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(20,12))

    # loss graph
    ax1.plot(self.train_loss[::5], color="b", label="mean train loss")
    ax1.plot(self.validation_loss[::5], color="r", label="mean validation loss")
    ax1.set_title("loss learning graph")
    ax1.set_xlabel(f"epoch number")
    ax1.set_ylabel(f"mean loss")
    ax1.legend()

    # edit thoe ticks
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func_x))



    # accuracy graph
    ax2.set_title("accuracy learning graph")
    ax2.set_xlabel(f"epoch number")
    ax2.set_ylabel(f"accuracy rate")


    ax2.plot(self.total_percenage[::5], color="r", label="total accuracy" )
    ax2.plot(self.train_total_percenage[::5], color="b", label="SoA")
    '''ax2.plot(self.types_percenage[0][0][::5], color="c", label="without manipulation accuracy", linewidth=1)
    ax2.plot(self.types_percenage[1][0][::5], color="m", label="temporal manipulation", linewidth=1)
    ax2.plot(self.types_percenage[2][0][::5], color="y", label="spatial manipulation", linewidth=1)
    ax2.plot(self.types_percenage[3][0][::5], color="k", label="anatomical manipulation", linewidth=1)'''
    ax2.legend()

    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_func_y))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func_x))


  def export(self, name):
    eval_lst = []
    eval_lst.append(self.train_loss)
    eval_lst.append(self.validation_loss)
    eval_lst.append(self.total_percenage)
    eval_lst.append(self.train_total_percenage)
    eval_lst.append(self.second_one_label_percenage)
    eval_lst.append(self.second_zero_label_percenage)

    for type_p in self.types_percenage:
      for line in type_p:
        eval_lst.append(line)

    for type_p in self.deep_types_percenage:
      for line in type_p:
        eval_lst.append(line)
    
    df = pd.DataFrame(eval_lst)

    df.to_csv(configurations.performance_path + name + ".csv", header=False, index=False)



  def import_p(self, name, train=False):
    df = pd.read_csv(configurations.performance_path + name + ".csv", header=None)
    
    eval_lst = df.values.tolist()
    
    train_offset = 1 if train else 0
    self.train_loss = [to_float(x) for x in eval_lst[0]]
    self.validation_loss = [to_float(x) for x in eval_lst[1]]
    self.total_percenage = [to_float(x) for x in eval_lst[2]]
    if train:
        self.train_total_percenage = [to_float(x) for x in eval_lst[3]]
    self.second_one_label_percenage = [to_float(x) for x in eval_lst[3 + train_offset]]
    self.second_zero_label_percenage =[to_float(x) for x in eval_lst[4 + train_offset]]


    counter = 5 + train_offset
    for i in range(4):
      for j in range(3):
        self.types_percenage[i][j] = eval_lst[counter]
        counter += 1

    type_accuracy = self.counter.deep_type_accuracy()
    for i in range(10):
      for j in range(3):
        self.deep_types_percenage[i][j] = eval_lst[counter]
        counter += 1











       
