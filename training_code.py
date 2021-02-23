# -*- coding: utf-8 -*-
import torch as tr
import torch.nn as nn
from evaluation import Evaluator
from torch.utils.data import DataLoader
import configurations
from copy import deepcopy
from models.attention import AttentionLayer
import numpy as np

def attentionAdjacment(model):
    for param in model.parameters():
        param.requires_grad = False
    size = model.attentionLayer.input_size
    model.attentionLayer = AttentionLayer(size)
    
    return model, model.attentionLayer.parameters()


def make_train_step(model, loss_fn, optimizer, device):
  # Builds function that performs a step in the train loop
  def train_step(x, y):
    # sets the model in a train mode
    model.train()
    # feed forward
    y_hat = model(x)
    # compute the loss
    #y = tr.unsqueeze(y, 1)
    # y = y.type(tr.FloatTensor).to(device)

    loss = loss_fn(y_hat, y)      
    
    #count correct predictions
    count = count_correct_predictions(y_hat, y)
    
    # compute gradients
    loss.backward()
    # update parameters
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), count

  return train_step

def count_correct_predictions(yhat, y):
  prediction = [1 if x > 0.5 else 0 for x in yhat]
  correct_predictions = [1 if prediction[i] == y[i] else 0 for i in range(len(y))]
  return sum(correct_predictions)


def training(model, train_dataset, val_dataset, optim="SGD", num_of_epochs=1000, lr=.01, save_model=False, save_performence=False, name="", onlyAttention=False): 

  #define the device
  device = 'cuda' if tr.cuda.is_available() else 'cpu'
  print(f"we are working on: {device}")

  
  
  #define the parameters to train
  parameters = model.parameters()

  if onlyAttention:
      print("only attention layer training")
      model, parameters = attentionAdjacment(model)
  
    # define the loss function to be log loss (or binary cross entropy loss)
  loss_fn = nn.BCELoss(reduction='mean')
  
  # move the model to the device
  model = model.to(device)
  
  # define the optimizer
  if optim == "SGD":
    optimizer = tr.optim.SGD(parameters, lr=lr)
  elif optim == "ADAM":
    optimizer = tr.optim.Adam(parameters, lr=lr)

  # create the step function
  train_step = make_train_step(model, loss_fn, optimizer, device)

  # create evaluator
  eval = Evaluator()

  # saving the best model
  best_model = model.state_dict()


  train_loader = DataLoader(dataset=train_dataset, batch_size=configurations.batch_size, shuffle=False,num_workers=0, pin_memory=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True)

  print(len(train_dataset))

  number_of_train_batches = int(len(train_dataset) / 16)
  number_of_val_batches = int(len(val_dataset) / len(val_dataset))

  print(f"number_of_train_batches: {number_of_train_batches}")
  print(f"number_of_val_batches: {number_of_val_batches}")

  # training loop
  for epoch in range(num_of_epochs):
    total_loss = 0
    total_val_loss = 0
    train_correct_counter = 0
    for i, (x_batch, y_batch,_) in enumerate(train_loader):
      #move the batches to the gpu (hopefully)
      x_batch = x_batch.to(device)

      y_batch = y_batch.to(device)

      # learm
      loss, count = train_step(x_batch, y_batch)
      
      #compute the total loss
      total_loss += loss
      
      #compute total correct answers in train
      train_correct_counter += count
      if (i+1)%(int(number_of_train_batches / 10)) == 0 and False:
          val, val_rate = val_eval(model, val_dataset, loss_fn)
          print(f"part number {round((i+1)/(number_of_train_batches / 10))} of epoch {epoch}: ")
          print(f"train loss is: {(total_loss/(i+1)):.4f} and train rate is {(train_correct_counter/(configurations.batch_size * (i+1))):.4f}")
          print(f"val loss is: {val:.4f} and val rate is {val_rate:.4f}")
          print()
      
      
    
    #compute the mean loss
    mean_loss = total_loss / number_of_train_batches
    with tr.no_grad():
      model.eval()
      for x_val, y_val, z_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            
            

            yhat = model(x_val)
            val_loss = loss_fn(yhat, y_val)

            eval.update_counter(yhat, y_val, z_val)


            total_val_loss += val_loss
    mean_val_loss = total_val_loss/number_of_val_batches
    eval.update_loss(mean_loss, mean_val_loss, train_correct_counter/len(train_dataset))
    # saving the best model so far
    if eval.is_new_best():
      best_model = deepcopy(model.state_dict())
    if (epoch + 1) % configurations.print_const == 0:
      #print("==================================================")
      print(f"epoch number {epoch + 1}:")
      eval.print_partial_statistics((epoch + 1) % configurations.deep_print_const == 0)
    
    # if the performance of the model not good, the model will stop early
    if epoch in configurations.early_stop and configurations.early_stop[epoch] > eval.mean_of_last_n(10):
      print("early stop due to low model performence")
      break
  eval.print_best_stats()
  eval.plot_stats()
  model.load_state_dict(best_model)

  if save_model:
    tr.save(model, configurations.model_path + name + ".pth")

  if save_performence:
    eval.export(name)
  
  return eval



def val_eval(model, validation_data, loss_fn):
    model.eval()
    eval = Evaluator()
    x_val = validation_data.X.to('cuda')
    y_val = validation_data.Y.to('cuda')
    z_val = validation_data.Z.to('cuda')
    yhat = model(x_val)
    val_loss = loss_fn(yhat, y_val)
    eval.update_counter(yhat, y_val, z_val)
    eval.update_loss(0,0,0)
    return val_loss, eval.total_percenage[0]


def print_aug_efficiency(rate_table, loss_table):
    for i in range(1, len(rate_table)):
        print("***********")
        print(f"augmentation number {i} ")
        total_rate = 0
        imporvements_rate = 0
        total_loss = 0
        imporvements_loss = 0
        for j in range(len(rate_table[i])):
            rate_gap = rate_table[i][j] - rate_table[i - 1][j]
            loss_gap = loss_table[i][j] - loss_table[i - 1][j]  
            if rate_gap > 0:
                imporvements_rate += 1
            if loss_gap < 0:
               imporvements_loss += 1 
            total_rate += rate_gap
            total_loss += loss_gap
        print(f"{imporvements_rate} / {len(rate_table[i])} rate improvments, total rate: {total_rate}")
        print(f"{imporvements_loss} / {len(rate_table[i])} loss improvments, total loss: {total_loss}")
        print("***********")
        print()
                


def training_aug(model, train_dataset, val_dataset, optim="SGD", num_of_epochs=6, lr=.01, save_model=False, save_performence=False, name=""): 
    
  #define the device
  device = 'cuda' if tr.cuda.is_available() else 'cpu'
  print(f"we are working on: {device}")
  
  
  percent_per_aug = np.zeros((51, num_of_epochs))
  loss_per_aug = np.zeros((51, num_of_epochs))

  
  
  #define the parameters to train
  parameters = model.parameters()

  
  best_model = model.state_dict()

  # define the loss function to be log loss (or binary cross entropy loss)
  loss_fn = nn.BCELoss(reduction='mean')
  
  # move the model to the device
  model = model.to(device)
  
  # define the optimizer
  if optim == "SGD":
    optimizer = tr.optim.SGD(parameters, lr=lr)
  elif optim == "ADAM":
    optimizer = tr.optim.Adam(parameters, lr=lr)

  # create the step function
  train_step = make_train_step(model, loss_fn, optimizer, device)

  # create evaluator
  eval = Evaluator()
  


  train_loader = DataLoader(dataset=train_dataset, batch_size=configurations.batch_size, shuffle=False,num_workers=0, pin_memory=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True)

  

  number_of_train_batches = int(len(train_dataset) / configurations.batch_size)
  number_of_val_batches = int(len(val_dataset) / len(val_dataset))

  print(f"number_of_train_batches: {number_of_train_batches}")
  print(f"number_of_val_batches: {number_of_val_batches}")

  # training loop
  for epoch in range(num_of_epochs):
    total_loss = 0
    total_val_loss = 0
    for i, (x_batch, y_batch,_) in enumerate(train_loader):
      #move the batches to the gpu (hopefully)
      x_batch = x_batch.to(device)

      y_batch = y_batch.to(device)

      # learm
      loss = train_step(x_batch, y_batch)
      #compute the total loss
      total_loss += loss
      if (i+1)%(int(len(train_dataset) / 10)) == 0:
          train, loss, rate = val_eval(model, val_dataset, loss_fn)
          print(f"part number {(i+1)/(int(len(train_dataset) / 10))}: train is: {train:.4f} loss is: {loss:.4f} and rate is {rate:.4f}")
          percent_per_aug[int((i+1)/50) - 1][ epoch] = rate
          loss_per_aug[int((i+1)/50) - 1][ epoch] = loss

    
    #compute the mean loss
    mean_loss = total_loss / number_of_train_batches
    with tr.no_grad():
      model.eval()
      for x_val, y_val, z_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            
            

            yhat = model(x_val)
            val_loss = loss_fn(yhat, y_val)

            eval.update_counter(yhat, y_val, z_val)


            total_val_loss += val_loss
    mean_val_loss = total_val_loss/number_of_val_batches
    eval.update_loss(mean_loss, mean_val_loss)
    if (epoch + 1) % configurations.print_const == 0:
      #print("==================================================")
      print(f"epoch number {epoch + 1}:")
      eval.print_partial_statistics((epoch + 1) % configurations.deep_print_const == 0)
  eval.print_best_stats()
  eval.plot_stats()
  model.load_state_dict(best_model)

  if save_model:
    tr.save(model, configurations.model_path + name + ".pth")

  if save_performence:
    eval.export(name)
  #print_aug_efficiency(percent_per_aug, loss_per_aug)
    