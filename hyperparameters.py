# -*- coding: utf-8 -*-
from run_preperation import data_preperation, create_model
import torch as tr
import torch.nn as nn
from evaluation import Evaluator
from torch.utils.data import DataLoader
import configurations
import optuna


bdn_cnf = ["NNN", "DDD", "BBB","BDD","BDN","BND","DBB","DBN","DNB"]
act_cnf = ["NNN", "RRR", "LLL", "RNN", "LNN", "NNR", "NNL"]
rnn_types =["lstm", "gru"]
is_bi = ["bi", "not"]
activation = ["N", "R", "L"]

optimizers = ['SGD', 'ADAM']


def make_train_step(model, loss_fn, optimizer, device):
  # Builds function that performs a step in the train loop
  def train_step(x, y):
    # sets the model in a train mode
    model.train()
    # feed forward
    y_hat = model(x)
    # compute the loss

    loss = loss_fn(y_hat, y)      
    
    # compute gradients
    loss.backward()
    # update parameters
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()

  return train_step



def training(trial, model, train_dataset, val_dataset, optim="SGD", num_of_epochs=1000, lr=.01, save_model=False, save_performence=False, name=""): 

  #define the device
  device = 'cuda' if tr.cuda.is_available() else 'cpu'
  print(f"we are working on: {device}")

  # move the model to the device
  model = model.to(device)
  #model = model.half()

  # define the loss function to be log loss (or binary cross entropy loss)
  loss_fn = nn.BCELoss(reduction='mean')
  # define the optimizer
  if optim == "SGD":
    optimizer = tr.optim.SGD(model.parameters(), lr=lr)
  elif optim == "ADAM":
    optimizer = tr.optim.Adam(model.parameters(), lr=lr)

  # create the step function
  train_step = make_train_step(model, loss_fn, optimizer, device)

  # create evaluator
  eval = Evaluator()

  # saving the best model
  best_model = model.state_dict()


  train_loader = DataLoader(dataset=train_dataset, batch_size=configurations.batch_size, shuffle=False,num_workers=0, pin_memory=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True)

  

  number_of_train_batches = int(len(train_dataset) / 16)
  number_of_val_batches = int(len(val_dataset) / len(val_dataset))


  # training loop
  for epoch in range(num_of_epochs):
    total_loss = 0
    total_val_loss = 0
    for x_batch, y_batch,_ in train_loader:
      #move the batches to the gpu (hopefully)
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)

      # learm
      loss = train_step(x_batch, y_batch)
      #compute the total loss
      total_loss += loss
    
    #compute the mean loss
    mean_loss = total_loss / number_of_train_batches
    
    #evaluation on validation set
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
    if (epoch + 1) % 100 == 0:
      #print("==================================================")
      print(f"epoch number {epoch + 1}:")
      eval.print_partial_statistics()

    # saving the best model so far
    if eval.is_new_best():
      best_model = model.state_dict()
      
    # if the performance of the model not good, the model will stop early
    if epoch in configurations.early_stop and configurations.early_stop[epoch] > eval.mean_of_last_n(10):
      print("early stop due to low model performence")
      raise optuna.TrialPruned()
    
    trial.report(eval.max_percentage_value(), epoch)
    if trial.should_prune():
        print(eval.max_percentage_value())
        if eval.interesting():
            if save_model:
              tr.save(model, configurations.model_path + name + ".pth")
          
            if save_performence:
              eval.export(name)

        raise optuna.TrialPruned()


  model.load_state_dict(best_model)

  if save_model:
    tr.save(model, configurations.model_path + name + ".pth")

  if save_performence:
    eval.export(name)

  return eval.max_percentage_value()



def objective(trial):
    cfg = {
        'bdn_cnn': trial.suggest_categorical('bdn_cnn', bdn_cnf),
        'act_cnn': trial.suggest_categorical('act_cnn', act_cnf),
        'first_layer_channels': trial.suggest_int('first_layer_channels', 6, 30, step=2),
        'second_layer_channels': trial.suggest_int('second_layer_channels', 6, 30, step=2),
        'third_layer_channels': trial.suggest_int('third_layer_channels', 6, 30, step=2),
        'drop_out': trial.suggest_float('drop_out', .1, .7, step=.1),
        
        'type': trial.suggest_categorical('type', rnn_types),
        'bi_rnn': trial.suggest_categorical('bi_rnn', is_bi),
        'act_rnn': trial.suggest_categorical('act_rnn', activation),
        'rnn_layers': trial.suggest_int('rnn_layers', 1, 3),
        
        #'sa_layers': trial.suggest_int('sa_layers', 1, 3),
        #'sa_heads': trial.suggest_int('sa_heads', 1, 4),
        #'sa_drop_out': trial.suggest_float('sa_drop_out',.1, .7, step=.1),
        
        'optimizer': trial.suggest_categorical('optimizer', optimizers),
        'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    }
    print(f"trial number: {trial.number}")
    '''if cfg['third_layer_channels'] % cfg['sa_heads'] != 0:
        cfg['sa_heads'] = 2'''
    bi = True if cfg['bi_rnn'] == 'bi' else False
    bi=False
    model = create_model('rnn', first_layer_channels=cfg['first_layer_channels'], 
                         second_layer_channels=cfg['second_layer_channels'],
                         third_layer_channels=cfg['third_layer_channels'],
                         cnnConfig=cfg['bdn_cnn']+cfg['act_cnn'], bi=bi,
                         middleLayers=cfg['rnn_layers'], dropout=cfg['drop_out'],
                         rnnType=cfg['type'])

    print(model)
    train_dataset, validation_dataset, test_dataset = data_preperation(-1, cross_subject=True, equal=True)

    accuracy = training(trial, model, train_dataset, validation_dataset, optim=cfg['optimizer'], num_of_epochs=1500, 
                        lr=cfg['lr'], save_model=True, save_performence=True, name='strnn'+str(trial.number))
    
    return accuracy

'''
    model = create_model('sa', first_layer_channels=cfg['first_layer_channels'], 
                         second_layer_channels=cfg['second_layer_channels'], 
                         third_layer_channels=cfg['third_layer_channels'],
                         cnnConfig=cfg['bdn_cnn']+cfg['act_cnn'], rnnType=cfg['type'], bi=bi,
                         middleLayers=cfg['sa_layers'], dropout=cfg['drop_out'], 
                         num_of_heads=cfg['sa_heads'])'''
