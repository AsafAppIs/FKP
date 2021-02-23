# -*- coding: utf-8 -*-
from run_preperation import data_preperation, create_model, import_model
from training_code import training, val_eval, training_aug
from import_data import import_split_data
from hyperparameters import objective
import optuna
from logging import StreamHandler
import torch.nn as nn
from sys import stdout
import joblib
from evaluation import Evaluator
from torchinfo import summary
from augmentation import createAugmentation


#createAugmentation(numOfAug=50, low=0, high=5)
#optuna.logging.get_logger("optuna").addHandler(StreamHandler(stdout))
#study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner(min_resource=3), sampler=optuna.samplers.TPESampler(n_startup_trials=40, seed=42))
#study = joblib.load("strnn2_study.pkl")
#study.optimize(objective, n_trials=50, gc_after_trial=True)
#joblib.dump(study, "strnn3_study.pkl")
lst = [0,1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23,24, 25, 27,28,  29, 31, 33, 34, 35, 36, 37, 38, 41, 43, 46, 47, 49, 50]
performence = []
#train_dataset, validation_dataset, test_dataset = import_split_data(indices=lst, mix=True)
#train_dataset, validation_dataset, test_dataset = data_preperation(-1, down=False)
for i in range(1,113):
    print(f"Augmentation file {i}:")
    train_dataset, validation_dataset, test_dataset = import_split_data(indices=[i])
    model = create_model(type="rnn", first_layer_channels=20, second_layer_channels=30, third_layer_channels=14, 
                         num_of_heads=2, cnnConfig="BNDNNN", rnnType="lstm", bi=True, rnnConfig="N", middleLayers=3,
                         dropout=.2, sa_dropout=.2)
    eval = training(model, train_dataset, validation_dataset ,optim="SGD", num_of_epochs=50, 
                    lr=0.0027982272717049276)
    performence.append([eval.train_loss[-1], eval.train_total_percenage[-1], eval.validation_loss[-1], eval.total_percenage[-1]])
print(*performence, sep='\n')
#training(model, train_dataset, validation_dataset ,optim="SGD", num_of_epochs=1500, lr=0.0027982272717049276, save_model=True, save_performence=True, name="best rnn augementation")

'''
eval = Evaluator()
eval.import_p("best rnn", train=False)
print(eval.train_total_percenage[0:100])
eval.plot_stats()

loss_fn = nn.BCELoss(reduction='mean')
old_model = import_model("best rnn")
val_eval(old_model, validation_dataset,loss_fn)
model = create_model(type="rnn", first_layer_channels=20, second_layer_channels=30, third_layer_channels=14, 
                     num_of_heads=2, cnnConfig="BNDNNN", rnnType="lstm", bi=True, rnnConfig="N", middleLayers=3,
                     dropout=.2, sa_dropout=.2,attention=True)
model.load_state_dict(old_model.state_dict())
training(model, train_dataset, validation_dataset ,optim="SGD",lr=0.7982272717049276, num_of_epochs=500,
         save_model=True, save_performence=True, name="best rnn OAH attention", onlyAttention=True)

old_model = import_model("best sa")
val_eval(old_model, validation_dataset,loss_fn)


model = create_model(type="sa", first_layer_channels=28, second_layer_channels=18, third_layer_channels=20, 
                     num_of_heads=2, cnnConfig="BNDRRR", rnnType="lstm", bi=True, rnnConfig="N", middleLayers=2,
                     dropout=.6, sa_dropout=.7,attention=True)
model.load_state_dict(old_model.state_dict())
training(model, train_dataset, validation_dataset ,optim="ADAM",lr=0.0002274107163111164, num_of_epochs=500, 
         save_model=True, save_performence=True, name="best sa OAH attention", onlyAttention=True)

for i in range(len(train_dataset)):
    model = create_model(type="rnn", first_layer_channels=20, second_layer_channels=30, third_layer_channels=14, 
                         num_of_heads=2, cnnConfig="BNDNNN", rnnType="lstm", bi=True, rnnConfig="N", middleLayers=3,
                         dropout=.2, sa_dropout=.2)
    eval = training(model, train_dataset[i], validation_dataset ,optim="SGD", num_of_epochs=100, 
                    lr=0.0027982272717049276)
    performence.append([eval.train_loss[-1], eval.train_total_percenage[-1], eval.validation_loss[-1], eval.total_percenage[-1]])


'''
