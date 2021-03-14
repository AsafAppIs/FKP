# -*- coding: utf-8 -*-
from run_preperation import data_preperation, create_model
from training_code import training

# loading the data and split it to train, validation and test sets
train_dataset, validation_dataset, test_dataset = data_preperation()

# creating our best performence model
model = create_model(type="sa", first_layer_channels=28, second_layer_channels=18, third_layer_channels=20, 
                     num_of_heads=2, cnnConfig="BNDRRR", rnnType="lstm", bi=True, rnnConfig="N", middleLayers=2,
                     dropout=.6, sa_dropout=.7,attention=True)

# train the model with his optimal hyperparameters 
training(model, train_dataset, validation_dataset ,optim="ADAM", num_of_epochs=1500, lr=0.0002274107163111164)
