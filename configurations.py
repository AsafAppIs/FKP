# -*- coding: utf-8 -*-
import torch as tr
import numpy as np
import matplotlib.pyplot as plt


random_seed = 42
number_of_subjects = 57
batch_size = 16
global_padding_size = 123
num_of_filters = 16
num_of_middle_channels = 10


min_tf = 110
max_tf = 130
finger_location = {"base": 1, "thumb_base": 2, "inter": 3, "distal": 4}

print_const = 10
deep_print_const = 100

early_stop = {250:0.6, 500:0.65, 1000:.7}

tr.manual_seed(random_seed)
tr.cuda.manual_seed(random_seed)
np.random.seed(random_seed)

general_path = "C:/Users/User/Documents/Asaf/fkp/"
data_path = "C:/Users/User/Documents/Asaf/fkp/data/"
general_path = "C:/Users/User/Documents/asaf/Finger Kinematic analysis/"
data_path = "C:/Users/User/Documents/asaf/Finger Kinematic analysis/data/"
model_path = "C:/Users/User/Documents/Asaf/fkp/results/models/"
performance_path = "C:/Users/User/Documents/Asaf/fkp/results/performence/"
augmentation_path = "C:/Users/User/Documents/Asaf/fkp/augmented_data/"


plt.style.use('fivethirtyeight')