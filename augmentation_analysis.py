# -*- coding: utf-8 -*-
import configurations
from functools import reduce
import math
import numpy as np
indices = [3, 10, 11, 18]

aug_cng_path = configurations.general_path + "aug cng.txt"
aug_result_path = configurations.general_path + "aug res.txt"
aug_sec_result_path = configurations.general_path + "aug res 2.txt"

def result_data():
    with open(aug_result_path, 'r') as reader:
        data = reader.read().split('***********')
        data = list(filter(lambda x: x != "\n\n", data))
        data = list(map(lambda x: [x.split()[index] for index in indices], data))
        data = [[float(y) for y in x] for x in data]
        return data
        
def sec_result_data():
    with open(aug_sec_result_path, 'r') as reader:
        data = reader.read().split("\n")
        data = [x.replace('[', '') for x in data]
        data = [x.replace(']', '') for x in data]
        data = [x.replace(',', '') for x in data]
        data = [x.replace('tensor(', '') for x in data]
        data = [x.replace('device=\'cuda:0\') ', '') for x in data]
        data = [[float(y) for y in x.split()] for x in data]
        return data
        
def cng_data_e():
    with open(aug_cng_path, 'r') as reader:
        data = reader.read().split("\n")
        data = list(map(lambda x: x.split('=')[1:], data))
        data = [list(map(lambda y: y.split(',')[:1], x)) for x in data] 
        data = [[float(y[0]) for y in x] for x in data]
        return data
        
def activate():
    res_2_data = sec_result_data()      
    res_data = result_data()
    cng_data = cng_data_e()[:-1]
    total_data = [cng_data[i] + res_2_data[i] for i in range(len(res_data))]
    total_devation = [math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) for x in cng_data]
    #total_devation = [abs(x[0])+ abs(x[1]) + abs(x[2]) for x in cng_data]
    total_devation = [abs(x[0]) for x in cng_data]
    
    loss = [x[2] for x in res_2_data]
    rate = [x[3] for x in res_2_data]
    
    print(np.corrcoef(total_devation, loss))
    print(np.corrcoef(total_devation, rate))
    
    print(np.corrcoef(rate, loss))
    
    
    total_data.sort(reverse=True, key=lambda x: x[6])
    
    
    for i in range(len(res_data)):
        print(f"{i+1} augmentation:")
        print(f"configurations: {total_data[i][:3]}")
        print(f"square distance is: {math.sqrt(total_data[i][0] ** 2 + total_data[i][1] ** 2 + total_data[i][2] ** 2)}")
        print(f"results: {total_data[i][3:]}")
        print("*********")
        print()
    
