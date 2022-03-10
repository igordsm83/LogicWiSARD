#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 15:22:17 2021

@author: igor
"""

import sys
sys.path.insert(0, '../lib/')
import logicwisard as lwsd
from wisard_tools import wisard_data_encode, eval_predictions, mnist_data_encode_b, write2file, mnist_data_encode_t, mnist_data_encode_z
from keras.datasets import mnist
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
import copy 



project_name = 'mnist'

try:
    os.remove("./log.txt")
except OSError:
    pass

start_time = time.time()
full_log = "--- RUNNING WISARD TRAINING "+project_name+" ---\n"

out_dir = './out/'

with open(out_dir+'/mWisard.pkl', 'rb') as inp:
    mWisard = pickle.load(inp)

addressSize = mWisard.address_size
thermo_resolution = 2 # mWisard.thermo_resolution

full_log += "\n> addressSize = " + str(addressSize) 
full_log += "\n> thermo_resolution = " + str(thermo_resolution) 
datetime_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H-%M-%S')
full_log += "\nStarting at: "+datetime_string+"\n"
write2file( full_log)


# Import 60000 images from mnist data set
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

classes = ['0','1','2','3','4','5','6','7','8','9']
print("Test set input: "+str(X_test.shape))
print("Test set output: "+str(Y_test.shape))


print('>>> Encoding test set...')
# X_test_lst = mnist_data_encode_b(X_test)
X_test_lst = mnist_data_encode_t(X_test, 0,255,thermo_resolution)
# X_test_lst, _, _ = mnist_data_encode_z(X_test, x_mean, x_std)
# X_test_lst = wisard_data_encode(X_test, classes, resolution=thermo_resolution, minimum=0, maximum=255)
X_test_lst = X_test_lst.astype(int)
Y_test = Y_test.astype(int)

    
word_cnt, max_value = mWisard.get_mem_info()
minterms_cnt = mWisard.get_minterms_info()

write2file('\n>>> Selection Results')
write2file('\nNumber of words: '+str(word_cnt))
write2file('\nNumber of minterms: '+str(minterms_cnt))
mWisard.model_conv = copy.deepcopy(mWisard.model)


write2file('\n>>> Evaluating test set...')
Y_test_pred = mWisard.classify(X_test_lst)

acc_test = eval_predictions(Y_test, Y_test_pred, classes, do_plot=True)    
write2file(f'\n>>> Test set accuracy: {acc_test:.01%}')

n_export = 0
mWisard.export2verilog(out_dir+'/', X_test_lst[0:n_export,:], Y_test_pred[0:n_export])

write2file( "\n\n--- Executed in %.02f seconds ---" % (time.time() - start_time))   
os.system("mv ./log.txt "+out_dir+"/log_reload.txt")

del X_test
del Y_test
del X_test_lst
del Y_test_pred
    
