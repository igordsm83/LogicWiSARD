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

seed = 420

addressSize = 14
thermo_resolution = 1
min_bleaching = 1
max_bleaching = 50

n_epochs = 1

# These are thresholds to stop the model search early
# If n_epochs models are trained and the criteria are not met, the smallest
# model with accuracy above min_test_acc is selected.
min_test_acc = 0.95
max_minterms_cnt = 188000

do_bleaching = 0

np.random.seed(seed)

project_name = 'mnist'

out_dir = './out/'

try:
    os.remove("./log.txt")
except OSError:
    pass

start_time = time.time()
full_log = "--- RUNNING WISARD TRAINING "+project_name+" ---\n"
full_log += "\n> seed = " + str(seed) 
full_log += "\n> addressSize = " + str(addressSize) 
full_log += "\n> thermo_resolution = " + str(thermo_resolution) 
full_log += "\n> min_test_acc = " + str(min_test_acc) 
full_log += "\n> max_minterms_cnt = " + str(max_minterms_cnt) 
full_log += "\n> n_epochs = " + str(n_epochs) 
datetime_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H-%M-%S')
full_log += "\nStarting at: "+datetime_string+"\n"
write2file( full_log)


# Import 60000 images from mnist data set
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Shuffle train set
n_shuffled = np.arange(X_train.shape[0])
np.random.shuffle(n_shuffled)
X_train = X_train[n_shuffled,:,:]
Y_train = Y_train[n_shuffled]

# Split train set into train and validation sets
X_val = X_train[55000:60000,:,:]
Y_val = Y_train[55000:60000]
X_train = X_train[0:55000,:,:]
Y_train = Y_train[0:55000]


classes = ['0','1','2','3','4','5','6','7','8','9']
print("Train set input: "+str(X_train.shape))
print("Train set output: "+str(Y_train.shape))
print("Val set input: "+str(X_val.shape))
print("Val set output: "+str(Y_val.shape))
print("Test set input: "+str(X_test.shape))
print("Test set output: "+str(Y_test.shape))


print('>>> Encoding train set...')
# X_train_lst = mnist_data_encode_b(X_train)
X_train_lst = mnist_data_encode_t(X_train, 0,255,thermo_resolution)
# X_train_lst, x_mean, x_std = mnist_data_encode_z(X_train, [], [])
# X_train_lst = wisard_data_encode(X_train, classes, resolution=thermo_resolution, minimum=0, maximum=255)
X_train_lst = X_train_lst.astype(int)
Y_train = Y_train.astype(int)

print('>>> Encoding val set...')
# X_val_lst = mnist_data_encode_b(X_val)
X_val_lst = mnist_data_encode_t(X_val, 0,255,thermo_resolution)
# X_val_lst, _, _ = mnist_data_encode_z(X_val, x_mean, x_std)
# X_val_lst = wisard_data_encode(X_val, classes, resolution=thermo_resolution, minimum=0, maximum=255)
X_val_lst = X_val_lst.astype(int)
Y_val = Y_val.astype(int)

print('>>> Encoding test set...')
# X_test_lst = mnist_data_encode_b(X_test)
X_test_lst = mnist_data_encode_t(X_test, 0,255,thermo_resolution)
# X_test_lst, _, _ = mnist_data_encode_z(X_test, x_mean, x_std)
# X_test_lst = wisard_data_encode(X_test, classes, resolution=thermo_resolution, minimum=0, maximum=255)
X_test_lst = X_test_lst.astype(int)
Y_test = Y_test.astype(int)

bin_acc = []
bin_acc_test = []
bin_mem = []
bin_minterms = []
bin_models = []

bleach_acc = []
bleach_mem = []

for epoch in range(n_epochs):
    print('>>>> EPOCH %d <<<<<' %(epoch))
    print('>>> Training Wisard...')
    mWisard = lwsd.logicwisard(classes, addressSize,min_bleaching, max_bleaching)
    mWisard.fit(X_train_lst, Y_train)
    
    if do_bleaching==1:
        ###### Non-binarized evaluation ######
        word_cnt, max_value = mWisard.get_mem_info()
        Y_test_pred = mWisard.classify(X_test_lst)    
        acc_test = eval_predictions(Y_test, Y_test_pred, classes, do_plot=False)   
        print('> Acc with bleaching: '+str(acc_test))
        bleach_acc.append(acc_test)
        bleach_mem.append(word_cnt)

    
    ###### Binarization ######
    print('>>> Finding the best threshold...')
    best_acc, best_threshold, best_cnt, max_acc, max_threshold, max_cnt = mWisard.find_threshold(X_val_lst, Y_val)
    print('Max results => acc: %f / threshold: %d / word_cnt: %d' % (max_acc, max_threshold, max_cnt))
    print('Best results => acc: %f / threshold: %d / word_cnt: %d' % (best_acc, best_threshold, best_cnt))
    
    mWisard.binarize_model()
    
    
    Y_test_pred = mWisard.classify(X_test_lst)
    acc_test = eval_predictions(Y_test, Y_test_pred, classes, do_plot=False)    
    
    word_cnt, max_value = mWisard.get_mem_info()
    minterms_cnt = mWisard.get_minterms_info()
    print('Test ACC: %f / Number of words: %d / Number of minterms: %d' % (acc_test, word_cnt, minterms_cnt))
    
    bin_acc.append(max_acc)
    bin_acc_test.append(acc_test)
    bin_mem.append(word_cnt)
    bin_minterms.append(minterms_cnt)
    bin_models.append(mWisard)


    if acc_test >= min_test_acc and minterms_cnt<=max_minterms_cnt:
        write2file(">>> Early stop. Criteria met at epoch: %d" %(epoch))
        break

del X_train
del Y_train
del X_train_lst
del X_val
del Y_val
del X_val_lst


# Plot search Results
plt.plot(bin_minterms, bin_acc_test,'g^')
plt.xlabel('Number of minterms')
plt.ylabel('Test set accuracy')
plt.savefig(out_dir+'/model_search_minterms.pdf')
plt.show()

if do_bleaching==1:
    plt.plot(np.array(bin_mem)*addressSize/1000, bin_acc_test,'g^', label="Binarized")
    plt.plot(np.array(bleach_mem)*(addressSize+8)/1000, bleach_acc,'ro', label='Bleaching')
    # plt.plot([280*1024*8/1000], [0.907],'bs', label="Hash tables") # Hash filter
    # plt.plot([819.049*1024*8/1000], [0.915],'ys', label="Bloom filters") # Bloom filter
    plt.xlabel('Required memory (Kbits)')
    plt.ylabel('Test set accuracy')
    plt.legend(loc="upper center")
    plt.savefig(out_dir+'/model_search.pdf')
    plt.show()


sorted_i = np.argsort(bin_acc_test)
sel_ind = -1
min_bin = 100000000
st_sorted = 0

for s in range(len(sorted_i)):
    if bin_acc_test[sorted_i[s]]>=min_test_acc:
        st_sorted = s
        break
    
for s in range(st_sorted,len(sorted_i)):
# for s in range(int(len(sorted_i)*0.8),len(sorted_i)):
    i = sorted_i[s]
    if bin_minterms[i]<min_bin:
        min_bin = bin_minterms[i]
        sel_ind = i


mWisard = copy.deepcopy(bin_models[sel_ind])
word_cnt = bin_mem[sel_ind]
minterms_cnt = bin_minterms[sel_ind]

write2file('\n>>> Selection Results')
write2file('\nNumber of words: '+str(word_cnt))
write2file('\nNumber of minterms: '+str(minterms_cnt))
mWisard.model_conv = copy.deepcopy(mWisard.model)



write2file('\n>>> Evaluating test set...')
Y_test_pred = mWisard.classify(X_test_lst)

acc_test = eval_predictions(Y_test, Y_test_pred, classes, do_plot=True)    
write2file(f'\n>>> Test set accuracy: {acc_test:.01%}')

n_export = 10000
mWisard.export2verilog(out_dir+'/', X_test_lst[0:n_export,:], Y_test_pred[0:n_export])

with open(out_dir+'/mWisard.pkl', 'wb') as outp:
    pickle.dump(mWisard, outp, pickle.HIGHEST_PROTOCOL)


del X_test
del Y_test
del X_test_lst
del Y_test_pred
    
write2file( "\n\n--- Executed in %.02f seconds ---" % (time.time() - start_time))    
os.system("mv ./log.txt "+out_dir)

