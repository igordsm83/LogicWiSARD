#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 05:28:50 2022

@author: igor
"""
from functools import reduce
import operator as op
import numpy as np
import math

def hamming_correction_sample (bits, address_size):    
    bits_c = np.copy(bits)
    if (np.sum(bits_c)>0):
        # next_pwr2 = 2**(math.ceil(math.log(address_size, 2)))
        next_pwr2 = 64
        diff_b = next_pwr2-address_size
        bits_c = np.concatenate((np.zeros((diff_b), dtype=int), bits_c))
        correction = reduce(op.xor, [i for i, bit in enumerate(bits_c) if bit])
        # print(correction)   
        bits_c[correction] = 1 - bits_c[correction]
        return bits_c[diff_b:]
    else:
        return bits_c

def hamming_correction(X, address_size):
    n_rams = int(X.shape[1]/address_size)
    X_r = np.copy(X)
    for i in range (X.shape[0]):
        xt = X[i,:].reshape(-1, address_size)
        for j in range (n_rams):
            xt[j,:] = hamming_correction_sample (xt[j,:], address_size)
        X_r[i,:] = xt.reshape(1,-1)
        
    return X_r