#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 05:44:20 2021

@author: igor
"""
import numpy as np
from discriminator import discriminator_train, discriminator_eval

def wisard_find_threshold (X, model, mapping, classes, address_size, min_bleaching=1, max_bleaching=100):
    n_samples = X.shape[0]
    X_mapped = X[:,mapping]
    Y_pred = []
    
    # Eval for each sample
    blc_cnt = 0
    for n in range (n_samples):
        xt = X_mapped[n,:].reshape(-1, address_size)
        xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
        scores = np.zeros((len(classes)))
        # Increase bleaching until the max or there is no tie       
        for bleaching in range(min_bleaching,max_bleaching+1):
            if(bleaching==max_bleaching):
                # print('# WARNING max bleaching reached: '+str(bleaching))
                blc_cnt+=1
            tie = False
            for c in range (len(classes)):
                scores[c] = discriminator_eval(xti.astype(int), model[classes[c]], bleaching)
                # Compare with the previous
                for c2 in range (c):
                    if scores[c]==scores[c2] and max_bleaching>1:
                        tie = True
                        break
                if tie==True and bleaching<max_bleaching:
                    break
            # Stops if there is no more ties
            if tie==False:
                break
        best_class = np.argmax(scores)    
        Y_pred.append(best_class)
    print('%d samples of %d reached the max bleaching' % (blc_cnt,n_samples))
    return np.array(Y_pred)