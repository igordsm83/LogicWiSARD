#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 07:29:19 2021

@author: igor
"""
from dict_ram import build_dict_ram

def discriminator_train(X):
    
    discriminator = []
    
    for r in range (X.shape[1]):
        ram_module = build_dict_ram(X[:,r])            
        discriminator.append(ram_module)
    return discriminator

def discriminator_eval(X, class_model, bleaching):
    score = 0
    
    for r in range(len(X)):
        if X[r] in class_model[r]:
            if class_model[r][X[r]]>= bleaching:
                score += 1

    return score