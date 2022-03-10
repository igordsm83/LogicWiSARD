#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 06:14:05 2021

@author: igor
"""

import numpy as np


def build_dict_ram(addresses):
    dict_ram = {}
    
    for a in range (len(addresses)):
        # if addresses[a] != 0:
        if addresses[a] in dict_ram:
            dict_ram[addresses[a]] += 1
        else:
            dict_ram[addresses[a]] = 1
    
    return dict_ram
