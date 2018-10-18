# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:38:51 2018

@author: bruno
"""

from utils.cleaning import load_clean_standardize


# Load clean and standardize the data
data_x, data_y, ids = load_clean_standardize("../../MLData2018/train.csv")
