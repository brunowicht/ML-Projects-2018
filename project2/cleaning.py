# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:04:09 2018

@author: bruno
"""

DATAPATH = "data/"
file = open(DATAPATH+"data_train.csv", encoding="utf-8")
text= file.read()
file.close()

text = text.replace('_', ',')
text = text.replace('r', '')
text = text.replace('c', '')

write_file = open(DATAPATH+"cleaned_data_train.csv", "w")
write_file.write(text[12:])
write_file.close()