# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 01:44:09 2019

@author: yiyuezhuo
"""

'''
4,5,6 
-> 
4_train, 5_train, 6_train, 
4_train_anno,5_train_anno,6_train_anno 
'''

import os
import random
import shutil

random.seed(8964)

config = {
    '4': 60,
    '5': 60,
    '6': 60
}

for root, num_sample in config.items():
    name_list = os.listdir(root)
    root_train = root+'_train'
    os.makedirs(root_train, exist_ok=True)
    os.makedirs(root_train+'_anno',exist_ok=True)
    random.shuffle(name_list)
    name_list_sampled = name_list[:num_sample]
    for name in name_list_sampled:
        ori_path = os.path.join(root, name)
        tar_path = os.path.join(root_train, name)
        shutil.copy(ori_path, tar_path)
        print('Copy {} -> {}'.format(ori_path, tar_path))
        
    
        