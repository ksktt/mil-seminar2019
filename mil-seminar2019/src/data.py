import torch
import os
import numpy as np
import pickle as pkl
from collections import defaultdict


class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, split='train',split_rate=0.8, data_dir='/Users/kusakatakuya/mil-seminar2019/mil-seminar2019/data/cifar100'):
        super(CifarDataset, self).__init__()
        data_path = os.path.join(data_dir, split+'.pkl')
        with open(data_path, 'rb') as f:
            data_dic = pkl.load(f, encoding='latin1')
        
        self.filenames = data_dic['filenames']
        self.fine_labels = data_dic['fine_labels']

        datas = []
        for data in data_dic['data']:
            data = data.reshape(3, 32, 32)
            data = data.swapaxes(0,2)
            data = data.swapaxes(0,1)
            datas.append(transform(data))
        self.datas = datas

    #Dataset class needs __getitem__ and  __len__   
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = self.fine_labels[idx]
        #img: Tensor 3 x 32 x 32
        img = self.datas[idx]
        return fname, label, img
  
  
    def __len__(self):
        return len(self.filenames)
