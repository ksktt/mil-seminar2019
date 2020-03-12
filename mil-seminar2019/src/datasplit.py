import torch
import os
import pickle as pkl
from collections import defaultdict

data_dir = '/Users/kusakatakuya/mil-seminar2019/mil-seminar2019/data/cifar-100-python'
out_dir = '/Users/kusakatakuya/mil-seminar2019/mil-seminar2019/data/cifar100'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

train_out_path = os.path.join(out_dir, 'train.pkl')
val_out_path = os.path.join(out_dir, 'val.pkl')
test_out_path = os.path.join(out_dir, 'test.pkl')

train_data_path = os.path.join(data_dir,'train')
test_data_path = os.path.join(data_dir, 'test')


######split train_dic for validation#####
split_rate=0.8

with open(train_data_path, 'rb') as f:
    train_dic = pkl.load(f, encoding='latin1')

filenames = train_dic['filenames']
fine_labels = train_dic['fine_labels']
datas = train_dic['data']


#data_dic: {label: [(fname, label, img_data), ...]}
data_dic = defaultdict(list)
for fname, label, data in zip(filenames, fine_labels, datas):
    data_dic[label].append((fname,label, data))


#get datas from every class
train_datas, val_datas = [], []
for label, tups in data_dic.items():
    split_ind = int(len(tups) * split_rate)
    train_datas += tups[:split_ind]
    val_datas += tups[split_ind:]

zipped_train_datas, zipped_val_datas = [], []
for t in zip(*train_datas):
    zipped_train_datas.append(list(t))

for v in zip(*val_datas):
    zipped_val_datas.append(list(v))


train_dic, val_dic = {},{}
train_dic['filenames'] = zipped_train_datas[0]
train_dic['fine_labels'] = zipped_train_datas[1]
train_dic['data'] = zipped_train_datas[2]
val_dic['filenames'] = zipped_val_datas[0]
val_dic['fine_labels'] = zipped_val_datas[1]
val_dic['data'] = zipped_val_datas[2]


with open(train_out_path, 'wb') as f, open(val_out_path, 'wb') as g:
    pkl.dump(train_dic, f)
    pkl.dump(val_dic, g)


#copy test data
with open(test_data_path, 'rb') as f, open(test_out_path, 'wb') as g:
    test_dic = pkl.load(f, encoding='latin1')
    pkl.dump(test_dic, g)