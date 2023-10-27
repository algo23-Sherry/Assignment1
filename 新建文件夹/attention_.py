# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:23:19 2023

@author: yuanxysx

已经加好了distance_encoding

合并生成cls阶段要生成distance矩阵！！！

新数据的merge_CLS_pro要重写

"""

import torch
import torch.nn as nn
from torch.functional import F
import math 
import pandas as pd
import numpy as np
import time
from torch.optim import lr_scheduler
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


# check = traindata.loc[traindata['code']=='600446.SH',:].copy()

train_num = 5
batch_size = 2
label_type = '3日绝对收益率2分类'
loss_type = 'crossentropy'
#学习率
lr = 0.001
step_size = 5
gamma = 0.03
#迭代次数
epochs = 30


label_type = 'class'
class_num = 2
window = 2
pctchg_window = 1
folder_name = 'window_{}_label'.format(window)
label_name = 'pctchg_{}'.format(pctchg_window)
threshold_up = 0.05
threshold_down = -0.05
train_path = './traindata_/'+folder_name+'/train.csv'
test_path = './traindata_/'+folder_name+'/test.csv'

"""
筛选训练样本
"""
traindata = pd.read_csv('./traindata_/'+folder_name+'/'+folder_name+'.csv',index_col=0)
#查看数据确定范围
# traindata.sort_values('pctchg_3',inplace=True)
# traindata.reset_index(drop=True,inplace=True)
# check = traindata['pctchg_3'].unique()
#不做样本筛选处理
# traindata = traindata.loc[(traindata[label_name]>threshold_up)|(traindata[label_name]<threshold_down),['code','T0','cls',label_name]].copy()
traindata[label_name] = traindata[label_name] .apply(lambda x :1 if x>0 else 0)
traindata.sort_values('T0',inplace=True)
traindata.reset_index(drop=True,inplace=True)

df = traindata.iloc[:int(len(traindata)*0.05),:].copy()

#临时生成一些distance
df['len'] = df['cls'].apply(lambda x :len(eval(x)))
df['distance'] = df['len'].apply(lambda x :[random.randint(0,10) for i in range(x)])


df.to_csv(train_path)
df = traindata.iloc[int(len(traindata)*0.8):,:]
df.to_csv(test_path)





class SelfAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, max_sequence_length, dropout_rate=0.1):
        super(SelfAttentionClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length

        self.embedding = nn.Linear(input_size, hidden_size)
        self.position_encoding = self.generate_position_encoding(hidden_size, max_sequence_length)
    
        self.self_attention = nn.MultiheadAttention(hidden_size, 1, dropout=dropout_rate)

        self.feed_forward = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 34),
            
        )

        self.output = nn.Linear(34, num_classes)

    def generate_position_encoding(self, hidden_size, max_sequence_length):
        position_encoding = torch.zeros(max_sequence_length, hidden_size)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding

    def forward(self, input_sequence,mask):
        batch_size = input_sequence.size(0)
        seq_len = input_sequence.size(1)
        input_sequence = input_sequence.view (seq_len,batch_size,self.input_size)
        input_embedding = self.embedding(input_sequence) +torch.stack([self.position_encoding[:seq_len]] * batch_size).permute(1, 0, 2) 
        self_attention_output, _ = self.self_attention(input_embedding, input_embedding, input_embedding, key_padding_mask=mask)
        attn_outpuself_attention_outputt = self_attention_output.permute(1, 0, 2)  # 恢复形状为 (batch_size, seq_len, hidden_dim)
        feed_forward_output = self.feed_forward(attn_outpuself_attention_outputt)
        pooled_output, _ = torch.max(feed_forward_output, dim=1)#沿着序列长度进行池化操作，即在每个样本中选择序列特征中的最大值作为代表
        logits = self.output(pooled_output)
        probabilities = torch.softmax(logits, dim=1)
        return logits

# 
input_size = 768
hidden_size = 256
num_classes = 2
data_list = list(map(eval,pd.read_csv(train_path,index_col=0).to_dict('list')['cls']))
len_list = [len(data) for data in data_list]
max_sequence_length =max(len_list)
del data_list
del len_list
dropout_rate = 0.1

model = SelfAttentionClassifier(input_size, hidden_size, num_classes, max_sequence_length, dropout_rate)


#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.dataset = pd.read_csv(train_path,index_col=0).to_dict('list')
        self.data  = list(map(eval, self.dataset['cls']))
        self.labels = self.dataset[label_name]
        self.distances = list(map(eval, self.dataset['distance']))
        # self.max_len = max(len(seq) for seq in self.data)
        # self.cls_len =  len(self.data[0][0])
        # self.cls_empty = [0 for i in range(self.cls_len)]
        self.dataset  = 0
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        label = self.labels[idx]
        distance = self.distances[idx]
        return seq,label,distance
    
dataset = Dataset(train_path)
test_dataset = Dataset(test_path)
print(type(dataset[0]))
print('trainset length:',len(dataset))
print('testset length:',len(test_dataset))


cls_empty = [0 for i in range(768)]
div_term = torch.exp(torch.arange(0, input_size, 2).float() * (-math.log(10000.0) / input_size))
div_term = torch.repeat_interleave(div_term, repeats=2)
position_ori = torch.arange(1, max_sequence_length+1, dtype=torch.float).unsqueeze(1)
position_ori = position_ori/position_ori
position_ori = position_ori.repeat(batch_size, 1).view(batch_size,max_sequence_length,1)


def collate_fn(batch):
    seqs, labels, distances = zip(*batch)
    # print(len(seqs))
    seqs = [seq.copy() for seq in seqs]  # 创建seq的副本
    sqs_len = [len(seq) for seq in seqs]
    padding_lens = [max_sequence_length-item for item in sqs_len]
   
    for i in range(batch_size):
        while len(seqs[i])<max_sequence_length:
            seqs[i].append(cls_empty)  # 添加padding
            distances[i].append(0)
            
    labels = torch.tensor(list(labels))
    padding_lens = torch.tensor(padding_lens)
    distances = [list(map(lambda x: [x],distance)) for distance in distances]
    distances = torch.tensor(distances)
    seqs = torch.tensor(seqs)
    position_dis = position_ori*distances
    distance_encoding = position_dis* div_term
    distance_encoding[:, :,0::2] = torch.sin(distance_encoding[:, :,0::2])
    distance_encoding[:, :,1::2] = torch.cos(distance_encoding[:, :,1::2])

    seqs = seqs + distance_encoding #在原CLS中加入distance_encoding
    return seqs, labels, padding_lens

#数据加载器
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

print('train_loader长度：',len(loader))
print('test_loader长度：',len(eval_loader))
for i, x in enumerate(loader):
    break

optimizer = Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)

#模型训练
#训练监测矩阵
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []


start_time = time.time()
for i in tqdm(range(epochs)):
    break
    #训练
    train_loss = 0.0
    correct = 0
    total = 0
    model.train()
    scheduler.step()
    
    for inputs, labels, padding_lens in loader:
        break
        mask = torch.arange(inputs.size(1)).unsqueeze(0)
        mask = mask >= max_sequence_length-padding_lens.unsqueeze(1)  #根据padding长度生成attention mask
        mask = mask.to(torch.bool)
        out = model(inputs, mask)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_train_loss = train_loss / len(loader) /batch_size
    train_accuracy = correct / total
    #测试       
    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():        
         for inputs, labels, padding_lens in eval_loader:
             mask = torch.arange(inputs.size(1)).unsqueeze(0)  # 创建一个 [1, seq_len] 的mask
             mask = mask >= max_sequence_length-padding_lens.unsqueeze(1)  # 根据padding长度生成attention mask
             mask = mask.to(torch.bool)
             out = model(inputs, mask)
             loss = criterion(out, labels)
             eval_loss += loss.item()
             _, predicted = torch.max(out.data, 1)
             total += labels.size(0)
             correct += (predicted == labels).sum().item()
    avg_test_loss = eval_loss / len(eval_loader) /batch_size
    test_accuracy = correct / total
    print("test loss:", avg_test_loss)    
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    
end_time = time.time()
print('运行时间：{}'.format(end_time-start_time))
    
df_loss = pd.DataFrame()
df_loss['train_loss'] = train_losses
df_loss['test_loss'] = test_losses
df_loss['train_accuracy'] = train_accuracy
df_loss['test_accuracy'] = test_accuracy
df_loss['epoch'] = df_loss.index
df_loss.to_csv('./attention/Attention_loss_{}.csv'.format(train_num))    
    
    
#模型保存到本地
# torch.save(model, './attention/Attention_{}.model'.format(train_num))
#model_load = torch.load('model/命名实体识别_中文.model')


# 画图
# import matplotlib.pyplot as plt
# import pandas as pd
# train_num = 1
# # df_loss = pd.read_csv('./attention/Attention_loss_{}.csv'.format(train_num),index_col=0)

# #loss
# plt.figure(figsize=(8, 6))
# plt.plot(df_loss['train_loss'], marker='.', linestyle='-', color='b', label='Training Loss')
# plt.plot(df_loss['test_loss'], marker='.', linestyle='--', color='red', label='eval_loss')
# plt.title('Loss Over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# # plt.savefig('./attention/Attention_loss_{}.jpg'.format(train_num))
# plt.show()

# #accuracy
# plt.figure(figsize=(8, 6))
# plt.plot(df_loss['train_accuracy'], marker='.', linestyle='-', color='b', label='train_accuracy')
# plt.plot(df_loss['test_accuracy'], marker='.', linestyle='--', color='red', label='test_accuracy')
# plt.title('Accuracy Over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# # plt.savefig('./attention/Attention_acccuracy_{}.jpg'.format(train_num))
# plt.show()
