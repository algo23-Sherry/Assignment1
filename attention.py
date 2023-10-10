# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:38:04 2023

@author: Sherry
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


train_num = 1
batch_size = 4
label_type = '3日绝对收益率2分类'
loss_type = 'crossentropy'
#学习率
lr = 0.01
step_size = 5
gamma = 0.03
#迭代次数
epochs = 3

class_num = 2
# folder_name = 'cls_class_ar'
folder_name = ''
label_type = 'class'
train_path = './cls_for_attention/'+folder_name+'/train.csv'
test_path = './cls_for_attention/'+folder_name+'/test.csv'



class SelfAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, max_sequence_length, dropout_rate=0.1):
        super(SelfAttentionClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length

        self.embedding = nn.Linear(input_size, hidden_size)
        self.position_encoding = self.generate_position_encoding(hidden_size, max_sequence_length)
        #初始化后 position enbedding就全部生成好了，每个input使用的pe都是一样的
        self.self_attention = nn.MultiheadAttention(hidden_size, 1, dropout=dropout_rate)

        self.feed_forward = nn.Sequential(#可以复杂化一点
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.output = nn.Linear(hidden_size, num_classes)

    def generate_position_encoding(self, hidden_size, max_sequence_length):
        position_encoding = torch.zeros(max_sequence_length, hidden_size)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding

    def forward(self, input_sequence,mask):
        seq_len = input_sequence.size(0)
        batch_size = input_sequence.size(1)
        input_sequence = input_sequence.view (batch_size,seq_len,self.input_size)
        input_embedding = self.embedding(input_sequence) + self.position_encoding[:seq_len]
        self_attention_output, _ = self.self_attention(input_embedding, input_embedding, input_embedding, key_padding_mask=mask)
        attn_outpuself_attention_outputt = self_attention_output.permute(1, 0, 2)  # 恢复形状为 (batch_size, seq_len, hidden_dim)
        feed_forward_output = self.feed_forward(self_attention_output)
        pooled_output, _ = torch.max(feed_forward_output, dim=0)#待推敲
        logits = self.output(pooled_output)#输出这个就可以了，crossentropy 自带softmax
        probabilities = torch.softmax(logits, dim=1)

        return logits

# 
input_size = 768
hidden_size = 256
num_classes = 2
# sequence_length = 10
max_sequence_length =10
dropout_rate = 0.1

model = SelfAttentionClassifier(input_size, hidden_size, num_classes, max_sequence_length, dropout_rate)


#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.dataset = pd.read_csv(train_path,index_col=0).to_dict('list')
        self.data  = list(map(eval, self.dataset['cls']))
        self.labels = self.dataset['label']
        # self.max_len = max(len(seq) for seq in self.data)
        # self.cls_len =  len(self.data[0][0])
        # self.cls_empty = [0 for i in range(self.cls_len)]
        self.dataset  = 0
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        label = self.labels[idx]
        return seq,label
    
dataset = Dataset(train_path)
test_dataset = Dataset(test_path)
print(type(dataset[0]))
print('trainset length:',len(dataset))
print('testset length:',len(test_dataset))


cls_empty = [0 for i in range(768)]

def collate_fn(batch):
    seqs, labels = zip(*batch)
    # print(len(seqs))
    seqs = [seq.copy() for seq in seqs]  # 创建seq的副本
    sqs_len = [len(seq) for seq in seqs]
    padding_lens = [10-item for item in sqs_len]
    for seq in seqs:
        while len(seq)<max_sequence_length:
            seq.append(cls_empty)  # 添加padding
    labels = list(labels)
    # padding_lens = list(padding_lens)
    # print( torch.tensor(seqs).shape)
    # print(padding_lens)
    return torch.tensor(seqs), torch.tensor(labels), torch.tensor(padding_lens)

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

#数据加载器
# loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
# eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


print('train_loader长度：',len(loader))
print('test_loader长度：',len(eval_loader))
for i, x in enumerate(loader):
    break
for inputs, labels, padding_lens in loader:
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
    #训练
    train_loss = 0.0
    correct = 0
    total = 0
    model.train()
    scheduler.step()
    
    for inputs, labels, padding_lens in loader:
        # break
        mask = torch.arange(inputs.size(1)).unsqueeze(0)  # 创建一个 [1, seq_len] 的mask
        mask = mask >= padding_lens.unsqueeze(1)  # 根据padding长度生成attention mask
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
             mask = mask >= padding_lens.unsqueeze(1)  # 根据padding长度生成attention mask
             mask = mask.to(torch.bool)
             out = model(inputs, mask)
             # print(out)
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
# df_loss.to_csv('./Attention/Attention_loss_{}.csv'.format(train_num))    
    
    
#模型保存到本地
# torch.save(model, './Attention/Attention_{}.model'.format(train_num))
#model_load = torch.load('model/命名实体识别_中文.model')


# 画图
#loss
plt.figure(figsize=(8, 6))
plt.plot(df_loss['train_loss'], marker='.', linestyle='-', color='b', label='Training Loss')
plt.plot(df_loss['test_loss'], marker='.', linestyle='--', color='red', label='eval_loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.savefig('./Attention/Attention_loss_{}.jpg'.format(train_num))
plt.show()

#accuracy
plt.figure(figsize=(8, 6))
plt.plot(df_loss['train_accuracy'], marker='.', linestyle='-', color='b', label='train_accuracy')
plt.plot(df_loss['test_accuracy'], marker='.', linestyle='--', color='red', label='test_accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# plt.savefig('./Attention/Attention_acccuracy_{}.jpg'.format(train_num))
plt.show()



