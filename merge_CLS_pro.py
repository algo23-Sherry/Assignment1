# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:07:01 2023

@author: yuanxysx
"""

"""
拼接CLS
"""
import pandas as pd
import numpy as np
import re
from tqdm import tqdm 

window = 3#使用过去3个交易日的QA数据
label_len = 3 #未来3个交易日的涨跌幅
label_column = 'pctchg_'+str(label_len)

with open('D:\cls\Q_ID_cls.csv', 'r') as f:
    total_lines = sum(1 for line in f)
# num_lines = total_lines // 5
# lines_list = [i for i in range(0,total_lines,num_lines)]

import pickle
# with open('lines_list.pickle', 'wb') as f:
#     pickle.dump(lines_list, f)
with open('lines_list.pickle', 'rb') as f:
    lines_list = pickle.load(f)


# lines_list =list(map(lambda x:x//100, lines_list))

total_lines = 1000

#拼接cls
def get_cls_df(filename):
    list_cls = []
    list_ID = []
    with open(filename, 'r') as f:
        for i in range(total_lines):
            try:
                line = f.readline()
                cls_ = eval(re.findall("\[.*?\]",line)[0])
                ID = line.split(',')[-1]
                list_cls.append(cls_)
                list_ID.append(ID)
            except Exception as e:
                print(e)
    df = pd.DataFrame(data={'cls':list_cls,'ID':list_ID})
    return df



def merge_cls(ID_df,filename):
    df_result = pd.DataFrame()
    cls_df = get_cls_df(filename)
    cls_df['ID'] = cls_df['ID'].apply(int)
    #merge
    df_temp = pd.merge(ID_df,cls_df,how='inner',left_on='ID',right_on='ID')
    df_temp.dropna(how='any',inplace=True)
    return df_temp




#读数据 ID+LABEL


DF_id_label = pd.read_csv('DF_ID_LABEL.csv',index_col=0)
print('样本总数：',len(DF_id_label.groupby(['T0','code']).count()))


DF_id_label.rename(columns = {label_column:'label'},inplace=True)
#分成 Q A
df_q = DF_id_label.loc[DF_id_label['type']=='q',:].copy()
df_a = DF_id_label.loc[DF_id_label['type']=='a',:].copy()

df_q = merge_cls(df_q,'D:\cls\Q_ID_cls.csv')
df_a = merge_cls(df_a,'D:\cls\A_ID_cls.csv')


#合并 sort by time
df_ = pd.concat([df_q,df_a])
df_.sort_values('time',inplace=True)
df_.reset_index(drop=True,inplace=True)
#grouby 合并cls 为序列
DF_result = df_.groupby(['code','T0'])['cls'].apply(lambda x :x.values.tolist()).to_frame()
#d对齐label
DF_result['label'] = df_.groupby(['code','T0'])['label'].apply(lambda x :x.values[0])
del df_
#查看最长的序列长度
# DF_result['len'] = DF_result['cls'].apply(len)
# print(max(DF_result['len']))
DF_result.sort_values('T0',inplace=True)
DF_result.reset_index(drop=False,inplace=True)
DF_result['shape'] = DF_result['cls'].apply(lambda x:np.array(x).shape)

# 输出保存
import os
# 定义要保存的文件路径
file_path = './traindata/window_{}_label_{}/'.format(window,label_len)
# 获取文件所在的目录路径
directory = os.path.dirname(file_path)
# 检查路径是否存在，如果不存在则创建
if not os.path.exists(directory):
    os.makedirs(directory)
    
df = DF_result.iloc[:int(len(DF_result)*0.8),:]
df = df[['cls','label','code','T0']].copy()
df.to_csv(file_path+'train.csv')
df = DF_result.iloc[int(len(DF_result)*0.8):,:]
df = df[['cls','label','code','T0']].copy()
df.to_csv(file_path+'test.csv')















# #保存数据到本地
# DF_train = DF_result.loc[DF_result['T0']<='2022-06-01',:].copy()
# DF_train = DF_train[['cls','label']].copy()
# DF_train.to_csv(file_path+'train.csv',mode='a',header=False)
# del DF_train

# DF_test = DF_result.loc[DF_result['T0']>'2022-06-01',:].copy()
# DF_test = DF_test[['cls','label']].copy()
# DF_test.to_csv(file_path+'test.csv')
# del DF_test
# del DF_result
        



# check = pd.read_csv(file_path+'train.csv',index_col=0)

# check['cls'] = check['cls'].apply(lambda x :x.replace('list(','').replace(')',''))
# check['cls'] = check['cls'].apply(eval) 
# check['cls'].apply(lambda x: True if type(x))

#大显存版本 合并cls
# Q_id = pd.read_csv('D:\cls\Q_ID_cls.csv',index_col = 0)
# def get_cls_df(filename):
#     list_cls = []
#     list_ID = []
#     with open(filename, 'r') as f:
#         for i in range(total_lines):
#             try:
#                 line = f.readline()
#                 cls_ = eval(re.findall("\[.*?\]",line)[0])
#                 ID = line.split(',')[-1]
#                 list_cls.append(cls_)
#                 list_ID.append(ID)
#             except Exception as e:
#                 print(e)
#     cls_df = pd.DataFrame(data={'cls':list_cls,'ID':list_ID})
#     cls_df['ID'] = cls_df['ID'].apply(int)
#     return cls_df

# cls_df =  get_cls_df('D:\cls\Q_ID_cls.csv')
# df_q = pd.merge(df_q,cls_df,how='left',left_on='ID',right_on='ID')
# df_q.drona(how='any',inplace=True)

# cls_df =  get_cls_df('D:\cls\A_ID_cls.csv')
# df_a = pd.merge(df_a,cls_df,how='left',left_on='ID',right_on='ID')
# df_a.drona(how='any',inplace=True)

# del cls_df
