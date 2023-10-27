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
import os
import time


"""
不筛，训练前再筛，全部算
"""
# label_list = ['pctchg_1', 'pctchg_3','pctchg_5','pctchg_10']
# window = 3
# label_data = pd.read_csv('./id_label/DF_ID_LABEL_{}.csv'.format(window),index_col=0)
# label_data.sort_values('T0',inplace=True)
# list_t0 = label_data['T0'].unique().tolist()
# T0len = 350
# start_T0s = []
# for i in range(0,len(list_t0),T0len):
#     print(i)
#     start_T0s.append(list_t0[i])
# if start_T0s[-1]<len(list_t0):
#     start_T0s.append(len(list_t0))

with open('./cls/Q_ID_cls.csv', 'r') as f:
    total_lines = sum(1 for line in f)
chunk_size = 20000  # 指定每次读取的行数
start_lines = []
for i in range(0,total_lines,chunk_size):
   start_lines.append(i)


# check =pd.read_csv('./cls/Q_ID_cls.csv',index_col=0)

#拼接cls
def get_cls_df(filename,start_line,chunk_size):
    list_cls = []
    list_ID = []
    with open(filename, 'r') as f:
        for _ in range(start_line):
            next(f)
        for i in range(chunk_size):
            try:
                line = f.readline()
                cls_ = eval(re.findall("\[.*?\]",line)[0])#文件首行会报错list index out of range，不用管
                ID = line.split(',')[-1]
                list_cls.append(cls_)
                list_ID.append(ID)
            except Exception as e:
                pass
                # print(e)
    df = pd.DataFrame(data={'cls':list_cls,'ID':list_ID})
    return df



def merge_cls(ID_df,filename):
    df_result = pd.DataFrame()
    for start_line in start_lines:#分组输入 进行拼接
        print('startline:',start_line)
        cls_df = get_cls_df(filename,start_line,chunk_size)
        cls_df['ID'] = cls_df['ID'].apply(int)
        #merge
        df_temp = pd.merge(ID_df,cls_df,how='inner',left_on='ID',right_on='ID')
        df_temp.dropna(how='any',inplace=True)
        df_result = pd.concat([df_result,df_temp])
    return df_temp


# check = DF_id_label.loc[DF_id_label['T0']=='2013-08-13',:].copy()



"""
trainset生成
"""
label_list = ['pctchg_1', 'pctchg_3','pctchg_5','pctchg_10']
window_list  = [2,5,10]
# batch_num = 5
for window in window_list:#使用过去window个交易日的QA数据
    window = 10
    # 定义要保存的文件路径
    file_path = './traindata_/window_{}_label/'.format(window)
    # 获取文件所在的目录路径
    directory = os.path.dirname(file_path)
    # 检查路径是否存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    #读数据 ID+LABEL
    # DF_id_label = pd.read_csv('./id_label_/DF_ID_LABEL_{}.csv'.format(window),index_col=0)#
    # total_len = len(DF_id_label)
    # interval = len(DF_id_label)//500000
    T0_list = ['2013-01-01','2016-01-01','2019-01-01','2021-01-01','2022-01-01','2023-01-01','2024-01-01']
    # check = DF_id_label.groupby(['T0']).count()
    # # for i  in range(len(start_T0s)-1):#分批次计算
    for i in range(len(T0_list)-1):
        # break
        DF_id_label = pd.read_csv('./id_label_/DF_ID_LABEL_{}.csv'.format(window),index_col=0)#
        DF_id_label = DF_id_label.loc[(DF_id_label['T0']>=T0_list[i]) & (DF_id_label['T0']<T0_list[i+1]),:].copy()
        print(len(DF_id_label))
        print('本批次样本总数：',len(DF_id_label.groupby(['T0','code']).count()))
        #分成 Q A
        df_q = DF_id_label.loc[DF_id_label['type']=='q',:].copy()
        df_a = DF_id_label.loc[DF_id_label['type']=='a',:].copy()
        
        df_q = merge_cls(df_q,'./cls/Q_ID_cls.csv')
        df_a = merge_cls(df_a,'./cls/A_ID_cls.csv')
        
        #合并 sort by time
        df_ = pd.concat([df_q,df_a])
        df_.sort_values('time',inplace=True)
        df_.reset_index(drop=True,inplace=True)
        #grouby 合并cls 为序列
        DF_result = df_.groupby(['code','T0'])['cls'].apply(lambda x :x.values.tolist()).to_frame()
        #d对齐label
        for label in label_list: 
            DF_result[label] = df_.groupby(['code','T0'])[label].apply(lambda x :x.values[0])
        del df_
        #查看最长的序列长度
        # DF_result['len'] = DF_result['cls'].apply(len)
        # print(max(DF_result['len']))
        DF_result.sort_values('T0',inplace=True)
        DF_result.reset_index(drop=False,inplace=True)
        if i ==0:
            DF_result.to_csv(file_path+'window_{}_label.csv'.format(window))
        else:  
            DF_result.to_csv(file_path+'window_{}_label.csv'.format(window),mode='a',header=None)
    # DF_result['shape'] = DF_result['cls'].apply(lambda x:np.array(x).shape)
    #分类任务标签
    # for label in label_list: 
        # DF_result[label+'_class'] = DF_result[label].apply(lambda x:1 if x>0 else 0)
    # 输出保存
    # if i==0:
    #     DF_result.to_csv(file_path+'window_{}_label.csv'.format(window))
    # else:
    #     DF_result.to_csv(file_path+'window_{}_label.csv'.format(window), mode='a', header=False)

# df = DF_result.iloc[:int(len(DF_result)*0.8),:]
# # df = df[['cls','label','code','T0']].copy()
# df.to_csv(file_path+'train.csv')
# df = DF_result.iloc[int(len(DF_result)*0.8):,:]
# # df = df[['cls','label','code','T0']].copy()
# df.to_csv(file_path+'test.csv')












