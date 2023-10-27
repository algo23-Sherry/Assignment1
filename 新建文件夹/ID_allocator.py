# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:22:40 2023

@author: yuanxysx
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

raw_data =  pd.read_csv('Ehudong_sh.csv',index_col=0)


text = pd.read_csv('Ehudong_sh.csv', header=0,index_col=0)
text.dropna(how='any',inplace=True)
text.rename(columns={'code':'company'},inplace=True)
text['code'] = text['company'].apply(lambda x : x[-7:-1]+'.SH')
text['a_date'] = text['a_time'].apply(lambda x : x[:10])
text['q_date'] = text['q_time'].apply(lambda x : x[:10])
text['QA'] =  text['q']  + text['a']

pctchg = pd.read_csv('pctchg.csv',index_col=0)
pctchg.set_index('TDATE',inplace=True)
pctchg = pctchg.stack().to_frame().reset_index(drop=False)
pctchg.rename(columns={'level_1':'code',0:'pctchg'},inplace=True)

df_trade = pd.DataFrame(pctchg['TDATE'].unique(),columns=['TDATE'])

start_date = max([min(text['q_date']),min(text['a_date']),min(df_trade['TDATE'])])
end_date = min([max(text['q_date']),max(text['a_date']),max(df_trade['TDATE'])])

text = text.loc[(text['q_date']>=start_date)&(text['a_date']<=end_date),:].copy()
df_trade = df_trade.loc[(df_trade['TDATE']>=start_date)&(df_trade['TDATE']<=end_date),:].copy()

#节假日问的问题作为下一交易日问的问题
text = pd.merge(text,df_trade,left_on='q_date',right_on='TDATE',how='left')
text = text.sort_values('q_time')
# text_weekenddate_fill = text.copy()
text['q_tdate'] = text.groupby('code')['TDATE'].fillna(method='bfill')#节假日问的问题作为下一交易日问的问题
null_counts = text.isnull().sum()
text.drop(columns=['TDATE'],inplace=True)
#节假日回答的问题作为下一交易日问答的问题
text = pd.merge(text,df_trade,left_on='a_date',right_on='TDATE',how='left')
text = text.sort_values('a_time')
# text_weekenddate_fill = text.copy()
text['a_tdate'] = text.groupby('code')['TDATE'].fillna(method='bfill')#节假日回答的问题作为下一交易日问答的问题
null_counts = text.isnull().sum()
text.drop(columns=['TDATE'],inplace=True)
text.dropna(how='any',inplace=True)


text.reset_index(drop=True,inplace=True)
text['ID'] = text.index

text.drop(columns = ['a_date','q_date'],inplace=True)
text.to_csv('text_ID.csv')


df_Q = text[['q','ID']].copy()[:500]
df_A = text[['a','ID']].copy()[:500]
df_QA = text[['QA','ID']].copy()[:500]

df_Q.rename(columns={'q':'text'},inplace=True)
df_A.rename(columns={'a':'text'},inplace=True)
df_QA.rename(columns={'QA':'text'},inplace=True)



df_Q.to_csv('Q_ID.csv')
df_A.to_csv('A_ID.csv')
df_QA.to_csv('QA_ID.csv')





# # 使用布尔索引过滤含有空值的行，并打印
# rows_with_null = text[text.isnull().any(axis=1)]
# #只有两条空值，drop掉
# text.dropna(how='any',inplace=True)