import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

'----------------------------------------------'
# 读取open数据,并对open数据进行处理
# 然后读取close数据，对close数据进行处理，最后返回完整的序列
# 在完整的序列中生成p_change_today列和p_change_next_day列,并返回最终序列
open_files = glob.glob(r'stock_price_prediction_open\*.csv')
close_files = glob.glob(r'stock_price_prediction_close\*.csv')
df = pd.DataFrame()


def format_open_file(df, file):
    file_name = file.lstrip(r'stock_price_prediction_open\\').rstrip('_open.csv')
    open_df = pd.read_csv(file)
    df.loc[file_name, 'open'] = open_df.loc[0, 'open']
    df.loc[file_name, 'open_next_day'] = open_df.loc[0, 'open_next_day']


def format_close_file(df, file):
    file_name = file.lstrip(r'stock_price_prediction_close\\').rstrip('_close.csv')
    close_df = pd.read_csv(file)
    df.loc[file_name, 'close'] = close_df.loc[0, 'close']
    df.loc[file_name, 'close_next_day'] = close_df.loc[0, 'close_next_day']


for open_file in open_files:
    format_open_file(df, open_file)

for close_file in close_files:
    format_close_file(df, close_file)

df['p_change_today'] = (df['close'] - df['open']) / df['close'] * 100
df['p_change_next_day'] = (df['close_next_day'] - df['open_next_day']) / df['close'] * 100



'----------------------------------------------'
#对数据列进行升序排序，返回涨幅最大的十只股票
time = datetime.now().strftime('%Y-%m-%d')
df = df.sort_values('p_change_next_day',ascending=False)
df.to_csv('stock_price_final_%s.csv' %time)

