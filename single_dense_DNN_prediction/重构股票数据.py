import pandas as pd
import tushare as ts
from abupy import ABuSymbolPd

'---------------------------------------------------'
#获取股价数据
def get_symbol():
    stock_list = []
    with open(r'D:\single_dense_DNN_prediction\data\stock.txt') as f:
        for stock in f:
            stock = stock[1:7]
            stock_list.append(stock)
    return stock_list

stock_list = get_symbol()

# 对股票数据进行重构
def df_to_df(df):
    new_df = pd.DataFrame()
    new_df['open'] = df.open
    new_df['close'] = df.close
    new_df['xopen'] = df.open.shift(-1)
    new_df['xclose'] = df.close.shift(-1)
    return new_df

# 将重构后的股票数据写入stock_data文件夹中
for stock in stock_list:
    df = ABuSymbolPd.make_kl_df(stock)
    if df is not None:
        new_df = df_to_df(df)
        new_df.to_csv(r'stock_data\%s.csv' %stock)
        print('%s is written' %stock)


'---------------------------------------------------'



