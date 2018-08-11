"""
从头开始计算会占用大量的计算资源，同时也很浪费时间，所以通过保存的模型和相应的权重参数来
直接进行预测是很必要的
"""
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
from keras.models import load_model
import os
import time


'--------------------------------------------------'
#制作时间计数器
def timer(func):
    def wrapper(*args,**kwargs):
        start_time = time.time()
        res = func(*args,**kwargs)
        end_time = time.time()
        print('执行时间：[%s seconds]' %(end_time-start_time))
        return res
    return wrapper

'--------------------------------------------------'

# 定义训练过程函数
@timer
def train_open(model_path,model_weights_path, df, stock):
    df_test = df.tail(1)

    x_test = df_test.open.values
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    y_pred = model.predict(x_test)
    y_pred = y_pred.flatten()
    new_df = pd.DataFrame()
    new_df['open'] = pd.Series(x_test)
    new_df['open_next_day'] = pd.Series(y_pred)
    new_df.to_csv(r'stock_price_prediction_open\%s_open.csv' % stock)
    print('%s open is done' % stock)

@timer
def train_close(model_path,model_weights_path, df, stock):
    df_test = df.tail(1)

    x_test = df_test.close.values
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    y_pred = model.predict(x_test)
    y_pred = y_pred.flatten()
    new_df = pd.DataFrame()
    new_df['close'] = pd.Series(x_test)
    new_df['close_next_day'] = pd.Series(y_pred)
    new_df.to_csv(r'stock_price_prediction_close\%s_close.csv' % stock)
    print('%s close is done' % stock)


'------------------------------------------'


# 训练模型并保存结果

def generate():
    stock_data_list = glob.glob(r'stock_data\*.csv')
    for stock in stock_data_list:
        stock_num = stock.lstrip(r'stock_data\\').rstrip('.csv')
        stock_df = pd.read_csv(stock)
        open_model_path = os.path.join('stock_model_data','%s_open.dat' %stock_num)
        open_model_weights_path = os.path.join('stock_model_data','%s_open_weights.dat' %stock_num)
        close_model_path = os.path.join('stock_model_data', '%s_close.dat' % stock_num)
        close_model_weights_path = os.path.join('stock_model_data', '%s_close_weights.dat' % stock_num)
        train_open(open_model_path, open_model_weights_path,stock_df, stock_num)
        train_close(close_model_path,close_model_weights_path, stock_df, stock_num)


if __name__ == '__main__':
    generate()
