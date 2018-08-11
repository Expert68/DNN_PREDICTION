import tensorflow as tf
import keras
from keras.utils import plot_model
import pandas as pd
import numpy as np
import glob
import time

'------------------------------------------'


# 制作时间计数器
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('执行时间：[%s seconds]' % (end_time - start_time))
        return res

    return wrapper


'------------------------------------------'


# 构建模型
def single_dense_dnn():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1000, name='single_dense_dnn', kernel_initializer='normal', input_dim=1))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1, kernel_initializer='normal'))
    return model


model = single_dense_dnn()
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
model.summary()
# tensorboard_callbacks1 = keras.callbacks.TensorBoard(log_dir='log/log_tmp/open',write_grads=True,write_images=True,write_graph=True)
# tensorboard_callbacks2 = keras.callbacks.TensorBoard(log_dir='log/log_tmp/close',write_grads=True,write_images=True,write_graph=True)

'------------------------------------------'


# 定义训练过程函数
@timer
def train_open(model, df, stock):
    row_count = df['close'].count()
    df_train = df.head(row_count - 1)
    df_test = df.tail(1)

    x_train = df_train.open.values
    y_train = df_train.xopen.values
    x_test = df_test.open.values
    model.fit(x_train, y_train, batch_size=160, epochs=2500, verbose=0)
    model.save(r'stock_model_data\%s_open.dat' % stock, overwrite=True)
    model.save_weights(r'stock_model_data\%s_open_weights.dat' % stock, overwrite=True)
    y_pred = model.predict(x_test)
    y_pred = y_pred.flatten()
    new_df = pd.DataFrame()
    new_df['open'] = pd.Series(x_test)
    new_df['open_next_day'] = pd.Series(y_pred)
    new_df.to_csv(r'stock_price_prediction_open\%s_open.csv' % stock)
    print('%s open is done' % stock)


@timer
def train_close(model, df, stock):
    row_count = df['close'].count()
    df_train = df.head(row_count - 1)
    df_test = df.tail(1)

    x_train = df_train.close.values
    y_train = df_train.xclose.values
    x_test = df_test.close.values
    model.fit(x_train, y_train, batch_size=160, epochs=2500, verbose=0)
    model.save(r'stock_model_data\%s_close.dat' % stock, overwrite=True)
    model.save_weights(r'stock_model_data\%s_close_weights.dat' % stock, overwrite=True)
    y_pred = model.predict(x_test)
    y_pred = y_pred.flatten()
    new_df = pd.DataFrame()
    new_df['close'] = pd.Series(x_test)
    new_df['close_next_day'] = pd.Series(y_pred)
    new_df.to_csv(r'stock_price_prediction_close\%s_close.csv' % stock)
    print('%s close is done' % stock)


'------------------------------------------'


# 训练模型并保存结果
@timer
def generate(model):
    stock_data_list = glob.glob(r'stock_data\*.csv')
    total_num = len(stock_data_list)
    count = 0
    for stock in stock_data_list:
        print('执行第%s/%s只股票完毕' % (count, total_num))
        stock_num = stock.lstrip(r'stock_data\\').rstrip('.csv')
        stock_df = pd.read_csv(stock)
        train_open(model, stock_df, stock_num)
        train_close(model, stock_df, stock_num)
        count += 1


if __name__ == '__main__':
    generate(model)
