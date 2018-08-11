from keras.models import Sequential
from keras.utils import plot_model
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import keras
from keras.models import load_model
from sklearn.model_selection import train_test_split



'----------------------------------------------------------'
#构建单层神经元模型,并通过summary函数获取神经元模型的信息
#注意如果用tf.keras.models.Sequential的方法导入Sequential类就会报错
#类似的layers也需要直接通过keras导入而不是通过tf.keras来导入
def single_dense_dnn():
    model = Sequential()
    model.add(keras.layers.Dense(1,input_dim=1))
    return model

model = single_dense_dnn()
model.summary()
'----------------------------------------------------------'
#绘制神经元模型，并保存到tmp文件夹

plot_model(model,to_file=r'tmp/single_dense_dnn.png')
plot_model(model,to_file=r'tmp/single_dense_dnn_1.png',show_shapes=True,show_layer_names=True)

'----------------------------------------------------------'


df = pd.read_csv(r'D:\single_dense_DNN_prediction\data\lin_reg02.csv')

'----------------------------------------------------------------------'
# 获取测试数据和训练数据

x_data = df.x.values
y_data = df.y.values
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=0)

model = load_model(r'D:\single_dense_DNN_prediction\tmp\single_dense_dnn.dat')
model.load_weights(r'D:\single_dense_DNN_prediction\tmp\single_dense_dnn_weights.dat')

y_pred = model.predict(x_test, verbose=1)
df2 = pd.DataFrame()
df2['x_test'] = x_test
df2['y_test'] = y_test
df2['y_pred'] = y_pred
print(df2.head())