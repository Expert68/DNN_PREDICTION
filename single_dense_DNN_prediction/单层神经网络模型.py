# 设置环境变量
import pandas as pd
import keras
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
import numpy as np

# 设置tensorflow的日志地址，如果日志存在则清空日志
rlog = 'log\log_tmp'
if os.path.exists(rlog):
    tf.gfile.DeleteRecursively(rlog)

EPOCHS = 10
BATCH_SIZE = 128

'----------------------------------------------------------------------'
# 读取数据，将数据保存在dataframe中，ndarray多维数组结构复杂，一般不使用，dataframe数据
# 建立在ndarray的基础上，并增加了很多高级功能，可以像使用SQL数据库一样来使用dataframe数据
# 而且dataframe数据轻松的用.values方法来获取ndarray数据

df = pd.read_csv(r'D:\single_dense_DNN_prediction\data\lin_reg02.csv')

'----------------------------------------------------------------------'
# 获取测试数据和训练数据

x_data = df.x.values
y_data = df.y.values
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=0)


# 构建单层神经网络模型
def single_dense_dnn():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(100, name='single_dense_dnn', kernel_initializer='normal', input_dim=1))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1, kernel_initializer='normal'))
    return model


model = single_dense_dnn()
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
model.summary()
plot_model(model, to_file='three_layers_dnn.png', show_shapes=True, show_layer_names=True)

'----------------------------------------------------------------------'
# 训练神经网络并进行预测,并将训练后的模型放到tensorboard中,最后将计算得到的模型进行保存
# 注意：用tensorboard读取event事件时一定要cd到event所在的文件夹中
# 然后再使用tensorboard --logdir ./ 命令，这样才能确保能够读取到数据
# save函数保存的是模型参数，为了减少模型计算量，所以采用save_weights函数，保存权重数据，方便下次直接使用
tensorboard_callbacks = keras.callbacks.TensorBoard(log_dir=rlog, write_graph=True, write_images=True)
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, callbacks=[tensorboard_callbacks])
model.save('tmp/single_dense_dnn.dat', overwrite=True)
model.save_weights('tmp/single_dense_dnn_weights.dat', overwrite=True)

'----------------------------------------------------------------------'
# 对测试数据进行预测
# model = keras.models.load_model(r'D:\single_dense_DNN_prediction\tmp\single_dense_dnn.dat')
y_pred = model.predict(x_test, verbose=1)
df2 = pd.DataFrame()
df2['x_test'] = x_test
df2['y_test'] = y_test
df2['y_pred'] = y_pred
df2.to_csv(r'D:\single_dense_DNN_prediction\data\lin_reg_predict.csv')

'----------------------------------------------------------------------'


# 定义准确度判别函数，对准确度进行预测

def accuracy(y_pred, y_true, devi):
    """
    使用最简单的思想求准确率，返回准确率和求准确率时使用的DataFrame对象
    :param y_pred: 预测值
    :param y_true: 准确值
    :param devi: deviation 允许的误差值
    :return: accuracy,df
    """
    df = pd.DataFrame()
    if len(y_true) == 0 and len(y_pred) == 0:
        acc, df = -1, df
        return acc, df
    df['y_pred'] = pd.Series(y_pred)
    df['y_true'] = pd.Series(y_true)
    df['diff'] = pd.Series(np.abs(df['y_pred'] - df['y_true']))
    # 使用更新语法，将y_true为0的列更新
    df.loc[df['y_true'] == 0, 'y_true'] = 0.0001
    df['k_diff'] = pd.Series(df['diff'] / df['y_true'] * 100)
    # 如果误差在允许的误差范围之内，则保存在新的DataFrame dfk中
    dfk = df[df['k_diff'] < devi]
    # 计算准确率
    accurate_num = dfk.y_pred.count()
    total_num = df.y_pred.count()
    acc = (accurate_num / total_num) * 100
    acc = round(acc)
    return acc, df


'----------------------------------------------------------------------'
# 计算准确度
acc, _ = accuracy(df2.y_pred.values, df2.y_test.values, devi=5)
print('acc:', acc)
