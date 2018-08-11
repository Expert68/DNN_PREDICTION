# import pandas as pd
# import numpy as np
# # def ai_acc_xed2x(y_true, y_pred, ky0=5, fgDebug=False):
# #     '''
# #     效果评估函数，用于评估机器学习算法函数的效果。
# #     输入：
# #     	y_true,y_pred，pandas的Series数据列格式。
# #     	ky0，结果数据误差k值，默认是5，表示百分之五。
# #     	fgDebug，调试模式变量，默认为False。
# #     返回：
# #         dacc,准确率，float格式
# #         df，结果数据，pandas列表格式DataFrame
# #
# #     '''
# #     # 1
# #     df, dacc = pd.DataFrame(), -1
# #     if (len(y_true) == 0) or (len(y_pred) == 0):
# #         return dacc, df
# #
# #     #
# #     y_num = len(y_true)
# #     df['y_true'], df['y_pred'] = pd.Series(y_true), pd.Series(y_pred)
# #     df['y_diff'] = np.abs(df.y_true - df.y_pred)
# #     # 2
# #     df['y_true2'] = df['y_true']
# #     df.loc[df['y_true'] == 0, 'y_true2'] = 0.00001
# #     df['y_kdif'] = df.y_diff / df.y_true2 * 100
# #     # 3
# #     dfk = df[df.y_kdif < ky0]
# #     knum = len(dfk['y_pred'])
# #     dacc = knum / y_num * 100
# #     #
# #     # 5
# #     dacc = round(dacc, 3)
# #     return dacc, df
#
# def accuracy(y_pred, y_true, devi):
#     """
#     使用最简单的思想求准确率，返回准确率和求准确率时使用的DataFrame对象
#     :param y_pred: 预测值
#     :param y_true: 准确值
#     :param devi: deviation 允许的误差值
#     :return: accuracy,df
#     """
#     df = pd.DataFrame()
#     if len(y_true) == 0 and len(y_pred) == 0:
#         acc, df = -1, df
#         return acc, df
#     df['y_pred'] = pd.Series(y_pred)
#     df['y_true'] = pd.Series(y_true)
#     print(df)
#     # df['diff'] = np.abs(df['y_pred'] - df['y_true'])
#     df['diff'] = np.abs(df.y_true - df.y_pred)
#     # 使用更新语法，将y_true为0的列更新
#     df.loc[df['y_true'] == 0, 'y_true'] = 0.0001
#     df['k_diff'] = df['diff'] / df['y_true'] * 100
#     # 如果误差在允许的误差范围之内，则保存在新的DataFrame dfk中
#     dfk = df[df['k_diff'] < devi]
#     # 计算准确率
#     accurate_num = dfk.y_pred.count()
#     total_num = df.y_pred.count()
#     acc = (accurate_num / total_num) * 100
#     acc = round(acc)
#     return acc, df
#
# df = pd.read_csv(r'D:\single_dense_DNN_prediction\data\lin_reg_predict.csv')
# acc,_ = accuracy(df.y_pred,df.y_test,devi=5)
# print(acc)

import pandas as pd
import numpy as np
import keras
from keras.models import load_model
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

