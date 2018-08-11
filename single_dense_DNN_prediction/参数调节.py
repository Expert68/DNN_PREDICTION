import keras
import arrow
import pandas as pd
#获取保存模型的路径
model_path = r'D:\single_dense_DNN_prediction\tmp\single_dense_dnn.dat'
data_path = r'D:\single_dense_DNN_prediction\data\lin_reg_predict.csv'


# def ai_mx_tst_epochs(f_mx, f_tg, df_train, df_test, kepochs=100, nsize=128, ky0=5):
#     ds, df = {}, pd.DataFrame()
#     for xc in range(1, 11):
#         print('\n#', xc)
#         dnum = xc * kepochs
#         mx = keras.models.load_model(f_mx)
#         t0 = arrow.now()
#         dacc = ai_mul_var_tst(mx, df_train, df_test, dnum, nsize, ky0=ky0)
#         tn = zt.timNSec('', t0)
#         ds['nepoch'], ds['epoch_acc'], ds['ntim'] = dnum, dacc, tn
#         df = df.append(ds, ignore_index=True)
#
#         #
#     df = df.dropna()
#     df['nepoch'] = df['nepoch'].astype(int)
#     print('\ndf')
#     print(df)
#     print('\nf,', f_tg)
#     df.to_csv(f_tg, index=False)
#     #
#     df.plot(kind='bar', x='nepoch', y='epoch_acc', rot=0)
#     df.plot(kind='bar', x='nepoch', y='ntim', rot=0)
#     #
#     return df
#
#
# def ai_mx_tst_bsize(f_mx, f_tg, df_train, df_test, nepochs=500, ksize=32, ky0=5):
#     ds, df = {}, pd.DataFrame()
#     for xc in range(1, 11):
#         print('\n#', xc)
#         dnum = xc * ksize
#         mx = ks.models.load_model(f_mx)
#         t0 = arrow.now()
#         dacc = ai_mul_var_tst(mx, df_train, df_test, nepochs, dnum, ky0=ky0)
#         tn = zt.timNSec('', t0)
#         ds['bsize'], ds['size_acc'], ds['ntim'] = dnum, dacc, tn
#         df = df.append(ds, ignore_index=True)
#
#         #
#     df = df.dropna()
#     df['bsize'] = df['bsize'].astype(int)
#     print('\ndf')
#     print(df)
#     print('\nf,', f_tg)
#     df.to_csv(f_tg, index=False)
#     #
#     df.plot(kind='bar', x='bsize', y='size_acc', rot=0)
#     df.plot(kind='bar', x='bsize', y='ntim', rot=0)
#     return df
#
def ai_mul_var_tst(mx,df_train,df_test,nepochs=200,nsize=128,ky0=5):
    x_train,y_train=df_train['x'].values,df_train['y'].values
    x_test, y_test = df_test['x'].values,df_test['y'].values
    #
    mx.fit(x_train, y_train, epochs=nepochs, batch_size=nsize)
    #
    y_pred = mx.predict(x_test)
    df_test['y_pred']=zdat.ds4x(y_pred,df_test.index,True)
    dacc,_=ai_acc_xed2x(df_test.y,df_test['y_pred'],ky0,False)
    #
    return dacc
