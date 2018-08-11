# 小插曲，因为计算的时候将open数据和close数据混合在一起了，所以用下面的代码将两个数据分开
import glob
import os
files_path = glob.glob(r'D:\single_dense_DNN_prediction\stock_price_prediction_open\*_close.csv')
dist_file_dir = r'D:\single_dense_DNN_prediction\stock_price_prediction_close'
def move(files):
    for file in files:
        file_name = file.lstrip(r'D:\single_dense_DNN_prediction\stock_price_prediction_open')
        file_path = os.path.join(r'D:\single_dense_DNN_prediction\stock_price_prediction_close',file_name)
        with open(file,'rb') as f1:
            with open(file_path,'wb') as f2:
                for line in f1:
                    f2.write(line)

        os.remove(file)
        print('%s is done' % file_name)

move(files_path)