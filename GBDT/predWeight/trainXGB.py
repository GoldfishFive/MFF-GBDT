# 划分数据集
import pandas as pd
import random
import numpy as np
import pickle
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error,r2_score

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os

# expName = 'xgb_data_20210206-1198-manuals'
expName = 'xgb_data_20210206-1198'
nowTime = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
expPath = os.path.join('GBDT/exps',expName,nowTime)
if not os.path.exists(expPath):
    os.makedirs(expPath)

def logger(log_str):
    with open(expPath + '/log.txt','a',encoding='utf-8') as file:
        file.write(log_str)



# #手工特征
# train_data_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198-train-0.8.csv'
# val_data_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198-test-0.2.csv'
# test_data_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198-test-0.2.csv'

# 归一化手工特征
train_data_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-train-0.8.csv'
val_data_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-test-0.2.csv'
test_data_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-test-0.2.csv'

# 手工特征+自动特征
# train_data_path = 'GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-train.csv'
# val_data_path = 'GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-val.csv'
# test_data_path = 'GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-val.csv'

# 归一化手工特征+自动特征
# train_data_path = 'GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-withnormal-train.csv'
# val_data_path = 'GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-withnormal-val.csv'
# test_data_path = 'GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-withnormal-val.csv'

df_train = pd.read_csv(train_data_path)
df_val = pd.read_csv(val_data_path)
df_test = pd.read_csv(test_data_path)
logger('train_data_path: '+train_data_path+'\nval_data_path: '+val_data_path+'\ntest_data_path: '+ test_data_path+'\n')
print(df_train.head())

# df_train = pd.read_csv('GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-withnormal-train.csv')
# df_val = pd.read_csv('GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-withnormal-val.csv')
# df_test = pd.read_csv('GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-withnormal-val.csv')

# 2D + 3D 特征
print(df_train.columns)
y_train = df_train['weight'] # 获取训练集的y
x_train = df_train.drop(['weight','imgName'],axis=1) # 获得训练集的x  1 按列舍弃
y_test = df_test['weight'] # 获取测试集的y
x_test = df_test.drop(['weight','imgName'],axis=1) # 获取测试集的x
y_val = df_val['weight']
x_val = df_val.drop(['weight','imgName'],axis=1)
# logger("使用全部手工特征\n")

#  2D特征
# x_train = pd.DataFrame(x_train.values[:, 0:15])
# x_test = pd.DataFrame(x_test.values[:, 0:15])
# x_val = pd.DataFrame(x_val.values[:, 0:15])
# print(x_train.shape,x_test.shape)
# logger("只使用2D手工特征\n")

#  3D特征
# x_train = pd.DataFrame(x_train.values[:, 15:24])
# x_test = pd.DataFrame(x_test.values[:, 15:24])
# x_val = pd.DataFrame(x_val.values[:, 15:24])
# print(x_train.shape,x_test.shape)
# logger("只使用3D手工特征\n")

all_manuals = ['area', 'perimeter', 'min_rect_width', 'min_rect_high', 'approx_area', 'approx_perimeter', 'extent', 'hull_perimeter', 'hull_area', 'solidity', 'max_defect_dist', 'sum_defect_dist', 'equi_diameter', 'ellipse_long', 'ellipse_short', 'eccentricity', 'volume', 'maxHeight', 'minHeight', 'max2min', 'meanHeight', 'mean2min', 'mean2max', 'stdHeight', 'heightSum']
new_manuals = ['approx_area', 'approx_perimeter', 'extent', 'hull_perimeter', 'hull_area', 'solidity', 'max_defect_dist', 'sum_defect_dist', 'equi_diameter', 'ellipse_long', 'ellipse_short', 'maxHeight', 'minHeight', 'max2min', 'meanHeight', 'mean2min', 'mean2max', 'stdHeight', 'heightSum']
old_manuals = ['area', 'perimeter', 'min_rect_width', 'min_rect_high', 'eccentricity', 'volume']

# #手工特征
# x_train = pd.DataFrame(x_train.values[:, :25],columns=all_manuals)
# x_test = pd.DataFrame(x_test.values[:, :25],columns=all_manuals)
# x_val = pd.DataFrame(x_val.values[:, :25],columns=all_manuals)
# logger("只使用手工特征\n")

#  自动特征
# x_train = pd.DataFrame(x_train.values[:, 26:2074])
# x_test = pd.DataFrame(x_test.values[:, 26:2074])
# x_val = pd.DataFrame(x_val.values[:, 26:2074])
# logger("只使用自动特征\n")

# print(x_train.columns)
# 新特征 19个
# x_train = x_train.drop(old_manuals,axis=1) # 获得训练集的x  1 按列舍弃
# x_test = x_test.drop(old_manuals,axis=1) # 获取测试集的x
# x_val = x_val.drop(old_manuals,axis=1) # 获得训练集的x  1 按列舍弃
# logger("只使用新的手工特征\n")


# 旧特征6个
# x_train = x_train.drop(new_manuals,axis=1) # 获得训练集的x  1 按列舍弃
# x_test = x_test.drop(new_manuals,axis=1) # 获取测试集的x
# x_val = x_val.drop(new_manuals,axis=1) # 获得训练集的x  1 按列舍弃
# logger("只使用旧的手工特征\n")




# dtrain = xgb.DMatrix(x_train, label=y_train)
# dtest = xgb.DMatrix(x_test, label=y_test)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1729)
# print(dtrain.shape, dtest.shape)

#模型参数设置
xlf = xgb.XGBRegressor( max_depth=5,
                        learning_rate=0.1,
                        n_estimators=2000,
                        silent=True,
                        objective='reg:linear',
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=1440,
                        gpu_id=0,
                        missing=None)


xlf.fit(x_train, y_train, eval_metric='rmse', verbose = True, eval_set = [(x_val, y_val)],early_stopping_rounds=100)

# 计算 auc 分数、预测
preds = xlf.predict(x_train)

t_preds = xlf.predict(x_test)

#训练集上：
train_log_str = '训练：\n' \
                '平均绝对误差: {:.6f}\n' \
                '均方误差mse: {:.6f}\n' \
                '均方根误差rmse: {:.6f}\n' \
                'R2: {:.6f}\n'.format(mean_absolute_error(y_train,preds),mean_squared_error(y_train, preds),
                                      mean_squared_error(y_train, preds) ** 0.5, r2_score(y_train,preds))

print(train_log_str),logger(train_log_str)
#测试集上
val_log_str = '测试：\n' \
                '平均绝对误差: {:.6f}\n' \
                '均方误差mse: {:.6f}\n' \
                '均方根误差rmse: {:.6f}\n' \
                'R2: {:.6f}\n'.format(mean_absolute_error(y_test,t_preds),mean_squared_error(y_test, t_preds),
                                      mean_squared_error(y_test, t_preds) ** 0.5, r2_score(y_test,t_preds))
print(val_log_str),logger(val_log_str)
logger('Feature importances:' + str(list(xlf.feature_importances_)))
pd.DataFrame({'gt':y_test,'pr':t_preds}).to_csv(expPath+'/test_predict.csv',index_label='index')
pd.DataFrame({'gt':y_train,'pr':preds}).to_csv(expPath+'/train_predict.csv',index_label='index')
joblib.dump(xlf, expPath + '/xgb.pkl')

