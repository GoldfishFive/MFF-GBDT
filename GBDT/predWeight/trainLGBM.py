import pandas as pd
import random
import numpy as np
import pickle
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os

expName = 'lgbm_data_20210206-1198'
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

## 归一化手工特征
# train_data_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-train-0.8.csv'
# val_data_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-test-0.2.csv'
# test_data_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-test-0.2.csv'

# 手工特征+自动特征
# train_data_path = 'GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-train.csv'
# val_data_path = 'GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-val.csv'
# test_data_path = 'GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-val.csv'

# 归一化手工特征+自动特征
train_data_path = 'GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-withnormal-train.csv'
val_data_path = 'GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-withnormal-val.csv'
test_data_path = 'GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-withnormal-val.csv'

df_train = pd.read_csv(train_data_path)
df_val = pd.read_csv(val_data_path)
df_test = pd.read_csv(test_data_path)
logger('train_data_path: '+train_data_path+'\nval_data_path: '+val_data_path+'\ntest_data_path: '+ test_data_path+'\n')
print(df_train.head())


# print(df_train.info())
# print(df_train.head())
print(df_train.columns)

# -----------
# 切分数据集
# -----------
# 2D + 3D 特征
# print(df_train.columns)
y_train = df_train['weight'] # 获取训练集的y
x_train = df_train.drop(['weight','imgName'],axis=1) # 获得训练集的x  1 按列舍弃
# x_train = df_train.drop(['weight','imgName','equi_diameter'],axis=1) # 获得训练集的x  1 按列舍弃
y_test = df_test['weight'] # 获取测试集的y
x_test = df_test.drop(['weight','imgName'],axis=1) # 获取测试集的x
# x_test = df_test.drop(['weight','imgName','equi_diameter'],axis=1) # 获取测试集的x
y_val = df_val['weight']
x_val = df_val.drop(['weight','imgName'],axis=1) # 获得训练集的x  1 按列舍弃
# x_val = df_val.drop(['weight','imgName','equi_diameter'],axis=1) # 获得训练集的x  1 按列舍弃
logger("使用全部特征\n")



# 2D特征
# x_train = pd.DataFrame(x_train.values[:, 0:16]) # 舍弃'equi_diameter'的话是 15
# x_test = pd.DataFrame(x_test.values[:, 0:16])
# x_val = pd.DataFrame(x_val.values[:, 0:16])
# x_train = pd.concat([x_train,train_life] ,axis=1)
# x_test = pd.concat([x_test,test_life],axis=1)
# x_val = pd.concat([x_val,val_life],axis=1)
# logger("只使用2D特征\n")

print(x_train.shape,x_test.shape)

#  3D特征
# x_train = pd.DataFrame(x_train.values[:, 16:24])
# x_test = pd.DataFrame(x_test.values[:, 16:24])
# x_val = pd.DataFrame(x_val.values[:, 16:24])
# print(x_train.shape,x_test.shape)
# logger("只使用3D特征\n")

all_manuals = ['area', 'perimeter', 'min_rect_width', 'min_rect_high', 'approx_area', 'approx_perimeter', 'extent', 'hull_perimeter', 'hull_area', 'solidity', 'max_defect_dist', 'sum_defect_dist', 'equi_diameter', 'ellipse_long', 'ellipse_short', 'eccentricity', 'volume', 'maxHeight', 'minHeight', 'max2min', 'meanHeight', 'mean2min', 'mean2max', 'stdHeight', 'heightSum']
new_manuals = ['approx_area', 'approx_perimeter', 'extent', 'hull_perimeter', 'hull_area', 'solidity', 'max_defect_dist', 'sum_defect_dist', 'equi_diameter', 'ellipse_long', 'ellipse_short', 'maxHeight', 'minHeight', 'max2min', 'meanHeight', 'mean2min', 'mean2max', 'stdHeight', 'heightSum']
old_manuals = ['area', 'perimeter', 'min_rect_width', 'min_rect_high', 'eccentricity', 'volume']

# #手工特征
# x_train = pd.DataFrame(x_train.values[:, :25],columns=all_manuals)
# x_test = pd.DataFrame(x_test.values[:, :25],columns=all_manuals)
# x_val = pd.DataFrame(x_val.values[:, :25],columns=all_manuals)
# logger("只使用手工特征\n")

#  自动特征
# x_train = pd.DataFrame(x_train.values[:, 25:2074])
# x_test = pd.DataFrame(x_test.values[:, 25:2074])
# x_val = pd.DataFrame(x_val.values[:, 25:2074])
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


# ------------------------------
# 给lightgbm 构造数据集
# ------------------------------
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

# 定义一个统计函数方便了解数据的分布
def show_stats(data):
    """data是输入进来的数据，然后求出下面的特征"""
    print("最小值：{}".format(np.min(data)))
    print("最大值：{}".format(np.max(data)))
    print("极值：{}".format(np.ptp(data)))
    print("均值：{}".format(np.mean(data)))
    print("标准差：{}".format(np.std(data)))
    print("方差：{}\n".format(np.var(data)))

#先写一个函数 输入model 输出网格搜索好的model、还有MAE
def model_train(model):
    #设置超参数的范围
    ## param_grid={'learning_rate':[0.01,0.05,0.1,0.2]}
    ##model=GridSearchCV(model,param_grid)
    #模型开始训练
    model.fit(x_train,y_train,eval_set=[(x_val, y_val)], eval_metric='l1', early_stopping_rounds=500)
    # model.fit(x_train,y_train)

    test_predict=model.predict(x_test)#预测的值这里不用返回了吧
    show_stats(y_train)#这里显示以预测的效果
    #model=model.best_estimator_#这里把网格搜索的最佳参数拿到
    train_predict=model.predict(x_train)#预测的值这里不用返回了吧
    #训练集上：
    train_log_str = '训练：\n' \
                    '平均绝对误差: {:.6f}\n' \
                    '均方误差mse: {:.6f}\n' \
                    '均方根误差rmse: {:.6f}\n' \
                    'R2: {:.6f}\n'.format(mean_absolute_error(y_train,train_predict),mean_squared_error(y_train, train_predict),
                                          mean_squared_error(y_train, train_predict) ** 0.5, r2_score(y_train,train_predict))

    print(train_log_str),logger(train_log_str)
    #测试集上
    val_log_str = '测试：\n' \
                    '平均绝对误差: {:.6f}\n' \
                    '均方误差mse: {:.6f}\n' \
                    '均方根误差rmse: {:.6f}\n' \
                    'R2: {:.6f}\n\n'.format(mean_absolute_error(y_test,test_predict),mean_squared_error(y_test, test_predict),
                                          mean_squared_error(y_test, test_predict) ** 0.5, r2_score(y_test,test_predict))
    print(val_log_str),logger(val_log_str)
    feature_log = 'Feature importances:' + str(list(model.feature_importances_))

    print(feature_log),logger(feature_log)
    pd.DataFrame({'gt':y_test,'pr':test_predict}).to_csv(expPath+'/test_predict.csv',index_label='index')
    pd.DataFrame({'gt':y_train,'pr':train_predict}).to_csv(expPath+'/train_predict.csv',index_label='index')
    return model


model_lgb=lgb.LGBMRegressor(n_estimators=4000,learning_rate=0.1,num_leaves=15,
                    max_depth=5,min_child_samples=15,min_child_weight=0.01,
                    subsample=0.8,colsample_bytree=1
                            )
model_lgb=model_train(model_lgb)
print(model_lgb.get_params())


lgb.plot_importance(model_lgb)
plt.savefig(expPath+'/importance.png')
plt.show()



# 模型存储
joblib.dump(model_lgb, expPath+'/result.pkl')

# # 模型加载
# gbm = joblib.load('model_lgb2.pkl')
# # 模型预测
# y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)
# # 模型评估
# print('测试集上的平均绝对误差:',mean_absolute_error(y_test,y_pred))
# print('均方根误差rmse:', mean_squared_error(y_test, y_pred) ** 0.5)
# print('均方误差mse:', mean_squared_error(y_test, y_pred))
# # 特征重要度
# print('Feature importances:', list(gbm.feature_importances_))
