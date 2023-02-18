import lightgbm as lgb
import pandas as pd
import random
import numpy as np
import pickle

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



# df_test = pd.read_csv('GBDT/csvData/20210206-200-1198-manuals/20210206-1198-test-0.2.csv')#手工特征
df_test = pd.read_csv('GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-test-0.2.csv')# 归一化手工特征


# y_train = df_train['weight'] # 获取训练集的y
# x_train = df_train.drop(['weight','imgName','equi_diameter'],axis=1) # 获得训练集的x  1 按列舍弃
x_test = df_test.drop(['weight','imgName'],axis=1) # 获取测试集的x
y_test = df_test['weight'] # 获取测试集的y
# print(x_test['ellipse_long'])

# 2D特征
# x_test = pd.DataFrame(x_test.values[:, 0:15])

#  3D特征
# x_test = pd.DataFrame(x_test.values[:, 15:24])


columns  = df_test.drop(['weight','imgName'],axis=1).columns.values
print(columns,len(columns))

# load model with pickle to predict
# model_weight = 'GBDT/exps/lgbm_data_20210206-1198/2021-12-16_21-11/result.pkl' # no normal
# model_weight = 'GBDT/exps/lgbm_data_20210206-1198/2021-12-16_21-12/result.pkl' # with normal

# model_weight = 'GBDT/exps/xgb_data_20210206-1198/2021-12-16_21-18/xgb.pkl' # no normal
model_weight = 'GBDT/exps/xgb_data_20210206-1198/2022-09-16_15-36/xgb.pkl' # with normal



with open(model_weight, 'rb') as fin:
    pkl_bst = pickle.load(fin)

print(pkl_bst)
features = list(pkl_bst.feature_importances_)
print(len(features))

preds = pkl_bst.predict(x_test)

print('平均绝对误差mae:',mean_absolute_error(y_test,preds))
print('均方误差mse:', mean_squared_error(y_test, preds))
print('均方根误差mse:', np.sqrt(mean_squared_error(y_test, preds)))
print('r2:', r2_score(y_test, preds))


#保存预测结果
# resList = []
# for p in preds:
#     resList.append(str(p)+'\n')
# with open('exps/20210911-90-model-1068-2d-re360_640_lgb200-predict.txt','w') as f:
#     f.writelines(resList)

# lgb.plot_importance(pkl_bst)
# plt.show()


# 计算5个区间各自的mae
y_test=y_test.to_numpy()
print(len(y_test))
print(len(preds))
min_weight = min(y_test)
max_weight = max(y_test)
inteval = (max_weight - min_weight)/5
print(min_weight,max_weight,inteval)
class_results = {
    0:[[],[]],
    1:[[],[]],
    2:[[],[]],
    3:[[],[]],
    4:[[],[]]
}

for i in range(len(y_test)):
    start = min_weight
    # end = min_weight
    end = min_weight + inteval
    for j in range(5):
        # end = end + inteval
        if j == 4 :
            if y_test[i] >= start and y_test[i] <= max_weight:
                class_results[j][0].append(y_test[i])
                class_results[j][1].append(preds[i])
        else:
            if y_test[i] >= start and y_test[i] <= end:
                class_results[j][0].append(y_test[i])
                class_results[j][1].append(preds[i])

        start = end
        end = end + inteval


for j in range(5):
    print('平均绝对误差:',j,mean_absolute_error(class_results[j][0],class_results[j][1]),"数量：",len(class_results[j][1]))

print(class_results[4])

