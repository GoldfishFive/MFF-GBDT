import torch
import numpy as np
import cv2
import copy
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from model.FC import FC_origin,FC_base,FC_plus
import pandas as pd
import matplotlib.pyplot as plt

# df_train = pd.read_csv('GBDT/csvData/20210206-200-1198-manuals/20210206-1198-train-0.8.csv')
df_train = pd.read_csv('GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-train-0.8.csv')
# df_val = pd.read_csv('../csvData/tvt-1068-23d_val_samples.csv')
# df_test = pd.read_csv('GBDT/csvData/20210206-200-1198-manuals/20210206-1198-test-0.2.csv')
df_test = pd.read_csv('GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-test-0.2.csv')

ss = StandardScaler() # 标准化
# ss = MinMaxScaler() # 归一化
# ss = MaxAbsScaler()

# 2D + 3D 特征
y_train = df_train['weight'] # 获取训练集的y
x_train = df_train.drop(['weight','imgName'],axis=1) # 获得训练集的x  1 按列舍弃
# x_train = df_train.drop(['weight','imgName','equi_diameter'],axis=1) # 获得训练集的x  1 按列舍弃
y_test = df_test['weight'] # 获取测试集的y
x_test = df_test.drop(['weight','imgName'],axis=1) # 获取测试集的x
# x_test = df_test.drop(['weight','imgName','equi_diameter'],axis=1) # 获取测试集的x
# y_val = df_val['weight']
# x_val = df_val.drop(['weight','imgName','equi_diameter'],axis=1)

#  2D特征
# x_train = pd.DataFrame(x_train.values[:, 0:15])
# #x_val = pd.DataFrame(x_val.values[:, 0:15])
# x_test = pd.DataFrame(x_test.values[:, 0:15])
# print(x_train.shape,x_test.shape)

#  3D特征
# x_train = pd.DataFrame(x_train.values[:, 15:24])
# #x_val = pd.DataFrame(x_val.values[:, 15:24])
# x_test = pd.DataFrame(x_test.values[:, 15:24])
print(x_train.shape,x_test.shape)

# ss = ss.fit(x_train)
# x_train = ss.transform(x_train)
# x_test = ss.transform(x_test)
# x_val = ss.transform(x_val)


x_train = torch.from_numpy(x_train.to_numpy()).float()
y_train = torch.from_numpy(y_train.to_numpy()).float()
x_test = torch.from_numpy(x_test.to_numpy()).float()
y_test = torch.from_numpy(y_test.to_numpy()).float()
# x_val = torch.from_numpy(x_val).float()
# y_val = torch.from_numpy(y_val.values).float()

loss_fn = torch.nn.L1Loss()
# model = myfc(15)
# model = myfc(9)
# model = myfc(24)
model_str = 'ANN_1'
# model_str = 'ANN_2'
# model_str = 'ANN_3'

if model_str == 'ANN_1':
    model = FC_origin(25)
    weightPath = 'exps/train_FC/2021-12-15 03-59/122-l0.14436717331409454-Weights.pth'
elif model_str == 'ANN_2':
    model = FC_plus(25)
    weightPath = 'exps/train_FC/2021-12-15 03-56/164-l0.14198146760463715-Weights.pth'
elif model_str == 'ANN_3':
    model = FC_base(25)
    weightPath = 'exps/train_FC/2021-12-15 22-21/189-l0.13980558514595032-Weights.pth'

print(model)

# weightPath = '../modelPkl/bs64-lr0.02-s24_19_1-t2021-06-26 06-42-26-e113-l0.17060346901416779-Weights.pth'

model.load_state_dict(torch.load(weightPath, map_location='cpu'))
test_result = []

# for i in range(len(x_test)):
#     with torch.no_grad():
#         testY = model(x_test[i])
#         # result_train = model(x_train)
#         test_result.append(testY)

with torch.no_grad():
    test_result = model(x_test)

suball = torch.abs(torch.sub(test_result,y_test))
print(np.min(suball.numpy()),np.max(suball.numpy()))





loss = loss_fn(torch.Tensor(test_result), y_test).item()

# train_loss = loss_fn(result_train, y_train).item()
# print("train:")
# print("测试MAE：{:.6f}".format(train_loss))



# bigErr = []
# # errPath = '../data/testBigErrImg/'
# errPath = '../data/trainBigErrImg/' #  筛选训练的
# df_test = df_train  #  筛选训练的
# for i in range(len(res)) :
#     if res[i] > 0.05:  # 误差大于100g的
#         print(df_test.loc[i])
#         print(df_test.loc[i,'imgName']," ",res[i])
#         bigErr.append(res[i])
#         img = cv2.imread('../data/1079/maskImg/'+df_test.loc[i,'imgName'])
#         cv2.imwrite(errPath+df_test.loc[i,'imgName']+str(res[i])+'.png',img)
# print(len(bigErr))

# print(x_test[np.argmax(res)])
# print(res[np.argmax(res)])
# print(np.sort(res))
#
# print(np.argmax(res))
print("测试MAE：{:.6f}".format(loss))
# print('最大误差的预测值：',testY[np.argmax(res)])
# print('实际值',y_test[np.argmax(res)])

#训练集上：
# print('平均绝对误差:',mean_absolute_error(y_train,preds))
# print('均方误差mse:', mean_squared_error(y_train, preds))
# print('均方根误差rmse:', mean_squared_error(y_train, preds) ** 0.5)
# print('R2:',r2_score(y_train,preds))
# print()

#测试集上
# print('平均绝对误差:',mean_absolute_error(test_result.numpy(),y_test.numpy()))
# print('均方误差mse:', mean_squared_error(test_result.numpy(),y_test.numpy()))
# print('均方根误差rmse:', mean_squared_error(test_result.numpy(),y_test.numpy()) ** 0.5)
# print('R2:',r2_score(test_result.numpy(),y_test.numpy()))

print('平均绝对误差:',mean_absolute_error(test_result,y_test.numpy()))
print('均方根误差rmse:', mean_squared_error(test_result,y_test.numpy()) ** 0.5)
print('R2:',r2_score(test_result,y_test.numpy()))


y_test=y_test.numpy()
print(len(y_test))

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
                class_results[j][1].append(test_result[i])
        else:
            if y_test[i] >= start and y_test[i] <= end:
                class_results[j][0].append(y_test[i])
                class_results[j][1].append(test_result[i])

        start = end
        end = end + inteval

for j in range(5):
    print('平均绝对误差:',j,mean_absolute_error(class_results[j][0],class_results[j][1]),"数量：",len(class_results[j][1]))

print(class_results[4])

