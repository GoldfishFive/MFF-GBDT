import torch
import numpy as np
import copy
import datetime
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from model.FC import FC_origin,FC_base,FC_plus
import time,os
import pandas as pd

def logger(log_str):
    with open(logFilePath,'a',encoding='utf-8') as file:
        file.write(log_str)

def makeEnv():
    global logFilePath
    expName = 'train_FC'
    model_name = 'fc'
    nowTime = time.strftime("%Y-%m-%d %H-%M", time.localtime())
    expPath = os.path.join('exps',expName,nowTime)
    if not os.path.exists(expPath):
        os.makedirs(expPath)

    num_classes = 1 #类别数
    batch_size = 8 #
    num_epochs = 200
    lr = 0.001
    weightPath = os.path.join(expPath, 'fianlEpochWeights.pth')
    logFilePath = os.path.join(expPath, 'log.txt' )
    log_str = "model_name " + model_name + '\n' + \
              "num_classes " + str(num_classes) + '\n' \
              "batch_size " + str(batch_size) + '\n' \
              "num_epochs " + str(num_epochs) + '\n' \
              "model_name " + model_name + '\n' \
              "learningRate " + str(lr) + '\n'
    logger(log_str)

    return expPath, model_name, num_epochs, num_classes, batch_size, lr, weightPath, logFilePath
def get_manual_features():
    csv_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_features.csv'
    # csv_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_normal_features.csv'
    logger('csv_path: '+csv_path+'\n')
    df = pd.read_csv(csv_path,index_col='imgName')
    print(df.head())
    # df = df.drop(['weight'],axis=1) # 获得训练集的x  1 按列舍弃  normal的已经舍弃了
    print(type(df.loc['1.1_Depth-0.png']))
    return df

def train_fc(model,x_train,y_train,x_test,y_test,loss_fn,
             optimizer, expPath, device,batch_size, logFilePath, num_epochs=100):

    y_train_copy = y_train
    best_loss = 10000
    bestEpoch = 0
    log_str = ''
    for epoch in range(num_epochs):
        weight_pr = []
        running_loss = 0.
        for start in range(0, len(x_train), batch_size):
            end = start + batch_size
            batchX = x_train[start:end]
            batchY = y_train[start:end]

            outputs= model(batchX)
            loss = loss_fn(outputs, batchY)

            predict_weight = outputs.cpu().detach().numpy()
            # print(predict_weight.shape)
            weight_pr.extend(predict_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size


        print('train:')
        # 回归
        epoch_loss = running_loss / len(x_train)
        print("epoch {} Phase {} loss: {}".format(epoch, 'train', epoch_loss*1000)) # 一轮的loss
        print('平均绝对误差:',"{:.6f}".format(mean_absolute_error(y_train_copy.cpu().numpy(),weight_pr)))
        print('均方误差mse:', "{:.6f}".format(mean_squared_error(y_train_copy.cpu().numpy(),weight_pr)))
        print('均方根误差rmse:', "{:.6f}".format(mean_squared_error(y_train_copy.cpu().numpy(),weight_pr) ** 0.5))
        print('R2:',"{:.6f}".format(r2_score(y_train_copy.cpu().numpy(),weight_pr)))


        # Find loss on val data
        val_outputs = model(x_test)
        loss = loss_fn(val_outputs, y_test).item()
        print('val:')
        print('Epoch:', epoch, 'val loss:', loss)
        print('平均绝对误差:',"{:.6f}".format(mean_absolute_error(val_outputs.cpu().detach().numpy(),y_test.cpu().detach().numpy())))
        print('均方误差mse:', "{:.6f}".format(mean_squared_error(val_outputs.cpu().detach().numpy(),y_test.cpu().detach().numpy())))
        print('均方根误差rmse:', "{:.6f}".format(mean_squared_error(val_outputs.cpu().detach().numpy(),y_test.cpu().detach().numpy()) ** 0.5))
        print('R2:',"{:.6f}\n".format(r2_score(val_outputs.cpu().detach().numpy(),y_test.cpu().detach().detach().numpy())))

        if loss < best_loss:
            best_loss = loss
            bestEpoch = epoch
            if loss < 0.20:
                best_model_wts = copy.deepcopy(model.state_dict())
                savePath = expPath +'/' + str(bestEpoch) + '-l' + str(best_loss)+ '-Weights.pth'
                torch.save(best_model_wts, savePath)

    print('训练MAE：{:.6f}'.format(best_loss))
    print('最佳轮次：',bestEpoch)

    #
    # model.load_state_dict(torch.load(bestPath, map_location='cpu'))
    # with torch.no_grad():
    #     testY = model(x_test)
    # #测试集上
    # print('平均绝对误差:',mean_absolute_error(testY.numpy(),y_test.numpy()))
    # print('均方误差mse:', mean_squared_error(testY.numpy(),y_test.numpy()))
    # print('均方根误差rmse:', mean_squared_error(testY.numpy(),y_test.numpy()) ** 0.5)
    # print('R2:',r2_score(testY.numpy(),y_test.numpy()))


def before_train_fc():
    # 原始的数据：
    train_data_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198-train-0.8.csv'
    test_data_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198-test-0.2.csv'

    # 归一化后的数据：
    # train_data_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-train-0.8.csv'
    # test_data_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-test-0.2.csv'
    logger('train_data_path: '+train_data_path+'\ntest_data_path: '+ test_data_path+'\n')


    df_train = pd.read_csv(train_data_path)
    # #df_val = pd.read_csv(test_data_path)
    df_test = pd.read_csv(test_data_path)


    # 2D + 3D 特征
    y_train = df_train['weight'] # 获取训练集的y
    x_train = df_train.drop(['weight','imgName'],axis=1) # 获得训练集的x  1 按列舍弃
    y_test = df_test['weight'] # 获取测试集的y
    x_test = df_test.drop(['weight','imgName'],axis=1) # 获取测试集的x
    # y_val = df_val['weight']
    # x_val = df_val.drop(['weight','imgName'],axis=1)
    logger("使用全部特征\n")

    #  2D特征
    # x_train = pd.DataFrame(x_train.values[:, 0:15])
    # x_val = pd.DataFrame(x_val.values[:, 0:15])
    # x_test = pd.DataFrame(x_test.values[:, 0:15])
    # print(x_train.shape,x_test.shape)
    # logger("只使用2D特征\n")

    # #  3D特征
    # x_train = pd.DataFrame(x_train.values[:, 15:24])
    # x_val = pd.DataFrame(x_val.values[:, 15:24])
    # x_test = pd.DataFrame(x_test.values[:, 15:24])
    # print(x_train.shape,x_test.shape)
    # logger("只使用3D特征\n")

    #  自动特征
    # x_train = pd.DataFrame(x_train.values[:, 26:2074])
    # x_test = pd.DataFrame(x_test.values[:, 26:2074])
    # x_val = pd.DataFrame(x_val.values[:, 26:2074])
    # logger("只使用自动特征\n")

    x_train = torch.from_numpy(x_train.values).float().cuda()
    y_train = torch.from_numpy(y_train.values).float().cuda()
    x_test = torch.from_numpy(x_test.values).float().cuda()
    y_test = torch.from_numpy(y_test.values).float().cuda()

    expPath, model_name, num_epochs, num_classes, batch_size, lr, weightPath, logFilePath = makeEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_str = 'ANN_1'
    # model_str = 'ANN_2'
    # model_str = 'ANN_3'

    if model_str == 'ANN_1':
        model = FC_origin(25)
    elif model_str == 'ANN_2':
        model = FC_plus(25)
    elif model_str == 'ANN_3':
        model = FC_base(25)
    logger(model_str+'\n')

    print(model)
    model = model.to(device)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=lr, momentum=0.9)
    loss_fn = nn.L1Loss()
    # loss_fn = torch.nn.MSELoss(reduction='sum')

    train_fc(model,x_train,y_train,x_test,y_test,loss_fn, optimizer, expPath, device,batch_size, logFilePath, num_epochs=num_epochs)


if __name__ == '__main__':
   before_train_fc()









