## 用resnet 等训练分类
import torch
import numpy as np
import os
import copy
import time
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
import torchvision.models as models
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from dataset.chicken200.chicken200 import Chicken_200_trainset, Chicken_200_testset
from model.FusonNet import fusonnet50


def logger(log_str):
    # with open(expPath+'/log.txt','a',encoding='utf-8') as file:
    with open('exps/test_fusonnet/2021-12-16 22-37/log.txt','a',encoding='utf-8') as file:
        file.write(log_str)

def makeEnv():
    global expPath

    expName = 'test_fusonnet'
    model_name = 'fusonnet'
    nowTime = time.strftime("%Y-%m-%d %H-%M", time.localtime())
    expPath = os.path.join('exps',expName,nowTime)
    if not os.path.exists(expPath):
        os.makedirs(expPath)

    num_classes = 1 #类别数
    batch_size = 8 # batch_size 只能设为1，一次传一个图片的手工进去的

    log_str = "model_name " + model_name + '\n' + \
              "num_classes " + str(num_classes) + '\n' \
              "batch_size " + str(batch_size) + '\n'
    logger(log_str)

    return expPath, model_name, num_classes, batch_size

def get_manual_features():
    # csv_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_features.csv'
    csv_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_normal_features.csv'
    df = pd.read_csv(csv_path,index_col='imgName')
    logger('csv_path: '+csv_path+'\n')
    # print(df.head())
    # df = df.drop(['weight'],axis=1) # 获得训练集的x  1 按列舍弃  normal的已经舍弃了
    # print(type(df.loc['1.1_Depth-0.png']))
    return df

def save_auto_features(model, dataloaders, expPath, device):
    df = get_manual_features()
    model.eval()

    for phase in ["train", "val"]:
        with open("GBDT/csvData/20210206-200-1198-withauto/20210206-200-1198-withauto-withnormal-"+phase+".csv",'w') as file:
            head = 'weight,imgName,area,perimeter,min_rect_width,min_rect_high,approx_area,approx_perimeter,extent,hull_perimeter,hull_area,' \
                'solidity,max_defect_dist,sum_defect_dist,equi_diameter,ellipse_long,ellipse_short,eccentricity,volume,maxHeight,minHeight,' \
                'max2min,meanHeight,mean2min,mean2max,stdHeight,heightSum'
            for i in range(2048):
                head += ',' + str(i)
            head += '\n'
            file.write(head)

            for inputs, labels, path in dataloaders[phase]:
                # print(labels,type(labels))
                labels_str = str(labels.item())
                inputs, labels= inputs.to(device), labels.to(device)

                # inputs 是图片，labels是体重，path 是路径
                # 处理路径，path，读取 25个手工参数，丢进去训练
                # print(path)
                path = path[0].split('/')[-1]
                # print(path)

                features = df.loc[path].values
                features_str = ','.join([str(i) for i in features])
                features_str = labels_str + ',' + path + ',' + features_str

                manual_features = torch.as_tensor(features, dtype=torch.float32)
                manual_features = torch.unsqueeze(manual_features,0).cuda()
                # print(manual_features)
                # print(type(manual_features),manual_features.size())
                with torch.no_grad():
                    outputs,auto_features = model(inputs, manual_features)
                auto_features = torch.squeeze(auto_features)
                auto_features = auto_features.cpu().numpy()
                # print(auto_features,type(auto_features),auto_features.shape)
                auto_features_str = ','.join([str(i) for i in auto_features])
                final_str = features_str + ',' + auto_features_str + '\n'
                file.write(final_str)

                # exit()
    return

def test_model(model, dataloaders, expPath, device):
    df = get_manual_features()
    model.eval()
    for phase in ["train", "val"]:
        weight_gt = []
        weight_pr = []
        for inputs, labels, path in dataloaders[phase]:
            # print(labels,type(labels))
            labels_gt = labels.numpy()
            # print(type(labels_gt))
            weight_gt.extend(labels_gt)
            inputs, labels= inputs.to(device), labels.to(device)

            # inputs 是图片，labels是体重，path 是路径
            # 处理路径，path，读取 25个手工参数，丢进去训练
            manual_features = []
            # print(path)
            for p in range(len(path)):
                name = path[p].split('/')[-1]
                # print(name)
                features = df.loc[name].values
                manual_features.append(features)


            manual_features = torch.as_tensor(manual_features, dtype=torch.float32)
            # print(manual_features)
            manual_features = manual_features.cuda()
            # print(type(manual_features),manual_features.size())
            with torch.no_grad():
                outputs,auto_features = model(inputs, manual_features)

            # print(outputs,type(outputs))
            predict_weight = outputs.cpu().numpy()
            # print(predict_weight.shape)
            weight_pr.extend(predict_weight)
            # print(weight_gt,weight_pr)
            # print(type(weight_gt),type(weight_pr))
            # exit()
        log_str = phase +'：\n' \
                '平均绝对误差: {:.6f}\n' \
                '均方误差mse: {:.6f}\n' \
                '均方根误差rmse: {:.6f}\n' \
                'R2: {:.6f}\n'.format(mean_absolute_error(weight_gt,weight_pr),mean_squared_error(weight_gt, weight_pr),
                                      mean_squared_error(weight_gt, weight_pr) ** 0.5, r2_score(weight_gt,weight_pr))
        print(log_str),logger(log_str)
        pd.DataFrame({'gt':weight_gt,'pr':weight_pr}).to_csv(expPath+'/'+phase+'_weight_pr.csv')

    return

def test_fusonnet():
    expPath, model_name, num_classes, batch_size = makeEnv()
    init_resnet_weightPath = "exps/resnet_fizze_train_fc/2021-12-12 06-36/epoch-148-0.10712745562195777-Weights.pth" #107g
    # weightPath = "exps/train_fusonnet/2021-12-15 00-10/epoch-50-0.14198220521211624-Weights.pth" # 141
    # weightPath = "exps/train_fusonnet/2021-12-15 05-00/epoch-10-0.11302029200706737-Weights.pth" # 157g
    # weightPath = "exps/train_fusonnet/2021-12-16 00-06/epoch-8-0.14731012320518494-Weights.pth" # 147g
    # weightPath = "exps/train_fusonnet/2021-12-16 05-12/epoch-134-0.1415516132513682-Weights.pth" # 147g

    logger("init_resnet_weightPath: "+init_resnet_weightPath+'\n')
    train_dataset = Chicken_200_trainset(transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((360,640))
    ]))
    val_dataset = Chicken_200_testset(transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((360,640))
    ]))

    image_datasets = {"train":train_dataset, "val":val_dataset}
    dataloaders_dict = {x: DataLoader(image_datasets[x],
        batch_size=batch_size, shuffle=False, num_workers=4) for x in ["train", "val"]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = next(iter(dataloaders_dict['train']))[0]
    print(img.shape)

    fusonnet = fusonnet50()
    fusonnet.load_state_dict(torch.load(init_resnet_weightPath, map_location='cpu'))
    print(fusonnet)

    fusonnet = fusonnet.to(device)

    test_model(fusonnet, dataloaders_dict, expPath, device)


def combineBPandGBDT():
    BP_train_path = 'exps/test_fusonnet/2021-12-16 22-37/train_weight_pr.csv'
    BP_test_path = 'exps/test_fusonnet/2021-12-16 22-37/val_weight_pr.csv'
    GBDT_train_path = 'GBDT/exps/lgbm_data_20210206-1198/2021-12-16_21-58/train_predict.csv'
    GBDT_test_path = 'GBDT/exps/lgbm_data_20210206-1198/2021-12-16_21-58/test_predict.csv'

    # GBDT_train_path = 'GBDT/exps/xgb_data_20210206-1198/2021-12-16_22-53/train_predict.csv'
    # GBDT_test_path = 'GBDT/exps/xgb_data_20210206-1198/2021-12-16_22-53/test_predict.csv'

    BP_train_df = pd.read_csv(BP_train_path)
    BP_test_df = pd.read_csv(BP_test_path)
    GBDT_train_df = pd.read_csv(GBDT_train_path,index_col='index')
    GBDT_test_df = pd.read_csv(GBDT_test_path,index_col='index')

    logger('BP_train_path '+BP_train_path+'\nBP_test_path '+BP_test_path+'\nGBDT_train_path '+GBDT_test_path+'\nGBDT_test_path '+GBDT_test_path+'\n')
    best_p = 0.
    best_q = 0. # q = 1-p
    best_MAE = 9999999
    for p in np.arange(0.00,1.01,0.01):
        q = 1.0 - p

        GBDT_train_df['cb'] =  GBDT_train_df['pr'] * p + BP_train_df['pr'] * q

        MAE =  mean_absolute_error(GBDT_train_df['cb'].values,GBDT_train_df['gt'].values)
        print(p,q,MAE)
        if MAE < best_MAE:
            best_MAE = MAE
            best_p = p
            best_q = q
            print('best_p '+ str(best_p)+'\nbest_q '+ str(best_q)+'\n')

    best_str = 'best_p '+ str(best_p)+'\nbest_q '+ str(best_q)+'\n'
    print(best_str),logger(best_str)

    GBDT_test_df['pr'] *= best_p
    BP_test_df['pr'] *= best_q
    GBDT_test_df['cb'] = BP_test_df['pr'] + GBDT_test_df['pr']
    # GBDT_test_df.to_csv(expPath + '/combined_result.csv')
    GBDT_test_df.to_csv('exps/test_fusonnet/2021-12-16 22-37/combined_result.csv')

    log_str = '平均绝对误差: {:.6f}\n' \
                '均方误差mse: {:.6f}\n' \
                '均方根误差rmse: {:.6f}\n' \
                'R2: {:.6f}\n'.format(mean_absolute_error(GBDT_test_df['cb'].values,GBDT_test_df['gt'].values),mean_squared_error(GBDT_test_df['cb'].values,GBDT_test_df['gt'].values),
                                      mean_squared_error(GBDT_test_df['cb'].values,GBDT_test_df['gt'].values) ** 0.5, r2_score(GBDT_test_df['cb'].values,GBDT_test_df['gt'].values))
    print(log_str),logger(log_str)

if __name__ == '__main__':
    # test_fusonnet()
    combineBPandGBDT()


