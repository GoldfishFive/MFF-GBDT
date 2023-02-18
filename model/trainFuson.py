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
from dataset.chicken200.chicken200 import Chicken_200_trainset, Chicken_200_testset
from model.FusonNet import fusonnet50
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR

def logger(log_str):
    with open(logFilePath,'a',encoding='utf-8') as file:
        file.write(log_str)

def makeEnv():
    global logFilePath
    expName = 'train_fusonnet'
    model_name = 'fusonnet_unfizze'
    nowTime = time.strftime("%Y-%m-%d %H-%M", time.localtime())
    expPath = os.path.join('exps',expName,nowTime)
    if not os.path.exists(expPath):
        os.makedirs(expPath)

    num_classes = 1 #类别数
    batch_size = 8 #
    num_epochs = 150
    lr = 0.001
    fizze_resnet = True
    weightPath = os.path.join(expPath, 'fianlEpochWeights.pth')
    logFilePath = os.path.join(expPath, 'log.txt' )
    init_resnet_weightPath = "exps/myresnet/2021-12-14 22-19/epoch-50-0.14810015708208085-Weights.pth"

    log_str = "model_name " + model_name + '\n' + \
              "num_classes " + str(num_classes) + '\n' \
              "batch_size " + str(batch_size) + '\n' \
              "num_epochs " + str(num_epochs) + '\n' \
              "fizze_resnet " + str(fizze_resnet) + '\n' \
              "learningRate " + str(lr) + '\n' \
              "init_resnet_weightPath " + init_resnet_weightPath + '\n'
    logger(log_str)

    return expPath, model_name, num_epochs, num_classes, batch_size, lr, weightPath, init_resnet_weightPath, fizze_resnet

class myresnet(nn.Module):
    def __init__(self):
        super( myresnet, self ).__init__()
        # self.model_ft=models.resnet18(pretrained=False)
        self.model_ft=models.resnet50(pretrained=False)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1048)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(1048,512)
        self.relu2=nn.ReLU()
        self.fc3 = nn.Linear(512,1)

    def forward(self, x):
        x=self.model_ft(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu2(x)
        x=self.fc3(x)
        x = torch.flatten(x) # 回归时加上

        return x
class myresnet_base(nn.Module):
    def __init__(self):
        super( myresnet_base, self ).__init__()
        # self.model_ft=models.resnet18(pretrained=False)
        self.model_ft=models.resnet50(pretrained=False)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1024)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(1024,512)
        self.relu2=nn.ReLU()
        self.fc3 = nn.Linear(512,1)

    def forward(self, x):
        x=self.model_ft(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu2(x)
        x=self.fc3(x)
        x = torch.flatten(x) # 回归时加上

        return x
class fusonnet_tiny(nn.Module):
    def __init__(self):
        super(fusonnet_tiny, self ).__init__()
        self.fc = nn.Linear(2073,1024)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(1024,512)
        self.relu2=nn.ReLU()
        self.fc3 = nn.Linear(512,1)

    def forward(self, x):
        x=self.fc(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu2(x)
        x=self.fc3(x)
        x = torch.flatten(x) # 回归时加上
        return x
class fusonnet_Base(nn.Module):
    def __init__(self):
        super(fusonnet_Base, self ).__init__()
        self.fc = nn.Linear(2073,2048)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(2048,1024)
        self.relu2=nn.ReLU()
        self.fc3 = nn.Linear(1024,1)

    def forward(self, x):
        x=self.fc(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu2(x)
        x=self.fc3(x)
        x = torch.flatten(x) # 回归时加上
        return x

def get_manual_features():
    # csv_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_features.csv'
    csv_path = 'GBDT/csvData/20210206-200-1198-manuals/20210206-1198_2D_3D_normal_features.csv'
    logger('csv_path: '+csv_path+'\n')
    df = pd.read_csv(csv_path,index_col='imgName')
    print(df.head())
    # df = df.drop(['weight'],axis=1) # 获得训练集的x  1 按列舍弃  normal的已经舍弃了

    print(type(df.loc['1.1_Depth-0.png']))
    return df

def train_model(model, dataloaders, loss_fn, optimizer,scheduler, expPath, device, num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 9999999
    mini_MAE= 9999999
    val_acc_history = []
    train_acc_history = []
    df = get_manual_features()
    for epoch in range(num_epochs):

        for phase in ["train", "val"]:
            weight_gt = []
            weight_pr = []
            running_loss = 0.
            if phase == "train":
                model.train()
            else:
                model.eval()

            for inputs, labels, path in dataloaders[phase]:
                labels_gt = labels.numpy()
                # print(type(labels_gt))
                weight_gt.extend(labels_gt)

                inputs, labels= inputs.to(device), labels.to(device)
                # print(labels.size())
                # print(inputs.size())

                # 处理路径，path，读取 25个手工参数，丢进去训练
                # print(path)
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

                with torch.autograd.set_grad_enabled(phase=="train"):
                    outputs,auto_features = model(inputs, manual_features)
                    # loss = loss_fn(outputs, labels)
                    # print(outputs.size())
                    loss = loss_fn(outputs.float(), labels.float())

                predict_weight = outputs.cpu().detach().numpy()
                # print(predict_weight.shape)
                weight_pr.extend(predict_weight)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                running_loss += loss.item() * inputs.size(0)

            # print(phase)
            val_MAE = mean_absolute_error(weight_gt,weight_pr)

            log_str = phase +'：\n' \
                '平均绝对误差: {:.6f}\n' \
                '均方误差mse: {:.6f}\n' \
                '均方根误差rmse: {:.6f}\n' \
                'R2: {:.6f}\n'.format(mean_absolute_error(weight_gt,weight_pr),mean_squared_error(weight_gt, weight_pr),
                                      mean_squared_error(weight_gt, weight_pr) ** 0.5, r2_score(weight_gt,weight_pr))
            print(log_str),logger(log_str)
            # 回归
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            log_str = "epoch {} Phase {} loss: {}\n\n".format(epoch, phase, epoch_loss*1000)
            print(log_str),logger(log_str) # 一轮的loss

            # if phase == "val" and epoch_loss < best_loss:
            #     best_loss = epoch_loss
            #     best_model_wts = copy.deepcopy(model.state_dict())
            #     torch.save(best_model_wts, expPath + '/epoch-' + str(epoch) + '-' + str(best_loss) + '-Weights.pth')
            if phase == "val" and val_MAE < mini_MAE: # 不是val阶段，直接短路
                mini_MAE = val_MAE
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, expPath + '/epoch-' + str(epoch) + '-' + str(mini_MAE) + '-Weights.pth')
            if phase == "val":
                val_acc_history.append(epoch_loss)
                print()
            if phase == 'train':
                train_acc_history.append(epoch_loss)

    logger('mini_loss: '+str(mini_MAE)+'\n')
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history


def fizze_resnet_parameter(model, fizze_resnet):
    if fizze_resnet:
        keylist = ['fc.weight', 'fc.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
        for name, param in model.named_parameters():
            if name not in keylist:
                param.requires_grad = False  # param.requires_grad == False 的参数不训练
                # print(name)

def init_Fusonmodel(fusonnet, weightPath, fizze_resnet):

    # model_pretrained = myresnet()
    model_pretrained = myresnet_base()
    model_pretrained.load_state_dict(torch.load(weightPath, map_location='cpu'))

    fusonnet_dict = fusonnet.state_dict()
    print(model_pretrained.state_dict().keys())
    print()
    # 将model_pretrained的建与自定义模型的建进行比较，剔除不同的
    pretrained_dict = {k[9:]: v for k, v in model_pretrained.state_dict().items() if k.startswith('model_ft.')} # 拿到 前面的参数
    pretrained_dict.pop('fc.weight')
    pretrained_dict.pop('fc.bias')  # 去掉参数不一样的第一层fa的参数
    print(pretrained_dict.keys())
    # 更新现有的model_dict
    fusonnet_dict.update(pretrained_dict)

    # 加载我们真正需要的state_dict
    fusonnet.load_state_dict(fusonnet_dict)

    if fizze_resnet:
        # 只更新后面的FC层，其他的设置成False，冻住前面的参数
        fizze_resnet_parameter(fusonnet, True)

    return  fusonnet

def train_fusonnet():
    # get_manual_features()
    # exit()
    expPath, model_name, num_epochs, num_classes, batch_size, lr, weightPath, init_resnet_weightPath, fizze_resnet = makeEnv()

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
    model = init_Fusonmodel(fusonnet, init_resnet_weightPath,fizze_resnet)
    print(model)

    model = model.to(device)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                   model.parameters()), lr=lr, momentum=0.9)

    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    logger('scheduler: CosineAnnealingLR(optimizer, T_max=10)\n')
    loss_fn = nn.L1Loss()
    model_trained, ohist, thist = train_model(model, dataloaders_dict, loss_fn, optimizer,scheduler, expPath, device, num_epochs=num_epochs)
    torch.save(model_trained.state_dict(), weightPath)


if __name__ == '__main__':
    train_fusonnet()

