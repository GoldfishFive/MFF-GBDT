## 用resnet 等训练分类
import collections

import torch
import numpy as np
import os
import time
from torch.utils.data import DataLoader
from torch import nn
from torchvision.transforms import transforms
import torchvision.models as models
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from dataset.chicken200.chicken200 import Chicken_200_trainset, Chicken_200_testset


def makeEnv():

    expName = 'test_myresnet'
    model_name = 'myresnet'
    nowTime = time.strftime("%Y-%m-%d %H-%M", time.localtime())
    expPath = os.path.join('exps',expName,nowTime)
    if not os.path.exists(expPath):
        os.makedirs(expPath)

    num_classes = 1 #类别数
    batch_size = 8
    num_epochs = 100
    lr = 0.001
    feature_extract = False # 【False】 训练整个网络finetune the whole model | 【True】 提取特征 only update the reshaped layer params

    weightPath = os.path.join(expPath, 'fianlEpochWeights.pth')
    logFilePath = os.path.join(expPath, 'train.log' )

    with open(logFilePath, 'w') as f:
        lines = list()
        lines.append("model_name " + model_name + '\n')
        lines.append("num_classes " + str(num_classes) + '\n')
        lines.append("batch_size " + str(batch_size) + '\n')
        lines.append("num_epochs " + str(num_epochs) + '\n')
        lines.append("model_name " + model_name + '\n')
        lines.append("learningRate " + str(lr) + '\n')
        lines.append("feature_extract " + str(feature_extract) + '\n')
        lines.append("weightPath " + weightPath + '\n')
        f.writelines(lines)

    return expPath, model_name, num_epochs, num_classes, batch_size, lr, weightPath, feature_extract

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
        self.model_ft.fc = nn.Linear(num_ftrs, 1048)
        # self.model_ft.fc = nn.Linear(num_ftrs, 1024)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(1048,512)
        # self.fc2 = nn.Linear(1024,512)
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
def test_model(model, dataloaders, expPath, device):
    model.eval()
    weight_gt = []
    weight_pr = []
    for phase in ["train", "val"]:
        for inputs, labels, path in dataloaders[phase]:
            # print(labels,type(labels))
            labels_gt = labels.numpy()
            # print(type(labels_gt))
            weight_gt.extend(labels_gt)
            inputs, labels= inputs.to(device), labels.to(device)

            # inputs 是图片，labels是体重，path 是路径
            # 处理路径，path，读取 25个手工参数，丢进去训练
            # print(path)
            path = path[0].split('/')[-1]
            # print(path)

            with torch.no_grad():
                outputs = model(inputs)

            # print(outputs,type(outputs))
            predict_weight = outputs.cpu().numpy()
            # print(predict_weight.shape)
            weight_pr.extend(predict_weight)

            # print(weight_gt,weight_pr)

        print(phase)
        print('平均绝对误差:',"{:.6f}".format(mean_absolute_error(weight_gt,weight_pr)))
        print('均方误差mse:', "{:.6f}".format(mean_squared_error(weight_gt,weight_pr)))
        print('均方根误差rmse:', "{:.6f}".format(mean_squared_error(weight_gt,weight_pr) ** 0.5))
        print('R2:',"{:.6f}".format(r2_score(weight_gt,weight_pr)))

    return

def before_test_resnet():
    expPath, model_name, num_epochs, num_classes, batch_size, lr, weightPath, feature_extract = makeEnv()
    weightPath = "exps/myresnet/2021-12-11 01-48/fianlEpochWeights.pth"
    # weightPath = "exps/myresnet/2021-12-14 22-19/epoch-50-0.14810015708208085-Weights.pth"

    train_dataset = Chicken_200_trainset(transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((360,640))
    ]))
    val_dataset = Chicken_200_testset(transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((360,640))
    ]))
    image_datasets = {"train":train_dataset, "val":val_dataset}
    dataloaders_dict = {x: DataLoader(image_datasets[x],batch_size=batch_size, shuffle=False, num_workers=4) for x in ["train", "val"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet = myresnet_base()
    resnet.load_state_dict(torch.load(weightPath, map_location='cpu'))
    print(resnet)

    resnet = resnet.to(device)
    test_model(resnet, dataloaders_dict, expPath, device)

def make_pth():
    """还原resnet权重文件，带上前缀“model_ft.”"""
    weightPath = "exps/myresnet/2021-12-11 01-48/epoch-95-0.15137851883967718-Weights.pth"

    pth = torch.load(weightPath, map_location='cpu')
    print(pth.keys(),type(pth))
    exceptkey = ['fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']

    mydict = collections.OrderedDict()
    for k in pth.keys():
        print(k,pth[k].size())
        if k not in exceptkey:
            mydict['model_ft.'+k] = pth[k]
        else:
            mydict[k] = pth[k]

    print(mydict.keys(),type(pth))
    # torch.save(mydict,weightPath)

if __name__ == '__main__':
    before_test_resnet()
    # make_pth()


