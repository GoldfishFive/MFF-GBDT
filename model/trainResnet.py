## 用resnet 等训练分类
import torch
import numpy as np
import os
import copy
import time
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
import torchvision.models as models
from dataset.chicken200.chicken200 import Chicken_200_trainset, Chicken_200_testset
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def makeEnv():

    expName = 'myresnet'
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

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
    if model_name == "Resnet":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # print(model_ft)
    elif model_name == "myresnet":
        model_ft = myresnet()
    else:
        print("model not implemented")
        return None
    return model_ft

class myresnet(nn.Module):
    def __init__(self):
        super( myresnet, self ).__init__()
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

def train_model(model, dataloaders, loss_fn, optimizer, expPath, num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 9999999
    val_acc_history = []
    train_acc_history = []

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

                with torch.autograd.set_grad_enabled(phase=="train"):
                    outputs = model(inputs)
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

                running_loss += loss.item() * inputs.size(0)  # L1loss默认除了batch_size,

            print(phase)
            print('平均绝对误差:',"{:.6f}".format(mean_absolute_error(weight_gt,weight_pr)))
            print('均方误差mse:', "{:.6f}".format(mean_squared_error(weight_gt,weight_pr)))
            print('均方根误差rmse:', "{:.6f}".format(mean_squared_error(weight_gt,weight_pr) ** 0.5))
            print('R2:',"{:.6f}".format(r2_score(weight_gt,weight_pr)))
            # 回归
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print("epoch {} Phase {} loss: {}".format(epoch, phase, epoch_loss*1000)) # 一轮的loss

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, expPath + '/epoch-' + str(epoch) + '-' + str(best_loss) + '-Weights.pth')
            if phase == "val":
                val_acc_history.append(epoch_loss)
                print()
            if phase == 'train':
                train_acc_history.append(epoch_loss)

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history

def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False



if __name__ == '__main__':
    expPath, model_name, num_epochs, num_classes, batch_size, lr, weightPath, feature_extract = makeEnv()

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

    unloader = transforms.ToPILImage()  # reconvert into PIL image

    # model_ft = initialize_model(model_name,
    #                 num_classes, feature_extract, use_pretrained=True)
    # print(model_ft)
    # print(type(model_ft))
    # # print(model_ft.layer1[0].conv1.weight.requires_grad) #查看模型前面的是否要训练
    # # print(model_ft.fc.weight.requires_grad) # 查看最后的全连接层是否要训练
    #
    # # 训练模型
    # model_ft = model_ft.to(device)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
    #                                model_ft.parameters()), lr=learningRate, momentum=0.9)
    # # loss_fn = nn.CrossEntropyLoss() # 分类损失函数
    # # loss_fn = nn.MSELoss()   # 回归 均方差损失
    # loss_fn = nn.L1Loss()
    # _, ohist, thist = train_model(model_ft, dataloaders_dict, loss_fn, optimizer, num_epochs=num_epochs)
    # torch.save(_.state_dict(), weightPath)

    # 全部重新训练
    model_ft= initialize_model(model_name,
                        num_classes, feature_extract, use_pretrained=False)
    print(model_ft)
    print(type(model_ft))
    model_scratch = model_ft.to(device),
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model_ft.parameters()), lr=lr, momentum=0.9)
    loss_fn = nn.L1Loss()
    ourModel, ohist, thist = train_model(model_ft, dataloaders_dict, loss_fn, optimizer, expPath, num_epochs=num_epochs)
    torch.save(ourModel.state_dict(), weightPath)

    with open(os.path.join(expPath, 'train.log'), 'a') as file:
        file.writelines(str(ohist)+'\n')
        file.writelines(str(thist))

    plt.figure(figsize=(20,8),dpi=100)
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,num_epochs+1),ohist,label="valacc")
    for x,y in zip(range(1,num_epochs+1),ohist):
        plt.text(x,y+0.01,"{:.3}".format(y),ha = 'center',va = 'bottom',fontsize=7)

    plt.plot(range(1,num_epochs+1),thist,label="trainacc")
    for x,y in zip(range(1,num_epochs+1),thist):
        plt.text(x,y+0.01,"{:.3}".format(y),ha = 'center',va = 'bottom',fontsize=7)

    plt.ylim((0,1.))# 轴范围
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()

    plt.savefig(expPath + '/result.png')
    plt.show()

