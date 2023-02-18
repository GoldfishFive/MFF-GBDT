from torch.utils.data import Dataset
import PIL.Image as Image
import os
import random
from sklearn.model_selection import train_test_split
import  numpy as np
import pandas as pd
import cv2

def data_split():
    dataPath = 'exps/data_20210206-200_weight_100-model-73-50.pth-result/maskImg'
    xlsxPath = 'data/20210206-200/20210206体重登记表.xlsx'
    df = pd.read_excel(xlsxPath)
    # print(df.loc[0]['体重/kg'])

    weightList = []
    nameList = []
    for i in sorted(os.listdir(dataPath)):
        idx = int(i.split('.')[0]) - 1
        weight = df.loc[idx]['体重/kg']
        # imgPath = os.path.join(dataPath,i)
        imgPath = dataPath + '/' + i
        nameList.append(imgPath)
        weightList.append(weight)
        # print(imgPath, weight)

    dataset = pd.DataFrame({'path':nameList, 'weight':weightList})

    # trainset,testset = train_test_split(dataset,test_size=0.2,random_state=43, shuffle=True)
    # trainset,testset = train_test_split(dataset,test_size=0.3,random_state=43, shuffle=True)
    trainset,testset = train_test_split(dataset,test_size=0.2,random_state=45, shuffle=True)
    # print(trainset.head(),testset.head())

    trainset.to_csv('data/20210206-200/dataset/train/trainset.csv')
    testset.to_csv('data/20210206-200/dataset/test/testset.csv')

    return trainset,testset



class Chicken_200_trainset(Dataset):
    def __init__(self, transform=None):
        trainset,testset = data_split()
        self.imgs = trainset.reset_index(drop=True)
        print( self.imgs.head())
        self.transform = transform

    def __getitem__(self, index):
        # print(self.imgs.loc[index])
        x_path,y = self.imgs.loc[index]
        # print(x_path,y)
        img_x = cv2.imread(x_path)
        # img_x = Image.open(x_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        return img_x, y, x_path

    def __len__(self):
        return len(self.imgs)

class Chicken_200_testset(Dataset):
    def __init__(self,transform=None):
        trainset,testset = data_split()
        self.imgs = testset.reset_index(drop=True)
        self.transform = transform

    def __getitem__(self, index):
        x_path,y= self.imgs.loc[index]
        img_x = cv2.imread(x_path)
        # img_x = Image.open(x_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        return img_x, y, x_path

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    # data_split()
    data = Chicken_200_trainset()
    print(data[0])
