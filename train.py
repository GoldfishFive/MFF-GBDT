from Penn_Fudan_dataset import PennFudanDataset
from Mask_rcnn_Model import get_model_instance_segmentation
import torch
import utils
import os
import cv2
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from engine import train_one_epoch, evaluate
import transforms as T
from transforms import  ToTensor,RandomHorizontalFlip,Compose
# import torchvision.transforms as T
# from torchvision.transforms import transforms

def gotoMain():
    # dataPath = 'data/100label'
    # dataPath = 'data/105'
    dataPath = 'data/105/mixData'
    saveModelName = 'weight/205-model-73-'
    trainValRate = 0.2

    main(dataPath, saveModelName, trainValRate)
    print("That's it!")

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    # if train:
    #     transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

def toTensor(img):
    assert type(img) == np.ndarrau, 'the img type is {}， but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2,0,1)))
    return img.float().div(255)

def PredictImg(image, model, device):
    img = cv2.imread(image)
    result = img.copy()
    dst = img.copy()
    # img = transforms.ToTensor(img)
    img = toTensor(img)

    names = {'0':'background', '1':'chicken'}

    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    boxes = prediction[0]['boxes']
    labels = prediction[0]['boxes']
    scores = prediction[0]['scores']
    masks = prediction[0]['masks']

    m_bOk = False
    for idx in range(boxes.shape[0]):
        if scores[idx] >= 0.5:
            m_bOk = True
            color = (255,255,255)
            mask = masks[idx, 0].mul(255).byte().cpu().numpy()
            thresh = mask
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            cv2.drawContours(dst,contours, -1, color, -1)

            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names.get(str(labels[idx].item()))
            cv2.rectangle(result,(x1,y1),(x2,y2),color,thickness=2)
            cv2.putText(result,text=name,org=(x1,y1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA,color=color)

            dst1 = cv2.addWeighted(result, 0.7,dst, 0.3, 0)

    if m_bOk:
        cv2.imshow('result', dst1)
        cv2.waitKey()
        cv2.destroyWindows()


def main(dataPath, saveModelName, trainValRate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2 # 背景也算一类

    dataset = PennFudanDataset(dataPath, get_transform(train=True))
    dataset_test = PennFudanDataset(dataPath, get_transform(train=False))
    # dataset = PennFudanDataset('105', transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.RandomHorizontalFlip(0.5),
    #     ]))
    # dataset_test = PennFudanDataset('105', transforms.Compose([
    #         transforms.ToTensor()
    #     ]))
    #     21 31  62
    indices = torch.randperm(len(dataset)).tolist()
    valLen = int(len(dataset)*trainValRate) # 验证集大小
    dataset = torch.utils.data.Subset(dataset, indices[:-valLen])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-valLen:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=utils.collate_fn
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1, collate_fn=utils.collate_fn
        )

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9,
                                weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 100
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device)
        if (epoch+1) % 10 == 0 :
            print("保存模型开始。。。")
            torch.save(model.state_dict(), saveModelName+ str(epoch+1)+'.pth')
            print("保存模型完成。。。")

    # print("保存模型开始。。。")
    # torch.save(model.state_dict(), saveModelName + str(num_epochs)+'.pth')
    # print("保存模型完成。。。")
    # utils.save_on_master(
    #     {'model': model.state_dict()},
    #     os.path.join('./', '105model.pth')
    # )

    # PredictImg('105/origin/100_Depth.png', model, device)

if __name__ == '__main__':

    gotoMain()

