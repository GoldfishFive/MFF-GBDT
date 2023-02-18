from PIL import Image
import os
import numpy as np
import torch
import torch.utils.data


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transfroms):
        self.root = root
        self.transfroms = transfroms
        # 在当前工作目录下获取所有排序好的文件名存入一个list
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'origin'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'mask'))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'origin', self.imgs[idx])
        mask_path = os.path.join(self.root, 'mask', self.masks[idx])
        # 确保图像为RGB模式，而mask不需要转换为RGB模式，因为mask背景为0，其他每种颜色代表一个实例
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        # 把PIL图像转换为numpy数组，得到mask中的实例编码并去掉背景
        mask = np.array(mask)
        obj_id = np.unique(mask)
        obj_id = obj_id[1:]

        # None就是newaxis,相当于多了一个维度
        # split the color-encoded mask to a set of binary masks
        # 下面这行代码的解释：以FudanPed000012为例，有两个目标，FudanPed000012_mask中像素为0表示背景，
        # 像素1表示目标1，像素2表示目标2，仅用于代表目标，而并非通过颜色显示，所以点开mask图像肉眼看到全部都是黑色的
        # mask是一个559*536的二维矩阵，obj_id=[0， 1, 2]
        # “obj_ids = obj_ids[1:]”去掉背景像素0 ， 故obj_id=[1, 2]
        # 而下面这行代码，创建了masks（2*559*536），包含两个大小为（559*536）的mask，分别对应第一个目标和第二个目标，
        # 第一个mask中，目标1所占像素为True，其余全为False，第二个mask中，目标2所占像素为True,其余全为False。
        
        masks = mask == obj_id[:, None, None]  # 即使图片的L模式为8字节单通道，而PIL读入时仍作为3通道处理

        # 对于每一个mask的边界框坐标
        num_objs= len(obj_id)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 数据集只有一个类别
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transfroms is not None:

            img ,target = self.transfroms(img, target)
            # target["masks"] = self.transfroms(target["masks"])

        return img, target

    def __len__(self):
        return len(self.imgs)

# 验证输出
# dataset = PennFudanDataset('PennFudanPed/')
# print(dataset[0])
