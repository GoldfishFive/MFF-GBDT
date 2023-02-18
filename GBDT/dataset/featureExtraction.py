import pandas as pd
import numpy as  np
import cv2
import os
import xlrd

def getWeightFromExcel():
    """
    从 每个文件夹的 excel 文件中 拿到 体重字典
    :return: dir_weightDict
    """
    dir_weightDict = dict()
    for imgDir in os.listdir('E:/chicken-mean/20210911mean/'):
        excelPath = 'E:/chicken-mean/20210911mean/' + imgDir
        pathList = os.listdir(excelPath)

        excelName = ''
        weightDict = dict()

        for p in pathList:
            if p.endswith('xlsx'):
                excelName = p

        wk = xlrd.open_workbook(os.path.join(excelPath,excelName))
        sheet = wk.sheet_by_index(0)
        # 遍历行
        for row_num in range(2, sheet.nrows): # 从第二行开始
            id = sheet.row_values(row_num)[0]  # 第一列是序号
            weight = sheet.row_values(row_num)[1] # 第而二列是体重/kg
            print(id, weight)
            weightDict[id] = weight

        dir_weightDict[imgDir] = weightDict

    return dir_weightDict


# TODO 改成从多个分割结果文件夹中 提取特征


imgDir = ['1-21','2-35','34-7']
# imgDir = ['56-110']
saveCSVname = 'csvData/20210911-90_2D_3D_features.csv'

imgNameList = []

# 2D特征
areaList = []
perimeterList = []
min_rect_widthList = []
min_rect_highList = []
approx_areaList = []
approx_perimeterList = []
extentList = []
hull_perimeterList = []
hull_areaList = []
solidityList = []
max_defect_distList = []
sum_defect_distList = []
equi_diameterList = []
ellipse_shortList = []
ellipse_longList = []
eccentricityList = []
weightList = []

# 3D特征
volumeList = []
maxHeightList = []
minHeightList = []
max2minList = []
meanHeightList = []
mean2minList = []
mean2maxList = []
stdHeightList = []
heightSumList = []


dataDirt = {'weight': weightList, 'imgName': imgNameList, 'area': areaList, 'perimeter': perimeterList, 'min_rect_width': min_rect_widthList,
            'min_rect_high': min_rect_highList, 'approx_area': approx_areaList, 'approx_perimeter': approx_perimeterList,
            'extent': extentList, 'hull_perimeter': hull_perimeterList, 'hull_area': hull_areaList, 'solidity': solidityList,
            'max_defect_dist': max_defect_distList, 'sum_defect_dist': sum_defect_distList, 'equi_diameter': equi_diameterList,
            'ellipse_long': ellipse_longList, 'ellipse_short': ellipse_shortList, 'eccentricity': eccentricityList,

            'volume': volumeList, 'maxHeight': maxHeightList, 'minHeight': minHeightList,'max2min':max2minList,
            'meanHeight': meanHeightList,'mean2min': mean2minList, 'mean2max': mean2maxList,'stdHeight': stdHeightList,
            'heightSum': heightSumList,
            }

dir_weightDict = getWeightFromExcel()

def getdep(rawImgPath,format,resolution):
    """
    获取 原始 raw 深度数据 转成numpy格式 单通道图像
    :param rawImgPath: raw 文件路径
    :param format: 'Z16' or 'DISPARITY32'
    :param resolution: (w,h)
    :return: numpy_arrary(w,h)
    """
    if format == 'Z16':
        type = np.int16 # format = Z16
        width, height = resolution
    # width, height = (720, 1280)
    elif format == 'DISPARITY32':
        type = np.float32 # format = DISPARITY32
        width, height = resolution

    imgData = np.fromfile(rawImgPath, dtype=type)
    imgData = imgData.reshape(width, height)

    return imgData

for dir in imgDir:

    dirPath = 'data/20210911/' + dir + '/maskImg/'
    maskPath = 'data/20210911/' + dir + '/mask/'
    rawPath = 'data/20210911/' + dir + '/raw/'

    imgNames = os.listdir(dirPath)
    imgNameList.extend(imgNames)

    for name in imgNames:
        print(name)
        weight = dir_weightDict[dir][int(name.split('.')[0])]
        print('鸡的体重:',weight)
        weightList.append(weight)

        # imgPath = os.path.join(dirPath, name)
        # img = cv2.imread(imgPath)
        # name = '20210312166-6.png'

        dataItem = []
        imgPath = os.path.join(maskPath, name)
        print(imgPath)
        mask = cv2.imread(imgPath)
        ret, thresh = cv2.threshold(mask[:,:,0], 200, 1,0) # 阈值分割一下 预测出来的mask


        # print(type(mask),mask.shape)
        # cv2.imshow('thresh',thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #查找轮廓
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        print(len(contours))
        areas = []
        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))
        #最大面积轮廓
        area = maxAreas = max(areas)  #投影面积
        print('投影面积', area)
        areaList.append(area)

        maxArea_id = areas.index(maxAreas)
        cnt = contours[maxArea_id] #cnt 为最大面积的轮廓

        perimeter = cv2.arcLength(cnt,True)  #轮廓周长
        print('轮廓周长', perimeter)
        perimeterList.append(perimeter)

        maxArea_min_rect = cv2.minAreaRect(cnt) # 最大面积轮廓的最小矩形框
        # print("最小矩形",maxArea_min_rect)  #左上角角点的坐标（x，y），矩形的宽和高（w，h），以及旋转角度

        print("最小矩形的宽",maxArea_min_rect[1][0])  #左上角角点的坐标（x，y），矩形的宽和高（w，h），以及旋转角度
        print("最小矩形的高",maxArea_min_rect[1][1])  #左上角角点的坐标（x，y），矩形的宽和高（w，h），以及旋转角度
        min_rect_widthList.append(maxArea_min_rect[1][0])
        min_rect_highList.append(maxArea_min_rect[1][1])

        epsilon = 0.01*cv2.arcLength(cnt,True) # 0.01倍的轮廓长度 作为近似计算的参数
        approx = cv2.approxPolyDP(cnt,epsilon,True)  #近似轮廓
        approx_area = cv2.contourArea(approx)  # 近似轮廓的面积
        approx_perimeter = cv2.arcLength(approx,True) #近似轮廓的周长
        print('近似轮廓的面积',approx_area)
        print('近似轮廓的周长',approx_perimeter)
        approx_areaList.append(approx_area)
        approx_perimeterList.append(approx_perimeter)

        x,y,w,h = cv2.boundingRect(cnt) #直边界矩形 一个直矩形（就是没有旋转的矩形）
        rect_area = w*h
        extent = float(maxAreas)/rect_area  #轮廓面积与边界矩形面积的比   实际上反映出区域的扩展范围程度
        print('轮廓面积与边界矩形面积的比', extent)
        extentList.append(extent)

        hull = cv2.convexHull(cnt)  #轮廓的凸包
        hull_perimeter = cv2.arcLength(hull,True) # 凸包周长
        hull_area = cv2.contourArea(hull)  # 凸包面积
        solidity = float(area)/hull_area  #轮廓面积与凸包面积的比  实际上反映出区域的固靠性程度
        print('凸包周长',hull_perimeter)
        print('凸包面积', hull_area)
        print('轮廓面积与凸包面积的比', solidity)
        hull_perimeterList.append(hull_perimeter)
        hull_areaList.append(hull_area)
        solidityList.append(solidity)

        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull) # 凸包的凸缺陷
        # print(defects.shape)

        distenceList = []  #每一个凸缺陷 到最远点的近似距离
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            # print(s,e,f,d)
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            distenceList.append(d)
            # cv2.line(mask,start,end,[0,255,0],1)
            # cv2.circle(mask,far,3,[0,0,255],-1)
        # cv2.imshow('mask',mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        max_defect_dist = max(distenceList) #凸缺陷到最远点的最大近似距离
        sum_defect_dist = sum(distenceList) #所有凸缺陷最远点的近似距离之和
        print('凸缺陷到最远点的最大近似距离', max_defect_dist)
        print('所有凸缺陷最远点的近似距离之和', sum_defect_dist)
        max_defect_distList.append(max_defect_dist)
        sum_defect_distList.append(sum_defect_dist)

        equi_diameter = np.sqrt(4*area/np.pi)  #与轮廓面积相等的圆形的直径
        print('与轮廓面积相等的圆形的直径', equi_diameter)
        equi_diameterList.append(equi_diameter)

        M = cv2.moments(cnt) # 轮廓的距
        # print (M)
        ellipse = cv2.fitEllipse(cnt) # 椭圆拟合  长轴和短轴
        # print(ellipse)  # （（中心点x,y), (短轴直径，长轴直径），旋转角度）
        print('椭圆拟合的短轴直径', ellipse[1][0])
        print('椭圆拟合的长轴直径', ellipse[1][1])
        ellipse_shortList.append(ellipse[1][0])
        ellipse_longList.append(ellipse[1][1])

        denominator = np.sqrt(pow(2 * M['mu11'], 2) + pow(M['mu20'] - M['mu02'], 2))
        eps = 1e-4   #定义一个极小值
        if (denominator > eps):
            #cosmin和sinmin用于计算图像协方差矩阵中较小的那个特征值λ2
            cosmin = (M['mu20'] - M['mu02']) / denominator
            sinmin = 2 * M['mu11'] / denominator
            #cosmin和sinmin用于计算图像协方差矩阵中较大的那个特征值λ1
            cosmax = -cosmin
            sinmax = -sinmin
             #imin为λ2乘以零阶中心矩μ00
            imin = 0.5 * (M['mu20'] + M['mu02']) - 0.5 * (M['mu20'] - M['mu02']) * cosmin - M['mu11'] * sinmin
             #imax为λ1乘以零阶中心矩μ00
            imax = 0.5 * (M['mu20'] + M['mu02']) - 0.5 * (M['mu20'] - M['mu02']) * cosmax - M['mu11'] * sinmax
            ratio = imin / imax   #椭圆的惯性率

            eccentricity  = np.sqrt(1- ratio*ratio)  # 椭圆离心率
        else:
            eccentricity  = 0  # 椭圆离心率 为 0 一个圆

        print("椭圆离心率",eccentricity)
        eccentricityList.append(eccentricity)


        #################  提取三维特征 ##############
        rawImgPath = os.path.join(rawPath, name.split('-')[0] + '.raw')
        print(rawImgPath)

        """
        修改成 统一使用conImg， 而不是直接用deepImg来提取3D特征
        """
        if dir == '34-7' or dir == '56-110':
            deepImg = getdep(rawImgPath,'DISPARITY32',(360,640)) # 深度矩阵
            """
            maxD = 1.5时 alpha = 0.17； alpha = 255/(maxD*10**3)
            """
            conImg = cv2.convertScaleAbs(deepImg, alpha=0.17)
            conImg = 255 - conImg
            conImg = np.where(conImg==255,0,conImg)
        else:
            deepImg = getdep(rawImgPath,'Z16',(360,640)) # 深度矩阵
            conImg = cv2.convertScaleAbs(deepImg, alpha=0.17)

        print(conImg.dtype)
        # mul_img = cv2.multiply(thresh.astype(np.int16), conImg)  # 保留了原始16bit的int16的数据
        mul_img = cv2.multiply(thresh.astype(np.uint8), conImg)  # 保留了原始16bit的int16的数据
        deepArr = mul_img[mul_img>0]
        print(deepArr)

        maxHeight = np.max(mul_img)
        minHeight = np.min(deepArr) # 非0的最小值
        meanHeight = np.mean(deepArr)
        max2min = maxHeight - minHeight
        mean2min = meanHeight - minHeight
        mean2max = maxHeight - meanHeight
        stdHeight = np.std(deepArr)
        heightSum=np.sum(mul_img)
        print('最大深度：',maxHeight)
        print('最小深度：',minHeight)
        print('平均深度：',meanHeight)
        print('深度极差：', max2min)
        print('均值到最小值的距离：', mean2min)
        print('均值到最大值的距离：', mean2max)
        print('距离高度的标准差：', stdHeight)
        print('深度之和：',heightSum)

        volumeZhu = area * maxHeight
        # print('Vzhu = ',Vzhu)
        volume = volumeZhu - heightSum    #肉鸡的大概体积
        print('体积：', volume)

        maxHeightList.append(maxHeight)
        minHeightList.append(minHeight)
        max2minList.append(max2min)
        meanHeightList.append(meanHeight)
        mean2minList.append(mean2min)
        mean2maxList.append(mean2max)
        stdHeightList.append(stdHeight)
        heightSumList.append(heightSum)
        volumeList.append(volume)

        # serier = pd.Series(dataItem)
        # print(serier)

        # break


print(dataDirt)
df = pd.DataFrame(dataDirt)
print(df.info)

df.to_csv(saveCSVname,index=False)

