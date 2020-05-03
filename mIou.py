#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import json
from PIL import Image
from os.path import join

#设标签宽W，长H
def fast_hist(a, b, n):#a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
    '''
	核心代码
	'''
    k = (a >= 0) & (a < n) #k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景） k=0或1
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)#np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)


def per_class_iu(hist):#分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
	核心代码
	'''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))#矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
#hist.sum(0)=按列相加  hist.sum(1)按行相加


#def label_mapping(input, mapping):#主要是因为CityScapes标签里面原类别太多，这样做把其他类别转换成算法需要的类别（共19类）和背景（标注为255）
#    output = np.copy(input)#先复制一下输入图像
#    for ind in range(len(mapping)):
#        output[input == mapping[ind][0]] = mapping[ind][1]#进行类别映射，最终得到的标签里面之后0-18这19个数加255（背景）
#    return np.array(output, dtype=np.int64)#返回映射的标签
'''
  compute_mIoU函数原始以CityScapes图像分割验证集为例来计算mIoU值的（可以根据自己数据集的不同更改类别数num_classes及类别名称name_classes），本函数除了最主要的计算mIoU的代码之外，还完成了一些其他操作，比如进行数据读取，因为原文是做图像分割迁移方面的工作，因此还进行了标签映射的相关工作，在这里笔者都进行注释。大家在使用的时候，可以忽略原作者的数据读取过程，只需要注意计算mIoU的时候每张图片分割结果与标签要配对。主要留意mIoU指标的计算核心代码即可。
'''
def compute_mIoU(gt_dir, pred_dir, devkit_dir):#计算mIoU的函数
    """
    Compute IoU given the predicted colorized images and 
    """
    with open('/home/ubuntu/DeepLab/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/info.json', 'r') as fp: 
        #读取info.json，里面记录了类别数目，类别名称。（我们数据集是VOC2011，相应地改了josn文件）
        info = json.load(fp) 
    num_classes = np.int(info['classes'])#读取类别数目，这里是20类
    print('Num classes', num_classes)#打印一下类别数目
    name_classes = np.array(info['label'], dtype=np.str)#读取类别名称
    #mapping = np.array(info['label2train'], dtype=np.int)#读取标签映射方式，详见博客中附加的info.json文件
    hist = np.zeros((num_classes, num_classes))#hist初始化为全零，在这里的hist的形状是[20, 20]
    '''
    原代码是有进行类别映射，所以通过json文件来存放类别数目、类别名称、 标签映射方式。而我们只需要读取类别数目和类别名称即可，可以按下面这段代码将其写死
    num_classes=20
    print('Num classes', num_classes)
    name_classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car", "cat","chair","cow","diningtable","dog","horse","motobike","person","pottedplant","sheep","sofa","train","tvmonitor"]
    hist = np.zeros((num_classes, num_classes))
    '''
    image_path_list = join(devkit_dir, 'val2.txt')#在这里打开记录分割图片名称的txt
    label_path_list = join(devkit_dir, 'val2.txt')#ground truth和自己的分割结果txt一样
    gt_imgs = open(label_path_list, 'r').read().splitlines()#获得验证集标签名称列表
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]#获得验证集标签路径列表，方便直接读取
    pred_imgs = open(image_path_list, 'r').read().splitlines()#获得验证集图像分割结果名称列表
    pred_imgs = [join(pred_dir, x) for x in pred_imgs]
    #pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]#获得验证集图像分割结果路径列表，方便直接读取


    for ind in range(len(gt_imgs)):#读取每一个（图片-标签）对
        pred = np.array(Image.open(pred_imgs[ind]))#读取一张图像分割结果，转化成numpy数组
        label = np.array(Image.open(gt_imgs[ind]))#读取一张对应的标签，转化成numpy数组
        #print pred.shape
        #print label.shape
        #label = label_mapping(label, mapping)#进行标签映射（因为没有用到全部类别，因此舍弃某些类别），可忽略
        if len(label.flatten()) != len(pred.flatten()):#如果图像分割结果与标签的大小不一样，这张图片就不计算
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)#对一张图片计算19×19的hist矩阵，并累加
        if ind > 0 and ind % 10 == 0:#每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
            print(per_class_iu(hist))
    
    mIoUs = per_class_iu(hist)#计算所有验证集图片的逐类别mIoU值
    for ind_class in range(num_classes):#逐类别输出一下mIoU值
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))#在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    return mIoUs


compute_mIoU('/home/ubuntu/DeepLab/datasets/VOCdevkit/VOC2012/SegmentationClass/',
                 '/home/ubuntu/DeepLab/datasets/VOCdevkit/VOC2012/zhuosetu/',
                 '/home/ubuntu/DeepLab/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/'
                )#执行主函数 三个路径分别为 ‘ground truth’,'自己的实验分割结果'，‘分割图片名称txt文件’






第二个

import numpy as np
import torch

def iou_mean(pred, target, n_classes = 1):
#n_classes ：the number of classes in your dataset,not including background
# for mask and ground-truth label, not probability map
  ious = []
  iousSum = 0
  pred = torch.from_numpy(pred)
  pred = pred.view(-1)
  target = np.array(target)
  target = torch.from_numpy(target)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
      iousSum += float(intersection) / float(max(union, 1))
  return iousSum/n_classes



def batch_intersection_union(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor 具体地为B(batch大小)*C(通道)*H*W
        target: label 3D tensor 具体地为B(batch大小)*H*W
        nclass: number of categories (int)
    """
    #在通道维上取最大值，注意predict为第二个返回值，因此是索引，从0开始
    #此时predict和target一样，维度均为B*H*W，且值均为0，1，2.........
    _, predict = torch.max(predict, 1) 
    mini = 1
    maxi = nclass
    nbins = nclass
    #将predict和target放入cpu并转换为numpy数组，同时+1
    #此时predict和target的值为1，2，......
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1

   #假如我除去背景类只有1类目标，则nbins为2。
   #此句似乎没有实际意义？predict值不变，因为target值均大于0
   # (target > 0)返回的均为true
    predict = predict * (target > 0).astype(predict.dtype)
    #求交集intersection，维度B*H*W，包含背景类0的交集
    #intersection交集处>0，非交集处为0（其内部像素值包括0，1，2）
    intersection = predict * (predict == target)
    # areas of intersection and union
    #绘制直方图,nbins个区间，range=(mini, maxi)左闭右开
    #假如我除去背景类只有1类目标，则nbins为2，range=(1,2)，则表示将数组均匀地分为2个区间：[1,1.5],[1.5,2]
    #第一个bins代表背景，第二个bins代表目标
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    #交集一定小于并集
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union


import numpy as np
import argparse
import json
from PIL import Image
from os.path import join

#设标签宽W，长H
def fast_hist(a, b, n):#a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
    '''
	核心代码
	'''
    k = (a >= 0) & (a < n)#k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景）
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)#np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)


def per_class_iu(hist):#分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
	核心代码
	'''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))#矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)


def label_mapping(input, mapping):#主要是因为CityScapes标签里面原类别太多，这样做把其他类别转换成算法需要的类别（共19类）和背景（标注为255）
    output = np.copy(input)#先复制一下输入图像
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]#进行类别映射，最终得到的标签里面之后0-18这19个数加255（背景）
    return np.array(output, dtype=np.int64)#返回映射的标签
'''
compute_mIoU函数是以CityScapes图像分割验证集为例来计算mIoU值的
由于作者个人贡献的原因，本函数除了最主要的计算mIoU的代码之外，还完成了一些其他操作，
比如进行数据读取，因为原文是做图像分割迁移方面的工作，因此还进行了标签映射的相关工作，在这里笔者都进行注释。
大家在使用的时候，可以忽略原作者的数据读取过程，只需要注意计算mIoU的时候每张图片分割结果与标签要配对。
主要留意mIoU指标的计算核心代码即可。
'''
def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):#计算mIoU的函数
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp: #读取info.json，里面记录了类别数目，类别名称，标签映射方式等等。
      info = json.load(fp)
    num_classes = np.int(info['classes'])#读取类别数目，这里是19类，详见博客中附加的info.json文件
    print('Num classes', num_classes)#打印一下类别数目
    name_classes = np.array(info['label'], dtype=np.str)#读取类别名称，详见博客中附加的info.json文件
    mapping = np.array(info['label2train'], dtype=np.int)#读取标签映射方式，详见博客中附加的info.json文件
    hist = np.zeros((num_classes, num_classes))#hist初始化为全零，在这里的hist的形状是[19, 19]

    image_path_list = join(devkit_dir, 'val.txt')#在这里打开记录验证集图片名称的txt
    label_path_list = join(devkit_dir, 'label.txt')#在这里打开记录验证集标签名称的txt
    gt_imgs = open(label_path_list, 'r').read().splitlines()#获得验证集标签名称列表
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]#获得验证集标签路径列表，方便直接读取
    pred_imgs = open(image_path_list, 'r').read().splitlines()#获得验证集图像分割结果名称列表
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]#获得验证集图像分割结果路径列表，方便直接读取

    for ind in range(len(gt_imgs)):#读取每一个（图片-标签）对
        pred = np.array(Image.open(pred_imgs[ind]))#读取一张图像分割结果，转化成numpy数组
        label = np.array(Image.open(gt_imgs[ind]))#读取一张对应的标签，转化成numpy数组
        label = label_mapping(label, mapping)#进行标签映射（因为没有用到全部类别，因此舍弃某些类别），可忽略
        if len(label.flatten()) != len(pred.flatten()):#如果图像分割结果与标签的大小不一样，这张图片就不计算
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)#对一张图片计算19×19的hist矩阵，并累加
        if ind > 0 and ind % 10 == 0:#每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    
    mIoUs = per_class_iu(hist)#计算所有验证集图片的逐类别mIoU值
    for ind_class in range(num_classes):#逐类别输出一下mIoU值
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))#在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    return mIoUs


def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)#执行计算mIoU的函数


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str, help='directory which stores CityScapes val gt images')#设置gt_dir参数，存放验证集分割标签的文件夹
    parser.add_argument('pred_dir', type=str, help='directory which stores CityScapes val pred images')#设置pred_dir参数，存放验证集分割结果的文件夹
    parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')#设置devikit_dir文件夹，里面有记录图片与标签名称及其他信息的txt文件
    args = parser.parse_args()
    main(args)#执行主函数


F:\ProgramFiles\Anaconda3\python.exe E:/Download/PythonProjects/readfilename/readmat.py
<class 'dict'>
dict_keys(['__header__', '__version__', '__globals__', 'colormap'])
dict_values([b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Thu May 12 14:16:07 2016', '1.0', [], array([[0.        , 0.        , 0.        ],
       [0.5       , 0.        , 0.        ],
       [0.99609375, 0.        , 0.        ],
       [0.        , 0.33203125, 0.        ],
       [0.6640625 , 0.        , 0.19921875],
       [0.99609375, 0.33203125, 0.        ],
       [0.        , 0.        , 0.33203125],
       [0.        , 0.46484375, 0.86328125],
       [0.33203125, 0.33203125, 0.        ],
       [0.        , 0.33203125, 0.33203125],
       [0.33203125, 0.19921875, 0.        ],
       [0.203125  , 0.3359375 , 0.5       ],
       [0.        , 0.5       , 0.        ],
       [0.        , 0.        , 0.99609375],
       [0.19921875, 0.6640625 , 0.86328125],
       [0.        , 0.99609375, 0.99609375],
       [0.33203125, 0.99609375, 0.6640625 ],
       [0.6640625 , 0.99609375, 0.33203125],
       [0.99609375, 0.99609375, 0.        ],
       [0.99609375, 0.6640625 , 0.        ]])])

Process finished with exit code 0


factor prob prob_s
[0.33973894 1.56649051 1.36568494 2.84453283 2.76846404 0.85907195
 1.77204006 1.02500099 2.62371924 1.46438956 2.12139959 2.84397328
 2.68099858 1.39502831 2.0552433  2.08102289 2.71828333 2.76395688
 2.8689779  2.94626711]
[6.02257408e-01 6.39380422e-03 4.68240456e-02 6.25033827e-04
 4.04391139e-04 9.66515475e-02 2.03776016e-02 7.44673899e-02
 1.30724730e-03 4.08044507e-02 1.48029708e-02 2.00836026e-03
 3.61471587e-03 4.46218898e-02 1.58529725e-02 1.69662156e-02
 3.79652777e-03 3.75840861e-03 2.21679062e-03 2.24822888e-03]
[0.39964146 0.0475178  0.06185648 0.00370327 0.00517887 0.12782062
 0.03620613 0.0990347  0.00822296 0.05431699 0.0220094  0.00371383
 0.00697903 0.05950366 0.02432731 0.02340655 0.00619749 0.00526885
 0.00324569 0.0018489 ]
