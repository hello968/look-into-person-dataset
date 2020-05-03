#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import json
from PIL import Image
from os.path import join

#���ǩ��W����H
def fast_hist(a, b, n):#a��ת����һά����ı�ǩ����״(H��W,)��b��ת����һά����ı�ǩ����״(H��W,)��n�������Ŀ��ʵ����������Ϊ19��
    '''
	���Ĵ���
	'''
    k = (a >= 0) & (a < n) #k��һ��һάbool���飬��״(H��W,)��Ŀ�����ҳ���ǩ����Ҫ��������ȥ���˱����� k=0��1
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)#np.bincount�����˴�0��n**2-1��n**2������ÿ�������ֵĴ���������ֵ��״(n, n)


def per_class_iu(hist):#�ֱ�Ϊÿ�������������19�ࣩ����mIoU��hist����״(n, n)
    '''
	���Ĵ���
	'''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))#����ĶԽ����ϵ�ֵ��ɵ�һά����/���������Ԫ��֮�ͣ�����ֵ��״(n,)
#hist.sum(0)=�������  hist.sum(1)�������


#def label_mapping(input, mapping):#��Ҫ����ΪCityScapes��ǩ����ԭ���̫�࣬���������������ת�����㷨��Ҫ����𣨹�19�ࣩ�ͱ�������עΪ255��
#    output = np.copy(input)#�ȸ���һ������ͼ��
#    for ind in range(len(mapping)):
#        output[input == mapping[ind][0]] = mapping[ind][1]#�������ӳ�䣬���յõ��ı�ǩ����֮��0-18��19������255��������
#    return np.array(output, dtype=np.int64)#����ӳ��ı�ǩ
'''
  compute_mIoU����ԭʼ��CityScapesͼ��ָ���֤��Ϊ��������mIoUֵ�ģ����Ը����Լ����ݼ��Ĳ�ͬ���������num_classes���������name_classes������������������Ҫ�ļ���mIoU�Ĵ���֮�⣬�������һЩ��������������������ݶ�ȡ����Ϊԭ������ͼ��ָ�Ǩ�Ʒ���Ĺ�������˻������˱�ǩӳ�����ع�������������߶�����ע�͡������ʹ�õ�ʱ�򣬿��Ժ���ԭ���ߵ����ݶ�ȡ���̣�ֻ��Ҫע�����mIoU��ʱ��ÿ��ͼƬ�ָ������ǩҪ��ԡ���Ҫ����mIoUָ��ļ�����Ĵ��뼴�ɡ�
'''
def compute_mIoU(gt_dir, pred_dir, devkit_dir):#����mIoU�ĺ���
    """
    Compute IoU given the predicted colorized images and 
    """
    with open('/home/ubuntu/DeepLab/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/info.json', 'r') as fp: 
        #��ȡinfo.json�������¼�������Ŀ��������ơ����������ݼ���VOC2011����Ӧ�ظ���josn�ļ���
        info = json.load(fp) 
    num_classes = np.int(info['classes'])#��ȡ�����Ŀ��������20��
    print('Num classes', num_classes)#��ӡһ�������Ŀ
    name_classes = np.array(info['label'], dtype=np.str)#��ȡ�������
    #mapping = np.array(info['label2train'], dtype=np.int)#��ȡ��ǩӳ�䷽ʽ����������и��ӵ�info.json�ļ�
    hist = np.zeros((num_classes, num_classes))#hist��ʼ��Ϊȫ�㣬�������hist����״��[20, 20]
    '''
    ԭ�������н������ӳ�䣬����ͨ��json�ļ�����������Ŀ��������ơ� ��ǩӳ�䷽ʽ��������ֻ��Ҫ��ȡ�����Ŀ��������Ƽ��ɣ����԰�������δ��뽫��д��
    num_classes=20
    print('Num classes', num_classes)
    name_classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car", "cat","chair","cow","diningtable","dog","horse","motobike","person","pottedplant","sheep","sofa","train","tvmonitor"]
    hist = np.zeros((num_classes, num_classes))
    '''
    image_path_list = join(devkit_dir, 'val2.txt')#������򿪼�¼�ָ�ͼƬ���Ƶ�txt
    label_path_list = join(devkit_dir, 'val2.txt')#ground truth���Լ��ķָ���txtһ��
    gt_imgs = open(label_path_list, 'r').read().splitlines()#�����֤����ǩ�����б�
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]#�����֤����ǩ·���б�����ֱ�Ӷ�ȡ
    pred_imgs = open(image_path_list, 'r').read().splitlines()#�����֤��ͼ��ָ��������б�
    pred_imgs = [join(pred_dir, x) for x in pred_imgs]
    #pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]#�����֤��ͼ��ָ���·���б�����ֱ�Ӷ�ȡ


    for ind in range(len(gt_imgs)):#��ȡÿһ����ͼƬ-��ǩ����
        pred = np.array(Image.open(pred_imgs[ind]))#��ȡһ��ͼ��ָ�����ת����numpy����
        label = np.array(Image.open(gt_imgs[ind]))#��ȡһ�Ŷ�Ӧ�ı�ǩ��ת����numpy����
        #print pred.shape
        #print label.shape
        #label = label_mapping(label, mapping)#���б�ǩӳ�䣨��Ϊû���õ�ȫ������������ĳЩ��𣩣��ɺ���
        if len(label.flatten()) != len(pred.flatten()):#���ͼ��ָ������ǩ�Ĵ�С��һ��������ͼƬ�Ͳ�����
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)#��һ��ͼƬ����19��19��hist���󣬲��ۼ�
        if ind > 0 and ind % 10 == 0:#ÿ����10�ž����һ��Ŀǰ�Ѽ����ͼƬ���������ƽ����mIoUֵ
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
            print(per_class_iu(hist))
    
    mIoUs = per_class_iu(hist)#����������֤��ͼƬ�������mIoUֵ
    for ind_class in range(num_classes):#��������һ��mIoUֵ
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))#��������֤��ͼ�������������ƽ����mIoUֵ������ʱ����NaNֵ
    return mIoUs


compute_mIoU('/home/ubuntu/DeepLab/datasets/VOCdevkit/VOC2012/SegmentationClass/',
                 '/home/ubuntu/DeepLab/datasets/VOCdevkit/VOC2012/zhuosetu/',
                 '/home/ubuntu/DeepLab/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/'
                )#ִ�������� ����·���ֱ�Ϊ ��ground truth��,'�Լ���ʵ��ָ���'�����ָ�ͼƬ����txt�ļ���






�ڶ���

import numpy as np
import torch

def iou_mean(pred, target, n_classes = 1):
#n_classes ��the number of classes in your dataset,not including background
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
        predict: input 4D tensor �����ΪB(batch��С)*C(ͨ��)*H*W
        target: label 3D tensor �����ΪB(batch��С)*H*W
        nclass: number of categories (int)
    """
    #��ͨ��ά��ȡ���ֵ��ע��predictΪ�ڶ�������ֵ���������������0��ʼ
    #��ʱpredict��targetһ����ά�Ⱦ�ΪB*H*W����ֵ��Ϊ0��1��2.........
    _, predict = torch.max(predict, 1) 
    mini = 1
    maxi = nclass
    nbins = nclass
    #��predict��target����cpu��ת��Ϊnumpy���飬ͬʱ+1
    #��ʱpredict��target��ֵΪ1��2��......
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1

   #�����ҳ�ȥ������ֻ��1��Ŀ�꣬��nbinsΪ2��
   #�˾��ƺ�û��ʵ�����壿predictֵ���䣬��Ϊtargetֵ������0
   # (target > 0)���صľ�Ϊtrue
    predict = predict * (target > 0).astype(predict.dtype)
    #�󽻼�intersection��ά��B*H*W������������0�Ľ���
    #intersection������>0���ǽ�����Ϊ0�����ڲ�����ֵ����0��1��2��
    intersection = predict * (predict == target)
    # areas of intersection and union
    #����ֱ��ͼ,nbins�����䣬range=(mini, maxi)����ҿ�
    #�����ҳ�ȥ������ֻ��1��Ŀ�꣬��nbinsΪ2��range=(1,2)�����ʾ��������ȵط�Ϊ2�����䣺[1,1.5],[1.5,2]
    #��һ��bins���������ڶ���bins����Ŀ��
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    #����һ��С�ڲ���
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union


import numpy as np
import argparse
import json
from PIL import Image
from os.path import join

#���ǩ��W����H
def fast_hist(a, b, n):#a��ת����һά����ı�ǩ����״(H��W,)��b��ת����һά����ı�ǩ����״(H��W,)��n�������Ŀ��ʵ����������Ϊ19��
    '''
	���Ĵ���
	'''
    k = (a >= 0) & (a < n)#k��һ��һάbool���飬��״(H��W,)��Ŀ�����ҳ���ǩ����Ҫ��������ȥ���˱�����
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)#np.bincount�����˴�0��n**2-1��n**2������ÿ�������ֵĴ���������ֵ��״(n, n)


def per_class_iu(hist):#�ֱ�Ϊÿ�������������19�ࣩ����mIoU��hist����״(n, n)
    '''
	���Ĵ���
	'''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))#����ĶԽ����ϵ�ֵ��ɵ�һά����/���������Ԫ��֮�ͣ�����ֵ��״(n,)


def label_mapping(input, mapping):#��Ҫ����ΪCityScapes��ǩ����ԭ���̫�࣬���������������ת�����㷨��Ҫ����𣨹�19�ࣩ�ͱ�������עΪ255��
    output = np.copy(input)#�ȸ���һ������ͼ��
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]#�������ӳ�䣬���յõ��ı�ǩ����֮��0-18��19������255��������
    return np.array(output, dtype=np.int64)#����ӳ��ı�ǩ
'''
compute_mIoU��������CityScapesͼ��ָ���֤��Ϊ��������mIoUֵ��
�������߸��˹��׵�ԭ�򣬱�������������Ҫ�ļ���mIoU�Ĵ���֮�⣬�������һЩ����������
����������ݶ�ȡ����Ϊԭ������ͼ��ָ�Ǩ�Ʒ���Ĺ�������˻������˱�ǩӳ�����ع�������������߶�����ע�͡�
�����ʹ�õ�ʱ�򣬿��Ժ���ԭ���ߵ����ݶ�ȡ���̣�ֻ��Ҫע�����mIoU��ʱ��ÿ��ͼƬ�ָ������ǩҪ��ԡ�
��Ҫ����mIoUָ��ļ�����Ĵ��뼴�ɡ�
'''
def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):#����mIoU�ĺ���
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp: #��ȡinfo.json�������¼�������Ŀ��������ƣ���ǩӳ�䷽ʽ�ȵȡ�
      info = json.load(fp)
    num_classes = np.int(info['classes'])#��ȡ�����Ŀ��������19�࣬��������и��ӵ�info.json�ļ�
    print('Num classes', num_classes)#��ӡһ�������Ŀ
    name_classes = np.array(info['label'], dtype=np.str)#��ȡ������ƣ���������и��ӵ�info.json�ļ�
    mapping = np.array(info['label2train'], dtype=np.int)#��ȡ��ǩӳ�䷽ʽ����������и��ӵ�info.json�ļ�
    hist = np.zeros((num_classes, num_classes))#hist��ʼ��Ϊȫ�㣬�������hist����״��[19, 19]

    image_path_list = join(devkit_dir, 'val.txt')#������򿪼�¼��֤��ͼƬ���Ƶ�txt
    label_path_list = join(devkit_dir, 'label.txt')#������򿪼�¼��֤����ǩ���Ƶ�txt
    gt_imgs = open(label_path_list, 'r').read().splitlines()#�����֤����ǩ�����б�
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]#�����֤����ǩ·���б�����ֱ�Ӷ�ȡ
    pred_imgs = open(image_path_list, 'r').read().splitlines()#�����֤��ͼ��ָ��������б�
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]#�����֤��ͼ��ָ���·���б�����ֱ�Ӷ�ȡ

    for ind in range(len(gt_imgs)):#��ȡÿһ����ͼƬ-��ǩ����
        pred = np.array(Image.open(pred_imgs[ind]))#��ȡһ��ͼ��ָ�����ת����numpy����
        label = np.array(Image.open(gt_imgs[ind]))#��ȡһ�Ŷ�Ӧ�ı�ǩ��ת����numpy����
        label = label_mapping(label, mapping)#���б�ǩӳ�䣨��Ϊû���õ�ȫ������������ĳЩ��𣩣��ɺ���
        if len(label.flatten()) != len(pred.flatten()):#���ͼ��ָ������ǩ�Ĵ�С��һ��������ͼƬ�Ͳ�����
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)#��һ��ͼƬ����19��19��hist���󣬲��ۼ�
        if ind > 0 and ind % 10 == 0:#ÿ����10�ž����һ��Ŀǰ�Ѽ����ͼƬ���������ƽ����mIoUֵ
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    
    mIoUs = per_class_iu(hist)#����������֤��ͼƬ�������mIoUֵ
    for ind_class in range(num_classes):#��������һ��mIoUֵ
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))#��������֤��ͼ�������������ƽ����mIoUֵ������ʱ����NaNֵ
    return mIoUs


def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)#ִ�м���mIoU�ĺ���


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str, help='directory which stores CityScapes val gt images')#����gt_dir�����������֤���ָ��ǩ���ļ���
    parser.add_argument('pred_dir', type=str, help='directory which stores CityScapes val pred images')#����pred_dir�����������֤���ָ������ļ���
    parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')#����devikit_dir�ļ��У������м�¼ͼƬ���ǩ���Ƽ�������Ϣ��txt�ļ�
    args = parser.parse_args()
    main(args)#ִ��������


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
