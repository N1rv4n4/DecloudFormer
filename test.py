import os
from osgeo import gdal
import numpy as np
import math
from math import floor,ceil
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as tfs
from copy import deepcopy
from scipy.spatial import distance
import sys
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


def read_img(path):

    if 'tif' in path :
        img_gdal=gdal.Open(path)

        img = img_gdal.ReadAsArray()
        if len(img.shape)==2:
            img=img[np.newaxis,...]
        #img=img.transpose((1,2,0))
    
    if 'jpg' in path or 'jpeg' in path or 'png' in path:
        img=cv2.imread(path).transpose([2,0,1])
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img is None:
        raise Exception("打开图像失败")
    
    return img

def write_img_8n16(data,path):
    cv2.imwrite(path,data.transpose([1,2,0]).astype(np.uint8))

def img2patch(image, block_size):

    if len(image.shape)==2:
        ex_name='png'
        image=image[:,:,np.newaxis]


    height = image.shape[1]
    width = image.shape[2]
    band = image.shape[0]

    
    area_w_begin=0
    area_h_begin=0
    area_w_end=width
    area_h_end=height  
    

    #计算图像需要分割为多少图像块
    image_height_block_num = ceil((area_h_end-area_h_begin)/block_size)
    image_width_block_num = ceil((area_w_end-area_w_begin)/block_size)

    # 读取图像数据
    image_array=image[:,area_h_begin:area_h_end,area_w_begin:area_w_end]

    patch_list=[]
    for h in range(image_height_block_num):
        for w in range(image_width_block_num):
            # 初始化输出块为block_size大小的全0矩阵
            test_patch = np.zeros((band,block_size, block_size))

            # 计算每个图像块宽高的起始和结尾坐标
            block_w_begin=w*block_size
            block_h_begin=h*block_size
            block_w_end=(w+1)*block_size
            block_h_end=(h+1)*block_size
            
            # 考虑边界情况
            if h==image_height_block_num-1:
                block_h_end=area_h_end
            if w==image_width_block_num-1:
                block_w_end=area_w_end

            #若超出边界，则选择边界
            if block_w_end > image_array.shape[2]:
                block_w_end = image_array.shape[2]
            if block_h_end > image_array.shape[1]:
                block_h_end = image_array.shape[1]

            # 计算图像块宽高
            block_w=block_w_end-block_w_begin
            block_h=block_h_end-block_h_begin

            # 当前图像块
            test_patch[:,:block_h,:block_w]=image_array[:,block_h_begin:block_h_end,block_w_begin:block_w_end]
            patch_list.append(test_patch)
    img_shape=image_array.shape
    return patch_list, img_shape

def patch2img(patch_list,img_shape,patch_size):
    if len(img_shape)==2:
        img_shape=np.array([1,img_shape[0],img_shape[1]])
        result=np.zeros((h_num*patch_size,w_num*patch_size),dtype=np.float32)
    h_num=math.ceil(img_shape[1]/patch_size)
    w_num=math.ceil(img_shape[2]/patch_size) 
    result=np.zeros((img_shape[0],h_num*patch_size,w_num*patch_size),dtype=np.float32)

    list_index=0
    for i in range(h_num):
        for j in range(w_num):
            y_off=i*patch_size
            x_off=j*patch_size
            y_end=y_off + patch_size
            x_end=x_off + patch_size
            pred = patch_list[list_index]

            if pred.shape[0] == 1:
                pred=pred[0]

            result[:,y_off:y_end,x_off:x_end]=pred

            list_index=list_index+1
    result=result[:,:img_shape[1],:img_shape[2]]
    return result

def preprocess_data_decloud_8bit_tf(img_data):
    
    haze= tfs.Compose([
        tfs.ToTensor()
    ])(img_data)[None,::]
    haze = haze*2-1
    return haze

def postprocess_data_decloud_8bit_tf(img_numpy):
    img_tif = (torch.squeeze(img_numpy.clamp(-1,1)).cpu().detach().numpy()+1)/2
    img_tif=np.clip(img_tif*255,0,255).astype(np.uint8)
    return img_tif


# 主函数
def main():
    img_path = '/home/server4/lmk/database/decloud_large/test6/'
    save_path = '/home/server4/lmk/database/decloud_large/result6/out_decloudformer_test/'
    model_path= '/home/server4/lmk/Decloudformer/decloudformer-b.pth'
    net=torch.load(model_path)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    net=net.to(device)
    net.eval()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_list = os.listdir(img_path)
    imgs_len=len(img_list)
    
    for idx in range(imgs_len):
        cloud_image=read_img(os.path.join(img_path, img_list[idx]))#.astype(np.float32)             
        image_for_dec = cloud_image.transpose(1,2,0)  # chw to hwc
        image_for_dec= preprocess_data_decloud_8bit_tf(image_for_dec)  
        output = net(image_for_dec.cuda())
        output_arr=postprocess_data_decloud_8bit_tf(output)    
        img_decloud=output_arr.astype(np.uint8)
        write_img_8n16(img_decloud, os.path.join(save_path, img_list[idx]))


if __name__ == "__main__":
    main()
