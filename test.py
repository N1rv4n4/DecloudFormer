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
