from osgeo import gdal
import numpy as np
import cv2



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count
            

# def read_img(filename):
#     img = cv2.imread(filename)
#     return img[:, :, ::-1].astype('float32') / 255.0

def read_img(path):
    """ 
    根据文件名使用不同的方式读取数据 
    """
    img=None
     
    if 'tif' in path :
        img_gdal=gdal.Open(path)
        img = img_gdal.ReadAsArray()
    
    if 'jpg' in path or 'jpeg' in path or 'png' in path:
        img=cv2.imread(path,cv2.IMREAD_UNCHANGED)
        if len(img.shape)>2:
            img=img.transpose((2,0,1))
        else:
            img=img[np.newaxis,...]
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

# def write_img(filename, img):
#     img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
#     cv2.imwrite(filename, img)

def write_img(data,path):
    if 'tif' in path :
        create_geotiff(path, data)
    if 'jpg' in path or 'jpeg' in path or 'png' in path or 'bmp' in path: 
        cv2.imwrite(path,data)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0]).copy()



# 创建GeoTIFF文件
def create_geotiff(output_path, data):
    driver = gdal.GetDriverByName("JPEG")
    if driver is None:
        raise Exception("无法获取GTiff驱动程序")

    # 创建输出数据集
    dataset = driver.Create(output_path, data.shape[2], data.shape[1], data.shape[0], gdal.GDT_UInt16)
    if dataset is None:
        raise Exception("无法创建输出数据集")

    # 写入数据到数据集
    # dataset.GetRasterBand(1).WriteArray(data)
    for k in range(data.shape[0]):
        dataset.GetRasterBand(k+1).WriteArray(data[k,:,:])
    dataset.FlushCache()
    del dataset