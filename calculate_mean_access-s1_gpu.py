import os
import data_processing_tool as dpt
from datetime import timedelta, date, datetime
from args_parameter import args
from PrepareData import ACCESS_v1

import torch,os,torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim

# from PIL import Image
import time
import model
import utility
from tqdm import tqdm
import math
import xarray as xr

args.file_ACCESS_dir="/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/"
# args.file_BARRA_dir="/g/data/ma05/BARRA_R/"
# ensemble=['e01','e02']
ensemble=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
access_rgb_mean= 2.9067910245780248e-05*86400

leading_time=217
leading_time_we_use=31


init_date=date(1970, 1, 1)
start_date=date(1990, 1, 2)
end_date=date(1990,12,30) #if 929 is true we should substract 1 day
dates=[start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]
print(access_rgb_mean)

train_transforms = transforms.Compose([
#     transforms.Resize(IMG_SIZE),
#     transforms.RandomResizedCrop(IMG_SIZE),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(30),
    transforms.ToTensor()
#     transforms.Normalize(IMG_MEAN, IMG_STD)
])

data_set=ACCESS_v1(start_date,end_date,transform=train_transforms)

train_dataloders =DataLoader(data_set,
                                        batch_size=args.batch_size,
                                        shuffle=False)
print(len(data_set))
args.cpu=False
# args.pre_train =False
# args.pre_train ="C:/Users/JIA059/climate_v1_csiro/High-resolution-seasonal-climate-forecast_v1_csiro/model/RCAN_BIX"+str(args.scale[0])+".pt"
# "C:/Users/JIA059/climate_v1_csiro/High-resolution-seasonal-climate-forecast_v1_csiro/model"
def prepare( l, volatile=False):
    device = torch.device('cpu' if args.cpu else 'cuda')
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(_l) for _l in l]
#training
training_name="temp01"

num=0
total=0

max_value=0
min_value=10000


start=time.time()
print(len(data_set))
    
for batch, lr in enumerate(train_dataloders):
    lr= prepare([lr])
    num+=lr[0].shape[0]*lr[0].shape[1]*lr[0].shape[2]*args.batch_size
    total+=torch.sum(lr[0])
    
    a=torch.max(lr[0])
    if max_value< a:
        max_value=a
        
    b=torch.min(lr[0])
    
    if min_value>b :
        min_value=b

    print("batch: %d,time cost %f s"%(batch,time.time()-start))
    break

    
print("rgb_mean: "+str(total/num))
print("rgb_mean_real: "+str((total/num)/max_value ))

print("max_value: "+str(max_value))
print("min_value: "+str(min_value))

