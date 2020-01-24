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
from sklearn.model_selection import StratifiedShuffleSplit
import model
import utility
from tqdm import tqdm
import math
import xarray as xr

args.file_ACCESS_dir="/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/"
args.file_BARRA_dir="/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/"

ensemble=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
access_rgb_mean= 2.9067910245780248e-05*86400

leading_time=217
leading_time_we_use=31


init_date=date(1970, 1, 1)
start_date=date(1990, 1, 2)
end_date=date(2018,12,31) #if 929 is true we should substract 1 day
dates=[start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]
print(access_rgb_mean)
###############################################################################
train_transforms = transforms.Compose([
#     transforms.Resize(IMG_SIZE),
#     transforms.RandomResizedCrop(IMG_SIZE),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(30),
    transforms.ToTensor()
#     transforms.Normalize(IMG_MEAN, IMG_STD)
])

data_set=ACCESS_BARRA_v1(start_date,end_date,transform=train_transforms)
train_data,test_data=random_split(data_set,[int(len(data_set)*0.8),len(data_set)-int(len(data_set)*0.8)])
len(train_data)
train_dataloders =DataLoader(train_data,
                                        batch_size=8,
                                        shuffle=False)
test_dataloders =DataLoader(test_data,
                                        batch_size=8,
                                        shuffle=False)



# args.pre_train ="C:/Users/JIA059/climate_v1_csiro/High-resolution-seasonal-climate-forecast_v1_csiro/model/RCAN_BIX"+str(args.scale[0])+".pt"
# "C:/Users/JIA059/climate_v1_csiro/High-resolution-seasonal-climate-forecast_v1_csiro/model"
def prepare( l, volatile=False):
    device = torch.device('cpu' if args.cpu else 'cuda')
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(_l) for _l in l]

checkpoint = utility.checkpoint(args)
net = model.Model(args, checkpoint).double()

criterion = nn.L1Loss()
optimizer_my = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)



#training
training_name="temp01"

max_error=10000
for e in range(args.epochs):
    start=time.time()
    if e % 10 == 0:
        for p in optimizer_my.param_groups:
            p['lr'] *= 0.9

    for batch, (lr, hr,_,_) in enumerate(train_dataloders):
    #     print(batch, (lr.size(), hr.size()))
        lr, hr = prepare([lr, hr])
        optimizer_my.zero_grad()
        sr = net(lr, 0)
        error = criterion(sr[:,:,:,0:403], hr)
        if error<max_error:
            max_error=error
#             torch.save(net,"C:/Users/JIA059/climate_v1_csiro/High-resolution-seasonal-climate-forecast_v1_csiro/model"+str(e)+".pkl")
            if not os.path.exists("./model/save/temp01/"):
                os.mkdir("./model/save/temp01/")
            torch.save(net,"./model/save/"+training_name+"/"+str(e)+".pkl")

        error.backward()
        optimizer_my.step()
        break
#     print("epoche: %d,time cost %f s, lr: %f, error: %f"%(e,time.time()-start,optimizer_my.state_dict()['param_groups'][0]['lr'],error.item()))
        
    break
    


