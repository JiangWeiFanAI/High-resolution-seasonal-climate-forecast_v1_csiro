import os
import data_processing_tool as dpt
from datetime import timedelta, date, datetime
from args_parameter import args
from PrepareData import ACCESS_BARRA_v3

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
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr,compare_mse
import platform 

def main():
    
    sys = platform.system()
    if sys == "Windows":
        args.file_ACCESS_dir="E:/climate/access-s1/"
        args.file_BARRA_dir="C:/Users/JIA059/barra/"
        args.file_DEM_dir="../DEM/"
    else:
        args.file_ACCESS_dir_pr="/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/"
        args.file_ACCESS_dir="/g/data/ub7/access-s1/hc/raw_model/atmos/"
        # training_name="temp01"
        args.file_BARRA_dir="/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/"

    args.channels=0
    if args.pr:
        args.channels+=1
    if args.zg:
        args.channels+=1
    if args.psl:
        args.channels+=1
    if args.tasmax:
        args.channels+=1
    if args.tasmin:
        args.channels+=1
    print(args.dem)
    if args.dem:
        args.channels+=1
    access_rgb_mean= 2.9067910245780248e-05*86400

    leading_time=217
    args.leading_time_we_use=7
    args.ensemble=2

    init_date=date(1970, 1, 1)
    start_date=date(1990, 1, 2)
    end_date=date(2012,12,25) #if 929 is true we should substract 1 day
    print(access_rgb_mean)

    print("training statistics:")
    print("  ------------------------------")
    print("  trainning name  |  %s"%args.train_name)
    print("  ------------------------------")
    print("  num of channels | %5d"%args.channels)
    print("  ------------------------------")
    print("  num of threads  | %5d"%args.n_threads)
    print("  ------------------------------")
    print("  batch_size     | %5d"%args.batch_size)
    print("  ------------------------------")
    print("  using cpu only？ | %5d"%args.cpu)
    checkpoint = utility.checkpoint(args)
    net = model.Model(args, checkpoint).double()

            
            
if __name__=='__main__':
    main()



