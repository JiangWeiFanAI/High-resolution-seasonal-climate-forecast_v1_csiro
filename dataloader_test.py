import os
import data_processing_tool as dpt
from datetime import timedelta, date, datetime
from args_parameter import args
from PrepareData import ACCESS_BARRA_v4

import torch,os,torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,random_split
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import platform

# from PIL import Image
import time
from sklearn.model_selection import StratifiedShuffleSplit
import model
from model import my_model
import utility
from tqdm import tqdm
import math
import xarray as xr
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr,compare_mse
import platform
from torch.autograd import Variable

args.file_ACCESS_dir="E:/climate/access-s1/"
args.file_BARRA_dir="C:/Users/JIA059/barra/"
args.file_DEM_dir="../DEM/"

init_date=date(1970, 1, 1)
start_date=date(1990, 1, 2)
end_date=date(1990,12,10)
args.channels=0
args.dem=True
args.psl=True
args.zg=True
args.tasmax=True
args.tasmin=True

sys = platform.system()
    
if sys == "Windows":
    init_date=date(1970, 1, 1)
    start_date=date(1990, 1, 2)
    end_date=date(1990,12,15) #if 929 is true we should substract 1 day   
#         args.file_ACCESS_dir="H:/climate/access-s1/" 
#         args.file_BARRA_dir="D:/dataset/accum_prcp/"
    args.file_ACCESS_dir="E:/climate/access-s1/"
    args.file_BARRA_dir="C:/Users/JIA059/barra/"
    args.file_DEM_dir="../DEM/"
else:
    args.file_ACCESS_dir_pr="/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/"
    args.file_ACCESS_dir="/g/data/ub7/access-s1/hc/raw_model/atmos/"
    # training_name="temp01"
    args.file_BARRA_dir="/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/"

if args.pr:
    args.channels+=1
if args.zg:
    args.channels+=22
if args.psl:
    args.channels+=1
if args.tasmax:
    args.channels+=1
if args.tasmin:
    args.channels+=1
if args.dem:
    args.channels+=1
access_rgb_mean= 2.9067910245780248e-05*86400

leading_time=217
args.leading_time_we_use=7
args.ensemble=2

class ACCESS_BARRA_v4_test(Dataset):
    '''
    scale is size(hr)=size(lr)*scale
    version_3_documention: compare with ver1, I modify:
    1. access file is created on getitem,the file list is access_date,barra,barra_date,time_leading
      in order to read more data like zg etc. more easier, we change access_filepath to access_date

    2. in ver., norm the every inputs 
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,train=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,_,date_for_BARRA,time_leading=self.filename_list[0]
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        data_exp=dpt.map_aust(data_high,domain=args.domain,xrarray=True)#,domain=domain)
        self.lat=data_exp["lat"]
        self.lon=data_exp["lon"]
        self.shape=(79,94)
        if self.args.dem:
            data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
            self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:
                
                    
                
#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[access_path]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        if self.args.nine2nine and self.args.date_minus_one==1:
            del _files[0]
        return _files

    

        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        access_filename_pr,en,access_date,date_for_BARRA,time_leading=self.filename_list[idx]
#         print(type(date_for_BARRA))
#         low_filename,high_filename,time_leading=self.filename_list[idx]

        lr=dpt.read_access_data(access_filename_pr,idx=time_leading).data[82:144,134:188]
#         lr=dpt.map_aust(lr,domain=self.args.domain,xrarray=False)
        lr=np.expand_dims(dpt.interp_tensor_2d(lr,self.shape),axis=2)

        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        label=dpt.map_aust(data_high,domain=self.args.domain,xrarray=False)#,domain=domain)

        if self.args.zg:
#             access_filename_zg=self.args.file_ACCESS_dir+"zg/daily/"+en+"/"+"da_zg_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            access_filename_zg="../data/da_zg_19900101_e01.nc"

            lr_zg=dpt.read_access_zg(access_filename_zg,idx=time_leading).data[:][83:145,135:188]
            lr_zg=dpt.interp_tensor_3d(lr_zg,self.shape)
        
        if self.args.psl:
            access_filename_psl=self.args.file_ACCESS_dir+"psl/daily/"+en+"/"+"da_psl_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            access_filename_psl="../data/da_psl_19900101_e01.nc"

            lr_psl=dpt.read_access_data(access_filename_psl,var_name="psl",idx=time_leading).data[82:144,134:188]
            lr_psl=dpt.interp_tensor_2d(lr_psl,self.shape)

        if self.args.tasmax:
            access_filename_tasmax=self.args.file_ACCESS_dir+"tasmax/daily/"+en+"/"+"da_tasmax_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            access_filename_tasmax="../data/da_tasmax_19900101_e01.nc"

            lr_tasmax=dpt.read_access_data(access_filename_tasmax,var_name="tasmax",idx=time_leading).data[82:144,134:188]
            lr_tasmax=dpt.interp_tensor_2d(lr_tasmax,self.shape)
            
        if self.args.tasmin:
            access_filename_tasmin=self.args.file_ACCESS_dir+"tasmin/daily/"+en+"/"+"da_tasmin_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            access_filename_tasmin="../data/da_tasmin_19900101_e01.nc"
            lr_tasmin=dpt.read_access_data(access_filename_tasmin,var_name="tasmin",idx=time_leading).data[82:144,134:188]
            lr_tasmin=dpt.interp_tensor_2d(lr_tasmin,self.shape)

            
#         if self.args.dem:
# #             print("add dem data")
#             lr=np.concatenate((lr,np.expand_dims(self.dem_data,axis=2)),axis=2)

            
#         print("end loading one data,time cost %f"%(time.time()-t))


        if self.transform:#channel 数量需要整理！！
            if self.args.channels==27:
                return self.transform(lr*86400),self.transform(self.dem_data),self.transform(lr_psl),self.transform(lr_zg),self.transform(lr_tasmax),self.transform(lr_tasmin),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
            elif self.args.channels==5:
                return self.transform(lr*86400),self.transform(self.dem_data),self.transform(lr_psl),self.transform(lr_tasmax),self.transform(lr_tasmin),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
            if self.args.channels==2:
                return self.transform(lr*86400),self.transform(self.dem_data),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)

        else:
            return lr*86400,label,torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
#         return np.reshape(train_data,(78,100,1))*86400,np.reshape(label,(312,400,1))
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


train_transforms = transforms.Compose([
    transforms.ToTensor()
])

dataset=ACCESS_BARRA_v4_test(
    start_date=date(1990, 1, 2),
    end_date=date(1990, 1, 10),
    transform=train_transforms,
    args=args,
)

train_dataloders =DataLoader(dataset,
                                            batch_size=2,
                                            shuffle=False,
                                num_workers=0)
def prepare( l, volatile=False):
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        if args.precision == 'single': tensor = tensor.float()
        return tensor.to(device)

    return [_prepare(_l) for _l in l]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("start")
max_error=np.inf
for e in range(args.epochs):
    #train
    start=time.time()
    for batch, (pr,dem,psl,zg,tasmax,tasmin, hr,_,_) in enumerate(train_dataloders):
        print(pr.dtype)
        print(dem.dtype)
        print(zg.dtype)
        print(tasmax.dtype)
        print(tasmin.dtype)
        print(hr.dtype)
        
        print("Train for batch %d,data loading time cost %f s"%(batch,start-time.time()))
        start=time.time()
        pr,dem,psl,zg,tasmax,tasmain, hr = prepare([pr,dem,psl,zg,tasmax,tasmin, hr])
        print(pr.dtype)
        print(dem.dtype)
        print(zg.dtype)
        print(tasmax.dtype)
        print(tasmin.dtype)
        print(hr.dtype)
        with torch.set_grad_enabled(True):
            start=time.time()
        break
    break


