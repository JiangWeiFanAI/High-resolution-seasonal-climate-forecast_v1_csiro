import os
import data_processing_tool as dpt
from datetime import timedelta, date, datetime
# import args_parameter as args
from args_parameter import args
import torch,torchvision
import numpy as np

from torch.utils.data import Dataset,random_split
from torchvision import datasets, models, transforms

import time
import xarray as xr
# from sklearn.model_selection import StratifiedShuffleSplit

file_ACCESS_dir="/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/"
file_BARRA_dir="/g/data/ma05/BARRA_R/analysis/acum_proc"
ensemble=['e01','e02']
# ensemble=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']

# leading_time=217
# leading_time_we_use=31


# init_date=date(1970, 1, 1)
# start_date=date(1990, 1, 1)
# end_date=date(1990,12,31) #if 929 is true we should substract 1 day
# dates=[start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

# domain = [111.975, 156.275, -44.525, -9.975]

# domain = [111.975, 156.275, -44.525, -9.975]

file_ACCESS_dir="F:/climate/access-s1/pr/daily/"#"/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/"
file_BARRA_dir="F:/climate/barra/"
class ACCESS_BARRA_v1(Dataset):
    '''
    scale is size(hr)=size(lr)*scale
    version_1_documention: the data we use is raw data that store at NCI
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,train=True,args=args):
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=7

        
#         if regin=="AUS":
#             self.shape=(314,403,1,1)
#             self.domain=[111.975, 156.275, -44.525, -9.975]
#         else:
#             self.shape=(768,1200,1,1)
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir)
        _,date_for_BARRA,time_leading=self.filename_list[0]

        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        data_exp=dpt.map_aust(data_high,domain=args.domain,xrarray=True)#,domain=domain)
        self.lat=data_exp["lat"]
        self.lon=data_exp["lon"]        
        
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
        '''get filename first and generate label '''
        _files = []
        for en in ensemble:
            for date in self.dates:
                filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        path=[access_path]
                        
#                         barra_path=file_BARRA_dir+"/accum_prcp-an-spec-PT0H-BARRA_R-v1-"+((date+timedelta(i)).strftime("%Y%m%d"))
                        barra_date=date+timedelta(i)
#                         self.data_dir+date.strftime('%m')+"/accum_prcp-an-spec-PT0H-BARRA_R-v1-"\
#                         +date.strftime('%Y%m%d')+"T"+enum[i]+"Z.nc"
                        path.append(barra_date)
                        path.append(i)
#                         print(path)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        if args.nine2nine and args.date_minus_one==1:
            del _files[0]
        return _files

    

        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        #read_data filemame[idx]
        access_filename,date_for_BARRA,time_leading=self.filename_list[idx]
#         print(type(date_for_BARRA))
#         low_filename,high_filename,time_leading=self.filename_list[idx]

        data_low=dpt.read_access_data(access_filename,idx=time_leading)
        lr_raw=dpt.map_aust(data_low,domain=args.domain,xrarray=False)
        
#         domain = [train_data.lon.data.min(), train_data.lon.data.max(), train_data.lat.data.min(), train_data.lat.data.max()]
#         print(domain)

        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        label=dpt.map_aust(data_high,domain=args.domain,xrarray=False)#,domain=domain)
        lr=dpt.interp_tensor_2d(lr_raw,(78,100))
        if self.transform:
            return self.transform( np.expand_dims(lr,axis=3)*86400),self.transform(np.expand_dims(label,axis=3)),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
        else:
            return np.expand_dims(lr,axis=3)*86400,np.expand_dims(label,axis=3),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
#         return np.reshape(train_data,(78,100,1))*86400,np.reshape(label,(312,400,1))

    
class ACCESS_v1(Dataset):
    '''
    scale is size(hr)=size(lr)*scale
    version_1_documention: the data we use is raw data that store at NCI
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,train=True,args=args):
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=31

        
#         if regin=="AUS":
#             self.shape=(314,403,1,1)
#             self.domain=[111.975, 156.275, -44.525, -9.975]
#         else:
#             self.shape=(768,1200,1,1)
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir)
        _,date_for_BARRA,time_leading=self.filename_list[0]

#         data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
#         data_exp=dpt.map_aust(data_high,domain=args.domain,xrarray=True)#,domain=domain)
#         self.lat=data_exp["lat"]
#         self.lon=data_exp["lon"]        
#         
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
        '''get filename first and generate label '''
        _files = []
        for en in ensemble:
            for date in self.dates:
                filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        path=[access_path]
                        barra_date=date+timedelta(i)
                        path.append(barra_date)
                        path.append(i)
#                         print(path)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        if args.nine2nine and args.date_minus_one==1:
            del _files[0]
        return _files

    

        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        #read_data filemame[idx]
        access_filename,date_for_BARRA,time_leading=self.filename_list[idx]
        data_low=dpt.read_access_data(access_filename,idx=time_leading)
        lr_raw=dpt.map_aust(data_low,domain=args.domain,xrarray=False)
        
        lr=dpt.interp_tensor_2d(lr_raw,(78,100))
        
        if self.transform:
            return self.transform( np.expand_dims(lr,axis=3)*86400)
        else:
            return np.expand_dims(lr,axis=3)*86400,np.expand_dims(label,axis=3),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
#         return np.reshape(train_data,(78,100,1))*86400,np.reshape(label,(312,400,1))

    


