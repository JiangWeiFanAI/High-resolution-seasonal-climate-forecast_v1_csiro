# import data_processing_tool as dpt
from netCDF4 import Dataset as netDataset
import os
from datetime import timedelta, date, datetime
from tqdm import tqdm
import time
import numpy as np
 



file_access_dir="/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/"
ensemble=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
# ensemble=['e01','e02']

init_date=date(1970, 1, 1)
start_date=date(1990, 1, 1)
end_date=date(2012,12,31)
leading_time=217
leading_time_we_use=31
dates=[start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

def get_filename(rootdir):
    '''get filename first and generate label '''
    _files = []
    for en in ensemble:
        for date in dates:
            filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
            access_path=file_access_dir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
            if os.path.exists(access_path):
                path=access_path
                _files.append(path)
    return _files


a=get_filename(file_access_dir)
print("the length of dataset: "+str(len(a)))
num=0
total=0

max_value=0
min_value=10000
shape=[]
for filename in tqdm(a):
    
    data=netDataset(filename)
    num+=data["pr"][:].shape[0]*data["pr"][:].shape[1]*data["pr"][:].shape[2]
    total+=np.sum(data["pr"][:])*86400
    if max_value< (np.max(data["pr"][:])*86400):
        max_value=np.max(data["pr"][:])*86400
    if min_value> np.min(np.max(data["pr"][:])*86400):
        min_value=np.min(np.max(data["pr"][:])*86400)
    print("\n")
#     print(data["pr"][:].sum())
print("rgb_mean: "+str(total/num))
print("rgb_mean_real: "+str((total/num)/max_value ))

print("max_value: "+str(max_value))
print("min_value: "+str(min_value))
