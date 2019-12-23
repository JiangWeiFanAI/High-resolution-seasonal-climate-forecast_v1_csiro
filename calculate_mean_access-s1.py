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
num=324*432*217*len(a)
print(num)
leading_time=217
total=0

for filename in tqdm(a):
    data=netDataset(filename)
    total+=np.sum(data["pr"][:])
    
#     print(data["pr"][:].sum())
print("rgb_mean: "+str(total/num))