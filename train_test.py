import os
import data_processing_tool as dpt
from datetime import timedelta, date, datetime
from args_parameter import args
from PrepareData import ACCESS_BARRA_v1,ACCESS_BARRA_v2

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
access_rgb_mean= 2.9067910245780248e-05*86400

leading_time=217
args.leading_time_we_use=7
args.ensemble=2

init_date=date(1970, 1, 1)
start_date=date(1990, 1, 2)
end_date=date(2012,12,25) #if 929 is true we should substract 1 day
dates=[start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]



############################################################################################

train_transforms = transforms.Compose([
#     transforms.Resize(IMG_SIZE),
#     transforms.RandomResizedCrop(IMG_SIZE),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(30),
    transforms.ToTensor()
#     transforms.Normalize(IMG_MEAN, IMG_STD)
])

data_set=ACCESS_BARRA_v2(start_date,end_date,transform=train_transforms,args=args)
train_data,val_data=random_split(data_set,[int(len(data_set)*0.8),len(data_set)-int(len(data_set)*0.8)])


print("Dataset statistics:")
print("  ------------------------------")
print("  total | %5d"%len(data_set))
print("  ------------------------------")
print("  train | %5d"%len(train_data))
print("  ------------------------------")
print("  val   | %5d"%len(val_data))

###################################################################################set a the dataLoader
train_dataloders =DataLoader(train_data,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                            num_workers=args.n_threads)
val_dataloders =DataLoader(val_data,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                          num_workers=args.n_threads)
##
def prepare( l, volatile=False):
    device = torch.device('cpu' if args.cpu else 'cuda')
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(_l) for _l in l]

checkpoint = utility.checkpoint(args)
net = model.Model(args, checkpoint).double()
args.lr=0.001
criterion = nn.L1Loss()
optimizer_my = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
# scheduler = optim.lr_scheduler.StepLR(optimizer_my, step_size=7, gamma=0.1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer_my, gamma=0.9)
# torch.optim.lr_scheduler.MultiStepLR(optimizer_my, milestones=[20,80], gamma=0.1)



#training
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




max_error=np.inf
# train_loss=
# print("batch size %d"%args.batch_size)
# print("batch size %d"%args.batch_size)

for e in range(args.epochs):
    #train
    print("start Training for %d"%e)
#     net.train()
    loss=0
    start=time.time()
    print("start Training for start loading data")

    for batch, (lr, hr,_,_) in enumerate(train_dataloders):
        print("start first batch train")

        lr, hr = prepare([lr, hr])
#         optimizer_my.zero_grad()
#         with torch.set_grad_enabled(True):
#             sr = net(lr, 0)
# #         error = criterion(sr[:,:,:,0:403], hr)
#             running_loss =criterion(sr, hr)
#             loss+=running_loss 
#         running_loss.backward()
#         optimizer_my.step()
        print("end first batch train")

        
    #validation
    net.eval()
    start=time.time()
    with torch.no_grad():
        eval_psnr=0
        eval_ssim=0
#         tqdm_val = tqdm(val_dataloders, ncols=80)
        for idx_img, (lr, hr,date,_) in enumerate(val_dataloders):
            print("start first batch validation")

            lr, hr = prepare([lr, hr])
#             sr = net(lr, 0)
#             val_loss=criterion(sr, hr)
#             for ssr,hhr in zip(sr,hr):
#                 eval_psnr+=compare_psnr(ssr[0].cpu().numpy(),hhr[0].cpu().numpy(),data_range=(hhr[0].cpu().max()-hhr[0].cpu().min()).item() )
#                 eval_ssim+=compare_ssim(ssr[0].cpu().numpy(),hhr[0].cpu().numpy(),data_range=(hhr[0].cpu().max()-hhr[0].cpu().min()).item() ) 
            print("end first batch validation")

    print("epoche: %d,time cost %f s, lr: %f, train_loss: %f,validation loss:%f "%(
              e,
              time.time()-start,
              optimizer_my.state_dict()['param_groups'][0]['lr'],
              running_loss.item()/len(train_data),
              val_loss
         ))
        
    if running_loss<max_error:
        print("saving")
        max_error=running_loss
        if not os.path.exists("./model/save/"+args.train_name+"/"):
            os.mkdir("./model/save/"+args.train_name+"/")
#         torch.save(net,"./model/save/"+args.train_name+"/"+str(e)+".pkl")
        print("end Training for %d"%e)





# #training
# max_error=10000
# train_loss=np.inf
# print("batch size %d"%args.batch_size)
# print("batch size %d"%args.batch_size)

# for e in range(args.epochs):
#     #train
#     print("start Training for %d"%e)
#     net.train()
#     loss=0
#     start=time.time()
#     print("start Training for start loading data")

#     for batch, (lr, hr,_,_) in enumerate(train_dataloders):
#         print("start first batch train")

#         lr, hr = prepare([lr, hr])
#         optimizer_my.zero_grad()
#         with torch.set_grad_enabled(True):
#             sr = net(lr, 0)
# #         error = criterion(sr[:,:,:,0:403], hr)
#             running_loss =criterion(sr, hr)
#             loss+=running_loss 
#         running_loss.backward()
#         optimizer_my.step()
#         print("end first batch train")

#         break
        
#     #validation
#     net.eval()
#     start=time.time()
#     with torch.no_grad():
#         eval_psnr=0
#         eval_ssim=0
# #         tqdm_val = tqdm(val_dataloders, ncols=80)
#         for idx_img, (lr, hr,date,_) in enumerate(val_dataloders):
#             print("start first batch validation")

#             lr, hr = prepare([lr, hr])
#             sr = net(lr, 0)
#             val_loss=criterion(sr, hr)
#             for ssr,hhr in zip(sr,hr):
#                 eval_psnr+=compare_psnr(ssr[0].cpu().numpy(),hhr[0].cpu().numpy(),data_range=(hhr[0].cpu().max()-hhr[0].cpu().min()).item() )
#                 eval_ssim+=compare_ssim(ssr[0].cpu().numpy(),hhr[0].cpu().numpy(),data_range=(hhr[0].cpu().max()-hhr[0].cpu().min()).item() ) 
#             print("end first batch validation")
#             break
#     print("epoche: %d,time cost %f s, lr: %f, train_loss: %f,validation loss:%f "%(
#               e,
#               time.time()-start,
#               optimizer_my.state_dict()['param_groups'][0]['lr'],
#               running_loss.item()/len(train_data),
#               val_loss
#          ))
        
#     if train_loss<max_error:
#         print("saving")
#         max_error=train_loss
#         if not os.path.exists("./model/save/"+args.train_name+"/"):
#             os.mkdir("./model/save/"+args.train_name+"/")
#         torch.save(net,"./model/save/"+args.train_name+"/"+str(e)+".pkl")
#         print("end Training for %d"%e)

#     break
        
    



