import cv2
import xarray as xr
from netCDF4 import Dataset, num2date, date2num
from libtiff import TIFF
import os, sys
import numpy as np

from datetime import datetime
from datetime import timedelta

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
# import cartopy.crs as ccrs
from matplotlib import cm
# from mpl_toolkits.basemap import Basemap
import warnings

warnings.filterwarnings("ignore")

levels = {}
levels["hour"]  = [0., 0.2,   1,   5,  10,  20,  30,   40,   60,   80,  100,  150]
levels["day"]   = [0., 0.2,  5, 10,  20,  30,  40,  60,  100,  150,  200,  300]
levels["week"]  = [0., 0.2,  10,  20,  30,  50, 100,  150,  200,  300,  500, 1000]
levels["month"] = [0.,  10,  20,  30,  40,  50, 100,  200,  300,  500, 1000, 1500]
levels["year"]  = [0.,  50, 100, 200, 300, 400, 600, 1000, 1500, 2000, 3000, 5000]
levels["psl"]  = [9000, 100000,  100100, 100300,  100500,  100700,  100900,  101000,  101200,  101400,  101600,  101800]

level["zg"]=[1400, 1411, 1422, 1433, 1444, 1455, 1466, 1477, 1488, 1499, 1510, 1521]

enum={0:"0600",1:"1200",2:"1800",3:"0000",4:"0600"}


prcp_colours_0 = [
                   "#FFFFFF", 
                   '#ffffd9',
                   '#edf8b1',
                   '#c7e9b4',
                   '#7fcdbb',
                   '#41b6c4',
                   '#1d91c0',
                   '#225ea8',
                   '#253494',
                   '#081d58',
                   "#4B0082"]

prcp_colours = [
                   "#FFFFFF", 
                   '#edf8b1',
                   '#c7e9b4',
                   '#7fcdbb',
                   '#41b6c4',
                   '#1d91c0',
                   '#225ea8',
                   '#253494',
                   '#4B0082',
                   "#800080",
                   '#8B0000']

prcp_colormap = matplotlib.colors.ListedColormap(prcp_colours)


def load_2D_netCDF(filename, var_name, lat_name="latitude", lon_name="longitude",xarray=True):
    """This function loads a two dimensional netCDF.
    Give filename as a string.
    Give name of measured variable.
    Give the names of the two spatial coordinates.
    The function returns the variable, and the two dimenstions
    for example : 
    lat_name = "latitude"
    lon_name = "longitude"
    data, lat, lon=load_2D_netCDF("./data/accum_prcp-an-spec-PT0H-BARRA_R-v1-20150207T0000Z.nc" ,'accum_prcp',lat_name,lon_name)
    
    return ndarray
    """
    data = Dataset(filename, 'r')
    var = data[var_name][:]
    lats = data[lat_name][:]
    lons = data[lon_name][:]
    data.close()
    if xarray:
        return xr.DataArray(var,coords=[lats,lons],dims=["lat","lon"])
    else:
        return var, lats, lons


def load_3D_netCDF(filename, var_name="pr", lat_name="lat", lon_name="lon",idx=0):
    data = Dataset(filename, 'r')
#     print(data)# lat(324), lon(432)
    var = data[var_name][idx]#!!!!!!!!!!!!!!!!!!!!!!!!!!!
    lats = data[lat_name][:]
    lons = data[lon_name][:]
    data.close()
    return xr.DataArray(var,coords=[lats,lons],dims=["lat","lon"])

def read_access_data(filename, var_name="pr", lat_name="lat", lon_name="lon",idx=0):
    data = Dataset(filename, 'r')
    var = data[var_name][idx]
    lats = data[lat_name][:]
    lons = data[lon_name][:]
    data.close()
    return xr.DataArray(var,coords=[lats,lons],dims=["lat","lon"])

def read_access_zg(filename, var_name="zg", lat_name="lat", lon_name="lon",idx=0,level="850"):
    data = Dataset(filename, 'r')
    var = data[var_name][idx][-3]
#     level=data["z1_p_level"][:]
    lats = data[lat_name][:]
    lons = data[lon_name][:]
    data.close()
    return xr.DataArray(var,coords=[lats,lons],dims=["lat","lon"])



def read_barra_data_an(root_dir,date_time,nine2nine=False):
    shape=(768,1200)
    enum={0:"1200",1:"1800",2:"0000",3:"0600"}
    daily=np.zeros(shape)
    if nine2nine:
        for i in range(4):
            if i<=2:
                filename=root_dir+date_time.strftime("%Y/%m/")+"accum_prcp-an-spec-PT0H-BARRA_R-v1.1-"+date_time.strftime("%Y%m%d")+"T"+enum[i]+"Z.nc"
            else:
                filename=root_dir+date_time.strftime("%Y/%m/")+"accum_prcp-an-spec-PT0H-BARRA_R-v1.1-"+(date_time+timedelta(1)).strftime("%Y%m%d")+"T"+enum[i]+"Z.nc"
                
            if not os.path.exists(filename):
                print("Error: "+filename+" not found")
#                 raise Exception("Error: "+filename+" not found")
                return

                
            var,lats,lons=load_2D_netCDF(filename,'accum_prcp',xarray=False)
            daily+=var
        return xr.DataArray(daily,coords=[lats,lons],dims=["lat","lon"])
    
    
    else:
        for i in range(4):
            filename=root_dir+date_time.strftime("%Y/%m/")+"accum_prcp-an-spec-PT0H-BARRA_R-v1.1-"+date_time.strftime("%Y%m%d")+"T"+enum[i]+"Z.nc"
            var,lats,lons=load_2D_netCDF(filename,'accum_prcp',xarray=False)
            daily+=var
        return xr.DataArray(daily,coords=[lats,lons],dims=["lat","lon"])



def get_file(root_dir,date_we_use,i):
    """
    some file name do not have v1, but they do have v1.1
    like :accum_prcp-fc-spec-PT1H-BARRA_R-v1.1-20100120T1800Z.sub.nc
    """
    filename=root_dir+date_we_use.strftime("%Y/%m/")+"accum_prcp-fc-spec-PT1H-BARRA_R-v1-"+date_we_use.strftime("%Y%m%d")+"T"+enum[i]+"Z.sub.nc"
    if not os.path.exists(filename):
        filename=root_dir+date_we_use.strftime("%Y/%m/")+"accum_prcp-fc-spec-PT1H-BARRA_R-v1.1-"+date_we_use.strftime("%Y%m%d")+"T"+enum[i]+"Z.sub.nc"
    return filename


def read_barra_data_fc(root_dir,date_time,nine2nine=True,date_minus_one=1):#argse
    """
    How to use: 
    
    root_dir usually is /g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/
   date time is the precipitation of that day you want to retrival
   nine2nine: boolean. the precipitation you retrival whether recording from 9am to 9pm. if nagative, it is accumulate from 0am to 24 pm
   date_minus_one: base on date_time, if 1, precipitation from (date_time-1)'s 9am to date_time's 9pm is regarded as the precipitation of the date_time. Example are followed: the following files are used to calculate the precipitation of 1990/01/31
          filename                                 subscript times
    accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900130T0600Z.sub.nc          4-6
    accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900130T1200Z.sub.nc          1-6
    accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900130T1800Z.sub.nc          1-6
    accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900131T0000Z.sub.nc          1-6
    accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900131T0600Z.sub.nc          1-3
    """
    shape=(768,1200)
    
    daily=np.zeros(shape)
    
    if nine2nine:
        enum={0:"0600",1:"1200",2:"1800",3:"0000",4:"0600"}
        for i in range(5):
            if i==0:
                date_we_use=date_time-timedelta(date_minus_one)
                filename=get_file(root_dir,date_we_use,i)
                
                dataset=Dataset(filename)
                daily+=np.sum(dataset["accum_prcp"][3:6],axis=0)
#                 print(var.shape)
            elif i==4:
                date_we_use=date_time-timedelta(date_minus_one-1)
                filename=get_file(root_dir,date_we_use,i)
                dataset=Dataset(filename)
                daily+=np.sum(dataset["accum_prcp"][0:3],axis=0)
                lats=dataset["latitude"][:]
                lons=dataset["longitude"][:]
#                 print(dataset["accum_prcp"][0:3].shape)
            elif i==3:
                date_we_use=date_time-timedelta(date_minus_one-1)
                filename=get_file(root_dir,date_we_use,i)
                dataset=Dataset(filename)
                daily+=np.sum(dataset["accum_prcp"][:],axis=0)       
            else:
                date_we_use=date_time-timedelta(date_minus_one)
                filename=get_file(root_dir,date_we_use,i)
                dataset=Dataset(filename)
                daily+=np.sum(dataset["accum_prcp"][:],axis=0)
            dataset.close()



        return xr.DataArray(daily,coords=[lats,lons],dims=["lat","lon"])
    
    
    else:
        enum={0:"0600",1:"1200",2:"1800",3:"0000"}
        for i in range(4):
            filename=root_dir+date_time.strftime("%Y/%m/")+"accum_prcp-fc-spec-PT1H-BARRA_R-v1-"+date_time.strftime("%Y%m%d")+"T"+enum[i]+"Z.sub.nc"

            dataset=Dataset(filename)
            daily+=np.sum(dataset["accum_prcp"][:],axis=0)

            lats=dataset["latitude"][:]
            lons=dataset["longitude"][:]

        return xr.DataArray(daily,coords=[lats,lons],dims=["lat","lon"])
    


def read_dem(filename):
    tif = TIFF.open(filename,mode='r')
    stack = []
    for img in list(tif.iter_images()):
        stack.append(img)
        
    dem_np=np.array(stack)
    dem_np=dem_np.transpose(1,2,0)
    return dem_np

def add_lat_lon(data,domian=[112.9, 154.00, -43.7425, -9.0],xarray=True):
    "data: is the something you want to add lat and lon, with first demenstion is lat,second dimention is lon,domain is DEM domain "
    new_lon=np.linspace(domian[0],domian[1],data.shape[1])
    new_lat=np.linspace(domian[2],domian[3],data.shape[0])
    if xarray:
        return xr.DataArray(data[:,:,0],coords=[new_lat,new_lon],dims=["lat","lon"])
    else:
        return data,new_lat,new_lon



def map_aust_old(data, lat=None, lon=None,domain = [111.85, 156.275, -44.35, -9.975],xrarray=True):
    '''
    domain=[111.975, 156.275, -44.525, -9.975]
    domain = [111.85, 156.275, -44.35, -9.975]for can be divide by 4
    xarray boolean :the out put data is xrray or not
    '''
    if str(type(data))=="<class 'xarray.core.dataarray.DataArray'>":
        da=data.data
        lat=data.lat.data
        lon=data.lon.data
    else:
        da=data
        
#     if domain==None:
#         domain = [111.85, 156.275, -44.35, -9.975]
    a = np.logical_and(lon>=domain[0], lon<=domain[1])
    b = np.logical_and(lat>=domain[2], lat<=domain[3])
    da=da[b,:][:,a].copy()
    llons, llats=lon[a], lat[b] # 将维度按照 x,y 横向竖向
    if str(type(data))=="<class 'xarray.core.dataarray.DataArray'>" and xrarray:
        return xr.DataArray(da,coords=[llats,llons],dims=["lat","lon"])
    else:
        return da
        
    
    return da,llats,llons


def draw_aus_psl(data, domain = [111.85, 155.875, -44.35, -9.975], level="day" ,titles_on = True, title = "BARRA-R precipitation", colormap = prcp_colormap, cmap_label = "Precipitation (mm)",save=False,path="./"):
    """ basema_ploting .py
This function takes a 2D data set of a variable from BARRA and maps the data on miller projection. 
The map default span is longitude between 111E and 155E, and the span for latitudes is -45 to -30, this is SE Australia. 
The colour scale is YlGnBu at 11 levels. 
The levels specifed are suitable for annual rainfall totals for SE Australia. 
From the BARRA average netCDF, the mean prcp should be multiplied by 24*365
"""
#    lats.sort() #this doesn't do anything for BARRA
#    lons.sort() #this doesn't do anything for BARRA
#     domain = [111.975, 156.275, -44.525, -9.975]#awap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from mpl_toolkits.basemap import Basemap
    fig=plt.figure()
#     level=levels[level]
    map = Basemap(projection = "mill", llcrnrlon = domain[0], llcrnrlat = domain[2], urcrnrlon = domain[1], urcrnrlat = domain[3], resolution = 'l')
    map.drawcoastlines()
    map.drawmapboundary()
    map.drawparallels(np.arange(-90., 120., 5.),labels=[1,0,0,0])
    map.drawmeridians(np.arange(-180.,180., 5.),labels=[0,0,0,1])
    llons, llats = np.meshgrid(data["lon"].values, data["lat"].values) # 将维度按照 x,y 横向竖向
#     print(lon.shape,llons.shape)
    x,y = map(llons,llats)
#     print(x.shape,y.shape)
    
    norm = BoundaryNorm(level, len(level)-1)
    cs = map.pcolormesh(x, y, data, norm = norm, cmap = colormap) 
#     cs = map.pcolormesh(x, y, data, cmap = prcp_colormap) 
    
    if titles_on:
        # label with title, latitude, longitude, and colormap
        
        plt.title(title)
        plt.xlabel("\n\nLongitude")
        plt.ylabel("Latitude\n\n")
        cbar = plt.colorbar(ticks = level[:-1], shrink = 0.8, extend = "max")
        cbar.ax.set_ylabel(cmap_label)
        cbar.ax.set_xticklabels(level)
    if save:
        plt.savefig(path+title)
    else:
        plt.show()
    plt.cla()
    plt.close("all")
    return



def draw_aus_old(data, domain = [111.85, 155.875, -44.35, -9.975], level="day" ,titles_on = True, title = "BARRA-R precipitation", colormap = prcp_colormap, cmap_label = "Precipitation (mm)",save=False,path="./"):
    """ basema_ploting .py
This function takes a 2D data set of a variable from BARRA and maps the data on miller projection. 
The map default span is longitude between 111E and 155E, and the span for latitudes is -45 to -30, this is SE Australia. 
The colour scale is YlGnBu at 11 levels. 
The levels specifed are suitable for annual rainfall totals for SE Australia. 
From the BARRA average netCDF, the mean prcp should be multiplied by 24*365
"""
#    lats.sort() #this doesn't do anything for BARRA
#    lons.sort() #this doesn't do anything for BARRA
#     domain = [111.975, 156.275, -44.525, -9.975]#awap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from mpl_toolkits.basemap import Basemap
    fig=plt.figure()
    level=levels[level]
    map = Basemap(projection = "mill", llcrnrlon = domain[0], llcrnrlat = domain[2], urcrnrlon = domain[1], urcrnrlat = domain[3], resolution = 'l')
    map.drawcoastlines()
    map.drawmapboundary()
    map.drawparallels(np.arange(-90., 120., 5.),labels=[1,0,0,0])
    map.drawmeridians(np.arange(-180.,180., 5.),labels=[0,0,0,1])
    llons, llats = np.meshgrid(data["lon"].values, data["lat"].values) # 将维度按照 x,y 横向竖向
#     print(lon.shape,llons.shape)
    x,y = map(llons,llats)
#     print(x.shape,y.shape)
    
    norm = BoundaryNorm(level, len(level)-1)
    cs = map.pcolormesh(x, y, data, norm = norm, cmap = colormap) 
#     cs = map.pcolormesh(x, y, data, cmap = prcp_colormap) 
    
    if titles_on:
        # label with title, latitude, longitude, and colormap
        
        plt.title(title)
        plt.xlabel("\n\nLongitude")
        plt.ylabel("Latitude\n\n")
        cbar = plt.colorbar(ticks = level[:-1], shrink = 0.8, extend = "max")
        cbar.ax.set_ylabel(cmap_label)
        cbar.ax.set_xticklabels(level)
    if save:
        plt.savefig(path+title)
    else:
        plt.show()
    plt.cla()
    plt.close("all")
    return


def map_aust(data, lat=None, lon=None,data_name="pr",domain = [111.85, 156.275, -44.35, -9.975],xrarray=True):
    '''
    domain=[111.975, 156.275, -44.525, -9.975]
    domain = [111.85, 156.275, -44.35, -9.975]for can be divide by 4
    '''
    if data_name=="pr":
        if str(type(data))=="<class 'xarray.core.dataarray.DataArray'>":
            da=data.data
            lat=data.lat.data
            lon=data.lon.data
        else:
            da=data

    #     if domain==None:
    #         domain = [111.85, 156.275, -44.35, -9.975]
        a = np.logical_and(lon>=domain[0], lon<=domain[1])
        b = np.logical_and(lat>=domain[2], lat<=domain[3])
        da=da[b,:][:,a]#.copy()
        llons, llats=lon[a], lat[b] # 将维度按照 x,y 横向竖向
        if str(type(data))=="<class 'xarray.core.dataarray.DataArray'>" and xrarray:
            return xr.DataArray(da,coords=[llats,llons],dims=["lat","lon"])
        else:
            return da


        return da,llats,llons
    else:
        if str(type(data))=="<class 'xarray.core.dataarray.DataArray'>":
            da=data.data
            level=data.level
            lat=data.lat.data
            lon=data.lon.data
        else:
            da=data
        a = np.logical_and(lon>=domain[0], lon<=domain[1])
        b = np.logical_and(lat>=domain[2], lat<=domain[3])
        da=da[:][:,b][:,:,a]#.copy()
        llons, llats=lon[a], lat[b] # 将维度按照 x,y 横向竖向
        if str(type(data))=="<class 'xarray.core.dataarray.DataArray'>" and xrarray:
            return xr.DataArray(da,coords=[level,llats,llons],dims=["level","lat","lon"])
        else:
            return da


        return da,llats,llons

    
def map_aust_return_dim(data, lat=None, lon=None,data_name="pr",domain = [111.85, 156.275, -44.35, -9.975],xrarray=True):
    '''
    domain=[111.975, 156.275, -44.525, -9.975]
    domain = [111.85, 156.275, -44.35, -9.975]for can be divide by 4
    '''
    if data_name=="pr":
        if str(type(data))=="<class 'xarray.core.dataarray.DataArray'>":
            da=data.data
            lat=data.lat.data
            lon=data.lon.data
        else:
            da=data

    #     if domain==None:
    #         domain = [111.85, 156.275, -44.35, -9.975]
        a = np.logical_and(lon>=domain[0], lon<=domain[1])
        b = np.logical_and(lat>=domain[2], lat<=domain[3])
        da=da[b,:][:,a]#.copy()
        llons, llats=lon[a], lat[b] # 将维度按照 x,y 横向竖向
        if str(type(data))=="<class 'xarray.core.dataarray.DataArray'>" and xrarray:
            return xr.DataArray(da,coords=[llats,llons],dims=["lat","lon"])
        else:
            return da,a,b


        return da,llats,llons
    else:
        if str(type(data))=="<class 'xarray.core.dataarray.DataArray'>":
            da=data.data
            level=data.level
            lat=data.lat.data
            lon=data.lon.data
        else:
            da=data
        a = np.logical_and(lon>=domain[0], lon<=domain[1])
        b = np.logical_and(lat>=domain[2], lat<=domain[3])
        da=da[:][:,b][:,:,a]#.copy()
        llons, llats=lon[a], lat[b] # 将维度按照 x,y 横向竖向
        if str(type(data))=="<class 'xarray.core.dataarray.DataArray'>" and xrarray:
            return xr.DataArray(da,coords=[level,llats,llons],dims=["level","lat","lon"])
        else:
            return da,a,b


        return da,llats,llons    
    



def interp_dim_scale(x, scale,linspace=True):
    '''get the corresponding lat and lon'''
    x0, xlast = x[0], x[-1]
    size=x.shape[0]*scale
    if linspace:
        y = np.linspace(x0,xlast,size)
    else:
        step = (x[1]-x[0])/scale
        y = np.arange(x0, xlast, step)
    return y



def interp_tensor_2d(X, size, fill=True):
    if fill:
        X[np.isnan(X)]=0
    scaled_tensor = cv2.resize(X, (size[1], size[0]),interpolation=cv2.INTER_CUBIC)
    return scaled_tensor

def interp_tensor_3d(X, size, fill=True):
    """
    hypothesis:
     dimensions is level,lat,lon(special design for zg)
    """
    if fill:
        X[np.isnan(X)]=0# if there is an exeption, ensure that x is numpy not xrarray
        
#     print(np.swapaxes(X,2,0).shape)
    scaled_tensor = cv2.resize(np.swapaxes(X,2,0), (size[0], size[1]),interpolation=cv2.INTER_CUBIC)
    return np.swapaxes(scaled_tensor,1,0)

def interp_da_2d_scale(da, scale):
    '''
    da is xarray
    Assume da is of dimensions ('lat', 'lon')
    single data input
    and return a xr array
    '''
    tensor = da.values
    # interpolate lat and lons
    latnew = interp_dim_scale(da[da.dims[0]].values, scale)
    lonnew = interp_dim_scale(da[da.dims[1]].values, scale)


    # lets store our interpolated data
    scaled_tensor = interp_tensor_2d(tensor, (latnew.shape[0],lonnew.shape[0]), fill=True)
    if latnew.shape[0] != scaled_tensor.shape[0]:
        raise ValueError("New shape is shitty")
    # intialize a new dataarray
    return xr.DataArray(scaled_tensor, coords=[latnew, lonnew],
                 dims=da.dims)

def interp_dim_shape(x, shape,linspace=True):
    '''get the corresponding lat and lon'''
    x0, xlast = x[0], x[-1]
    size=shape
    if linspace:
        y = np.linspace(x0,xlast,size)
    else:
        step = (x[1]-x[0])/scale
        y = np.arange(x0, xlast, step)
    return y

def interp_da_2d_shape(da, shape):
    '''
    da is xarray
    Assume da is of dimensions ('lat', 'lon')
    single data input
    and return a xr array
    '''
    tensor = da.values
    # interpolate lat and lons
    latnew = interp_dim_shape(da[da.dims[0]].values, shape[0])
    lonnew = interp_dim_shape(da[da.dims[1]].values, shape[1])


    # lets store our interpolated data
    scaled_tensor = interp_tensor_2d(tensor, (latnew.shape[0],lonnew.shape[0]), fill=True)
    if latnew.shape[0] != scaled_tensor.shape[0]:
        raise ValueError("New shape is shitty")
    # intialize a new dataarray
    return xr.DataArray(scaled_tensor, coords=[latnew, lonnew],
                 dims=da.dims)

