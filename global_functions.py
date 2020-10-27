from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib import cm
import pandas as pd
import numpy as np
import colorcet as cc
import geopandas
import netCDF4


## COLORS DEFINITION

# set the colormap and centre the colorbar
class MidpointNormalize(Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def define_cmap(hydro_var_name,hydro_basin,year,month):
    ''' called in each basin, for each hydrological variable, at a given month'''
    vmin=hydro_basin['{} {}-{}-15'.format(hydro_var_name,year,month)].min()
    vmax=hydro_basin['{} {}-{}-15'.format(hydro_var_name,year,month)].max()

    if hydro_var_name=='TWS':
        norm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
        cmap='RdBu_r'
    elif hydro_var_name=='P':
        norm = Normalize(vmin=0, vmax=vmax)
        cmap='Blues'
    elif hydro_var_name=='R':
        norm = Normalize(vmin=0, vmax=vmax)
        cmap='Purples'
    else: # ET
        norm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
        cmap='BrBG'
    return norm,cmap


def define_cmap_perf(metric,discrete=True):
    if metric=='NSE':
        if discrete:
            cmap=cm.YlGn
            bounds = [0,0.2,0.5,0.65,0.85,1]
            norm = BoundaryNorm(bounds, cmap.N)
        else:
            top = cm.get_cmap('YlGn', 128)
            newcolors = np.vstack((np.ones((128,4)),
                           top(np.linspace(0, 1, 128))))
            cmap = ListedColormap(newcolors, name='WhiteGreen')
            norm = MidpointNormalize(midpoint=0, vmin=0,vmax=1)
    if metric=='PBIAS':
        if discrete:
            bottom = cm.get_cmap('Blues', 128)
            top = cm.get_cmap('Oranges_r',128)
            newcolors = np.vstack((bottom(np.linspace(0, 1, 128)),
                           top(np.linspace(0, 1, 128))))
            cmap = ListedColormap(newcolors, name='BlueOrange')

            bounds = [-0.6,-0.4,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.4,0.6]
            norm = BoundaryNorm(bounds, cmap.N)
        else:
            norm = MidpointNormalize(midpoint=0)
            cmap='Spectral_r'
    if metric=='corr':
        norm = MidpointNormalize(midpoint=0, vmin=-1,vmax=1)
        cmap=cc.m_diverging_isoluminant_cjm_75_c24
    if metric=='corr_basins':
        norm = MidpointNormalize(midpoint=0, vmin=-1,vmax=1)
        cmap='Spectral'
    return norm,cmap




## GENERAL FUNCTIONS
def normalize(X,a=-1,b=1):
    m=X.min()
    M=X.max()
    Xnorm=(b-a)*(X-m)/(M-m)+a
    return Xnorm



## HYDROLOGICAL VARIABLES OVER GRID
def load_hydro_data(hydro_var_name,dataset_name,time_idx,path='../datasets/hydrology',version=1,fill_nan=True):
    # load dataset
    X=netCDF4.Dataset("{}/{}_{}.nc".format(path,hydro_var_name,dataset_name))

    # create time indexes
    if version==1:
        db=pd.DataFrame({'year':np.asarray(X.variables['time'][:][0,:]).astype(int),
                        'month':np.asarray(X.variables['time'][:][1,:]).astype(int),
                        'day':15})
        time_X=pd.to_datetime(db)
    if version==2:
        year=[d[3:] for d in np.asarray(X.variables['time'])]
        month=[d[:2] for d in np.asarray(X.variables['time'])]

        db=pd.DataFrame({'year':np.asarray(year).astype(int),
                        'month':np.asarray(month).astype(int),
                        'day':15})
        time_X=pd.to_datetime(db)

    # dataframe of all grid points
    if version==1:
        lat=np.asarray(X.variables['Lat'][:][0])
        long=np.asarray(X.variables['Long'][:][0])
        df=pd.DataFrame({'x':long,'y':lat})
    if version==2:
        lat=np.asarray(X.variables['Lat'])
        long=np.asarray(X.variables['Long'])
        (lat_flat,long_flat)=np.meshgrid(lat,long)
        df=pd.DataFrame({'x':long_flat.flatten(),'y':lat_flat.flatten()})

    spatial_grid=geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.x, df.y))

    # dataframe of each variable at each month over all grid point
    hydro_var=np.asarray(X.variables['{}_mm'.format(hydro_var_name)])
    X_grid=hydrological_variables_grid(hydro_var,time_X,hydro_var_name,spatial_grid,time_idx, version=version,fill_nan=fill_nan)

    return spatial_grid, X_grid, time_X



def hydrological_variables_grid(hydro_var,time_X,hydro_var_name,spatial_grid,time_idx,version=1,fill_nan=True):
    ''' returns a DataFrame with as many rows as the number of grid points
    in each column, the hydrological variable given for each month of time '''
    ind_time=np.where(time_X.isin(time_idx))[0]

    columns=['{} {}'.format(hydro_var_name,d.date()) for d in time_idx]
    db=spatial_grid.copy()

    if version==1:
        if fill_nan:
            # replace all missing values (=fill_value) by nans
            fill_value=0.0
            hydro_var=np.where(hydro_var==fill_value,np.nan,hydro_var)

        # select hydrological variable at overlapping time points
        hydro_grid=np.asarray(hydro_var)[ind_time,:]
        hydro_grid=pd.DataFrame(hydro_grid.T,index=db.index,columns=columns)

    if version==2:
        # hydro_var is of shape (nb_month x latitude points x longitude points
        if fill_nan:
            # replace all missing values (=fill_value) by nans
            fill_value=-9999
            hydro_var=np.where(hydro_var==fill_value,np.nan,hydro_var)
        Nx=hydro_var.shape[2]
        Ny=hydro_var.shape[1]
        Nt=ind_time.shape[0]
        hydro_grid=hydro_var[ind_time].T.reshape(Nx*Ny,Nt)
        hydro_grid=pd.DataFrame(hydro_grid,index=db.index,columns=columns)


    # join coordinates and hydrological variables
    db=db.join(hydro_grid)
    return db


def load_basins_data(approximate=True,path='../datasets/basins'):
    if approximate:
        basins=geopandas.read_file("{}/basins_with_approx_climate_zones_and_runoff_stations.shp".format(path))
    else:
        basins=geopandas.read_file("{}/basins_with_climate_zones_and_runoff_stations.shp".format(path))
    basins.set_index("NAME",inplace=True)
    basins.drop(['CATCH_ID', 'DB_ID', 'SORTING'],axis=1,inplace=True)
    basins.columns=['RASTAREA', 'LATITUDE', 'MAIN_CLIMATE', 'CLIMATE_AREA_%',
    'COLOR','NB_RUNOFF','geometry']
    basins.crs = 'epsg:4326'

    return basins



## BASIN SELECTION
def find_coordinates_inside_basin(my_basin,spatial_grid,basins,plot=False):
    spatial_grid_sindex=spatial_grid.sindex

    # Get the bounding box coordinates of the Polygon as a list
    bounds = list(my_basin.bounds.values[0])

    # Get the indices of the Points that are likely to be inside the bounding box of the given Polygon
    point_candidate_idx = list(spatial_grid_sindex.intersection(bounds))
    point_candidates = spatial_grid.loc[point_candidate_idx]

    if plot:
        # Let's see what we have now
        ax = my_basin.plot(color='blue', alpha=0.5)
        ax = point_candidates.plot(ax=ax, color='black', markersize=2)

    # Make the precise Point in Polygon query
    final_selection = point_candidates.loc[point_candidates.intersects(my_basin.values[0])]

    if plot:
        # Let's see what we have now
        ax = my_basin.plot(color='blue', alpha=0.5)
        ax = final_selection.plot(ax=ax, color='black', markersize=2)

    return final_selection


def my_fillna_temporal(hydro_basin,hydro_var_name,time_idx,method='cubic'):
    ''' reconstruct missing values in hydro_basin with interpolation '''
    columns=['{} {}'.format(hydro_var_name,d.date()) for d in time_idx]

    # cubic interpolation can be performed only on dates
    df=pd.DataFrame(hydro_basin[columns].values,columns=time_idx)
    try:
        df=df.interpolate(axis=1,method='cubic')

        # we change columns and index of df
        df=pd.DataFrame(df.values,columns=columns,index=hydro_basin.index)
        hydro_basin[columns]=df
        return hydro_basin,True
    # if there are too many missing values, interpolation cannot be performed
    except:
        return hydro_basin,False



def my_fillna_spatial(hydro_basin,hydro_var_name,time_idx,dataset_name,version=1,threshold=5,path='../datasets/hydrology'):
    columns=['{} {}'.format(hydro_var_name,d.date()) for d in time_idx]
    missing_points=hydro_basin.loc[np.isnan(hydro_basin[columns]).sum(axis=1)>0] #grid points with at least 1 missing value non recovered temporally
    if missing_points.shape[0]==0: # there is no missing point
        return hydro_basin,True

    if missing_points.shape[0]<hydro_basin.shape[0]*threshold/100: # there are not too many missing points
        for idx in missing_points.index:
            x=missing_points.loc[idx,'x']
            y=missing_points.loc[idx,'y']
            (res_lat,res_long)=compute_spatial_resolution(hydro_var_name,dataset_name,version=version)

            # if one point is not is the basin, the dataframe returned has 0 row
            under=hydro_basin.loc[(hydro_basin['x']==x)&(hydro_basin['y']==y-res_lat)]
            # remove the row if it is unknow
            if under.shape[0]>0:
                if np.isin(under.index[0],missing_points.index):
                    under=under.drop(under.index[0])

            above=hydro_basin.loc[(hydro_basin['x']==x)&(hydro_basin['y']==y+res_lat)]
            if above.shape[0]>0:
                if np.isin(above.index[0],missing_points.index):
                    above=above.drop(above.index[0])

            left=hydro_basin.loc[(hydro_basin['x']==x-res_long)&(hydro_basin['y']==y)]
            if left.shape[0]>0:
                if np.isin(left.index[0],missing_points.index):
                    left=left.drop(left.index[0])

            right=hydro_basin.loc[(hydro_basin['x']==x+res_long)&(hydro_basin['y']==y)]
            if right.shape[0]>0:
                if np.isin(right.index[0],missing_points.index):
                    right=right.drop(right.index[0])

            if under.shape[0]+above.shape[0]+left.shape[0]+right.shape[0]==4: # all neighbours are present
                new=(under[columns].values+above[columns].values+left[columns].values+right[columns].values)/4
                hydro_basin.loc[idx,columns]=new.flatten()

            if under.shape[0]+above.shape[0]==2: # vertical neighbours are present
                new=(under[columns].values+above[columns].values)/2
                hydro_basin.loc[idx,columns]=new.flatten()

            if left.shape[0]+right.shape[0]==2: # horizontal neighbours are present
                new=(left[columns].values+right[columns].values)/2
                hydro_basin.loc[idx,columns]=new.flatten()

    missing_points=hydro_basin.loc[np.isnan(hydro_basin[columns]).sum(axis=1)>0]
    return hydro_basin,(missing_points.shape[0]==0)


def my_fillna(hydro_basin,hydro_var_name,time_idx,dataset_name,version=1,threshold=5,path='../datasets/hydrology',method='cubic'):
    hydro_basin,temporal_filling=my_fillna_temporal(hydro_basin,hydro_var_name,time_idx,method=method)
    hydro_basin,spatial_filling=my_fillna_spatial(hydro_basin,hydro_var_name,time_idx,dataset_name,version=version,threshold=threshold,path=path)
    return hydro_basin,temporal_filling&spatial_filling



## SPATIAL AVERAGING
def compute_spatial_resolution(hydro_var_name,dataset_name,version=1,path='../datasets/hydrology'):
    X=netCDF4.Dataset("{}/{}_{}.nc".format(path,hydro_var_name,dataset_name))
    if version==1:
        lat=np.asarray(X.variables['Lat'][:][0])
        long=np.asarray(X.variables['Long'][:][0])
    if version==2:
        lat=np.asarray(X.variables['Lat'])
        long=np.asarray(X.variables['Long'])
    res_lat=np.abs(np.unique(lat)[0]-np.unique(lat)[1])
    res_long=np.abs(np.unique(long)[1]-np.unique(long)[0])

    return (res_lat,res_long)


def area_square(lat,lat_res=0.5,long_res=0.5,a=6.378e6):
    ''' gives the area in km^2 of a square af length spatial_res located at latitude lat'''
    radius=a*np.cos(lat*np.pi/180) # radius of the parallel supporting the square
    area=np.pi**2*a*radius/(4*90)**2 # area of a square of length 1°
    return area*lat_res*long_res/1e6


def weighted_average(hydro_basin,hydro_var_name,time_idx,lat_res=0.5,long_res=0.5,a=6.378e6):
    ''' compute the spatial average over a basin taking into account the Earth curvature '''
    area=area_square(hydro_basin['y'],lat_res=lat_res,long_res=long_res,a=a).values.reshape(hydro_basin.shape[0],1)
    total_area=np.sum(area)
    weighted_basin=hydro_basin[['{} {}'.format(hydro_var_name,d.date()) for d in time_idx]]*area
    mean_basin=np.sum(weighted_basin,axis=0)/total_area
    return mean_basin


def hydrological_variables_basin_filtered(hydro_basin,hydro_var_name,time_idx,dataset_name,a=6.378e6,version=1,path='../datasets/hydrology'):
    ''' compute the 3 point derivative of TWS
    filters hydrological variables to match TWS derivation '''

    # we check that there are no more missing values
    nb_nan=hydro_basin.loc[:,['{} {}'.format(hydro_var_name,d.date()) for d in time_idx]].isna().sum().sum()
    if nb_nan>0:
        raise Exception('There are missing values in {}'.format(hydro_var_name))

    (lat_res,long_res)=compute_spatial_resolution(hydro_var_name,dataset_name,version=version,path=path)

    # if terrestrial water storage, need to differentiate
    if hydro_var_name=='TWS':
        # mean over the basin at each time point
        hydro_mean_basin=weighted_average(hydro_basin,hydro_var_name,time_idx,lat_res=lat_res,long_res=long_res,a=a)

        # derivative f(x+1)-f(x-1)/2
        hydro_mean_basin=0.5*hydro_mean_basin.iloc[2:].values-0.5*hydro_mean_basin.iloc[:-2].values

        # formatting
        df_filter=pd.Series(hydro_mean_basin.flatten(),index=time_idx[1:-1],name='{} mean filtered'.format(hydro_var_name))
        return hydro_mean_basin,df_filter

    # otherwise, filtering to match the differential
    else:
        #mean over the basin at each time point
        hydro_mean_basin=weighted_average(hydro_basin,hydro_var_name,time_idx,lat_res=lat_res,long_res=long_res,a=a)

        # temporal filtering [f(i-1)+2f(i)+f(i+1)]/4
        hydro_mean_basin_filter=0.25*hydro_mean_basin.iloc[:-2].values+0.5*hydro_mean_basin.iloc[1:-1].values+0.25*hydro_mean_basin.iloc[2:].values

        # formatting
        df_filter=pd.Series(hydro_mean_basin_filter.flatten(),index=time_idx[1:-1],name='{} mean filtered'.format(hydro_var_name))
        return hydro_mean_basin,df_filter



## PERFORMANCE METRICS
def percentage_bias(X,Y): # should be as close from 0 as possible
    ''' Y is the observed variable
    X is the estimated variable'''
    if X.shape!=Y.shape:
        raise Exception("Shape of X {} is different from shape of Y {}".format(X.shape,Y.shape))
    #res=np.sum(Y-X)/np.sum(Y)
    res=1-X.mean()/Y.mean()
    return res

def compute_NSE(X,Y): # should be as close from 1 as possible
    ''' X is the estimated variable
    Y is the observed variables'''
    if X.shape!=Y.shape:
        raise Exception("Shape of X {} is different from shape of Y {}".format(X.shape,Y.shape))
    res=1-np.sum((X-Y)**2)/np.sum((X-Y.mean())**2)
    return res







