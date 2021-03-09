from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib import cm
import matplotlib.pyplot as plt
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


def define_cmap(hydro_var_name,values,percentile=False):
    ''' called in each basin, for each hydrological variable, at a given month'''
    #vmin=hydro_basin['{} {}-{}-15'.format(hydro_var_name,year,month)].min()
    values=values[np.where(~np.isnan(values))]
    if percentile:
        vmin=np.percentile(values,2)
        vmax=np.percentile(values,98)
    else:
        vmin=np.min(values)
        vmax=np.max(values)

    if hydro_var_name=='TWS':
        norm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
        cmap='RdBu'
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
            cmap = cm.get_cmap('YlGn')
            norm = MidpointNormalize(midpoint=0, vmin=0,vmax=1)
    elif metric=='NSE_large':
        if discrete:
            cmap=cm.YlGn
            bounds = [0.19,0.2,0.5,0.75,1]
            norm = BoundaryNorm(bounds, cmap.N)
        else:
            cmap = cm.get_cmap('YlGn')
            norm = MidpointNormalize(midpoint=0, vmin=0,vmax=1)
    elif metric=='PBIAS':
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
    elif metric=='corr':
        norm = MidpointNormalize(midpoint=0, vmin=-1,vmax=1)
        cmap=cc.m_diverging_isoluminant_cjm_75_c24
    elif metric=='corr_basins':
        norm = MidpointNormalize(midpoint=0, vmin=-1,vmax=1)
        cmap='Spectral'
    else:
        norm = MidpointNormalize(midpoint=0)
        cmap='viridis'
    return norm,cmap




## GENERAL FUNCTIONS
def normalize(X,a=-1,b=1):
    m=X.min()
    M=X.max()
    Xnorm=(b-a)*(X-m)/(M-m)+a
    return Xnorm


def derivative(X):
    ''' compute the 2 points derivative'''
    Y=0.5*(X[2:].values-X[:-2].values)
    return Y


def uncertainty_derivative(X):
    Y=0.5*np.sqrt(X[2:].values**2+X[:-2].values**2)
    return Y


def time_filter(X):
    ''' filter to match derivation of TWS'''
    Y=0.25*X[:-2].values+0.5*X[1:-1].values+0.25*X[2:].values
    return Y



## HYDROLOGICAL VARIABLES OVER GRID
def load_hydro_data(hydro_var_name,dataset_name,fill_value=-9999,path='../datasets/hydrology',version=1,fill_nan=True):
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
    X_grid=hydrological_variables_grid(hydro_var,time_X,hydro_var_name,spatial_grid, fill_value=fill_value,version=version,fill_nan=fill_nan)

    return spatial_grid, X_grid, time_X



def hydrological_variables_grid(hydro_var,time_X,hydro_var_name,spatial_grid,fill_value,version=1,fill_nan=True):
    ''' returns a DataFrame with as many rows as the number of grid points
    in each column, the hydrological variable given for each month of time '''
    columns=['{} {}'.format(hydro_var_name,d.date()) for d in time_X]
    db=spatial_grid.copy()

    if version==1:
        if fill_nan:
            # replace all missing values (=fill_value) by nans
            hydro_var=np.where(hydro_var==fill_value,np.nan,hydro_var)

        # select hydrological variable
        hydro_grid=pd.DataFrame(hydro_var.T,index=db.index,columns=columns)

    if version==2:
        # hydro_var is of shape (nb_month x latitude points x longitude points
        if fill_nan:
            # replace all missing values (=fill_value) by nans
            hydro_var=np.where(hydro_var==fill_value,np.nan,hydro_var)
        Nx=hydro_var.shape[2]
        Ny=hydro_var.shape[1]
        Nt=time_X.shape[0]

        #hydro_grid=hydro_var[ind_time].T.reshape(Nx*Ny,Nt)

        Nt=time_X.shape[0]
        hydro_grid=hydro_var.T.reshape(Nx*Ny,Nt)

        hydro_grid=pd.DataFrame(hydro_grid,index=db.index,columns=columns)


    # join coordinates and hydrological variables
    db=db.join(hydro_grid)
    return db


def load_basins_data(approximate=True,path='../datasets/basins'):
    if approximate:
        basins=geopandas.read_file("{}/basins_with_approx_climate_zones.shp".format(path))
    else:
        basins=geopandas.read_file("{}/basins_with_climate_zones.shp".format(path))
    basins.set_index("NAME",inplace=True)
    basins.drop(['CONTINENT','OCEAN','CLIMATE_AR','COLOR_HEX'],axis=1,inplace=True)
    basins.columns=['RASTAREA', 'MAIN_CLIMATE','zone','geometry']
    basins.crs = 'epsg:4326'
    return basins



def test_same_spatial_grid(spatial_grid_ref,spatial_grid_test):
    ''' Test if two spatial grids are the same to avoid unnecessary computations'''
    if spatial_grid_ref.shape!=spatial_grid_test.shape:
        return False
    if (spatial_grid_ref['x']!=spatial_grid_test['x']).any():
        return False
    if (spatial_grid_ref['y']!=spatial_grid_test['y']).any():
        return False
    return True



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
        df=df.interpolate(axis=1,method=method)

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

    # if there is only one point missing, we remove it from hydro_basin
    if missing_points.shape[0]==1:
            hydro_basin.drop(missing_points.index[0],inplace=True)
            return hydro_basin,True

    return hydro_basin,(missing_points.shape[0]==0)


def my_fillna(hydro_basin,hydro_var_name,time_X,dataset_name,version=1,threshold=5,path='../datasets/hydrology',method='cubic',fill_spatial=True):
    if hydro_var_name=='TWS':
        hydro_basin,temporal_filling=my_fillna_temporal(hydro_basin,hydro_var_name,time_X,method=method)

    if fill_spatial:
        hydro_basin,spatial_filling=my_fillna_spatial(hydro_basin,hydro_var_name,time_X,dataset_name,version=version,threshold=threshold,path=path)

    hydro_basin=hydro_basin[['x','y','geometry']+['{} {}'.format(hydro_var_name,d.date()) for d in time_X]]
    nb_nans=np.sum(np.sum(np.isnan(hydro_basin[['{} {}'.format(hydro_var_name,d.date()) for d in time_X]])))
    filling=nb_nans==0
    return hydro_basin,filling






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
    ''' gives the area in km^2 of a square of length spatial_res located at latitude lat'''
    radius=a*np.cos(lat*np.pi/180) # radius of the parallel supporting the square
    area=np.pi**2*a*radius/(4*90)**2 # area of a square of length 1°
    return area*lat_res*long_res/1e6


def weighted_average(hydro_basin,hydro_var_name,time_idx,lat_res=0.5,long_res=0.5,a=6.378e6):
    ''' compute the spatial average over a basin taking into account the Earth curvature
    if there are missing values, then the total area is oversetimated compared to the area really covered by values'''
    area=area_square(hydro_basin['y'],lat_res=lat_res,long_res=long_res,a=a).values.reshape(hydro_basin.shape[0],1)
    total_area=np.sum(area)
    weighted_basin=hydro_basin[['{} {}'.format(hydro_var_name,d.date()) for d in time_idx]]*area
    mean_basin=np.sum(weighted_basin,axis=0)/total_area # counts nans as zero
    return mean_basin


def hydrological_variables_basin_filtered(hydro_basin,hydro_var_name,time_idx,dataset_name,a=6.378e6,version=1,path='../datasets/hydrology',missing_threshold=5):
    ''' compute the 3 point derivative of TWS
    filters hydrological variables to match TWS derivation '''

    # we check that the number of points with missing months does not exceed the threshold (in %)
    missing_points=hydro_basin.loc[np.sum(np.isnan(hydro_basin[['{} {}'.format(hydro_var_name,d.date()) for d in time_idx]]),axis=1)>0].index
    if 100*missing_points.shape[0]/hydro_basin.shape[0]>missing_threshold:
        df=pd.Series(np.nan*np.ones(time_idx.shape[0]),index=time_idx)
        df_filter=pd.Series(np.nan*np.ones(time_idx.shape[0]-2),index=time_idx[1:-1])
        return df,df_filter

    (lat_res,long_res)=compute_spatial_resolution(hydro_var_name,dataset_name,version=version,path=path)

    # mean over the basin at each time point
    hydro_mean_basin=weighted_average(hydro_basin,hydro_var_name,time_idx,lat_res=lat_res,long_res=long_res,a=a)

    # if terrestrial water storage, need to differentiate
    if hydro_var_name=='TWS':
        hydro_mean_basin_filter=derivative(hydro_mean_basin)

    # if water storage uncertainty, we apply the formula for uncertainty addition
    elif hydro_var_name=='TWS_uncertainty':
        hydro_mean_basin_filter=uncertainty_derivative(hydro_mean_basin)

    # otherwise, filtering to match the differential
    elif hydro_var_name in ['P','ET','R','PET','LE']:
        hydro_mean_basin_filter=time_filter(hydro_mean_basin)

    else:
        raise Exception('Variable {} is unknown'.format(hydro_var_name))

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
    ''' X is the estimated variable from the budget equation
    Y is the reference variable (measured)'''
    if X.shape!=Y.shape:
        raise Exception("Shape of X {} is different from shape of Y {}".format(X.shape,Y.shape))
    if np.sum(X.index==Y.index)!=X.shape[0]:
        raise Exception('Indices of X and Y do not match')
    res=1-np.sum((X-Y)**2)/np.sum((Y-Y.mean())**2)
    return res



def compute_cyclic_NSE(X,Y): # should be as close from 1 as possible
    ''' X is the estimated variable from the budget equation
    Y is the reference variable (measured)'''
    if X.shape!=Y.shape:
        raise Exception("Shape of X {} is different from shape of Y {}".format(X.shape,Y.shape))
    if np.sum(X.index==Y.index)!=X.shape[0]:
        raise Exception('Indices of X and Y do not match')

    Y_df=Y.to_frame()

    # compute the mean value per month
    Y_df['month']=Y_df.index.month
    Y_month=Y_df.groupby('month').mean()
    Y_month.columns=['mean month']

    # associate each date to the mean monthly value
    Y_df=Y_df.join(Y_month,on='month')

    res=1-np.sum((X-Y)**2)/np.sum((Y-Y_df['mean month'])**2)
    return res


def compute_RMSE(X,Y,normalized=False):
    if X.shape!=Y.shape:
        raise Exception("Shape of X {} is different from shape of Y {}".format(X.shape,Y.shape))

    RMSE=np.sqrt(np.sum((X-Y)**2)/X.shape[0])

    if normalized:
        q1=np.percentile(Y,25)
        q3=np.percentile(Y,75)
        return RMSE/(q3-q1)
    else:
        return RMSE




## SEASONAL DECOMPOSITION
def seasonal_Fourier_reconstruction(X,Y,P,nP,dP,linear=False):
    if linear: # to be used by TWS where there can be a trend. Trend is suppressed by derivationin TWSC
        # remove the linear trend
        model = LinearRegression()
        model.fit(X.reshape(X.shape[0],1), Y)
        Ydetrend=Y-model.predict(X.reshape(X.shape[0],1))
    else:
        Ydetrend=Y

    # constant term
    a0=np.sum(Ydetrend[0:nP*int(P/dP)]*dP)/(P*nP)
    if linear and np.abs(a0)>1e-8:
        raise Exception("a0 should be 0, it is {:.2e}".format(a0))

    # annual sine wave
    a1=np.sum(Ydetrend[0:nP*int(P/dP)]*np.cos(2*np.pi*X[0:nP*int(P/dP)]/P)*dP)*(2/(P*nP))
    b1=np.sum(Ydetrend[0:nP*int(P/dP)]*np.sin(2*np.pi*X[0:nP*int(P/dP)]/P)*dP)*(2/(P*nP))
    A1=np.sqrt(a1**2+b1**2) # amplitude of the annual signal
    phi1=np.arctan2(b1,a1) # phase of the annual signal

    # semi-annual sine wave
    a2=np.sum(Ydetrend[0:nP*int(P/dP)]*np.cos(4*np.pi*X[0:nP*int(P/dP)]/P)*dP)*(2/(P*nP))
    b2=np.sum(Ydetrend[0:nP*int(P/dP)]*np.sin(4*np.pi*X[0:nP*int(P/dP)]/P)*dP)*(2/(P*nP))
    A2=np.sqrt(a2**2+b2**2)
    phi2=np.arctan2(b2,a2)

    # sum all components: constant term + annual sine + semi-annual sine (+ linear trend)
    if linear:
        bias=model.intercept_
        trend=model.coef_[0]
        Yrecons=bias+trend*X+A1*np.cos(2*np.pi*X/P-phi1)+A2*np.cos(4*np.pi*X/P-phi2)
        return bias,trend,Yrecons
    else:
        bias=a0
        Yrecons=bias+A1*np.cos(2*np.pi*X/P-phi1)+A2*np.cos(4*np.pi*X/P-phi2)
        return bias,Yrecons


def deseasone(Y_filter,P=12,dP=1):
    nP=Y_filter.shape[0]//P
    X=np.arange(0,Y_filter.shape[0],dP)[:nP*P] # month indices
    Y=Y_filter.iloc[:nP*P] # restrict to an integer number of periods of length P
    bias,Yrecons=seasonal_Fourier_reconstruction(X,Y,P,nP,dP) # Yrecons is the seasonal component of the signal
    return Y-Yrecons







## PLOTS
def plot_water_budget(TWSC_filter,A_filter,time_idx,basin_name,data_P,data_ET,data_R,data_TWS,save_fig=False):
    TWSC_filter.index=time_idx

    corr=A_filter.corr(TWSC_filter)
    PBIAS=percentage_bias(A_filter,TWSC_filter)
    NSE=compute_NSE(A_filter,TWSC_filter)

    TWSC_mean=TWSC_filter.mean()
    A_mean=A_filter.mean()
    plt.figure()
    plt.plot(TWSC_filter,'k',label='dTWS/dt mean={:.2f}'.format(TWSC_mean))
    plt.plot([TWSC_filter.index[0],TWSC_filter.index[-1]],[TWSC_mean,TWSC_mean],'k--',alpha=0.5)
    plt.plot(A_filter,color='darkviolet',label='P-ET-R mean={:.2f}'.format(A_mean))
    plt.plot([A_filter.index[0],A_filter.index[-1]],[A_mean,A_mean],'--',color='darkviolet',alpha=0.5)
    plt.legend()
    plt.title("Water budget equation in {} \n P {}, ET {}, R {}, TWS {} \n NSE={:.2f} PBIAS={:.2f} correlation={:.2f}".format(basin_name,
                                        data_P,data_ET,data_R,data_TWS,NSE,PBIAS,corr))
    plt.xlabel("month")
    plt.ylabel("mm/month")
    plt.xlim([time_idx[0],time_idx[-1]])
    plt.tight_layout()

    if save_fig:
        plt.savefig("../plots/water_budget/{}_P_{}_ET_{}_R_{}.png".format(basin_name,
                                                data_P,data_ET,data_R))
    plt.show()


def plot_water_budget_details(TWSC_filter,P_filter,ET_filter,R_filter,time_selec,basin_name,data_P,data_ET,data_R,data_TWS,save_fig=False,figsize=None):
    dates=['P_{} {}'.format(data_P,d.date()) for d in time_selec]
    P=pd.Series(P_filter.loc[dates].values,index=time_selec)

    dates=['ET_{} {}'.format(data_ET,d.date()) for d in time_selec]
    ET=pd.Series(ET_filter.loc[dates].values,index=time_selec)

    dates=['R_{} {}'.format(data_R,d.date()) for d in time_selec]
    R=pd.Series(R_filter.loc[dates].values,index=time_selec)

    dates=['TWS_{} {}'.format(data_TWS,d.date()) for d in time_selec]
    TWSC=pd.Series(TWSC_filter.loc[dates].values,index=time_selec)

    A_filter=P-ET-R
    NSE=compute_NSE(A_filter,TWSC)

    if figsize:
        plt.figure(figsize=figsize)
    else:
        plt.figure()

    if data_TWS=='GRACE_JPL_mascons':
        TWS_uncertainty_month=pd.read_csv('../results/hydrology/TWS_uncertainty_{}_monthly_filtered.csv'.format(data_TWS),index_col=[0])
        TWSC_uncertainty_filter=TWS_uncertainty_month.loc[basin_name,['TWS_uncertainty_{} {}'.format(data_TWS,d.date()) for d in time_selec]]
        TWSC_uncertainty_filter.index=time_selec
        plt.fill_between(time_selec,TWSC-TWSC_uncertainty_filter,
                 TWSC+TWSC_uncertainty_filter,color='gray',alpha=0.5,label='TWSC uncertainty')

    plt.plot(time_selec,TWSC,'k--',label='dTWS/dt')
    plt.plot(time_selec,P,color='dodgerblue',label='P')
    plt.plot(time_selec,P-ET,color='teal',label='P-ET')
    plt.plot(time_selec,P-ET-R,color='rebeccapurple',label='P-ET-R')

    plt.legend()
    plt.title("Water budget equation in {} \n P {}, ET {}, R {},TWS {} \n NSE={:.2f}".format(basin_name,
                                        data_P,data_ET,data_R,data_TWS,NSE))
    plt.xlabel("month")
    plt.ylabel("mm")
    #plt.xlim([time_selec[0],time_selec[-1]])
    plt.tight_layout()

    if save_fig:
        plt.savefig("../plots/water_budget/{}_P_{}_ET_{}_R_{}_details.png".format(basin_name,
                                                data_P,data_ET,data_R))
    plt.show()


def decompose_dataset(combination):
    iP=combination.find('_ET')
    data_P=combination[2:iP]
    iET=combination.find('_R')
    data_ET=combination[iP+4:iET]
    iR=combination.find('_TWS')
    data_R=combination[iET+3:iR]
    data_TWS=combination[iR+5:]

    return data_P,data_ET,data_R,data_TWS


def find_best_dataset(basin_name,max_NSE):
    best_comb=max_NSE.loc[basin_name,'best_dataset']

    return decompose_dataset(best_comb)






