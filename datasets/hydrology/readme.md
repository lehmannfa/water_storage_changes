all hydrological variables (precipitations, runoff, evapotranspiration, water storage) available from difference datasets.
Datasets are too large to be stored on Github, they need to be downloaded from [SharePoint](https://uob.sharepoint.com/teams/grp-globalmass/Shared%20Documents/Forms/AllItems.aspx?FolderCTID=0x012000C86F09E9396E694EBC6704F6B0D20797&viewid=fefbffbe%2D6bc8%2D45a0%2D96bc%2Dde244b9f68bb&id=%2Fteams%2Fgrp%2Dglobalmass%2FShared%20Documents%2FGlobalMass%2FMRes%5F2020%5Fwater%5Fcycle%2FDatasets).

## Current datasets needed

Evapotranspiration: 

* ET_GLDAS21_CLSM25 ([link](https://disc.gsfc.nasa.gov/datasets/GLDAS_CLSM10_M_2.1/summary?keywords=GLDAS_CLSM10_M_2.1)) : , monthly, Jan 2000 - June 2020, 1° x 1°, on land
* ET_GLDAS21_NOAH36 ([link](https://disc.gsfc.nasa.gov/datasets/GLDAS_NOAH10_M_2.1/summary?keywords=GLDAS_NOAH10_M_2.1)) : , monthly, Jan 2000 - June 2020, 1° x 1°, on land
* ET_GLDAS_VIC412 ([link](https://disc.gsfc.nasa.gov/datasets/GLDAS_VIC10_M_2.1/summary?keywords=GLDAS_VIC10_M_2.1)) : , monthly, Jan 2000 - June 2020, 1° x 1°, on land
* ET_GLDAS20_CLSM25 ([link](https://disc.gsfc.nasa.gov/datasets/GLDAS_CLSM10_M_2.0/summary?keywords=GLDAS)) : monthly, Jan 1948-Dec 2014 (restricted to Jan 1979 - Dec 2014), 1.0° x 1.0°, -60° x 90°, on land
* ET_GLDAS20_NOAH36 ([link](https://disc.gsfc.nasa.gov/datasets/GLDAS_NOAH10_M_2.0/summary?keywords=GLDAS)): monthly, Jan 1948-Dec 2014 (restricted to Jan 1979 - Dec 2014), 1.0° x 1.0°, -60° x 90°, on land 
* ET_GLDAS20_VIC412 ([link](https://disc.gsfc.nasa.gov/datasets/GLDAS_VIC10_M_2.0/summary?keywords=GLDAS)): monthly, Jan 1948-Dec 2014 (restricted to Jan 1979 - Dec 2014), 1.0° x 1.0°, -60° x 90°, on land 
* ET_MERRA2 ([link](https://disc.gsfc.nasa.gov/datasets/M2TMNXLND_5.12.4/summary?keywords=MERRA-2)) : , monthly, Jan 1980 - Sep 2020, 0.5° x 0.625°, on land
* ET_SEBBop : , monthly, Jan 2003 - Sep 2020, 0.5° x 0.5°, -60° x 80° on land
* ET_GLEAM ([Global Land Evaporation Amsterdam Model](https://www.gleam.eu/)) : reanalysis, monthly, Jan 1980 - Dec 2018, 0.25° x 0.25° mapped to 0.5° x 0.5°, global on land
* ET_ERA5-Land ([ECMWF](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview)) : monthly, Jan 1981 - Aug 2020, 0.1° x 0.1°, global on land

Precipitations : 

* P_CPC : observation based, monthly, Jan 2002 - Dec 2017, 0.5° x 0.5°, global (modelled over seas)
* P_GLDAS21 : forcing data (Princeton), monthly, Jan 2000 - June 2020, 1° x 1°, on land
* P_GPCC ([DWD](https://opendata.dwd.de/climate_environment/GPCC/html/fulldata-monthly_v2018_doi_download.html)) : 
* P_GPCP ([link](https://psl.noaa.gov/data/gridded/data.gpcp.html)) : monthly, Jan 1979 - Oct 2020, 2.5° x 2.5° interpolated to 0.5° x 0.5°
* P_GPM ([NASA](https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGM_06/summary?keywords=%22IMERG%20final%22)) : monthly, June 2000 - Aug 2020, 0.1° x 0.1°, global
* P_ERA5-Land ([ECMWF](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview)) : monthly, Jan 1981 - Aug 2020, 0.1° x 0.1°, global on land
* P_MERRA2 ([link](https://disc.gsfc.nasa.gov/datasets/M2TMNXLND_5.12.4/summary?keywords=MERRA-2)) : , monthly, Jan 1980 - Sep 2020, 0.5° x 0.625°, on land
* P_MSWEP : satellite, monthly, Jan 1979 - Oct 2017, 0.5° x 0.5°, global
* P_TRMM : , monthly, Jan 2002 - Dec 2016, 0.5° x 0.5°, -50° x 50°
* P_CRU ([Climatic Research Unit](https://catalogue.ceda.ac.uk/uuid/89e1e34ec3554dc98594a5732622bce9)) : observation based, monthly, Jan 1901-Dec 2019 (restricted to Jan 1979 - Dec 2019), 0.5° x 0.5°, -60°x90°

Runoff : 

* R_ERA5-Land ([ECMWF](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview)) : monthly, Jan 1981 - Aug 2020, 0.1° x 0.1°, global on land
* R_GLDAS21_CLSM25 ([link](https://disc.gsfc.nasa.gov/datasets/GLDAS_CLSM10_M_2.1/summary?keywords=GLDAS_CLSM10_M_2.1)) : satellite + ground based observations, monthly, Jan 2000 - June 2020, 1° x 1°, on land
* R_GLDAS21_NOAH36 ([link](https://disc.gsfc.nasa.gov/datasets/GLDAS_NOAH10_M_2.1/summary?keywords=GLDAS_NOAH10_M_2.1)): satellite + ground based observations, monthly, Jan 2000 - June 2020, 1° x 1°, on land
* R_GLDAS21_VIC412 ([link](https://disc.gsfc.nasa.gov/datasets/GLDAS_VIC10_M_2.1/summary?keywords=GLDAS_VIC10_M_2.1)) : satellite + ground based observations, monthly, Jan 2000 - June 2020, 1° x 1°, on land
* ET_GLDAS20_CLSM25 ([link](https://disc.gsfc.nasa.gov/datasets/GLDAS_CLSM10_M_2.0/summary?keywords=GLDAS)) : monthly, Jan 1948-Dec 2014 (restricted to Jan 1979 - Dec 2014), 1.0° x 1.0°, -60° x 90°, on land
* ET_GLDAS20_NOAH36 ([link](https://disc.gsfc.nasa.gov/datasets/GLDAS_NOAH10_M_2.0/summary?keywords=GLDAS)): monthly, Jan 1948-Dec 2014 (restricted to Jan 1979 - Dec 2014), 1.0° x 1.0°, -60° x 90°, on land
* ET_GLDAS20_VIC412 ([link](https://disc.gsfc.nasa.gov/datasets/GLDAS_VIC10_M_2.0/summary?keywords=GLDAS)): monthly, Jan 1948-Dec 2014 (restricted to Jan 1979 - Dec 2014), 1.0° x 1.0°, -60° x 90°, on land 
* R_GRUN ([link](https://figshare.com/articles/GRUN_Global_Runoff_Reconstruction/9228176)): machine learning from GRDC observations, monthly, Jan 1902 - Dec 2014 (restricted to Jan 1979 - Dec 2014), 0.5° x 0.5°, -60° x 80° on land 
* R_MERRA2 ([link](https://disc.gsfc.nasa.gov/datasets/M2TMNXLND_5.12.4/summary?keywords=MERRA-2)) : , monthly, Jan 1980 - Sep 2020, 0.5° x 0.625°, on land

Terrestrial water storage : 

* TWS_GRACE_JPL ([Jet Propulsion Laboratory](https://podaac.jpl.nasa.gov/dataset/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06_V2)) : mascons, monthly, Apr 2002 - Present, 0.5° x 0.5°, global 
* TWS_GRACE_CSR ([Center for Space Research](http://www2.csr.utexas.edu/grace/RL06_mascons.html)) : mascons
* TWS_GRACE_CSR_grid ([Center for Space Research](https://podaac.jpl.nasa.gov/dataset/TELLUS_GRAC_L3_CSR_RL06_LND_v03)) : land mass grids from spherical harmonics, monthly, Apr 2002 - June 2017, 1°x1° on land

Potential EvapoTranspiration : 
* USGS ([United States Geological Survey](https://earlywarning.usgs.gov/fews/product/81)) : 
* CRU ([Climatic Research Unit](https://catalogue.ceda.ac.uk/uuid/89e1e34ec3554dc98594a5732622bce9)) : observation based, monthly, Jan 1901-Dec 2019 (restricted to Jan 1979 - Dec 2019), 0.5° x 0.5°, -60°x90°
* PET ([Global Land Evaporation Amsterdam Model](https://www.gleam.eu/)) : reanalysis, monthly, Jan 1980 - Dec 2018, 0.25° x 0.25° mapped to 0.5° x 0.5°, global on land


CLSM simulates shallow groundwater. Terrestrial Water Storage in CLSM = soil water + snow water equivalent + canopy water + groundwater

TWS in Noah = soil moisture in all layers + accumulated snow + plant canopy surface water

Groundwater storage is included in CLSM TWS : GWS = TWS - RootZoneSoilMoisture - SnowWaterEquivalent - CanopyInterception
