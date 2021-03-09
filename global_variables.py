import numpy as np

## CLIMATE ZONES COLOR
climate_color_dict={'A': '#ff0000',
 'Af': '#960000',
 'Am': '#ff0000',
 'Afm':'#960000',
 'As': '#ff9999',
 'Aw': '#ffcccc',
 'Asw':'#ffcccc',
 'B': '#ccaa54',
 'BWk': '#ffff63',
 'BWh': '#ffcc00',
 'BSk': '#ccaa54',
 'BSh': '#cc8c14',
 'C': '#007800',
 'Cfa': '#003300',
 'Cfb': '#004f00',
 'Cfc': '#007800',
 'Cf': '#004f00',
 'Csa': '#00ff00',
 'Csb': '#95ff00',
 'Csc': '#c8ff00',
 'Cs': '#95ff00',
 'Cwa': '#593b00',#'#b56400',
 'Cwb': '#966400',
 'Cwc': '#593b00',
 'D': '#ff6eff',
 'Dfa': '#330033',
 'Dfb': '#630063',
 'Dfc': '#c700c7',
 'Dfd': '#c71686',
 'Dsa': '#ff6eff',
 'Dsb': '#ffb5ff',
 'Dsc': '#e6c7ff',
 'Dsd': '#cccccc',
 'Dw': '#997fb3',
 'Dwa': '#c7b3c7',
 'Dwb': '#997fb3',
 'Dwc': '#8759b3',
 'Dwd': '#6d24b3',
 'E': '#6395ff',
 'EF': '#6395ff',
 'ET': '#63ffff'}


climate_name_dict={'Af': 'Equatorial rainforest, fully humid',
 'Am': 'Equatorial monsoon',
 'Afm' :'Equatorial rainforest/monsoon', # gathers Af and Am
 'Asw':'Equatorial savannah', #gathers As and Aw
 'As': 'Equatorial savannah with dry summer',
 'Aw': 'Equatorial savannah with dry winter',
 'A': 'Equatorial',
 'BSh': 'Arid Steppe hot',
 'BSk': 'Arid Steppe cold',
 'BWh': 'Arid desert hot',
 'BWk': 'Arid desert cold',
 'B': 'Arid',
 'Cfa': 'Warm temperate fully humid with hot summer',
 'Cfb': 'Warm temperate fully humid with warm summer',
 'Cfc': 'Warm temperate fully humid with cool summer',
 'Cf': 'Warm temperate fully humid',
 'Csa': 'Warm temperate with dry, hot summer',
 'Csb': 'Warm temperate with dry, warm summer',
 'Csc': 'Warm temperate with dry, cool summer',
 'Cs': 'Warm temperate with dry summer',
 'Cwa': 'Warm temperate with dry winter and hot summer',
 'Cwb': 'Warm temperate with dry winter and warm summer',
 'Cwc': 'Warm temperate with dry winter and cool summer',
 'C':'Warm temperate',
 'Dfa': 'Snow with fully humid hot summer',
 'Dfb': 'Snow fully humid warm summer',
 'Dfc': 'Snow fully humid cool summer',
 'Dfd': 'Snow fully humid extremely continental',
 'Dsa': 'Snow dry, hot summer',
 'Dsb': 'Snow dry, warm summer',
 'Dsc': 'Snow dry, cool summer',
 'Dw': 'Snow dry winter', # gathers DWa, Dwb, Dwc, Dwd
 'Dwa': 'Snow dry winter hot summer',
 'Dwb': 'Snow dry winter warm summer',
 'Dwc': 'Snow dry winter cool summer',
 'Dwd': 'Snow dry winter extremely continental',
 'D': 'Snow',
 'EF': 'Polar frost',
 'ET': 'Polar tundra',
 'E':'Polar'}


dict_color_zone={'Amazon':'deepskyblue',
                'Brazos':'tab:orange',
                'Chad':'limegreen',
                'Congo':'tab:red',
                'Danube':'violet',
                'Ganges':'tab:brown',
                'Lake Eyre':'gold',
                'Lena':'tab:gray',
                'Orinoco':'rebeccapurple',
                'Rhine':'tab:cyan',
                'Volga':'forestgreen',
                'Yangtze':'tab:olive',
                'Yenisey':'dodgerblue'}


 ## DATASETS

version_data={'GLDAS21_NOAH36':2,'GLDAS21_CLSM25':2,'GLDAS21_VIC412':2,
'GLDAS20_NOAH36':2,'GLDAS20_CLSM25':2,'GLDAS20_VIC412':2,'GLDAS22_CLSM25':2,
'GLDAS21':2,'MERRA2':2,'SSEBop':2,'GRACE_CSR_mascons':2,'GRUN':2,'GRACE_JPL_mascons':2,'GRACE_CSR_grid':2,'CRU':2,'GPCP':2,'GLEAM':2,'GPM':2,'ERA5_Land':2,'GPCC':2,'MOD16':2,
'JRA55':2,'FLUXCOM':2,'GLDAS20':2,
'GRACE_JPL':1,'CPC':2,'TRMM':2,'GLDAS':1,'SEB':1,'MSWEP':1}

dict_fill_value={'GLDAS21_NOAH36':-9999,'GLDAS21_CLSM25':-9999,'GLDAS21_VIC412':-9999,
'GLDAS20_NOAH36':-9999,'GLDAS20_CLSM25':-9999,'GLDAS20_VIC412':-9999,
'GPCP':-9999,'GLEAM':-9999,'GPM':-9999,'ERA5_Land':-9999,'GPCC':-9999,'MOD16':-9999,'GLDAS20':-9999,
'GLDAS21':-9999,'MERRA2':-9999,'SSEBop':-9999,'GRUN':-9999,'GRACE_JPL_mascons':-9999,'GRACE_CSR_grid':-9999,'CRU':-9999,'GLDAS22_CLSM25':-9999,'JRA55':-9999,'FLUXCOM':-9999,
'GRACE_JPL':-9999,'GRACE_CSR_mascons':-9999,'CPC':-9999,'TRMM':-9999,'GLDAS':0,'SEB':0,'MSWEP':-9999}


dict_dataset_name={'CLSM2.0':'GLDAS20_CLSM25','CLSM2.1':'GLDAS21_CLSM25','CLSM2.2':'GLDAS22_CLSM25',
                  'NOAH2.0':'GLDAS20_NOAH36','NOAH2.1':'GLDAS21_NOAH36','Princeton':'GLDAS20',
                  'VIC2.0':'GLDAS20_VIC412','VIC2.1':'GLDAS21_VIC412',
                  'ERA5_Land':'ERA5 Land','MERRA2':'MERRA2','JRA55':'JRA55',
                   'CPC':'CPC','CRU':'CRU','GPCC':'GPCC','GPCP':'GPCP','GPM':'GPM',
                   'TRMM':'TRMM','MSWEP':'MSWEP','GLEAM':'GLEAM','MOD16':'MOD16',
                   'SSEBop':'SSEBop','FLUXCOM':'FLUXCOM','GRUN':'GRUN',
                   'GLDAS20_CLSM25':'CLSM2.0','GLDAS21_CLSM25':'CLSM2.1','GLDAS22_CLSM25':'CLSM2.2',
                   'GLDAS20_NOAH36':'NOAH2.0','GLDAS21_NOAH36':'NOAH2.1','GLDAS20':'Princeton',
                   'GLDAS20_VIC412':'VIC2.0','GLDAS21_VIC412':'VIC2.1',
                   'ERA5 Land':'ERA5_Land'
                  }


colors_dataset={'ERA5_Land':'firebrick','MERRA2':'darkorange','JRA55':'tomato',
               'CPC':'olive','CRU':'forestgreen','GPCC':'limegreen',
                'GPCP':'mediumpurple','GPM':'rebeccapurple',
               'TRMM':'hotpink','MSWEP':'violet','GLDAS20':'lightpink','GLDAS2.0-f':'lightpink',
               'CLSM2.0':'teal','CLSM2.1':'lightgreen','CLSM2.2':'aquamarine',
                'GLDAS20_CLSM25':'teal','GLDAS21_CLSM25':'lightgreen','GLDAS22_CLSM25':'aquamarine',
                'NOAH2.0':'lightskyblue','NOAH2.1':'deepskyblue',
               'GLDAS20_NOAH36':'lightskyblue','GLDAS21_NOAH36':'deepskyblue',
                'VIC2.0':'blue','VIC2.1':'darkblue',
               'GLDAS20_VIC412':'blue','GLDAS21_VIC412':'darkblue',
               'GLEAM':'rebeccapurple','MOD16':'violet','SSEBop':'hotpink','FLUXCOM':'mediumpurple',
               'GRUN':'#eaee5e'}


def format_combination(comb):
    comb='P: '+comb[2:-22]
    comb=comb.replace('P: GLDAS20', 'P: Princeton')
    comb=comb.replace('_ET_',' ; ET: ')
    comb=comb.replace('_R_',' ; R: ')

    comb=comb.replace('ERA5_Land','ERA5 Land')
    comb=comb.replace('GLDAS20_CLSM25','CLSM2.0')
    comb=comb.replace('GLDAS21_CLSM25','CLSM2.1')
    comb=comb.replace('GLDAS22_CLSM25','CLSM2.2')
    comb=comb.replace('GLDAS20_NOAH36','NOAH2.0')
    comb=comb.replace('GLDAS21_NOAH36','NOAH2.1')
    comb=comb.replace('GLDAS20_VIC412','VIC2.0')
    comb=comb.replace('GLDAS21_VIC412','VIC2.1')
    return comb

