import numpy as np

## CLIMATE ZONES COLOR
climate_color_dict={'A': '#ff0000',
 'Af': '#960000',
 'Am': '#ff0000',
 'As': '#ff9999',
 'Aw': '#ffcccc',
 'B': '#ccaa54',
 'BWk': '#ffff63',
 'BWh': '#ffcc00',
 'BSk': '#ccaa54',
 'BSh': '#cc8c14',
 'C': '#007800',
 'Cfa': '#003300',
 'Cfb': '#004f00',
 'Cfc': '#007800',
 'Csa': '#00ff00',
 'Csb': '#95ff00',
 'Csc': '#c8ff00',
 'Cwa': '#b56400',
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
 'Dwa': '#c7b3c7',
 'Dwb': '#997fb3',
 'Dwc': '#8759b3',
 'Dwd': '#6d24b3',
 'E': '#6395ff',
 'EF': '#6395ff',
 'ET': '#63ffff'}



 ## EARTH DATA
G=6.674e-11 # (m^3 kg^-1 s^-2) Gravitational constant
M=5.9722e24 # (kg) Earth mass
a=6.378e6 # (m) semi major axis of the ellipse describing the Earth ~ Earth radius
rho_av=3*M/(4*np.pi*a**3) # (kg m^-3) average density of the Earth


version_data={'GLDAS21_NOAH36':2,'GLDAS21_CLSM25':2,'GLDAS21_VIC412':2,'GLDAS21':2,'MERRA2':2,'SSEBop':2,'GRACE_CSR':2,'GRUN':2,
'GRACE_JPL':1,'CPC':1,'TRMM':1,'GLDAS':1,'SEB':1,'MSWEP':1}

dict_fill_value={'GLDAS21_NOAH36':-9999,'GLDAS21_CLSM25':-9999,'GLDAS21_VIC412':-9999,
'GLDAS21':-9999,'MERRA2':-9999,'SSEBop':-9999,'GRUN':-9999,
'GRACE_JPL':-9999,'GRACE_CSR':-9999,'CPC':-9999,'TRMM':-9999,'GLDAS':0,'SEB':0,'MSWEP':-9999}


