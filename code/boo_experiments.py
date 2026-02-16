# -*- coding: utf-8 -*-
"""
Gel-to-Crystal transition script

This script analyzes crystallization kinetics from experimentally obtained particle transitions. It calculates the 2D bond-order parameters psi4 and psi6 and uses that to compute the crystal fraction. 

It fur
Author: Steven van Kesteren
Date: 2025-09-10
"""

# ============================================================================ #
# Imports
# ============================================================================ #

import numpy as np
import freud
import pandas as pd
from bokeh.io import output_notebook
from bokeh.plotting import figure, show

import matplotlib as mpl
import matplotlib.pyplot as plt






#%%psi 4 and psi 6 over time for non-classical crystallization
import os
dir_path =  os.path.dirname(os.path.abspath(__file__))

file_path =dir_path + r'/Confocal_data/C1-standard_0718_allspots.csv'
data = pd.read_csv(file_path)


ii = 0
r_ncn = range(10,731,3)
psi6_ncn = np.zeros(len(r_ncn))
psi4_ncn = np.zeros(len(r_ncn))
box = freud.box.Box(50,50,is2D = True)

for f in r_ncn:
    print(f)
    # fig, ax = plt.subplots(1, 1)
    x_1 = np.array(data[data['FRAME']==f]['POSITION_X'])
    y_1 = np.array(data[data['FRAME']==f]['POSITION_Y'])

    
    points =np.array([x_1,y_1,np.zeros(len(x_1))]).T
     
     
     
    sq_order = freud.order.Hexatic(k=4)
    sq_order.compute(system=(box, points),neighbors={"num_neighbors": 4})
     
    hex_order = freud.order.Hexatic(k=6)
    hex_order.compute(system=(box, points),neighbors={"num_neighbors": 6})
    psi_6 = hex_order.particle_order
    psi6_ncn[ii]= np.nanmean(abs(psi_6))
    psi_4 = sq_order.particle_order

    psi4_ncn[ii] =(abs(psi_4)>0.7).mean()
    ii+=1

#Frame rate of the experiment
t = np.array(r_ncn)*2.57 
#%% quick plot


fig,axs = plt.subplots(dpi = 300)
fig.tight_layout()

axs.plot(t,psi4_ncn)

plt.show()


#%%plot the slope
import scipy
sqrt_t = np.sqrt(t-t[0])[:]
log_x = -np.log(1-psi4_ncn)[:]


fig,axs = plt.subplots(figsize = (3,3),dpi = 300)
fig.tight_layout()

axs.plot(sqrt_t,log_x,'ko',alpha = 0.1)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(sqrt_t[10:],log_x[10:])


axs.plot(sqrt_t,intercept + slope*sqrt_t,'r--')
axs.set(xlabel="$\sqrt{t}$",ylabel="-ln(1 - X)")




plt.show()
print(slope)