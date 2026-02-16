#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Steven van Kesteren
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt




# Parameters
K0 = 0.5e-4     # forward rate constant dependent on I
K_rev = 0.005     # backward rate constant
A_tot =0.3     # total concentration of A + B
B0 = 0.0         # initial concentration of B


# Time points
t_eval = np.linspace(0, 1500, 1500)


I0 = np.ones(90)*150 # 90s High illumination
I1 = np.ones(110)*0 #110s light off
I2 = np.ones(1500-200)*16.5 #low illumination

# Array of intensities in mW
I_array = np.concatenate((I0,I1,I2))



# Interpolate I(t)
I_func = interp1d(t_eval, I_array, kind='linear', fill_value='extrapolate')

# Differential equation
def dB_dt(t, B):
    forward_rate = K0 * I_func(t) * (A_tot - B)
    backward_rate = K_rev * B
    return forward_rate - backward_rate

# Solve ODE
sol = solve_ivp(dB_dt, [t_eval[0], t_eval[-1]], [B0], t_eval=t_eval)




plt.figure(figsize=(8,5))
plt.plot(sol.t, sol.y[0], label='B(t)')

plt.xlabel('t (s)')
plt.ylabel('[SP]')
plt.legend()
plt.show()

#%%

def dB_dt(t, B):
    forward_rate = K0 * I_func(t) * (A_tot - B)
    backward_rate = K_rev * B
    return forward_rate - backward_rate


def simulate_SP_recovery(I_rec = 0,t_eval =np.linspace(0, 1500, 1500),K0 = 2.5e-5 ,K_rev = 0.004,A_tot =0.5,B0 = 0.0):
    
    
    
    
    # Time points
    t_eval = np.linspace(0, 1500, 1500)


    I0 = np.ones(90)*150 # 90s High illumination
    I1 = np.ones(110)*0 #180s
    I2 = np.ones(1500-200)*I_rec

    # Array of intensities in mW
    I_array = np.concatenate((I0,I1,I2))# could be any array



    # Interpolate I(t)
    I_func = interp1d(t_eval, I_array, kind='linear', fill_value='extrapolate')

    # Differential equation
    def dB_dt(t, B):
        forward_rate = K0 * I_func(t) * (A_tot - B)
        backward_rate = K_rev * B
        return forward_rate - backward_rate

    # Solve ODE
    sol = solve_ivp(dB_dt, [t_eval[0], t_eval[-1]], [B0], t_eval=t_eval)
    
    return sol.t,sol.y[0]
    

I_ls = [8,13,16,16.5]

results= []

mat = np.array([])
for I in I_ls:
    t, SP = simulate_SP_recovery(I)
    
    results.append({"t":t,"SP": SP})
    


figure_width = 7 # cm
figure_height = 5 # cm
left_right_magrin = 3 # cm
top_bottom_margin = 3 # cm

# Don't change
left   = left_right_magrin / figure_width # Percentage from height
bottom = top_bottom_margin / figure_height # Percentage from height3
width  = 1 - left*2
height = 1 - bottom*2
cm2inch = 1/2.54 # inch per cm

fig, ax1 = plt.subplots(figsize=(figure_width*cm2inch,figure_height*cm2inch), dpi = 500)
fig.tight_layout()


lines = ['k:','k-.','k--','k-']

for r,line in zip(results,lines):

    ax1.plot(r['t'],r['SP'],line,lw = 1)

    
    
    
#%%


figure_width = 9 # cm
figure_height = 8 # cm
left_right_magrin = 3 # cm
top_bottom_margin = 3 # cm

# Don't change
left   = left_right_magrin / figure_width # Percentage from height
bottom = top_bottom_margin / figure_height # Percentage from height3
width  = 1 - left*2
height = 1 - bottom*2
cm2inch = 1/2.54 # inch per cm

fig, ax1 = plt.subplots(figsize=(figure_width*cm2inch,figure_height*cm2inch), dpi = 500)
fig.tight_layout()


lines = ['k:','k-.','k--','k-']

for r,line,I in zip(results,lines, I_ls):
    SP = r['SP']
    SP = SP[90:]
    SP = np.where((SP >= 0.05) & (SP <= 0.06), SP, np.nan)
    
    ax1.plot(SP,line,lw = 1, label = '%.1f $\mu W/cm^2$'% I)
    

ax1.legend()
ax1.set(xlabel = 't (s)',ylabel = '[SP] mM'  )
plt.show()
