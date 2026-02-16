
# -*- coding: utf-8 -*-
"""
Crystal Growth Analysis Script

This script analyzes crystallization kinetics from GSD simulation trajectories.
It calculates the crystallinity fraction (q6 order parameter), fits kinetic
models (KJMA and others), and relates the crystallization rate to colloid
interaction potentials.

Author: Steven van Kesteren
Date: 2025-09-10
"""

# ============================================================================ #
# Imports
# ============================================================================ #
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

import gsd.fl
import freud

import openmm
from openmm import unit

from colloids.colloid_potentials_algebraic import ColloidPotentialsAlgebraic
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters
import numpy as np
import matplotlib.pyplot as plt
import pwlf  

# ============================================================================ #
# Crystal Analysis Functions
# ============================================================================ #
def calculate_crystallinity(file_path, frame_step=5, max_frame=2000,
                            q6_threshold=0.6, num_neighbors=6):
    """
    Compute crystallinity fraction from a GSD trajectory using the Steinhardt q6 order parameter.
    """
    trajectory = gsd.fl.open(file_path, 'r')
    frame_indices = range(0, max_frame, frame_step)
    crystallinity_fraction = np.zeros(len(frame_indices))

    # Large box to avoid periodicity
    box = freud.box.Box(30000, 30000, 30000)
    ql = freud.order.Steinhardt(6)

    for idx, frame in enumerate(frame_indices):
        positions = trajectory.read_chunk(frame, 'particles/position')
        q6_values = ql.compute((box, positions),
                               neighbors={"num_neighbors": num_neighbors}).particle_order
        crystallinity_fraction[idx] = np.nanmean(q6_values > q6_threshold)

    return crystallinity_fraction, np.array(frame_indices)






def simple_exp_piecewise_fit(x, y, n_segments=3, plot=True):
    """
    Automatically fit multiple linear regions in an -log(1-Y), X plot

    Parameters
    ----------
    x : array-like
        Time values (must be > 0).
    y : array-like
        Crystallinity fraction (0 < y < 1).
    n_segments : int
        Number of linear segments to fit (default = 3).
    plot : bool
        Whether to plot results.

    Returns
    -------
    slopes : list
        Avrami exponents (slopes) for each region.
    breaks : list
        Breakpoints in log-time separating regions.
    """
    # Avrami log–log transformation
    # logt = np.log(x)
    logt = x**0.5
    logy = -np.log(1 - y)

    # Piecewise linear fit
    model = pwlf.PiecewiseLinFit(logt, logy,disp_res=True)
    breaks = model.fit(n_segments)
    slopes = model.slopes

    if plot:
        plt.figure(figsize=(5,4), dpi=150)
        plt.scatter(logt, logy, s=10, c="k", alpha=0.5, label="Data")
        x_hat = np.linspace(logt.min(), logt.max(), 200)
        y_hat = model.predict(x_hat)
        plt.plot(x_hat, y_hat, "r-", lw=2, label="Piecewise fit")
        for br in breaks[1:-1]:
            plt.axvline(br, ls="--", c="gray", lw=1)
        for i, slope in enumerate(slopes):
            plt.text((breaks[i]+breaks[i+1])/2, 
                     np.interp((breaks[i]+breaks[i+1])/2, logt, logy),
                     f"n ≈ {slope:.2f}",
                     ha="center", va="bottom", fontsize=8, color="blue")
        plt.xlabel("t")
        plt.ylabel("-ln(1 - X)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return slopes, breaks,logt, logy



def avrami_loglog_analysis(x, y, t_start=36, t_end=50, plot=True):
    """
    Perform Avrami log–log analysis: ln(-ln(1-Y)) vs ln(t).

    Parameters
    ----------
    x : np.ndarray
        Time values (s).
    y : np.ndarray
        Crystallinity fraction.
    t_start : int
        Start index for linear regime fitting.
    t_end : int
        End index for linear regime fitting.
    plot : bool
        If True, plot analysis.

    Returns
    -------
    slope : float
        Avrami exponent (growth dimensionality).
    intercept : float
        Fit intercept.
    """
    y2 = -np.log(1 - y)
    y3 = np.log(y2)
    x_log = np.log(x)

    # Select linear region
    x_fit = np.log(x[t_start:t_end] - x[t_start])
    y_fit = y3[t_start:t_end]

    coeffs = np.polyfit(x_fit, y_fit, 1)
    slope, intercept = coeffs

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
        ax1.plot(x, y)
        ax1.set(ylabel="Crystallinity", xlabel="t (s)")

        ax2.plot(x_log, y2)
        ax2.set(ylabel="-ln(1-Y)", xlabel="ln(t)")

        ax3.plot(x_log, y3)
        ax3.plot(x_fit, intercept + slope * x_fit, "--", label=f"slope={slope:.2f}")
        ax3.set(ylabel="ln(-ln(1-Y))", xlabel="ln(t)")
        ax3.legend()

        plt.tight_layout()
        plt.show()

    return slope, intercept
def simple_exp_segmented_fit(t, X, n_segments=3, plot=True, min_cryst=0.01, n= 4):
    """
    Perform piecewise exponential fits treating each segment as independent.
    
    Parameters
    ----------
    t : array_like
        Time array
    X : array_like
        Crystallinity fraction (0 < X < 1)
    n_segments : int
        Number of segments for piecewise linear fit
    plot : bool
        Whether to plot the fits
    min_cryst : float
        Minimum X to avoid log(0)
    
    Returns
    -------
    segment_results : list of dicts
        Each dict contains {'slope': n, 'intercept': lnK, 't_fit', 'X_fit'}
    pwlf_breaks : list
        Breakpoints in log-time space
    """
    # --- preprocess ---
    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    
    # clip crystallinity to avoid log(0) or log(1)
    X = np.clip(X, min_cryst, 0.999999)
    
    
    logt = t**0.5
    # logt = np.log(t)
    avrami_y = -np.log(1 - X)
    
    # remove non-finite points
    mask = np.isfinite(logt) & np.isfinite(avrami_y)
    logt = logt[mask]
    avrami_y = avrami_y[mask]
    
    # --- piecewise linear fit ---
    model = pwlf.PiecewiseLinFit(logt, avrami_y)
    breaks = model.fit(n_segments)
    
    segment_results = {}
    for i in range(len(breaks)-1):
        seg_mask = (logt >= breaks[i]) & (logt <= breaks[i+1])
        t_seg = t[seg_mask]
        X_seg = X[seg_mask]
        
        if len(t_seg) < 5:
            continue  # skip too short segments
        
        # shift time to t=0 for this segment
        # t_shift = np.log(t_seg[1:] - t_seg[0])**n
        t_shift = (t_seg[1:] - t_seg[0])**0.5
        avrami_y_seg = -np.log(1 - X_seg[1:])
        
        slope, intercept = np.polyfit(t_shift, avrami_y_seg, 1)
        
        segment_results[i+1] = {
            'slope': slope,
            'intercept': intercept,
            't_raw': t_seg,
            'X_raw': X_seg,
            't_shift': t_shift,
            'X_fit':   avrami_y_seg
        }
        
        # optional plotting
        if plot:
            plt.plot(t_shift, avrami_y_seg, 'o', label=f'Segment {i+1} data')
            plt.plot(t_shift, intercept + slope*t_shift, '-', 
                     label=f'Segment {i+1} fit (n={slope:.2f})')
    
    if plot:
        plt.xlabel('time (s)')
        plt.ylabel('-ln(1-X]')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return segment_results, breaks

def avrami_segmented_fit(t, X, n_segments=3, plot=True, min_cryst=0.01):
    """
    Perform piecewise Avrami fits treating each segment as independent.
    
    Parameters
    ----------
    t : array_like
        Time array
    X : array_like
        Crystallinity fraction (0 < X < 1)
    n_segments : int
        Number of segments for piecewise linear fit
    plot : bool
        Whether to plot the fits
    min_cryst : float
        Minimum X to avoid log(0)
    
    Returns
    -------
    segment_results : list of dicts
        Each dict contains {'slope': n, 'intercept': lnK, 't_fit', 'X_fit'}
    pwlf_breaks : list
        Breakpoints in log-time space
    """
    # --- preprocess ---
    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    
    # clip crystallinity to avoid log(0) or log(1)
    X = np.clip(X, min_cryst, 0.999999)
    
    logt = np.log(t)
    # logt = t
    avrami_y = np.log(-np.log(1 - X))
    
    # remove non-finite points
    mask = np.isfinite(logt) & np.isfinite(avrami_y)
    logt = logt[mask]
    avrami_y = avrami_y[mask]
    
    # --- piecewise linear fit ---
    model = pwlf.PiecewiseLinFit(logt, avrami_y)
    breaks = model.fit(n_segments)
    
    segment_results = []
    for i in range(len(breaks)-1):
        # segment indices
        seg_mask = (logt >= breaks[i]) & (logt <= breaks[i+1])
        t_seg = t[seg_mask]
        X_seg = X[seg_mask]
        
        if len(t_seg) < 5:
            continue  # skip too short segments
        
        # shift time to t=0 for this segment
        t_shift = t_seg[1:] - t_seg[0]
        logt_shift = np.log(t_shift)
        avrami_y_seg = np.log(-np.log(1 - X_seg[1:]))
        
        # avrami_y_seg = -np.log(1 - X_seg[1:])
        # linear regression
        slope, intercept = np.polyfit(logt_shift, avrami_y_seg, 1)
        
        segment_results.append({
            'slope': slope,
            'intercept': intercept,
            't_fit': t_shift,
            'X_fit': X_seg
        })
        
        # optional plotting
        if plot:
            plt.plot(logt_shift, avrami_y_seg, 'o', label=f'Segment {i+1} data')
            plt.plot(logt_shift, intercept + slope*logt_shift, '-', label=f'Segment {i+1} fit (n={slope:.2f})')
    
    if plot:
        plt.xlabel('ln t (s)')
        plt.ylabel('ln[-ln(1-X)]')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return segment_results, breaks

def avrami_piecewise_fit(x, y, n_segments=3, plot=True):
    """
    Automatically fit multiple linear regions in an Avrami log–log plot.

    Parameters
    ----------
    x : array-like
        Time values (must be > 0).
    y : array-like
        Crystallinity fraction (0 < y < 1).
    n_segments : int
        Number of linear segments to fit (default = 3).
    plot : bool
        Whether to plot results.

    Returns
    -------
    slopes : list
        Avrami exponents (slopes) for each region.
    breaks : list
        Breakpoints in log-time separating regions.
    """
    # Avrami log–log transformation
    logt = np.log(x)
    # logt = x
    avrami_y = np.log(-np.log(1 - y))

    # Piecewise linear fit
    model = pwlf.PiecewiseLinFit(logt, avrami_y,disp_res=True)
    breaks = model.fit(n_segments)
    slopes = model.slopes

    if plot:
        plt.figure(figsize=(5,4), dpi=150)
        plt.scatter(logt, avrami_y, s=10, c="k", alpha=0.5, label="Data")
        x_hat = np.linspace(logt.min(), logt.max(), 200)
        y_hat = model.predict(x_hat)
        plt.plot(x_hat, y_hat, "r-", lw=2, label="Piecewise fit")
        for br in breaks[1:-1]:
            plt.axvline(br, ls="--", c="gray", lw=1)
        for i, slope in enumerate(slopes):
            plt.text((breaks[i]+breaks[i+1])/2, 
                     np.interp((breaks[i]+breaks[i+1])/2, logt, avrami_y),
                     f"n ≈ {slope:.2f}",
                     ha="center", va="bottom", fontsize=8, color="blue")
        plt.xlabel("ln t")
        plt.ylabel("ln[-ln(1 - X)]")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return slopes, breaks

# ============================================================================ #
# PACS Potential Computation
# ============================================================================ #
def get_pacs_potential(h_values, radius_one, radius_two,
                       surface_potential_one, surface_potential_two,
                       parameters, platform_name="Reference"):
    """
    Compute PACS (Polymer–Colloid–Salt) potential energies for two colloids.
    """
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors([10000.0, 0.0, 0.0],
                                        [0.0, 10000.0, 0.0],
                                        [0.0, 0.0, 10000.0])

    colloid_potentials = ColloidPotentialsAlgebraic(parameters, use_log=False)
    system.addParticle(1.0)
    colloid_potentials.add_particle(radius=radius_one, surface_potential=surface_potential_one)
    system.addParticle(1.0)
    colloid_potentials.add_particle(radius=radius_two, surface_potential=surface_potential_two)

    for potential in colloid_potentials.yield_potentials():
        system.addForce(potential)

    platform = openmm.Platform.getPlatformByName(platform_name)
    integrator = openmm.LangevinIntegrator(parameters.temperature.value_in_unit(unit.kelvin), 0.0, 0.0)
    context = openmm.Context(system, integrator, platform)

    energies = np.zeros(len(h_values))
    for i, h in enumerate(h_values):
        context.setPositions([[radius_one + radius_two + h, 0, 0], [0, 0, 0]])
        state = context.getState(getEnergy=True)
        energies[i] = state.getPotentialEnergy() / (
            unit.BOLTZMANN_CONSTANT_kB * parameters.temperature * unit.AVOGADRO_CONSTANT_NA
        )
    return energies


# ============================================================================ #
# Example Driver Script
# ============================================================================ #
import pandas as pd

def process_batch(file_paths, potentials, frame_skip=20, sf=None,frame_step=5):
    """
    Batch process multiple trajectories and extract kinetic parameters.

    Parameters
    ----------
    file_paths : list[str]
        Paths to GSD trajectory files.
    potentials : list[float]
        Associated surface potentials (mV).
    frame_skip : int, optional
        Skip first N frames before fitting (default = 20).
    sf : float, optional
        Scaling factor to convert frames to seconds. If None, frames are used directly.
    fit_func : callable
        Kinetic model function to fit (default = kjma).
    p0 : list
        Initial guess for fit parameters.
    t_avrami : tuple[int, int]
        Frame indices for Avrami log–log fitting window.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with results for each trajectory:
        [potential, k_fit, scale_fit, avrami_exponent]
    """
    records = []

    for file_path, pot in zip(file_paths, potentials):
        print(["processing:   " + file_path])
        # Compute crystallinity
        cryst_frac, frames = calculate_crystallinity(file_path,frame_step=frame_step)
        # Convert frames to time
        x = frames[frame_skip:] - frames[frame_skip]
        if sf is not None:
            x = x * 500000 * sf
            x = x + 1
        y = cryst_frac[frame_skip:]

        # Fit kinetics
        # coeffs = fit_kinetics(x, y, fit_func=fit_func, p0=p0, plot=False)
        # k_fit, scale_fit = coeffs

        # Avrami exponent
        # slope, intercept = avrami_loglog_analysis(x, y,
        #                                           t_start=t_avrami[0],
        #                                           t_end=t_avrami[1],
                                                  # plot=False)
        simple_exp_piecewise_fit(x, y)

        slopes, breaks,logt, logy = simple_exp_piecewise_fit(x, y)
        simple_exp_segmented_fit(x,y)

        records.append({
            "potential": pot,
            "exponent": slopes,
            "breaks": breaks,
            "logt": logt,
            "logy": logy
        })
        
    results_df = pd.DataFrame(records)
    return results_df


# ============================================================================ #
# Plotting Helpers
# ============================================================================ #
def plot_crystallinity_curves(file_paths, potentials, frame_skip=20, sf=None,cmap = 'viridis'):
    """
    Plot crystallinity growth curves for all potentials.
    """
    plt.figure(figsize=(4, 2.5), dpi=300)

    cmap_m=plt.get_cmap(cmap)
    # lin_map = mpl.colors.LinearSegmentedColormap.from_list(cmap, cmap_m.colors[:])
  
    # cmap = plasma_map

    colors = cmap_m(np.linspace(240,10,len(potentials)).astype(int))
    
    for file_path, pot,color in zip(file_paths, potentials,colors):
        cryst_frac, frames = calculate_crystallinity(file_path)
        x = frames[frame_skip:] - frames[frame_skip]
        if sf is not None:
            x = x * 500000 * sf
        y = cryst_frac[frame_skip:]
        plt.plot(x, y, label=f"{pot} mV",c = color)    

    plt.xlabel("t (s)" if sf else "Frame")
    plt.ylabel("Crystallinity (q6 > 0.6)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def plot_crystallinity_curves_ln(file_paths, potentials, frame_skip=0, sf=None,cmap = 'viridis'):
    """
    Plot crystallinity growth curves for all potentials.
    """
    plt.figure(figsize=(5, 4), dpi=300)

    cmap_m=plt.get_cmap(cmap)
    # lin_map = mpl.colors.LinearSegmentedColormap.from_list(cmap, cmap_m.colors[:])
  
    # cmap = plasma_map

    colors = cmap_m(np.linspace(240,10,len(potentials)).astype(int))
    
    for file_path, pot,color in zip(file_paths, potentials,colors):
        cryst_frac, frames = calculate_crystallinity(file_path)
        x = frames[frame_skip:] - frames[frame_skip+1]
        if sf is not None:
            x = x * 500000 * sf
        y = cryst_frac[frame_skip:]
        
        plt.plot(-np.log(1-x), np.sqrt(y), label=f"{pot} mV",c = color)    

    plt.xlabel("t (s)" if sf else "Frame")
    plt.ylabel("Crystallinity (q6 > 0.6)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def plot_fit_results(results, n_segments=3,seg = 1,avrami = False,cmap = 'viridis', n =0.5):
    """
    Plot fitted kinetic parameters vs potential.

    Parameters
    ----------
    results: from batch processing
    
    n_segments: number of linear segments to fit
    
    seg: index of the segment of interest
    
    avrami: switches between exponential and avrami type fitting
    
    cmap: cmap that is used
    
    
    """
    
    import matplotlib as mpl
    import scipy
    
    
    plt.figure(figsize=(5, 4), dpi=150)
    fig,axs=plt.subplots(ncols = 3, nrows = 2, figsize=(8,5), dpi=300)
    fig2,axs2 = plt.subplots(ncols = 3, nrows = 2, figsize=(8,5), dpi=300)
    # fig3,axs3 = plt.subplots(ncols = 1,nrows = 1,figsize = (4,4), dpi=300)
    axs = axs.flatten()
    axs2 =axs2.flatten()
    fig4, (ax2,ax1) = plt.subplots(ncols = 2,figsize=(5,3), dpi = 500)
    fig4.tight_layout(pad =2)
    
    cmap_m=plt.get_cmap(cmap)
    # lin_map = mpl.colors.LinearSegmentedColormap.from_list(cmap, cmap_m.colors[:])
  
    # cmap = plasma_map

    colors = cmap_m(np.linspace(240,10,len(results)).astype(int))
    f = []
    fit = []
    for i in range(len(results)):
        
        logt = results['logt'][i]
       
        logy = results['logy'][i]
        if avrami:
            logt = results['logt'][i]
            logy = np.log(logy)
        slopes = results['exponent'][i]
        breaks = results['breaks'][i]
        
        
    
        model = pwlf.PiecewiseLinFit(logt, logy,disp_res=True)
        breaks = model.fit(n_segments)
        
        # if results['potential'][i] == 22.5:
        #     breaks = model.fit(2)
        slopes = model.slopes
        x_hat = np.linspace(logt.min(), logt.max(), 200)
        y_hat = model.predict(x_hat)
        
        
        axs[i].plot(x_hat, y_hat, "r--", lw=1, label="fit")
        axs[i].scatter(logt, logy, s=10, c=colors[i], alpha=0.5, label="data")
        
        
    
        
        print(breaks)
        for br in breaks[1:-1]:
            axs[i].axvline(br, ls="--", c="gray", lw=1)
        # for i, slope in enumerate(slopes):
            # axs[i].text((breaks[i]+breaks[i+1])/2, 
            #          np.interp((breaks[i]+breaks[i+1])/2, logt, logy),
            #          f"k ≈ {slope:.2f}",
            #          ha="center", va="bottom", fontsize=8, color="blue")
            
        axs[i].set(xlabel="$\sqrt{t}$",ylabel="-ln(1 - X)")
        if avrami:
            axs[i].set(xlabel="ln(t)",ylabel="ln(-ln(1 - X))")
        ax2.set(xlabel="$\sqrt{t}$",ylabel="-ln(1 - X)")
        axs[i].set(title = "%.1f mV" % results['potential'][i])
        # plt.legend()
        
        
        # segment indices
        seg_mask = (logt >= breaks[seg]) & (logt <= breaks[seg+1])
        print(breaks[seg])
        t_seg = logt[seg_mask]**2
        # t_seg = logt[seg_mask]
        X_seg = logy[seg_mask]
        
        if len(t_seg) < 3:
            continue  # skip too short segments
        
        # shift time to t=0 for this segment
        t_shift = t_seg[1:] - t_seg[0]
        # print(t_shift)
        logt_shift = t_shift**n
        # logt_shift = np.log(t_shift)**4
        X_seg = X_seg[1:]
        # linear regression
        print(logt_shift)
        print(X_seg)
        # slope, intercept = np.polyfit(logt_shift,X_seg, 1)
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(logt_shift[logt_shift>10],X_seg[logt_shift>10])

        # optional plotting
        
        axs2[i].plot(logt_shift, X_seg, 'o',c=colors[i],alpha = 0.6, label=f'$R^2$:%0.2f' % r_value**2)
        axs2[i].plot(logt_shift, intercept + slope*logt_shift, 'r--')
        axs2[i].set(xlabel="$\sqrt{t}$",ylabel="-ln(1 - X)")
        axs2[i].set(title = "%.1f mV" % results['potential'][i])
        axs2[i].legend()
        # axs3.plot(logt_shift, X_seg, 'o',c=colors[i],label="%.1f mV" % results['potential'][i]+f'  fit (k={slope:.2f})')
        ax2.plot(logt_shift, X_seg, 'o',c=colors[i],label="%.1f mV" % results['potential'][i],alpha = 0.6)
        ax2.plot(logt_shift, intercept + slope*logt_shift, 'r--')
        print("%.1f mV" % results['potential'][i]+f'  fit (k={slope:.2f})')
        ax2.set(xlabel="$\sqrt{t}$",ylabel="-ln(1 - X)")
        if avrami:
            
            ax2.set(xlabel="ln(t)",ylabel="ln(-ln(1 - X))")
    
        # ax2.legend()
        
        ax1.errorbar(results['potential'][i],slope,std_err,fmt='ko', mfc = colors[i],alpha = 0.6)

        print(r_value)




       
        f = np.append(f,results['potential'][i])
        fit = np.append(fit,slope)
        
    

    fig.tight_layout()
    fig2.tight_layout()
    # fig3.tight_layout()
    

    figure_width = 5.5 # cm
    figure_height =5 # cm
    left_right_magrin = 3 # cm
    top_bottom_margin = 3 # cm
       
    # Don't change
    left   = left_right_magrin / figure_width # Percentage from height
    bottom = top_bottom_margin / figure_height # Percentage from height3
    width  = 1 - left*2
    height = 1 - bottom*2
    cm2inch = 1/2.54 # inch per cm
       
  
    
    np.polyfit(f[-4:], fit[-4:], 1)
    a = np.polyfit(f[-3:], fit[-3:], 1)
    b = np.polyfit( fit[-4:],f[-4:], 1)
    fit_lm = np.poly1d(a)

    ax1.set(ylabel= r'$K_{crystal}$ ($s^{-1/2}$)')
    ax1.set(xlabel= r'$\zeta_p$ (mV)')
    ax1.set_xlim(45,0)
    
    ax1.set(ylabel= r'$K_{crystal}$ ($s^{-1/2}$)')
    if avrami:
        
        ax1.set(ylabel= r'$n_{Avrami}$')
        
    ax1.set(xlabel= r'$\zeta_p$ (mV)')
    plt.show()


#%%
def plot_fit_results_final(results, n_segments=3,seg = 1,avrami = False,cmap = 'viridis', n =0.5):
    """
    Plot fitted kinetic parameters vs potential.

    Parameters
    ----------
    results: from batch processing
    
    n_segments: number of linear segments to fit
    
    seg: index of the segment of interest
    
    avrami: switches between exponential and avrami type fitting
    
    cmap: cmap that is used
    
    
    """
    
    import matplotlib as mpl
    import scipy
    
    figure_width = 4.5 # cm
    figure_height =4 # cm
    left_right_magrin = 3 # cm
    top_bottom_margin = 3 # cm
       
    # Don't change
    left   = left_right_magrin / figure_width # Percentage from height
    bottom = top_bottom_margin / figure_height # Percentage from height3
    width  = 1 - left*2
    height = 1 - bottom*2
    cm2inch = 1/2.54 # inch per cm
    
    fig,axs= plt.subplots(figsize=(figure_width * cm2inch, figure_height * cm2inch), dpi=300)
    fig.tight_layout()
    cmap_m=plt.get_cmap(cmap)
    # lin_map = mpl.colors.LinearSegmentedColormap.from_list(cmap, cmap_m.colors[:])
  
    # cmap = plasma_map

    colors = cmap_m(np.linspace(240,10,len(results)-3).astype(int))
    f = []
    fit = []
    for i in range(len(results)):
        
        logt = results['logt'][i]
       
        logy = results['logy'][i]
        if avrami:
            logt = results['logt'][i]
            logy = np.log(logy)
        slopes = results['exponent'][i]
        breaks = results['breaks'][i]
        
        
    
        model = pwlf.PiecewiseLinFit(logt, logy,disp_res=True)
        breaks = model.fit(n_segments)
        
        # if results['potential'][i] == 22.5:
        #     breaks = model.fit(2)
        slopes = model.slopes
        x_hat = np.linspace(logt.min(), logt.max(), 200)
        y_hat = model.predict(x_hat)
        

        
   

        # plt.legend()
        
        
        # segment indices
        seg_mask = (logt >= breaks[seg]) & (logt <= breaks[seg+1])
        t_seg = logt[seg_mask]**2
        # t_seg = logt[seg_mask]
        X_seg = logy[seg_mask]
        
        if len(t_seg) < 3:
            continue  # skip too short segments
        
        # shift time to t=0 for this segment
        t_shift = t_seg[1:] - t_seg[0]
        logt_shift = t_shift**n
        # logt_shift = np.log(t_shift)**4
        X_seg = X_seg[1:]
        # linear regression

        # slope, intercept = np.polyfit(logt_shift,X_seg, 1)
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(logt_shift[logt_shift>10],X_seg[logt_shift>10])

        # optional plotting
        
        
    
        # ax2.legend()
        if i<3:
            axs.errorbar(results['potential'][i],slope,std_err,fmt='ko',mec = 'k',mfc = 'white',alpha = 0.6)
        else:
            axs.errorbar(results['potential'][i],slope,std_err,fmt='ko', mfc = colors[i-3],alpha = 0.6)

            
        print(r_value)




       
        f = np.append(f,results['potential'][i])
        fit = np.append(fit,slope)
        
    

    
    


       
  
    
    np.polyfit(f[-4:], fit[-4:], 1)
    a = np.polyfit(f[-4:], fit[-4:], 1)
    b = np.polyfit( fit[-4:],f[-4:], 1)
    fit_lm = np.poly1d(a)
    # axs.plot(f[-4:],fit_lm(f[-4:]),'k--')
    axs.plot(17,0.0038,'kx')
    axs.set(ylabel= r'$K_{crystal}$ ($s^{-\frac{1}{2}}$)')
    axs.set(xlabel= r'$\zeta_p$ (mV)')
    axs.set_xlim(45,0)
    
    # axs.set(ylabel= r'$K_{crystal}$ ($s^{-1/2}$)')

    axs.set(xlabel= r'$\zeta_p$ (mV)')
    fig.savefig(dir_path+r'/boo_exp.svg')
    plt.show()
    return fig
#%%
# ============================================================================ #
# Example Batch Driver
# ============================================================================ #
dir_path =  os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    # ---- Example: Analyze one trajectory ----
    
    dir_path =  os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.normpath(dir_path + r"/Simulations/oscillate_surfacepotentialpositive+50to+17/trajectory.gsd")
   
    

    cryst_frac, frames = calculate_crystallinity(file_path,frame_step=3)

    # Convert frames to time (example scaling)
    sf = (0.0015 / 1.18e-12) * (0.015e-12)
    time = (frames - frames[0]) * 500000 * sf  # align to frame 0
    y = cryst_frac[:]
    
    
    
    
    x = frames
    x = x + 1
    y = cryst_frac
    
    # slopes, breaks = avrami_piecewise_fit(x, y, n_segments=3)
    # slopes, breaks = avrami_segmented_fit(x, y, n_segments=3)
    simple_exp_piecewise_fit(x, y)
    
    # coeffs = fit_kinetics(time, y, fit_func=kjma, p0=[0.01, 2])
    # slope, intercept = avrami_loglog_analysis(time, y)

    # print(f"Fitted k = {coeffs[0]:.3e}, scale = {coeffs[1]:.2f}")
    # print(f"Avrami slope = {slopes[1]:.2f}")

#%%Process and show selected trajectories

if __name__ == "__main__":
    # Example input lists
    
    dir_path =  os.path.dirname(os.path.abspath(__file__))

    file_paths = [
        '/Simulations/oscillate_surfacepotentialpositive+50to+25/trajectory.gsd',
        "/Simulations/oscillate_surfacepotentialpositive+50to+22.5/trajectory.gsd",
        "/Simulations/oscillate_surfacepotentialpositive+50to+20/trajectory.gsd",
        "/Simulations/oscillate_surfacepotentialpositive+50to+18/trajectory.gsd",
        "/Simulations/oscillate_surfacepotentialpositive+50to+17/trajectory.gsd",
        "/Simulations/oscillate_surfacepotentialpositive+50to+15/trajectory.gsd",
    ]
    
    file_paths = [dir_path + f for f in file_paths ]
    
    potentials = [25,22.5,20,18,17,15]

    # Scaling factor (adjust to your system)
    sf = (0.0015 / 1.18e-12) * (0.015e-12)

    # Process all trajectories
    results = process_batch(file_paths, potentials, sf=sf,frame_step=2)

    # Show results
    print(results)
    plot_fit_results(results)
    plot_fit_results(results,avrami=True)

    # Plot curves
    plot_crystallinity_curves(file_paths, potentials, sf=sf)
    plot_crystallinity_curves_ln(file_paths, potentials, sf=sf)

#%% Process all trajectories
if __name__ == "__main__":
    # Example input lists
    
    dir_path =  os.path.dirname(os.path.abspath(__file__))

    file_paths = [
        '/Simulations/oscillate_surfacepotentialpositive+50to+40/trajectory.gsd',
        '/Simulations/oscillate_surfacepotentialpositive+50to+30/trajectory.gsd',
        '/Simulations/oscillate_surfacepotentialpositive+50to+25/trajectory.gsd',
        "/Simulations/oscillate_surfacepotentialpositive+50to+22.5/trajectory.gsd",
        "/Simulations/oscillate_surfacepotentialpositive+50to+20/trajectory.gsd",
        "/Simulations/oscillate_surfacepotentialpositive+50to+18/trajectory.gsd",
        "/Simulations/oscillate_surfacepotentialpositive+50to+17/trajectory.gsd",
        "/Simulations/oscillate_surfacepotentialpositive+50to+15/trajectory.gsd",
    ]
    
    file_paths = [dir_path + f for f in file_paths ]
    
    potentials = [40,30,25,22.5,20,18,17,15]

    # Scaling factor (adjust to your system)
    sf = (0.0015 / 1.18e-12) * (0.015e-12)

    # Process all trajectories
    results = process_batch(file_paths, potentials, sf=sf,frame_step=2)

    # Show results
    print(results)


#%%
fig = plot_fit_results_final(results)


#%%TESTS
#Test data

k1 = 0.0002
n1 = 0.5
k2 = 0.002
n2 =0.5
t = np.linspace(0,10000,1000)
y0= np.ones(100)*0.2
y1 = (1 - np.exp(-k1*t[:100]**n1))+0.2
y2 = (1 - np.exp(-k2*t[:800]**n2))*0.2+max(y1)

y = np.concatenate((y0,y1,y2))

plt.plot(y)
plt.show()
plt.plot(np.log(t),-np.log(1-y))

t = t+1

avrami_piecewise_fit(t, y, n_segments=3)
avrami_segmented_fit(t, y, n_segments=3)
simple_exp_piecewise_fit(t, y)
a,b = simple_exp_segmented_fit(t, y,n=1)


#%%
#Test data 2

k1 = 0.1
n1 = 1
k2 = 0.003
n2 =0.5
t = np.linspace(0,10000,1000)
y0= np.ones(100)*0.2
y1 = (1 - t[1:101]**-k1)+0.2
y2 = (1 - np.exp(-k2*t[:800]**n2))*0.2+max(y1)

y = np.concatenate((y0,y1,y2))

plt.plot(y)
plt.show()
plt.plot(np.log(t),-np.log(1-y))

t = t+1

avrami_piecewise_fit(t, y, n_segments=3)
avrami_segmented_fit(t, y, n_segments=3)
simple_exp_piecewise_fit(t, y)
a,b = simple_exp_segmented_fit(t, y)



