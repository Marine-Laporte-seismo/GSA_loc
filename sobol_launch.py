#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:42:38 2022
main code of GSA-launch 
@author: laportem
"""

##############################################################################
### Scripts 


## Choose one of the two earthquake location algorithm
algo = 'H71' #('H71')


if algo == 'NLL':
    from locate_nll import Locate
elif algo == 'H71':
    from locate_h71 import Locate
else:
    sys.exit("Not known: Please choose between 'NLL' or 'H71' for the earthquake location algoritm")
    
## Plotting scripts
from plotting.sobol_radial_plot import plot_sobol_indices
from plotting.plot_elementary_effects import filter_simulations, EE_subplots, EE_subplots_imshow, EE_hist
from plotting.ellipse_func import  confidence_ellipse

## Libraries
import SALib
from SALib.sample import saltelli
from SALib.test_functions import Ishigami
from SALib.analyze import sobol
from SALib.analyze import morris
from io import StringIO
import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import time 
import os
import seaborn as sns
import logging
import sys
import math
import scipy
from datetime import datetime

from math import sin, cos,sqrt

t0=time.time()
now = datetime.now()
original_stdout = sys.stdout

##############################################################################
#-------------------------- Define random variables--------------------------#
##############################################################################
# Set the number of samples (must be a power of 2)
# n_samples = n *(2*num_vars+2)
n = 2**10 # 2**13 => 245760 samples with second-order interactions; 131072 without

# Define GSA Random Variables (RV)
print(">> Start Sobol Analysis")
problem = {
    'num_vars': 14,
    'names': ['Gap1', 'Dmin', 'Dmoy', 'Emoy', 'H1', 'Vp1', 'Vp2', 'Vp3', 'Vp/Vs',
              'ErrP', 'ErrS', 'ErrP_close', 'ErrS_close', 'nb_S'],
    'bounds': [
        [0, 1], [0, 1], [0, 1], [0, 1],       # adimensional 0-100%
        [-5, 5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0, 0.023],  # km, km/s, etc.
        [0, 0.05], [0, 0.15], [0, 0.05], [0, 0.15], [0.5, 1]  # time and adimensional
    ],
    'dists': [
        'unif', 'unif', 'unif', 'unif',      # Uniform distributions
        'unif', 'unif', 'unif', 'unif', 'norm',  # Gaussian for Vp/Vs
        'norm', 'norm', 'norm', 'norm', 'unif'  # Gaussian for errors, Uniform for nb_S
    ]
}

# Generate samples with the Saltelli sampler
# (add : calc_second_order=False to remove inetractions)
samples = saltelli.sample(problem, n)  # With interactions 
n_samples = samples.shape[0]
print(f"> Problem defined for {n_samples} simulations")


##############################################################################
#-------------------------- Define simulation parameters----------------- ---#
##############################################################################
# Simulation parameters
lat_synth, lon_synth = 0, 0   # Earthquake location (0,0)
depth_synth = 25              # Earthquake depth (km)
name_simu = f"SOBOL_{algo}_Z_{depth_synth}km_N{int(np.log2(n))}_{n_samples}_Vars_{problem['num_vars']}"
os.system('mkdir ./out/'+name_simu)
##############################################################################
#-------------------------- Define Outputs------------------------------- ---#
##############################################################################
# Define output arrays for the simulation results
La, Lo, Z, RMS, DT, R0 = (np.zeros(n_samples) for _ in range(6))
delta_h, delta_z, ErrH, ErrZ = (np.zeros(n_samples) for _ in range(4))
# Output directory
path_out = 'out/'+name_simu


##############################################################################
#-------------------------- Simulation phase---------------------------------#
##############################################################################



# Start of Simulation Phase : relocation over the "samples" exploration scheme
# n_samples iterations
print(">> Starting Simulation Phase")
start_phase = time.time()

Err_Lon, Err_Lat = [], [] #to have latitude and longitude error instead of horizontal errors

# Run simulations
for i, X in enumerate(samples):
    deb_simu = time.time()
    variables_simu = ', '.join(map(str, X))
    print(f". Simulation {i+1}/{n_samples} : Variables -> {variables_simu}")

    # Run simulation and calculate outputs
    La[i], Lo[i], Z[i], RMS[i], DT[i], delta_h[i], delta_z[i] = Locate(X, depth_synth)
    
    # Calculate horizontal and vertical differences with inintial loc 
    # Different than delta_h and delta_z which are the location uncertainties
    ErrH[i] = math.sqrt((La[i] * 110.852)**2 + (Lo[i] * 110.852 * math.cos(math.radians(La[i])))**2)
    ErrZ[i] = depth_synth - Z[i]
    
    # Print horizontal error
    print(f". Location differences in horizontal coordinates: {ErrH[i]:.2f} km")
    
    # Calculate Longitude and Latitude error relative to the synthetic earthquake location
    dlo = lon_synth - Lo[i]
    dist = 110.852 * math.cos(math.radians(lat_synth)) * dlo
    Err_Lon.append(dist)
    dlat = (lat_synth - La[i]) * 110.852
    Err_Lat.append(dlat)



    # Report depth error and simulation time
    time_simu = round(time.time() - deb_simu, 2)
    print(f". Depth: {Z[i]:.2f} km, Depth Variation: {ErrZ[i]:.2f} km")
    print(f". End of simulation {i+1}/{n_samples} : {time_simu} seconds\n")
    
    
# Convert lists to arrays for consistency
Err_Lon = np.array(Err_Lon)
Err_Lat = np.array(Err_Lat)
# Calculate total phase duration
duration_phase = round((time.time() - start_phase) / 60, 2)
print(f"> End of all {n_samples} simulations: {duration_phase} minutes")

# Create DataFrames for inputs and results
samples_df = pd.DataFrame(samples, columns=problem['names'])
outputs_df = pd.DataFrame({'La': La, 'Lo': Lo, 'Z': Z, 'RMS': RMS, 'DT': DT})
loc_differences_df = pd.DataFrame({'ErrZ': ErrZ, 'ErrH': ErrH, 'Err_Lon': Err_Lon, 'Err_Lat': Err_Lat})
# Combine all data into a single DataFrame
all_data = pd.concat([samples_df, outputs_df, loc_differences_df], axis=1) 



##############################################################################
#----------------------- COMPUTES AND PLOT SOBOL INDEXES---------------------#
##############################################################################

# Sobol sensitivity analysis for each output of the problem
original_stdout = sys.stdout


for label_output, values_output in outputs_df.items():
    out_file = os.path.join(path_out, f"Sobol_Indexes_{label_output}.txt")
    with open(out_file, "w") as sobol_fileout:
        output = StringIO()
        sys.stdout = output
        try:
            print(f"## Sobol Analysis for {label_output} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            SI = sobol.analyze(problem, values_output.to_numpy(), print_to_console=True)
            sobol_fileout.write(output.getvalue())
        finally:
            sys.stdout = original_stdout
    print(f"> Sobol analysis completed for {label_output}.")

    # Extract Main(S1)/Total(ST) Sobol indexes for analysis and visualization
    SI_filtered = {k: SI[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
    SI_df = pd.DataFrame(SI_filtered, index=problem['names'])
    # Sort by total effect (ST)
    SI_sorted = SI_df.sort_values('ST', ascending=False) 
     

    # Separate Indexes and Confidence Intervals (Epsilon)
    Sobol_Indexes = SI_df[['S1', 'ST']]
    Sobol_Confidence = SI_df[['S1_conf', 'ST_conf']]
        
        
    ### [Plotting]
    ### For each random variable Plot Total and Main Effect Contributions as a Bar Chart
    # (Laporte et al.,2024 -> as Fig.5)
    fig_SI, ax_SI = plt.subplots(figsize=(10, 7))
    ax_SI.bar(SI_sorted.index, SI_sorted.ST, yerr=SI_sorted.ST_conf, color='salmon', label='ST : Total Effect')
    ax_SI.bar(SI_sorted.index, SI_sorted.S1, yerr=SI_sorted.S1_conf, color='steelblue', label='S1 : Main Effect')
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    ax_SI.grid()
    ax_SI.set_ylim(0, 0.9)
    plt.title(f"{n_samples} simulations: {round((time.time() - t0) / 60, 2)} min\nOutput: {label_output}")
    plt.ylabel('Relative contribution', fontsize=15)
    plt.xlabel('Inputs', fontsize=16)
    plt.legend(loc='upper right', fontsize=15)
    fig_SI.savefig(os.path.join(path_out, f"F5_Bars_Sobol_Indexes_{label_output}.pdf"))
    #fig_SI.savefig(os.path.join(path_out, f"F5_Bars_Sobol_Indexes_{label_output}.jpg"))

    # Using SALiB plotting tool
    # fig_EE, ax_EE = plt.subplots(figsize=(10, 12))
    # Sobol_Indexes.plot.bar(yerr=Sobol_Confidence.values.T, ax=ax_EE)
    # fig_EE.set_size_inches(8, 4)
    # plt.xticks(rotation=45, fontsize=14)
    # plt.grid()
    # plt.ylabel('Relative contribution')
    # plt.xlabel('Parameters', fontsize=20)
    # fig_EE.savefig(os.path.join(path_out, f"{name_simu}_Parameters_EE_{label_output}.pdf"))
    # fig_EE.savefig(os.path.join(path_out, f"{name_simu}_Parameters_EE_{label_output}.jpg"))

    # Radial Plot for Sensitivity Analysis with Threshold
    # (Laporte et al.,2024 -> Fig.6)
    sns.set_style('whitegrid')
    fig_RP = plot_sobol_indices(SI, problem, label_output, criterion='ST', threshold=0.005)
    fig_RP.set_size_inches(7, 7)
    fig_RP.savefig(os.path.join(path_out, f"F6_Radial_Plot_Interactions_{label_output}.pdf"))
    #fig_RP.savefig(os.path.join(path_out, f"F6_Radial_Plot_Interactions_{label_output}.jpg"))
    plt.show()
        
    


    # choice between a scatter plot (EE_subplots) or an image (EE_subplots_imshow)
    # if n_samples is too large, the scatter plot takes too long 
    #EE_subplots(samples, problem['names'], [label_output, values_output], path_out)
    EE_subplots_imshow(samples, problem['names'], [label_output, values_output], path_out)    
    

 ##############################################################################
 #----------------------- ADDITIONAL PLOTS------------------------------------#
 ##############################################################################   
####  1 - Generate EE subplots for each parameter (Laporte et al.,2024 -> Fig.7)
####  Option : Filter to keep simulations with ErrZ<(10)km
#### -------------------------------------------------------------------------------------- 
# # Apply filtering on samples and outputs based on depth error threshold
filtered_data = filter_simulations(all_data, filter_Z=10)
plot_data = filtered_data # or all_data
# Visualizations using custom EE_subplots and EE_hist functions
plot_parameters =  pd.DataFrame({'Depth error (km)': plot_data['ErrZ'],
    'Latitude error (km)': plot_data['Err_Lat'],
    'Longitude error (km)': plot_data['Err_Lon'],
    'Horizontal error (km)': plot_data['ErrH'],
})
for label_output, values_output in plot_parameters.items() :
    # choice between a scatter plot (EE_subplots) or an image (EE_subplots_imshow)
    # if n_samples is too large, the scatter plot takes too long 
    #EE_subplots(plot_data[samples_df.columns], problem['names'], [label, data_series], path_out, name_simu)
    EE_subplots_imshow(plot_data[samples_df.columns].to_numpy(), problem['names'], [label_output, values_output], path_out)

###  2 - Representation of the 5 × 5 symmetric matrix of output inter-dependancies  (Laporte et al.,2024 -> Fig.4)
#### Using the Pairgrid plot of seaborn librairy 
#### --------------------------------------------------------------------------------------  
#  Option : Shows information vp2_info, dt_info, or rms_info based on conditions 
vp2_info = [0 if vp2 < -0.1 else 1 if vp2 > 0.1 else 0.5 for vp2 in samples_df['Vp2']]
dt_info = [1 if dti > 0.4 or dti < -0.4 else 0 for dti in DT]
rms_info = [1 if rmsi > 0.1 else 0 for rmsi in RMS]
# Select considered outputs into a dataframe
plot_seaborn_outputs = pd.DataFrame({
    'Longitude error (km)': Err_Lon,
    'Latitude error (km)': Err_Lat,
    'Depth error (km)': ErrZ,
    'RMS (s)': RMS,
    'DT (s)': DT,
    'DT_0.4': dt_info
})

# Begin seaborn figure
mpl.rcParams["axes.labelsize"] = 24
g=sns.PairGrid(plot_seaborn_outputs,vars=['Longitude error (km)','Latitude error (km)','Depth error (km)','DT (s)','RMS (s)'], hue='DT_0.4')
g.map_upper(sns.scatterplot,alpha=0.3)
axes_g = g.axes
# to restrict axes
# for i in range (len(axes_g)):
#     axes_g[i,0].set_xlim(-10,10)
#     axes_g[i,1].set_xlim(-10,10)
#     axes_g[0,i].set_ylim(-3,3)
#     axes_g[1,i].set_ylim(-3,3)
#     axes_g[i,2].set_xlim(-20,20)
#     axes_g[2,i].set_ylim(-20,20)
#     axes_g[i,3].set_xlim(-2,2)
#     axes_g[3,i].set_ylim(-2,2)

# Shows ellipse at 1$\sigma
dx1_XY, mx1_XY, dy1_XY, my1_XY = confidence_ellipse(Err_Lat, Err_Lon, axes_g[0,1], n_std=1,edgecolor='firebrick')
dx1_ZX, mx1_ZX, dy1_ZX, my1_ZX = confidence_ellipse( ErrZ,Err_Lon, axes_g[0,2], n_std=1, edgecolor='firebrick')
dx1_ZY, mx1_ZY, dy1_ZY, my1_ZY = confidence_ellipse( ErrZ, Err_Lat,axes_g[1,2], n_std=1, edgecolor='firebrick')
    
# Shows ellipse at 2$\sigma
dx2_XY, mx2_XY, dy2_XY, my2_XY = confidence_ellipse(Err_Lat, Err_Lon, axes_g[0,1], n_std=2, edgecolor='black', linestyle='--')
dx2_ZX, mx2_ZX, dy2_ZX, my2_ZX = confidence_ellipse( ErrZ,Err_Lon, axes_g[0,2], n_std=2, edgecolor='black', linestyle='--')
dx2_ZY, mx2_ZY, dy2_ZY, my2_ZY = confidence_ellipse( ErrZ, Err_Lat,axes_g[1,2], n_std=2, edgecolor='black', linestyle='--')

# Shows ellipse at 3$\sigma
dx3_XY, mx3_XY, dy3_XY, my3_XY = confidence_ellipse(Err_Lat, Err_Lon, axes_g[0,1], n_std=3, edgecolor='black', linestyle=':')
dx3_ZX, mx3_ZX, dy3_ZX, my3_ZX = confidence_ellipse( ErrZ,Err_Lon, axes_g[0,2], n_std=3, edgecolor='black', linestyle=':')
dx3_ZY, mx3_ZY, dy3_ZY, my3_ZY = confidence_ellipse( ErrZ, Err_Lat,axes_g[1,2], n_std=3, edgecolor='black', linestyle=':')


misloc2_XY = round(sqrt((float(dx2_XY))**2+(float(dy2_XY))**2),2)
misloc2_ZX = round(sqrt((float(dx2_ZX))**2+(float(dy2_ZX))**2),2)
misloc2_ZY = round(sqrt((float(dx2_ZY))**2+(float(dy2_ZY))**2),2)

g.map_lower(sns.kdeplot,fill=True, hue=None,cmap='Spectral_r')
g.map_diag(sns.histplot, kde=True, stat='proportion')  
g.add_legend(fontsize= 20)
g.savefig(path_out+'/F4_Dependancies_outputs_matrix_5x5.jpg')



#### 3- Compare Errors (difference from initial loc) and Uncertainties (computed from reloc)
#### --------------------------------------------------------------------------------------  
Diff_errH = np.zeros([samples.shape[0]])
Diff_errZ = np.zeros([samples.shape[0]])
for i in range(len(ErrH)):
    Diff_errH[i] = float(delta_h[i])-float(ErrH[i])
    Diff_errZ[i] = float(delta_z[i])-float(ErrZ[i])
fig_ERR,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
mean_errH = round( np.mean(Diff_errH),1)
ax1.hist(Diff_errH,bins=np.linspace(-3,5,40),color='steelblue', label='Mean of differences : '+str(mean_errH)+' km')
ax1.set_xlabel('Difference ErrH_real - ErrH_calc(Nll)',fontsize=14)
mean_errZ = round(np.mean(Diff_errZ),1)
ax2.hist(Diff_errZ,bins=np.linspace(-20,20,40),color='salmon',label='Mean of differences : '+str(mean_errZ)+' km')
ax2.set_xlabel('Difference ErrZ_real - ErrZ_calc(Nll)',fontsize=14)
plt.suptitle('Location erros analysis',fontsize=20)
ax1.legend()
ax2.legend()
fig_ERR.savefig(path_out+'/Error_vs_Uncertainty_analysis.pdf')





##############################################################################
#------------------MORRIS ANALYSIS-------------------------------------------#
### Uncomment to perform a Morris sensitivity analysis
### Parameters of the Morris sensitivity analysis
sigma = 0.95 # confidence level
levels = 200
deb_morris = time.time()
## Exemple for depth :
EE_morris = morris.analyze(problem, samples, Z, conf_level=sigma, 
                            print_to_console=True, num_levels=levels)
print("> Morris analyse : %s seconds"%(str(round(time.time()-deb_morris,2))))

      
## Figure Morris
fig_M, ax_M = plt.subplots(figsize=(10,10))
ax_M.scatter(EE_morris['mu_star'],EE_morris['sigma'],marker='*', s =300, color='firebrick')
ax_M.set_xlabel('Mean(|EE|)',fontsize=14)
ax_M.set_ylabel('Standard deviation($EE$)',fontsize=14)
ax_M.tick_params(axis='y', colors='black',labelsize=14,grid_markersize=14)
ax_M.tick_params(axis='x', colors='black',labelsize=14,grid_markersize=14)
for i, txt in enumerate(EE_morris['names']):
    ax_M.annotate(txt, (EE_morris['mu_star'][i], EE_morris['sigma'][i]), fontsize=14)
plt.title('Distribution of Elementary effects',fontsize=16)
name_fig_M = 'Morris_Elementary_Effects_Analysis.pdf'
fig_M.savefig(os.path.join(path_out,name_fig_M))

