#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:42:38 2022

@author: laportem
"""

##############################################################################
# A faire : rajouter rms et t0 pour faire morris
# Scripts 
from locate_nll import Locate
from ellipse import graphe_ellipse_synth, plot_ellipses

# Libraries
import SALib
from SALib.sample import saltelli
from SALib.test_functions import Ishigami
from SALib.analyze import sobol
from SALib.analyze import morris
import SALib.sample.morris as ms



import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)



import numpy as np
import pandas as pd
import time 
import os
import sys
import math
from datetime import datetime
from time import strftime



t0=time.time()
now = datetime.now()
original_stdout = sys.stdout
trajectoires = 10

num_vars=14
# La methode de Morris effectue une analyse de la sensibilite un parametre a la fois (OAT)
# appelee trajectoire et d effectuer  cette evaluation dans plusieurs points de l espace des parametres

# Combine rapidite d une methode AOT + explore un maillage sur une echelle globale
##############################################################################
#-------------------------- Define random variables--------------------------#
print(">> Start Morris Analysis ")

# La methode de Morris n autorise que l evaluation de loi uniforme sur une plage de variation de 0 a 1
# problem = {
#     'num_vars' : 12,
#     'names' : ['gap1',  'nsta', 'nclose', 'dmin', 
#                'vp1','vp2', 'vp3', 'vpvs',
#                'errP', 'errS', 'wS'],
#     'bounds' : [[0,1], [0,1], [0,1], [0,1], [0,1],
#                 [-1,1],[-1,1], [-1,1], [-1,1],
#                 [0,0.05], [0,0.15], [1,2]],

#     'dists':['unif', 'unif','unif','unif','unif',
#              'unif', 'unif','unif','unif',
#              'norm', 'norm','unif']
#     }


problem = {
    'num_vars' : 14,
    'names' : ['G1',  'Dmin', 'Dmoy','Emoy',
               'Taille_C1','Vp1','Vp2', 'Vp3','Vp/Vs',
               'ErrP', 'ErrS', 'ErrP_c', 'ErrS_c', 'nS'],
    'bounds' : [[0,1], [0,1], [0,1], [0,1],
                [-5,5],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5], [-0.07,0.07],
                [0,0.05], [0,0.15], [0,0.05], [0,0.15], [0.5,1]],

    }

# La plage de variation des 12 parametres est discretisee en N niveaux
# Le croisement de 4 niveaux definit un ensemble de noeuds (4**12 = 16777216)
# La methode de Morris permet d echantillonner ces 17 millions de noeuds 
# L echantillonnage est realise a travers un ensemble de 40 trajectoires aleatoires
# Chaque trajectoire passe par 13 noeuds de maniere a ce que chaque facteur ne varie qu une seule fois par trajectoire
# = Exploration d un maillage sur une echelle globale


niveaux = 4
jump=4
sigma = 0.95

param_values = ms.morris.sample(problem,N=100, num_levels=niveaux,optimal_trajectories=10)
#param_values = ms.sample(problem,N=trajectoires, num_levels=niveaux,grid_jump=jump, optimal_trajectories=None)
n_simu = param_values.shape[0]
name_simu = 'MORRIS_NLL_Z_25km_N10_Trajectoires_'+str(trajectoires)+'_Niveaux_'+str(niveaux)+'_'+str(n_simu)
print('> Problem defined for %s simulations'%(str(n_simu)))



##############################################################################
#-------------------------- Define random variables--------------------------#
La = np.zeros([param_values.shape[0]])
Lo = np.zeros([param_values.shape[0]])
Z = np.zeros([param_values.shape[0]])
ErrZ = np.zeros([param_values.shape[0]])
RMS = np.zeros([param_values.shape[0]])
DT = np.zeros([param_values.shape[0]])
R0 = np.zeros([param_values.shape[0]])
ERRX = np.zeros([param_values.shape[0]])
errors_h = np.zeros([param_values.shape[0]])
errors_z = np.zeros([param_values.shape[0]])
out_fig = '~/Documents'
out_file = os.path.join(out_fig,name_simu+'.txt')
##############################################################################
#-------------------------- Simulation phase---------------------------------#
for i, X in enumerate(param_values):
    deb_simu = time.time()
    variables_simu = ' ,'.join(str(e) for e in X)
    print(". Simulation %s/%s : %s"%(str(i),str(n_simu),str(variables_simu)))
    La[i],Lo[i],Z[i], RMS[i], DT[i], errors_h[i], errors_z[i] = Locate(X)
    ERRX[i] = math.sqrt((float(La[i])*110.852)**2+(float(Lo[i])*110.852*math.cos(float(La[i])))**2)
    ErrZ[i] = float(Z[i])-25
    print('Error in geographic coordinates :', round(ERRX[i],2))
    depth_simu = float(Z[i])
    depth_variation = abs(25-depth_simu)
    time_simu = round(float(time.time()-deb_simu),2)
    print(". Depth : %s km - variation : %s km"%(str(depth_simu), str(depth_variation)))
    print(". End of simulation %s/%s : %s sec"%(str(i),str(n_simu),str(time_simu)))
    print('\n')
    print('\n')
duration_phase = round(float(time.time()-t0)/60,2)
print("> End of all %s simulations : %s minutes"%(str(n_simu),str(duration_phase)))




lat_synth = 0
lon_synth = 0
depth_synth = 25

##############################################################################
#------------------------------- Morris--------------------------------------#
#------------------------------- Morris--------------------------------------#
# Parameters Morris

out_parameters = [['ErrZ',ErrZ], ['ErrX',ERRX],['RMS',RMS],['DT',DT]]
deb_morris = time.time()
# L influence de chaque facteur Xi est evaluee en comparant des simulations d'une trajectoire 
# entre lesquelles seul le facteur Xi a varié --> donne l effet elementaire de Xi pour un point de l espace
# L analyse de chaque trajectoire permet d'extraire les influences de chaque facteur sur la profondeur 
# 1 -- par la moyenne de la valeur absolue des effets X = Mean
# 2 -- par la variabilité des effets par l ecart type Y = Sigma

# EE : epsilon_i = (y_j*x_0)-(y_j*(x_0+delta)
#plot_ellipses(lon_synth, lat_synth, depth_synth, Lo, La, Z, n_simu)



Si_morris_all = []
with open(out_file, "w") as morris_fileout : 
    sys.stdout = morris_fileout      
    print('## README These Laporte 2022 - Morris analysis - '+now.strftime("%Y-%m-%d  %H:%M:%S"))
    print('## Simulation : '+name_simu)
    print('\n')
    for k in range (len(out_parameters)) :
        parameter = out_parameters[k]
        
        print ('# Morris indices for paramer '+parameter[0])
        Si_morris = morris.analyze(problem, param_values, parameter[1], conf_level=sigma, 
                                print_to_console=True, num_levels=niveaux)
        print('\n')

        Si_morris_all.append(Si_morris)
sys.stdout = original_stdout        
# Une fois places en X/Y : on obtient une estimation du niveau d interaction 
# et/ou de non linearite du parametre dans l espace      
# Figure 
fig_m, axs = plt.subplots( 2,2, figsize=(15,15))
axs = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]
for i in range (len(out_parameters)) :
    parameter = out_parameters[i]    
    ax_m = axs[i]   
    Si_morris_par = Si_morris_all[i]
    ax_m.errorbar(Si_morris_par['mu_star'],Si_morris_par['sigma'], xerr =Si_morris_par['mu_star_conf'], fmt='o' , lw=2, c='darkgrey', zorder=0)
    ax_m.scatter(Si_morris_par['mu_star'],Si_morris_par['sigma'],marker='o',color='salmon',s =200, zorder=1)
    #ax_m.plot(np.array([0,Si_morris['mu_star'][0]]),2*np.array([0,Si_morris['sigma'][0]/np.sqrt(300)]),'--',alpha=0.5)
    

    
    if parameter[0]=='ErrZ':
        units = 'km'
        a = 0.2
        ax_m.set_xlim(0,30)
        ax_m.set_ylim(0,30)
    elif parameter[0]=='ErrX' or parameter[0]=='Lo':
        units='km'
        a=0.1
        ax_m.set_xlim(0,8)
        ax_m.set_ylim(0,8)
    elif parameter[0]=='DT' :
        a = 0.02
        units ='seconds'
        ax_m.set_xlim(0,3)
        ax_m.set_ylim(0,3)
    elif parameter[0]=='RMS' :
        a = 0.01
        units ='seconds'
        ax_m.set_xlim(0,0.6)
        ax_m.set_ylim(0,0.6)
        
        
        
    for i, txt in enumerate(Si_morris_par['names']):
        ax_m.annotate(txt, (Si_morris_par['mu_star'][i]+a, Si_morris_par['sigma'][i]+a), size=18)
    
    ax_m.set_title('Morris method' +'\n'+'Distribution of Elementary effects : '+parameter[0],fontsize=16)
    ax_m.set_xlabel('Mean(|$EE$|) {%s}'%(str(units)),fontsize=16)
    ax_m.set_ylabel('Standard deviation($EE$) {%s}'%(str(units)),fontsize=16)    
    ax_m.grid()
    fig_m.tight_layout()
    fig_m.suptitle('Morris method '+'\n'+'Trajectoires : %s'%(str(trajectoires))+'\n'+'Niveaux : %s'%(str(niveaux))+'\n'+'Temps de simulation : %s min'%(str(duration_phase)),y=1.07, fontsize=20, fontweight='bold')  
        
    name_fig_m = name_simu +'.pdf'
    fig_m.savefig(os.path.join(out_fig,name_fig_m))
    


