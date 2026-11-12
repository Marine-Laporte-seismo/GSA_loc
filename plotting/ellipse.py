#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 11:22:14 2022

@author: laportem
"""
import numpy as np
import matplotlib.pyplot as plt 
from plotting.ellipse_func import calc_distance, graphe_ellipse, graphe_ellipse2, gaussian_points, confidence_ellipse
from math import cos, sin, tan, pi, sqrt
import os
from datetime import datetime

def graphe_ellipse_synth(dep_ref,deps, lon_ref,lons, lat_ref):
    x=[]
    y=[]
    flag=0
    for i in range (len(deps)):
        deg=lon_ref- lons[i]
        dist=110.852*cos(lat_ref)*float(deg)*1000

        x.append(dist)
        ddep=(dep_ref-deps[i])*1000
        y.append(ddep)              
    x=np.array(x)
    y=np.array(y)
    return x, y

def graphe_ellipse_synth2(lat_ref,lats, lon_ref,lons):
    x=[]
    y=[]
    flag=0
    for i in range (len(lats)):
        deg=lon_ref- lons[i]
        dist=110.852*cos(lat_ref)*float(deg)*1000
                
        x.append(dist)
        dlat=(lat_ref-lats[i])*110.852*1000

        y.append(dlat)
             
    x=np.array(x)
    y=np.array(y)
    return x, y

def graphe_ellipse_synth3(dep_ref,deps, rms):
    x=[]
    y=[]
    flag=0
    for i in range (len(deps)):
        rms_i = rms[i]
        x.append(rms_i)
        ddep=(dep_ref-deps[i])*1000
        y.append(ddep)              
    x=np.array(x)
    y=np.array(y)
    return x, y


def plot_ellipses(lon_synth, lat_synth, depth_synth, Lo, La, Z,RMS, n_iter, out_dir, name_simu):
    # ELLIPSES
    now = datetime.now()
    x1, y1= graphe_ellipse_synth(depth_synth,Z, lon_synth,Lo, lat_synth)
    x2, y2= graphe_ellipse_synth2(lat_synth,La, lon_synth, Lo)
    #x3, y3 = graphe_ellipse_synth3(depth_synth,Z, RMS)

## PLOT 
    limo=20000
    limo2 = 20000
    plt.figure(figsize=(8, 8))
    #plt.title('Adding gaussian error distribution S=0.1s P / S=0.3s S')
    left, width = 0.1, 0.65
    bottom, height = 0.1,0.65
    spacing = 0.03
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom,0.2, height]
    axs = plt.axes(rect_scatter)
    axs.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    
    

    labels=['Lons-Depth','Lats-Lons']
    
    xp,yp,zp=gaussian_points(x1,y1) 
    axs.scatter(xp,yp,c=zp,cmap='Spectral_r', s=25, edgecolor='grey', label=labels[0], alpha=0.7)
    #axs[0].scatter(xp,yp,c='grey', s=25, edgecolor='gray', label=labels[0], alpha=0.7)
    dx1, mx1, dy1, my1 = confidence_ellipse(x1, y1, axs, n_std=1,
                    label=r'$1\sigma$', edgecolor='firebrick')
    dx2, mx2, dy2, my2 = confidence_ellipse(x1, y1, axs, n_std=2,
                    label=r'$2\sigma$', edgecolor='black', linestyle='--')
    dx3, mx3, dy3, my3 = confidence_ellipse(x1, y1, axs, n_std=3,
                    label=r'$3\sigma$', edgecolor='black', linestyle=':')
    axs.set_xlabel('Longitude[m]',fontsize=18)
    axs.set_ylabel('Depth[m]',fontsize=18)
    axs.set_xlim(-limo2,limo2)
    axs.set_ylim(-limo,limo)
    axs.legend(fancybox=True, framealpha=0.5)
    axs.grid()
    aire1 = round(float(dx1)/1000 * float(dy1)/1000 * pi,1)
    aire2 = round(float(dx2)/1000 * float(dy2)/1000 * pi,1)
    aire3 = round(float(dx3)/1000 * float(dy3)/1000 * pi,1)
    misloc1 = round(sqrt((float(mx1)/1000)**2+(float(my1)/1000)**2),2)
    misloc2 = round(sqrt((float(mx2)/1000)**2+(float(my2)/1000)**2),2)
    misloc3 = round(sqrt((float(mx3)/1000)**2+(float(my3)/1000)**2),2)
    ## Histograms
    binwidth1 = 1000
    binwidth2 =1000
    # lim = np.ceil(np.abs([xp, yp]).max() / binwidth) * binwidth

    bins = np.arange(-limo2, limo2, binwidth2)
    ax_histx.hist(xp, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax_histx.set_ylabel('Nb events',fontsize=18)
    ax_histx.set_xlim(-limo2,limo2)
    ax_histx.grid()
    
    bins2 = np.arange(-limo, limo, binwidth1)
    ax_histy.hist(yp, bins=bins2, orientation='horizontal', color='steelblue',edgecolor='black', alpha=0.7)
    ax_histy.set_xlabel('Nb events',fontsize=18)
    ax_histy.set_ylim(-limo,limo)
    #ax_histy.set_xticks([0,250,500])
    ax_histy.grid()
    
    #plt.title('Error Ellipses while adding a pick error +/-0.05s P et +/-0.15s S ')

    plt.savefig(os.path.join(out_dir,'Saltelli_XZ_'+name_simu+'.pdf'))
    #plt.show()
    
    
    
    
    
    ##  DEUXIEME PLOT
    limo3 = 1000
    binwidth3 =50
    plt.figure(figsize=(8, 8))
    #plt.title('Adding gaussian error distribution S=0.1s P / S=0.3s S')

    axs2 = plt.axes(rect_scatter)
    axs2.tick_params(direction='in', top=True, right=True)
    ax_histx2 = plt.axes(rect_histx)
    ax_histx2.tick_params(direction='in', labelbottom=False)
    ax_histy2 = plt.axes(rect_histy)
    ax_histy2.tick_params(direction='in', labelleft=False)
    
    

    labels=['Lons-Depth','Lats-Lons']
    xd,yd,zd=gaussian_points(x2,y2)
    axs2.scatter(xd,yd,c=zd,cmap='Spectral_r', s=25, edgecolor='grey', label=labels[1], alpha=0.7)
    #axs[0].scatter(xp,yp,c='grey', s=25, 179 evedgecolor='gray', label=labels[0], alpha=0.7)
    dx4, mx4, dy4, my4 = confidence_ellipse(x2, y2, axs2, n_std=1,
                    label=r'$1\sigma$', edgecolor='firebrick')
    dx5, mx5, dy5, my5 = confidence_ellipse(x2, y2, axs2, n_std=2,
                    label=r'$2\sigma$', edgecolor='black', linestyle='--')
    dx6, mx6, dy6, my6 = confidence_ellipse(x2, y2, axs2, n_std=3,
                    label=r'$3\sigma$', edgecolor='black', linestyle=':')
    axs2.set_xlabel('Longitude[m]',fontsize=18)
    axs2.set_ylabel('Latitude[m]',fontsize=18)
    axs2.set_xlim(-limo3,limo3)
    axs2.set_ylim(-limo3,limo3)
    axs2.legend(fancybox=True, framealpha=0.5)
    axs2.grid()
    
    aire4 = round(float(dx4)/1000 * float(dy4)/1000 * pi,1)
    aire5 = round(float(dx5)/1000 * float(dy5)/1000 * pi,1)
    aire6 = round(float(dx6)/1000 * float(dy6)/1000 * pi,1)
    misloc4 = round(sqrt((float(mx4)/1000)**2+(float(my4)/1000)**2),2)
    misloc5 = round(sqrt((float(mx5)/1000)**2+(float(my5)/1000)**2),2)
    misloc6 = round(sqrt((float(mx6)/1000)**2+(float(my6)/1000)**2),2)
    ## Histograms

    # lim = np.ceil(np.abs([xp, yp]).max() / binwidth) * binwidth

    bins3 = np.arange(-limo3, limo3, binwidth3)
    ax_histx2.hist(xd, bins=bins3, color='steelblue', edgecolor='black', alpha=0.7)
    ax_histx2.set_ylabel('Nb events',fontsize=18)
    ax_histx2.set_xlim(-limo3,limo3)
    ax_histx2.grid()
    
    bins4 = np.arange(-limo3, limo3, binwidth3)
    ax_histy2.hist(yd, bins=bins4, orientation='horizontal', color='steelblue',edgecolor='black', alpha=0.7)
    ax_histy2.set_xlabel('Nb events',fontsize=18)
    #ax_histy2.set_xticks([0,250,500,750])
    ax_histy2.grid()
    
    #plt.title('Error Ellipses while adding a pick error +/-0.05s P et +/-0.15s S ')

    plt.savefig(os.path.join(out_dir, 'Saltelli_ALL_XY_'+name_simu+'.pdf'))
    plt.show()    
    
    
    out_file = os.path.join(out_dir,name_simu+'_Ellipses.txt')
    with open(out_file, "w") as ellipses_fileout : 
        ellipses_fileout.write('## README These Laporte 2022 - Sobol analysis - '+now.strftime("%Y-%m-%d  %H:%M:%S")+'\n')
        ellipses_fileout.write('## Simulation : '+name_simu+'\n')
        ellipses_fileout.write('# Uncertainty ellipses'+'\n')
        ellipses_fileout.write('\n')
        ellipses_fileout.write('# Covariance Longitude/Depth'+'\n')
        ellipses_fileout.write('\t'.join(['1:68%',str(mx1), str(my1), str(dx1), str(dy1), str(aire1), str(misloc1),'\n' ]))
        ellipses_fileout.write('\t'.join(['2:95%',str(mx2), str(my2), str(dx2), str(dy2), str(aire2), str(misloc2),'\n'  ]))
        ellipses_fileout.write('\t'.join(['3:99.7%',str(mx3), str(my3), str(dx3), str(dy3), str(aire3), str(misloc3),'\n'  ]))
        ellipses_fileout.write('# Covariance Longitude/Latitude')
        ellipses_fileout.write('\t'.join(['1:68%',str(mx4), str(my4), str(dx4), str(dy4), str(aire4), str(misloc4),'\n'  ] ))
        ellipses_fileout.write('\t'.join(['2:95%',str(mx5), str(my5), str(dx5), str(dy5), str(aire5), str(misloc5),'\n'  ]))
        ellipses_fileout.write('\t'.join(['3:99.7%',str(mx6), str(my6), str(dx6), str(dy6), str(aire6), str(misloc6),'\n'  ]))
    
    # ##  TROISIEME PLOT ##
    # plt.figure(figsize=(8, 8))
    # #plt.title('Adding gaussian error distribution S=0.1s P / S=0.3s S')

    # axs3 = plt.axes(rect_scatter)
    # axs3.tick_params(direction='in', top=True, right=True)
    # ax_histx3 = plt.axes(rect_histx)
    # ax_histx3.tick_params(direction='in', labelbottom=False)
    # ax_histy3 = plt.axes(rect_histy)
    # ax_histy3.tick_params(direction='in', labelleft=False)
    
    

    # labels=['Lons-Depth','Lats-Lons', 'RMS-Depth']
    # xr,yr,zr=gaussian_points(x3,y3)
    # axs3.scatter(xr,yr,c=zr,cmap='Spectral_r', s=25, edgecolor='', label=labels[2], alpha=0.7)
    # #axs[0].scatter(xp,yp,c='grey', s=25, 179 evedgecolor='gray', label=labels[0], alpha=0.7)
    # confidence_ellipse(x3, y3, axs3, n_std=1,
    #                 label=r'$1\sigma$', edgecolor='firebrick')
    # confidence_ellipse(x3, y3, axs3, n_std=2,
    #                 label=r'$2\sigma$', edgecolor='black', linestyle='--')
    # confidence_ellipse(x3, y3, axs3, n_std=3,
    #                 label=r'$3\sigma$', edgecolor='black', linestyle=':')
    # axs3.set_xlabel('Longitude[m]')
    # axs3.set_ylabel('Latitude[m]')
    # axs3.set_xlim(0,2)
    # axs3.set_ylim(-limo2,limo2)
    # axs3.legend(fancybox=True, framealpha=0.5)
    # axs3.grid()
    
    
    
    # ## Histograms

    # # lim = np.ceil(np.abs([xp, yp]).max() / binwidth) * binwidth

    # bins5 = np.arange(0, 2, 0.1)
    # ax_histx3.hist(xr, bins=bins5, color='steelblue', edgecolor='black', alpha=0.7)
    # ax_histx3.set_ylabel('Nb events')
    # ax_histx3.set_xlim(0,2)
    # ax_histx3.grid()
    

    # ax_histy3.hist(yr, bins=bins4, orientation='horizontal', color='steelblue',edgecolor='black', alpha=0.7)
    # ax_histy3.set_xlabel('Nb events')
    # #ax_histy2.set_xticks([0,250,500,750])
    # ax_histy3.grid()
    
    #plt.title('Error Ellipses while adding a pick error +/-0.05s P et +/-0.15s S ')

    # plt.savefig('/media/laportem/THESE_HD/Saltelli_ALL_Rms'+str(n_iter)+'.pdf')
    # plt.show()
    
    
def plot_ellipses_new(ErrZ, ErrH, n_iter, out_dir, name_simu):
    # ELLIPSES
    now = datetime.now()
    y = np.array(ErrZ)
    x = np.array(ErrH)


## PLOT 
    limx=25
    limy = 25
    plt.figure(figsize=(15, 15))
    #plt.title('Adding gaussian error distribution S=0.1s P / S=0.3s S')
    left, width = 0.1, 0.65
    bottom, height = 0.1,0.65
    spacing = 0.025
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom,0.2, height]
    axs = plt.axes(rect_scatter)
    axs.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    
    

    labels=['ErrH - ErrZ']
    
    xp,yp,zp=gaussian_points(x,y) 
    axs.scatter(xp,yp,c=zp,cmap='Spectral_r', s=25, edgecolor='', label=labels[0], alpha=0.7)
    #axs[0].scatter(xp,yp,c='grey', s=25, edgecolor='gray', label=labels[0], alpha=0.7)
    dx1, mx1, dy1, my1 = confidence_ellipse(x, y, axs, n_std=1,
                    label=r'$1\sigma$', edgecolor='firebrick')
    dx2, mx2, dy2, my2 = confidence_ellipse(x, y, axs, n_std=2,
                    label=r'$2\sigma$', edgecolor='black', linestyle='--')
    dx3, mx3, dy3, my3 = confidence_ellipse(x, y, axs, n_std=3,
                    label=r'$3\sigma$', edgecolor='black', linestyle=':')
    axs.set_xlabel('Erreur horizontale [km]')
    axs.set_ylabel('Erreur verticale [km]')
    axs.set_xlim(-limx,limx)
    axs.set_ylim(-limy,limy)
    
    
    
    axs.legend(fancybox=True, framealpha=0.5)
    axs.grid()
    aire1 = round(float(dx1) * float(dy1) * pi,1)
    aire2 = round(float(dx2) * float(dy2) * pi,1)
    aire3 = round(float(dx3)* float(dy3) * pi,1)
    misloc1 = round(sqrt((float(mx1)**2)+(float(my1)**2)),2)
    misloc2 = round(sqrt((float(mx2)**2)+(float(my2)**2)),2)
    misloc3 = round(sqrt((float(mx3)**2)+(float(my3)**2)),2)
    ## Histograms
    binwidth1 = 1
    binwidth2 =1
    # lim = np.ceil(np.abs([xp, yp]).max() / binwidth) * binwidth

    bins = np.arange(-limx, limx, binwidth2)
    ax_histx.hist(xp, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax_histx.set_ylabel('Nb events')
    ax_histx.set_xlim(-limx,limx)
    ax_histx.grid()
    
    bins2 = np.arange(-limy, limy, binwidth1)
    ax_histy.hist(yp, bins=bins2, orientation='horizontal', color='steelblue',edgecolor='black', alpha=0.7)
    ax_histy.set_xlabel('Nb events')
    ax_histy.set_ylim(-limy,limy)
    #ax_histy.set_xticks([0,250,500])
    ax_histy.grid()
    
    #plt.title('Error Ellipses while adding a pick error +/-0.05s P et +/-0.15s S ')

    plt.savefig(os.path.join(out_dir,'Saltelli_ALL_HZ_'+str(n_iter)+'.pdf'))
    #plt.show()
    
    
    
    
    
    
    out_file = os.path.join(out_dir,name_simu+'_Ellipses_ErrH_ErrZ.txt')
    with open(out_file, "w") as ellipses_fileout : 
        ellipses_fileout.write('## README These Laporte 2022 - Sobol analysis - '+now.strftime("%Y-%m-%d  %H:%M:%S")+'\n')
        ellipses_fileout.write('## Simulation : '+name_simu+'\n')
        ellipses_fileout.write('# Uncertainty ellipses'+'\n')
        ellipses_fileout.write('\n')
        ellipses_fileout.write('# Covariance ErrH / ErrZ'+'\n')
        ellipses_fileout.write('\t'.join(['1:68%',str(mx1), str(my1), str(dx1), str(dy1), str(aire1), str(misloc1),'\n' ]))
        ellipses_fileout.write('\t'.join(['2:95%',str(mx2), str(my2), str(dx2), str(dy2), str(aire2), str(misloc2),'\n'  ]))
        ellipses_fileout.write('\t'.join(['3:99.7%',str(mx3), str(my3), str(dx3), str(dy3), str(aire3), str(misloc3),'\n'  ]))
