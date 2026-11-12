#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:24:04 2020

@author: laportem
"""
# Libraries
import sys
from obspy.core import UTCDateTime
from math import pi, acos, cos, asin, sin, tan, atan, sqrt
import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import gaussian_kde

# Functions


def calc_distance(lat1, lat2, lon1, lon2):
    # Pour chaque évènement commun : calcul distances et azimuts entre les épicentres RSC-HIKNET
    d2r = pi/180
    Dlon=abs(float(lon2)-float(lon1))
            
    # Distance entre épicentres RSC et HiKNET
    Delta=acos(sin(float(lat2)*d2r)*sin(float(lat1)*d2r)+cos(float(lat2)*d2r)*cos(float(lat1)*d2r)*cos(Dlon*d2r)) # en radians
    distance=Delta*6370 # en kilomètres
    
    return distance

def graphe_ellipse(date1, date2, depth1,depth2, lon1,lon2, lat1):
    to =time.time()
    x=[]
    y=[]
    flag=0
    for i in range (len(date1)):
        date_i=date1[i]
        if i<len(date2)-20:
            for j in range(i-20,i+20):
    
                date_j=date2[j]
                diff=abs(date_i-date_j)
                if diff<1 :
                    flag+=1
                    deg=lon1[i]-lon2[j]
                    dist=110.852*cos(lat1[i])*float(deg)*1000
                    x.append(dist)
                    ddep=(depth1[i]-depth2[j])*1000
                    y.append(ddep)
                    
        else:
            for j in range(len(date2)):
    
                date_j=date2[j]
                diff=abs(date_i-date_j)
                if diff<1 :
                    flag+=1
                    deg=lon1[i]-lon2[j]
                    dist=110.852*cos(lat1[i])*float(deg)*1000
                    x.append(dist)
                    ddep=(depth1[i]-depth2[j])*1000
                    y.append(ddep)
                
    x=np.array(x)
    y=np.array(y)
    duree=time.time()-to
    print( 'Graphe ellipse en %s secondes'%(str(duree)))
    return x, y

def graphe_ellipse_synth(date_ref, dates, depth_ref,deps, lon_ref,lons, lat_ref):
    to =time.time()
    x=[]
    y=[]
    flag=0
    for i in range (len(dates)):
        date_i=dates[i]
        deg=lon_ref- lons[i]
        dist=110.852*cos(lat_ref)*float(deg)*1000
        x.append(dist)
        ddep=(dep_ref-deps[i])*1000
        y.append(ddep)              
    x=np.array(x)
    y=np.array(y)
    duree=time.time()-to
    print( 'Graphe ellipse en %s secondes'%(str(duree)))
    return x, y

def graphe_ellipse2(date1, date2, lat1,lat2, lon1,lon2):
    top=time.time()
    x=[]
    y=[]
    for i in range (0,len(date1)-1):
        date_i=date1[i]
        if i<len(date2)-20:
            for j in range(i-20,i+20):

                date_j=date2[j]
                diff=abs(date_i-date_j)
                if diff<1 :
                    deg=lon1[i]-lon2[j]
                    dist=110.852*cos(lat1[i])*float(deg)*1000
                    x.append(dist)
                    dlat=(lat1[i]-lat2[j])*110.852*1000
                    y.append(dlat)
        else:
            for j in range(len(date2)):

                date_j=date2[j]
                diff=abs(date_i-date_j)
                if diff<1 :
                    deg=lon1[i]-lon2[j]
                    dist=110.852*cos(lat1[i])*float(deg)*1000
                    x.append(dist)
                    dlat=(lat1[i]-lat2[j])*110.852*1000
                    y.append(dlat)
            
                
    x=np.array(x)
    y=np.array(y)
    duree=time.time()-top
    print( 'Graphe ellipse en %s secondes'%(str(duree)))
    return x, y



def gaussian_points(xo,yo):
    x=np.array(xo)
    y=np.array(yo)
    xy = np.vstack([xo,yo])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    return x,y,z

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    

    
    #print( 'Demi grand axe x : ', scale_x)
    mean_x = np.mean(x)
    #print( 'Moyenne x : ', mean_x)
    maxx=scale_x+mean_x
    minx=-scale_x+mean_x
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    #print( 'Demi grand axe y :', scale_y)
    mean_y = np.mean(y)
    #print( 'Moyenne y : ', mean_y)
    #print( '\t')
    maxy=scale_y+mean_y
    miny=-scale_y+mean_y

 
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    
    
    sx = round(scale_x,2)
    sy = round(scale_y,2)
    mx = round(mean_x,2)
    my=round(mean_y,2)
    return sx, mx, sy, my