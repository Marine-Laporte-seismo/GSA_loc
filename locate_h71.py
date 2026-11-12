#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:46:16 2021
@author: laportem
Locate a synthetic event (xo,yo,zo) using Hypo71 earthquake algorithm (Lee and Lahr., 1975)
using random variables for 10 receivers locations, velocity model errors, picking errors, 
defined by sampling strategy for the global sensitivity analysis
"""

##############################################################################
# =============================================================================
# Scripts 

# =============================================================================
# from network_map import show_network
# Libraries
import numpy as np
import math
import sys
import os
import string
import time
import glob
from obspy import  UTCDateTime
import logging

#################################
# MAIN FUNCTION
####################################


def Locate(args, depth_synth):
    """
   Executes a series of steps to compute an earthquake location 
   while adding errors in the velocity model, picking observations 
   and changing the network geometry according to random variables given in args.
   
   Parameters:
   -----------
   args : list
       A list of input parameters affecting various elements of the seismic simulation:
           - args[0] to args[3]: Parameters for defining network geometry and station distribution:
               - index_gap1 : int
                   Index controlling gap between stations.
               - index_dist_min : int
                   Index controlling minimum station distance.
               - index_dist_moy : int
                   Index controlling average station distance.
               - index_dist_chaos : int
                   Index controlling randomness in station distribution.
           - args[4] to args[8]: Parameters affecting the velocity model configuration:
               - index_H1 : int
                   Index modifying the depth of the initial layer.
               - index_Vp1 : int
                   Index for primary velocity model layer.
               - index_Vp2 : int
                   Index for secondary velocity model layer.
               - index_Vp3 : int
                   Index for tertiary velocity model layer.
               - index_ratio : int
                   Ratio index for adjusting Vp/Vs velocity model.
           - args[9] to args[13]: Parameters defining observation errors:
               - error_P : float
                   Error level for P-wave arrivals.
               - error_S : float
                   Error level for S-wave arrivals.
               - error_P_close : float
                   Error for P-waves at close distances.
               - error_S_close : float
                   Error for S-waves at close distances.
               - index_prob_S : float
                   Probability factor for S-wave observations.
            - depth_synth : initial depth 
   
   Returns:
   --------
   tuple
       A tuple containing:
       - lat_end : float
           Final estimated latitude of the event.
       - lon_end : float
           Final estimated longitude of the event.
       - depth : float
           Final estimated depth of the event.
       - rms_end : float
           Root Mean Square error of the final solution.
       - time_end : UTCDateTime
           The computed event origin time.
       - errh : float
           Horizontal error of the event location.
       - errz : float
           Vertical error of the event location.
   """
    phasefile = './hypo71/input_phase.pha'
    name_simu = './hypo71/Simu'

    # Initial Parameters
    lat_synth = 0
    lon_synth = 0
    depth_synth = depth_synth
    origin_time = UTCDateTime(2000, 1, 1, 1, 1, 30)
    model_init = np.array([[0, 20, 55], [5.4, 6.4, 8.1]])
    ratio_init = 1.73
    nsta = 10
    dist_close = 50
    dist_far = 150
    dist_chaos = 20
    elev_min = 0
    elev_max = 0
    value_error_max_velocity = 5
    weightP = 0
    weightS = 2 # for hypo71 input phase file (0 full weight, 2 half weight)

    # Parse input arguments
    index_gap1, index_dist_min, index_dist_moy, index_dist_chaos = args[:4]
    index_H1, index_Vp1, index_Vp2, index_Vp3, index_ratio = args[4:9]
    error_P, error_S, error_P_close, error_S_close, index_prob_S = args[9:14]
    error_P_rand, error_S_rand = 0, 0  # defaults

    # Predefined randomness for chaos, P, S, and observation errors
    scheme_chaos_dist = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1] 
    scheme_P = [0.90, -0.95, -1.00, 1.1, 0.85, -1.15, -0.85, 1.00, 1.2, -0.80, -1.00, 1.05, 1.10, -0.8, -0.85, 1.10, -1.20, -0.80, 0.8, 1.00]
    scheme_S = [-0.80, 1.20, 1.00, -1.10, -0.90, 1.15, -0.95, 0.85, -0.90, 1.05, -1.00, 0.80, -1.15, -0.95, -1.05, 1.10, -0.80, -0.85, 1.20, 0.95]
    scheme_obs_S = [0.75, 0.5, 1, 0.8, 0.65, 0.55, 0.7, 0.9, 0.85, 0.6, 0.725, 0.975, 0.575, 0.825, 0.675, 0.625, 0.925, 0.875, 0.525, 0.775]

    # 1 - Create Network from geometrical parameters
    stations_lats, stations_lons, stations_elevs, stations_names, ids = Create_Network_chaos(
        index_gap1, index_dist_min, index_dist_moy, index_dist_chaos, nsta, scheme_chaos_dist, 
        dist_close, dist_far, dist_chaos, elev_min, elev_max, lat_synth, lon_synth
    )

    # 2 - Write NonLinLoc control file for computing theoretical arrivals 
    # (with the initial velocity model )
    Write_NLL_control_tt(
        stations_names, stations_lats, stations_lons, stations_elevs, model_init, ratio_init, synth_lat=lat_synth, synth_lon=lon_synth, synth_depth=depth_synth
    )
    
    # 3 - Compute travel-times
    liste_ttP, liste_ttS = get_travel_times(nsta)

    # 4 - Generate random velocity model
    model_aranged, ratio_aranged = Create_Random_Model(
        model_init, ratio_init, index_H1, index_Vp1, index_Vp2, index_Vp3, index_ratio, value_error_max_velocity
    )

    # 5 - Write Phase file for input of NonLinLoc/Hypo71
    Write_Phase_file(
        stations_names, liste_ttP, liste_ttS, origin_time, error_P, error_S, weightP, weightS, 
        scheme_P, scheme_S, error_P_close, error_S_close, error_P_rand, error_S_rand, scheme_obs_S, 
        index_prob_S, phasefile
    )

    # 6 -  Write Hypo71 control/parameters file for relocation
    
    Write_Hypo71_Hypodata(stations_names, stations_lats, stations_lons, stations_elevs, model_aranged, ratio_aranged, dist_close, dist_far, 
                       tests=['0.1', '30.', '1.', '0.05', '2.5', '1', '-0.87', '2.0000', '0.0035', '100.', '18.', '0.5', '1.'])

    # 7 - Launch NonLinLoc simulation and extract results
    test = Launch_Hypo71(phasefile, name_simu, args)
    #lat_end, lon_end, depth, rms_end, time_end, errh, errz 
    return test









# =============================================================================
# # 0 - Tools
# =============================================================================

def open_catalog (catalog_path):
    """Try to open a text file and store its lines in a list
    output : lines (list of length n=number of lines in catalog)"""
    try:
        cat = open(catalog_path, "r")
    except:
        print( "Error : %s does not exist !" % catalog_path)
        sys.exit()
    else:
        lines = cat.readlines()
        cat.close()
    return lines


def get_point_from_bearing_distance(lat1,lon1, az, dist):
    """" Compute destination in geographical coordinates (degrees)
    from origin (lat1-lon1), bearing (az) in degrees 
    and distance (dist) in kilometers """
    # given: lat1, lon1, b = bearing in degrees, d = distance in kilometers
    R = 6378.1 #Radius of the Earth   
    az = math.radians(az)
    lat1 = math.radians(lat1) #Current lat point converted to radians
    lon1 = math.radians(lon1) #Current long point converted to radians
    
    lat2 = math.asin( math.sin(lat1)*math.cos(dist/R) +
         math.cos(lat1)*math.sin(dist/R)*math.cos(az))  
    lon2 = lon1 + math.atan2(math.sin(az)*math.sin(dist/R)*math.cos(lat1),
                 math.cos(dist/R)-math.sin(lat1)*math.sin(lat2))
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    return lat2, lon2


# =============================================================================
# 1 - Create Network from geometrical parameters
# =============================================================================
def get_name(bearing, distance):
    """Code a nametag of 4 characters for every station depending on bearing and distance
   Additional : verify two stations do not have the same nametag !!! """
    flag=False
    alphabet = list(string.ascii_uppercase)
    deb_staname = str(int(bearing/10)).zfill(2)
    end_staname = str(int(distance/10)).zfill(2)
    if str(deb_staname[0]) == '0':
        deb_staname = 'W'+str(deb_staname[1])
    elif str(deb_staname[0]) == '1':
        deb_staname = 'X'+str(deb_staname[1])
    elif str(deb_staname[0]) == '2':
        deb_staname = 'Y'+str(deb_staname[1])
    elif str(deb_staname[0]) == '3':
        deb_staname = 'Z'+str(deb_staname[1])
    if str(end_staname[0]) == '0':
        for i in range(0,10):
            if flag==False and abs(int(distance) -int(end_staname)*10)  == i :
                end_staname = alphabet[i] + str(end_staname)[1]
                flag = True
    elif str(end_staname[0]) == '1':
        for j in range(0,10):
            if flag==False and abs(int(distance) - int(end_staname)*10) == j :
                end_staname = alphabet[j] + str(end_staname)[1]  
                flag = True
            
    sta_code = deb_staname + end_staname
    return sta_code


def verify_names(names):
    """ For Create_Network function, just check if the station name is available 
    if not : returns a new station name """
    for k in range(3) :
        alphabet = list(string.ascii_uppercase)
        replace = alphabet[-1-k]
        once = set()
        seenOnce = once.add
        twice = set(n for n in names if n in once or seenOnce(n))
        for j in range (len(list(twice))):
            flag_correct = False 
            doublon = list(twice)[j]
            for i in range(len(names)):
                namesta = names[i]
                if doublon == namesta  and flag_correct == False :
                    names[i]=namesta[:-1]+replace
                    flag_correct = True
    return names
        
        


def Create_Network(index_gap1, index_dist_min, index_dist_moy, nsta=10, dist_close = 50, dist_far = 150, elev_min=0.0, elev_max=0.0, lat_synth=0,lon_synth=0):
    """ Computes geometrical parameters of a network based on several indexes used for earthquakes locations qualification (Bondar et al., 2004)
    # Random variable of azimuths  :
     - index_gap1 :float: primary azimutal gap, purcentage of 360°, value between 0 and 1  (RV gap1)
    # Random variable station distance :
     - index_dist_min :float: value of the minimum source station distance, purcentage of 1-dist_close value between 0 and 1 (RV dmin)
     - index_dist_moy :float: value of the mean source station distance, purcentage of 1-dist_close value between 0 and 1 (RV dmoy)
    
    # Parameters
     - nsta :float: number of stations, default = 10
     - dist_close :float: area of the closest source-station distance (km), default = 50
     - dist_far   :float: largest source-station distance (km), default = 150
         
    # Stations elevations (default, to vary for further analysis)
    - elev_min :float: minimal station elevation in meters default=0.0
    - elev_max :float: maximal station elevation in meters default=0.0
    
    # Source epicentral parameters for reference (default) : 
    - lat_synth :float: latitude en degres, default=0.0
    - lon_synth :float: longitude en degres, default=0.0
    
    # OUTPUTS :
    - liste_lats :list of floats: length = nsta : latitudes of stations in decimal degrees
    - liste_lons :list of floats: length = nsta : longitude of stations in decimal degrees
    - liste_elev :list of floats: length = nsta : elevation of stations in meters
    - stations_names :list of strings: length = nsta : nametag of station in 4 characters
    - stations_ids :list of integers: length = nsta : just the num of station from 1 to 10
    
    """
    print(">> Generate new seismological network...")
    #to = time.time()
    # Pick gap primaire 
    gap_azi_minimal_1 = float(360/nsta)
    gap_azi_maximal_1 = 230
    gap_azi_1 = gap_azi_minimal_1 + index_gap1 * (gap_azi_maximal_1 - gap_azi_minimal_1)    
    #print( '.Primary gap : ', gap_azi_1)

    # 2 - Get number of stations to fulfill between azimutal gaps
    #print(".Number of stations :", nsta)
    # 3 - Get minimal distance (km)
    dist_min =  2 + index_dist_min  * (dist_close-2)
    dist_away = 100
    dist_moy = dist_close + index_dist_moy * (dist_away-dist_close)
    
    # __init__ listes outputs 
    stations_lats = [0]*int(nsta)
    stations_lons = [0]*int(nsta)
    stations_elev = [0]*int(nsta)
    names = ['XXXX']*int(nsta)
    stations_ids = np.arange(1,nsta+1,1)
    
    
    azi_sta = gap_azi_1
    for sta in range (1,int(nsta)+1):
        if sta == 1 :
            dm = dist_moy
            bearing = 0
        elif sta == 2 :
            dm = dist_moy
            bearing = gap_azi_1
        elif sta == 3 :
            dm = dist_min          
            azi_sta = azi_sta + (360-gap_azi_1)/float(nsta-1)
            bearing = azi_sta 
        else :
            dm = dist_moy           
            azi_sta = azi_sta + (360-gap_azi_1)/float(nsta-1)
            bearing = azi_sta 

        lat_sta, lon_sta = get_point_from_bearing_distance(lat_synth,lon_synth, bearing, dm)
        name_sta = get_name(bearing,dm)
        elev_sta =  np.random.uniform(elev_min,elev_max) 
        # store in lists
        stations_lats[sta-1] = round(lat_sta,3)
        stations_lons[sta-1] = round(lon_sta,3)
        stations_elev[sta-1] = elev_sta
        names[sta-1] = name_sta
    stations_names = verify_names(stations_names)
    #print ('Times', time.time()-to)
    return stations_lats, stations_lons,stations_elev, stations_names, stations_ids




def Create_Network_chaos(index_gap1, index_dist_min, index_dist_moy, index_dist_chaos,nsta, scheme_chaos_dist,dist_close=50 , dist_far=150, max_dist_chaos=20,  elev_min=0, elev_max=0, lat_synth=0,lon_synth=0):
    """ Computes geometrical parameters of a network based on several indexes used for earthquakes locations qualification (Bondar et al., 2004)
    # Random variable of azimuths  :
     - index_gap1 :float: rd primary azimutal gap, drawn between gap_azi_minimal and gap_azi_maximal, value between 0 and 1  (RV gap1)
    # Random variable station distance :
     - index_dist_min :float: rd value of the minimum source station distance, drawn between 5km and dist_close, value between 0 and 1 (RV dmin)
     - index_dist_moy :float: rd value of the mean source station distance, drawn between dist_close and dist_far, value between 0 and 1 (RV dmoy)
    # To generate a chaos using a predefined scheme
     - index_dist_chaos :float: rd value of +/-dmoy, between 0 and max_dist_chaos, value between 0 and 1 
     - scheme_chaos_dist :list (10): a predefined scheme defining which stations are pull closer (-1) and pull back (+1) 
     - max_dist_chaos :float: largest deviation of dmoy for chaos default =20 km
    # Parameters
     - nsta :float: number of stations, default = 10
     - dist_close :float: area of the closest source-station distance (km), default = 50
     - dist_far   :float: largest source-station distance (km), default = 150
         
    # Stations elevations (default, to vary for further analysis)
    - elev_min :float: minimal station elevation in meters default=0.0
    - elev_max :float: maximal station elevation in meters default=0.0
    
    # Source epicentral parameters for reference (default) : 
    - lat_synth :float: latitude en degres, default=0.0
    - lon_synth :float: longitude en degres, default=0.0
    
    # OUTPUTS :
    - stations_lats :list of floats: length = nsta : latitudes of stations in decimal degrees
    - stations_lons :list of floats: length = nsta : longitude of stations in decimal degrees
    - stations_elev :list of floats: length = nsta : elevation of stations in meters
    - stations_names :list of strings: length = nsta : nametag of station in 4 characters
    - stations_ids :list of integers: length = nsta : just the num of station from 1 to 10
    
    """
    print(">> Generate new seismological network...")
    epsilon = 5
    # Pick gap primaire 
    gap_azi_minimal_1 = float(360)/nsta
    gap_azi_maximal_1 = 110
    gap_azi_1 = gap_azi_minimal_1 + index_gap1 * (gap_azi_maximal_1 - gap_azi_minimal_1)    

    # 3 - Get minimal distance (km)
    dist_min_0 = epsilon
    dist_min_1 = 30
    dist_min =  dist_min_0 + index_dist_min  * (dist_min_1-dist_min_0)
    #print('DISTANCE MIN:  ', dist_min)
    
    dist_moy_0 = dist_min_1 + max_dist_chaos + 2
    dist_moy_1 = 150 - max_dist_chaos - epsilon
    dist_moy = dist_moy_0 + index_dist_moy * (dist_moy_1-dist_moy_0)
    #print('DISTANCE MOY :  ', dist_moy)
    
    
    # __init__ listes outputs 
    stations_lats = [0]*int(nsta)
    stations_lons = [0]*int(nsta)
    stations_elev = [0]*int(nsta)
    stations_names = ['XXXX']*int(nsta)
    stations_ids = np.arange(1,nsta+1,1)
    
    chaos_dist = index_dist_chaos * max_dist_chaos
    
    
    azi_sta = gap_azi_1
    for sta in range (1,int(nsta)+1):
        if sta == 1 :
            dm = dist_moy + (chaos_dist*scheme_chaos_dist[sta-1])
            bearing = 0
        elif sta == 2 :
            dm = dist_moy + (chaos_dist*scheme_chaos_dist[sta-1])
            bearing = gap_azi_1
        elif sta == 3 :
            dm = dist_min          
            azi_sta = azi_sta + (360-gap_azi_1)/float(nsta-1)
            bearing = azi_sta 
        else :
            dm =dist_moy + (chaos_dist*scheme_chaos_dist[sta-1]) 
            azi_sta = azi_sta + (360-gap_azi_1)/float(nsta-1)
            bearing = azi_sta 

        lat_sta, lon_sta = get_point_from_bearing_distance(lat_synth,lon_synth, bearing, dm)
        name_sta = get_name(bearing,dm)
        elev_sta =  np.random.uniform(elev_min,elev_max) 
        # store in lists
        stations_lats[sta-1] = round(lat_sta,3)
        stations_lons[sta-1] = round(lon_sta,3)
        stations_elev[sta-1] = elev_sta
        stations_names[sta-1] = name_sta 
    stations_names = verify_names(stations_names)
    return stations_lats, stations_lons,stations_elev, stations_names, stations_ids

# =============================================================================
#  2 - Write NonLinLoc control file for computing theoretical arrivals  
# =============================================================================

def Write_NLL_control_tt( stations_names, stations_lats, stations_lons, stations_elevs, velocity_model, 
    vp_vs_ratio, synth_lat=0, synth_lon=0, synth_depth=25 ):
    """
    Creates the control file for NonLinLoc for computing theoretical travel-times in the 
    folder ./nlloc
    """
    print(">> Write control file for computing theoretical arrivals using NonLinLoc...") 
    control_lines = open_catalog('./nlloc/control_ref_tt.txt')
    with open('./nlloc/control_tt.txt', 'w') as control_new:
        n_lay = velocity_model.shape[1]
        n_sta = len(stations_names)

        # Write initial lines
        control_new.writelines(control_lines[:10])

        # VGGRID line
        line_vggrid = 'VGGRID 2 404 202 0 0 -5.0 0.5 0.5 0.5 SLOW_LEN\n'
        control_new.write(line_vggrid)
        control_new.write(control_lines[12])

        # Layer information
        for j_line in range(n_lay):
            layer_line_reference = control_lines[13].split()
            layer_depth = velocity_model[0, j_line]
            layer_velocity = velocity_model[1, j_line]
            layer_ratio = layer_velocity / vp_vs_ratio

            layer_line = f'LAYER {layer_depth:.2f} {layer_velocity:.2f} 0 {layer_ratio:.2f} 0 2.67 0\n'
            control_new.write(layer_line)

        # Write lines from control reference
        control_new.writelines(control_lines[16:23])

        # Station information
        for h_line in range(n_sta):
            station_name = stations_names[h_line]
            station_lat = stations_lats[h_line]
            station_lon = stations_lons[h_line]
            station_elev = stations_elevs[h_line] / 1000

            station_line = f'GTSRCE {station_name} LATLON {station_lat:.2f} {station_lon:.2f} 0.0 {station_elev:.2f}\n'
            control_new.write(station_line)

        # Additional control lines
        control_new.writelines(control_lines[31:38])

        # Earthquake event line
        line_eqevent = f'EQEVENT EQ_REF {synth_lat} {synth_lon} {synth_depth} 0.0\n'
        control_new.write(line_eqevent)

        # Phase information for each station
        for station in stations_names:
            station_line_P = f'EQSTA {station} P GAU 0 GAU 0\n'
            station_line_S = f'EQSTA {station} S GAU 0 GAU 0\n'
            control_new.write(station_line_P)
            control_new.write(station_line_S)
            control_new.write(f'EQVPVS {vp_vs_ratio:.2f}\n')

        # Final line from control reference
        control_new.write(control_lines[46])
        
# =============================================================================
#   3 - Compute travel-times
# =============================================================================        
def get_travel_times(n_sta):
    """
    Launches NonLinLoc to compute theoretical travel times at each station in the network
    based on the initial velocity model. This function cleans previous model and time files,
    calculates travel times, and returns them for P- and S-waves.

    Parameters:
    -----------
    n_sta :integer: number of stations in the network

    Returns:
    --------
    tuple
        A tuple containing:
        - liste_ttP : ndarray
            Array of computed P-wave travel times for each station.
        - liste_ttS : ndarray
            Array of computed S-wave travel times for each station.
    """
    print(">> Computing theoretical travel times...")
    



    # Clear previous model and time files
    model_files = glob.glob('./nlloc/theoretical/model/*')
    time_files = glob.glob('./nlloc/theoretical/time/*')
    for file_path in model_files + time_files:
        os.remove(file_path)

    # Run NonLinLoc programs for travel time calculation
    start_time = time.time()
    os.system("Vel2Grid ./nlloc/control_tt.txt")
    os.system("Grid2Time ./nlloc/control_tt.txt")
    os.system("Time2EQ ./nlloc/control_tt.txt")
    elapsed_time = time.time() - start_time
    #print(f"Travel time computation completed in {elapsed_time:.2f} seconds.")

    # Open generated synthetic observation catalog
    travel_times_catalog = open_catalog('./nlloc/synth.obs')

    # Initialize travel time arrays for P- and S-waves
    liste_ttP = np.zeros(n_sta)
    liste_ttS = np.zeros(n_sta)
    # Retrieve travel time arrays by parsing synth.obs
    for i in range(n_sta):
        try:
            # P-wave travel time (even lines in catalog)
            liste_ttP[i] = float(travel_times_catalog[2 + i * 2].split()[8])
            # S-wave travel time (odd lines in catalog)
            liste_ttS[i] = float(travel_times_catalog[3 + i * 2].split()[8])
        except (IndexError, ValueError):
            # Set default travel time if data is missing or corrupted
            liste_ttP[i] = 20.0
            liste_ttS[i] = 20.0

    return liste_ttP, liste_ttS
        
# =============================================================================
#   4 - Generate random velocity model
# =============================================================================


def Create_Random_Model(model_init, ratio_init, index_H1,  index_Vp1, index_Vp2, index_Vp3, index_ratio, value_error_max_vitesse = 10) : 
    """ Create a random velocity model close to a reference velocity model 
    # Reference model :
    - model_init : np.array (2*3): initial velocity model for P waves L1 : couches (km) / L2: Vp (m/s)
    - ratio_init : float : VP/VS ratio initial, default = 1.73 for sandstones (1.68,1.78)
    
    # Add random error between -value/+value% (4 VARIABLES):
    - index_H1 : norm : value between -5 and 5 km (bounds)
    - index_Vp1 : norm: value between -0.05 and 0.05 km/s
    determining purcentage of error for the velocity in the first layer
    - index_Vp2 : norm: value between -0.05 and 0.05 km/s
    determining purcentage of error for the velocity in the second layer
    - index_Vp0 : uniform: value between 0 and 0.1 km/s
    determining purcentage of error for the velocity in the third layer
    - index_ratio : norm: value between -0.05 and 0.05 km/s
    determining purcentage of error for the Vp/Vs ratio

    
    # Output :
    - model_aranged : array (2 * nlayer):
    - ratio_aranged : float: value of Vp/Vs ratio
    """
    print(">> Generate new velocity model...")
    new_model = model_init
    n_layer = len(model_init[0])
    
    # Modification vitesses P
    #error_Vp = [float(index_Vp1) * float(value_error_max_vitesse)/100, float(index_Vp2) * float(value_error_max_vitesse)/100, float(index_Vp3) * float(value_error_max_vitesse)/100]
    error_Vp  = [float(index_Vp1), float(index_Vp2), float(index_Vp3)]
    for layer in range(0,n_layer):

        old_Vp = float(model_init[1,layer]) 
        new_Vp  = old_Vp + error_Vp[layer] 
        #print('New velocity :', new_Vp)
        new_model[1,layer] = new_Vp    

    # Modification Layer size
    new_model[0,1] = new_model[0,1] + index_H1
    
    # Modification ratio Vp/Vs
    new_ratio = ratio_init +  index_ratio
    #print('Nouveau ratio :', new_ratio)
    #print('RATIO : ', new_ratio)
    return new_model, new_ratio

       

# =============================================================================
#  5 - Write Phase file for input of NonLinLoc/Hypo71
# =============================================================================
def Write_Phase_file(stations_names,liste_ttP,liste_ttS, origin_time, err_pick_P, 
                      err_pick_S, weightP,weightS,tirage_P, tirage_S, errP_close, errS_close,errP_rand, errS_rand,
                      tirage_obs_S, index_prob_S, phase_file):
    """
    Creates the input phase file required for NonLinLoc/Hypo71 by computing 
    travel times with added noise and writing formatted data for each station.
    """
    n_sta = len(stations_names)
    # Erase old phase_file
    try :
        os.remove(phase_file)
    except OSError:
        print('-> Create a new .pha')
        
    # Write a new phase_file for the simulation
    phase_new = open(phase_file, "w+") 

    for i in range(n_sta):
        tp = origin_time.strftime("%y%m%d%H%M")
        proba_S = float(tirage_obs_S[i])*index_prob_S
        if i==2 : 
            err_P = errP_close
            err_S = errS_close
        elif i==8:
            err_P = errP_rand
            err_S = errS_rand
        else :
            err_P = tirage_P[i]*err_pick_P
            err_S = tirage_S[i]*err_pick_S
        
        if proba_S < 0.5 :
            weightS = 4
        else :
            weightS = 2
        #err_P = round(float(err_pick_P), 3)
        secP = origin_time.second+liste_ttP[i] + err_P# ADD 0
        #err_S = round(float(err_pick_S), 3)
        secS = origin_time.second+liste_ttS[i]+err_S # ADD 0
        linepha =str(stations_names[i])+'.P.'+str(weightP)+' '+tp+str('{:02.2f}'.format(secP)).zfill(5)+ '       '+str('{:02.2f}'.format(secS)).zfill(5)+'.S.'+str(weightS)+'                                   '+'\n'
        phase_new.writelines((linepha))
    phase_new.writelines('                 10')
    phase_new.writelines('\n')
    phase_new.close()    
   

# =============================================================================
#   6 -  Write NonLinLoc control file for computing theoretical arrivals 
# =============================================================================
           
def Write_Hypo71_Hypodata(liste_name, liste_lats, liste_lons, liste_elev, model_aranged, ratio_aranged, dist_close, dist_far, tests=['0.1','30.','1.',
        """
        Generates a 'hypodata' file for the Hypo71 location input.
        
        Parameters:
            station_names (list of str): Names of stations.
            latitudes (list of float): Latitudes of stations.
            longitudes (list of float): Longitudes of stations.
            elevations (list of int): Elevations of stations in meters.
            model_arranged (2D array): Model data arranged as [[depth], [velocity]] for each layer.
            ratio_arranged (float): Ratio for depth layer settings.
            dist_close (int): Minimum distance parameter.
            dist_far (int): Maximum distance parameter.
            tests (list of str): Hypo71 test parameters.
        """                                                                                                                          '0.05','2.5','1','-0.87','2.0000','0.0035','100.','18.', '0.5','1.']):
    space = ' '
    lines_tests=[0]*13      
    nsta = len(liste_name)
    lines_stations= [0]*nsta  
                                                                                                                 
    with open('./hypo71/hypodata','w') as hypo :

        hypo.writelines('HEAD'+21*space+'SobolAnalysis'+'\n')
        for i in range(1,14):
            lines_tests[i-1] = 'RESET TEST(%02d)=%s'%(i,str(tests[i-1]))
        hypo.writelines('\n'.join(lines_tests))  
        hypo.writelines('\n'+ space*27+'SELECTION CARD 1 OU BLANC EN COLONNE1'+'\n')

        for j in range(nsta):
            tag_sta = liste_name[j]
            ent_lats = int(liste_lats[j])
            ent_lons = int(liste_lons[j])
            val_elev = int(liste_elev[j])
            dec_lats = round((abs(liste_lats[j])%1)*60,2)
            dec_lons = round((abs(liste_lons[j])%1)*60,2)
    
            if liste_lats[j]>=0:
                hem_lat = 'N'
            else :
                hem_lat='S'
            if liste_lons[j]>=0 :
                hem_lon='E'
            else :
                hem_lon='W'
            lines_stations[j] = '  %s%02d%05.2f%s%03d%05.2f%s%04d'%(tag_sta, abs(ent_lats), dec_lats, hem_lat, abs(ent_lons), dec_lons, hem_lon, val_elev)
        
        hypo.writelines('\n'.join(lines_stations)) 
        hypo.writelines('\n'+'\n')
        nlayers = len(model_aranged[0,:]) 
        lines_layers = [0]*nlayers
        for k in range(nlayers):
            lines_layers[k] = '  %01.1f   %05.2f'%(model_aranged[1,k], model_aranged[0,k])
        hypo.writelines('\n'.join(lines_layers)) 
        hypo.writelines('\n'+'\n')
        final_line = '   5. %02d. %03d.  %01.2f    3    0         0    0         0   1 '%(dist_close, dist_far, ratio_aranged)
        hypo.writelines(final_line) 
        hypo.writelines('\n')
        

# =============================================================================
#  7 - Launch NonLinLoc simulation and extract results
# =============================================================================


def Launch_Hypo71(phase_file, name_simu, args):
    """Launch Hypo71 location and read output files."""
    
    # Define output log file
    logout = f"{name_simu}.out"
    log = open(logout, "a+")
    
    # Define relocation output files
    reloc_1 = f"{phase_file}.hyp"
    reloc_2 = f"{phase_file}.prt"
    
    # Erase old relocation files if they exist
    try:
        os.remove(reloc_1)
        os.remove(reloc_2)
    except OSError:
        print('-> Create a new reloc.hyp file') 
        logging.basicConfig(filename='hypo71_log.txt', level=logging.INFO)
        logging.info(">> Launch Hypo71 and retrieve final loc..") 
    
    # Launch Hypo71 in terminal
    os.system(f"./hypo71/c_hypo {phase_file}")  
    
    # Read new relocation files: depth only
    try:
        with open(reloc_1, "r+") as cat_1, open(reloc_2, "r+") as cat_2:
            line_hyp = cat_1.readlines()
            line_prt = cat_2.readlines()
    except Exception as e:
        print('Location error:', e)
        return None, None, None, None, None, None  # Early exit on error

    # Check for location errors in the output
    line_test_1 = line_prt[39]
    line_test_2 = line_prt[40]
    
    if line_test_1.startswith(' **') or line_test_2.startswith(' **'):
        res0 = 10
        lat_end = 0.2
        lon_end = 0.2
        rms_end = 10
        time_end = 10
        depth = 0
        delta_h_end = 10
        delta_z_end = 10
    else:
        try:
            res0 = float(line_prt[46].split()[11]) 
        except ValueError:
            res0 = str(line_prt[46].split()[11])[:3]  # Use substring if conversion fails
        
        # Extract depth, latitude, longitude, RMS, and time from the output
        try:
            depth = float(line_hyp[0][37:42])
            lat_end = float(line_hyp[0][18:20]) + float(line_hyp[0][21:26]) / 60
            lon_end = float(line_hyp[0][28:30]) + float(line_hyp[0][31:36]) / 60
            rms_end = float(line_hyp[0][44:47])
            delta_h_end = float(line_hyp[0][60:63])
            delta_z_end = float(line_hyp[0][64:68])
            time_end = 30 - float(line_hyp[0][12:17])
        except (ValueError, IndexError):
            # Default values in case of extraction failure or empty Hypo71 output
            depth = 2.0          # à choisir : valeur par défaut de la profondeur
            lat_end = 0.2
            lon_end = 0.2
            rms_end = 10
            time_end = 10
            delta_h_end = 11
            delta_z_end = 11

    
    # Log parameters and results
    log.write(f"{args[0]}\t{args[1]}\t{args[2]}\t{args[3]}\t{args[4]}\t")
    #log.write(str(line_hyp[0]))  # Write first line of reloc_log file
    log.write(str(line_hyp[0]) if line_hyp else "NO LOCATION (insufficient data)\n")
    log.close()

    return lat_end, lon_end, depth, rms_end, time_end, delta_h_end, delta_z_end


        

    
if __name__ == "__main__":
    # Here it is just a test 
   
    depth_synth=25
    # ARGS : index_gap1, index_gap2, index_nsta, index_nsta_close, index_dist_min
    index_gap1 = 0
    index_gap2 = 0.95
    index_nsta = 0.05
    index_nsta_close = 0.25
    index_dist_min = 0.9

    sta_max = 25
    dist_close = 50
    dist_far = 150
    elev_min=0
    elev_max = 1000
    
    index_Vp1 = -0.8
    index_Vp2 = 0.3
    index_Vp3 = 0
    index_ratio = 0.7
    

    error_P = 0.2
    error_S = 0
    weightP=0.1
    weightS=1
    
    args = [ 0.29919434,  0.24450684,  0.9954834 ,  0.10388184,  2.50366211,
       -0.4276123 ,  0.06896973,  0.30236816, -0.03892442,  0.07715578,
        0.17461453, -0.00252545, -0.00849562,  0.59014893]
    #args = [0.666666666667 ,0.666666666667 ,0.0 ,0.333333333333 ,0.333333333333 ,-1.0 ,-0.333333333333 ,1.0 ,-0.333333333333 ,0.0166666666667 ,0.05 ,1.0]
     #   args =[ 0.56005859,  0.94287109,  0.05322266,  0.25634766,  0.90576172, -0.80371094,
    # 0.35058594 , 0.04199219,  0.66699219 , 0.18036317 ,0.03348294  ,0.13623047]
    z = Locate(args,depth_synth)
    

