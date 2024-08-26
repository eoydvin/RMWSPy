#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:01:03 2023

@author: Erlend Øydvin
https://github.com/eoydvin

Simple algorithm that finds points and lines that are close to each other

This alorithm could be improved for instance by using shapes or projections, 
but I like it this way since it do not relies on too many libraries and it is 
simple touderstand. Due to its simplifications it might miss some points that 
were actually within distance to the link. However these occations are few and
only happen to points that fall outside the circles drawn inside the while loop


Projection could be implemented by using:
https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
"""

import numpy as np
import tqdm
def haversine(lat1, lon1, lat2, lon2):
    # return: distance [km] between two points, assuming earth is a sphere
    # https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    R = 6373.0 # approximate radius of earth in km
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return abs(distance)


def group_points_near_lines(points_ds, cmls, distance, max_linklength = None):
    """
    Function that, given points and CMLs, finds point stations that are  
    close to a CML. Short CMLs are removed. 
    
    Parameters
    ----------
    points_ds: Xarray Dataset [station_id]
        Contains information on raingauge/disdrometer. Must have 
        coordinates: [sublink_id, lat (latitude), lon (longitude)]. 
    cmls: Xarray Dataset [sublink_id]
        Contains information on CMLs. Must have coordinates [sublink_id
        site_b_latitude, site_b_longitude, site_a_latitude, 
        site_a_longitude, length]. 
    distance: float
        Maximum distance from CML to raingauge/disdrometer and max length cml. 

    Returns
    -------
    point_with_short_cml: list [point measurement with nearby CML]
    point_cmls: dictionary {point_name: [list of nearby CMLs]}     

    """
    bound_distance = distance # in km, gauges that are close to link
    if max_linklength is None:
        max_linklength = distance # in km, max length of CMLs to concider
    point_cmls = {} # tracks gauge id as key and all CML close to the gauge
    x_bound_distance_per_linklength = distance # 
    for i, point in tqdm.tqdm(enumerate(points_ds.station_id)):
        lat_point = points_ds.station_id[i].lat
        lon_point = points_ds.station_id[i].lon
        point_cmls[str(point.station_id.values)] = []
        for j, cml in enumerate(cmls.length.where(
                cmls.length < max_linklength, drop = True).cml_id):
            lat1 = cml.site_0_lat.values # lines
            lon1 = cml.site_0_lon.values
            lat2 = cml.site_1_lat.values
            lon2 = cml.site_1_lon.values
            lat3 = lat_point.values # points 
            lon3 = lon_point.values
            
            # draw a box around link, check if point is within this box
            if lat1 > lat2:
                within_lat = lat3 <= lat1 and lat3 >= lat2
            elif lat1 <= lat2:
                within_lat = lat3 >= lat1 and lat3 <= lat2    
            if lon1 > lon2:
                within_lon = lon3 <= lon1 and lon3 >= lon2
            elif lon1 <= lon2:
                within_lon = lon3 >= lon1 and lon3 <= lon2
            
            # Exception: Check if close to endpoints
            within_point1 = haversine(lat1, lon1, lat3, lon3) <= bound_distance
            within_point2 = haversine(lat2, lon2, lat3, lon3) <= bound_distance
            if within_point1 or within_point2:
                point_cmls[str(point.station_id.values)].append(
                    str(cml.cml_id.values))
                
            elif within_lat and within_lon:
                link_length = cml.length
                
                # make a line between the two points and discretize, note that
                # this is probably an approximation
                lats_link = np.linspace(lat1, lat2, 
                    int(x_bound_distance_per_linklength*link_length/bound_distance))
                lons_link = np.linspace(lon1, lon2, 
                    int(x_bound_distance_per_linklength*link_length/bound_distance))
                
                k = 0
                within_link = False
                while (k < len(lats_link)) and (not within_link):
                    within_link = haversine(
                        lats_link[k], lons_link[k], lat3, lon3) <= bound_distance
                    k += 1
                    if within_link:
                        point_cmls[str(point.station_id.values)].append(
                            str(cml.cml_id.values))
    
    point_with_short_cml = [i for i in point_cmls if len(point_cmls[i]) != 0]
    return point_with_short_cml, point_cmls


def check_points_near_lines(points_ds, cmls, distance):
    """
    Check if given point/points is close to a line. This can for instance be 
    pixels in a radar grid. The function return true if any of the points 
    given is within range of a cml, so for a radar grid it must be re-run
    for each pixel in order to test all pixels separately
    
    Parameters
    ----------
    points_ds : Xarray Dataset [sublink_id]
        Contains information on raingauge/disdrometer. Must have 
        coordinates station_id, lat (latitude) and lon (longitude). 
    cmls : Xarray Dataset [sublink_id]
        Contains information on CMLs. Must have coordinates sublink_id
        site_b_latitude, site_b_longitude, site_a_latitude, 
        site_a_longitude and length. 
    distance : float
        True if point is close to a line

    Returns
    -------
    None.

    """
    bound_distance = distance # in km, gauges that are close to link
    max_linklength = cmls.length.max() # in km, max length of CMLs to concider
    point_cmls = {} # tracks gauge id as key and all CML close to the gauge
    x_bound_distance_per_linklength = distance # discertization steps in approximation
    for i, point in enumerate(points_ds.station_id):
        lat_point = points_ds.station_id[i].lat
        lon_point = points_ds.station_id[i].lon
        point_cmls[str(point.station_id.values)] = []
        
        for j, cml in enumerate(cmls.length.where(
                cmls.length < max_linklength).dropna().cml_id):
                        
            lat1 = cml.site_a_latitude.values # lines
            lon1 = cml.site_a_longitude.values
            lat2 = cml.site_b_latitude.values
            lon2 = cml.site_b_longitude.values
            lat3 = lat_point.values # points 
            lon3 = lon_point.values
            
            # draw a box around link, check if point is within this box
            if lat1 > lat2:
                within_lat = lat3 <= lat1 and lat3 >= lat2
            elif lat1 <= lat2:
                within_lat = lat3 >= lat1 and lat3 <= lat2    
            if lon1 > lon2:
                within_lon = lon3 <= lon1 and lon3 >= lon2
            elif lon1 <= lon2:
                within_lon = lon3 >= lon1 and lon3 <= lon2
                
            # Exception: Check if close to endpoints
            within_point1 = haversine(lat1, lon1, lat3, lon3) <= bound_distance
            within_point2 = haversine(lat2, lon2, lat3, lon3) <= bound_distance
            if within_point1 or within_point2:
                return True
                
            elif within_lat and within_lon:
                link_length = cml.length
                
                # make a line between the two points and discretize, note that
                # this is probably an approximation
                lats_link = np.linspace(lat1, lat2, 
                    int(x_bound_distance_per_linklength*link_length/bound_distance))
                lons_link = np.linspace(lon1, lon2, 
                    int(x_bound_distance_per_linklength*link_length/bound_distance))
                
                k = 0
                within_link = False
                while (k < len(lats_link)) and (not within_link):
                    within_link = haversine(
                        lats_link[k], lons_link[k], lat3, lon3) <= bound_distance
                    k += 1
                    if within_link:
                        return True
    
    return False