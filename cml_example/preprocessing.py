import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import yaml

import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from tqdm import tqdm

import rain_data as rd


def process_rawData(setup, give=None):
    """
    Pre-processing of observational data. Includes time selection, resampling,
    and spatial sanity check.
    """

    with open("settings.yml") as file:
        dict_options = yaml.safe_load(file)
    globals().update(dict_options[setup])

    # array that stores considered time steps
    t_array = np.array(
        pd.date_range(
            "%s-%s-%sT%s:%s" % (year, months[0], start_day, start_hour, start_minute),
            periods=n_tsteps,
            freq=freq,
        )
    )

    # load data
    if len(months) > 1:
        cml_files = [cml_data_raw % (year, m) for m in months]
        ds_cml = xr.open_mfdataset(cml_files)
    else:
        cml_file = cml_data_raw % (year, months[0])
        ds_cml = xr.open_dataset(cml_file)
    ds_rg = xr.open_dataset(rg_data_raw)

    # pre-select by time
    ds_cml = ds_cml.sel(time=slice(t_array[0] - pd.Timedelta("59 min"), t_array[-1]))
    ds_rg = ds_rg.sel(time=slice(t_array[0] - pd.Timedelta("59 min"), t_array[-1]))

    # not as dask objects (spatial sanity check take long otherwise)
    ds_cml = ds_cml.compute()

    # remove duplicates that stem from loading with xr.open_mfdataset
    ds_cml = ds_cml.sel(time=~ds_cml.get_index("time").duplicated())

    # select one channel
    ds_cml = ds_cml.sel(channel_id="channel_1", drop=True)

    # remove superfluous variables
    ds_cml = ds_cml.drop_vars(
        ["rx", "tx", "txrx", "polarization", "frequency", "length"]
    )

    # rename variables for consistency
    ds_cml = ds_cml.rename({"cml_id": "obs_id", "R": "rain"})
    ds_rg = ds_rg.rename({"station_id": "obs_id", "rainfall_amount": "rain"})

    # select only time of interest
    ds_cml = ds_cml.sel(time=slice(t_array[0] - pd.Timedelta("59 min"), t_array[-1]))
    ds_rg = ds_rg.sel(time=slice(t_array[0] - pd.Timedelta("59 min"), t_array[-1]))

    print("Resampling ...")
    # resample
    ds_cml["rain"] = ds_cml.rain.resample(
        time="60min", closed="right", label="right", base=50, restore_coord_dims=True
    ).mean(dim="time", skipna=False)

    ds_rg["rain"] = ds_rg.rain.resample(
        time="60min", closed="right", label="right", base=50, restore_coord_dims=True
    ).sum(dim="time", skipna=False)
    print("... done.")

    # select hourly values
    ds_cml = ds_cml.sel(time=t_array)
    ds_rg = ds_rg.sel(time=t_array)

    # project
    ds_cml = rd.projection(ds_cml, "CML")
    ds_rg = rd.projection(ds_rg, "LIN")

    # cut area
    ds_cml = rd.take_subset_obs(ds_cml, box, "CML")
    ds_rg = rd.take_subset_obs(ds_rg, box, "LIN")

    # midpoint
    ds_cml = rd.center_along_link(ds_cml, grid="latlon")

    # project midpoint
    ds_cml = rd.projection(ds_cml, datatype="LIN")

    print("Spatial sanity check ...")
    # label - spatial sanity check
    ds_cml = rd.label_outliers(ds_cml)
    # ds_rg = rd.label_outliers(ds_rg)
    print("... done.")

    if give == "CML":
        return ds_cml
    else:
        ds_cml.to_netcdf(datafolder + "temp/CML_preprocessed.nc")
        ds_rg.to_netcdf(datafolder + "temp/rg_preprocessed.nc")


def process_prior_to_reconstruction(setup):
    """
    Second part of preprocessing.
    """

    with open("settings.yml") as file:
        dict_options = yaml.safe_load(file)
    globals().update(dict_options[setup])

    # array that stores considered time steps
    t_array = np.array(
        pd.date_range(
            "%s-%s-%sT%s:%s" % (year, months[0], start_day, start_hour, start_minute),
            periods=n_tsteps,
            freq=freq,
        )
    )

    # load preprocessed observations
    ds_cml = xr.open_dataset(datafolder + obsfolder + "CML_preprocessed.nc")
    ds_rg = xr.open_dataset(datafolder + obsfolder + "rg_preprocessed.nc")

    # label duplicates considering both datasets (first one is kept preferably)
    ds_rg, ds_cml = rd.label_combined_duplicates(ds_rg, ds_cml)

    # set rain to nan where duplicate check was not passed
    # ds_cml["rain"] = ds_cml.rain.where(ds_cml.label_dupl)
    ds_cml = ds_cml.where(ds_cml.label_dupl, drop=True)

    # shift towards origin
    ds_cml = rd.shift_indices(ds_cml, box, "CML")
    ds_cml = rd.shift_indices(ds_cml, box, "LIN")
    ds_rg = rd.shift_indices(ds_rg, box, "LIN")

    # set rain to nan where spatial sanity check was not passed
    ds_cml["rain"] = ds_cml.rain.where(ds_cml.label_sp_sanity)

    # label time steps whether they are wet enough
    ds_cml["label_wet"] = (
        xr.where(ds_cml.rain > wet_thld, True, False).sum(dim="obs_id")
        / len(ds_cml.obs_id)
        >= wet_min
    )
    ds_rg["label_wet"] = (
        xr.where(ds_rg.rain > wet_thld, True, False).sum(dim="obs_id")
        / len(ds_rg.obs_id)
        >= wet_min
    )

    # combine label wet for CMLs and gauges
    label_wet = ds_cml.label_wet & ds_rg.label_wet

    # label time steps by enough/not enough available data
    ds_cml["label_available"] = ds_cml.rain.isnull().sum(dim="obs_id") <= nan_max * len(
        ds_cml.obs_id
    )

    # combine to dataset with info on timesteps
    ds_timeseries = xr.Dataset({"time": ("time", t_array)})
    ds_timeseries["label_wet"] = ("time", label_wet)
    ds_timeseries["label_available"] = ("time", ds_cml.label_available)
    ds_timeseries["label_calculated"] = (
        "time",
        ds_timeseries.label_wet & ds_timeseries.label_available,
    )

    for label in [label_wet, ds_cml.label_available]:
        t_array = filter_timesteps(label, t_array)

    # save
    ds_timeseries.to_netcdf(datafolder + "temp/timeseries_labels.nc")
    ds_cml.to_netcdf(datafolder + "temp/CML_for_reconstruction.nc")
    ds_rg.to_netcdf(datafolder + "temp/rg_for_reconstruction.nc")


def filter_timesteps(label, t_array):
    """
    Filter indiviudal timesteps from the series based on a flag (label).
    """

    t_array_new = []
    for i, t in enumerate(t_array):
        if label.sel(time=t):
            t_array_new.append(t)
    t_array_new = np.array(t_array_new)

    L1 = len(t_array)
    L2 = len(t_array_new)

    print(
        "%i time steps filtered based on %s. %i time steps remain."
        % (L1 - L2, label.name, L2)
    )

    return t_array_new


def define_spatial_masks(box, fn_border, fn_cml, fn_rg, radii=[30], outfolder=None):
    """
    Define potential areas for analysis, based on observation distance and border.
    """

    import RM_processing as rmp
    from shapely.geometry import Point, Polygon

    # ===================
    # masks based on distances from observations

    # preprocessing
    domainsize = (box[2] - box[0], box[3] - box[1])

    # load preprocessed observations
    ds_cml = xr.open_dataset(fn_cml)
    ds_rg = xr.open_dataset(fn_rg)

    # reduce time dimension
    ds_cml_at_t = ds_cml.isel(time=0)
    ds_rg_at_t = ds_rg.isel(time=0)

    # get coords
    cml_yx = (
        np.vstack(
            (
                ds_cml_at_t.y_a.values,
                ds_cml_at_t.x_a.values,
                ds_cml_at_t.y_b.values,
                ds_cml_at_t.x_b.values,
            )
        ).T
    ).astype(int)
    rg_yx = (np.vstack((ds_rg_at_t.y.values, ds_rg_at_t.x.values)).T).astype(int)

    # create dummy rain array (required by function)
    cml_prec = np.ones(len(ds_cml_at_t.obs_id))
    rg_prec = np.ones(len(ds_rg_at_t.obs_id))

    # initialize dataset
    ds_mask = xr.Dataset(
        coords={
            "y": (("y"), np.arange(box[0], box[2])),
            "x": (("x"), np.arange(box[1], box[3])),
        }
    )

    # loop over radii and test which pixels are within radius of obs.
    # define union (pixels need to be within radius of any obs. one of the datasets)
    # and intersection (pixels need to be with radius of an obs. of both datasets)
    for radius in tqdm(radii):

        mask_cml_dist = rmp.genMask_CML(cml_yx, cml_prec, domainsize, radius)
        mask_rg_dist = rmp.genMask(rg_yx, rg_prec, domainsize, radius)

        mask_union_dist = mask_cml_dist | mask_rg_dist
        mask_inter_dist = mask_cml_dist & mask_rg_dist

        ds_mask["%i_union" % radius] = (("y", "x"), mask_union_dist)
        ds_mask["%i_inter" % radius] = (("y", "x"), mask_inter_dist)

    # ===================
    # German border mask

    # read polygon data of German border
    df_shape_ger = pd.read_csv(fn_border, index_col=0)

    # constants of RADOLAN projection
    xstart = -523.4621669218558
    ystart = -4658.644724265572

    # initialize mask
    mask_ger = np.zeros((900, 900))

    # loop over polygons that define Germany (islands separate polygons)
    for part, df_part in df_shape_ger.groupby("part"):
        border_y = df_part.y_dwd.values / 1000 - ystart
        border_x = df_part.x_dwd.values / 1000 - xstart

        polygon = Polygon(list(zip(border_x, border_y)))

        # loop over grid coordinates
        for y in tqdm(range(mask_ger.shape[0])):
            for x in range(mask_ger.shape[1]):

                # test if grid point is within polygon
                # only if not already in any of the other polygons
                if mask_ger[y, x] == False:
                    point = Point((x, y))
                    mask_ger[y, x] = polygon.contains(point)

    # cut mask to bounding box and change data type
    mask_ger = mask_ger[box[0] : box[2], box[1] : box[3]]
    mask_ger = mask_ger.astype(bool)

    # add mask to dataset
    ds_mask["German_border"] = (("y", "x"), mask_ger)

    # output
    if outfolder is not None:
        ds_mask.to_netcdf(outfolder + "spatial_masks.nc")
