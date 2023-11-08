import numpy as np
import xarray as xr
import pandas as pd
import yaml
from tqdm import tqdm
import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging

import rain_data as rd


def run(setup, workers=None):
    """
    Wrapper for the Kriging calculation. Involves filtering
    of the observational data.
    """

    with open("settings.yml") as file:
        dict_options = yaml.safe_load(file)
    globals().update(dict_options[setup])

    # array that stores considered time steps
    ds_t = xr.open_dataset(datafolder + "timeseries_labels.nc")
    t_array_calculated = ds_t.time.where(ds_t.label_calculated, drop=True).values

    # load observations
    ds_cml = xr.open_dataset(datafolder + obsfolder + "CML_for_reconstruction.nc")
    ds_rg = xr.open_dataset(datafolder + obsfolder + "rg_for_reconstruction.nc")

    if workers is not None:
        # run kriging parallel

        # lazy evaluation
        results = [
            dask.delayed(run_timestep)(
                ds_cml_timestep=ds_cml.sel(time=t),
                ds_rg_timestep=ds_rg.sel(time=t),
                box=box,
                vario_model=vario_model,
                nclosest=nclosest,
                nlags=nlags,
                outfolder=datafolder + krifolder,
            )
            for t in t_array_calculated
        ]

        # computation
        csz = workers  # chunk size
        ds_list = []
        for i in tqdm(range(int(len(t_array_calculated) / csz) + 1)):
            ds_list.extend(dask.compute(*results[csz * i : csz * (i + 1)]))

    else:

        [
            run_timestep(
                ds_cml_timestep=ds_cml.sel(time=t),
                ds_rg_timestep=ds_rg.sel(time=t),
                box=box,
                vario_model=vario_model,
                nclosest=nclosest,
                nlags=nlags,
                outfolder=datafolder + krifolder,
            )
            for t in t_array_calculated
        ]


def combine_save_timeseries(setup):
    """
    Load single time steps, combine, and save again.
    """

    with open("settings.yml") as file:
        dict_options = yaml.safe_load(file)
    globals().update(dict_options[setup])

    # array that stores considered time steps
    ds_t = xr.open_dataset(datafolder + "timeseries_labels.nc")
    t_array_calculated = ds_t.time.where(ds_t.label_calculated, drop=True).values

    # combine timesteps
    ds_list = []
    for t in tqdm(t_array_calculated):
        ds_ = xr.open_dataset(datafolder + krifolder + "kriging_%s.nc" % t)
        ds_list.append(ds_)
    ds = xr.concat(ds_list, dim="time")

    # define encoding
    encoding = {
        "rainfall_amount": {
            "zlib": True,
            "chunksizes": [1, len(ds.y), len(ds.x)],
        }
    }

    # save specifying chunks
    ds.to_netcdf(datafolder + "temp/kriging_%s.nc" % year, encoding=encoding)


def run_timestep(
    ds_cml_timestep,
    ds_rg_timestep,
    box,
    vario_model,
    nclosest,
    nlags,
    calc_vario_params_separately=False,
    outfolder="",
):
    """
    Run Kriging for a single time step.
    """

    t = ds_cml_timestep.time.values

    # observations
    y = np.concatenate((ds_cml_timestep.y.values, ds_rg_timestep.y.values))
    x = np.concatenate((ds_cml_timestep.x.values, ds_rg_timestep.x.values))
    rain = np.concatenate((ds_cml_timestep.rain.values, ds_rg_timestep.rain.values))

    # filter nans
    y, x, rain = rd.filter_nans_b(y, x, rain)

    # if variogram parameters shall be calculated by skgstat
    if calc_vario_params_separately:
        vario_params = calc_vario_params(y, x, rain, vario_model)
    else:
        vario_params = None

    # Kriging
    ok = OrdinaryKriging(
        x,
        y,
        rain,
        variogram_model=vario_model,
        variogram_parameters=vario_params,
        verbose=False,
        enable_plotting=False,
        nlags=nlags,
    )

    # needs coords to be floats not integers
    rain, ss = ok.execute(
        "grid",
        np.arange(box[3] - box[1]).astype(float),
        np.arange(box[2] - box[0]).astype(float),
        backend="C",
        n_closest_points=nclosest,
    )

    # variogram points
    parameters = ok.get_variogram_points()

    ds = build_dataset(
        t,
        box,
        nclosest,
        vario_model,
        rain=rain,
        parameters=parameters,
    )

    save_timestep(t, ds, outfolder)


def filter_save_timesteps(label, t_array, box, nlags, outfolder, save=False):
    """
    Save time steps for which no Kriging calculation shall be conducted
    as NaN dataset. The time steps are removed from the time array.
    """

    cnt = 0
    t_array_new = []
    for i, t in enumerate(t_array):

        if ~label.sel(time=t):
            cnt += 1

            if save:
                ds = build_dataset(
                    t,
                    box,
                    nclosest,
                    vario_model,
                    rain=np.ones((box[2] - box[0], box[3] - box[1])) * np.nan,
                    parameters=np.ones((2, nlags - 2)) * np.nan,
                )
                save_timestep(t, ds, outfolder)

        else:
            t_array_new.append(t)
    t_array_new = np.array(t_array_new)

    print(
        "%i time steps filtered based on %s. %i time steps remain."
        % (cnt, label.name, len(t_array_new))
    )

    return t_array_new


def build_dataset(t, box, nclosest, vario_model, rain, parameters):
    """
    Create an xarray dataset of results.
    """

    # combine to dataset
    ds = xr.Dataset(
        {
            "rainfall_amount": (["y", "x"], rain),
            "variogram_dist": (["nlags"], parameters[0]),
            "variogram_value": (["nlags"], parameters[1]),
            "n_closest": nclosest,
            "vario_model": vario_model,
            "time": t,
        },
        coords={
            "y": (["y"], np.arange(box[0], box[2]).astype(float)),
            "x": (["x"], np.arange(box[1], box[3]).astype(float)),
        },
    )

    return ds


def save_timestep(t, ds, outfolder):
    """
    Save encoded.
    """

    # define encoding
    encoding = {
        "rainfall_amount": {
            "zlib": True,
        }
    }

    # save
    ds.to_netcdf(
        outfolder + "kriging_%s.nc" % t,
        encoding=encoding,
    )


def calc_vario_params(y, x, rain, vario_model):
    """
    Wrapper function for estimating kriging parameters.
    """

    # calculate variogram parameters
    variogram = skg.Variogram(
        coordinates=np.vstack((x, y)).T, values=rain, model=vario_model
    )

    # variogram parameters in order: sill, range, nugget as required by pykrige
    vario_params = np.array(
        (variogram.parameters[1], variogram.parameters[0], variogram.parameters[2])
    )

    return vario_params
