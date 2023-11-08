import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import yaml

import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from tqdm import tqdm
import datetime

import plot_functions as pf
import rain_data as rd
import SAL_calculation as sc


def calculate_metrics(setup, workers=None):
    """
    Calculate and save performance metrics.
    """

    with open("settings.yml") as file:
        dict_options = yaml.safe_load(file)
    globals().update(dict_options[setup])

    # array that stores considered time steps
    ds_t = xr.open_dataset(datafolder + "timeseries_labels.nc")
    t_array_calculated = ds_t.time.where(ds_t.label_calculated, drop=True).values

    # =================================
    # load data

    # random mixing
    ds_rms = rd.load_field_at_time(datafolder + rm_data, t_array_calculated)
    rms = rd.get_rain_dataarray(ds_rms, "final_fields", chunk=True)

    # kriging
    ds_kri = rd.load_field_at_time(datafolder + kri_data, t_array_calculated)
    ds_kri = rd.coords_as_int(ds_kri)
    ds_kri = rd.take_subset_field(ds_kri, box)
    kri = rd.get_rain_dataarray(ds_kri, "rainfall_amount", chunk=True)

    # radolan
    ds_rad = rd.load_field_at_time(rad_data, t_array_calculated)
    ds_rad = rd.shift_coords_to_origin(ds_rad)
    ds_rad = rd.coords_as_int(ds_rad)
    ds_rad = rd.take_subset_field(ds_rad, box)
    rad = rd.get_rain_dataarray(ds_rad, "rainfall_amount", chunk=True)

    # spatial mask
    ds_dist_mask = xr.open_dataset(datafolder + "spatial_masks.nc")
    dist_mask = ds_dist_mask[mask_type]

    # define spatial rad masks
    rad_mask = ~rad.isnull()

    # mask fields (if rad will be interpolated don't use rad_mask here)
    rms, rad, kri = rd.mask_fields([rms, rad, kri], [rad_mask, dist_mask])

    # set negative values of kriging to zero
    kri = kri.where(kri >= 0, 0)

    # # new arrays where missing values are set to zero for SAL calculation
    rms0 = rms.where(~rms.isnull(), 0)
    kri0 = kri.where(~kri.isnull(), 0)
    rad0 = rad.where(~rad.isnull(), 0)

    #     # =====================
    #     # test INTERPOLATION for revision (tried but not used in the end)

    #     rad_mask_inside = ~dist_mask | rad_mask
    #     radNan = rad0.where(rad_mask_inside, np.nan)

    #     # ---------------------
    #     # first approach
    #     rad_interpx = radNan.interpolate_na(dim="x")
    #     rad_interpy = radNan.interpolate_na(dim="y")
    #     rad_interp = (rad_interpx + rad_interpy) / 2

    #     # ---------------------
    #     # second approach: wradlib
    #     import wradlib.ipol as ipol

    #     radNan.load()

    #     # Target coordinates
    #     trgx = radNan.x.values
    #     trgy = radNan.y.values
    #     trg = np.meshgrid(trgx, trgy)
    #     trg = np.vstack((trg[0].ravel(), trg[1].ravel())).T

    #     # interpolation loop
    #     interp_fields = []
    #     for t in tqdm(radNan.time):
    #         orig = radNan.sel(time=t).values
    #         orig_flat = orig.flatten()
    #         interp_flat = ipol.interpolate(trg, trg, orig_flat, ipol.Linear)
    #         interp = interp_flat.reshape(orig.shape)
    #         interp_fields.append(interp)
    #     radInterp = radNan.copy()
    #     radInterp = (["time", "y", "x"], np.array(radInterp))

    # ===========================================
    # SAL calculation (choose rad0 or rad_interp or radInterp)

    print("Calculating SAL parameters ...")
    # lazy evaluation or direct (non-parallel) calculation
    sal_rms = sc.SAL_timeseries(rms0, rad0, t_array_calculated, box, workers=workers)
    sal_kri = sc.SAL_timeseries(kri0, rad0, t_array_calculated, box, workers=workers)

    # save SAL results to disk
    sal_rms.to_netcdf(datafolder + "temp/SAL_rms.nc")

    # save SAL results to disk
    sal_kri.to_netcdf(datafolder + "temp/SAL_kri.nc")

    # SAL of mean fields
    for n in [3, 5, 10, 20]:
        ind = np.arange(n)
        rmm0 = rms0.isel(nfields=ind).mean("nfields")
        sal_rmm = sc.SAL_timeseries(
            rmm0, rad0, t_array_calculated, box, workers=workers
        )
        sal_rmm.to_netcdf(datafolder + "temp/SAL_rmm_mean%i.nc" % n)

    # ============================================
    # standard performance indices

    pcc_rms = xr.corr(rms, rad, dim=["y", "x"])
    # pcc_kri = xr.corr(kri, rad, dim=["y", "x"])

    rmse_rms = ((rms - rad) ** 2).mean(["y", "x"])
    # rmse_kri = ((kri - rad) ** 2).mean(["y", "x"])

    bias_rms = (rms - rad).mean(["y", "x"]) / rad.mean(["y", "x"])
    # bias_kri = (kri - rad).mean(["y", "x"]) / rad.mean(["y", "x"])

    # to datasets
    ds_rms_pixel = xr.Dataset()
    ds_rms_pixel["pcc"] = (("time", "nfields"), pcc_rms)
    ds_rms_pixel["rmse"] = (("time", "nfields"), rmse_rms)
    ds_rms_pixel["bias"] = (("time", "nfields"), bias_rms)
    ds_rms_pixel = ds_rms_pixel.assign_coords({"time": ("time", t_array_calculated)})

    ds_kri_pixel = xr.Dataset()
    ds_kri_pixel["pcc"] = (("time"), pcc_kri)
    ds_kri_pixel["rmse"] = (("time"), rmse_kri)
    ds_kri_pixel["bias"] = (("time"), bias_kri)
    ds_kri_pixel = ds_kri_pixel.assign_coords({"time": ("time", t_array_calculated)})

    # save to disk
    ds_rms_pixel.to_netcdf(datafolder + "temp/Pixelmetrics_rms.nc")
    ds_kri_pixel.to_netcdf(datafolder + "temp/Pixelmetrics_kri.nc")

    # pixelmetrics of mean fields
    for n in [3, 5, 10, 20]:
        ind = np.arange(n)
        rmm_ = rms.isel(nfields=ind).mean("nfields")
        pcc_rmm_ = xr.corr(rmm_, rad, dim=["y", "x"])
        rmse_rmm_ = ((rmm_ - rad) ** 2).mean(["y", "x"])
        bias_rmm_ = (rmm_ - rad).mean(["y", "x"]) / rad.mean(["y", "x"])

        ds_rmm_pixel_ = xr.Dataset()
        ds_rmm_pixel_["pcc"] = (("time"), pcc_rmm_)
        ds_rmm_pixel_["rmse"] = (("time"), rmse_rmm_)
        ds_rmm_pixel_["bias"] = (("time"), bias_rmm_)
        ds_rms_pixel_ = ds_rms_pixel.assign_coords(
            {"time": ("time", t_array_calculated)}
        )

        ds_rmm_pixel_.to_netcdf(datafolder + "temp/Pixelmetrics_rmm_mean%i.nc" % n)

    # =================================
    # for revision
    # calculate several combinations of mean fields for each size or revision

    # holds for both
    runs = 100
    indexes = np.arange(20)

    # SAL of mean fields
    for n in [
        # 3, 5,
        10,
        20,
    ]:
        ds_list = []
        for i in range(runs):
            np.random.shuffle(indexes)
            ind = indexes[:n]
            rmm0 = rms0.isel(nfields=ind).mean("nfields")
            sal_rmm = sc.SAL_timeseries(
                rmm0, rad0, t_array_calculated, box, workers=workers
            )
            sal_rmm["run"] = i
            ds_list.append(sal_rmm)
        sal_out = xr.concat(ds_list, dim="run")
        sal_out.to_netcdf(datafolder + "temp/SAL_rmm_mean%i_revision.nc" % n)
        print("saved SAL for ens. mean of %i members" % n)

    # pixelmetrics of mean fields
    for n in [3, 5, 10]:
        ds_list = []
        for i in range(runs):
            np.random.shuffle(indexes)
            ind = indexes[:n]
            rmm_ = rms.isel(nfields=ind).mean("nfields")
            pcc_rmm_ = xr.corr(rmm_, rad, dim=["y", "x"])
            rmse_rmm_ = ((rmm_ - rad) ** 2).mean(["y", "x"])
            bias_rmm_ = (rmm_ - rad).mean(["y", "x"]) / rad.mean(["y", "x"])

            ds_rmm_pixel_ = xr.Dataset()
            ds_rmm_pixel_["pcc"] = (("time"), pcc_rmm_)
            ds_rmm_pixel_["rmse"] = (("time"), rmse_rmm_)
            ds_rmm_pixel_["bias"] = (("time"), bias_rmm_)
            ds_rmm_pixel_ = ds_rmm_pixel_.assign_coords(
                {"time": ("time", t_array_calculated)}
            )

            ds_rmm_pixel_["run"] = i
            ds_list.append(ds_rmm_pixel_)
        pixel_out = xr.concat(ds_list, dim="run")
        pixel_out.to_netcdf(datafolder + "temp/Pixelmetrics_rmm_mean%i_revision.nc" % n)


# ================================================================================
# ================================================================================


def analysis(setup):
    """
    Paper analysis. Loads fields and metrics and calls plot functions.
    """

    # =================================
    # load settings and create time array

    with open("settings.yml") as file:
        dict_options = yaml.safe_load(file)
    globals().update(dict_options[setup])

    # array that stores considered time steps
    ds_t = xr.open_dataset(datafolder + "timeseries_labels.nc")
    t_array_calculated = ds_t.time.where(ds_t.label_calculated, drop=True).values

    tMapsList = pd.to_datetime([t1, t2, t3])
    tLabels = ["June 28, 01:50"]

    # =================================
    # load data

    # random mixing
    ds_rms = rd.load_field_at_time(datafolder + rm_data, t_array_calculated)
    rms = rd.get_rain_dataarray(ds_rms, "final_fields", chunk=False)

    # kriging
    ds_kri = rd.load_field_at_time(datafolder + kri_data, t_array_calculated)
    ds_kri = rd.coords_as_int(ds_kri)
    ds_kri = rd.take_subset_field(ds_kri, box)
    kri = rd.get_rain_dataarray(ds_kri, "rainfall_amount", chunk=False)

    # radolan
    ds_rad = rd.load_field_at_time(rad_data, t_array_calculated)
    ds_rad = rd.shift_coords_to_origin(ds_rad)
    ds_rad = rd.coords_as_int(ds_rad)
    ds_rad = rd.take_subset_field(ds_rad, box)
    rad = rd.get_rain_dataarray(ds_rad, "rainfall_amount", chunk=False)

    # spatial mask
    ds_dist_mask = xr.open_dataset(datafolder + "spatial_masks.nc")
    dist_mask = ds_dist_mask[mask_type]

    # wetness indicator
    rad_mean_all = xr.open_dataset(
        datafolder + "radolan_spatial_mean_timeseries.nc"
    ).rainfall_amount

    # load preprocessed observations
    ds_cml = xr.open_dataset(datafolder + obsfolder + "CML_preprocessed.nc")
    ds_rg = xr.open_dataset(datafolder + obsfolder + "rg_preprocessed.nc")

    # =================================
    # find arrays of time steps where CMLs are not available
    # and where observations are too wet
    beginDryPeriod, endDryPeriod = rd.find_periods(
        ds_t.time.values, ds_t.time.where(~ds_t.label_wet, drop=True).values
    )

    beginNanPeriod, endNanPeriod = rd.find_periods(
        ds_t.time.values, ds_t.time.where(~ds_t.label_available, drop=True).values
    )

    beginCalcPeriod, endCalcPeriod = rd.find_periods(
        ds_t.time.values, ds_t.time.where(ds_t.label_calculated, drop=True).values
    )

    # handle overlapping periods (Nan over Dry)
    for startNan, endNan in zip(beginNanPeriod, endNanPeriod):
        idx = len(np.where(startNan >= np.array(endDryPeriod))[0])
        endDryPeriod.insert(idx, startNan)
        beginDryPeriod.insert(idx + 1, endNan)

    # time of beginning, end plus selected time steps
    specialDates = [pd.to_datetime(ds_t.time.values[0])]
    specialDates.extend(tMapsList)
    specialDates.append(pd.to_datetime(ds_t.time.values[-1]))

    # =================================
    # color options

    # base colormap
    baseCmp = mpl.cm.YlGnBu(np.arange(256))[50:]
    baseCmp = ListedColormap(baseCmp, name="myColorMap", N=baseCmp.shape[0])

    # levels and colors of standard rain colormap
    cLevels = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cMapRain = pf.colormap(cLevels, extend=2, belowColor="#FFFFFF", baseCmp=baseCmp)

    # several binary maps (w: white, g:grey, b:black, o:orange)
    cmap_binary_wb, cmap_binary_wg, cmap_binary_wo = pf.binary_colormaps()

    # invert colormap for obs. distance mask
    cmap_binary_wg_inv = cmap_binary_wg[::-1]

    # ---------------------------------------------
    # special colormap for showing Nans too

    # slightly different levels
    cLevels0 = [-0.00001, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cLevels0_legend = [0, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    precipColors = []

    # nan (grey) and zero (white) color
    precipColors.append("#818181")
    precipColors.append("#FFFFFF")

    # rest of colormap filled by colors of base colormap
    nBaseColors = len(cLevels0) - 1
    for i in np.linspace(0, 1, nBaseColors):
        rgb = baseCmp(i)[:3]
        precipColors.append(mpl.colors.rgb2hex(rgb))

    # transform to cmap
    cMapRain0 = ListedColormap(precipColors)

    # =================================
    # plot radolan coverage for answer

    # define spatial rad masks
    rad_mask = ~rad.isnull()

    pf.Radolan_coverage_answer(
        dist_mask,
        rad_mask,
        cmap_binary_wg,
        border_data=datafolder + border_data,
        outfolder="temp/",
    )

    # =================================
    # filter visualization

    if plot_maps:

        for location_name in ["west", "south"]:

            if location_name == "west":
                # spatial subset
                # ymin, xmin, ymax, xmax = 450, 350, 550, 450
                ymin, xmin, ymax, xmax = 550, 500, 650, 600

                # inset position
                xins, yins, wins, hins = 0.47, 0.08, 0.5, 0.5

                # select specific time steps
                t_sels = np.array(
                    pd.to_datetime(
                        [
                            "2019-07-28T13:50",
                            "2019-08-18T12:50",
                        ]
                    )
                )
                tLabelsSpatialSanityExamples = ["Jul 28, 13:50", "Jun 11, 02:50"]

            elif location_name == "south":
                # spatial subset
                ymin, xmin, ymax, xmax = 150, 550, 250, 650

                # inset position
                xins, yins, wins, hins = 0.03, 0.47, 0.5, 0.5

                # select specific time steps
                t_sels = np.array(
                    pd.to_datetime(
                        ["2019-06-11T07:50", "2019-07-07T01:50", "2019-07-28T10:50"]
                    )
                )
                tLabelsSpatialSanityExamples = [
                    "Jun 11, 07:50",
                    "Jul 7, 01:50",
                    "Jul 28, 10:50",
                ]

            cxins = xins + wins / 8
            cyins = yins + 0.03
            cwins = wins * 3 / 4
            chins = 0.01

            # first of list is example time step for paper, others are situations with more rain
            for t_sel, tLabel in zip(t_sels[:1], tLabelsSpatialSanityExamples[:1]):

                # subset
                ds_cml_t_subset = rd.reduce_time_and_space(
                    ds_cml, t_sel, ymin, xmin, ymax, xmax
                )

                # skip time step if all cmls are zero
                if ds_cml_t_subset.rain.max() == 0:
                    print("No plot as all CMLs are dry.")
                    continue

                # don't show nans
                ds_cml_t_subset = ds_cml_t_subset.where(
                    ~ds_cml_t_subset.rain.isnull(), drop=True
                )

                pf.filter_visualization(
                    rad,
                    ds_cml_t_subset,
                    t_sel,
                    tLabel,
                    box,
                    inset_loc=[xins, yins, wins, hins],
                    c_loc=[cxins, cyins, cwins, chins],
                    location_name=location_name,
                    ymin=ymin,
                    ymax=ymax,
                    xmin=xmin,
                    xmax=xmax,
                    cMap_standard=cMapRain,
                    cLevels_standard=cLevels,
                    border_data=datafolder + border_data,
                    fig_width=3.5,
                    outfolder="temp/",
                )

    # =================================
    # observation overview map

    if plot_maps:
        pf.overview_observation_map(
            ds_cml,
            ds_rg,
            box,
            dist_mask,
            cmap_binary_wg_inv,
            datafolder + border_data,
            fig_width=3.5,
            outfolder="temp/",
        )

    # =================================
    # further processing

    # selected time steps
    rmsTMaps = rms.sel(time=tMapsList)
    kriTMaps = kri.sel(time=tMapsList)
    radTMaps = rad.sel(time=tMapsList)

    # bring selected maps to xarray object (faster)
    # rmsTMaps = rmsTMaps.compute()
    # radTMaps = radTMaps.compute()

    # set Radolan nans to -99
    radTMaps_overviewPlot = radTMaps.where(~radTMaps.isnull(), other=-99)

    # reduce observation data set to selected time steps
    ds_cmlTMaps = ds_cml.sel(time=tMapsList)
    ds_rgTMaps = ds_rg.sel(time=tMapsList)

    # =================================
    # results overview map

    if plot_maps:
        pf.overview_tseries_plus_selected(
            rad_mean_all,
            ds_cmlTMaps,
            ds_rgTMaps,
            radTMaps_overviewPlot,
            rmsTMaps,
            ds_t.time.values,
            tMapsList,
            tLabelsFull,
            box,
            beginCalcPeriod,
            endCalcPeriod,
            beginNanPeriod,
            endNanPeriod,
            beginDryPeriod,
            endDryPeriod,
            specialDates,
            cLevels,
            cMapRain,
            cLevels0,
            cMapRain0,
            datafolder + border_data,
            fig_width=5.5,
            outfolder="temp/",
        )

    # =================================
    # further processing

    # set negative values of kriging to zero
    kriTMaps = kriTMaps.where(kriTMaps >= 0, 0)

    # define spatial masks
    rad_mask = ~radTMaps.isnull()

    # mask fields (by radolan mask and observation distance mask)
    rmsTMaps, radTMaps, kriTMaps = rd.mask_fields(
        [rmsTMaps, radTMaps, kriTMaps], [rad_mask, dist_mask]
    )

    # set missing values to zero
    rmsTMaps0 = rmsTMaps.where(~rmsTMaps.isnull(), 0)
    kriTMaps0 = kriTMaps.where(~kriTMaps.isnull(), 0)
    radTMaps0 = radTMaps.where(~radTMaps.isnull(), 0)

    # =================================
    # load metrics

    # load SAL
    ds_rms_SAL = xr.open_dataset(datafolder + "metrics/march_22/SAL_rms.nc")
    ds_kri_SAL = xr.open_dataset(datafolder + "metrics/march_22/SAL_kri.nc")

    # load SAL results of mean fields
    ds_rmm_3_SAL = xr.open_dataset(datafolder + "metrics/march_22/SAL_rmm_mean3.nc")
    ds_rmm_5_SAL = xr.open_dataset(datafolder + "metrics/march_22/SAL_rmm_mean5.nc")
    ds_rmm_10_SAL = xr.open_dataset(datafolder + "metrics/march_22/SAL_rmm_mean10.nc")
    ds_rmm_20_SAL = xr.open_dataset(datafolder + "metrics/march_22/SAL_rmm_mean20.nc")

    # load pixelmetrics
    ds_rms_pixel = xr.open_dataset(datafolder + "metrics/march_22/Pixelmetrics_rms.nc")
    ds_kri_pixel = xr.open_dataset(datafolder + "metrics/march_22/Pixelmetrics_kri.nc")

    # load pixelmetrics of mean fields
    ds_rmm_3_pixel = xr.open_dataset(
        datafolder + "metrics/march_22/Pixelmetrics_rmm_mean3.nc"
    )
    ds_rmm_5_pixel = xr.open_dataset(
        datafolder + "metrics/march_22/Pixelmetrics_rmm_mean5.nc"
    )
    ds_rmm_10_pixel = xr.open_dataset(
        datafolder + "metrics/march_22/Pixelmetrics_rmm_mean10.nc"
    )
    ds_rmm_20_pixel = xr.open_dataset(
        datafolder + "metrics/march_22/Pixelmetrics_rmm_mean20.nc"
    )

    # =================================
    # plot test combinations of mean fields for answer to reviewer

    various_combinations(ds_rms_SAL, ds_rms_pixel, datafolder)

    # =================================
    # SAL maps for selected time steps

    if plot_maps:
        for t, outname, title in zip(np.array(tMapsList), ["A", "B", "C"], tLabelsFull):

            # RMM maps with sal bars
            pf.RMM_with_SAL_bars_extended(
                t,
                outname,
                title,
                rmsTMaps0.sel(time=t),
                radTMaps0.sel(time=t),
                kriTMaps0.sel(time=t),
                ds_rmm_3_SAL.sel(time=t),
                ds_rmm_5_SAL.sel(time=t),
                ds_rmm_10_SAL.sel(time=t),
                ds_rmm_20_SAL.sel(time=t),
                ds_kri_SAL.sel(time=t),
                nfields,
                cMapRain,
                cLevels,
                cmap_binary_wb,
                datafolder + border_data,
                outfolder="temp/",
            )

            # maps with sal bars
            pf.map_with_SAL_bars(
                t,
                outname,
                title,
                rmsTMaps0.sel(time=t),
                radTMaps0.sel(time=t),
                kriTMaps0.sel(time=t),
                ds_rms_SAL.sel(time=t),
                ds_kri_SAL.sel(time=t),
                nfields,
                cMapRain,
                cLevels,
                cmap_binary_wb,
                datafolder + border_data,
                outfolder="temp/",
            )

        pf.map_with_SAL_bars_together(
            ts=tMapsList[1:],
            outname="BC",
            titles=tLabelsFull[1:],
            rms0_s=[rmsTMaps0.sel(time=tMapsList[1]), rmsTMaps0.sel(time=tMapsList[2])],
            rad0_s=[radTMaps0.sel(time=tMapsList[1]), radTMaps0.sel(time=tMapsList[2])],
            kri0_s=[kriTMaps0.sel(time=tMapsList[1]), kriTMaps0.sel(time=tMapsList[2])],
            ds_rm_SAL_s=[
                ds_rms_SAL.sel(time=tMapsList[1]),
                ds_rms_SAL.sel(time=tMapsList[2]),
            ],
            ds_kri_SAL_s=[
                ds_kri_SAL.sel(time=tMapsList[1]),
                ds_kri_SAL.sel(time=tMapsList[2]),
            ],
            nfields=nfields,
            cmap_rain=cMapRain,
            clevs_rain=cLevels,
            cmap_binary=cmap_binary_wb,
            border_data=datafolder + border_data,
            fig_width=5.5,
            outfolder="temp/",
        )

    # ------------------------------------------------------------------
    # calculate median over time

    df_median_SAL = pd.DataFrame(
        index=["1_rm", "3_rm", "5_rm", "10_rm", "20_rm", "kri"]
    )

    for param in ["S", "A", "L"]:
        medianList = []
        medianList.append(
            ds_rms_SAL["m%s" % param].isel(nfields_sim=0).quantile(0.5).values
        )
        medianList.append(ds_rmm_3_SAL[param].quantile(0.5).values)
        medianList.append(ds_rmm_5_SAL[param].quantile(0.5).values)
        medianList.append(ds_rmm_10_SAL[param].quantile(0.5).values)
        medianList.append(ds_rmm_20_SAL[param].quantile(0.5).values)
        medianList.append(ds_kri_SAL[param].quantile(0.5).values)

        df_median_SAL[param] = medianList

    df_median_pixel = pd.DataFrame(
        index=["1_rm", "3_rm", "5_rm", "10_rm", "20_rm", "kri"]
    )

    for param in ["pcc", "rmse", "bias"]:
        medianList = []
        medianList.append(ds_rms_pixel[param].isel(nfields=0).quantile(0.5).values)
        medianList.append(ds_rmm_3_pixel[param].quantile(0.5).values)
        medianList.append(ds_rmm_5_pixel[param].quantile(0.5).values)
        medianList.append(ds_rmm_10_pixel[param].quantile(0.5).values)
        medianList.append(ds_rmm_20_pixel[param].quantile(0.5).values)
        medianList.append(ds_kri_pixel[param].quantile(0.5).values)

        df_median_pixel[param] = medianList

    # =================================
    # statistical plots

    if plot_statistics:

        # Pixel boxplots
        pf.Pixel_boxplots(
            ds_rms_pixel,
            ds_rmm_20_pixel,
            ds_kri_pixel,
            fig_width=3.5,
            outfolder="temp/",
        )

        # pixel boxplot for answer
        pf.Pixel_boxplots_answer(
            ds_rms_pixel,
            fig_width=5.5,
            outfolder="temp/",
        )

        fig = pf.SAL_boxplot_contour(
            ds_rms_SAL, ds_rmm_20_SAL, ds_kri_SAL, fig_width=5.5, outfolder="temp/"
        )

        pf.SAL_six_scatter(
            ds_rms_SAL,
            ds_kri_SAL,
            rad_mean_all,
            ds_t.label_calculated,
            fig_width=5.5,
            outfolder="temp/",
        )

        # various ensemble sizes
        pf.parameters_for_mean_fields_split(
            df_median_SAL, df_median_pixel, fig_width=5.5, outfolder="temp/"
        )

    # ======================================================================
    # TABLE

    # ----------------------------------------------------------------------
    # median table
    df_tMaps = pd.DataFrame(index=tLabelsFull)

    params = ["S", "A", "L", "L1"]

    for p in params:
        df_tMaps["%s (RM)" % p] = ds_rms_SAL[p].sel(time=tMapsList).values
        df_tMaps["%s (RMM)" % p] = ds_rmm_20_SAL[p].sel(time=tMapsList).values
        df_tMaps["%s (KRI)" % p] = ds_kri_SAL[p].sel(time=tMapsList).values

    # add standard perform indices
    df_tMaps["PCC (RM)"] = (
        ds_rms_pixel["pcc"].sel(time=tMapsList).quantile(0.5, "nfields").values
    )
    df_tMaps["PCC (RMM)"] = ds_rmm_20_pixel["pcc"].sel(time=tMapsList).values
    df_tMaps["PCC (KRI)"] = ds_kri_pixel["pcc"].sel(time=tMapsList).values
    df_tMaps["RMSE (RM)"] = (
        ds_rms_pixel["rmse"].sel(time=tMapsList).quantile(0.5, "nfields").values
    )
    df_tMaps["RMSE (RMM)"] = ds_rmm_20_pixel["rmse"].sel(time=tMapsList).values
    df_tMaps["RMSE (KRI)"] = ds_kri_pixel["rmse"].sel(time=tMapsList).values
    df_tMaps["BIAS (RM)"] = (
        ds_rms_pixel["bias"].sel(time=tMapsList).quantile(0.5, "nfields").values
    )
    df_tMaps["BIAS (RMM)"] = ds_rmm_20_pixel["bias"].sel(time=tMapsList).values
    df_tMaps["BIAS (KRI)"] = ds_kri_pixel["bias"].sel(time=tMapsList).values

    # df for median over time
    df_all = pd.DataFrame(index=["Median over time"])

    params = ["S", "A", "L", "L1"]

    for p in params:
        df_all["%s (RM)" % p] = (
            ds_rms_SAL[p].sel(time=t_array_calculated).quantile(0.5).values
        )
        df_all["%s (RMM)" % p] = (
            ds_rmm_20_SAL[p].sel(time=t_array_calculated).quantile(0.5).values
        )
        df_all["%s (KRI)" % p] = (
            ds_kri_SAL[p].sel(time=t_array_calculated).quantile(0.5).values
        )

    # add standard perform indices
    df_all["PCC (RM)"] = (
        ds_rms_pixel["pcc"]
        .sel(time=t_array_calculated)
        .quantile(0.5, "nfields")
        .quantile(0.5)
        .values
    )
    df_all["PCC (RMM)"] = (
        ds_rmm_20_pixel["pcc"].sel(time=t_array_calculated).quantile(0.5).values
    )
    df_all["PCC (KRI)"] = (
        ds_kri_pixel["pcc"].sel(time=t_array_calculated).quantile(0.5).values
    )
    df_all["RMSE (RM)"] = (
        ds_rms_pixel["rmse"]
        .sel(time=t_array_calculated)
        .quantile(0.5, "nfields")
        .quantile(0.5)
        .values
    )
    df_all["RMSE (RMM)"] = (
        ds_rmm_20_pixel["rmse"].sel(time=t_array_calculated).quantile(0.5).values
    )
    df_all["RMSE (KRI)"] = (
        ds_kri_pixel["rmse"].sel(time=t_array_calculated).quantile(0.5).values
    )
    df_all["BIAS (RM)"] = (
        ds_rms_pixel["bias"]
        .sel(time=t_array_calculated)
        .quantile(0.5, "nfields")
        .quantile(0.5)
        .values
    )
    df_all["BIAS (RMM)"] = (
        ds_rmm_20_pixel["bias"].sel(time=t_array_calculated).quantile(0.5).values
    )
    df_all["BIAS (KRI)"] = (
        ds_kri_pixel["bias"].sel(time=t_array_calculated).quantile(0.5).values
    )

    # ----------------------------------------------------------------------
    # combine selected time steps with total mean / median
    df_table = pd.concat([df_tMaps, df_all]).transpose()
    df_table = np.around(df_table, decimals=3)

    print(df_table.to_latex())


# secondary function
def various_combinations(ds_rms_SAL, ds_rms_pixel, datafolder):

    # load SAL results of mean fields
    ds_rmm_3_SAL_revision = xr.open_dataset(
        datafolder + "metrics/july_22/SAL_rmm_mean3_revision.nc"
    )
    ds_rmm_5_SAL_revision = xr.open_dataset(
        datafolder + "metrics/july_22/SAL_rmm_mean5_revision.nc"
    )
    ds_rmm_10_SAL_revision = xr.open_dataset(
        datafolder + "metrics/july_22/SAL_rmm_mean10_revision.nc"
    )

    # load pixelmetrics of mean fields
    ds_rmm_3_pixel_revision = xr.open_dataset(
        datafolder + "metrics/july_22/Pixelmetrics_rmm_mean3_revision.nc"
    )
    ds_rmm_5_pixel_revision = xr.open_dataset(
        datafolder + "metrics/july_22/Pixelmetrics_rmm_mean5_revision.nc"
    )
    ds_rmm_10_pixel_revision = xr.open_dataset(
        datafolder + "metrics/july_22/Pixelmetrics_rmm_mean10_revision.nc"
    )

    # ===========================
    # median over time
    runs = len(ds_rmm_3_SAL_revision.run)

    # --------------------------------------------
    # SAL
    df_median_SAL_revision = pd.DataFrame()

    size_list = []
    param_list = []
    run_list = []
    value_list = []

    for size in [1, 3, 5, 10]:
        for param in ["S", "A", "L"]:
            for run in range(runs):
                if size == 1:
                    value = (
                        ds_rms_SAL["m%s" % param]
                        .quantile(0.5, dim="time")
                        .isel(nfields_sim=run)
                        .values
                    )
                elif size == 3:
                    value = (
                        ds_rmm_3_SAL_revision[param]
                        .quantile(0.5, dim="time")
                        .isel(run=run)
                        .values
                    )
                elif size == 5:
                    value = (
                        ds_rmm_5_SAL_revision[param]
                        .quantile(0.5, dim="time")
                        .isel(run=run)
                        .values
                    )
                elif size == 10:
                    value = (
                        ds_rmm_10_SAL_revision[param]
                        .quantile(0.5, dim="time")
                        .isel(run=run)
                        .values
                    )

                size_list.append(size)
                param_list.append(param)
                run_list.append(run)
                value_list.append(value)

    df_median_SAL_revision["Size"] = size_list
    df_median_SAL_revision["Parameter"] = param_list
    df_median_SAL_revision["Runs"] = run_list
    df_median_SAL_revision["Value"] = np.array(value_list, dtype=float)

    # --------------------------------------------
    # Pixel
    df_median_pixel_revision = pd.DataFrame()

    size_list = []
    param_list = []
    run_list = []
    value_list = []

    for size in [1, 3, 5, 10]:
        for param in ["pcc", "rmse", "bias"]:
            for run in range(runs):
                if size == 1:
                    value = (
                        ds_rms_pixel[param]
                        .quantile(0.5, dim="time")
                        .isel(nfields=run)
                        .values
                    )
                elif size == 3:
                    value = (
                        ds_rmm_3_pixel_revision[param]
                        .quantile(0.5, dim="time")
                        .isel(run=run)
                        .values
                    )
                elif size == 5:
                    value = (
                        ds_rmm_5_pixel_revision[param]
                        .quantile(0.5, dim="time")
                        .isel(run=run)
                        .values
                    )
                elif size == 10:
                    value = (
                        ds_rmm_10_pixel_revision[param]
                        .quantile(0.5, dim="time")
                        .isel(run=run)
                        .values
                    )

                size_list.append(size)
                if param == "pcc":
                    param_list.append("PCC")
                if param == "rmse":
                    param_list.append("RMSE")
                if param == "bias":
                    param_list.append("BIAS")
                run_list.append(run)
                value_list.append(value)

    df_median_pixel_revision["Size"] = size_list
    df_median_pixel_revision["Parameter"] = param_list
    df_median_pixel_revision["Runs"] = run_list
    df_median_pixel_revision["Value"] = np.array(value_list, dtype=float)

    # --------------------------------------------
    # plots

    pf.various_combinations_of_mean_fields(
        df_median_SAL_revision, outfolder="temp/", metric="SAL"
    )
    pf.various_combinations_of_mean_fields(
        df_median_pixel_revision, outfolder="temp/", metric="pixel"
    )
