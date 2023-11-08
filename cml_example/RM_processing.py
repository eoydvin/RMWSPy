import os
import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats as st
import scipy.spatial as sp
import scipy.interpolate as interpolate
from cml import *
import gcopula_sparaest as sparest
import bresenhamline
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut
import datetime
from tqdm import tqdm
import yaml
import psutil

# import dask
# from dask.distributed import Client
# from dask_jobqueue import SLURMCluster

import rain_data as rd


def run(
    setup,
    ygrid,
    xgrid,
    nfields=20,
    t_array_calculated=None,
    ds_cml=None,
    ds_rg=None,
    workers=None,
):
    """
    Wrapper for the Random Mixing calculation if a time series is given. Involves filtering of the
    observational data.
    """

    with open("settings.yml") as file:
        dict_options = yaml.safe_load(file)
    globals().update(dict_options[setup])

    if t_array_calculated is None:
        # array that stores considered time steps
        ds_t = xr.open_dataset(datafolder + "timeseries_labels.nc")
        t_array_calculated = ds_t.time.where(ds_t.label_calculated, drop=True).values

    # load observations
    if ds_cml is None:
        ds_cml = xr.open_dataset(datafolder + obsfolder + "CML_for_reconstruction.nc")
    if ds_rg is None:
        ds_rg = xr.open_dataset(datafolder + obsfolder + "rg_for_reconstruction.nc")

    # remove already calculated time steps from t_array
    rm_already_calculated(t_array_calculated, datafolder + rmfolder)

    if workers is not None:
        # lazy evaluation
        results = []
        for t in t_array_calculated:
            results.append(
                dask.delayed(run_timestep)(
                    ds_cml_at_t=ds_cml.sel(time=t),
                    ds_rg_at_t=ds_rg.sel(time=t),
                    ygrid=ygrid,
                    xgrid=xgrid,
                    nfields=nfields,
                    lin_constraints=True,
                    nonlin_constraints=True,
                    outfolder=datafolder + rmfolder,
                )
            )

        # computation
        # dask.compute(*results)
        for i in tqdm(range(int(len(t_array_calculated) / workers) + 1)):

            dask.compute(*results[workers * i : workers * (i + 1)])

    else:

        [
            run_timestep(
                ds_cml_at_t=ds_cml.sel(time=t),
                ds_rg_at_t=ds_rg.sel(time=t),
                ygrid=ygrid,
                xgrid=xgrid,
                nfields=nfields,
                lin_constraints=True,
                nonlin_constraints=True,
                outfolder=datafolder + rmfolder,
            )
            for t in t_array_calculated
        ]


def rm_already_calculated(t_array, outfolder):
    """
    Remove already calculated time steps from t_array.
    """

    cnt = 0
    ind_del = []
    for i, t in enumerate(t_array):
        if os.path.isfile(outfolder + "parameters_%s.nc" % t):
            cnt += 1
            ind_del.append(i)
    t_array = np.delete(t_array, ind_del)
    print(
        "%i time steps have already been calculated. %i time steps remain."
        % (cnt, len(t_array))
    )

    return t_array


def combine_save_timeseries(setup, t_array_calculated=None):
    """
    load single time steps, combine them, and save again.
    """

    with open("settings.yml") as file:
        dict_options = yaml.safe_load(file)
    globals().update(dict_options[setup])

    # array that stores considered time steps
    # from file if not given explicitly
    if t_array_calculated is None:
        ds_t = xr.open_dataset(datafolder + "timeseries_labels.nc")
        t_array_calculated = ds_t.time.where(ds_t.label_calculated, drop=True).values

    # flag whether all time steps are available
    incomplete = False
    ds_list = []
    for t in tqdm(t_array_calculated):
        try:
            ds_ = xr.open_dataset(datafolder + rmfolder + "parameters_%s.nc" % t)
        except:
            incomplete = True
        else:
            ds_list.append(ds_)

    # print info on completeness
    if incomplete:
        print("Not for all timesteps is a file available.")
    else:
        print("File available for all timesteps.")

    # concatenate
    ds_rm = xr.concat(ds_list, dim="time")

    # save specifying chunks
    ds_rm.to_netcdf(
        datafolder + "temp/parameters_%i_%s.nc" % (nfields, year),
        encoding={
            "final_fields": {
                "zlib": True,
                "dtype": "uint16",
                "scale_factor": 0.01,
                "_FillValue": -9999,
                "chunksizes": [1, 1, len(ds_rm.y), len(ds_rm.x)],
            },
        },
    )


def run_timestep(
    ds_cml_at_t,
    ds_rg_at_t,
    ygrid,
    xgrid,
    nfields=20,
    lin_constraints=True,
    nonlin_constraints=True,
    outfolder="",
    outname=6666,
):
    """
    Wrapper for running Random Mixing for a single time step.

    tstep - datetime (string)
    nfields - number of realizations (int)
    lin_constraints - consider linear constraints in RM?
    nonlin_constraints - consider nonlinear constraints in RM?
    """

    # # check wheter time step has already been calculated
    # if os.path.isfile(
    #     outfolder + "parameters_%s.nc" % ds_cml_at_t.time.values):
    #     return

    # start time
    t0_tstep = datetime.datetime.now()
    mem0_tstep = psutil.Process(os.getpid()).memory_info().rss

    # use random seed if you want to ensure reproducibility
    # np.random.seed(121)

    # store domainsize
    domainsize = (len(ygrid), len(xgrid))

    # ------------------------------
    # extract data

    # linear constraints: filter nans, multiply by 10 (later reversed)
    lin_yx = np.vstack((ds_rg_at_t.y.values, ds_rg_at_t.x.values)).T.astype(int)
    lin_prec = ds_rg_at_t.rain.values
    lin_yx, lin_prec = rd.filter_nans_a(lin_yx, lin_prec)
    lin_prec *= 10.0

    # nonlinear constraints
    cml_yx = np.vstack(
        (
            ds_cml_at_t.y_a.values,
            ds_cml_at_t.x_a.values,
            ds_cml_at_t.y_b.values,
            ds_cml_at_t.x_b.values,
        )
    ).T.astype(int)
    cml_prec = ds_cml_at_t.rain.values
    cml_yx, cml_prec = rd.filter_nans_a(cml_yx, cml_prec)

    # =================================================================================
    # Preparation

    # calculate marginal
    marginal = calculate_marginal(lin_prec, cml_prec)

    # fit a Gaussian copula -> spatial model
    cmod, t_copula, mem_copula = calculate_copula(
        lin_yx,
        lin_prec,
        outputfile=None,
        covmods=["Exp", "Sph"],
        ntries=6,
        nugget=0.001,
    )

    # get linear constraints
    if lin_constraints == True:
        cp, cv, lecp, lecv = get_linear_constraints(lin_yx, lin_prec, marginal)
    else:
        cp, cv, lecp, lecv = (None, None, None, None)

    # initialize CMLModel
    if nonlin_constraints == True:
        nonlin_integrals = nl_integrals(cml_yx)
        my_CMLModel = CMLModel(cml_prec, marginal, nonlin_integrals)
        optmethod = "circleopt"
    else:
        my_CMLModel = None
        optmethod = "no_nl_constraints"

    # =================================================================================
    # SIMULATION USING RMWSPy

    # initialize Random Mixing Whittaker-Shannon
    CS = RMWS(
        my_CMLModel,
        domainsize=domainsize,
        covmod=cmod,
        nFields=nfields,
        cp=cp,
        cv=cv,
        le_cp=lecp,
        le_cv=lecv,
        optmethod=optmethod,
        minObj=0.4,
        maxbadcount=20,
        maxiter=300,
        pyfftwmode=True,
        # seed=121,
        tstep=ds_cml_at_t.time.values,
        t0=t0_tstep,
    )

    # run RMWS
    CS()

    # ==================================================================================
    # POST-PROCESSING

    # backtransform simulated fields to original data space
    inner_fields = backtransform(CS.innerFields, marginal)
    final_fields = backtransform(CS.finalFields, marginal)

    # record time needed
    t_tstep = (datetime.datetime.now() - t0_tstep).total_seconds()
    mem_tstep = psutil.Process(os.getpid()).memory_info().rss - mem0_tstep
    print("time needed:", t_tstep)

    # =================================================================================
    # NETCDF OUTPUT

    # reduce memory usage when saved as netcdf
    encoding = {
        "final_fields": {
            "zlib": True,
            "dtype": "uint16",
            "scale_factor": 0.01,
            "_FillValue": -9999,
        }
    }

    ds_parameters = xr.Dataset(
        data_vars={
            "time": ds_cml_at_t.time,
            "copula": cmod,
            "t_copula": t_copula,
            "mem_copula": mem_copula,
            "t_uncon": CS.t_uncon,
            "mem_uncon": CS.mem_uncon,
            "t_MHRW": CS.t_MHRW,
            "mem_MHRW": CS.mem_MHRW,
            "t_final": CS.t_final,
            "mem_final": CS.mem_final,
            "t_total": t_tstep,
            "mem_total": mem_tstep,
            # "obj_fin": (["nfields"], CS.objfin_arr),
            # "final_fields": (["nfields", "y", "x"], final_fields),
        },
        coords={
            # "y": (["y"], ygrid),
            # "x": (["x"], xgrid),
            # "nfields": (["nfields"], np.arange(nfields)),
        },
    )

    ds_parameters.to_netcdf(
        outfolder + "parameters_%s_%s.nc" % (ds_cml_at_t.time.values, outname),
        encoding=encoding,
    )

    print("END")


def filter_save_timesteps(label, t_array, nfields, ygrid, xgrid, outfolder, save=False):

    cnt = 0
    ind_del = []
    for i, t in enumerate(t_array):

        if ~label.sel(time=t):

            cnt += 1
            ind_del.append(i)

            if save:

                # reduce memory usage when saved as netcdf
                encoding = {
                    "final_fields": {
                        "zlib": True,
                        "dtype": "uint16",
                        "scale_factor": 0.01,
                        "_FillValue": -9999,
                    }
                }

                # save nan dataset
                ds_parameters = xr.Dataset(
                    data_vars={
                        "time": t,
                        "copula": str(np.nan),
                        "time_tstep": np.nan,
                        "obj_fin": (["nfields"], np.ones((nfields)) * np.nan),
                        "final_fields": (
                            ["nfields", "y", "x"],
                            np.ones((nfields, len(ygrid), len(xgrid))) * np.nan,
                        ),
                    },
                    coords={
                        "y": (["y"], ygrid),
                        "x": (["x"], xgrid),
                        "nfields": (["nfields"], np.arange(nfields)),
                    },
                )
                ds_parameters.to_netcdf(
                    path=outfolder + "parameters_%s.nc" % t, encoding=encoding
                )

    t_array = np.delete(t_array, ind_del)
    print(
        "%i time steps were filtered based on %s. %i time steps remain."
        % (cnt, label.name, len(t_array))
    )

    return t_array


def nl_integrals(cml_yx):
    """
    Define line integrals between the two coordinates of cml using Bresenham's Line Algorithm
    """

    yxyx = np.rint(cml_yx).astype(int)

    nl_integrals = []
    for integ in range(yxyx.shape[0]):
        nl_integrals.append(np.array(bresenhamline.get_line(yxyx[integ, :2], yxyx[integ, 2:])))
    return nl_integrals


def calculate_marginal(prec, cml_prec=None):
    """
    Calculate marginal distribution either by rain observed by gauges (prec) only,
    or including the high CML values also if they exceed the gauge values.
    """

    if cml_prec is not None:
        # add the cml_prec that are higher than prec
        hcmlp = np.copy(cml_prec) * 10.0
        hcmlp = hcmlp[hcmlp > prec.max()]
        prec = np.concatenate((prec, hcmlp))

    print("max prec: ", np.max(prec) / 10)

    # fit a non-parametric marginal distribution using KDE with Gaussian kernel
    # this assumes that there are wet observations
    p0 = 1.0 - np.float(prec[prec > 0].shape[0]) / prec.shape[0]
    print("p0 = ", p0)

    if len(prec[prec > 0]) < 5:
        cv = 2
    else:
        cv = 5

    # optimize the kernelwidth
    prec_wet = np.log(prec[prec > 0])
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=cv
    )
    grid.fit(prec_wet[:, None])

    # use optimized kernel for kde
    kde = KernelDensity(bandwidth=grid.best_params_["bandwidth"], kernel="gaussian")
    kde.fit(prec_wet[:, None])

    # build cdf and invcdf from pdf
    xx = np.arange(prec_wet.min() - 1.0, prec_wet.max() + 1.0, 0.001)
    logprob = np.exp(kde.score_samples(xx[:, None]))
    cdf_ = np.cumsum(logprob) * 0.001
    cdf_ = np.concatenate(([0.0], cdf_))
    cdf_ = np.concatenate((cdf_, [1.0]))
    xx = np.concatenate((xx, [prec_wet.max() + 1.0]))
    xx = np.concatenate(([prec_wet.min() - 1.0], xx))
    cdf = interpolate.interp1d(xx, cdf_, bounds_error=False)
    invcdf = interpolate.interp1d(cdf_, xx)

    # marginal distribution variables
    marginal = {}
    marginal["p0"] = p0
    marginal["cdf"] = cdf
    marginal["invcdf"] = invcdf

    return marginal


def calculate_copula(
    yx, prec, outputfile=None, covmods=["Mat", "Exp", "Sph"], ntries=6, nugget=0.05
):
    """
    Wrapper function for copula / spatial dependence calculation
    """

    print("\nCalculating Copula...")
    t0_copula = datetime.datetime.now()
    mem0_copula = psutil.Process(os.getpid()).memory_info().rss

    # transform to rank values
    u = (st.rankdata(prec) - 0.5) / prec.shape[0]

    # set subset size
    if len(prec[prec > 0]) < 5:
        n_in_subset = 2
    else:
        n_in_subset = 5

    # calculate copula models
    cmods = sparest.paraest_multiple_tries(
        np.copy(yx),
        u,
        ntries=[ntries, ntries],
        n_in_subset=n_in_subset,
        # number of values in subsets
        neighbourhood="nearest",
        # subset search algorithm
        covmods=covmods,  # covariance functions
        outputfile=outputfile,
    )  # store all fitted models in an output file

    # take the copula model with the highest likelihood
    # reconstruct from parameter array
    likelihood = -666
    for model in range(len(cmods)):
        for tries in range(ntries):
            if cmods[model][tries][1] * -1.0 > likelihood:
                likelihood = cmods[model][tries][1] * -1.0
                #                 cmod = "0.05 Nug(0.0) + 0.95 %s(%1.3f)" % (
                #                     covmods[model], cmods[model][tries][0][0])
                cmod = "%1.3f Nug(0.0) + %1.3f %s(%1.3f)" % (
                    nugget,
                    1 - nugget,
                    covmods[model],
                    cmods[model][tries][0][0],
                )
                if covmods[model] == "Mat":
                    cmod += "^%1.3f" % (cmods[model][tries][0][1])

    print(cmod)

    t1_copula = datetime.datetime.now()
    t_copula = t1_copula - t0_copula
    t_copula = t_copula.total_seconds()
    mem_copula = psutil.Process(os.getpid()).memory_info().rss - mem0_copula
    print("...done in ", t_copula, "\n")

    return cmod, t_copula, mem_copula


def get_linear_constraints(yx, prec, marginal):
    """
    Transform observations to standard normal using the fitted cdf;
    """

    # zero (dry) observations
    mp0 = prec == 0.0
    lecp = yx[mp0]
    lecv = np.ones(lecp.shape[0]) * st.norm.ppf(marginal["p0"])

    # wet observations
    yx_wet = yx[~mp0]
    prec_wet = np.log(prec[~mp0])

    # non-zero (wet) observations
    cp = yx_wet
    cv = st.norm.ppf(
        (1.0 - marginal["p0"]) * marginal["cdf"](prec_wet) + marginal["p0"]
    )

    # delete NaNs that appear when values are out of the interpolation range of cdf
    ind_nan = np.where(np.isnan(cv))
    cp = np.delete(cp, ind_nan, axis=0)
    cv = np.delete(cv, ind_nan, axis=0)

    #     lin_data = np.delete(lin_data, ind_nan, axis=0)

    return cp, cv, lecp, lecv


def backtransform(data, marginal):
    """
    Transform from standard normal to acutal value space.
    """

    rain = st.norm.cdf(data)
    mp0 = rain <= marginal["p0"]
    rain[mp0] = 0.0
    rain[~mp0] = (rain[~mp0] - marginal["p0"]) / (1.0 - marginal["p0"])
    rain[~mp0] = marginal["invcdf"](rain[~mp0])
    rain[~mp0] = np.exp(rain[~mp0]) / 10.0

    return rain


def get_copula_params(ds):
    """
    Copula parameters
    """

    # range and model
    cov_rng = []
    cov_mod = []

    for i in range(len(ds.time)):
        name_cop = (ds.copula.values)[i]

        if name_cop == "nan":
            cov_rng.append(np.nan)
            cov_mod.append("nan")

        else:
            name_cop = name_cop.split("+")[1].strip().split(" ")[1]
            cov_rng.append(float(name_cop[4:-1]))
            cov_mod.append(name_cop[:3])

    return cov_rng, cov_mod
