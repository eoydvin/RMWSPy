import numpy as np
import xarray as xr
import pandas as pd
import scipy.spatial as sp
import scipy.interpolate as interpolate
import datetime
from tqdm import tqdm


def center_along_link(ds_cml, grid="RAD"):
    """
    Calculate the central point along CML paths.
    Either on latlon or RADOLAN (yx) grid
    """

    if grid == "RAD":
        ds_cml["y"] = (ds_cml.y_a + ds_cml.y_b) / 2
        ds_cml["x"] = (ds_cml.x_a + ds_cml.x_b) / 2

    if grid == "latlon":
        ds_cml["lat"] = (ds_cml.site_a_latitude + ds_cml.site_b_latitude) / 2
        ds_cml["lon"] = (ds_cml.site_a_longitude + ds_cml.site_b_longitude) / 2

    return ds_cml


def extract_virtual_gauges(ds_cml, linperc=100, exclusive=False):
    """
    If only non-linear (CML) data is used, extract virtual gauges.

    Parameters:

    linperc
        Percentage of data that will be transformed to virtual gauges
    exclusive
        If true the data that is used as virtual gauges
        is not considered as non-linear data.
    """

    # index to get at random entries, use 'linperc' % of entries
    L = len(ds_cml.obs_id)
    idx = np.arange(L)
    np.random.shuffle(idx)
    idx = idx[: int(L * (linperc / 100))]
    ids = ds_cml.obs_id.isel(obs_id=idx)

    # separate data
    extracted_data = ds_cml.where(ds_cml.obs_id in ids)

    if exclusive:
        ds_cml = ds_cml.where(ds_cml.obs_id not in ids, drop=True)

    # get central point on link
    lat_center = (extracted_data.site_a_latitude + extracted_data.site_b_latitude) / 2
    lon_center = (extracted_data.site_a_longitude + extracted_data.site_b_longitude) / 2

    # combine to dataset of virtual gauges
    ds_vg = xr.Dataset(
        {"rain": (["obs_id", "time"], extracted_data.rain)},
        coords={
            "lat": (["obs_id"], lat_center),
            "lon": (["obs_id"], lon_center),
            "time": (["time"], extracted_data.time),
        },
    )

    return ds_vg, ds_cml


def projection(ds, datatype, projtype="RAD"):
    """
    Projection from lat, lon to y, x grid.
    """

    if datatype == "LIN":
        if projtype == "RAD":
            y, x = projRADOLAN(ds.lat.values, ds.lon.values)
        #         elif projtype == 'PRG':
        #             y, x = projPRG(ds.lat, ds.lon)
        ds["y"] = (["obs_id"], y)
        ds["x"] = (["obs_id"], x)

    elif datatype == "CML":
        if projtype == "RAD":
            ya, xa = projRADOLAN(ds.site_a_latitude.values, ds.site_a_longitude.values)
            yb, xb = projRADOLAN(ds.site_b_latitude.values, ds.site_b_longitude.values)
        #         elif projtype == 'PRG'
        ds["y_a"] = (["obs_id"], ya)
        ds["x_a"] = (["obs_id"], xa)
        ds["y_b"] = (["obs_id"], yb)
        ds["x_b"] = (["obs_id"], xb)

    return ds


def projRADOLAN(lat=None, lon=None, y=None, x=None):
    """
    Projection on quasi-RADOLAN grid ([0,899],[0,899] instead of actual RADOLAN coordinates).
    Calculates either from lat,lon to y,x or inverse depending on what is given.
    """

    # constants of RADOLAN projection
    radius_earth = 6370.04
    lon0 = 10 * (np.pi / 180)
    lat0 = 60 * (np.pi / 180)
    xstart = -523.4621669218558
    xend = 375.5378330781442
    ystart = -4658.644724265572
    yend = -3759.644724265572

    if lat is not None and lon is not None:
        # data coordinates in radiant
        lat = lat * (np.pi / 180)
        lon = lon * (np.pi / 180)

        # exact y,x
        y_exact = (
            -radius_earth
            * ((1 + np.sin(lat0)) / (1 + np.sin(lat)))
            * np.cos(lat)
            * np.cos(lon - lon0)
        )
        x_exact = (
            radius_earth
            * ((1 + np.sin(lat0)) / (1 + np.sin(lat)))
            * np.cos(lat)
            * np.sin(lon - lon0)
        )

        # transform [0,899] grid
        y = y_exact - ystart
        x = x_exact - xstart

        return y, x

    elif y is not None and x is not None:

        # shift
        y_ = y + ystart
        x_ = x + xstart

        # calculation
        fact_1 = radius_earth**2 * (1 + np.sin(lat0)) ** 2
        fact_2 = x_**2 + y_**2
        lat = np.arcsin((fact_1 - fact_2) / (fact_1 + fact_2))
        lon = np.arctan(-x_ / y_) + lon0

        # in degree
        lat = lat / (np.pi / 180)
        lon = lon / (np.pi / 180)

        return lat, lon


def get_data_range(lat_list, lon_list):
    """
    Returns lat lon extent of data
    """

    lat_all, lon_all = [], []
    for lat_array, lon_array in zip(lat_list, lon_list):
        lat_all.extend(lat_array)
        lon_all.extend(lon_array)
    rng = np.array([np.min(lat_all), np.min(lon_all), np.max(lat_all), np.max(lon_all)])

    return rng


def projDataExtent(lat, lon, rng, gridsize=(100, 100)):
    """
    Project adjusted to data extent.
    """

    yinc = (rng[2] - rng[0]) / (gridsize[0] - 1)
    xinc = (rng[3] - rng[1]) / (gridsize[1] - 1)

    y = (lat - rng[0]) / yinc
    x = (lon - rng[1]) / xinc

    return y, x


def take_subset_obs(ds, box, datatype):
    """
    Take only a spatial subset of the data.

    box: bounding box of subset [y_min, x_min, y_max, x_max]
    """

    if datatype == "LIN":
        ds = ds.where(ds.y >= box[0], drop=True)
        ds = ds.where(ds.y < box[2], drop=True)
        ds = ds.where(ds.x >= box[1], drop=True)
        ds = ds.where(ds.x < box[3], drop=True)
    elif datatype == "CML":
        ds = ds.where(ds.y_a >= box[0], drop=True)
        ds = ds.where(ds.y_a < box[2], drop=True)
        ds = ds.where(ds.x_a >= box[1], drop=True)
        ds = ds.where(ds.x_a < box[3], drop=True)
        ds = ds.where(ds.y_b >= box[0], drop=True)
        ds = ds.where(ds.y_b < box[2], drop=True)
        ds = ds.where(ds.x_b >= box[1], drop=True)
        ds = ds.where(ds.x_b < box[3], drop=True)
    return ds


def shift_indices(ds, box, datatype, mode="reduce"):
    """
    Shift indices if a spatial subset is used. Necessary, because
    coordinates should be in [0, N] for RM calculation, independent
    of the actual coordinates.
    mode: 'reduce' | 'expand'
        reduce - reduces indices from RADOLAN grid to box size
        expand - brings indices back to RADOLAN grid
    """

    if mode == "expand":
        box = np.array(box) * -1

    if datatype == "LIN":
        ds["y"] = ds.y - box[0]
        ds["x"] = ds.x - box[1]

    if datatype == "CML":
        ds["y_a"] = ds.y_a - box[0]
        ds["x_a"] = ds.x_a - box[1]
        ds["y_b"] = ds.y_b - box[0]
        ds["x_b"] = ds.x_b - box[1]

    return ds


def label_combined_duplicates(ds_a, ds_b=None):
    """
    Label observations if the have coordinates that are occupied
    by another observation that comes first in listing.
    """

    if ds_b is None:
        # in case only one dataset is given
        label = check_for_duplicates(ds_a.y.values, ds_a.x.values)
        ds_a["label_dupl"] = (["obs_id"], label)
        return ds_a

    else:
        # otherwise combine and then look for duplicates
        y = np.concatenate((ds_a.y.values, ds_b.y.values))
        x = np.concatenate((ds_a.x.values, ds_b.x.values))

        label = check_for_duplicates(y, x)

        ds_a["label_dupl"] = (["obs_id"], label[: len(ds_a.obs_id)])
        ds_b["label_dupl"] = (["obs_id"], label[len(ds_a.obs_id) :])

        return ds_a, ds_b


def check_for_duplicates(y, x):
    """
    If location is occupied more than once all but first get label False.
    """

    yx_list = []

    label = np.ones(len(y), dtype=bool)

    for i, yx in enumerate(zip(y, x)):

        yx_tuple = tuple(yx)

        # test whether location has already been looked at
        if yx_tuple in yx_list:
            label[i] = False

        # store tuples of locations that have already been looked at
        yx_list.append(yx_tuple)

    return label


def label_outliers(ds, radius=15, min_neighbors=5, tol_wet_neighbors=0):

    """
    Delete observations if they do not fit the ones in their surroundings.

    Parameters:

    ds: xarray dataset
        with rainfall information for dimensions `time` and observation id (`obs_id`),
        coordinates `y` and `x` (for CMLs they refer to the midpoint along the path)
    radius: float
        radius in which neighbors are considered
    min_neighbors: int
        only filter if number of neighbors in radius is not below min_neighbors
    tol_wet: int,
        number of non-zero neighbors tolerated, i.e. if this number is not
        exceeded, the filter applies.

    Returns:

    ds: xarray dataset
        input dataset, but with flag indicating if sanity check was passed.
    """

    # extract data to use numpy based function
    y_coords = ds.y.values
    x_coords = ds.x.values

    # array of locations, shape: (no. obs, 2)
    locs = np.vstack((y_coords, x_coords)).T

    assert (
        locs.shape[1] == 2
    ), "Both y- and x-coordinates must be one-dimensional. Check if coordinates depend on time!"

    # mutual distances
    dist = sp.distance_matrix(locs, locs)

    # store flags in array
    isSane = np.ones((len(ds.obs_id), len(ds.time)), dtype=bool)

    # loop over time
    for i_t in tqdm(range(len(ds.time))):

        # extract rain at time
        rain = ds.isel(time=i_t).rain.values

        # loop over observations
        for i_obs in range(len(rain)):

            # label missing data and zeros as if they passed spatial sanity check
            if (rain[i_obs] == np.nan) or (rain[i_obs] == 0):
                continue

            # extract ids of those observations that are close
            neighbor_ids = np.argwhere(dist[i_obs, :] < radius)[:, 0]

            # do not consider obs. in question as a neighbor of itself
            neighbor_ids = neighbor_ids[neighbor_ids != i_obs]

            # rain recorded by neighbors
            neighbor_rain = rain[neighbor_ids]

            # do not consider neighbors with missing data
            neighbor_rain = neighbor_rain[~np.isnan(neighbor_rain)]

            # make sure enough neighbors are considered
            if len(neighbor_rain) < min_neighbors:
                continue

            # apply filter if neighbors are (mostly) dry
            if len(neighbor_rain[neighbor_rain > 0]) <= tol_wet_neighbors:
                isSane[i_obs, i_t] = False

    # add flags to dataset
    ds["label_sp_sanity"] = (["obs_id", "time"], isSane)

    return ds


def label_outliers_old(ds, radius=20, min_neighbors=10, multi_tol=5, add_tol=0.00):

    """
    Delete observations if they do not fit the ones in their surroundings.

    Parameters:

    ds: xarray dataset
        with rainfall information for dimensions `time` and observation id (`obs_id`),
        coordinates `y` and `x` (for CMLs they refer to the midpoint along the path)
    radius: float
        radius in which neighbors are considered
    min_neighbors: int
        only filter if number of neighbors in radius is not below min_neighbors
    tol: float,
        tolerance, number of standard deviations the observation can deviate
        from the mean of the neighbors

    Returns:

    ds: xarray dataset
        input dataset, but with flag indicating if sanity check was passed.
    """

    # extract data to use numpy based function
    y_coords = ds.y.values
    x_coords = ds.x.values

    # array of locations, shape: (no. obs, 2)
    locs = np.vstack((y_coords, x_coords)).T

    assert (
        locs.shape[1] == 2
    ), "Both y- and x-coordinates must be one-dimensional. Check if coordinates depend on time!"

    # mutual distances
    dist = sp.distance_matrix(locs, locs)

    # store flags in array
    isSane = np.ones((len(ds.obs_id), len(ds.time)), dtype=bool)

    # loop over time
    for i_t in tqdm(range(len(ds.time))):

        # extract rain at time
        rain = ds.isel(time=i_t).rain.values

        # loop over observations
        for i_obs in range(len(rain)):

            # label missing data as if they passed spatial sanity check
            if rain[i_obs] == np.nan:
                continue

            # extract ids of those observations that are close
            neighbor_ids = np.argwhere(dist[i_obs, :] < radius)[:, 0]

            # do not consider obs. in question as a neighbor of itself
            neighbor_ids = neighbor_ids[neighbor_ids != i_obs]

            # rain recorded by neighbors
            neighbor_rain = rain[neighbor_ids]

            # do not consider neighbors with missing data
            neighbor_rain = neighbor_rain[~np.isnan(neighbor_rain)]

            # make sure enough neighbors are considered
            if len(neighbor_rain) < min_neighbors:
                continue

            # calculate upper and lower threshold with respect to mean and std of neighbors
            tol = multi_tol * np.std(neighbor_rain) + add_tol
            thld_up = np.mean(neighbor_rain) + tol
            thld_low = np.mean(neighbor_rain) - tol

            # if observation in question is outside the two thresholds it is not sane
            if rain[i_obs] > thld_up or rain[i_obs] < thld_low:

                isSane[i_obs, i_t] = False

    # add flags to dataset
    ds["label_sp_sanity"] = (["obs_id", "time"], isSane)

    return ds


def label_outliers_noTime(ds, radius=30, min_neighbors=5, tol=3):

    """
    Delete observations if they do not fit the ones in their surroundings (to be called for data of one time step).

    Options:
        - 'radius' in which neighbors are considered
        - 'min_neighbors': only filter if number of neighbors in radius is not below min_neighbors
        - 'tol': tolerance, number of standard deviations the observation can deviate from the mean of the neighbors
    """

    # extract data to use numpy based function
    y_coords = ds.y.values
    x_coords = ds.x.values

    # array of locations, shape: (no. obs, 2)
    locs = np.vstack((y_coords, x_coords)).T

    # mutual distances
    dist = sp.distance_matrix(locs, locs)

    # store flags in array
    isSane = np.ones(len(ds.obs_id), dtype=bool)

    # extract rain at time
    rain = ds.rain.values

    # loop over observations
    for i_obs in range(len(rain)):

        # label missing data as if they passed spatial sanity check
        if rain[i_obs] == np.nan:
            continue

        # extract ids of those observations that are close
        neighbor_ids = np.argwhere(dist[i_obs] < radius)[:, 0]

        # do not consider obs. in question as a neighbor of itself
        neighbor_ids = neighbor_ids[neighbor_ids != i_obs]

        # rain recorded by neighbors
        neighbor_rain = rain[neighbor_ids]

        # do not consider neighbors with missing data
        neighbor_rain = neighbor_rain[~np.isnan(neighbor_rain)]

        # make sure enough neighbors are considered
        if len(neighbor_rain) < min_neighbors:
            continue

        # calculate upper and lower threshold with respect to mean and std of neighbors
        thld_up = np.mean(neighbor_rain) + tol * np.std(neighbor_rain)
        thld_low = np.mean(neighbor_rain) - tol * np.std(neighbor_rain)

        # if observation in question is outside the two thresholds it is not sane
        if rain[i_obs] > thld_up or rain[i_obs] < thld_low:

            isSane[i_obs] = False

    # add flags to dataset
    ds["label_sp_sanity"] = ("obs_id", isSane)

    return ds


def filter_nans_a(yx, rain):

    yx = yx[~np.isnan(rain)]
    rain = rain[~np.isnan(rain)]

    return yx, rain


def filter_nans_b(y, x, rain):

    y = y[~np.isnan(rain)]
    x = x[~np.isnan(rain)]
    rain = rain[~np.isnan(rain)]

    return y, x, rain


def give_info_on_labels(ds_list, label_list):

    for i, ds in enumerate(ds_list):

        print("\n%i. dataset:" % i)

        for j, label in enumerate(label_list):

            print(
                "%.2f percent of observations are negatively labeled according to %s"
                % (
                    (1 - (ds[label].sum() / (len(ds.obs_id) * len(ds.time))).values)
                    * 100,
                    label,
                )
            )


# ===============================================================
# helper functions for analysis


def load_field_at_time(fn, t_array):
    return xr.open_dataset(fn).sel(time=t_array)


def take_subset_field(ds, box):
    return ds.sel(x=slice(box[1], box[3]), y=slice(box[0], box[2]))


def shift_coords_to_origin(ds):
    ds["x"] = ds.x - ds.x.min()
    ds["y"] = ds.y - ds.y.min()
    return ds


def coords_as_int(ds):
    ds["x"] = ds.x.astype(int)
    ds["y"] = ds.y.astype(int)
    return ds


def get_rain_dataarray(ds, var_name, chunk=False):
    if chunk:
        da = ds[var_name].chunk({"time": 1})
    else:
        da = ds[var_name]
    return da


def find_periods(t_array, t_selected):
    """
    Find out boundaries of certain (Nan, dry, etc.) periods.
    """

    # list of start dates and list of end dates
    beginPeriod = []
    endPeriod = []

    # append beginnings and ends of periods
    for t in t_array[:-1]:
        tNext = t + pd.Timedelta("1h")
        if t not in t_selected and tNext in t_selected:
            beginPeriod.append(t)
        if t in t_selected and tNext not in t_selected:
            endPeriod.append(t)

    # set edges of array if needed
    if t_array[0] in t_selected:
        beginPeriod.insert(0, np.datetime64(t_array[0] - pd.Timedelta("1h")))
    if t_array[-1] in t_selected:
        endPeriod.append(t_array[-1])

    return beginPeriod, endPeriod


def find_periods_old(t_array, t_selected):
    """
    Find out boundaries of certain (Nan, dry, etc.) periods.
    """

    # list of start dates and list of end dates
    beginPeriod = []
    endPeriod = []

    # append beginnings and ends of periods
    for t in t_array[:-1]:
        tNext = t + pd.Timedelta("1h")
        if t not in t_selected and tNext in t_selected:
            beginPeriod.append(t)
        if t in t_selected and tNext not in t_selected:
            endPeriod.append(t)

    # set edges of array if needed
    if t_array[0] in t_selected:
        beginPeriod.insert(0, np.datetime64(t_array[0] - pd.Timedelta("1h")))
    if t_array[-1] in t_selected:
        endPeriod.append(t_array[-1])

    return beginPeriod, endPeriod


def reduce_time_and_space(ds, t, ymin, xmin, ymax, xmax, datatype="CML"):
    """
    Take a temporal and spatial subset of observational data.
    """

    # selection by time
    ds_t = ds.sel(time=t)

    # selection by space
    if datatype == "CML":
        yaCond = (ds_t.y_a > ymin) & (ds_t.y_a <= ymax)
        xaCond = (ds_t.x_a > xmin) & (ds_t.x_a <= xmax)
        ybCond = (ds_t.y_b > ymin) & (ds_t.y_b <= ymax)
        xbCond = (ds_t.x_b > xmin) & (ds_t.x_b <= xmax)
        yCond = yaCond & ybCond
        xCond = xaCond & xbCond
        ds_t_subset = ds_t.where(yCond & xCond, drop=True)

    elif datapye == "LIN":
        yCond = (ds_t.y > ymin) & (ds_t.y <= ymax)
        xCond = (ds_t.x > xmin) & (ds_t.x <= xmax)
        ds_t_subset = ds_t.where(yCond & xCond, drop=True)

    # bring label back to boolean type
    ds_t_subset["label_sp_sanity"] = ds_t_subset.label_sp_sanity.astype(bool)

    return ds_t_subset


def mask_fields(infields, masks):
    """
    Apply masks on fields.

    infields: list of fields (dataarrays)
    masks: list of masks (dataarrays)
    """

    outfields = []

    for field in infields:
        for mask in masks:
            field = field.where(mask)
        outfields.append(field)
    outfields = tuple(outfields)

    return outfields


def get_wetness_indicator_timeseries(
    da, option="quantile", q=0.95, norm=False, only_wet=False
):
    """
    Gives back a time series (xarray dataarray) that is an
    indication of how wet each time step is.
    Nans are not considered.

    If option 'quantile', q can be defined [0, 1].
    norm: Output will be normalized to [0, 1].
    only_wet: Only wet pixels will be considered.

    """

    space = ["y", "x"]

    if only_wet:
        da = da.where(da > 0)

    if option == "quantile":
        wetness = da.quantile(q, dim=space)
    elif option == "mean":
        wetness = da.mean(space)
    elif option == "percWet":
        wetness = (da > 0).sum(space) / ((da == 0).sum(space) + (da > 0).sum(space))

    if norm:
        wetness = wetness * (1 / wetness.max())

    return wetness


def set_nan_to_zero(da, da_tmask):
    """
    Set to zero instead of nan for SAL. Only not available time steps (too dry)
    shall still be nan and produce nans when calculating SAL.
    """

    # first set all nans to zero
    da0 = da.where(~np.isnan(da), 0)

    # second set dry time steps back to nan
    da0 = da0.where(da_tmask)

    return da0


# ===============================================================
# DEPRECATED


def label_outliers_outer(ds, radius=30, min_neighbors=5, tol=3):
    """
    Wrapper for spatial sanity check over dimension time
    """

    # store flags in array
    isSane = np.ones((len(ds.obs_id), len(ds.time)), dtype=bool)

    # loop over time
    for i_t in tqdm(range(len(ds.time))):

        # inner function
        isSane[:, i_t] = label_outliers(
            ds.y.values, ds.x.values, ds.rain.isel(time=i_t).values
        )

    # add flags to dataset
    ds["spatial_sanity"] = (["obs_id", "time"], isSane)

    return ds


def distinct_ids_for_duplicates(ds, y="x", x="x"):
    """
    Label entries by distinct locations on grid - not just True or False
    """

    # label_list stores numeric labels so that all similar positions have the same and all distinc positions distinct labels
    yx_list, label_list = [], []
    for yx in zip(ds[y].values, ds[x].values):

        yx_tuple = tuple(yx)

        # test whether location has already looked at
        cond = yx_tuple in yx_list

        # give label accordingly
        if len(yx_list) == 0:
            label = 0
        elif cond:
            label = label_list[yx_list.index(yx_tuple)]
        else:
            label = np.max(label_list) + 1

        # store tuples of locations that have already been looked at
        yx_list.append(yx_tuple)

        # store labels
        label_list.append(label)

    # add information to dataset
    ds["obs_loc_id"] = (["obs_id"], label_list)

    return ds


# ================================
# for synthetic input


def get_synt_lin_data(mode="gauss", n_obs=400, rng=(0, 30, 0, 30)):
    if mode == "2017_paper":

        p_data = np.genfromtxt(
            os.path.join(datafolder, "syn_obs_RG_2015_08.csv"), delimiter="\t"
        )

    elif mode == "minimal":

        lat = [0, 3, 2]
        lon = [0, 1, 3]
        p_data = [2, 5, 0]

        p_data = np.vstack((lat, lon, p_data)).T

    elif mode == "gauss":

        # 2D GAUSSIAN DISTRIBUTION
        lat = np.linspace(1, 29, int(n_obs**0.5))
        lon = np.linspace(1, 29, int(n_obs**0.5))

        # randomly offset position of observations
        lat = lat + np.random.uniform(-0.5, 0.5, len(lat))
        lon = lat + np.random.uniform(-0.5, 0.5, len(lon))

        sig = 1
        scl = 100  # scale factor

        # several gauss shapes
        mu = np.array([[5, 5], [15, 15], [25, 25]])
        lat_gauss = np.zeros(len(lat))
        lon_gauss = np.zeros(len(lon))
        for k in range(np.shape(mu)[0]):  # k: number of gauss shapes per dimension
            lat_gauss += (1 / (sig * (2 * np.pi) ** (1 / 2))) * np.exp(
                (-1 / 2) * ((lat - mu[k, 0]) / sig) ** 2
            )
            lon_gauss += (1 / (sig * (2 * np.pi) ** (1 / 2))) * np.exp(
                (-1 / 2) * ((lon - mu[k, 1]) / sig) ** 2
            )

        # # single gauss shape
        # mu = np.array([np.mean(lat), np.mean(lon)])
        # lat_gauss = (1/(sig*(2*np.pi)**(1/2)))*np.exp((-1/2)*((lat-mu[0])/sig)**2)
        # lon_gauss = (1/(sig*(2*np.pi)**(1/2)))*np.exp((-1/2)*((lon-mu[1])/sig)**2)

        R = np.ones((len(lat), len(lon)))
        for i in range(len(lat)):
            for j in range(len(lon)):
                R[i, j] = lat_gauss[i] * lon_gauss[j] * scl

        lat, lon = np.meshgrid(lat, lon)

        lat = np.ndarray.flatten(
            lat, order="F"
        )  # may change order to default for different gauss patterns
        lon = np.ndarray.flatten(lon, order="F")
        p_data = np.ndarray.flatten(R)
        p_data[p_data < 0.1] = 0.0

        p_data = np.vstack((lat, lon, p_data)).T

    return p_data


def get_synt_nl_data(mode=None, n_obs=400, rng=(0, 30, 0, 30)):

    if mode == "paper_2017":

        nl_data = np.genfromtxt(
            os.path.join(datafolder, "syn_obs_CML_2015_08.csv"), delimiter="\t"
        )

    elif mode == "minimal":

        lat_a = [1, 2]
        lon_a = [0, 1]
        lat_b = [3, 0]
        lon_b = [0, 3]
        values = [3, 1]

        nl_data = np.vstack((lat_a, lon_a, lat_b, lon_b, values)).T

    else:

        ### SIMPLE EXAMPLE DATA NON-LINEAR CONSTRAINTS
        a = rng[0]
        b = rng[1]
        c = rng[2]
        d = rng[3]
        CML_lat_lon = np.zeros([30, 4])
        CML_data = np.zeros([30, 1])
        for i in range(np.shape(CML_lat_lon)[0]):
            CML_lat_lon[i] = [
                (b - a) * np.random.rand() + a,
                (d - c) * np.random.rand() + c,
                (b - a) * np.random.rand() + a,
                (d - c) * np.random.rand() + c,
            ]
            if i % 3 == 0:
                CML_data[i, :] = 10
        nl_data = np.concatenate((CML_lat_lon, CML_data), axis=1)

    return nl_data
