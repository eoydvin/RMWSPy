import numpy as np
import scipy.spatial as sp
from collections import namedtuple
import xarray as xr


def semivariogram(
    xy, v, bin_parameter=6, mode="n_bins", max_dist=None, output="return"
):

    """
    Calculate a semivariogram.

    mode: 'n_bins': define number of bins by bin_parameter
          'width': define width of bins by bin_parameter
          'n_in_bin': define number of observation within each bin by bin_parameter
          'output': either plot (to plot directly) or return (to return 2D array)

    If n_bins or wdth is given the width of each bin is equal. The returned distance is the center of the bin.
    If n_in_bin is given the width of the bins can vary. The returned distance is the mean of the distances of the observation in the bin.
    """

    # calculate distances
    h = sp.distance_matrix(xy, xy)

    # calculate squares of value differences
    vv = np.vstack((v, np.zeros(len(v)))).T
    dv = sp.distance_matrix(vv, vv) ** 2

    # flatten arrays to 1D
    h1d = np.ndarray.flatten(h)
    dv1d = np.ndarray.flatten(dv)

    # delete where values are compared with themselves (distance = 0)
    ind_0 = h1d == 0
    h1d = np.delete(h1d, ind_0)
    dv1d = np.delete(dv1d, ind_0)

    if max_dist != None:
        ind_max = h1d > max_dist
        h1d = np.delete(h1d, ind_max)
        dv1d = np.delete(dv1d, ind_max)

    # initialize output (distances and and value differences)
    h_ = []
    dv_ = []

    if mode == "n_bins" or mode == "width":

        # define number of bins in case width of bins is given
        if mode == "n_bins":
            n_bins = bin_parameter
        elif mode == "width":
            n_bins = int(np.max(h1d) / bin_parameter)

        # append mean value within every bin (devided by two since each pair appears twice in distance matrix(?) - definition of semivariogram)
        for i in range(n_bins):

            # boundaries of bin
            h_low = (i / n_bins) * np.max(h1d)
            h_high = ((i + 1) / n_bins) * np.max(h1d)

            # indices of observations within bin
            cond1 = h1d > h_low
            cond2 = h1d <= h_high
            cond = cond1 == cond2

            # output (distance, value difference)
            if len(cond) > 0:
                h_.append((h_high + h_low) / 2)
                dv_.append(np.mean(dv1d[cond]) / 2)

    elif mode == "n_in_bin":

        # assign number of observations considered in each bin
        n_in_bin = bin_parameter

        # sort distance and value arrays
        i_sort = np.argsort(h1d)
        h1d = h1d[i_sort]
        dv1d = dv1d[i_sort]

        # array of boundariy indices of bins
        step_array = np.arange(0, len(h1d), n_in_bin)
        if step_array[-1] != len(h1d):
            step_array = np.hstack((step_array, len(h1d)))

        # step over sorted data in steps defined by n_in_bin
        for i in range(len(step_array) - 1):

            # boundary indices
            low = step_array[i]
            high = step_array[i + 1]

            # output (distance, value difference)
            h_.append(np.mean(h1d[low:high]))
            dv_.append(np.mean(dv1d[low:high]))

    # combine distance and values
    hv = np.vstack((h_, dv_)).T

    if output == "plot":
        plt.xlabel("distance in km")
        plt.ylabel("mean difference of values in mm/h")
        plt.scatter(data[:, 0], data[:, 1])
        fig.savefig("semivariogram")

    elif output == "return":
        return hv


def fraction_skill_score(reference, prediction, thld=0.9, clmns=10, rows=None):
    """
    Calculate the Fraction Skill Score.

    reference, prediction: 2D arrays
    thld: threshold for separation of pixels.
    cmlns: number of columns to consider, i.e. neighborhood size in 1D
    rows: similar as columns to define rectangular neighborhoods;
        if None, neighborhood is quadratic (clmns^2)
    """

    sizey = np.shape(reference)[0]
    sizex = np.shape(reference)[1]

    if rows == None:
        rows = clmns

    # cuts off right and lower edge from analysis if clmns and rows are not a divisor of sizey respec. sizex
    n_domainsy = int(sizey / clmns)
    n_domainsx = int(sizex / rows)

    mask_pre = np.zeros(np.shape(prediction))
    mask_ref = np.zeros(np.shape(reference))

    mask_pre[np.isnan(reference) | np.isnan(prediction)] = np.nan
    mask_ref[np.isnan(reference) | np.isnan(prediction)] = np.nan

    mask_pre[prediction > thld] = 1
    mask_ref[reference > thld] = 1

    f_ref = mask_ref[mask_ref == 1].sum() / len(mask_ref.flatten())  # [mask_ref==0])
    qual_ref = 0.5 + (f_ref / 2)

    p_ref = []
    p_pre = []
    for i in range(n_domainsy):
        for j in range(n_domainsx):
            p_ref.append(
                mask_ref[i * rows : (i + 1) * rows, j * clmns : (j + 1) * clmns].sum()
            )
            p_pre.append(
                mask_pre[i * rows : (i + 1) * rows, j * clmns : (j + 1) * clmns].sum()
            )

            if np.isnan(p_ref[-1]) or np.isnan(p_pre[-1]):
                p_ref = p_ref[:-1]
                p_pre = p_pre[:-1]

    p_ref = np.array(p_ref)
    p_pre = np.array(p_pre)

    FSS = 1 - (((p_ref - p_pre) ** 2).sum() / ((p_ref**2).sum() + (p_pre**2).sum()))

    return FSS, qual_ref


def quantile_quantile(reference, prediction, n=100, dry=True):
    """
    Calculate n equal quantiles for a reference and prediction field. If dry is True, zeros are considered, otherwise they are not.
    """

    # values in 1D
    reference = reference.flatten()
    prediction = prediction.flatten()

    # remove dry
    if dry == False:
        reference = reference[reference > 0]
        prediction = prediction[prediction > 0]

    ref_sorted = np.sort(reference)
    pred_sorted = np.sort(prediction)

    indices_ref = (np.linspace(0, len(ref_sorted) - 1, n)).astype(int)
    indices_pred = (np.linspace(0, len(pred_sorted) - 1, n)).astype(int)

    ref_values = ref_sorted[indices_ref]
    pred_values = pred_sorted[indices_pred]

    return ref_values, pred_values
