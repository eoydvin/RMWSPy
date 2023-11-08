import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import proplot as pplt
from bokeh.plotting import figure, show
from bokeh.models import (
    ColorBar,
    LinearColorMapper,
    LogColorMapper,
    LogTicker,
    ColumnDataSource,
    CustomJS,
)
from bokeh.io import output_notebook
import bokeh.palettes as bp
from bokeh.transform import linear_cmap, log_cmap
from bokeh.layouts import gridplot

import seaborn as sns


# ================================================================
# PAPER PLOTS of DOI: https://doi.org/10.1029/2022WR032563 (Combining Commercial Microwave Link and Rain Gauge Observations to Estimate Countrywide Precipitation: A Stochastic Reconstruction and Pattern Analysis Approach)


def overview_observation_map(
    ds_cml, ds_rg, box, mask, cmap_mask, border_data, fig_width=5.5, outfolder=None
):

    """
    Figure 1
    """

    # figure options that are dependent on size
    lw_border = 0.5
    c_border = "k"
    lw_cml = 0.5
    c_cml = "darkgoldenrod"  # "seagreen"
    lw_rg = 0.1
    s_rg = 3
    c_rg = "maroon"  # "firebrick"
    resolution = 400
    plt.rcParams["font.size"] = 8

    fig_height = (9 / 7) * fig_width

    fig, ax = pplt.subplots(figsize=(fig_width, fig_height))

    # mask
    ax.pcolormesh(mask, cmap=cmap_mask)

    # border
    plot_German_border(ax, border_data, c=c_border, linewidth=lw_border)

    # positions CMLs
    for lata, lona, latb, lonb in zip(
        ds_cml.y_a.values,
        ds_cml.x_a.values,
        ds_cml.y_b.values,
        ds_cml.x_b.values,
    ):

        ax.plot([lona, lonb], [lata, latb], "-", c=c_cml, lw=lw_cml, zorder=4)

    # position of rain gauges
    ax.scatter(
        ds_rg.x,
        ds_rg.y,
        c=c_rg,
        # ec='grey',
        linewidth=lw_rg,
        s=s_rg,
        label="Rain Gauges",
    )

    # for legend (involves re-ordering)
    ax.plot([], [], c=c_border, ls=":", lw=lw_border, label="German border")
    ax.plot([], [], "-", c=c_cml, lw=lw_cml, label="CMLs")
    ax.axvspan(0, 0, color=cmap_mask[0], label="Area not considered")
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 3, 0, 2]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncols=2)

    # format
    ax.format(
        title="",
        ylim=[box[0], box[2]],
        xlim=[box[1], box[3]],
        ylabel="y [km]",
        xlabel="x [km]",
        aspect="equal",
    )

    if outfolder is not None:
        fig.savefig(outfolder + "dist_mask.jpg", dpi=resolution)


def overview_tseries_plus_selected(
    rad_mean_all,
    ds_cmlTMaps,
    ds_rgTMaps,
    radTMaps,
    rmsTMaps,
    t_array,
    tMapsList,
    tLabelsFull,
    box,
    beginAvailPeriod,
    endAvailPeriod,
    beginNanPeriod,
    endNanPeriod,
    beginDryPeriod,
    endDryPeriod,
    specialDates,
    cLevels,
    cMapRain,
    cLevels0,
    cMapRain0,
    border_data,
    fig_width=5.5,
    outfolder=None,
):
    """
    Figure 2
    """

    plot_array = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 3.0, 6.0, 7.0, 10.0, 11.0],
            [4.0, 5.0, 8.0, 9.0, 12.0, 13.0],
        ]
    )

    # figure options
    hspace = np.ones(plot_array.shape[0] - 1) * 0.00
    hspace[0] = 0.2

    wspace = np.ones(plot_array.shape[1] - 1) * 0.00
    wspace[[1, 3]] = 0.1

    fontsz_small = 6  # 4
    plt.rcParams["font.size"] = 6  # 4

    mrksz_rg = 1
    lw_rg = 0.05
    lw_cml = 0.2
    lw_border = 0.2
    cbar_width = 0.03

    resolution = 400

    alpha_periods = 0.2
    textXOffset = pd.Timedelta("12h")
    textY = 0.07

    # plot ____________
    outname = "timeSeries_and_map_obs_rec_newSize.jpg"
    ifield = 0

    fig_height = (1 / 0.71) * fig_width * (plot_array.shape[0] / plot_array.shape[1])

    fig, axs = pplt.subplots(
        plot_array,
        figsize=(fig_width, fig_height),
        hspace=tuple(hspace),
        wspace=tuple(wspace),
        share=False,
    )

    # time series ---------------------------------
    ax = axs[0]

    # radolan spatial mean curve
    ax.plot(
        rad_mean_all, c="grey", lw=0.5, linestyle="-", label="RADOLAN-RW Spatial Mean"
    )

    # mark selected time steps
    for t_ in tMapsList:
        ax.axvline(x=pd.to_datetime(t_), c="k", lw=1.5, zorder=0)
        # ax.text(pd.to_datetime(t_)+textXOffset, textY, tLabel, c='white', bbox=dict(facecolor='black', alpha=0.5))

    # for legend only
    ax.axvline(
        x=pd.to_datetime(t_array[-1]) + pd.Timedelta("24h"),
        c="k",
        lw=1.5,
        zorder=0,
        label="Selected Time Steps",
    )

    # color where RM calculated
    for i, (start, end) in enumerate(zip(beginAvailPeriod, endAvailPeriod)):
        if i == 0:
            ax.axvspan(
                start,
                end,
                alpha=alpha_periods,
                color="darkgreen",
                ec="k",
                label="Reconstructions Available",
            )
        else:
            ax.axvspan(start, end, alpha=alpha_periods, color="darkgreen")

    # color where CML not available
    for i, (start, end) in enumerate(zip(beginNanPeriod, endNanPeriod)):
        if i == 0:
            ax.axvspan(
                start,
                end,
                alpha=alpha_periods,
                hatch="////",
                color="dimgrey",
                ec="k",
                label="CML Not Available",
            )
        else:
            ax.axvspan(start, end, alpha=alpha_periods, hatch="////", color="dimgrey")

    # color where observations are too dry
    for i, (start, end) in enumerate(zip(beginDryPeriod, endDryPeriod)):
        if i == 0:
            ax.axvspan(
                start, end, alpha=alpha_periods, color="white", ec="k", label="Too Dry"
            )
        else:
            ax.axvspan(start, end, alpha=alpha_periods, color="white")

    ax.set_ylim([0, 1])
    ax.set_xlim([np.datetime64(t_array[0] - pd.Timedelta("1h")), t_array[-1]])
    ax.legend(ncols=3, loc="ul", fontsize=fontsz_small)

    ax.set_xticks(specialDates)

    ax.format(
        xlocator=("month", range(3)),  # range(0,32,3)),
        xformatter="%b %d",
        xtickminor=[],
        xrotation=0,
        yticks=[0, 0.5, 1],
        ylabel="P [mm]",
        grid=False,
    )

    # lower part ---------------------------------
    # leftFigMargin = axs[1].get_window_extent().x0
    # rightFigMargin = fig.get_window_extent().width - axs[10].get_window_extent().x1
    # leftrightDiff = leftFigMargin - rightFigMargin

    # number of axes per time step
    n_per_t = 4
    for i, (t_, tLabel) in enumerate(zip(tMapsList, tLabelsFull)):

        # for axes selection
        k = 1

        # # titles
        # fig.text((axs[n_per_t*i+k].get_window_extent().x1 - leftrightDiff/2) / fig.get_window_extent().width,
        #                 0.67, tLabel, fontweight='bold', va="center", ha="center")

        fig.text(
            (axs[n_per_t * i + k].get_window_extent().x1)
            / fig.get_window_extent().width
            - 0.012,
            0.68,  # 0.67
            tLabel,
            fontweight="bold",
            va="center",
            ha="center",
        )

        # fig.text((axs[n_per_t*i+k].get_window_extent().x1) / fig.bbox.width - 0.02,
        #                 0.67, '|', fontweight='bold', va="center", ha="center")

        # CML ==============================
        cml_norm_col = transform_to_linear_colormap(
            ds_cmlTMaps.rain.sel(time=t_).values, cLevels, cMapRain
        )
        for lata, lona, latb, lonb, colo in zip(
            ds_cmlTMaps.y_a.values,
            ds_cmlTMaps.x_a.values,
            ds_cmlTMaps.y_b.values,
            ds_cmlTMaps.x_b.values,
            cml_norm_col,
        ):
            axs[n_per_t * i + k].plot(
                [lona, lonb], [lata, latb], "-", c="k", lw=lw_cml * 1.1, zorder=0
            )
            axs[n_per_t * i + k].plot(
                [lona, lonb], [lata, latb], "-", c=colo, lw=lw_cml
            )  # , zorder=4)
        plot_German_border(
            ax=axs[n_per_t * i + k], border_data=border_data, linewidth=lw_border
        )
        axs[n_per_t * i + k].format(ultitle="CMLs")

        # RGs ============================
        k += 1
        rg_norm_col = transform_to_linear_colormap(
            ds_rgTMaps.rain.sel(time=t_).values, cLevels, cMapRain
        )
        axs[n_per_t * i + k].scatter(
            ds_rgTMaps.x,
            ds_rgTMaps.y,
            c=rg_norm_col,
            ec="k",
            linewidth=lw_rg,
            s=mrksz_rg,
        )
        plot_German_border(
            ax=axs[n_per_t * i + k], border_data=border_data, linewidth=lw_border
        )
        axs[n_per_t * i + k].format(ultitle="Rain Gauges")

        # RAD =======================
        k += 1
        axs[n_per_t * i + k].pcolormesh(
            radTMaps.sel(time=t_), levels=cLevels0, cmap=cMapRain0, extend="both"
        )
        plot_German_border(
            ax=axs[n_per_t * i + k], border_data=border_data, linewidth=lw_border
        )
        axs[n_per_t * i + k].format(ultitle="RADOLAN-RW")

        # RM =======================
        k += 1
        im = axs[n_per_t * i + k].pcolormesh(
            rmsTMaps.sel(time=t_).isel(nfields=0),
            levels=cLevels,
            cmap=cMapRain,
            extend="both",
        )
        plot_German_border(
            ax=axs[n_per_t * i + k], border_data=border_data, linewidth=lw_border
        )
        axs[n_per_t * i + k].format(ultitle="RM single\nmember")

    # settings for all maps
    axs[1:].format(
        ylim=[box[0], box[2]],
        xlim=[box[1], box[3]],
        yticks=[],
        xticks=[],
        ylabel="",
        xlabel="",
        aspect="equal",
    )

    # colorbar
    fig.colorbar(
        im,
        ticks=cLevels,
        loc="b",
        width=cbar_width,
        extend="both",
        shrink=0.6,
        cols=(1, 6),
        label="P [mm]",
    )

    if outfolder is not None:
        fig.savefig(outfolder + outname, dpi=resolution)


def map_with_SAL_bars(
    t,
    outname,
    title,
    rms0_,
    rad0_,
    kri0_,
    ds_rm_SAL_,
    ds_kri_SAL_,
    nfields,
    cmap_rain,
    clevs_rain,
    cmap_binary,
    border_data,
    fig_width=5.5,
    outfolder=None,
):
    """
    Figure 3
    """

    plot_array = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            # [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ]
    )

    hspace = tuple(np.ones(plot_array.shape[0] - 1) * 0.0)  # 0.05
    wspace = tuple(np.ones(plot_array.shape[1] - 1) * 0.0)  # 0.05

    # constants of RADOLAN projection
    xstart = -523.4621669218558
    ystart = -4658.644724265572

    ymin = 0
    ymax = 900
    xmin = 200
    xmax = 900

    # plot ____________
    mapsz = 7
    # fontsz = 14

    # options
    cbar_width = 0.05
    sallinelw = 0.8
    lw_border = 0.2
    lw_thld = 0.2
    mrksz_tcm = 20
    lw_tcm = 0.5
    mrksz_sal = 8  # 6
    mrksz_sal_factor = 3
    mrklw_sal = 0.5
    mrk_tcm = "X"
    mrk_kri = "o"
    mrk_rm = "s"
    ifield = 0
    c_tcm_other = "grey"
    c_tcm_main = "red"
    alpha_tcm = 1
    fontsz_small = 6
    plt.rcParams["font.size"] = 8  # 4
    resolution = 400

    # .94
    fig_height = (1 / 0.88) * fig_width * (plot_array.shape[0] / plot_array.shape[1])

    fig, axs = pplt.subplots(
        plot_array,
        figsize=(fig_width, fig_height),
        hspace=hspace,
        wspace=wspace,
        # space=0.1,
        sharey=0,
        sharex=0,
    )

    # map plots -----------------

    mapi = 0

    # ================================
    # RAD: field, contour, center of mass, formatting
    im = axs[mapi].pcolormesh(rad0_, levels=clevs_rain, cmap=cmap_rain, extend="both")
    axs[mapi].contour(
        rad0_,
        levels=np.array([-1, float(ds_rm_SAL_["thld_ref"].values)]),
        colors=cmap_binary,
        lw=lw_thld,
    )

    axs[mapi].scatter(
        ds_rm_SAL_["tcm_x_sim"].isel(nfields_sim=0).values,
        ds_rm_SAL_["tcm_y_sim"].isel(nfields_sim=0).values,
        marker=mrk_tcm,
        color=c_tcm_other,
        ec="white",
        lw=lw_tcm,
        alpha=alpha_tcm,
        s=mrksz_tcm,
        zorder=4,
    )
    axs[mapi].scatter(
        ds_kri_SAL_["tcm_x_sim"].values,
        ds_kri_SAL_["tcm_y_sim"].values,
        marker=mrk_tcm,
        color=c_tcm_other,
        ec="white",
        lw=lw_tcm,
        alpha=alpha_tcm,
        s=mrksz_tcm,
        zorder=4,
    )
    axs[mapi].scatter(
        ds_rm_SAL_["tcm_x_ref"].values,
        ds_rm_SAL_["tcm_y_ref"].values,
        marker=mrk_tcm,
        color=c_tcm_main,
        ec="white",
        lw=lw_tcm,
        s=mrksz_tcm,
        zorder=4,
    )

    axs[mapi].set_title("RADOLAN-RW")
    plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

    # ================================
    # KRI: field, contour, center of mass, formatting
    mapi += 1
    im = axs[mapi].pcolormesh(kri0_, levels=clevs_rain, cmap=cmap_rain, extend="both")
    axs[mapi].contour(
        kri0_,
        levels=np.array([-1, float(ds_kri_SAL_["thld_sim"].values)]),
        colors=cmap_binary,
        lw=lw_thld,
    )

    axs[mapi].scatter(
        ds_rm_SAL_["tcm_x_sim"].isel(nfields_sim=0).values,
        ds_rm_SAL_["tcm_y_sim"].isel(nfields_sim=0).values,
        marker=mrk_tcm,
        color=c_tcm_other,
        ec="white",
        lw=lw_tcm,
        alpha=alpha_tcm,
        s=mrksz_tcm,
        zorder=4,
    )
    axs[mapi].scatter(
        ds_rm_SAL_["tcm_x_ref"].values,
        ds_rm_SAL_["tcm_y_ref"].values,
        marker=mrk_tcm,
        color=c_tcm_other,
        ec="white",
        lw=lw_tcm,
        alpha=alpha_tcm,
        s=mrksz_tcm,
        zorder=4,
    )
    axs[mapi].scatter(
        ds_kri_SAL_["tcm_x_sim"].values,
        ds_kri_SAL_["tcm_y_sim"].values,
        marker=mrk_tcm,
        color=c_tcm_main,
        ec="white",
        lw=lw_tcm,
        s=mrksz_tcm,
        zorder=4,
    )

    axs[mapi].set_title("KRI")
    plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

    # ================================
    # 1. RM: field, contour, center of mass, formatting
    mapi += 1
    im = axs[mapi].pcolormesh(
        rms0_.isel(nfields=0),
        levels=clevs_rain,
        cmap=cmap_rain,
        extend="both",
    )
    axs[mapi].contour(
        rms0_.isel(nfields=ifield),
        levels=np.array([-1, float(ds_rm_SAL_["thld_sim"].values)]),
        colors=cmap_binary,
        lw=lw_thld,
    )

    axs[mapi].scatter(
        ds_rm_SAL_["tcm_x_ref"].values,
        ds_rm_SAL_["tcm_y_ref"].values,
        marker=mrk_tcm,
        color=c_tcm_other,
        ec="white",
        lw=lw_tcm,
        alpha=alpha_tcm,
        s=mrksz_tcm,
        zorder=4,
    )
    axs[mapi].scatter(
        ds_kri_SAL_["tcm_x_sim"].values,
        ds_kri_SAL_["tcm_y_sim"].values,
        marker=mrk_tcm,
        color=c_tcm_other,
        ec="white",
        lw=lw_tcm,
        alpha=alpha_tcm,
        s=mrksz_tcm,
        zorder=4,
    )
    axs[mapi].scatter(
        ds_rm_SAL_["tcm_x_sim"].isel(nfields_sim=ifield).values,
        ds_rm_SAL_["tcm_y_sim"].isel(nfields_sim=ifield).values,
        marker=mrk_tcm,
        color=c_tcm_main,
        ec="white",
        lw=lw_tcm,
        s=mrksz_tcm,
        zorder=4,
    )

    axs[mapi].set_title("sRM")
    plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

    # ================================
    # settings for all maps
    axs[:3].format(
        ylim=[ymin, ymax],
        xlim=[xmin, xmax],
        yticks=[],
        xticks=[],
        ylabel="",
        xlabel="",
        aspect="equal",
    )

    # colorbar
    axs[2].colorbar(
        im,
        ticks=clevs_rain[::2],
        width=cbar_width,
        extend="both",
        shrink=0.8,
        label="P [mm]",
    )

    # legend
    axs[0].plot([], [], c="%s" % cmap_binary[1], lw=lw_thld, label="Threshold")
    axs[0].scatter(
        [],
        [],
        marker=mrk_tcm,
        color=c_tcm_main,
        ec="white",
        lw=lw_tcm,
        s=mrksz_tcm,
        label="C. o. mass\n(this)",
    )
    axs[0].scatter(
        [],
        [],
        marker=mrk_tcm,
        color=c_tcm_other,
        ec="white",
        lw=lw_tcm,
        s=mrksz_tcm,
        label="C. o. mass\n(other two)",
    )

    if outname == "A" or outname == "C":
        axs[0].legend(loc="lr", ncols=1, fontsize=fontsz_small)  # , fontsize=14)
    else:
        axs[0].legend(loc="ul", ncols=1, fontsize=fontsz_small)  # , fontsize=14)

    # =======================================================================
    # SAL Plot -------------------

    # RM eSAL ================================
    axs[3].scatter(
        ds_rm_SAL_["S"].values,
        3,
        c="orange",
        marker=mrk_rm,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        lw=mrklw_sal,
        zorder=4,
        label="eRM",
    )
    axs[3].scatter(
        ds_rm_SAL_["A"].values,
        2,
        c="orange",
        marker=mrk_rm,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        lw=mrklw_sal,
        zorder=4,
    )
    axs[3].scatter(
        ds_rm_SAL_["L"].values,
        1,
        c="orange",
        marker=mrk_rm,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        lw=mrklw_sal,
        zorder=4,
    )

    # RM SAL shown member ================================
    axs[3].scatter(
        ds_rm_SAL_["mS"].isel(nfields_sim=ifield),
        3,
        c="orange",
        marker="D",
        s=mrksz_sal,
        lw=mrklw_sal,
        zorder=4,
        label="sRM (shown)",
    )
    axs[3].scatter(
        ds_rm_SAL_["mA"].isel(nfields_sim=ifield),
        2,
        c="orange",
        marker="D",
        s=mrksz_sal,
        lw=mrklw_sal,
        zorder=4,
    )
    axs[3].scatter(
        ds_rm_SAL_["mL"].isel(nfields_sim=ifield),
        1,
        c="orange",
        marker="D",
        s=mrksz_sal,
        lw=mrklw_sal,
        zorder=4,
    )

    # KRI SAL ================================
    axs[3].scatter(
        ds_kri_SAL_["S"].values,
        3,
        c="purple",
        marker=mrk_kri,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        lw=mrklw_sal,
        zorder=4,
        label="KRI",
    )
    axs[3].scatter(
        ds_kri_SAL_["A"].values,
        2,
        c="purple",
        marker=mrk_kri,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        lw=mrklw_sal,
        zorder=4,
    )
    axs[3].scatter(
        ds_kri_SAL_["L"].values,
        1,
        c="purple",
        marker=mrk_kri,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        lw=mrklw_sal,
        zorder=4,
    )

    # RM SAL single mebers ================================
    axs[3].scatter(
        ds_rm_SAL_["mS"],
        np.ones(nfields) * 3,
        c="grey",
        marker="D",
        s=mrksz_sal,
        lw=mrklw_sal,
        zorder=3,
        label="sRM (not shown)",
    )
    axs[3].scatter(
        ds_rm_SAL_["mA"],
        np.ones(nfields) * 2,
        c="grey",
        marker="D",
        s=mrksz_sal,
        lw=mrklw_sal,
        zorder=3,
    )
    axs[3].scatter(
        ds_rm_SAL_["mL"],
        np.ones(nfields) * 1,
        c="grey",
        marker="D",
        s=mrksz_sal,
        lw=mrklw_sal,
        zorder=3,
    )

    # ================================
    # vertical zero line
    axs[3].axvline(0, c="k", linewidth=1, zorder=2)

    # horizontal bars
    axs[3].axhline(3, c="gray", lw=sallinelw)
    axs[3].axhline(2, c="gray", lw=sallinelw)
    axs[3].axhline(1, xmin=0.5, c="gray", lw=sallinelw)

    # formatting
    axs[3].set_ylim((0.5, 3.5))
    axs[3].set_yticks([1, 2, 3])
    axs[3].set_yticklabels(["L", "A", "S"])
    axs[3].set_xlim((-2, 2))
    axs[3].set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    axs[3].set_xticklabels([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    axs[3].format(
        xlabel="",
        ygrid=False,
        ytickminor=False,
    )

    axs[3].yaxis.set_label_position("right")
    axs[3].yaxis.tick_right()

    axs[3].legend(loc="ll", ncols=2, fontsize=fontsz_small)

    axs.format(
        suptitle="\n%s" % title,
        # suptitle='\n%s %s'%(str(t).split('T')[0], str(t).split('T')[1].split('.')[0]),
    )
    if outfolder is not None:
        fig.savefig(outfolder + outname + ".jpg", dpi=resolution)


def map_with_SAL_bars_together(
    ts,
    outname,
    titles,
    rms0_s,
    rad0_s,
    kri0_s,
    ds_rm_SAL_s,
    ds_kri_SAL_s,
    nfields,
    cmap_rain,
    clevs_rain,
    cmap_binary,
    border_data,
    fig_width=5.5,
    outfolder=None,
):
    """
    Figure 4
    """

    plot_array = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            # [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7],
            [5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7],
            [5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7],
            [5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7],
            [5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7],
            [5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7],
            [5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7],
            [5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7],
            [5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7],
            # [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ]
    )

    hspace = list(np.ones(plot_array.shape[0] - 1) * 0.0)
    hspace[10] = 0.4
    wspace = tuple(np.ones(plot_array.shape[1] - 1) * 0.0)

    # constants of RADOLAN projection
    xstart = -523.4621669218558
    ystart = -4658.644724265572

    ymin = 0
    ymax = 900
    xmin = 200
    xmax = 900

    # plot ____________
    mapsz = 7
    # fontsz = 14

    # options
    cbar_width = 0.05
    sallinelw = 0.8
    lw_border = 0.2
    lw_thld = 0.2
    mrksz_tcm = 20
    lw_tcm = 0.5
    mrksz_sal = 6
    mrksz_sal_factor = 3
    mrklw_sal = 0.5
    mrk_tcm = "X"
    mrk_kri = "o"
    mrk_rm = "s"
    ifield = 0
    c_tcm_other = "grey"
    c_tcm_main = "red"
    alpha_tcm = 1
    fontsz_small = 6
    plt.rcParams["font.size"] = 8
    resolution = 400

    fig_height = (1 / 0.96) * fig_width * (plot_array.shape[0] / plot_array.shape[1])

    fig, axs = pplt.subplots(
        plot_array,
        figsize=(fig_width, fig_height),
        hspace=hspace,
        wspace=wspace,
        # space=0.1,
        sharey=0,
        sharex=0,
    )

    for i_timestep in range(2):

        t = ts[i_timestep]
        title = titles[i_timestep]
        rms0_ = rms0_s[i_timestep]
        rad0_ = rad0_s[i_timestep]
        kri0_ = kri0_s[i_timestep]
        ds_rm_SAL_ = ds_rm_SAL_s[i_timestep]
        ds_kri_SAL_ = ds_kri_SAL_s[i_timestep]

        # map plots -----------------

        mapi = 0 + 4 * i_timestep

        # ================================
        # RAD: field, contour, center of mass, formatting
        im = axs[mapi].pcolormesh(
            rad0_, levels=clevs_rain, cmap=cmap_rain, extend="both"
        )
        axs[mapi].contour(
            rad0_,
            levels=np.array([-1, float(ds_rm_SAL_["thld_ref"].values)]),
            colors=cmap_binary,
            lw=lw_thld,
        )

        axs[mapi].scatter(
            ds_rm_SAL_["tcm_x_sim"].isel(nfields_sim=0).values,
            ds_rm_SAL_["tcm_y_sim"].isel(nfields_sim=0).values,
            marker=mrk_tcm,
            color=c_tcm_other,
            ec="white",
            lw=lw_tcm,
            alpha=alpha_tcm,
            s=mrksz_tcm,
            zorder=4,
        )
        axs[mapi].scatter(
            ds_kri_SAL_["tcm_x_sim"].values,
            ds_kri_SAL_["tcm_y_sim"].values,
            marker=mrk_tcm,
            color=c_tcm_other,
            ec="white",
            lw=lw_tcm,
            alpha=alpha_tcm,
            s=mrksz_tcm,
            zorder=4,
        )
        axs[mapi].scatter(
            ds_rm_SAL_["tcm_x_ref"].values,
            ds_rm_SAL_["tcm_y_ref"].values,
            marker=mrk_tcm,
            color=c_tcm_main,
            ec="white",
            lw=lw_tcm,
            s=mrksz_tcm,
            zorder=4,
        )

        axs[mapi].set_ylabel(t, fontweight="bold")
        axs[mapi].set_title("RADOLAN-RW")
        plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

        # ================================
        # KRI: field, contour, center of mass, formatting
        mapi += 1
        im = axs[mapi].pcolormesh(
            kri0_, levels=clevs_rain, cmap=cmap_rain, extend="both"
        )
        axs[mapi].contour(
            kri0_,
            levels=np.array([-1, float(ds_kri_SAL_["thld_sim"].values)]),
            colors=cmap_binary,
            lw=lw_thld,
        )

        axs[mapi].scatter(
            ds_rm_SAL_["tcm_x_sim"].isel(nfields_sim=0).values,
            ds_rm_SAL_["tcm_y_sim"].isel(nfields_sim=0).values,
            marker=mrk_tcm,
            color=c_tcm_other,
            ec="white",
            lw=lw_tcm,
            alpha=alpha_tcm,
            s=mrksz_tcm,
            zorder=4,
        )
        axs[mapi].scatter(
            ds_rm_SAL_["tcm_x_ref"].values,
            ds_rm_SAL_["tcm_y_ref"].values,
            marker=mrk_tcm,
            color=c_tcm_other,
            ec="white",
            lw=lw_tcm,
            alpha=alpha_tcm,
            s=mrksz_tcm,
            zorder=4,
        )
        axs[mapi].scatter(
            ds_kri_SAL_["tcm_x_sim"].values,
            ds_kri_SAL_["tcm_y_sim"].values,
            marker=mrk_tcm,
            color=c_tcm_main,
            ec="white",
            lw=lw_tcm,
            s=mrksz_tcm,
            zorder=4,
        )

        axs[mapi].set_title("KRI")
        plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

        # ================================
        # 1. RM: field, contour, center of mass, formatting
        mapi += 1
        im = axs[mapi].pcolormesh(
            rms0_.isel(nfields=0),
            levels=clevs_rain,
            cmap=cmap_rain,
            extend="both",
        )
        axs[mapi].contour(
            rms0_.isel(nfields=ifield),
            levels=np.array([-1, float(ds_rm_SAL_["thld_sim"].values)]),
            colors=cmap_binary,
            lw=lw_thld,
        )

        axs[mapi].scatter(
            ds_rm_SAL_["tcm_x_ref"].values,
            ds_rm_SAL_["tcm_y_ref"].values,
            marker=mrk_tcm,
            color=c_tcm_other,
            ec="white",
            lw=lw_tcm,
            alpha=alpha_tcm,
            s=mrksz_tcm,
            zorder=4,
        )
        axs[mapi].scatter(
            ds_kri_SAL_["tcm_x_sim"].values,
            ds_kri_SAL_["tcm_y_sim"].values,
            marker=mrk_tcm,
            color=c_tcm_other,
            ec="white",
            lw=lw_tcm,
            alpha=alpha_tcm,
            s=mrksz_tcm,
            zorder=4,
        )
        axs[mapi].scatter(
            ds_rm_SAL_["tcm_x_sim"].isel(nfields_sim=ifield).values,
            ds_rm_SAL_["tcm_y_sim"].isel(nfields_sim=ifield).values,
            marker=mrk_tcm,
            color=c_tcm_main,
            ec="white",
            lw=lw_tcm,
            s=mrksz_tcm,
            zorder=4,
        )

        axs[mapi].set_title("sRM")
        plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

        # ================================
        # settings for all maps
        axs[4 * i_timestep].format(
            ylabel=titles[i_timestep],
        )

        axs[4 * i_timestep : 4 * i_timestep + 3].format(
            # suptitle=title,
            ylim=[ymin, ymax],
            xlim=[xmin, xmax],
            yticks=[],
            xticks=[],
            # ylabel="",
            xlabel="",
            aspect="equal",
        )

        axs[4 * i_timestep + 1 : 4 * i_timestep + 3].format(
            # suptitle=title,
            # ylim=[ymin, ymax],
            # xlim=[xmin, xmax],
            # yticks=[],
            # xticks=[],
            ylabel="",
            # xlabel="",
            # aspect="equal",
        )

        if i_timestep == 0:
            # colorbar
            axs[2].colorbar(
                im,
                # loc='ul',
                ticks=clevs_rain[::2],
                width=cbar_width,
                extend="both",
                shrink=0.8,
                label="P [mm]",
            )

            # legend
            axs[0].plot([], [], c="%s" % cmap_binary[1], lw=lw_thld, label="Threshold")
            axs[0].scatter(
                [],
                [],
                marker=mrk_tcm,
                color=c_tcm_main,
                ec="white",
                lw=lw_tcm,
                s=mrksz_tcm,
                label="C. o. mass\n(this)",
            )
            axs[0].scatter(
                [],
                [],
                marker=mrk_tcm,
                color=c_tcm_other,
                ec="white",
                lw=lw_tcm,
                s=mrksz_tcm,
                label="C. o. mass\n(other two)",
            )
            axs[0].legend(loc="ur", ncols=1, fontsize=fontsz_small)  # , fontsize=14)
            # if outname == "C":
            #     axs[0].legend(loc="lr", ncols=1, fontsize=fontsz_small)  # , fontsize=14)
            # else:
            #     axs[0].legend(loc="ul", ncols=1, fontsize=fontsz_small)  # , fontsize=14)

        # =======================================================================
        # SAL Plot -------------------
        ax_sal = axs[3 + 4 * i_timestep]

        # RM eSAL ================================
        ax_sal.scatter(
            ds_rm_SAL_["S"].values,
            3,
            c="orange",
            marker=mrk_rm,
            s=mrksz_sal * mrksz_sal_factor,
            ec="k",
            lw=mrklw_sal,
            zorder=4,
            label="eRM",
        )
        ax_sal.scatter(
            ds_rm_SAL_["A"].values,
            2,
            c="orange",
            marker=mrk_rm,
            s=mrksz_sal * mrksz_sal_factor,
            ec="k",
            lw=mrklw_sal,
            zorder=4,
        )
        ax_sal.scatter(
            ds_rm_SAL_["L"].values,
            1,
            c="orange",
            marker=mrk_rm,
            s=mrksz_sal * mrksz_sal_factor,
            ec="k",
            lw=mrklw_sal,
            zorder=4,
        )

        # RM SAL shown member ================================

        ax_sal.scatter(
            ds_rm_SAL_["mS"].isel(nfields_sim=ifield),
            3,
            c="orange",
            marker="D",
            s=mrksz_sal,
            lw=mrklw_sal,
            zorder=4,
            label="sRM (shown)",
        )
        ax_sal.scatter(
            ds_rm_SAL_["mA"].isel(nfields_sim=ifield),
            2,
            c="orange",
            marker="D",
            s=mrksz_sal,
            lw=mrklw_sal,
            zorder=4,
        )
        ax_sal.scatter(
            ds_rm_SAL_["mL"].isel(nfields_sim=ifield),
            1,
            c="orange",
            marker="D",
            s=mrksz_sal,
            lw=mrklw_sal,
            zorder=4,
        )

        # KRI SAL ================================
        ax_sal.scatter(
            ds_kri_SAL_["S"].values,
            3,
            c="purple",
            marker=mrk_kri,
            s=mrksz_sal * mrksz_sal_factor,
            ec="k",
            lw=mrklw_sal,
            zorder=4,
            label="KRI",
        )
        ax_sal.scatter(
            ds_kri_SAL_["A"].values,
            2,
            c="purple",
            marker=mrk_kri,
            s=mrksz_sal * mrksz_sal_factor,
            ec="k",
            lw=mrklw_sal,
            zorder=4,
        )
        ax_sal.scatter(
            ds_kri_SAL_["L"].values,
            1,
            c="purple",
            marker=mrk_kri,
            s=mrksz_sal * mrksz_sal_factor,
            ec="k",
            lw=mrklw_sal,
            zorder=4,
        )

        # RM SAL single mebers ================================
        ax_sal.scatter(
            ds_rm_SAL_["mS"],
            np.ones(nfields) * 3,
            c="grey",
            marker="D",
            s=mrksz_sal,
            lw=mrklw_sal,
            zorder=3,
            label="sRM (not shown)",
        )
        ax_sal.scatter(
            ds_rm_SAL_["mA"],
            np.ones(nfields) * 2,
            c="grey",
            marker="D",
            s=mrksz_sal,
            lw=mrklw_sal,
            zorder=3,
        )
        ax_sal.scatter(
            ds_rm_SAL_["mL"],
            np.ones(nfields) * 1,
            c="grey",
            marker="D",
            s=mrksz_sal,
            lw=mrklw_sal,
            zorder=3,
        )

        # ================================
        # vertical zero line
        ax_sal.axvline(0, c="k", linewidth=1, zorder=2)

        # horizontal bars
        ax_sal.axhline(3, c="gray", lw=sallinelw)
        ax_sal.axhline(2, c="gray", lw=sallinelw)
        ax_sal.axhline(1, xmin=0.5, c="gray", lw=sallinelw)

        # formatting
        ax_sal.set_ylim((0.5, 3.5))
        ax_sal.set_yticks([1, 2, 3])
        ax_sal.set_yticklabels(["L", "A", "S"])
        ax_sal.set_xlim((-2, 2))
        ax_sal.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        ax_sal.set_xticklabels([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        ax_sal.format(
            xlabel="",
            ygrid=False,
            ytickminor=False,
        )

        ax_sal.yaxis.set_label_position("right")
        ax_sal.yaxis.tick_right()

        if i_timestep == 0:
            ax_sal.legend(loc="ll", ncols=2, fontsize=fontsz_small)

        # axs.format(
        #     suptitle="\n%s" % title,
        #     # suptitle='\n%s %s'%(str(t).split('T')[0], str(t).split('T')[1].split('.')[0]),
        # )
    if outfolder is not None:
        fig.savefig(outfolder + outname + ".jpg", dpi=resolution)


def SAL_boxplot_contour(
    ds_rms_SAL, ds_rmm_SAL, ds_kri_SAL, fig_width=5.5, outfolder=None
):
    """
    Figure 5
    """

    # Calculations ================================================
    # RM single fields
    rms_data = np.array(
        [
            ds_rms_SAL.S.values,
            ds_rms_SAL.A.values,
            ds_rms_SAL.L.values,
            # ds_rms_SAL.Q.values
        ]
    ).T
    df_rms_sal = pd.DataFrame(
        data=rms_data, index=ds_rms_SAL.time.values, columns=["S", "A", "L"]
    )  # , "|SAL|"])
    df_rms_sal["Reconstruction"] = "eRM"

    # RM mean fields
    rmm_data = np.array(
        [
            ds_rmm_SAL.S.values,
            ds_rmm_SAL.A.values,
            ds_rmm_SAL.L.values,
            # ds_rmm_SAL.Q.values
        ]
    ).T
    df_rmm_sal = pd.DataFrame(
        data=rmm_data, index=ds_rmm_SAL.time.values, columns=["S", "A", "L"]
    )  # , "|SAL|"])
    df_rmm_sal["Reconstruction"] = "mRM"

    # Kriging
    kri_data = np.array(
        [
            ds_kri_SAL.S.values,
            ds_kri_SAL.A.values,
            ds_kri_SAL.L.values,
            # ds_kri_SAL.Q.values
        ]
    ).T
    df_kri_sal = pd.DataFrame(
        data=kri_data, index=ds_kri_SAL.time.values, columns=["S", "A", "L"]
    )  # , '|SAL|'])
    df_kri_sal["Reconstruction"] = "KRI"

    # combine
    df_sal = pd.concat([df_rms_sal, df_rmm_sal, df_kri_sal])
    df_sal.index.rename("time", inplace=True)
    df_sal = df_sal.melt(id_vars=["Reconstruction"])

    # figure ================================================

    # some figure options
    colors = {"rms": "orange", "kri": "purple"}
    lnstyle = {"rms": "-", "kri": ":"}
    ec = "k"
    lw = 0.5
    lw_zero = 1
    s = 100
    alpha = 0.75
    resolution = 400
    plt.rcParams["font.size"] = 8  # 4
    fig_height = fig_width / 2  # 2

    fig = plt.figure(figsize=(fig_width, fig_height), tight_layout=True)
    gs0 = fig.add_gridspec(
        1,
        2,
    )  # wspace=0.3) # 0.2)
    ax0 = fig.add_subplot(gs0[0])
    # ax1 = fig.add_subplot(gs0[1])
    gssub = gs0[1].subgridspec(2, 2, wspace=0)
    ax = [
        fig.add_subplot(gssub[0, 0]),
        fig.add_subplot(gssub[1, 0]),
        fig.add_subplot(gssub[0, 1]),
        fig.add_subplot(gssub[1, 1]),
    ]

    sns.boxplot(
        x="value",
        y="variable",
        hue="Reconstruction",
        data=df_sal,
        palette=["orange", "goldenrod", "purple"],
        width=0.5,
        linewidth=1,
        flierprops=dict(
            markerfacecolor="white",
            # markeredgecolor="black",
            markersize=0.5,
            marker="D",
        ),
        ax=ax0,
    )

    ax0.set_ylabel("")
    ax0.set_xlabel("")
    ax0.set_yticks(np.arange(3))
    ax0.set_yticklabels(
        # ["Structure (S)", "Amplitude (A)", "Location (L)"]
        ["S", "A", "L"]
    )  # , "Combined (|SAL|)"])
    ax0.set_yticks([], minor=True)
    ax0.set_xticks([-2, -1, 0, 1, 2])
    ax0.set_xticks([], minor=True)
    ax0.set_xlim((-2, 2))

    ax0.axvline(x=0, c="grey", lw=lw_zero, ls="--", label="optimal", zorder=0)
    ax0.legend()

    # ======================
    # Contour

    # find data range
    # axlims = find_SAL_data_range([ds_rms_SAL, ds_kri_SAL])
    axlims = {"S": [-2, 2], "A": [-2, 2], "L": [0, 1]}

    for i, p1 in enumerate(["S", "A"]):

        for j, p2 in enumerate(["A", "L"]):

            axi = 2 * i + j

            if p1 == p2:
                ax[axi].set_visible(False)
                continue

            # main part
            for ds, k in zip([ds_rms_SAL, ds_kri_SAL], ["rms", "kri"]):

                # for contours
                sns.kdeplot(
                    x=ds[p1].values,
                    y=ds[p2].values,
                    levels=[0.25, 0.5, 0.75, 1],
                    color="black",
                    linewidths=lw,
                    linestyles=lnstyle[k],
                    ax=ax[axi],
                )

                # for fill of contours
                sns.kdeplot(
                    x=ds[p1].values,
                    y=ds[p2].values,
                    #                              label = k,
                    levels=[0.25, 0.5, 0.75, 1],
                    fill=True,
                    alpha=alpha,
                    color=colors[k],
                    ax=ax[axi],
                )

                # adjustments
                ax[axi].set_xlim(axlims[p1])
                ax[axi].set_ylim(axlims[p2])
                ax[axi].set_xlabel(p1)
                ax[axi].set_ylabel(p2)
                ax[axi].set_xticks([-1, 0, 1])
                if p2 == "A":
                    ax[axi].set_yticks([-1, 0, 1])
                elif p2 == "L":
                    ax[axi].set_yticks([0, 0.5])
                ax[axi].set_yticks([], minor=True)
                ax[axi].set_xticks([], minor=True)

            # zero lines
            ax[axi].plot(axlims[p1], [0, 0], c="grey", lw=lw_zero, ls="--")
            ax[axi].plot([0, 0], axlims[p2], c="grey", lw=lw_zero, ls="--")

    # ax[3].set_yticks([])
    ax[3].set_yticklabels("")
    ax[3].set_ylabel("")
    ax[0].set_xticklabels("")
    ax[0].set_xlabel("")

    # legend
    ax[0].plot([], [], c="k", lw=lw, linestyle="-", label="eRM")
    ax[0].plot([], [], c="k", lw=lw, linestyle=":", label="KRI")
    _ = ax[0].legend()

    fig.subplots_adjust(hspace=0, wspace=0)

    ax[3].tick_params(left=False)
    ax[0].tick_params(bottom=False)

    # fig.align_ylabels()

    if outfolder is not None:
        fig.savefig(
            outfolder + "SAL_diagram_boxplot_contour_summer.jpg", dpi=resolution
        )


def SAL_triple_scatter(
    which, ds_rms_SAL, ds_kri_SAL, ref_rain, da_tmask, fig_width=5.5, outfolder=None
):
    """
    Figure 6
    """

    # some figure options
    ec = None
    lw = 0.5
    lw_zero = 0.5
    s = 1
    alpha_low = 0.8
    alpha_middle = 0.8
    alpha_high = 0.8
    mrk = "o"
    resolution = 400
    plt.rcParams["font.size"] = 4
    fig_height = fig_width

    fig, axs = pplt.subplots(
        ncols=2,
        nrows=2,
        figsize=(fig_width, fig_height),
        sharey=3,
        sharex=3,
        spany=0,
        spanx=0,
        hspace=0,
        wspace=0,
    )

    # for length of colorbar
    x0pos = axs[0, 1].get_window_extent().x0
    x1pos = axs[0, 1].get_window_extent().x1
    cbar_length = 0.9 * (x1pos - x0pos) / 100

    # # for colormap
    # cmin = 0
    # cmax = ref_rain.max().values
    # cLevels = (np.linspace(0,cmax,10)*100).astype(int) / 100

    # find data range
    # axlims = find_SAL_data_range([ds_rms_SAL, ds_kri_SAL])
    axlims = {"S": [-2, 2], "A": [-2, 2], "L": [0, 1]}

    # # for title
    # if which == 'rms':
    #     title = 'RM eSAL'
    # elif which == 'kri':
    #     title = 'Kriging SAL'

    ref_rain = ref_rain.where(da_tmask, drop=True)

    # special colormap for categorial coloring
    cLevels = [
        0,
        ref_rain.quantile(0.25).values,
        ref_rain.quantile(0.75).values,
        ref_rain.max(),
    ]

    # separation of different quantiles
    cond_low = ref_rain < cLevels[1]
    cond_middle = (ref_rain >= cLevels[1]) & (ref_rain <= cLevels[2])
    cond_high = ref_rain > cLevels[2]
    ref_rain_low = ref_rain.where(cond_low, drop=True)
    ref_rain_middle = ref_rain.where(cond_middle, drop=True)
    ref_rain_high = ref_rain.where(cond_high, drop=True)
    ds_rms_SAL_low = ds_rms_SAL.where(cond_low, drop=True)
    ds_rms_SAL_middle = ds_rms_SAL.where(cond_middle, drop=True)
    ds_rms_SAL_high = ds_rms_SAL.where(cond_high, drop=True)
    ds_kri_SAL_low = ds_kri_SAL.where(cond_low, drop=True)
    ds_kri_SAL_middle = ds_kri_SAL.where(cond_middle, drop=True)
    ds_kri_SAL_high = ds_kri_SAL.where(cond_high, drop=True)

    # cMap = ListedColormap(['#ffd700', '#648c11', '#0000cd']) original
    cMap = ListedColormap(["#d90101", "#49dd48", "#013dd9"])

    for i, p1 in enumerate(["S", "A"]):

        for j, p2 in enumerate(["A", "L"]):

            if p1 == p2:

                # only show legend for KRI field as both will appear together
                if which == "kri":

                    im = axs[j, i].scatter(
                        np.arange(3) + 666,
                        np.arange(3),
                        c=np.arange(3),
                        alpha=alpha_middle,
                        levels=[0, 0.25, 0.75, 1],
                        cmap=cMap,
                        # vmin=cmin, vmax=cmax,
                        colorbar="ur",
                        colorbar_kw={
                            "length": cbar_length,  #'%iem'%cbar_length,
                            "label": "Quantiles of mean reference rain",
                            #'align': 'center',
                            "ticks": [[0, 0.25, 0.75, 1]],
                            # 'drawedges': False,
                            "linewidth": 0,
                        },
                    )

                axs[j, i].axis("off")
                continue

            # main part
            if which == "rms":

                im = axs[j, i].scatter(
                    ds_rms_SAL_middle[p1].values,
                    ds_rms_SAL_middle[p2].values,
                    c=ref_rain_middle.values,
                    alpha=alpha_middle,
                    s=s,
                    ec=ec,
                    marker=mrk,
                    levels=cLevels,
                    cmap=cMap,
                    lw=lw,
                )

                im = axs[j, i].scatter(
                    ds_rms_SAL_low[p1].values,
                    ds_rms_SAL_low[p2].values,
                    c=ref_rain_low.values,
                    alpha=alpha_low,
                    s=s,
                    ec=ec,
                    marker=mrk,
                    levels=cLevels,
                    cmap=cMap,
                    lw=lw,
                )

                im = axs[j, i].scatter(
                    ds_rms_SAL_high[p1].values,
                    ds_rms_SAL_high[p2].values,
                    c=ref_rain_high.values,
                    alpha=alpha_high,
                    s=s,
                    ec=ec,
                    marker=mrk,
                    levels=cLevels,
                    cmap=cMap,
                    lw=lw,
                )

            elif which == "kri":

                im = axs[j, i].scatter(
                    ds_kri_SAL_middle[p1].values,
                    ds_kri_SAL_middle[p2].values,
                    c=ref_rain_middle.values,
                    alpha=alpha_middle,
                    s=s,
                    ec=ec,
                    marker=mrk,
                    levels=cLevels,
                    cmap=cMap,
                    lw=lw,
                )

                im = axs[j, i].scatter(
                    ds_kri_SAL_low[p1].values,
                    ds_kri_SAL_low[p2].values,
                    c=ref_rain_low.values,
                    alpha=alpha_low,
                    s=s,
                    ec=ec,
                    marker=mrk,
                    levels=cLevels,
                    cmap=cMap,
                    lw=lw,
                )

                im = axs[j, i].scatter(
                    ds_kri_SAL_high[p1].values,
                    ds_kri_SAL_high[p2].values,
                    c=ref_rain_high.values,
                    alpha=alpha_high,
                    s=s,
                    ec=ec,
                    marker=mrk,
                    levels=cLevels,
                    cmap=cMap,
                    lw=lw,
                )

            # adjustments
            axs[j, i].set_xlim(axlims[p1])
            axs[j, i].set_ylim(axlims[p2])
            axs[j, i].set_xlabel(p1)
            axs[j, i].set_ylabel(p2)
            axs[j, i].set_xticks(np.linspace(axlims[p1][0], axlims[p1][1], 5))
            axs[j, i].set_yticks(np.linspace(axlims[p2][0], axlims[p2][1], 5))
            axs[j, i].set_xticks([-1, 0, 1])
            if p2 == "A":
                axs[j, i].set_yticks([-1, 0, 1])
            elif p2 == "L":
                axs[j, i].set_yticks([0, 0.5])
            axs[j, i].set_xticks([], minor=True)
            axs[j, i].set_yticks([], minor=True)

            # zero lines
            axs[j, i].plot(axlims[p1], [0, 0], c="grey", lw=lw_zero)
            axs[j, i].plot([0, 0], axlims[p2], c="grey", lw=lw_zero)

    axs[1, 1].tick_params(left=False)
    axs[0, 0].tick_params(bottom=False)

    # axs.format(
    # suptitle='%s'%title,
    #     aspect='equal',
    # )

    fig.align_ylabels()

    if outfolder is not None:
        fig.savefig(outfolder + "SAL_diagram_crain_%s.jpg" % which, dpi=resolution)


def Pixel_boxplots(
    ds_rms_pixel, ds_rmm_pixel, ds_kri_pixel, fig_width=5.5, outfolder=None
):
    """
    Figure 7
    """

    # Calculations =================================================
    # RM single fields
    rms_data = np.array(
        [
            ds_rms_pixel.pcc.quantile(0.5, "nfields").values,
            ds_rms_pixel.rmse.quantile(0.5, "nfields").values,
            ds_rms_pixel.bias.quantile(0.5, "nfields").values,
        ]
    ).T
    df_rms_pixel = pd.DataFrame(
        data=rms_data, index=ds_rms_pixel.time.values, columns=["PCC", "RMSE", "BIAS"]
    )
    df_rms_pixel["Reconstruction"] = "eRM"

    # RM mean fields
    rmm_data = np.array(
        [
            ds_rmm_pixel.pcc.values,
            ds_rmm_pixel.rmse.values,
            ds_rmm_pixel.bias.values,
        ]
    ).T
    df_rmm_pixel = pd.DataFrame(
        data=rmm_data, index=ds_rmm_pixel.time.values, columns=["PCC", "RMSE", "BIAS"]
    )
    df_rmm_pixel["Reconstruction"] = "mRM"

    # Kriging
    kri_data = np.array(
        [
            ds_kri_pixel.pcc.values,
            ds_kri_pixel.rmse.values,
            ds_kri_pixel.bias.values,
        ]
    ).T
    df_kri_pixel = pd.DataFrame(
        data=kri_data, index=ds_kri_pixel.time.values, columns=["PCC", "RMSE", "BIAS"]
    )
    df_kri_pixel["Reconstruction"] = "KRI"

    # combine
    df_pixel = pd.concat([df_rms_pixel, df_rmm_pixel, df_kri_pixel])
    df_pixel.index.rename("time", inplace=True)
    df_pixel = df_pixel.melt(id_vars=["Reconstruction"])

    # figure ============================================================
    resolution = 400
    plt.rcParams["font.size"] = 8
    fig_height = fig_width / (13 / 8)  # 16/9
    lw_zero = 1

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), tight_layout=True)

    ax = sns.boxplot(
        x="value",
        y="variable",
        hue="Reconstruction",
        data=df_pixel,
        palette=["orange", "goldenrod", "purple"],
        width=0.5,  ###### 0.5
        linewidth=1,  #### None
        flierprops=dict(
            markerfacecolor="white",
            markersize=0.5,  #### 1
            marker="D",
        ),
    )

    ax.axvline(
        1, ymin=0.666, ymax=1, c="grey", lw=lw_zero, ls="--", label="optimal", zorder=0
    )
    ax.axvline(0, ymin=0, ymax=0.666, c="grey", lw=lw_zero, ls="--", zorder=0)
    ax.legend(loc="upper right")

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xlim((-1, 2.5))
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(["PCC", "RMSE", "BIAS"])
    ax.set_yticks([], minor=True)

    if outfolder is not None:
        fig.savefig(outfolder + "Pixelbased_boxplot_summer.jpg", dpi=resolution)


def parameters_for_mean_fields_split(
    df_mean_SAL, df_mean_pixel, fig_width=5.5, outfolder=None
):
    """
    Figure 8
    """

    resolution = 400
    # small_font = 5
    plt.rcParams["font.size"] = 8  # 6

    fig_height = fig_width / 1.5
    fig, axs = pplt.subplots(nrows=2, sharey=0, figsize=(fig_width, fig_height))

    ensSizes = [1, 3, 5, 10, 20]

    c_S = [0.6, 0.6, 1, 1]
    c_A = [0.4, 0.4, 0.7, 1]
    c_L = [0.1, 0.1, 0.5, 1]
    c_pcc = [1, 0.6, 0.6, 1]
    c_rmse = [0.7, 0.4, 0.4, 1]
    c_bias = [0.5, 0.1, 0.1, 1]

    rmls = "-"
    krils = ":"

    ax = axs[0]

    # SAL RM
    ax.plot(ensSizes, df_mean_SAL["S"][:-1], c=c_S, label="S (mRM)")
    ax.plot(ensSizes, df_mean_SAL["A"][:-1], c=c_A, label="A (mRM)")
    ax.plot(ensSizes, df_mean_SAL["L"][:-1], c=c_L, label="L (mRM)")

    # SAL KRI
    ax.plot(
        ensSizes, np.ones(5) * df_mean_SAL["S"][-1], c=c_S, ls=krils, label="S (KRI)"
    )
    ax.plot(
        ensSizes, np.ones(5) * df_mean_SAL["A"][-1], c=c_A, ls=krils, label="A (KRI)"
    )
    ax.plot(
        ensSizes, np.ones(5) * df_mean_SAL["L"][-1], c=c_L, ls=krils, label="L (KRI)"
    )

    # adjustments
    ax.legend(ncols=3, loc="lr")  # , fontsize=small_font)
    ax.set_ylabel("SAL Metrics")

    # ========================
    # second subplot
    ax = axs[1]

    # Standard Metrics RM
    ax.plot(ensSizes, df_mean_pixel["pcc"][:-1], c=c_pcc, label="PCC (mRM)")
    ax.plot(ensSizes, df_mean_pixel["rmse"][:-1], c=c_rmse, label="RMSE (mRM)")
    ax.plot(ensSizes, df_mean_pixel["bias"][:-1], c=c_bias, label="BIAS (mRM)")

    # Standard Metrics KRI
    ax.plot(
        ensSizes,
        np.ones(5) * df_mean_pixel["pcc"][-1],
        c=c_pcc,
        ls=krils,
        label="PCC (KRI)",
    )
    ax.plot(
        ensSizes,
        np.ones(5) * df_mean_pixel["rmse"][-1],
        c=c_rmse,
        ls=krils,
        label="RMSE (KRI)",
    )
    ax.plot(
        ensSizes,
        np.ones(5) * df_mean_pixel["bias"][-1],
        c=c_bias,
        ls=krils,
        label="BIAS (KRI)",
    )

    # adjustments
    ax.legend(ncols=3, loc="lr")  # , fontsize=small_font)
    ax.set_ylabel("Standard Indices")

    axs.format(
        # ylabel="",
        ylim=(-0.5, 0.8),
        yticks=(-0.4, -0.2, 0, 0.2, 0.4, 0.6),
        xlabel="Number of RM ensemble members",
        xticks=ensSizes,
        xtickminor=[],
        xlim=(1, 20),
    )
    if outfolder is not None:
        fig.savefig(outfolder + "meanFieldAnalysis_1.jpg", dpi=resolution)


# ==================================================
# SUPPORTING MATERIAL of DOI: https://doi.org/10.1029/2022WR032563


def RMM_with_SAL_bars_extended(
    t,
    outname,
    title,
    rms0_,
    rad0_,
    kri0_,
    ds_rmm_SAL_3,
    ds_rmm_SAL_5,
    ds_rmm_SAL_10,
    ds_rmm_SAL_20,
    ds_kri_SAL_,
    nfields,
    cmap_rain,
    clevs_rain,
    cmap_binary,
    border_data,
    fig_width=5.5,
    outfolder=None,
):
    """
    Figures S1, S2, S3
    """

    plot_array = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
            [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
            [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
            [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
            [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
            [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
            [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
            [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
            [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        ]
    )

    hspace = tuple(np.ones(plot_array.shape[0] - 1) * 0.0)  # 0.05
    wspace = tuple(np.ones(plot_array.shape[1] - 1) * 0.0)  # 0.05

    ymin = 0
    ymax = 900
    xmin = 200
    xmax = 900

    # plot ____________
    mapsz = 7
    # fontsz = 14

    # options
    cbar_width = 0.05
    sallinelw = 0.8
    lw_border = 0.2
    lw_thld = 0.2
    mrksz_tcm = 20
    lw_tcm = 0.5
    mrksz_sal = 8  # 6
    mrksz_sal_factor = 3
    mrklw_sal = 0.5
    mrk_tcm = "X"
    mrk_kri = "o"
    mrk_rm = "s"
    ifield = 0
    c_tcm_other = "grey"
    c_tcm_main = "red"
    c_3 = "#faed00"
    c_5 = "#fabb00"
    c_10 = "#fa8400"
    c_20 = "#fa2500"
    mrk_3 = "v"
    mrk_5 = "^"
    mrk_10 = "s"
    mrk_20 = "D"
    alpha_sal = 0.5
    alpha_tcm = 1
    fontsz_small = 6
    plt.rcParams["font.size"] = 8  # 4
    resolution = 400

    # .94 / .93
    fig_height = (1 / 0.99) * fig_width * (plot_array.shape[0] / plot_array.shape[1])

    fig, axs = pplt.subplots(
        plot_array,
        figsize=(fig_width, fig_height),
        hspace=hspace,
        wspace=wspace,
        # space=0.1,
        sharey=0,
        sharex=0,
    )

    # map plots -----------------

    mapi = 0

    # ================================
    # RAD: field, contour, center of mass, formatting
    im = axs[mapi].pcolormesh(rad0_, levels=clevs_rain, cmap=cmap_rain, extend="both")
    axs[mapi].contour(
        rad0_,
        levels=np.array([-1, float(ds_rmm_SAL_20["thld_ref"].values)]),
        colors=cmap_binary,
        lw=lw_thld,
    )

    axs[mapi].format(ultitle="RADOLAN-RW")
    plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

    # ================================
    # KRI: field, contour, center of mass, formatting
    mapi += 1
    im = axs[mapi].pcolormesh(kri0_, levels=clevs_rain, cmap=cmap_rain, extend="both")
    axs[mapi].contour(
        kri0_,
        levels=np.array([-1, float(ds_kri_SAL_["thld_sim"].values)]),
        colors=cmap_binary,
        lw=lw_thld,
    )

    axs[mapi].format(ultitle="KRI")
    plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

    # ================================
    # 1. RM: field, contour, center of mass, formatting
    mapi += 1
    im = axs[mapi].pcolormesh(
        rms0_.isel(nfields=slice(0, 3)).mean("nfields"),
        levels=clevs_rain,
        cmap=cmap_rain,
        extend="both",
    )
    axs[mapi].contour(
        rms0_.isel(nfields=slice(0, 3)).mean("nfields"),
        levels=np.array([-1, float(ds_rmm_SAL_3["thld_sim"].values)]),
        colors=cmap_binary,
        lw=lw_thld,
    )

    axs[mapi].format(ultitle="mRM(3)")
    plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

    # ================================
    # 1. RM: field, contour, center of mass, formatting
    mapi += 1
    im = axs[mapi].pcolormesh(
        rms0_.isel(nfields=slice(0, 5)).mean("nfields"),
        levels=clevs_rain,
        cmap=cmap_rain,
        extend="both",
    )
    axs[mapi].contour(
        rms0_.isel(nfields=slice(0, 5)).mean("nfields"),
        levels=np.array([-1, float(ds_rmm_SAL_5["thld_sim"].values)]),
        colors=cmap_binary,
        lw=lw_thld,
    )

    axs[mapi].format(ultitle="mRM(5)")
    plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

    # ================================
    # 1. RM: field, contour, center of mass, formatting
    mapi += 1
    im = axs[mapi].pcolormesh(
        rms0_.isel(nfields=slice(0, 10)).mean("nfields"),
        levels=clevs_rain,
        cmap=cmap_rain,
        extend="both",
    )
    axs[mapi].contour(
        rms0_.isel(nfields=slice(0, 10)).mean("nfields"),
        levels=np.array([-1, float(ds_rmm_SAL_10["thld_sim"].values)]),
        colors=cmap_binary,
        lw=lw_thld,
    )

    axs[mapi].format(ultitle="mRM(10)")
    plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

    # ================================
    # 1. RM: field, contour, center of mass, formatting
    mapi += 1
    im = axs[mapi].pcolormesh(
        rms0_.mean("nfields"),
        levels=clevs_rain,
        cmap=cmap_rain,
        extend="both",
    )
    axs[mapi].contour(
        rms0_.mean("nfields"),
        levels=np.array([-1, float(ds_rmm_SAL_20["thld_sim"].values)]),
        colors=cmap_binary,
        lw=lw_thld,
    )

    axs[mapi].format(ultitle="mRM(20)")
    plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

    # ================================
    # settings for all maps
    axs[:6].format(
        ylim=[ymin, ymax],
        xlim=[xmin, xmax],
        yticks=[],
        xticks=[],
        ylabel="",
        xlabel="",
        aspect="equal",
    )

    # colorbar
    fig.colorbar(
        im,
        loc="r",
        # orientation="vertical",
        ticks=clevs_rain[::2],
        width=cbar_width,
        extend="both",
        shrink=0.8,
        label="P [mm]",
        rows=(1, 18),
    )

    # cb.set_label(label="P [mm]", fontsize=fontsz_small)

    # legend
    axs[0].plot([], [], c="%s" % cmap_binary[1], lw=lw_thld, label="Threshold")

    # if outname == "A" or outname == "C":
    axs[0].legend(loc="lr", ncols=1, fontsize=fontsz_small)  # , fontsize=14)
    # else:
    #     axs[0].legend(loc="ul", ncols=1, fontsize=fontsz_small)  # , fontsize=14)

    # =======================================================================
    # SAL Plot -------------------

    # RMM 3 SAL ================================
    axs[6].scatter(
        ds_rmm_SAL_3["S"].values,
        3,
        c=c_3,
        marker=mrk_3,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
        label="mRM(3)",
    )
    axs[6].scatter(
        ds_rmm_SAL_3["A"].values,
        2,
        c=c_3,
        marker=mrk_3,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
    )
    axs[6].scatter(
        ds_rmm_SAL_3["L"].values,
        1,
        c=c_3,
        marker=mrk_3,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
    )

    # RMM 5 SAL ================================
    axs[6].scatter(
        ds_rmm_SAL_5["S"].values,
        3,
        c=c_5,
        marker=mrk_5,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
        label="mRM(5)",
    )
    axs[6].scatter(
        ds_rmm_SAL_5["A"].values,
        2,
        c=c_5,
        marker=mrk_5,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
    )
    axs[6].scatter(
        ds_rmm_SAL_5["L"].values,
        1,
        c=c_5,
        marker=mrk_5,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
    )

    # RMM 10 SAL ================================
    axs[6].scatter(
        ds_rmm_SAL_10["S"].values,
        3,
        c=c_10,
        marker=mrk_10,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
        label="mRM(10)",
    )
    axs[6].scatter(
        ds_rmm_SAL_10["A"].values,
        2,
        c=c_10,
        marker=mrk_10,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
    )
    axs[6].scatter(
        ds_rmm_SAL_10["L"].values,
        1,
        c=c_10,
        marker=mrk_10,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
    )

    # RMM 20 SAL ================================
    axs[6].scatter(
        ds_rmm_SAL_20["S"].values,
        3,
        c=c_20,
        marker=mrk_20,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
        label="mRM(20)",
    )
    axs[6].scatter(
        ds_rmm_SAL_20["A"].values,
        2,
        c=c_20,
        marker=mrk_20,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
    )
    axs[6].scatter(
        ds_rmm_SAL_20["L"].values,
        1,
        c=c_20,
        marker=mrk_20,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
    )

    # KRI SAL ================================
    axs[6].scatter(
        ds_kri_SAL_["S"].values,
        3,
        c="purple",
        marker=mrk_kri,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
        label="KRI",
    )
    axs[6].scatter(
        ds_kri_SAL_["A"].values,
        2,
        c="purple",
        marker=mrk_kri,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
    )
    axs[6].scatter(
        ds_kri_SAL_["L"].values,
        1,
        c="purple",
        marker=mrk_kri,
        s=mrksz_sal * mrksz_sal_factor,
        ec="k",
        alpha=alpha_sal,
        lw=mrklw_sal,
        zorder=4,
    )

    # ================================
    # vertical zero line
    axs[6].axvline(0, c="k", linewidth=1, zorder=2)

    # horizontal bars
    axs[6].axhline(3, c="gray", lw=sallinelw)
    axs[6].axhline(2, c="gray", lw=sallinelw)
    axs[6].axhline(1, xmin=0.5, c="gray", lw=sallinelw)

    # formatting
    axs[6].set_ylim((0.5, 3.5))
    axs[6].set_yticks([1, 2, 3])
    axs[6].set_yticklabels(["L", "A", "S"])
    axs[6].set_xlim((-2, 2))
    axs[6].set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    axs[6].set_xticklabels([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    axs[6].format(
        xlabel="",
        ygrid=False,
        ytickminor=False,
    )

    axs[6].yaxis.set_label_position("right")
    axs[6].yaxis.tick_right()

    axs[6].legend(loc="ll", ncols=2, fontsize=fontsz_small)

    axs.format(
        suptitle="\n%s" % title,
        # suptitle='\n%s %s'%(str(t).split('T')[0], str(t).split('T')[1].split('.')[0]),
    )
    if outfolder is not None:
        fig.savefig(outfolder + outname + "_RMM_special.jpg", dpi=resolution)


def filter_visualization(
    rad,
    ds_cml_t_subset,
    t_sel,
    tLabel,
    box,
    inset_loc,
    c_loc,
    location_name,
    ymin,
    ymax,
    xmin,
    xmax,
    cMap_standard,
    cLevels_standard,
    border_data,
    fig_width=5.5,
    outfolder=None,
):
    """
    Figures S4, S5
    """

    # figure options that depend on size
    resolution = 400
    plt.rcParams["font.size"] = 8
    lw_border = None
    cml_margin_lw = 5
    cml_inner_lw = 3
    cbar_width = 0.05

    fig_height = (9 / 7) * fig_width

    fig, ax = pplt.subplots(figsize=(fig_width, fig_height))

    # radolan
    im = ax.pcolormesh(
        rad.sel(time=t_sel),
        cmap=cMap_standard,
        levels=cLevels_standard,
        extend="both",
    )

    # border
    plot_German_border(ax, border_data, linewidth=lw_border)

    # outline of subset
    ax.plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], c="k")

    # format
    ax.format(
        title=tLabel,
        ylim=[box[0], box[2]],
        xlim=[box[1], box[3]],
        # ylim=[ymin, ymax],
        # xlim=[xmin, xmax],
        ylabel="y [km]",
        xlabel="x [km]",
        aspect="equal",
    )

    # inset axis ==================================
    axins = ax.inset_axes(bounds=inset_loc)

    # colorscale subset
    cLevels_subset, cMapRain_subset = adjusted_colormap(ds_cml_t_subset)
    cml_norm_col = transform_to_linear_colormap(
        ds_cml_t_subset.rain.values, cLevels_subset, cMapRain_subset
    )

    if np.max(cLevels_subset) > 0:
        # radolan
        imins = axins.pcolormesh(
            rad.sel(time=t_sel),
            cmap=cMapRain_subset,
            levels=cLevels_subset,
            extend="both",
        )

    # positions observations
    for lata, lona, latb, lonb, colo, l in zip(
        ds_cml_t_subset.y_a.values,
        ds_cml_t_subset.x_a.values,
        ds_cml_t_subset.y_b.values,
        ds_cml_t_subset.x_b.values,
        cml_norm_col,
        ds_cml_t_subset.label_sp_sanity.values,
    ):
        if l:
            c = "k"
            zorder = 3
        else:
            c = "r"
            zorder = 4

        axins.plot(
            [lona, lonb], [lata, latb], "-", c=c, lw=cml_margin_lw, zorder=zorder
        )
        axins.plot(
            [lona, lonb], [lata, latb], "-", c=colo, lw=cml_inner_lw, zorder=zorder
        )

    # inset legend
    axins.plot([], [], "r", label="Filtered")
    axins.legend(loc="ul")

    # inset formatting
    axins.format(
        # title=t_sel,
        # ylim=[box[0], box[2]],
        # xlim=[box[1], box[3]],
        yticks=[],
        xticks=[],
        ylim=[ymin, ymax],
        xlim=[xmin, xmax],
        ylabel="",
        xlabel="",
        aspect="equal",
    )

    # zoom lines
    axins.indicate_inset_zoom()

    # space for small inset colorbar
    cax = ax.inset_axes(bounds=c_loc, zoom=False)

    # end of inset ================================

    # colorbars
    ax.colorbar(im, loc="ll", label="P [mm]", width=cbar_width, shrink=0.6)

    if np.max(cLevels_subset) > 0:
        fig.colorbar(
            imins,
            cax=cax,
            label="P [mm]",
            orientation="horizontal",
            ticks=cLevels_subset[::3],
            extend="both",
        )

    if outfolder is not None:
        fig.savefig(
            outfolder + "exmpl_%s_sp_sanity_%s.jpg" % (location_name, t_sel),
            dpi=resolution,
        )


# ====================================================
# ANSWERS TO REVIEWERS


def Pixel_boxplots_answer(ds_rms_pixel, fig_width=5.5, outfolder=None):
    """
    Figure used in answer to reviewers
    """

    # Calculations =================================================

    # data for single members --------------------------------------
    df_list = []
    for ind in np.arange(len(ds_rms_pixel.nfields)):

        m_data = np.array(
            [
                ds_rms_pixel.pcc.isel(nfields=ind).values,
                ds_rms_pixel.rmse.isel(nfields=ind).values,
                ds_rms_pixel.bias.isel(nfields=ind).values,
            ]
        ).T
        df_m = pd.DataFrame(
            data=m_data, index=ds_rms_pixel.time.values, columns=["PCC", "RMSE", "BIAS"]
        )
        df_m["Member"] = ind
        df_list.append(df_m)

    df_single = pd.concat(df_list)
    df_single.index.rename("time", inplace=True)
    df_single = df_single.melt(id_vars=["Member"])

    # median data --------------------------------------------------
    m_data = np.array(
        [
            ds_rms_pixel.pcc.quantile(0.5, "nfields").values,
            ds_rms_pixel.rmse.quantile(0.5, "nfields").values,
            ds_rms_pixel.bias.quantile(0.5, "nfields").values,
        ]
    ).T
    df_median = pd.DataFrame(
        data=m_data, index=ds_rms_pixel.time.values, columns=["PCC", "RMSE", "BIAS"]
    )
    df_median["Member"] = -99
    df_median.index.rename("time", inplace=True)
    df_median = df_median.melt(id_vars=["Member"])

    # combine -----------------------------------------------------
    df_pixel = pd.concat([df_single, df_median])

    # figure ============================================================
    resolution = 400
    plt.rcParams["font.size"] = 8
    fig_height = fig_width / (13 / 8)  # 16/9
    lw_zero = 1

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), tight_layout=True)

    # some shortcut to name colors
    col_ = ["orange"]
    for i in range(len(ds_rms_pixel.nfields)):
        col_.append("w")

    sns.boxplot(
        x="value",
        y="variable",
        hue="Member",
        data=df_pixel,
        palette=col_,  # , "goldenrod", "purple"],
        width=0.9,  ###### 0.5
        linewidth=1,  #### None
        flierprops=dict(
            markerfacecolor="white",
            markersize=0.5,  #### 1
            marker="D",
        ),
        ax=ax,
    )

    ax.axvline(
        1, ymin=0.666, ymax=1, c="grey", lw=lw_zero, ls="--", label="optimal", zorder=0
    )
    ax.axvline(0, ymin=0, ymax=0.666, c="grey", lw=lw_zero, ls="--", zorder=0)

    # only use the first elements in legend
    handles, labels = ax.get_legend_handles_labels()
    l = ax.legend(
        handles[:3],
        ["optimal", "eRM", "sRM"],  # labels[:3],
        loc="upper right",
        borderaxespad=0.0,
    )

    # ax.legend(loc="upper right")
    # ax.legend([],[], frameon=False)

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xlim((-1, 2.5))
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(["PCC", "RMSE", "BIAS"])
    ax.set_yticks([], minor=True)

    if outfolder is not None:
        fig.savefig(outfolder + "Pixelbased_boxplot_summer_answer.jpg", dpi=resolution)


def various_combinations_of_mean_fields(df_median, outfolder, metric):
    """
    Figures used in answer to reviewers
    """

    fig, ax = plt.subplots(figsize=(5.5, 3), tight_layout=True)

    sns.boxplot(
        x="Size",
        y="Value",
        hue="Parameter",
        data=df_median,
        # color="k",
        width=0.5,
        linewidth=0.5,
        flierprops=dict(
            markerfacecolor="white",
            # markeredgecolor="black",
            markersize=0.1,
            marker="D",
        ),
        ax=ax,
    )

    ax.set_xticks([], minor=True)

    if metric == "SAL":
        ax.set_ylabel("SAL Metrics")
    elif metric == "pixel":
        ax.set_ylabel("Standard Indices")

    ax.set_xlabel("Number of RM ensemble members")

    # FOUND
    for i, artist in enumerate(ax.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        col = lighten_color(artist.get_facecolor(), 1.2)
        artist.set_edgecolor(col)

        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)
            line.set_linewidth(0.5)  # ADDITIONAL ADJUSTMENT

    if outfolder is not None:
        if metric == "SAL":
            fig.savefig(outfolder + "SAL_mean_fields_answer.jpg", dpi=400)
        elif metric == "pixel":
            fig.savefig(outfolder + "Pixelbased_mean_fields_answer.jpg", dpi=400)


import colorsys
import matplotlib.colors as mc


def lighten_color(color, amount=0.5):
    # --------------------- SOURCE: @IanHincks ---------------------
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def Radolan_coverage_answer(
    dist_mask, rad_mask, cmap_binary, border_data, fig_width=5.5, outfolder=None
):
    """
    Figure used in answer to reviewers
    """

    # calculations
    rad_nan_inside = dist_mask & ~rad_mask
    ex_123 = (
        rad_nan_inside.where(rad_nan_inside.sum(["y", "x"]) == 123, drop=True)
        .isel(time=0)
        .astype(bool)
    )
    # ex_2804 = rad_nan_inside.where(rad_nan_inside.sum(["y","x"]) == 2804, drop=True).isel(time=0).astype(bool)
    ex_10228 = (
        rad_nan_inside.where(rad_nan_inside.sum(["y", "x"]) == 10228, drop=True)
        .isel(time=0)
        .astype(bool)
    )

    # plot
    plot_array = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
        ]
    )

    plot_array = plot_array[:, :14]

    hspace = tuple(np.ones(plot_array.shape[0] - 1) * 0.0)  # 0.05
    wspace = tuple(np.ones(plot_array.shape[1] - 1) * 0.0)  # 0.05

    ymin = 0
    ymax = 900
    xmin = 200
    xmax = 900

    # plot ____________
    mapsz = 7
    # fontsz = 14

    # options

    cbar_width = 0.05
    lw_border = 0.2
    ifield = 0
    fontsz_small = 6
    plt.rcParams["font.size"] = 8  # 4
    resolution = 400

    # .94 / .93
    fig_height = (1 / 0.95) * fig_width * (plot_array.shape[0] / plot_array.shape[1])

    fig, axs = pplt.subplots(
        plot_array,
        figsize=(fig_width, fig_height),
        hspace=hspace,
        wspace=wspace,
        # space=0.1,
        sharey=0,
        sharex=0,
    )

    # map plots -----------------

    mapi = 0
    im = axs[mapi].pcolormesh(
        ex_123,
        # levels=clevs_rain,
        cmap=cmap_binary,
        alpha=0.3,
        zorder=0,
    )

    axs[mapi].set_title("Usual situation\n661 time steps affected\n123 missing values")
    plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

    axs[mapi].axvspan(6666, 6666, color="#818181", label="Missing Values")
    axs[mapi].legend(loc="ur")

    mapi += 1
    im = axs[mapi].pcolormesh(
        ex_10228,
        # levels=cLevels,
        cmap=cmap_binary,
        extend="both",
    )

    axs[mapi].set_title(
        "Situation with most gaps\n2 time steps affected\n10228 missing values"
    )
    plot_German_border(axs[mapi], border_data=border_data, linewidth=lw_border)

    # ================================
    # settings for all maps
    axs[:3].format(
        ylim=[ymin, ymax],
        xlim=[xmin, xmax],
        yticks=[],
        xticks=[],
        ylabel="",
        xlabel="",
        aspect="equal",
    )

    if outfolder is not None:
        fig.savefig(outfolder + "ex_rad_nans.jpg", dpi=resolution)


# ============================================================
# Colormap and other helper functions


def plot_German_border(ax, border_data, c="grey", linewidth=None, shift=[0, 0]):

    df_shape_ger = pd.read_csv(border_data, index_col=0)

    c = "k"
    linewidth = 0.3
    ls = "-"

    # constants of RADOLAN projection
    xstart = -523.4621669218558
    ystart = -4658.644724265572

    for part, df_part in df_shape_ger.groupby("part"):
        border_y = df_part.y_dwd.values / 1000 - ystart - shift[0]
        border_x = df_part.x_dwd.values / 1000 - xstart - shift[1]

        if linewidth is None:
            ax.plot(
                border_x,
                border_y,
                c=c,
                linestyle=ls,
            )
        else:
            ax.plot(border_x, border_y, c=c, linestyle=ls, lw=linewidth)


def colormap(
    cLevels=[0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    extend=2,
    belowColor=None,
    aboveColor=None,
    baseCmp=mpl.cm.YlGnBu,
):
    """
    Returns a colormap that is based on a standard map, with optional extra colors at beginning and end.

    extend: If colors are needed for values below and above range defined by cLevels, extend should be 2;
            if colors are needed for either below or above the range then extend=1;
            if only values within range are considered then extend=0.
            Extend does not make belowColor or aboveColor mandatory:
            if those are None, basemap colors are used so that they extend the range.

    belowColor and aboveColor: Give colors as hex value
    """

    # if specific edge colors are given, less base map colors are needed
    for edgeC in [belowColor, aboveColor]:
        if edgeC is not None:
            extend -= 1

    # number of colors that need to be filled by baseCmp.
    nBaseColors = len(cLevels) - 1 + extend

    # list of hex color codes of baseCmp.
    precipColors = []
    for p in np.linspace(0, 1, nBaseColors):
        rgb = baseCmp(p)[:3]
        precipColors.append(mpl.colors.rgb2hex(rgb))

    # add specific edge values
    if belowColor is not None:
        precipColors.insert(0, belowColor)
    if aboveColor is not None:
        precipColors.append(aboveColor)

    # transform to cmap
    cmap = ListedColormap(precipColors)

    return cmap


def adjusted_colormap(ds_t_subset):

    baseCmp = mpl.cm.Blues(np.arange(256))[50:]
    baseCmp = ListedColormap(baseCmp, name="myColorMap", N=baseCmp.shape[0])

    # max value
    max_rain = np.rint(ds_t_subset.rain.max().values * 100) / 100

    cLevels_subset = np.linspace(max_rain / 10, max_rain, 10)
    cMapRain_subset = colormap(
        cLevels_subset, extend=2, belowColor="#FFFFFF", baseCmp=baseCmp
    )

    return cLevels_subset, cMapRain_subset


def transform_to_linear_colormap(values, cLevels, cMap):
    rTrans = np.ones((len(values), 4))

    # for values within bounds of lowest and highest level
    N = len(cLevels) - 1
    for i in range(N):
        ind = np.asarray((values >= cLevels[i]) & (values < cLevels[i + 1])).nonzero()[
            0
        ]
        rTrans[ind, :] = np.array(cMap(i / N))

    # for values at the edges
    ind = np.asarray(values < cLevels[0]).nonzero()[0]
    rTrans[ind, :] = np.array(cMap(0))

    ind = np.asarray(values >= cLevels[-1]).nonzero()[0]
    rTrans[ind, :] = np.array(cMap((N - 1) / N))

    return rTrans


def transform_to_linear_colormap_old(rain, cLevels, cMap, nan_grey=True):

    rTrans = np.ones((len(rain), 4))

    # for values within bounds of lowest and highest level
    for i in range(len(cLevels) - 1):
        ind = np.asarray((rain >= cLevels[i]) & (rain < cLevels[i + 1])).nonzero()[0]
        rTrans[ind, :] = np.array(cMap(i + 1))

    # for values at the edges
    ind = np.asarray(rain < cLevels[0]).nonzero()[0]
    rTrans[ind, :] = np.array(cMap(0.0))

    ind = np.asarray(rain >= cLevels[-1]).nonzero()[0]
    rTrans[ind, :] = np.array(cMap(1.0))

    # handle nans
    ind = np.asarray(np.isnan(rain)).nonzero()[0]
    if nan_grey:
        rTrans[ind, :] = np.array([0.5, 0.5, 0.5, 1])
    else:
        rTrans[ind, :] = np.array(cMap(0.0))

    return rTrans


def getcmap():

    # boundaries of colormap
    clevels = [0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
    # clevs_rain = np.logspace(0.1,20,12)

    cmap = plt.cm.get_cmap("YlGnBu", len(clevels) - 1)

    cmap.set_under("#ef0000")
    cmap.set_over("navy")
    #     cm = cmap_rain(range(-1,len(clevs_rain)))

    return cmap, clevels


def find_SAL_data_range(ds_SAL_list):

    # find data range
    Smax, Amax, Lmax = 0, 0, 0
    for ds in ds_SAL_list:

        if np.nanmax(np.abs(ds.S.values)) > Smax:
            Smax = np.nanmax(np.abs(ds.S.values))
        if np.nanmax(np.abs(ds.A.values)) > Amax:
            Amax = np.nanmax(np.abs(ds.A.values))
        if ds.L.max().values > Lmax:
            Lmax = ds.L.max().values

    Smax = 1.1 * Smax
    Amax = 1.1 * Amax
    Lmax = 1.1 * Lmax

    Smin = Smax * -1
    Amin = Amax * -1
    Lmin = 0

    axlims = {"S": [Smin, Smax], "A": [Amin, Amax], "L": [Lmin, Lmax]}

    return axlims


# ================================================================
# Plots NOT in paper


def parameters_for_mean_fields_old(
    df_mean_SAL, df_mean_pixel, fig_width=5.5, outfolder=None
):
    """
    Old version of Figure 8
    """

    resolution = 400
    small_font = 5
    plt.rcParams["font.size"] = 8  # 6

    fig_height = fig_width / (18 / 9)
    fig, ax = pplt.subplots(figsize=(fig_width, fig_height))

    ensSizes = [1, 3, 5, 10, 20]

    c_S = [0.6, 0.6, 1, 1]
    c_A = [0.4, 0.4, 0.7, 1]
    c_L = [0.1, 0.1, 0.5, 1]
    c_pcc = [1, 0.6, 0.6, 1]
    c_rmse = [0.7, 0.4, 0.4, 1]
    c_bias = [0.5, 0.1, 0.1, 1]

    rmls = "-"
    krils = ":"

    ax.plot(ensSizes, df_mean_SAL["S"][:-1], c=c_S, label="S (RM)")
    ax.plot(ensSizes, df_mean_SAL["A"][:-1], c=c_A, label="A (RM)")
    ax.plot(ensSizes, df_mean_SAL["L"][:-1], c=c_L, label="L (RM)")

    ax.plot(ensSizes, df_mean_pixel["pcc"][:-1], c=c_pcc, label="PCC (RM)")
    ax.plot(ensSizes, df_mean_pixel["rmse"][:-1], c=c_rmse, label="RMSE (RM)")
    ax.plot(ensSizes, df_mean_pixel["bias"][:-1], c=c_bias, label="BIAS (RM)")

    ax.plot(
        ensSizes, np.ones(5) * df_mean_SAL["S"][-1], c=c_S, ls=krils, label="S (KRI)"
    )
    ax.plot(
        ensSizes,
        np.ones(5) * df_mean_SAL["A"][-1],
        c=c_A,
        lw=2,
        ls=krils,
        label="A (KRI)",
    )
    ax.plot(
        ensSizes, np.ones(5) * df_mean_SAL["L"][-1], c=c_L, ls=krils, label="L (KRI)"
    )

    ax.plot(
        ensSizes,
        np.ones(5) * df_mean_pixel["pcc"][-1],
        c=c_pcc,
        ls=krils,
        label="PCC (KRI)",
    )
    ax.plot(
        ensSizes,
        np.ones(5) * df_mean_pixel["rmse"][-1],
        c=c_rmse,
        ls=krils,
        label="RMSE (KRI)",
    )
    ax.plot(
        ensSizes,
        np.ones(5) * df_mean_pixel["bias"][-1],
        c=c_bias,
        ls=krils,
        label="BIAS (KRI)",
    )

    # adjustments
    ax.legend(ncols=6, loc="lr", fontsize=small_font)
    ax.set_xticks(ensSizes)
    ax.set_xticks([], minor=True)
    ax.set_xlabel("Number of RM ensemble members")
    ax.set_xlim((1, 20))
    ax.set_ylim((-0.4, 0.8))
    ax.set_ylabel("SAL, PCC, BIAS [-], and RMSE [mm]")

    if outfolder is not None:
        fig.savefig(outfolder + "meanFieldAnalysis_1.jpg", dpi=resolution)


def boxplot_PCC_S(
    ds_rms_pixel,
    ds_rmm_20_pixel,
    ds_kri_pixel,
    ds_rms_SAL,
    ds_rmm_20_SAL,
    ds_kri_SAL,
    fig_width=5.5,
    outfolder=None,
):

    # Calculations =================================================
    # RM single fields
    rms_data = np.array(
        [
            ds_rms_SAL.S.values,
            ds_rms_pixel.pcc.mean("nfields").values,
            # ds_rms_pixel.bias.mean("nfields").values,
        ]
    ).T
    df_rms = pd.DataFrame(
        data=rms_data,
        index=ds_rms_pixel.time.values,
        columns=[
            "PCC",
            # "BIAS",
            "S",
        ],
    )
    df_rms["Reconstruction"] = "RM single"

    # RM mean fields
    rmm_data = np.array(
        [
            ds_rmm_20_SAL.S.values,
            ds_rmm_20_pixel.pcc.values,
            # ds_rmm_20_pixel.bias.values,
        ]
    ).T
    df_rmm = pd.DataFrame(
        data=rmm_data,
        index=ds_rmm_20_pixel.time.values,
        columns=[
            "PCC",
            # "BIAS",
            "S",
        ],
    )
    df_rmm["Reconstruction"] = "RM ens. mean"

    # Kriging
    kri_data = np.array(
        [
            ds_kri_SAL.S.values,
            ds_kri_pixel.pcc.values,
            # ds_kri_pixel.bias.values,
        ]
    ).T
    df_kri = pd.DataFrame(
        data=kri_data,
        index=ds_kri_pixel.time.values,
        columns=[
            "PCC",
            # "BIAS",
            "S",
        ],
    )
    df_kri["Reconstruction"] = "Kriging"

    # combine
    df = pd.concat([df_rms, df_rmm, df_kri])
    df.index.rename("time", inplace=True)
    df = df.melt(id_vars=["Reconstruction"])

    # figure ============================================================
    resolution = 400
    plt.rcParams["font.size"] = 12
    fig_height = fig_width / (16 / 9)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), tight_layout=True)

    ax = sns.boxplot(
        x="value",
        y="variable",
        hue="Reconstruction",
        data=df,
        palette=["orange", "goldenrod", "purple"],
        width=0.5,  ###### 0.5
        linewidth=1,  #### None
        flierprops=dict(
            markerfacecolor="white",
            markersize=0.5,  #### 1
            marker="D",
        ),
    )

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xlim((-2, 2))
    ax.set_xticks(np.arange(-2, 2.1, 1))
    ax.set_yticks(np.arange(2))
    ax.set_yticklabels(
        [
            "Structure Error",
            "Pearson Correlation\n(PCC)",
            # "BIAS",
        ]
    )
    ax.set_yticks([], minor=True)

    if outfolder is not None:
        fig.savefig(outfolder + "boxplot_PCC_S.jpg", dpi=resolution)


def SAL_six_scatter(
    ds_rms_SAL, ds_kri_SAL, ref_rain, da_tmask, fig_width=5.5, outfolder=None
):

    # some figure options
    ec = None
    lw = 0.5
    lw_zero = 0.5
    s = 1
    alpha_low = 0.8
    alpha_middle = 0.8
    alpha_high = 0.8
    mrk = "o"
    resolution = 400
    plt.rcParams["font.size"] = 8
    fig_height = fig_width / 2

    hspace = 0
    wspace = tuple([0, 0.1, 0])

    fig, axs = pplt.subplots(
        ncols=4,
        nrows=2,
        figsize=(fig_width, fig_height),
        sharey=3,
        sharex=3,
        spany=0,
        spanx=0,
        hspace=hspace,
        wspace=wspace,
        # space=0.1,
    )

    # for length of colorbar
    x0pos = axs[0, 1].get_window_extent().x0
    x1pos = axs[0, 1].get_window_extent().x1
    cbar_length = 0.9 * (x1pos - x0pos) / 100

    # find data range
    axlims = {"S": [-2, 2], "A": [-2, 2], "L": [0, 1]}

    ref_rain = ref_rain.where(da_tmask, drop=True)
    # special colormap for categorial coloring
    cLevels = [
        0,
        ref_rain.quantile(0.25).values,
        ref_rain.quantile(0.75).values,
        ref_rain.max(),
    ]

    # separation of different quantiles
    cond_low = ref_rain < cLevels[1]
    cond_middle = (ref_rain >= cLevels[1]) & (ref_rain <= cLevels[2])
    cond_high = ref_rain > cLevels[2]
    ref_rain_low = ref_rain.where(cond_low, drop=True)
    ref_rain_middle = ref_rain.where(cond_middle, drop=True)
    ref_rain_high = ref_rain.where(cond_high, drop=True)
    ds_rms_SAL_low = ds_rms_SAL.where(cond_low, drop=True)
    ds_rms_SAL_middle = ds_rms_SAL.where(cond_middle, drop=True)
    ds_rms_SAL_high = ds_rms_SAL.where(cond_high, drop=True)
    ds_kri_SAL_low = ds_kri_SAL.where(cond_low, drop=True)
    ds_kri_SAL_middle = ds_kri_SAL.where(cond_middle, drop=True)
    ds_kri_SAL_high = ds_kri_SAL.where(cond_high, drop=True)

    # cMap = ListedColormap(['#ffd700', '#648c11', '#0000cd']) original
    cMap = ListedColormap(["#d90101", "#49dd48", "#013dd9"])

    for i, p1 in enumerate(["S", "A"]):

        for j, p2 in enumerate(["A", "L"]):

            if p1 == p2:

                im = axs[j, i].scatter(
                    np.arange(3) + 666,
                    np.arange(3),
                    c=np.arange(3),
                    alpha=alpha_middle,
                    levels=[0, 0.25, 0.75, 1],
                    cmap=cMap,
                    # vmin=cmin, vmax=cmax,
                    colorbar="ur",
                    colorbar_kw={
                        "length": cbar_length,  #'%iem'%cbar_length,
                        "label": "Quantiles of\nmean reference rain",
                        #'align': 'center',
                        "ticks": [[0, 0.25, 0.75, 1]],
                        # 'drawedges': False,
                        "linewidth": 0,
                    },
                )

                axs[j, i].axis("off")
                axs[j, i + 2].axis("off")
                continue

            # main part
            im = axs[j, i + 2].scatter(
                ds_rms_SAL_middle[p1].values,
                ds_rms_SAL_middle[p2].values,
                c=ref_rain_middle.values,
                alpha=alpha_middle,
                s=s,
                ec=ec,
                marker=mrk,
                levels=cLevels,
                cmap=cMap,
                lw=lw,
            )

            im = axs[j, i + 2].scatter(
                ds_rms_SAL_low[p1].values,
                ds_rms_SAL_low[p2].values,
                c=ref_rain_low.values,
                alpha=alpha_low,
                s=s,
                ec=ec,
                marker=mrk,
                levels=cLevels,
                cmap=cMap,
                lw=lw,
            )

            im = axs[j, i + 2].scatter(
                ds_rms_SAL_high[p1].values,
                ds_rms_SAL_high[p2].values,
                c=ref_rain_high.values,
                alpha=alpha_high,
                s=s,
                ec=ec,
                marker=mrk,
                levels=cLevels,
                cmap=cMap,
                lw=lw,
            )

            im = axs[j, i].scatter(
                ds_kri_SAL_middle[p1].values,
                ds_kri_SAL_middle[p2].values,
                c=ref_rain_middle.values,
                alpha=alpha_middle,
                s=s,
                ec=ec,
                marker=mrk,
                levels=cLevels,
                cmap=cMap,
                lw=lw,
            )

            im = axs[j, i].scatter(
                ds_kri_SAL_low[p1].values,
                ds_kri_SAL_low[p2].values,
                c=ref_rain_low.values,
                alpha=alpha_low,
                s=s,
                ec=ec,
                marker=mrk,
                levels=cLevels,
                cmap=cMap,
                lw=lw,
            )

            im = axs[j, i].scatter(
                ds_kri_SAL_high[p1].values,
                ds_kri_SAL_high[p2].values,
                c=ref_rain_high.values,
                alpha=alpha_high,
                s=s,
                ec=ec,
                marker=mrk,
                levels=cLevels,
                cmap=cMap,
                lw=lw,
            )

            # adjustments
            axs[j, i + 2].set_xlim(axlims[p1])
            axs[j, i + 2].set_ylim(axlims[p2])
            axs[j, i + 2].set_xlabel(p1)
            axs[j, i + 2].set_ylabel(p2)
            axs[j, i + 2].set_xticks([-1, 0, 1])
            if p2 == "A":
                axs[j, i + 2].set_yticks([-1, 0, 1])
            elif p2 == "L":
                axs[j, i + 2].set_yticks([0, 0.5])
            axs[j, i + 2].set_xticks([], minor=True)
            axs[j, i + 2].set_yticks([], minor=True)

            # zero lines
            axs[j, i + 2].plot(axlims[p1], [0, 0], c="grey", lw=lw_zero)
            axs[j, i + 2].plot([0, 0], axlims[p2], c="grey", lw=lw_zero)

            # adjustments
            axs[j, i].set_xlim(axlims[p1])
            axs[j, i].set_ylim(axlims[p2])
            axs[j, i].set_xlabel(p1)
            axs[j, i].set_ylabel(p2)
            axs[j, i].set_xticks([-1, 0, 1])
            if p2 == "A":
                axs[j, i].set_yticks([-1, 0, 1])
            elif p2 == "L":
                axs[j, i].set_yticks([0, 0.5])
            axs[j, i].set_xticks([], minor=True)
            axs[j, i].set_yticks([], minor=True)

            # zero lines
            axs[j, i].plot(axlims[p1], [0, 0], c="grey", lw=lw_zero)
            axs[j, i].plot([0, 0], axlims[p2], c="grey", lw=lw_zero)

            # titles
            axs[j, i].format(ultitle="KRI")
            axs[j, i + 2].format(ultitle="eRM")

    axs[1, 1].tick_params(left=False)
    axs[1, 3].tick_params(left=False)
    axs[0, 0].tick_params(bottom=False)
    axs[0, 2].tick_params(bottom=False)

    # axs.format(
    # suptitle='%s'%title,
    #     aspect='equal',
    # )

    fig.align_ylabels()

    if outfolder is not None:
        fig.savefig(outfolder + "SAL_diagram_crain_both.jpg", dpi=resolution)


def SAL_triple_contour(ds_rms_SAL, ds_kri_SAL, fig_width=5.5, outfolder=None):
    """
    SAL diagram with contours / scatter density for S A and L combined
    """

    # some figure options
    colors = {"rms": "orange", "kri": "purple"}
    lnstyle = {"rms": "-", "kri": ":"}
    ec = "k"
    lw = 0.5
    lw_zero = 0.5
    s = 100
    alpha = 0.75
    resolution = 400
    plt.rcParams["font.size"] = 4
    fig_height = fig_width

    hspace = 0
    wspace = 0

    fig, ax = pplt.subplots(
        ncols=2,
        nrows=2,
        figsize=(fig_width, fig_height),
        sharey=3,
        sharex=3,
        spany=0,
        spanx=0,
        hspace=hspace,
        wspace=wspace,
    )

    # find data range
    # axlims = find_SAL_data_range([ds_rms_SAL, ds_kri_SAL])
    axlims = {"S": [-2, 2], "A": [-2, 2], "L": [0, 1]}

    for i, p1 in enumerate(["S", "A"]):

        for j, p2 in enumerate(["A", "L"]):

            if p1 == p2:
                ax[j, i].set_visible(False)
                continue

            # main part
            for ds, k in zip([ds_rms_SAL, ds_kri_SAL], ["rms", "kri"]):

                sns.kdeplot(
                    x=ds[p1].values,
                    y=ds[p2].values,
                    #                              label = k,
                    levels=[0.25, 0.5, 0.75, 1],
                    fill=True,
                    alpha=alpha,
                    color=colors[k],
                    lw=lw,
                    ls=lnstyle[k],
                    ax=ax[j, i],
                )

                # adjustments
                ax[j, i].set_xlim(axlims[p1])
                ax[j, i].set_ylim(axlims[p2])
                ax[j, i].set_xlabel(p1)
                ax[j, i].set_ylabel(p2)
                # ax[j, i].set_xticks(np.linspace(axlims[p1][0], axlims[p1][1], 5))
                # ax[j, i].set_yticks(np.linspace(axlims[p2][0], axlims[p2][1], 5))
                ax[j, i].set_xticks([-1, 0, 1])
                if p2 == "A":
                    ax[j, i].set_yticks([-1, 0, 1])
                elif p2 == "L":
                    ax[j, i].set_yticks([0, 0.5])
                # ax[j, i].set_yticks([], minor=True)
                # ax[j, i].set_xticks([], minor=True)
                # ax.set_aspect('equal', 'box')

            # zero lines
            ax[j, i].plot(axlims[p1], [0, 0], c="grey", lw=lw_zero)
            ax[j, i].plot([0, 0], axlims[p2], c="grey", lw=lw_zero)

    ax[1, 1].tick_params(left=False)
    ax[0, 0].tick_params(bottom=False)

    # legend
    ax[0, 0].plot([], [], c="k", lw=lw, linestyle="-", label="RM")
    ax[0, 0].plot([], [], c="k", lw=lw, linestyle=":", label="KRI")
    _ = ax[0, 0].legend(ncols=2)

    ax.set_yticks([], minor=True)
    ax.set_xticks([], minor=True)

    fig.align_ylabels()

    if outfolder is not None:
        fig.savefig(outfolder + "SAL_diagram_contour_summer.jpg", dpi=resolution)


def binary_colormaps():

    # binary colormaps
    cmap_binary_wb = ["#FFFFFF", "#000000"]
    cmap_binary_wg = ["#FFFFFF", "#d3d3d3"]  # "#818181"]
    cmap_binary_wo = ["#FFFFFF", "#ef9400"]

    return cmap_binary_wb, cmap_binary_wg, cmap_binary_wo


def SAL_boxplots(ds_rms_SAL, ds_rmm_SAL, ds_kri_SAL, fig_width=5.5, outfolder=None):

    # Calculations ================================================
    # RM single fields
    rms_data = np.array(
        [
            ds_rms_SAL.S.values,
            ds_rms_SAL.A.values,
            ds_rms_SAL.L.values,
            # ds_rms_SAL.Q.values
        ]
    ).T
    df_rms_sal = pd.DataFrame(
        data=rms_data, index=ds_rms_SAL.time.values, columns=["S", "A", "L"]
    )  # , "|SAL|"])
    df_rms_sal["Reconstruction"] = "RM"

    # Kriging
    kri_data = np.array(
        [
            ds_kri_SAL.S.values,
            ds_kri_SAL.A.values,
            ds_kri_SAL.L.values,
            # ds_kri_SAL.Q.values
        ]
    ).T
    df_kri_sal = pd.DataFrame(
        data=kri_data, index=ds_kri_SAL.time.values, columns=["S", "A", "L"]
    )  # , '|SAL|'])
    df_kri_sal["Reconstruction"] = "KRI"

    # combine
    #     df_sal = df_rms_sal.append(df_rmm_sal.append(df_kri_sal))
    df_sal = pd.concat([df_rms_sal, df_kri_sal])
    df_sal.index.rename("time", inplace=True)
    df_sal = df_sal.melt(id_vars=["Reconstruction"])

    # figure ================================================
    resolution = 400
    plt.rcParams["font.size"] = 8
    fig_height = fig_width / (16 / 9)
    # fig_height = fig_width

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), tight_layout=True)

    ax = sns.boxplot(
        x="value",
        y="variable",
        hue="Reconstruction",
        data=df_sal,
        palette=["orange", "purple"],
        width=0.5,
        linewidth=1,
        flierprops=dict(
            markerfacecolor="white",
            markersize=0.5,
            marker="D",
        ),
    )

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(
        # ["Structure (S)", "Amplitude (A)", "Location (L)"]
        ["S", "A", "L"]
    )  # , "Combined (|SAL|)"])
    ax.set_yticks([], minor=True)
    ax.set_xlim((-2, 2))

    ax.axvline(x=0, c="k", lw=1)

    ax.legend(loc="lower left")

    if outfolder is not None:
        fig.savefig(outfolder + "SAL_boxplot_summer.jpg", dpi=resolution)
