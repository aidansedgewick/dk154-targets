import copy
import inspect
import logging
import traceback
#from dataclasses import dataclass # >=py3.7 only!
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.visualization import ZScaleInterval

from astroplan import Observer

from dk154_targets.queries import FinkQuery

logger = logging.getLogger(__name__.split(".")[-1])

lc_gs = plt.GridSpec(3,4)
zscaler = ZScaleInterval()

class TargetData:
    def __init__(
        self, 
        lightcurve: pd.DataFrame=None, 
        probabilities: pd.DataFrame=None, 
        parameters: dict=None, 
        cutouts: dict=None, 
        meta: dict=None
    ):
        self.lightcurve = lightcurve
        self.probabilities = probabilities
        self.parameters = parameters or {}
        self.cutouts = cutouts or {}
        self.meta = dict(
            #lightcurve_update_time=None,
            #probabilities_update_time=None,
            #parameters_update_time=None,
            #cutout_update_time=None, 
        )
        meta = meta or {}
        self.meta.update(meta)

class Target:

    default_base_score = 100.

    def __init__(
        self, 
        objectId: str, 
        ra: float, 
        dec: float,
        fink_lightcurve: pd.DataFrame=None,
        alerce_lightcurve: pd.DataFrame=None,
        alerce_probabilities = None,
        base_score: float=None,
        meta: dict=None,
    ):
        #===== basics
        self.objectId = objectId
        self.ra = ra
        self.dec = dec
        self.coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        self.base_score = base_score or self.default_base_score

        if (not np.isfinite(self.ra)) or (not np.isfinite(self.dec)):
            raise ValueError(f"ra and dec should not be {self.ra}, {self.dec}")

        #===== keep track of target data 
        self.fink_data = TargetData(lightcurve=fink_lightcurve)
        self.alerce_data = TargetData(lightcurve=alerce_lightcurve, probabilities=alerce_probabilities)
        self.atlas_data = TargetData()
        self.tns_data = TargetData()

        #self.cutouts = {}
        #self.cutout_update_time = None

        self.target_history = self.compile_target_history()

        #===== models
        self.models = []
        self.updated = True
        self.target_of_opportunity = False

        #===== rank/score history
        self.score_history = {"no_observatory": []}
        self.rank_history = {"no_observatory": []}
        self.reset_target_figures()
        self.last_score_comments = {}
        self.reject_comments = None

        #===== meta
        meta = meta or {}
        self.meta = {
            "init_time": Time.now()
        }
        self.meta.update(meta)

        logger.debug(f"{self.objectId} initalised")


    @classmethod
    def from_fink_query(cls, objectId, ra=None, dec=None, base_score=None, **kwargs):
        fink_lightcurve = FinkQuery.query_objects(objectId=objectId, **kwargs)
        if fink_lightcurve is None:
            logger.warn(f"no fink_lightcurve {objectId} query")
            return None
        if isinstance(fink_lightcurve, pd.DataFrame) and fink_lightcurve.empty:
            logger.warning(f"fink data is None")
            return None
        return cls.from_fink_lightcurve(
            objectId, fink_lightcurve, ra=ra, dec=dec, base_score=base_score
        )


    @classmethod
    def from_fink_lightcurve(
        cls, objectId, fink_lightcurve, ra=None, dec=None, base_score=None
    ):
        if isinstance(fink_lightcurve, str) or isinstance(fink_lightcurve, Path):
            logger.info(f"interpret {fink_lightcurve} as path")
            fink_lightcurve = pd.read_csv(fink_lightcurve)

        fink_lightcurve = fink_lightcurve.copy(deep=True)
        fink_lightcurve.sort_values("jd", inplace=True)
        if "tag" in fink_lightcurve.columns:
            detections = fink_lightcurve.query("tag=='valid'")
        else:
            detections = fink_lightcurve
        if detections.empty:
            logger.warning(f"init {objectId}: no valid detections!")
            return None
        if ra is None or dec is None:
            ra = fink_lightcurve["ra"].dropna().values[-1]
            dec = fink_lightcurve["dec"].dropna().values[-1]
        if (not np.isfinite(ra)) or (not np.isfinite(dec)):
            logger.warning(f"ra and dec should not be {ra}, {dec}")
            return None
        target = cls(objectId, ra, dec, fink_lightcurve=fink_lightcurve, base_score=base_score)
        return target

    @classmethod
    def from_alerce_lightcurve(
        cls, objectId, alerce_lightcurve, ra=None, dec=None, base_score=None
    ):
        alerce_lightcurve = alerce_lightcurve.copy(deep=True)
        alerce_lightcurve.sort_values("jd", inplace=True)
        if "tag" in alerce_lightcurve.columns:
            detections = alerce_lightcurve.query("tag=='valid'")
        else:
            detections = alerce_lightcurve
        if detections.empty:
            logger.warning(f"init {objectId}: no valid detections")
            return None
        if ra is None or dec is None:
            ra = alerce_lightcurve["ra"].dropna().values[-1]
            dec = alerce_lightcurve["dec"].dropna().values[-1]
        if (not np.isfinite(ra)) or (not np.isfinite(dec)):
            logger.warning(f"ra and dec should not be {ra}, {dec}")
            return None
        target = cls(objectId, ra, dec, alerce_lightcurve=alerce_lightcurve, base_score=base_score)
        return target

    def evaluate_target(
        self, scoring_function: Callable, observatory: Observer, **kwargs
    ):
        obs_name = getattr(observatory, "name", "no_observatory")
        if obs_name == "no_observatory":
            assert observatory is None
        t_ref = kwargs.get("t_ref", Time.now()) 
        if not isinstance(t_ref, Time):
            raise ValueError(f"t_ref should be astropy.time.Time, not {type(t_ref)}")

        scoring_function_message = (
            "`scoring_function` should accept two arguments `target` and `observatory` "
            "(which could be `None`), and should return float and optionally two lists of strings."
        )

        scoring_result = scoring_function(self, observatory, **kwargs)
        if isinstance(scoring_result, tuple):
            if len(scoring_result) != 3:
                raise ValueError(scoring_function_message)
            score, score_comments, reject_comments = scoring_result
            self.last_score_comments[obs_name] = score_comments
            self.reject_comments = reject_comments
        elif isinstance(scoring_result, (float, int)):
            score = scoring_result
            score_comments = []
            reject_comments = []
        else:
            raise ValueError(scoring_function_message)

        if not isinstance(score, (float, int)):
            raise ValueError(scoring_function_message)

        if obs_name not in self.score_history:
            self.score_history[obs_name] = []
        assert obs_name in self.score_history
        self.score_history[obs_name].append((score, t_ref))
        return score, score_comments, reject_comments


    def get_last_score(self, obs_name, return_time=False):
        if not isinstance(obs_name, str):
            observatory = obs_name
            obs_name = getattr(observatory, "name", "no_observatory")
            if obs_name == "no_observatory":
                assert observatory is None
            else:
                assert isinstance(observatory, Observer)

        if len(self.score_history[obs_name]) == 0:
            return None
            
        if return_time:
            return self.score_history[obs_name][-1]
        else:
            return self.score_history[obs_name][-1][0]


    def get_description(self, t_ref: Time=None):
        t_ref = t_ref or Time.now()
        
        msg_components = [
            f"{self.objectId}\n",
            f"ra={self.ra:.6f} dec={self.dec:.5f}",
        ]
        if self.target_history is not None:
            N_obs = {}
            for band, band_history in self.target_history.groupby("band"):
                if "tag" in band_history:
                    band_detections = band_history.query("(tag=='valid') or (tag=='badquality')")
                else:
                    band_detections = band_history
                N_obs[band] = len(band_detections)
            detection_str = f"detections: " + " ".join(f"{band}={N}" for band, N in N_obs.items())
            msg_components.append(detection_str)
        msg_components.append(f"https://fink-portal.org/{self.objectId}")

        description = "\n".join(msgc for msgc in msg_components)
        return description
        

    def compile_target_history(
        self, ztf_source_priority=("fink", "alerce", ), t_ref=None,
    ):
        t_ref = t_ref or Time.now()

        ztf_band_lookup = {1: "ztfg", 2: "ztfr"}

        df_list = []

        ### sort ztf data
        #ztf_source = None
        for ztf_source in ztf_source_priority:
            source_data = getattr(self, f"{ztf_source}_data", None)
            if source_data is None:
                raise ValueError(f"{ztf_source} data should not be None!")
            ztf_lightcurve = source_data.lightcurve
            if ztf_lightcurve is not None:
                if not ztf_lightcurve.empty:
                    break

        if ztf_source == "fink":
            use_fink_cols = ["jd", "magpsf", "sigmapsf", "tag", "diffmaglim"]
            fink_cols = [c for c in use_fink_cols if c in self.fink_data.lightcurve.columns]
            fink_rename = {"magpsf": "mag", "sigmapsf": "magerr", "fid": "band"}
            with pd.option_context('mode.chained_assignment', None):
                fink_df = self.fink_data.lightcurve[fink_cols]
                fink_df.loc[:,"band"] = self.fink_data.lightcurve["fid"].map(ztf_band_lookup)
                fink_df.rename(fink_rename, axis=1, inplace=True)
                df_list.append(fink_df)
        elif ztf_source == "alerce":
            use_alerce_cols = ["jd", "magpsf", "sigmapsf", "tag", "diffmaglim"]
            alerce_cols = [c for c in use_alerce_cols if c in self.alerce_data.lightcurve.columns]
            alerce_rename = {"magpsf": "mag", "sigmapsf": "magerr", "fid": "band"}
            with pd.option_context('mode.chained_assignment', None):
                alerce_df = self.alerce_data.lightcurve[alerce_cols]
                alerce_df.loc[:,"band"] = self.alerce_data.lightcurve["fid"].map(ztf_band_lookup)
                alerce_df.rename(alerce_rename, axis=1, inplace=True)
                df_list.append(alerce_df)


        ### sort atlas data
        with pd.option_context('mode.chained_assignment', None):
            if self.atlas_data.lightcurve is not None and (not self.atlas_data.lightcurve.empty):
                atlas_band_lookup = {"o": "atlaso", "c": "atlasc"}

                atlas_cols = ["m", "dm", "mag5sig"]
                atlas_rename = {"m": "mag", "dm": "magerr", "mag5sig": "diffmaglim"}
                full_atlas_df = self.atlas_data.lightcurve[atlas_cols]
                full_atlas_df.reset_index(drop=True, inplace=True)
                full_atlas_df.loc[:,"band"] = self.atlas_data.lightcurve["F"].map(atlas_band_lookup)
                full_atlas_df.loc[:,"jd"] = Time(self.atlas_data.lightcurve["MJD"].values, format="mjd").jd

                # know snr~1/dm

                atlas_df = full_atlas_df.query("m > 0")
                atlas_df.reset_index(drop=True, inplace=True)
                tag_data = np.full(len(atlas_df), 'valid', dtype="object")
                # badqual_mask = (atlas_df["m"] < 0.)
                # tag_data[ badqual_mask ] = 'badquality'
                upperlim_mask = (atlas_df["m"] > atlas_df["mag5sig"])
                tag_data[ upperlim_mask ] = 'upperlim'
                atlas_df.loc[:,"tag"] = pd.Series(tag_data)

                atlas_df.rename(atlas_rename, axis=1, inplace=True)
                df_list.append(atlas_df)

        target_history = pd.concat(df_list)
        self.target_history = target_history.query(f"jd < @t_ref.jd")
        return


    def plot_lightcurve(self, t_ref=None, fig=None):
        #try:
        logger.info(f"lc for {self.objectId}")
        lc_fig = plot_lightcurve(self, t_ref=t_ref, fig=fig)
        self.latest_lc_fig = lc_fig
        return lc_fig
        #except Exception as e:
        #    logger.warning(f"NO LIGHTCURVE FOR {self.objectId}")
        #    print(e)
        #    return None


    def plot_observing_chart(self, observatory: Observer, t_ref: Time=None):
        obs_name = getattr(observatory, "name", "no_observatory")
        logger.info(f"oc for {self.objectId} {obs_name}")
        if observatory is not None:
            return plot_observing_chart(observatory, self, t_ref=t_ref)
        return None

    def reset_target_figures(self,):
        self.latest_lc_fig = None
        self.latest_oc_figs = []


def plot_lightcurve(
    target: Target, t_ref: Time=None, fig=None, forecast_days=5., **kwargs
):

    t_ref = t_ref or Time.now()
    if not isinstance(t_ref, Time):
        raise ValueError(f"t_ref should be astropy.time.Time, not {type(t_ref)}")
    xlabel = kwargs.get(
        "xlabel", 
        f"time before now ({t_ref.datetime.strftime('%Y-%m-%d %H:%M')})"
    )
    


    ##======== initialise figure
    if fig is None:
        fig = plt.figure(figsize=(8, 4.8))
        ax = fig.add_subplot(lc_gs[:,:-1])
    else:
        ax = fig.axes[0]

    if target.target_history is None:
        logger.warning("{target.objectId} - no ")
        return None
    full_detections = target.target_history
    time_grid = np.arange(full_detections["jd"].min()-5., t_ref.jd + forecast_days, 1.0)

    if not target.models:
        model = None
    else:
        model = target.models[-1]    
        
    brightest_values = [] # so we can adjust the axes ylimit if we need.
    legend_handles = []

    color_lookup = {"ztfg": "C0", "ztfr": "C1", "atlasc": "C2", "atlaso": "C3"}
    for ii, (band, band_history) in enumerate(target.target_history.groupby("band")):
        band_history.sort_values("jd", inplace=True)
        band_color = color_lookup.get(band, ii+len(color_lookup))
        label = f"{band[:-1].upper()}-" + r"$" + band[-1] + "$"
        if band.startswith("ztf"):
            zorder = 10
            alpha = 1.0
            ms=6
        else:
            zorder = 6
            alpha = 0.6
            ms=4

        if "tag" in band_history.columns:
            detections = band_history.query("tag=='valid'")
        else:
            detections = band_history
        if len(detections) == 0:
            logger.warning(f"{target.objectId} no detections for {band}")
            #continue      

        if "tag" in band_history.columns:
            detections = band_history.query("tag=='valid'")
            ulimits = band_history.query("tag=='upperlim'")
            badqual = band_history.query("tag=='badquality'")

            if not (len(detections) + len(ulimits) + len(badqual)) == len(band_history):
                logger.warning(
                    f"for band {band}:\n    len(det+ulimits+badqual)="
                    f"{len(detections)}+{len(ulimits)}+{len(badqual)}"
                    f" != len(df)={len(band_history)}"
                )
            if len(ulimits) > 0:
                ax.errorbar(
                    ulimits["jd"].values - t_ref.jd, ulimits["diffmaglim"],
                    yerr=None, 
                    ls="none", marker="v", color=band_color, mfc="none", ms=ms,
                    zorder=zorder, alpha=alpha
                )
            if len(badqual) > 0:
                ax.errorbar(
                    badqual["jd"].values - t_ref.jd, badqual["mag"],
                    yerr=badqual["magerr"].values, 
                    ls="none", marker="o", color=band_color, mfc="none", ms=ms,
                    zorder=zorder, alpha=alpha
                )
        else:
            detections = band_history
        if len(detections) > 0:
            detections_scatter = ax.errorbar(
                detections["jd"].values - t_ref.jd, detections["mag"],
                yerr=detections["magerr"].values, 
                ls="none", marker="o", color=band_color, ms=ms,
                zorder=zorder, alpha=alpha, label=label
            )
            legend_handles.append(detections_scatter)
            brightest_values.append(np.nanmin(detections["mag"]))

        

        #===== add models
        if model is None:
            continue # Still inside the band loop...
        #TODO: make this more "generic"? non-SN models might not have this function signature...
        model_copy = copy.deepcopy(model)
        try:
            test_flux = model.bandflux(band, time_grid, zp=25., zpsys="ab")
        except AttributeError as e:
            logger.info("couldn't call `bandflux` on model... ")

        if "samples" in model.res:
            pdict = np.nanmedian(model.res["samples"], axis=0) 
            best_parameters = {
                k: v for k,v in zip( model.res["vparam_names"], pdict )
            }
            model_copy.update(best_parameters)
        model_flux = model_copy.bandflux(band, time_grid, zp=25., zpsys="ab")
        #except AttributeError as e:
        #logger.info("couldn't call `bandflux` on model... ")

        pos_mask = model_flux > 0
        model_mag = -2.5 * np.log10(model_flux[ pos_mask ]) + 8.9
        brightest_values.append(np.min(model_mag[ np.isfinite(model_mag) ]))
        model_time = time_grid[ pos_mask ] - t_ref.jd
        ax.axvline(model["t0"]-t_ref.jd, color="k", ls="--")
        ax.plot(model_time, model_mag, color=band_color, zorder=zorder, alpha=alpha)

        if "samples" in model.res:
            #model_copy = copy.deepcopy(model)
            lc_evaluations = []
            for p_jj, params in enumerate(model.res["samples"][::50]):
                pdict = {k: v for k, v in zip(model.res["vparam_names"], params)}
                model_copy.update(pdict)
                lc_flux_jj = model_copy.bandflux(band, time_grid, zp=25., zpsys="ab")
                with np.errstate(divide="ignore", invalid="ignore"):
                    lc_mag_jj = -2.5 * np.log10(lc_flux_jj[ pos_mask ]) + 8.9
                lc_evaluations.append(lc_mag_jj)
            lc_evaluations = np.vstack(lc_evaluations)

            lc_bounds = np.nanquantile(lc_evaluations, q=[0.16, 0.84], axis=0)
            ax.fill_between(
                model_time, lc_bounds[0,:], lc_bounds[1,:], 
                color=band_color, alpha=0.2, zorder=zorder, 
            )            

    legend = ax.legend(handles=legend_handles, loc=2)
    ax.add_artist(legend)

    ax.set_xlabel(xlabel, fontsize=14)
    y_bright = min(min(brightest_values) - 0.2, 16.0)
    ax.set_ylim(22., y_bright)
    #ax.set_ylim(ax.get_ylim()[::-1])
    ax.axvline(t_ref.jd-t_ref.jd, color="k")

    title = f"{target.objectId}, ra={target.ra:.4f} dec={target.dec:.5f}"
    known_redshift = target.tns_data.parameters.get("Redshift", None)
    if known_redshift is not None:
        title = title + r" $z_{\rm TNS}=" + f"{known_redshift:.3f}" + "$"

    ax.text(
        0.5, 1.0, title, fontsize=14,
        ha="center", va="bottom", transform=ax.transAxes
    )

    ##======== add postage stamps
    for ii, imtype in enumerate(["Science", "Template", "Difference"]):
        if len(fig.axes) == 4:
            im_ax = fig.axes[ii+1]
        else:
            im_ax = fig.add_subplot(lc_gs[ii:ii+1, -1:])

        im_ax.set_xticks([])
        im_ax.set_yticks([])
        im_ax.text(
            1.02, 0.5, imtype, rotation=90, transform=im_ax.transAxes, ha="left", va="center"
        )

        im = target.fink_data.cutouts.get(imtype, None)
        if im is None:
            continue

        im_finite = im[ np.isfinite(im) ]

        vmin, vmax = zscaler.get_limits(im_finite.flatten())

        im_ax.imshow(im, vmin=vmin, vmax=vmax)

        xl_im = len(im.T)
        yl_im = len(im)
        im_ax.plot([0.5 * xl_im, 0.5 * xl_im], [0.2*yl_im, 0.4*yl_im], color="r")
        im_ax.plot([0.2*yl_im, 0.4*yl_im], [0.5*yl_im, 0.5*yl_im], color="r")
    fig.tight_layout()

    return fig


def plot_observing_chart(observer: Observer, target: Target=None, t_ref=None):
    t_ref = t_ref or Time.now()

    fig, ax = plt.subplots()

    time_grid = t_ref + np.linspace(0, 24, 24 * 4) * u.hour

    timestamps = np.array([x.mjd for x in time_grid])

    moon_altaz = observer.moon_altaz(time_grid)
    sun_altaz = observer.sun_altaz(time_grid)

    civil_night = observer.tonight(t_ref, horizon=0*u.deg)
    astro_night = observer.tonight(t_ref, horizon=-18*u.deg)

    ax.fill_between( # civil night
        timestamps, -90*u.deg, 90*u.deg, (sun_altaz.alt < 0*u.deg), color="0.9", 
    )
    ax.fill_between( # naval ??
        timestamps, -90*u.deg, 90*u.deg, (sun_altaz.alt < -6*u.deg), color="0.7", 
    )
    ax.fill_between( # civil night
        timestamps, -90*u.deg, 90*u.deg, (sun_altaz.alt < -12*u.deg), color="0.4", 
    )
    ax.fill_between( # astronomical night
        timestamps, -90*u.deg, 90*u.deg, sun_altaz.alt < -18*u.deg, color="0.3", 
    )


    ax.plot(timestamps, moon_altaz.alt.deg, color="0.5", ls="--", label="moon")
    ax.plot(timestamps, sun_altaz.alt.deg, color="0.5", ls=":", label="sun")
    ax.set_ylim(0, 90)
    ax.set_ylabel("Altitude [deg]", fontsize=16)


    if target is not None:
        target_altaz = observer.altaz(time_grid, target.coord)
        ax.plot(timestamps, target_altaz.alt.deg, color="b", label="target")

        if all(target_altaz.alt < 30*u.deg):
            ax.text(
                0.5, 0.5, f"target alt never >30 deg", color="red", rotation=45,
                ha="center", va="center", transform=ax.transAxes, fontsize=18
            )

    #obs_name = getattr(vf.observatory.info, "name", vf.obs_str) or vf.obs_str
    obs_name = observer.name
    title = f"Observing from {obs_name}"
    title = title + f"\n starting at {t_ref.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    ax.text(
        0.5, 1.0, title, fontsize=14,
        ha="center", va="bottom", transform=ax.transAxes
    )

    iv = 3 # tick marker interval.
    fiv = 24 / iv # interval fraction of day.

    xticks = round(timestamps[0] * fiv, 0) / fiv + np.arange(0, 1, 1. / fiv)
    hourmarks = [Time(x, format="mjd").datetime for x in xticks]
    xticklabels = [hm.strftime("%H:%M") for hm in hourmarks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_xlim(timestamps[0], timestamps[-1])

    if target is not None:
        ax2 = ax.twinx()
        mask = target_altaz.alt > 10. * u.deg
        airmass_time = timestamps[ mask ]
        airmass = 1. / np.cos(target_altaz.zen[ mask ]).value
        ax2.plot(airmass_time, airmass, color="red")
        ax2.set_ylim(1.0, 4.0)
        ax2.set_ylabel("Airmass", color="red", fontsize=14)
        ax2.tick_params(axis='y', colors='red')
        ax2.set_xlim(ax.get_xlim())

    ax.legend()

    return fig
