import copy
import inspect
import logging
import traceback
#from dataclasses import dataclass # >=py3.7 only!
import warnings
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.visualization import ZScaleInterval

from astroplan import Observer

#from dk154_targets.queries import FinkQuery, AtlasQueryManager
from dk154_targets.queries import FinkQuery
#from dk154_targets.scoring import ScoringBadSignatureError, ScoringBadReturnValueError
#from dk154_targets.visibility_forecast import plot_observing_chart

logger = logging.getLogger(__name__.split(".")[-1])

lc_gs = plt.GridSpec(3,4)
zscaler = ZScaleInterval()

class Target:

    default_base_score = 100.
    default_band_lookup = {1: "ztfg", 2: "ztfr"}

    def __init__(
        self, 
        objectId: str, 
        ra: float, 
        dec: float,
        target_history: pd.DataFrame=None,
        base_score: float=None,
        band_lookup: dict=None,
    ):
        #===== basics
        self.objectId = objectId
        self.ra = ra
        self.dec = dec
        self.coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        self.base_score = base_score or self.default_base_score
        self.band_lookup = band_lookup or self.default_band_lookup

        #===== keep track of target data 
        self.target_history = target_history
        self.atlas_data = None
        self.cutouts = {}
        self.cutout_update_time = None

        #===== models
        self.models = []
        self.updated = True
        self.target_of_opportunity = False

        #===== rank/score history
        self.score_history = {"no_observatory": []}
        self.rank_history = {"no_observatory": []}
        self.last_score_comments = {}
        self.reject_comments = None


    @classmethod
    def from_fink_query(cls, objectId, ra=None, dec=None, base_score=None, **kwargs):
        target_history = FinkQuery.query_objects(objectId=objectId, **kwargs)
        if target_history is None:
            logger.warn(f"no target history from {objectId}")
            return None
        if isinstance(target_history, pd.DataFrame) and target_history.empty:
            return None
        return cls.from_target_history(objectId, target_history, ra=ra, dec=dec, base_score=base_score)


    @classmethod
    def from_target_history(cls, objectId, target_history, ra=None, dec=None, base_score=None):
        target_history = target_history.copy(deep=True)
        target_history.sort_values("jd", inplace=True)
        if ra is None or dec is None:
            ra = target_history["ra"].values[-1]
            dec = target_history["dec"].values[-1]
        target = cls(objectId, ra, dec, target_history=target_history, base_score=base_score)
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

    #def get_last_magnitude(self, data="fink", fid=None)
    #    target_data = target.data[data]
    #    if fid is not None:
    #        fid.


    def model_target(self, modelling_function: Callable):
        model = modelling_function(self)
        if model is not None:
            #logger.info(f"{self.objectId} {type(model).__name__} model built")
            self.models.append(model)
            self.updated = True
        else:
            pass
            #logger.warning()


    def update_target_history(
        self, new_df: pd.DataFrame, keep_old=True, date_col="jd",
    ):
        """
        Concatenate new data to the existing target_history

        Parameters
        ----------
        new_df
            new data to include in target
        keep_old
            if the earliest data in the new data is older than the last data in the 
            existing target_history, we need to decide whether to truncate the old 
            data or new to avoid inluding repeat observations.
            By default, truncate the new data to only include observations after the
            last data (`keep_old=True` default.)
        date_col
            The column used to truncate either the `target_history` or the `new_df`.
            default date_col=`jd`.

        """

        if self.target_history is None:
            self.target_history = new_df
            logger.info("no target_history, use new data")
            self.target_history.sort_values(date_col, inplace=True)
            return None
            
        if new_df[date_col].min() > self.target_history[date_col].max():
            logger.info(f"{self.objectId} update: simple concat")
            self.target_history = pd.concat([self.target_history, new_df])
            self.target_history.sort_values(date_col, inplace=True)
            return None

        # the other case... where there is overlap between existing data and new data.

        gt_min = self.target_history[date_col].min() <= new_df[date_col]
        lt_max = new_df[date_col] <= self.target_history[date_col].max()

        if all( gt_min.values & lt_max.values ):
            if len(new_df) == 1:
                # This is the case where the alert is 'old' and already included in the target_history.
                is_close = np.isclose(
                    new_df[date_col], self.target_history[date_col], atol=1./(60.*24.)
                )
                if any(is_close):
                    logger.info(f"{self.objectId} alert data already in lc data")
                    pass
                else:
                    raise ValueError("new data within existing data, but none are close date match")
            else:
                # This case something wrong has happened.
                raise ValueError(f"new_df in date range but more than one alert: len={len(new_df)}")
        else:
            min_new_date = new_df[date_col].min()
            max_existing_date = self.target_history[date_col].max()
            if keep_old:
                logger.info(f"{self.objectId} update: truncate update data")
                existing_data = self.target_history
                new_data = new_df.query(f"{date_col} > @max_existing_date")
            else:
                logger.info(f"{self.objectId} update: truncate existing data")
                existing_data = self.target_history.query(f"{date_col} < @min_new_date")
                print(new_data["jd"])
                new_data = new_df
            if not existing_data[date_col].max() < new_data[date_col].min():
                print(
                    f"existing max: {existing_data[date_col].max()}\n "
                    f"new_min {new_data[date_col].min()}\n "
                )
                print("newdf datecol\n", new_df[date_col])
                raise ValueError
            target_history = pd.concat([existing_data, new_data])
            self.target_history = target_history

        self.updated = True
        self.target_history.sort_values(date_col, inplace=True)


    def update_cutouts(self, n_days=2):
        """
        Update the fink cutouts.
        TODO: fink a better way to do this?? ie - ask the fink_query_manager to do it?
        """
        cutouts_are_None = any([im is None for im in self.cutouts.values()])
        no_cutouts = len(self.cutouts) == 0
        if self.cutout_update_time is None:
            cutouts_are_old = True
        else:
            cutouts_are_old = Time.now() - self.cutout_update_time > n_days * u.day
        if no_cutouts or cutouts_are_None or cutouts_are_old:
            logger.info(f"update {self.objectId} cutouts")
            for imtype in FinkQuery.imtypes:
                im = FinkQuery.get_cutout(imtype, objectId=self.objectId)
                self.cutouts[imtype] = im
            self.cutout_update_time = Time.now()


    def plot_lightcurve(self, t_ref=None, fig=None):
        #try:
        logger.info(f"lc for {self.objectId}")
        return plot_lightcurve(self, t_ref=t_ref, fig=fig)
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


def plot_lightcurve(target: Target, t_ref: Time=None, fig=None):

    t_ref = t_ref or Time.now()
    if not isinstance(t_ref, Time):
        raise ValueError(f"t_ref should be astropy.time.Time, not {type(t_ref)}")
    xlabel = f"time before now ({t_ref.datetime.strftime('%Y-%m-%d %H:%M')})"

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
    time_grid = np.arange(full_detections["jd"].min()-5., t_ref.jd + 5., 1.0)

    band_lookup = target.band_lookup

    if not target.models:
        model = None
    else:
        model = target.models[-1]    

    for ii, (fid, fid_history) in enumerate(target.target_history.groupby("fid")):
        fid_history.sort_values("jd", inplace=True)
        if "tag" in fid_history.columns:
            detections = fid_history.query("tag=='valid'")
            ulimits = fid_history.query("tag=='upperlim'")
            badqual = fid_history.query("tag=='badquality'")
            
            if not (len(detections) + len(ulimits) + len(badqual)) == len(fid_history):
                logger.warning(
                    f"len(det)+len(ulimits)+len(badqual) {len(det)}+{len(ulimits)}+{len(badqual)}"
                    f" != len(df)={len(fid_history)}"
                )
            ax.errorbar(
                ulimits["jd"].values - t_ref.jd, ulimits["diffmaglim"],
                yerr=None, 
                ls="none", marker="v", color=f"C{ii}", mfc="none"
            )
            ax.errorbar(
                badqual["jd"].values - t_ref.jd, badqual["magpsf"],
                yerr=badqual["sigmapsf"].values, 
                ls="none", marker="o", color=f"C{ii}", mfc="none"
            )
        else:
            detections = fid_history
        ax.errorbar(
            detections["jd"].values - t_ref.jd, detections["magpsf"],
            yerr=detections["sigmapsf"].values, 
            ls="none", marker="o", color=f"C{ii}"
        )
        y_bright = detections["magpsf"].min()

        

        #===== add models
        #for model in models:
        if model is None:
            continue # Still inside the fid loop...
        #TODO: make this more "generic"? non-SN models might not have this function signature...
        try:
            model_flux = model.bandflux(band_lookup[fid], time_grid, zp=25., zpsys="ab")
        except AttributeError as e:
            logger.info("couldn't call `bandflux` on model... ")
        pos_mask = model_flux > 0
        model_mag = -2.5 * np.log10(model_flux[ pos_mask ]) + 8.9
        y_bright = min(y_bright, min(model_mag))
        model_time = time_grid[ pos_mask ] - t_ref.jd
        ax.axvline(model["t0"]-t_ref.jd, color="k", ls="--")
        ax.plot(model_time, model_mag, color=f"C{ii}")

        if "samples" in model.res:
            model_copy = copy.deepcopy(model)
            param_dicts = [
                {k: v} for params in model.res["samples"] for k, v in zip(
                    model.res["vparam_names"], params
                )
            ]
            lc_evaluations = []
            for p_jj, params in enumerate(model.res["samples"][::50]):
                pdict = {k: v for k, v in zip(model.res["vparam_names"], params)}
                model_copy.update(pdict)
                lc_flux_jj = model_copy.bandflux(band_lookup[fid], time_grid, zp=25., zpsys="ab")
                with np.errstate(divide="ignore", invalid="ignore"):
                    lc_mag_jj = -2.5 * np.log10(lc_flux_jj[ pos_mask ]) + 8.9
                lc_evaluations.append(lc_mag_jj)
            lc_evaluations = np.vstack(lc_evaluations)

            lc_bounds = np.nanquantile(lc_evaluations, q=[0.16, 0.84], axis=0)
            ax.fill_between(model_time, lc_bounds[0,:], lc_bounds[1,:], color=f"C{ii}", alpha=0.2)
            
    if target.atlas_data is not None:
        for band, full_band_lc in target.atlas_data.groupby("F"):
            band_lc = full_band_lc.query("(duJy < uJy) & (m > 0) & (m < mag5sig)")
            atlas_jd = Time(band_lc["MJD"].values, format="mjd").jd - t_ref.jd

            ax.errorbar(atlas_jd, band_lc["m"], yerr=band_lc["dm"], marker="x")


    ax.set_xlabel(xlabel, fontsize=14)
    y_bright = min(y_bright - 0.2, 16.0)
    ax.set_ylim(22., y_bright)
    #ax.set_ylim(ax.get_ylim()[::-1])
    ax.axvline(t_ref.jd-t_ref.jd, color="k")

    title = f"{target.objectId}, ra={target.ra:.4f} dec={target.dec:.5f}"
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

        im = target.cutouts.get(imtype, None)
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
