import copy
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

#from dk154_targets.queries import FinkQuery, AtlasQueryManager
from dk154_targets.queries import FinkQuery
from dk154_targets.scoring import ScoringBadSignatureError, ScoringBadReturnValueError
from dk154_targets.visibility_forecast import VisibilityForecast

logger = logging.getLogger(__name__.split(".")[-1])

lc_gs = plt.GridSpec(3,4)
zscaler = ZScaleInterval()

class Target:
    def __init__(
        self, 
        objectId: str, 
        ra: float, 
        dec: float,
        target_history: pd.DataFrame=None,
        base_score: float=100.,
        **kwargs
    ):
        #===== basics
        self.objectId = objectId
        self.ra = ra
        self.dec = dec
        self.coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        self.base_score = base_score

        #===== keep track of target data #- each should be a TargetData
        self.target_history = target_history
        self.cutouts = {}

        self.visibility_forecasts = {}

        #===== models
        self.models = []
        self.updated = True
        self.target_of_opportunity= False

        #===== rank/score history
        self.score_history = {"no_observatory": []}
        self.rank_history = {"no_observatory": []}
        self.last_score_comments = {}
        self.reject_comments = None


    def evaluate_target(
        self, scoring_function: Callable, observatory: EarthLocation, t_ref: Time=None
    ):
        obs_name = getattr(observatory, "name", "no_observatory")
        if obs_name == "no_observatory":
            assert observatory is None
        t_ref = t_ref or Time.now()

        scoring_result = scoring_function(self, observatory)
        if isinstance(scoring_result, tuple):
            if len(scoring_result) != 3:
                raise ValueError
            score, score_comments, reject_comments = scoring_result
            self.last_score_comments[obs_name] = score_comments
            self.reject_comments = reject_comments
        elif isinstance(scoring_result, ()):
            score = scoring_result
            score_comments = []
            reject_comments = []

        if obs_name not in self.score_history:
            self.score_history[obs_name] = []
        assert obs_name in self.score_history
        self.score_history[obs_name].append((score, t_ref))


    def get_last_score(self, obs_name, return_time=False):
        if not isinstance(obs_name, str):
            observatory = obs_name
            obs_name = getattr(observatory, "name", "no_observatory")
            if obs_name == "no_observatory":
                assert observatory is None
            else:
                assert isisntance(observatory, EarthLocation)
            
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
            logger.info(f"model built for {self.objectId}")
            self.models.append(model)
            self.updated = True


    def update_target_history(self, new_df, keep_old=True):
        if self.target_history is None:
            self.target_history = new_df

        if new_df["jd"].min() < self.target_history["jd"].max():
            min_new_df_jd = new_df["jd"].min()
            max_target_history_jd = self.target_history["jd"].max()
            if keep_old:
                existing_data = self.target_history
                new_data = new_df.query("jd>@max_target_history_jd")
            else:
                existing_data = self.target_history.query("jd<@min_new_df_jd")
                new_data = new_df
            assert old_target_history["jd"].max() < new_target_history["jd"].min()
            target_history = pd.concat([existing_data, new_data])
            self.target_history = target_history
        else:
            self.target_history = pd.concat([self.target_history, new_df])
        self.target_history.sort_values("jd", inplace=True)


    @classmethod
    def from_fink_query(cls, objectId, ra=None, dec=None):
        target_history = FinkQuery.query_objects(objectId=objectId)
        if target_history is None:
            logger.warn(f"no target history from {objectId}")
            return None
        if isinstance(target_history, pd.DataFrame) and target_history.empty:
            return None
        target_history.sort_values("jd", inplace=True)
        if ra is None or dec is None:
            ra = target_history["ra"].values[-1]
            dec = target_history["dec"].values[-1]
        target = cls(objectId, ra, dec, target_history=target_history)
        return target


    def plot_lightcurve(self,):
        return plot_lightcurve(self)


    def plot_observing_chart(self, observatory):
        vf = VisibilityForecast(self, observatory)
        return vf.plot_observing_chart()


def plot_lightcurve(target: Target, t_ref: Time=None, cutouts="fink"):

    t_ref = t_ref or Time.now()
    xlabel = f"time before now ({t_ref.datetime.strftime('%Y-%m-%d %H:%M')})"

    ##======== initialise figure
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(lc_gs[:,:-1])


    full_detections = target.target_history
    time_grid = np.arange(full_detections["jd"].min()-5., Time.now().jd + 5., 1.0)


    #for data_name, target_data in target.data.items():

    if not target.models:
        model = None
    else:
        model = target.models[-1]

    for ii, (fid, fid_history) in enumerate(target.target_history.groupby("fid")):
        fid_history.sort_values("jd", inplace=True)
        if "tag" in fid_history.columns:
            detections = fid_history.query("tag=='valid'")
            ulimits = fid_history.query("tag==upperlim'")
            badqual = fid_history.query("tag==badquality")
            assert len(detections) + len(ulimits) + len(badqual) == len(target_history)
            ax.errorbar(
                ulimits["jd"].values - t_ref.jd, ulimits["diffmaglim"],
                yerr=ulimits["sigmapsf"].values, 
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
            continue
        model_flux = model.bandflux(band_lookup[fid], time_grid, zp=25., zpsys="ab")
        pos_mask = model_flux > 0
        model_mag = -2.5 * np.log10(model_flux[ pos_mask ]) + 8.9
        y_bright = min(y_bright, min(model_mag))
        model_time = time_grid[ pos_mask ] - t_ref.jd
        ax.axvline(model["t0"]-t_ref.jd, color="k", ls="--")
        ax.plot(model_time, model_mag, color=f"C{ii}")

    ax.set_xlabel(xlabel, fontsize=14)
    y_bright = min(y_bright - 0.2, 16.0)
    ax.set_ylim(22., y_bright)
    #ax.set_ylim(ax.get_ylim()[::-1])
    ax.axvline(Time.now().jd-t_ref.jd, color="k")

    ##======== add postage stamps
    for ii, imtype in enumerate(["Science", "Template", "Difference"]):
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