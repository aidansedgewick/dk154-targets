import logging

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from dk154_targets.visibility_forecast import VisibilityForecast

class ScoringFunctionError(Exception):
    pass

class ScoringBadSignatureError(Exception):
    pass

class ScoringBadReturnValueError(Exception):
    pass


def logistic(x, L, k, x0):
    return L / (1. + np.exp(-k * (x-x0)))

def gauss(x, A, mu, sig):
    return A * np.exp(-(x-mu)**2 / sig ** 2)

def tuned_interest_function(x):
    return logistic(x, 10.0, -1./2., -6.) + gauss(x, 4., 0., 1.)

def default_score(target: "Target", observatory: EarthLocation):

    logger = logging.getLogger("default_score")

    jds = target.target_history['jd'].values
    assert all(jds[:-1] < jds[1:])
    # make sure they're in ascending order.

    t_now = Time.now()

    if "tag" in target.target_history.columns:
        detections = target.target_history.query("tag=='valid'")
    else:
        detections = target.target_history

    ###================== set up some things ================###
    score = 100. # base score.
    reject = False
    negative_score = False
    reject_comments = []
    score_comments = [f"base score {score:.2f}"]
    factors = {}


    ###====== Are there at least 2 observations in each band? =======###
    N_obs = {}
    for fid, fid_detections in detections.groupby("fid"):
        N_obs[fid] = len(fid_detections)
    if N_obs.get(1, 0) < 2 or N_obs.get(2, 0) < 2:
        reject = True
        reject_comments.append(f"N obs: {N_obs.get(1, 0)} g, {N_obs.get(2, 0)} r")


    ###===== if it's way too faint =====###
    last_mag = target.target_history["magpsf"].values[-1]
    last_fid = target.target_history["fid"].values[-1]
    band_lookup = {1: "g", 2: "r"}
    last_band = band_lookup[last_fid]
    if last_mag > 18.5:
        reject_comments.append(f"Too faint: {last_band}={last_mag:.2f}")
        reject = True
    else:
        mag_factor = 10 ** ((18.5 - last_mag) / 2)
        factors["mag"] = mag_factor
        score = score * mag_factor
        score_comments.append(f"mag {last_mag:.2f} gives f={mag_factor:.2f}")


    ###===== Is the target very old? ======###    
    timespan = detections["jd"].max() - detections["jd"].min()
    if timespan > 20.:
        timespan_factor = 1. / (timespan - 19.) # -19 means that t - 19 > 1 always.
        factors["timespan"] = timespan_factor
        score = score * timespan_factor
        score_comments.append(f"timespan {timespan:.1f} gives f={timespan_factor:.2f}")


    ###===== Is the target still rising ======###
    rising_fractions = []
    for fid, fid_history in detections.groupby("fid"):
        fid_history.sort_values("jd", inplace=True)
        if len(fid_history) > 1:
            interval_rising = fid_history["magpsf"].values[:-1] > fid_history["magpsf"].values[1:]
            rising_fraction = sum(interval_rising) / len(interval_rising)
        else:
            rising_fraction = 0.5
            score_comments.append(f"{len(fid_history)} band {fid} obs gives {rising_fraction}")
        rising_fractions.append(max(rising_fraction, 0.05))
        
    rising_factor = np.prod(rising_fractions)
    factors["rising"] = rising_factor
    rstr = ','.join(f'{f:.2f}' for f in rising_fractions)
    score_comments.append(f"rising f={rstr}")
    score = score * rising_factor
    if any([f < 0.4 for f in rising_fractions]):
        reject = True
        reject_comments.append("more than half declining...")


    ###===== is there a model =====###
    if len(target.models) > 0:
        model = target.models[-1]
    else:
        model = None

    if model is not None:
        ###===== Time from peak?
        peak_dt = t_now.jd - model["t0"]
        interest_factor = tuned_interest_function(peak_dt)
        score = score + np.log10(interest_factor)
        score_comments.append(f"interest {interest_factor:.2f} from\n     peak={peak_dt:.2f} days")

        if peak_dt > 20.:
            reject_comments.append("too far past peak")
            reject = True

        ###===== Blue colour?
        model_g = model.bandflux("ztfg", t_now.jd)
        model_r = model.bandflux("ztfr", t_now.jd)
        model_gr = 2.5 * np.log10(model_r / model_g)
        color_factor = 1. + np.exp(-5. * model_gr)
        score_comments.append(f"model color g-r={model_gr:.2f}\n    gives f={color_factor:.2f}")
        if not np.isfinite(color_factor):
            color_factor = 0.1
        factors["color"] = color_factor

        score = score * color_factor

    if not np.isfinite(score):
        print(f"score is nan!\n", factors)

    if observatory is not None:
        t_now = Time.now()
        dummy_vf = VisibilityForecast(target=None, observatory=observatory, t0=t_now)
        t_ref = dummy_vf.nearest_night(t_ref=t_now) # return t_now if nighttime, else sunset


        vf = VisibilityForecast(target, observatory, t0=t_ref)
        current_airmass = 1. / np.cos(vf.target_altaz.zen[0])
        current_alt = vf.target_altaz.alt[0]

        if current_airmass < 0:
            current_airmass = 100.
        score = score / current_airmass
        score_comments.append(f"airmass = {current_airmass:.2f}")

        #if current_alt < 20. * u.deg:
        #    negative_score = True



    # normalise score!
    score = logistic(np.log10(score), 1., 1., 1.)
    if not np.isfinite(score):
        reject = True
        reject_comments.append("non-finite score")
        logger.warning(f"{target.objectId} score NaN")
        logger.info(f"{target.objectId}")

        #print("========", target.objectId, "========")
        #print(f"target_history len = {len(target.target_history)}")
        #for l in score_comments:
        #    print(l)
    if negative_score:
        return -1.

    if reject:
        score = np.nan
    return score, reject_comments, score_comments


def uniform_score(target: "Target", observatory: EarthLocation):
    return 1., [], []

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = np.linspace(-20, 10, 1000)
    y = tuned_interest_function(x)
    fig, ax = plt.subplots()
    ax.set_ylabel("factor", fontsize=16)
    ax.set_xlabel("days from model peak", fontsize=16)
    ax.plot(x, y)
    plt.show()