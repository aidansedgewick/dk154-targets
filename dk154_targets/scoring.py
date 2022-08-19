import logging

import time

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from astroplan import Observer
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

def default_score(target: "Target", observatory: Observer, **kwargs):
    times = {}
    logger = logging.getLogger("default_score")

    jds = target.target_history['jd'].values
    assert all(jds[:-1] <= jds[1:])
    # make sure they're in ascending order.

    t_ref = kwargs.get("t_ref", Time.now())
    if not isinstance(t_ref, Time):
        raise ValueError(f"t_ref should be astropy.time.Time, not {type(t_ref)}")

    if "tag" in target.target_history.columns:
        detections = target.target_history.query("tag=='valid'")
    else:
        # If there is no 'tag' column, it means all the rows are 'valid'
        detections = target.target_history

    ###============== set up some things to keep track =============###
    score = getattr(target, "base_score", 100.) # base score.
    t_ref = kwargs.get("t_ref", Time.now())
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
    mag_factor = 10 ** ((18.5 - last_mag) / 2)
    factors["mag"] = mag_factor
    score = score * mag_factor
    if last_mag > 18.5:
        reject_comments.append(f"Too faint: {last_band}={last_mag:.2f}")
        reject = True
    else:
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
        peak_dt = t_ref.jd - model["t0"]
        interest_factor = tuned_interest_function(peak_dt)
        score = score * interest_factor
        score_comments.append(f"interest {interest_factor:.2f} from\n     peak={peak_dt:.2f} days")
        factors["interest"] = interest_factor

        if peak_dt > 20.:
            reject_comments.append("too far past peak")
            reject = True

        ###===== Blue colour?
        model_g = model.bandflux("ztfg", t_ref.jd)
        model_r = model.bandflux("ztfr", t_ref.jd)
        model_gr = 2.5 * np.log10(model_r / model_g)
        color_factor = 1. + np.exp(-5. * model_gr)
        score_comments.append(f"model color g-r={model_gr:.2f}\n    gives f={color_factor:.2f}")
        if not np.isfinite(color_factor):
            color_factor = 0.1
        factors["color"] = color_factor

        score = score * color_factor


    if not np.isfinite(score) or score < 0.:
        print(f"{target.objectId} score is -ve or nan! {score} \n", factors, score_comments)
        print()
    #else:
    #    print(f"{target.objectId} score is currently {score}")

    if observatory is not None:
        minimum_altitude = 30. # deg.
        """
        tonight = observatory.tonight(horizon=-18*u.deg) # Now if night, else nearest sunset time.
        tonight = fast_observer_tonight(observatory, horizon=-18*u.deg, n_grid_points=50)
        target_altaz = observatory.altaz(tonight[0], target.coord)
        times["t_tonight"] = time.perf_counter() - t1
        """


        tonight = kwargs.get("tonight", None)
        if tonight is None:
            tonight = observatory.tonight(time=kwargs.get())

        dt = 15 / (24 * 60)
        t1 = time.perf_counter()
        night_grid = Time(
            np.arange(tonight[0].mjd, tonight[1].mjd, dt), format="mjd"
        )
        if night_grid[-1] > tonight[-1]:
            night_grid = night_grid[:-1]
        altaz_grid = observatory.altaz(night_grid, target)


        if all(altaz_grid.alt.deg < minimum_altitude):
            negative_score = True

        if altaz_grid[0].alt.deg < minimum_altitude:
            negative_score = True
        else:
            alt_above_minimum = np.maximum(altaz_grid.alt.deg, minimum_altitude) - minimum_altitude
            integral = np.trapz(alt_above_minimum, x=night_grid.mjd)
            assert integral >= 0

            norm = (tonight[-1].mjd - tonight[0].mjd) * (45. - minimum_altitude) # time above 30 deg.

            observing_factor = 1. / (integral / norm)
            factors["observing"] = observing_factor
            score_comments.append(f"observing f={observing_factor}")
            score = score * observing_factor

    # normalise score!
    if not np.isfinite(score):
        reject = True
        reject_comments.append("non-finite score")
        logger.warning(f"{target.objectId} score NaN")
        print(factors)
    score = logistic(np.log10(score), 1., 1., 1.)
    if not np.isfinite(score):
        reject = True
        reject_comments.append("non-finite score")
        logger.warning(f"{target.objectId} score NaN")
        print(factors)
    #print(" ".join(f"{k}:{v:.2e}" for k, v in times.items()))
    if negative_score:
        return -1.

    if reject:
        score = np.nan
    return score, score_comments, reject_comments


def uniform_score(target: "Target", observatory: Observer):
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