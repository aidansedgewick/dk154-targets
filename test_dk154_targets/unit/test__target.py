import datetime
import pytest

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

from astroplan import Observer

from sncosmo.models import Model

from dk154_targets.target import Target, plot_lightcurve, plot_observing_chart
from dk154_targets.scoring import ScoringBadSignatureError, ScoringBadReturnValueError
from dk154_targets.modelling import default_sncosmo_model


t_test = Time(datetime.datetime(year=1993, month=2, day=2, hour=6, minute=30))

class BasicModel:
    def __init__(self,):
        pass

    def bandflux(self, band, time, zp=None, zpsys=None):
        f0 = 5.7547e-4
        if hasattr(time, "__len__"):
            return np.array([f0] * len(time)) # should be mag~17.0
        return f0



def basic_scoring_function(target, observatory):
    return 99., ["a comment"], []


def another_basic_scoring_function(target, observatory):
    return 45., ["another comment"], []
    

def test__init():
    target = Target("t1", 30.0, -4.0)
    assert target.objectId == "t1"
    assert np.isclose(target.ra, 30.)
    assert np.isclose(target.dec, -4.0
    )
    assert isinstance(target.coord, SkyCoord)
    assert np.isclose(target.coord.ra.deg, 30.)

    assert target.target_history is None
    assert isinstance(target.models, list) and len(target.models) == 0


def test__basic_evaluate():
    target = Target("t2", 0., 0.,)
    target.evaluate_target(basic_scoring_function, None)
    assert set(target.score_history.keys()) == set(["no_observatory"])
    assert len(target.score_history["no_observatory"]) == 1
    assert len(target.score_history["no_observatory"][0]) == 2
    assert np.isclose(target.score_history["no_observatory"][0][0], 99.)
    assert isinstance(target.score_history["no_observatory"][0][1], Time)
    # Should def take <5 seconds to run this test!
    assert abs(target.score_history["no_observatory"][0][1].mjd - Time.now().mjd) < 5. / 86400.
    assert set(target.last_score_comments.keys()) == set(["no_observatory"])
    assert set(target.last_score_comments["no_observatory"]) == set(["a comment"])

    location = EarthLocation(lat=1, lon=55, height=20)
    observer = Observer(location=location, name="astro_lab")
    target.evaluate_target(another_basic_scoring_function, observer)
    assert set(target.score_history.keys()) == set(["no_observatory", "astro_lab"])
    assert len(target.score_history) == 2
    assert len(target.score_history["no_observatory"]) == 1
    assert len(target.score_history["astro_lab"]) == 1
    assert np.isclose(target.score_history["astro_lab"][-1][0], 45.)
    assert set(target.last_score_comments.keys()) == set(["no_observatory", "astro_lab"])
    assert set(target.last_score_comments["astro_lab"]) == set(["another comment"])


    t1 = target.score_history["no_observatory"][0][1].mjd % 1
    t2 = target.score_history["astro_lab"][0][1].mjd % 1
    assert t2 > t1 # This score was later.


def test__bad_scoring_function():

    def scoring_bad_signature():
        return 99., [], []

    res = scoring_bad_signature()
    assert len(res) == 3 and isinstance(res, tuple)
    assert np.isclose(res[0], 99.)
    assert isinstance(res[1], list) and isinstance(res[2], list)

    target = Target("t1", 30., 60.,)
    with pytest.raises(TypeError):
        target.evaluate_target(scoring_bad_signature, None)

    def bad_return_two_floats(target, observatory):
        return 99., 99.
    
    with pytest.raises(ValueError):
        target.evaluate_target(bad_return_two_floats, None)

    def bad_return_str(target, observatory):
        return "score", [], []

    with pytest.raises(ValueError):
        target.evaluate_target(bad_return_str, None)


def test__get_last_score():
    target = Target("t3", 90., 45.)
    target.evaluate_target(basic_scoring_function, None)

    last_score = target.get_last_score("no_observatory")
    assert isinstance(last_score, float)
    assert np.isclose(last_score, 99.)

    last_scores_with_time = target.get_last_score("no_observatory", return_time=True)
    assert isinstance(last_scores_with_time, tuple)
    assert isinstance(last_scores_with_time[0], float)
    assert np.isclose(last_scores_with_time[0], 99.)
    assert isinstance(last_scores_with_time[1], Time)
    assert abs(target.score_history["no_observatory"][0][1].mjd - Time.now().mjd) < 5. / 86400.

    # test with obervatory = None

    none_score = target.get_last_score(None)
    assert np.isclose(none_score, 99.)
    

def test__basic_model():
    test_model = BasicModel()
    assert np.isclose(test_model.bandflux(None, 2459000.5), 5.7547e-4)

    def basic_modelling_function(target):
        return BasicModel()

    target = Target("t4", 180., 20.)
    assert target.updated # should start as updated
    target.updated = False # switch to not, so we can test if modelling switches it.
    assert not target.updated

    target.model_target(basic_modelling_function)
    assert len(target.models) == 1
    assert isinstance(target.models[0], BasicModel)
    

def test__update_target_history():
    main_date = 2459000.5 # this is 00:00 on May 31st 2020.

    t1 = Target("t1", 30., 30.)
    new_data = pd.DataFrame(
        {
            "jd": np.array([0.0, 1.1, 1.9, 3.3]) + main_date,
            "magpsf": np.array([19.5, 19.0, 18.5, 18.0]),
            "sigmapsf": np.array([0.1, 0.1, 0.1, 0.1]),
        }
    )
    assert t1.target_history is None
    t1.update_target_history(new_data)
    assert isinstance(t1.target_history, pd.DataFrame)
    assert len(t1.target_history)


    existing_th = pd.DataFrame(
        {
            "jd": np.array([-2.0, -1.0, 0.1, 0.5]) + main_date,
            "magpsf": np.array([20.5, 20.0, 19.4, 19.2]),
            # NO sigmapsf to see if columns carry properly.
        }
    )
    t2 = Target("t2", 60., 30., target_history=existing_th)
    assert len(t2.target_history) == 4
    assert t2.target_history is not None


    print("t2 target hist", t2.target_history)
    t2.update_target_history(new_data)
    assert len(t2.target_history) == 7
    assert np.allclose(
        t2.target_history["jd"], np.array([-2.0, -1.0, 0.1, 0.5, 1.1, 1.9, 3.3]) + 2459000.5
    )
    assert np.allclose(
        t2.target_history["magpsf"],
        np.array([20.5, 20.0, 19.4, 19.2, 19.0, 18.5, 18.0])
    )
    assert all(not np.isfinite(x) for x in t2.target_history["sigmapsf"].values[:4])
    assert np.allclose(t2.target_history["sigmapsf"].values[4:], [0.1, 0.1, 0.1])


def test__plotting():

    ## TODO make this better...

    objectId = "ZTF20acyvzbr"

    target = Target.from_fink_query(objectId, withupperlim=True)
    if target is None:
        raise ValueError("fink query failed")
    target.model_target(default_sncosmo_model)
    t0 = Time(target.models[0]["t0"], format="jd")
    lc_fig = target.plot_lightcurve(t_ref=t0)
    assert isinstance(lc_fig, plt.Figure)

    observatory = Observer.at_site("lapalma")
    target.plot_observing_chart(observatory, t_ref=t0)