import datetime
import pytest

import numpy as np

import pandas as pd

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

from sncosmo.models import Model

from dk154_targets.target import Target
from dk154_targets.scoring import ScoringBadSignatureError, ScoringBadReturnValueError


t_test = Time(datetime.datetime(year=1993, month=2, day=2, hour=6, minute=30))

class BasicModel:
    def __init__(self,):
        pass

    def bandflux(self, band=None):
        return 5.7547e-4 # should be mag~17.0


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

    astro_lab = EarthLocation(lat=1, lon=55, height=20)
    astro_lab.name = "astro_lab"
    target.evaluate_target(another_basic_scoring_function, astro_lab)
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
    

def test__basic_model():
    test_model = BasicModel()
    assert np.isclose(test_model.bandflux(), 5.7547e-4)

    def basic_modelling_function(target):
        return BasicModel()

    target = Target("t4", 180., 20.)
    assert target.updated # should start as updated
    target.updated = False # switch to not, so we can test if modelling switches it.
    assert not target.updated

    target.model_target(basic_modelling_function)
    assert len(target.models) == 1
    assert isinstance(target.models[0], BasicModel)
    

