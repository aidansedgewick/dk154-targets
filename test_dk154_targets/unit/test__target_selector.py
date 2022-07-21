import os
import pytest
import yaml
from pathlib import Path

import numpy as np

from astropy.coordinates import EarthLocation
from astropy.time import Time

from dk154_targets.query_managers import FinkQueryManager
from dk154_targets.target import Target
from dk154_targets.target_selector import TargetSelector

from dk154_targets import paths

basic_config = {
    "query_managers": {
        "fink": { 
            "username" :"test_user",
            "group_id": "user_group",
            "server": "192.0.2.0",
        }
    }
}



test_config = {
    "sleep_time": 2.0,
    "query_managers": {
        "fink": {
            "username" :"test_user",
            "group_id": "user_group",
            "server": "192.0.2.0",
            "topics": ["cool_space_targets"]
        },
    },
    "observatories": {
        "lasilla": "La Silla Observatory", # Can use EarthLocation.of_site() values.
        "astrolab": {"lat": 55., "lon": 1., "height": 20.}, 
        "north_pole": {"lat": 90, "lon": 0, "height": 0}, # can always see polaris from here.
    }
}

def test__init():

    selector = TargetSelector(test_config)
    
    assert isinstance(selector.selector_config, dict)
    assert set(selector.selector_config.keys()) == set(["sleep_time", "query_managers", "observatories"])

    assert np.isclose(selector.selector_config["sleep_time"], 2.0) # duh

    assert isinstance(selector.observatory_config, dict)
    assert len(selector.observatory_config) == 3
    assert len(selector.observatories) == 4 # always initialise with "None" as first observatory.
    assert selector.observatories[0] is None
    assert all(isinstance(obs, EarthLocation) for obs in selector.observatories[1:])
    assert np.isclose(selector.observatories[1].lon.deg, -70.73) # of_site worked correctly.
    assert np.isclose(selector.observatories[1].lat.deg, -29.256667)

    assert isinstance(selector.fink_query_manager, FinkQueryManager)
    assert isinstance(selector.fink_query_manager.credential_config, dict)
    assert (
        set(selector.fink_query_manager.credential_config.keys()) == 
        set(["username", "group_id", "server"])
    )
    assert isinstance(selector.fink_query_manager.topics, list)
    assert set(selector.fink_query_manager.topics) == set(["cool_space_targets"])

    assert isinstance(selector.target_lookup, dict)
    assert len(selector.target_lookup) == 0


def test__from_config():
    default_path_rel = TargetSelector.default_selector_config_path.relative_to(paths.base_path)
    assert default_path_rel == Path("config/selector_config.yaml")
    assert TargetSelector.default_selector_config_path.exists()

    test_config_path = paths.test_path / "test_config.yaml"
    with open(test_config_path, "w+") as f:
        yaml.dump(test_config, f)

    ts = TargetSelector.from_config(config_file=test_config_path)
    assert len(ts.observatories) == 4
    assert set(obs.name for obs in ts.observatories[1:]) == set(["lasilla", "astrolab", "north_pole"])
    os.remove(test_config_path)
    assert not test_config_path.exists()


def test__bad_fink_config():
    empty_config = {}
    ts = TargetSelector(empty_config)
    assert ts.fink_query_manager is None

    fink_no_credentials_config = {
        "query_managers": {
            "fink": {
                "blah": "double blah"
            }
        }
    }

    with pytest.raises(ValueError):
        ts = TargetSelector(fink_no_credentials_config)

    required_kwargs = ["username", "group_id", "server"]
    for kw in ["username", "group_id", "server"]:
        bad_fink_config = {x: "blah blah" for x in required_kwargs if x != kw}
        bad_config = {"query_managers": {"fink": bad_fink_config}}
        with pytest.raises(ValueError):
            ts = TargetSelector(bad_config)

def test__add_targets():
    empty_config = {}
    ts = TargetSelector(empty_config)
    assert len(ts.target_lookup) == 0

    t1 = Target("target1", 30., 45.)
    ts.add_target(t1)
    assert len(ts.target_lookup) == 1
    assert "target1" in ts.target_lookup
    t1_recover = ts.target_lookup["target1"]
    assert np.isclose(t1_recover.coord.ra.deg, 30.)
    assert np.isclose(t1_recover.coord.dec.deg, 45.)

    assert t1_recover is t1


def test__evaluate_all_targets():
    def score_equal_ra(target, observatory):
        return target.coord.ra.deg, [], []

    t1 = Target("t1", 0., 45.)
    t2 = Target("t2", 45., 60.)

    ts = TargetSelector(basic_config)
    ts.add_target(t1)
    ts.add_target(t2)
    
    assert len(ts.target_lookup) == 2
    assert set(ts.target_lookup.keys()) == set(["t1", "t2"])

    t_eval = Time.now()
    ts.evaluate_all_targets(score_equal_ra) # use default "no_observatory"
    assert len(t1.score_history["no_observatory"]) == 1
    assert np.isclose(t1.score_history["no_observatory"][0][0], 0.)
    t1_score_time = t1.score_history["no_observatory"][0][1]
    assert isinstance(t1_score_time, Time)
    assert abs(t1_score_time.mjd - t_eval.mjd) < 2. / 86400. # less than 2 seconds.