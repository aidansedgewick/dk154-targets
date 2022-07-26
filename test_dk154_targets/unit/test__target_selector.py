import os
import pytest
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from astropy.coordinates import EarthLocation
from astropy.time import Time

from astroplan import Observer

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


class BasicModel:
    def __init__(self):
        pass



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
    assert all(isinstance(obs, Observer) for obs in selector.observatories[1:])
    assert np.isclose(selector.observatories[1].location.lon.deg, -70.73) # of_site worked correctly.
    assert np.isclose(selector.observatories[1].location.lat.deg, -29.256667)
    assert isinstance(selector.observatories[1], Observer)

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
    #assert default_path_rel == Path("config/selector_config.yaml")
    #assert TargetSelector.default_selector_config_path.exists()

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


def test__remove_bad_targets():
    empty_config = {}
    ts = TargetSelector(empty_config)
    assert ts.observatories[0] is None 
    assert len(ts.observatories) == 1

    t1 = Target("t1", 0., 60.,)
    t2 = Target("t2", 0., -20.,)
    t3 = Target("t3", 0., 40)
    t4 = Target("t4", 0., 0.)
    ts.add_target(t1)
    ts.add_target(t2)
    ts.add_target(t3)
    ts.add_target(t4)

    def reject_above_dec30(target, observatory):
        if target.dec > 30:
            return np.nan, [], []
        else:
            return 99., [], []

    ts.evaluate_all_targets(reject_above_dec30) # default observatory=None
    assert len(ts.target_lookup) == 4 # Haven't removed anything yet...

    assert not np.isfinite(t1.score_history["no_observatory"][0][0])
    assert np.isfinite(t2.score_history["no_observatory"][0][0])
    assert np.isclose(t2.score_history["no_observatory"][0][0], 99.)
    assert not np.isfinite(t3.score_history["no_observatory"][0][0])
    assert np.isfinite(t4.score_history["no_observatory"][0][0])
    assert np.isclose(t2.score_history["no_observatory"][0][0], 99.)
    
    ts.remove_bad_targets()

    assert len(ts.target_lookup) == 2
    assert set(ts.target_lookup.keys()) == set(["t2", "t4"])


def test__model_targets():
    empty_config = {}
    ts = TargetSelector(empty_config)
    ts.add_target(Target("t0", 50., 30.))
    assert ts.target_lookup["t0"].updated
    ts.model_targets(None)
    assert len(ts.target_lookup["t0"].models) == 0
    assert ts.target_lookup["t0"].updated
    
    def build_basic_model(target):
        return BasicModel()

 
    ts = TargetSelector(empty_config)
    ts.add_target(Target("target1", 34., -4.))
    ts.add_target(Target("target2", 50., 25.))
    ts.add_target(Target("target3", 270., 65.))

    assert len(ts.target_lookup) == 3
    for objectId, target in ts.target_lookup.items():
        assert target.updated
        assert len(target.models) == 0

    ts.model_targets(build_basic_model)
    for objectId, target in ts.target_lookup.items():
        assert not target.updated
        assert len(target.models) == 1
        assert isinstance(target.models[0], BasicModel)

    ts.model_targets(build_basic_model, lazy_modelling=True)
    for objectId, target in ts.target_lookup.items():
        assert not target.updated
        assert len(target.models) == 1
        # No extra models added!

    ts.model_targets(build_basic_model, lazy_modelling=False)
    for objectId, target in ts.target_lookup.items():
        assert not target.updated
        assert len(target.models) == 2
        # New model added!


def test__model_and_score():
    pass


def test__targets_of_opportunity():
    opp_targets_config = {
        "targets_of_opportunity_path": "test_dk154_targets/test_targets_of_opportunity"
    }

    expected_opp_target_path = paths.base_path / "test_dk154_targets/test_targets_of_opportunity"
    if expected_opp_target_path.exists():
        for file_path in expected_opp_target_path.glob("*"):
            assert "test_dk154_targets" in file_path.parts
            os.remove(file_path)
        expected_opp_target_path.rmdir()
    assert not expected_opp_target_path.exists()

    ts = TargetSelector(opp_targets_config)
    assert expected_opp_target_path.exists()
    assert len(ts.target_lookup) == 0

    assert ts.targets_of_opportunity_path.parts[-1] == "test_targets_of_opportunity"
    t_opp_path = ts.targets_of_opportunity_path
    f_list = ts.targets_of_opportunity_path.glob("*.yaml")
    assert sum([1 for f in f_list]) == 0 # f_list from glob is a generator, not a list...

    # Read a basic example - there will be no fink data for this.
    opp_target1 = dict(
        objectId="TargetOpp1", ra=30., dec=45.
    )
    opp_target1_path =  ts.targets_of_opportunity_path / "target_opp1.yaml"
    assert not opp_target1_path.exists()
    with open(opp_target1_path, "w+") as f:
        yaml.dump(opp_target1, f)
    assert opp_target1_path.exists()

    f_list = ts.targets_of_opportunity_path.glob("*.yaml")
    assert sum([1 for f in f_list]) == 1
    ts.check_for_targets_of_opportunity()
    assert len(ts.target_lookup) == 1
    assert "TargetOpp1" in ts.target_lookup
    t1 = ts.target_lookup["TargetOpp1"]
    assert isinstance(t1, Target)
    assert np.isclose(t1.base_score, 100.)

    f_list = ts.targets_of_opportunity_path.glob("*.yaml")
    assert sum([1 for f in f_list]) == 0 # The file has been deleted!

    # Example with no objectID - should return None.
    opp_target2 = dict()
    opp_target2_path =  ts.targets_of_opportunity_path / "target_opp1.yaml"
    assert not opp_target2_path.exists()
    with open(opp_target2_path, "w+") as f:
        yaml.dump(opp_target2, f)
    ts.check_for_targets_of_opportunity()
    assert len(ts.target_lookup) == 1

    # clean up
    expected_opp_target_path.rmdir()
    assert not expected_opp_target_path.exists()

    
def test__build_ranked_target_list():
    ranked_list_config = {
        "target_list_dir": "test_dk154_targets/ranked_lists",
        "observatories": {
            "fake_observatory": {"lat": 10., "lon": 90., "height": 10.}
        }
    }

    expected_target_list_dir = paths.base_path / "test_dk154_targets/ranked_lists"
    if expected_target_list_dir.exists():
        for file_path in expected_target_list_dir.glob("*"):
            assert "test_dk154_targets" in file_path.parts
            os.remove(file_path)
        expected_target_list_dir.rmdir()
    
    assert not expected_target_list_dir.exists()

    def test_scoring_function(target, observatory):
        obs_factor = observatory.location.height.to("m").value if observatory is not None else 0.

        if target.dec < 30.0:
            score = -(target.dec + obs_factor)
        else:
            score = np.nan

        return score, [], []

    ts = TargetSelector(ranked_list_config)
    assert len(ts.observatories) == 2

    assert expected_target_list_dir.exists()

    ts.add_target(Target("t1", ra=30., dec=-40.)) # Both lists
    ts.add_target(Target("t2", ra=60., dec=-60.)) # Both lists
    ts.add_target(Target("t3", ra=65., dec=-5.)) # no_obs only.
    ts.add_target(Target("t4", ra=80., dec=10.)) # neither list
    ts.add_target(Target("t5", ra=85., dec=25.)) # neither list, should not reject
    ts.add_target(Target("t6", ra=92., dec=32.)) # should reject.
    ts.add_target(Target("t7", ra=95., dec=45.)) # reject.

    exepcted_no_obs_path = (
        paths.base_path / 
        "test_dk154_targets/ranked_lists/no_observatory_ranked_list.csv"
    )
    assert not exepcted_no_obs_path.exists()

    expected_fake_obs_path = (
        paths.base_path / 
        "test_dk154_targets/ranked_lists/fake_observatory_ranked_list.csv"
    )
    assert not expected_fake_obs_path.exists()

    for observatory in ts.observatories:
        ts.evaluate_all_targets(test_scoring_function, observatory=observatory)

    assert len(ts.target_lookup) == 7
    ts.remove_bad_targets()
    assert len(ts.target_lookup) == 5
    for observatory in ts.observatories:
        ts.build_ranked_target_list(observatory, plots=False)

    assert exepcted_no_obs_path.exists()
    no_obs_list = pd.read_csv(exepcted_no_obs_path)
    assert len(no_obs_list) == 3
    assert no_obs_list.iloc[0].objectId == "t2"
    assert np.isclose(no_obs_list.iloc[0].score, 60)
    assert no_obs_list.iloc[1].objectId == "t1"
    assert np.isclose(no_obs_list.iloc[1].score, 40)
    assert no_obs_list.iloc[2].objectId == "t3"
    assert np.isclose(no_obs_list.iloc[2].score, 5)


    assert expected_fake_obs_path.exists()
    fake_obs_list = pd.read_csv(expected_fake_obs_path)
    assert len(fake_obs_list) == 2
    assert fake_obs_list.iloc[0].objectId == "t2"
    assert np.isclose(fake_obs_list.iloc[0].score, 50)
    assert fake_obs_list.iloc[1].objectId == "t1"
    assert np.isclose(fake_obs_list.iloc[1].score, 30)

    os.remove(exepcted_no_obs_path)
    assert not exepcted_no_obs_path.exists()
    os.remove(expected_fake_obs_path)
    assert not expected_fake_obs_path.exists()

    expected_target_list_dir.rmdir()
    assert not expected_target_list_dir.exists()

def test__start():
    empty_config = {
        "sleep_time": 1.0,
        "observatories": {
            "lasilla": "lasilla"
        },
        "target_list_dir": "test_dk154_targets/test_start_ranked_lists",
        "targets_of_opportunity_path": "test_dk154_targets/test_start_topp"
    }

    def basic_score(target, observatory):
        score = target.dec
        if observatory is not None:
            score = score * 2
            if (0 < target.dec) & (target.dec < 50):
                score = score * 10 # will make t1 and t2 v high scoring for "lasilla"
        if target.dec < -30:
            score = np.nan
        return score, [], []

    def build_basic_model(target):
        return BasicModel()

    topp_dir_expected = paths.base_path / "test_dk154_targets/test_start_topp"
    if topp_dir_expected.exists():
        for file_path in topp_dir_expected.glob("*"):
            os.remove(file_path)
        topp_dir_expected.rmdir()
    assert not topp_dir_expected.exists()

    ranked_list_dir_expected = paths.base_path / "test_dk154_targets/test_start_ranked_lists"
    if ranked_list_dir_expected.exists():
        for file_path in ranked_list_dir_expected.glob("*"):
            os.remove(file_path)
        ranked_list_dir_expected.rmdir()

    ts = TargetSelector(empty_config)
    assert topp_dir_expected.exists()
    assert ranked_list_dir_expected.exists()
    ts.add_target(Target("t1", ra=30., dec=30.))
    ts.add_target(Target("t2", ra=45., dec=-40.))
    ts.add_target(Target("t3", ra=60., dec=40.))
    ts.add_target(Target("t4", ra=75., dec=-20.))

    assert ts.fink_query_manager is None

    # add these after __init__ so we can check that the TS() makes the dir correctly.
    topp1_path = topp_dir_expected / "topp1.yaml"
    assert not topp1_path.exists()
    with open(topp1_path, "w+") as f:
        yaml.dump(dict(objectId="t5", ra=90., dec=60.), f)
    assert topp1_path.exists()
    topp2_path = topp_dir_expected / "topp2.yaml"
    assert not topp2_path.exists()
    with open(topp2_path, "w+") as f:
        yaml.dump(dict(objectId="t6", ra=105., dec=-60.), f)
    assert topp2_path.exists()

    ts.start(basic_score, build_basic_model, break_after_one=True)
    assert len(ts.target_lookup) == 4
    assert set(ts.target_lookup.keys()) == set(["t1", "t3", "t4", "t5"])

    for objectId, target in ts.target_lookup.items():
        assert len(target.models) == 1
        assert isinstance(target.models[0], BasicModel)

    assert np.isclose(ts.target_lookup["t1"].score_history["no_observatory"][0][0], 30.)
    assert np.isclose(ts.target_lookup["t1"].score_history["lasilla"][0][0], 600.)
    assert ts.target_lookup["t1"].rank_history["no_observatory"][0][0] == 3
    assert ts.target_lookup["t1"].rank_history["lasilla"][0][0] == 2

    assert "t2" not in ts.target_lookup

    assert np.isclose(ts.target_lookup["t3"].score_history["no_observatory"][0][0], 40.)
    assert np.isclose(ts.target_lookup["t3"].score_history["lasilla"][0][0], 800.)
    assert ts.target_lookup["t3"].rank_history["no_observatory"][0][0] == 2
    assert ts.target_lookup["t3"].rank_history["lasilla"][0][0] == 1

    assert np.isclose(ts.target_lookup["t4"].score_history["no_observatory"][0][0], -20)
    assert np.isclose(ts.target_lookup["t4"].score_history["lasilla"][0][0], -40.)
    assert ts.target_lookup["t4"].rank_history["no_observatory"][0][0] == 99
    assert ts.target_lookup["t4"].rank_history["lasilla"][0][0] == 99

    assert np.isclose(ts.target_lookup["t5"].score_history["no_observatory"][0][0], 60.)
    assert np.isclose(ts.target_lookup["t5"].score_history["lasilla"][0][0], 120.)
    assert ts.target_lookup["t5"].rank_history["no_observatory"][0][0] == 1
    assert ts.target_lookup["t5"].rank_history["lasilla"][0][0] == 3
    assert ts.target_lookup["t5"].target_of_opportunity

    assert not topp1_path.exists()
    assert not topp2_path.exists()

    no_obs_path = ranked_list_dir_expected / "no_observatory_ranked_list.csv"
    no_obs_list = pd.read_csv(no_obs_path)
    assert len(no_obs_list) == 3
    assert np.allclose(no_obs_list["score"].values, [60., 40., 30.,])
    assert np.allclose(no_obs_list["ra"].values, [90., 60, 30,])

    lasilla_path = ranked_list_dir_expected / "lasilla_ranked_list.csv"
    lasilla_list = pd.read_csv(lasilla_path)

    assert np.allclose(lasilla_list["score"].values, [800, 600, 120,])
    assert np.allclose(lasilla_list["ra"].values, [60., 30., 90.,])

