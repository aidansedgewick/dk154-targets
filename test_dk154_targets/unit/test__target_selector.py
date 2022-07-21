import pytest
from pathlib import Path

import numpy as np

from astropy.coordinates import EarthLocation

from dk154_targets.query_managers import FinkQueryManager
from dk154_targets.target import Target
from dk154_targets.target_selector import TargetSelector

basic_config = {
    "sleep_time": 2.0,
    "query_managers": {
        "fink": {
            "credential": {
                "username" :"test_user",
                "group_id": "user_group",
                "servers": "192.0.2.0"
            },
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

    selector = TargetSelector(basic_config)
    
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
        set(["username", "group_id", "servers"])
    )
    assert isinstance(selector.fink_query_manager.topics, list)
    assert set(selector.fink_query_manager.topics) == set(["cool_space_targets"])

    assert isinstance(selector.target_lookup, dict)
    assert len(selector.target_lookup) == 0



"""
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
"""