import pytest

import numpy as np

from astropy.coordinates import EarthLocation

from dk154_targets.target import Target
from dk154_targets.target_selector import TargetSelector


def test__init():
    empty_config = {}
    selector = TargetSelector(empty_config)
    
    assert isinstance(selector.selector_config, dict)
    assert len(selector.selector_config) == 0

    assert isinstance(selector.observatory_config, dict)
    assert len(selector.observatory_config) == 0
    assert len(selector.observatories) == 1
    assert selector.observatories[0] is None

    assert isinstance(selector.target_lookup, dict)
    assert len(selector.target_lookup) == 0


    ## now with observatories
    config_with_observatories = dict(
        observatories = dict(
            lapalma="lapalma",
            astrolab=dict(lat=55.0, lon=1.0, height=30.)
        )
    )
    ts = TargetSelector(config_with_observatories)
    assert len(ts.observatories) == 3
    assert ts.observatories[0] is None
    assert isinstance(ts.observatories[1], EarthLocation)
    assert ts.observatories[1].name == "lapalma"
    assert isinstance(ts.observatories[2], EarthLocation)
    assert ts.observatories[2].name == "astrolab"

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
