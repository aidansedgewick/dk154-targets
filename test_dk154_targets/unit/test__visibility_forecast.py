import datetime

import numpy as np

import pandas as pd

from astropy.coordinates import EarthLocation
from astropy.time import Time

from dk154_targets.target import Target
from dk154_targets.visibility_forecast import VisibilityForecast

def test__initialise():
    
    polaris_history = pd.DataFrame(
        {
            "jd": [Time.now().jd, Time.now().jd-1.],  # intentionally not in order.
            "fid": [1, 2],
            "magpsf": [2.0, 1.99], # Fake data.
        }
    )
    polaris = Target(
        "polaris", ra=37.95456067, dec=+89.26410897, model_fit=False
        #target_history=polaris_history
    )
    north_pole = EarthLocation(lat=90., lon=0., height=0.)

    # without time
    vf1 = VisibilityForecast(target=polaris, observatory=north_pole)
    assert Time.now().jd - vf1.t0.jd < 5. * 1 / (24. * 3600.) # Should initialise in <5 sec.

    test_time = Time(datetime.datetime(year=2022, month=1, day=1)) # Happy New Year!
    vf2 = VisibilityForecast(target=polaris, observatory=north_pole, t0=test_time)