[![codecov](https://codecov.io/gh/aidansedgewick/dk154-targets/branch/main/graph/badge.svg?token=RKGJ98TT9I)](https://codecov.io/gh/aidansedgewick/dk154-targets)

# dk154-targets

A tool to rank targets based (currently) on the [Fink](https://fink-broker.org) alert broker.
We 'score' targets by converting features of their lightcurves to 'factors', and multiplying them all together.


## Install

- Clone this repo, `git clone https://github.com/aidansedgewick/dk154-targets.git`
- Preferably start a new virtualenv `python3 -m virtualenv dk154_env` and source `source dk154_env/bin/activate`
    (source each time).
- Install requirements `python3 -m pip install -r requirements.txt`
- Install this package *as developer* `python3 -m pip install -e .`

## Quickstart

- Look at `config/selector_config.yaml`. Add your fink credentials
- Add any observatories you're interested in.
- Start the selector: `python3 dk154_targets/main.py`

## Main elements

### TargetSelector

`TargetSelector` is the main part of the tool. TargetSelector knows a list of targets, and ranks them
according to their computed score.

A simple way to start the target selector is:
```
from dk154_targets import TargetSelector

# selector = TargetSelector.from_config(config_file="path/to/config")
selector = TargetSelector.from_config() # Read from default location config/selector_config.yaml
selector.start()
```

This will (infintely) loop:
- listen for fink alerts 
- update models for targets who have new data
- score targets (for each observatory)
- reject bad targets, based on the score
- build the ranked target list (for each observatory)
- sleep for some time (default 5 minutes)


The `TargetSelector` has:
- `target_lookup`: a dict with {"ZTF22abcdef": `Target`} pairs.
- `FinkQueryManager`, which is responsible for asking the Fink database for alerts.
- `observatories`: a list of observatories to consider, the first is *always* `None`.

### Target

Some important attributes `Target` has:
- `objectId` (eg. ZTF22abcdef), 
- `coord` (an astropy SkyCoord), 
- `ra`, `dec` (in deg)
- `target_history`, which is a pd.DataFrame,
- `base_score` (default=100.). This is multiplied by all the 'factors' which are described in the scoring function (see below).
- `models` - a list of models, which are useful to access during scoring.

### Targets of Opportunity

If there's another target that you find that you want to include in the target list, but it's not going to be found
by the fink topics you've chosen, you can add it by including a yaml file in the `targets_of_opportunity` directory.
The TargetSelector periodically checks this folder for new targets, and then deletes the yaml.

your yaml should look like:
```
objectId: ZTF22abcdef
# ra: # these are optional if it's a ZTF target. Fink returns the ra, dec of the target.
# dec:
# base_score: 1000 # if your target is inherentely more interesting to a target.  


### Scoring function

Change the way the targets are scored by providing a new scoring function.
You can write your own scoring function, although there is a default one to rank supernova targets.

Your function should have the two arguements and `**kwargs`:
 - `target`, which is an instance of [`dk154_targets.target.Target`](https://github.com/aidansedgewick/dk154-targets/blob/main/dk154_targets/target.py#L30),
 - `observatory`, which will be `None`, or an [`astroplan.observer.Observer`](https://astroplan.readthedocs.io/en/latest/api/astroplan.Observer.html#astroplan.Observer)
 which is similar to `astropy.coordinates.EarthLocation`, but with some nice extra functions.
 - `**kwargs`, which will contain `t_ref` (an astropy time, the time at the start of the loop).
You should give your scoring function to the TargetSelector on `start`.

- If your target is "bad" and should be removed from the target list
(ie, you'll never be interested in it), the score returned should be `np.nan`.
- If you may want to observe it in future, but not include it in the next ranked list 
(eg. because it is currently below the horizon), the score should be negative (ie, < 0). ##>
- The target with the highest score will be in the first row of the output.

Example:

```
import pandas as pd

from astropy import units as u
from astropy.time import Time

from dk154_targets import TargetSelector

def prefer_brightest_targets(target, observatory, **kwargs):
    """
    Choose the brightest targets, but not targets who are 
    currently below 30 degrees altitude.
    Reject targets whose last magnitude was fainter than 22.5
    """

    t_ref = kwargs.get(
    score = target.base_score # defaults to 100.
    score_comments = []
    reject_comments = []

    target.target_history.sort_values("jd", inplace=True) # Make sure the last row is the latest.
    last_mag = target.target_history["magpsf"].values[-1] # The last value in the column.

    if last_mag > 22.5:
        score = np.nan # This is so faint we'll never want to observe it...
        reject_comments.append(f"too faint, mag={last_mag:.2f}")
    else:
        last_flux = 10 ** -0.4 * (last_mag - 22.5) # brighter objects have higher flux.
        score = score * last_flux
        score_comments.append(f"mag={last_mag:.2f}")

    if observatory is not None:
        # observatory is an `astroplan.observer.Observer`
        t_now = Time.now()
        target_altaz = observer.altaz(t_now, target.coord) # target.coord is a SkyCoord
        if target_altaz.alt < 30 * u.deg:
            # much too low!
            score = -1.0 # Don't delete this target, but not interested now.
            score_comments.append("target too low")

    return score, score_comments, reject_comments
        

selector = TargetSelector.from_config() # Reads from the default config file.
selector.start(scoring_function=prefer_brightest_targets)

```

### Modelling function

In future there should be the option to change the models built. Currently only the default one works,
as the functions which plot the lightcurves calls `model.bandflux()`.

You should be able to write your own modelling function. See dk154_targets/modelling.py for the default example.

The function should have one parameter, `target`, which will be an instance of `Target`.

```
import numpy as np

import sncosmo

from dk154_targets import TargetSelector

def build_sncosmo_model(target):
    model = sncosmo.Model(source="salt2")

    data = target.target_history

    try:
        fitted_model = # model fitting code goes here
    except Exception as e:
        print(f"{target.objectId} fitting failed!")
        print(e)
        fitted_model = None
    
    return fitted_model

selector = TargetSelector.from_config() # Reads from the default config file.
selector.start(scoring_function=prefer_brightest_targets, modelling_function=build_sncosmo_model)
```





