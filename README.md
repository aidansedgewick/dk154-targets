[![codecov](https://codecov.io/gh/aidansedgewick/dk154-targets/branch/main/graph/badge.svg?token=RKGJ98TT9I)](https://codecov.io/gh/aidansedgewick/dk154-targets)

# dk154-targets

A tool to rank targets based (currently) on the [Fink](https://fink-broker.org) alert broker.

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

The `TargetSelector` knows about a list of `Target`s, and ranks them according to their computed score.
`TargetSelector` has `FinkQueryManager`, which is responsible for asking the Fink database for 

### Target

`Target` has `objectId` (eg. ZTF22abcdef), `coord` (an astropy SkyCoord), `ra`, `dec` (in deg), and `target_history`,
which is a pd.DataFrame.

### Scoring function

Change the way the targets are scored by providing a new scoring function.
You can write your own scoring function, although there is a default one to rank supernova targets.

Your function should have (exactly) two arguements `target`, which is an instance of `dk154_targets.target.Target`,
and `observatory`, which will be `None`, or `astropy.coordinates.EarthLocation` (with the added `name` attribute`).
You should give your scoring function to the TargetSelector on `start`.

- If your target is "bad" and should be removed from the target list
(ie, you'll never be interested in it), the score returned should be `np.nan`.
- If you may want to observe it, but not include it in the next ranked list, the score should be negative (<0)
- The targets with the highest score will  >


Example:

```
import pandas as pd

from dk154_targets import TargetSelector, VisibilityForecast

def prefer_brightest_targets(target, observatory):
    """
    Choose the brightest targets, but not targets who are 
    currently below 30 degrees altitude.
    Reject targets whose last magnitude was fainter than 22.5
    """

    score_comments = []
    reject_comments = []

    last_mag = target.target_history["magpsf"].values[-1] # The latest value

    if last_mag > 22.5:
        score = np.nan # This is so faint we'll never want to observe it...
        reject_comments
    else:
        last_flux = 10 ** -0.4 * (last_mag - 22.5) # brighter objects have higher flux.
        score = last_flux

    if observatory is not None:
        vf = VisibilityForecast(target, observatory) 
        next_altitude = vf.get_immediate_altitude() 
        # Returns altitude *now* if it's currently night time at the observatory,
        # else returns the altitude if the 


selector = TargetSelector(scoring_function=prefer_brightest_targets)

```













