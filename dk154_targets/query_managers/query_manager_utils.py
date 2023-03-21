import logging
from pathlib import Path

import numpy as np

import pandas as pd

from astropy.time import Time

from dk154_targets.target import Target, TargetData

logger = logging.getLogger(__name__.split(".")[-1])

def get_file_update_interval(file_path: Path, t_ref: Time) -> float:
    "return interval IN DAYS between modify time (st_mtime) of file, and t_ref"
    file_path = Path(file_path)
    if not file_path.exists():
        return np.inf
    file_mod_time = Time(file_path.stat().st_mtime, format="unix")
    dt = t_ref - file_mod_time
    return dt.jd

def update_target_data_lightcurve(
    target_data: TargetData, updates, 
    keep_existing=True, integrate_missing=False,
    date_col: str="jd", date_tol: float=1./(60.*24.)
):

    ###==== the simple cases: no data at all, or simple concat =====###
    if target_data.lightcurve is None:
        target_data.lightcurve = updates
        target_data.lightcurve.sort_values(date_col, inplace=True)
        logger.info("no existing lightcurve, use new data")
        return


    ### six cases:

    #  1:    2:    3:    4:    5:    6:
    #  l u   l u   l u   l u   l u   l u
    #    |     |         |     |       |
    #  |     | |   |     |     |     | |
    #  |     |     | |   |     | |   | |
    #  |     |     |       |     |     |
    #
    # only really ever expect 1,2,3

    lc_min = target_data.lightcurve[date_col].min()
    lc_max = target_data.lightcurve[date_col].max()
    update_min = updates[date_col].min()
    update_max = updates[date_col].max()

    if integrate_missing:
        raise NotImplementedError()

    # This is UGLY and SLOW. TODO improve!
    cases_matched = []
    if update_min > lc_max:
        # case 1
        target_data.lightcurve = pd.concat([target_data.lightcurve, updates])
        target_data.lightcurve.sort_values(date_col, inplace=True)
        target_data.lightcurve.reset_index(inplace=True)
        logger.info("new data: simple concat")
        cases_matched.append(1)

    if (update_min <= lc_max) and (update_max > lc_max):
        # case 2
        if keep_existing:
            existing_data = target_data.lightcurve
            new_data = updates.query(f"{date_col} > @lc_max")
            logger.info("keep existing data in overlap")
        else:
            existing_data = target_data.lightcurve.query(f"{date_col} < @update_min")
            new_data = updates
            logger.info("keep updates in overlap")
        lc = pd.concat([existing_data, new_data])
        target_data.lightcurve = lc
        cases_matched.append(2)

    if (update_min >= lc_min) and (update_max <= lc_max):
        # case 3
        dates_are_close = np.array(
            [
                any(np.isclose(u, target_data.lightcurve[date_col].values, atol=date_tol)) 
                for u in updates[date_col].values
            ]
        )
        print(dates_are_close)
        if all(dates_are_close):
            logger.info(f"all {len(updates)} updates already in lc")
        else:
            raise ValueError("updates overlap")
        cases_matched.append(3)

    if len(cases_matched) > 1:
        raise ValueError(f"matched cases {cases_matched}")
    if len(cases_matched) == 0:
        raise ValueError("no cases matched?!")
    return

    