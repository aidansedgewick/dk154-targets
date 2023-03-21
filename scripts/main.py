import datetime
import logging
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from dk154_targets import TargetSelector, Target
from dk154_targets.queries import FinkQuery
from dk154_targets.query_managers import FinkQueryManager
from dk154_targets.utils import chunk_list

from dk154_targets import paths

logger = logging.getLogger("main")


# some default paths
default_fink_data_path = FinkQueryManager.fink_lightcurve_path
# default_alerce_data_path = AlerceQueryManager.alerce_lightcurve_path
default_config_path = TargetSelector.default_selector_config_path

parser = ArgumentParser()
parser.add_argument("--fink-data", nargs="?", const=default_fink_data_path, default=False)
parser.add_argument("--objectId-list", nargs="?", const=default_fink_data_path, default=False)
parser.add_argument("--fink-query", action="store_true", default=False)
parser.add_argument("--start-date", default=False, type=str, nargs="+")
parser.add_argument("--n-days", default=30, type=float)
#parser.add_argument("--new", default=False, action="store_true")
parser.add_argument("-c", "--config-file", default=default_config_path, type=str)
parser.add_argument("--faint-lim", default=18.5, type=float)

args = parser.parse_args()


choose_one_from = [
    args.fink_data, args.start_date, args.objectId_list
]
n_options = sum([bool(arg) for arg in choose_one_from])
if n_options > 1:
    print("AAAAAAAAA")
    sys.exit()

if n_options == 0:
    logger.info("new selector")




selector = TargetSelector.from_config(config_file=args.config_file)

if args.fink_data:

    fink_data_path = Path(args.fink_data)
    try:
        print_path = fink_data_path.relative_to(paths.base_path)
    except:
        print_path = fink_data_path
    if not fink_data_path.exists():
        print(f"\nNo fink_data at:\n    {fink_data_path.absolute()}")
        sys.exit()
    logger.info(f"read data from {print_path}")
    lightcurve_list = [f for f in fink_data_path.glob("*.csv")]
    logger.info(f"read {len(lightcurve_list)} files...")

    new_targets = []
    updated_targets = []
    for lightcurve_path in lightcurve_list:
        objectId = lightcurve_path.stem
        fink_lightcurve = pd.read_csv(lightcurve_path)
        target = selector.target_lookup.get(objectId, None)
        if target is None:
            target = Target.from_fink_lightcurve(objectId, fink_lightcurve)
            if target is not None:
                selector.add_target(target)
                new_targets.append(objectId)
        else:
            target.fink_data.lightcurve = fink_lightcurve
            updated_targets.append(objectId)
    logger.info(f"FINK: load {len(new_targets)}, add data for {len(updated_targets)}")

msg = (
    f"\n\n"
    f"Start with {len(selector.target_lookup)} targets.\n"
)
selector.start() #observatory=observatory)

"""
elif args.n_days or args.start_date:
    if args.start_date and not args.n_days:
        t0 = datetime.datetime.strptime(" ".join(args.start_date), "%Y %m %d %H %M %S")
        #t0 = datetime.datetime(2022, 7, 8, 19, 0, 0)
        ndays = int((Time.now().datetime - t0).days) + 1.
    elif args.n_days and not args.start_date:
        ndays = args.n_days
        t0 = Time.now() - (ndays + 1.) * u.day # +1 for messy hours
    else:
        print(f"start_date: {args.start_date} and n_days: {args.n_days}\n  - provide one, not both.")
        sys.exit()

    logger.info(f"scrape {ndays:.1f} days: t0={t0.strftime('%Y-%m-%d')}")


    interval = 1 * u.hour
    n_intervals = int(ndays * 24 * u.hour / interval)

    df_list = []
    for ii in range(n_intervals):
        start_time = Time(t0) + ii * interval
        start_str = start_time.datetime.strftime("%Y-%m-%d %H:%M:%S")
        end_time = Time(t0) + (ii + 1) * interval
        end_str = end_time.datetime.strftime("%Y-%m-%d %H:%M:%S")

        logger.info(f"{start_str} to {end_str}")
        for class_ in ["SN candidate", "Early SN Ia candidate"]:
            df = FinkQuery.query_latest_alerts(
                return_df=True,
                fix_column_names=True,
                **{
                    "class": class_,
                    "n": 20000, 
                    "startdate": start_str, 
                    "stopdate": end_str,
                    "withupperlim": True,
                }
            )
            if len(df) == 0:
                continue
            logger.info(f"{len(df)} new {class_} targets")
            df_list.append(df)

    targets = pd.concat(df_list)
    logger.info(f"{len(targets)} targets scraped")
    query_str = f"(magpsf < {args.faint_lim}) & (lapse < 30)"
    good_targets = targets.query(query_str)
    logger.info(f"{len(good_targets)} bright, unique targets <{args.faint_lim}, <30 days old")


    fink_kwargs = dict(withupperlim=True)
    objectId_list = np.unique(good_targets["objectId"])
    if isinstance(objectId_list, str):
        objectId_list = [objectId_list]
    logger.info(f"initialise {len(objectId_list)} objects")

    chunk_size=20
    df_list = []
    for ii, objectId_chunk in enumerate(chunk_list(objectId_list, size=chunk_size)):
        objectId_str = ",".join(objId for objId in objectId_chunk)
        df = FinkQuery.query_objects(return_df=True, objectId=objectId_str, **fink_kwargs)
        
        logger.info(f"chunk {ii+1} of {int(len(objectId_list)/chunk_size)+1} (n_rows={len(df)})")
        df_list.append(df)
    df = pd.concat(df_list)


    n_groups = len(np.unique(df["objectId"]))
    for ii, (objectId, fink_data) in enumerate(df.groupby("objectId")):
        if ii % 10 == 0:
            logger.info(f"target {ii+1} of {n_groups}")

        fink_data_path = paths.data_path / "fink/lightcurves"
        fink_data_path.mkdir(exist_ok=True, parents=True)

        fixed_df = FinkQuery.fix_column_names(fink_data)
        
        object_data_path = fink_data_path / f"{objectId}.csv"
        fixed_df.to_csv(object_data_path, index=False)
"""

