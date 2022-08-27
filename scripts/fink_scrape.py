import logging
import os

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.time import Time, TimeDelta

from dk154_targets.queries import FinkQuery
from dk154_targets.utils import chunk_list, readstamp

from dk154_targets import paths

logger = logging.getLogger(__name__.split(".")[-1])

archive_path = paths.data_path / "archive"
archive_path.mkdir(exist_ok=True, parents=True)

alert_df_path = archive_path / "alerts"
alert_df_path.mkdir(exist_ok=True, parents=True)

lightcurve_df_path = archive_path / "light_curves"
lightcurve_df_path.mkdir(exist_ok=True, parents=True)

start_date = Time("2020-10-01 00:00:00")
end_date = Time("2021-09-30 00:00:00")
dt = TimeDelta(3*u.hour)

n_intervals = int((end_date - start_date) / dt)

fink_classes = ["SN candidate", "Early SN Ia candidate"]

alert_df_list = []

full_alert_df_path = archive_path / "alert_archive.csv"

if not full_alert_df_path.exists():

    for ii in range(n_intervals):
        t0 = start_date + ii * dt
        t1 = start_date + (ii+1) * dt

        alert_df_name = t0.strftime("%Y%m%d_%H%M%S")

        start_time = t0.strftime("%Y-%m-%d %H:%M:%S")
        end_time = t1.strftime("%Y-%m-%d %H:%M:%S")

        alert_df_outpath = alert_df_path / f"{alert_df_name}.csv"
        if alert_df_outpath.exists():
            if os.path.getsize(alert_df_outpath) < 10:
                logger.info(f"skip read {alert_df_outpath.name}")
                continue
            alert_df = pd.read_csv(alert_df_outpath)
            logger.info(f"read existing {alert_df_outpath.name}")
            if len(alert_df.columns) < 2:
                raise ValueError("no columns?!")
            alert_df_list.append(alert_df)
            continue

        logger.info(f"scrape {start_time}")

        df_list = []
        for class_ in fink_classes:
            df = FinkQuery.query_latest_alerts(
                return_df=True,
                fix_column_names=True,
                **{
                    "class": class_,
                    "n": 20000, 
                    "startdate": start_time, 
                    "stopdate": end_time,
                    "withupperlim": True,
                }
            )
            if len(df) == 0:
                continue
            df["class"] = class_.lower().replace(" ", "_")
            df_list.append(df)

        if len(df_list) > 0:
            alert_df = pd.concat(df_list)
            alert_df.query("magpsf<19.5", inplace=True)
        else:
            alert_df = pd.DataFrame(list())
        logger.info(f"save alerts to {alert_df_outpath.name}")
        alert_df.to_csv(alert_df_outpath, index=False)
        alert_df_list.append(alert_df)


    full_alert_df = pd.concat(alert_df_list)
    full_alert_df.to_csv(full_alert_df_path, index=False)

else:
    logger.info("read existing alert df")
    full_alert_df = pd.read_csv(full_alert_df_path)

brighest_alert = full_alert_df.groupby("objectId")["magpsf"].min()
assert all(brighest_alert < 19.5)

objectId_list = np.unique(full_alert_df["objectId"])




print(len(full_alert_df))
logger.info(f"scrape {len(objectId_list)} obj")


full_archive_df_path = archive_path / "full_archive.csv"
if not full_archive_df_path.exists():
    existing_objectIds = []
    missing_objectIds = []
    for objectId in objectId_list:
        lightcurve_df_outpath = lightcurve_df_path / f"{objectId}.csv"
        if lightcurve_df_outpath.exists():
            existing_objectIds.append(objectId)
        else:
            missing_objectIds.append(objectId)

    archive_df_list = []
    if len(missing_objectIds) > 0:
        chunk_size = 20
        for ii, objectId_chunk in enumerate(chunk_list(missing_objectIds, size=chunk_size)):

            objectId_str = ",".join(objId for objId in objectId_chunk)
            df = FinkQuery.query_objects(
                return_df=True, 
                objectId=objectId_str,
                withupperlim=True,
            )
            
            logger.info(f"chunk {ii+1} of {int(len(missing_objectIds)/chunk_size)+1} (n_rows={len(df)})")
            archive_df_list.append(df)
            for objectId, obj_df in df.groupby("objectId"):
                assert objectId not in existing_objectIds
                lightcurve_df_outpath = lightcurve_df_path / f"{objectId}.csv"
                obj_df.to_csv(lightcurve_df_outpath, index=False)

    for existing_path in existing_objectIds:
        lightcurve_df_outpath = lightcurve_df_path / f"{objectId}.csv"
        obj_df = pd.read_csv(lightcurve_df_outpath)
        archive_df_list.append(obj_df)

    full_archive_df = pd.concat(df_list)
    full_archive_df.to_csv(full_archive_df_path)

