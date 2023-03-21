import time
import tqdm
import logging
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table
from astropy.time import Time, TimeDelta

from dk154_targets.queries import FinkQuery
from dk154_targets.utils import chunk_list, readstamp

from dk154_targets import paths

parser = ArgumentParser()
parser.add_argument("--scrape-alerts", action="store_true", default=False)
parser.add_argument("--skip-lc-scrape", action="store_true", default=False)

args = parser.parse_args()
print(args)


logger = logging.getLogger(__name__.split(".")[-1])

fink_archive_path = paths.archive_path / "fink"

alert_df_path = fink_archive_path / "alerts"
alert_df_path.mkdir(exist_ok=True, parents=True)

lightcurve_dir_path = fink_archive_path / "light_curves"
lightcurve_dir_path.mkdir(exist_ok=True, parents=True)

start_date = Time("2020-10-01 00:00:00")
end_date = Time("2021-09-30 00:00:00")
dt = TimeDelta(3*u.hour)

n_intervals = int((end_date - start_date) / dt)

fink_classes = ["SN candidate", "Early SN Ia candidate"]

alert_df_list = []

full_alert_df_path = fink_archive_path / "alert_archive.csv"

if not full_alert_df_path.exists() or args.scrape_alerts:

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




existing_objectIds = []
missing_objectIds = []
for objectId in objectId_list:
    lightcurve_df_path = lightcurve_dir_path / f"{objectId}.csv"
    if lightcurve_df_path.exists():
        existing_objectIds.append(objectId)
        if "ZTF21acemach" in str(lightcurve_df_path):
            print("aaaaaaaaaaaaaa")
            assert lightcurve_df_path.exists()
    else:
        missing_objectIds.append(objectId)

print(len(existing_objectIds), len(missing_objectIds))

if not args.skip_lc_scrape:
    if len(missing_objectIds) > 0:
        chunk_size = 100
        for ii, objectId_chunk in enumerate(chunk_list(missing_objectIds, size=chunk_size)):

            objectId_str = ",".join(objId for objId in objectId_chunk)
            logger.info(f"chunk {ii+1} of {int(len(missing_objectIds)/chunk_size)+1}")
            t1 = time.perf_counter()
            #try:
            df = FinkQuery.query_objects(
                return_df=True, 
                objectId=objectId_str,
                withupperlim=True,
            )
            #except Exception as e:
            #    logger.warning("FAILED!")
            #    continue
            t2 = time.perf_counter()
            logger.info(f"done (n_rows={len(df)}) in {t2-t1:.2f}s")
            
            #archive_df_list.append(df)
            for objectId, obj_df in df.groupby("objectId"):
                assert objectId not in existing_objectIds
                lightcurve_df_path = lightcurve_dir_path / f"{objectId}.csv"
                obj_df.to_csv(lightcurve_df_path, index=False)
            logger.info("sleep")
            time.sleep(20)


lc_summaries_path = fink_archive_path / "light_curve_summaries.csv"
if not lc_summaries_path.exists():

    lc_gen = lightcurve_df_path.glob("*.csv")

    lc_list = [p for p in lc_gen]

    logger.info("produce summary")

    data_list = []

    for ii, lc_path in tqdm.tqdm(enumerate(lc_list), total=len(lc_list)):
        objectId = lc_path.stem
        target_history = pd.read_csv(lc_path, encoding="utf8")
        
        detections = target_history.query("tag=='valid'")
        #ulimits = target_history.query("tag=='upperlim'")
        #badquality = target_history.query("tag=='badquality'")
        detections.sort_values("jd", inplace=True)  
        
        #N_obs = {}
        #for fid, fid_detections in detections.groupby("fid"):
        #    N_obs[fid] = len(fid_detections)
        N_obs = detections.groupby("fid").size().to_dict()
        

        ra = detections["ra"].iloc[0] #.mean()
        dec = detections["dec"].iloc[0] #.mean()
        start_jd = detections["jd"].iloc[0] #.min()
        end_jd = detections["jd"].iloc[-1] #.max()
        timespan = end_jd - start_jd
        
        start_date = Time(start_jd, format="jd").strftime("%Y-%m-%d %H:%M:%S")
        end_date = Time(end_jd, format="jd").strftime("%Y-%m-%d %H:%M:%S")
        
        summary = dict(
            objectId=objectId,
            ra=ra,
            dec=dec,
            start_date=start_date, 
            end_date=end_date, 
            start_jd=start_jd, 
            end_jd=end_jd, 
            timespan=timespan,
            peak_mag=detections["magpsf"].min(),
            n_detections=len(detections), 
            N_obs_1=N_obs.get(1, 0), 
            N_obs_2=N_obs.get(2, 0), 
        )        
        data_list.append(summary)

    summary_df = pd.DataFrame(data_list)
    summary_df.to_csv(lc_summaries_path, index=False)

else:
    logger.info("read existing summaries")
    summary_df = pd.read_csv(lc_summaries_path)



full_archive_df_path = fink_archive_path / "full_archive.fits"
archive_df_list = []
if True: #not full_archive_df_path.exists():
    
    


lc_summaries_path = fink_archive_path / "light_curve_summaries.csv"





tns_path = paths.archive_path / "tns/confirmed_sne.csv"
if tns_path.exists():
    tns_data = pd.read_csv(tns_path)

    summary_copy = summary_df.copy(deep=True)
    summary_copy.sort_values("end_jd", ascending=False, inplace=True, ignore_index=True)

    tns_coord = SkyCoord(ra=tns_data["RA"], dec=tns_data["DEC"], unit=[u.hourangle, u.deg])
    fink_coord = SkyCoord(ra=summary_copy["ra"], dec=summary_copy["dec"], unit="deg")

    fink_match_idxs, tns_match_idxs, sep2d, _ = search_around_sky(fink_coord, tns_coord, 5 * u.arcsec)

    print(len(fink_match_idxs), len(np.unique(fink_match_idxs)))
    print(len(tns_match_idxs), len(np.unique(tns_match_idxs)))

    _, unique_fink_matches = np.unique(fink_match_idxs, return_index=True)
    unique_fink_mask = np.full(len(fink_match_idxs), False)
    unique_fink_mask[unique_fink_matches] = True

    _, unique_tns_matches = np.unique(tns_match_idxs, return_index=True)
    unique_tns_mask = np.full(len(tns_match_idxs), False)
    unique_tns_mask[unique_tns_matches] = True

    idx_mask = unique_fink_mask & unique_tns_mask

    print(idx_mask.sum())


    valid_fink_idxs = fink_match_idxs[ idx_mask ]
    valid_tns_idxs = tns_match_idxs[ idx_mask ]

    assert len(valid_fink_idxs) == len(np.unique(valid_fink_idxs))
    assert len(valid_tns_idxs) == len(np.unique(valid_tns_idxs))

    print(tns_data.iloc[valid_tns_idxs, :])

    res = pd.concat(
        [
            summary_copy, 
            tns_data.iloc[valid_tns_idxs, :].set_index(valid_fink_idxs)
        ], axis=1
    )

    matched_path = paths.archive_path / "fink_summary_tns_matched.csv"
    res.to_csv(matched_path, index=False)

    




