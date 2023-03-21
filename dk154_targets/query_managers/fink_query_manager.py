import json
import logging
import os
import pickle
import requests
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.time import Time

from fink_client.avroUtils import write_alert, _get_alert_schema, AlertReader
from fink_client.consumer import AlertConsumer

from dk154_targets import paths
from dk154_targets.target import Target
from dk154_targets.queries import FinkQuery

from dk154_targets.query_managers import GenericQueryManager
from dk154_targets.query_managers.query_manager_utils import (
    get_file_update_interval, update_target_data_lightcurve
)

from dk154_targets.utils import readstamp

logger = logging.getLogger(__name__.split(".")[-1])

def get_query_results_path(query_name):
    return FinkQueryManager.fink_query_results_path / f"{query_name}.csv"

def get_fink_cutout_path(objectId: str):
    return FinkQueryManager.fink_cutout_path / f"{objectId}.pkl"

def read_existing_cutouts(cutout_path):
    cutout_path = Path(cutout_path)
    try:
        with open(cutout_path, "rb") as f:
            cutouts = pickle.load(f)
    except Exception as e:
        logger.error(f"reading cutouts at {cutout_path.stem}")
        logger.error(e)
        cutouts = {}
    assert isinstance(cutouts, dict)
    return cutouts

def write_cutouts(cutouts: dict, cutout_path: Path):
    cutout_path = Path(cutout_path)
    try:
        with open(cutout_path, "wb+") as f:
            pickle.dump(cutouts, f)
    except Exception as e:
        logger.error(f"writing cutouts {cutout_path.stem}")
        logger.error(e)
    return None

def get_fink_lightcurve_path(objectId: str):
    return FinkQueryManager.fink_lightcurve_path / f"{objectId}.csv"

def save_fink_lightcurve(lightcurve_df: pd.DataFrame, lightcurve_path: Path):
    lightcurve_path = Path(lightcurve_path)
    logger.info(f"{lightcurve_path.stem} saving lightcurve")
    lightcurve_df.to_csv(lightcurve_path, index=False)

# def target_from_fink_query(
#     objectId, ra=None, dec=None, base_score=None, **kwargs
# ):
#     fink_lightcurve = FinkQuery.query_objects(objectId=objectId, **kwargs)
#     if fink_lightcurve is None:
#         logger.warn(f"no fink_lightcurve {objectId} query")
#         return None
#     if isinstance(fink_lightcurve, pd.DataFrame) and fink_lightcurve.empty:
#         logger.warning(f"fink data is None")
#         return None
#     return cls.from_fink_lightcurve(
#         objectId, fink_lightcurve, ra=ra, dec=dec, base_score=base_score
#     )

# def target_from_fink_lightcurve(
#     objectId, fink_lightcurve, ra=None, dec=None, base_score=None
# ):
#     if isinstance(fink_lightcurve, str) or isinstance(fink_lightcurve, Path):
#         logger.info(f"interpret {fink_lightcurve} as path")
#         fink_lightcurve = pd.read_csv(fink_lightcurve)

#     fink_lightcurve = fink_lightcurve.copy(deep=True)
#     fink_lightcurve.sort_values("jd", inplace=True)
#     if "tag" in fink_lightcurve.columns:
#         detections = fink_lightcurve.query("tag=='valid'")
#     else:
#         detections = fink_lightcurve
#     if detections.empty:
#         logger.warning(f"init {objectId}: no valid detections!")
#         return None
#     if ra is None or dec is None:
#         ra = fink_lightcurve["ra"].dropna().values[-1]
#         dec = fink_lightcurve["dec"].dropna().values[-1]
#     if (not np.isfinite(ra)) or (not np.isfinite(dec)):
#         logger.warning(f"ra and dec should not be {ra}, {dec}")
#         return None
#     target = cls(objectId, ra, dec, fink_lightcurve=fink_lightcurve, base_score=base_score)
#     return target

class FinkQueryManager(GenericQueryManager):
    
    name = "fink"
    default_num_alerts = 5
    default_timeout = 10

    fink_data_path = paths.fink_data_path
    fink_cutout_path = paths.fink_data_path / "cutouts"
    fink_lightcurve_path = paths.fink_data_path / "lightcurves"
    fink_query_results_path = paths.fink_data_path / "query_results"

    def __init__(self, consumer_config: dict, target_lookup: dict):
        self.consumer_config = consumer_config
        self.credential_config = {
            x: self.consumer_config.get(x) for x in ["username", "group_id", "bootstrap.servers"]
        }
        if any([x is None for x in self.credential_config.values()]):
            msg = (
                "your selector_config should contain fink:\n"
                "query_managers:\n  fink:\n    "
                "username: <username>\n    group_id: <group-id>\n    bootstrap.servers: <server>\n"
            )
            raise ValueError(msg)

        topics = self.consumer_config.get("topics", None)
        self.topics = topics or ["fink_sso_ztf_candidates_ztf"]
        query_classes = self.consumer_config.get("query_classes", None)
        self.query_classes = query_classes or ["SN candidate", "Early SN Ia candidate"]
        self.last_query_update = Time(0., format="unix")
        self.query_update_interval = self.consumer_config.get("query_update_interval", 0.25)
        self.query_lookback_time = self.consumer_config.get("query_lookback_time", 20.)
        self.cutout_update_interval = self.consumer_config.get("cutout_update_interval", 2.)

        self.fink_data_path.mkdir(exist_ok=True, parents=True)
        self.fink_cutout_path.mkdir(exist_ok=True, parents=True)
        self.fink_lightcurve_path.mkdir(exist_ok=True, parents=True)
        self.fink_query_results_path.mkdir(exist_ok=True, parents=True)

        # The target_lookup!
        self.target_lookup = target_lookup


    def listen_for_alerts(self):
        num_alerts = self.consumer_config.get("num_alerts", self.default_num_alerts)
        timeout = self.consumer_config.get("timeout", self.default_timeout)
        latest_alerts = []
        logger.info(f"listen for {round(timeout)} sec, for {num_alerts} alerts")
        topic_counter = {topic: 0 for topic in self.topics}
        for current_topic in self.topics:
            with AlertConsumer([current_topic], self.credential_config) as consumer:
                # consumer expects list of topics.
                logger.info(f"poll for {current_topic}")
                for ii in range(num_alerts):
                    try:
                        topic, alert, key = consumer.poll(timeout=timeout)
                    except json.decoder.JSONDecodeError as e:
                        logger.error(traceback.format_exc())
                        continue
                    if any([x is None for x in [topic, alert, key]]):
                        logger.info(f"break after {len(latest_alerts)} alerts")
                        break
                    topic_counter[topic] = topic_counter[topic] + 1
                    latest_alerts.append( (topic, alert, key,) )
        summary_str = ", ".join(
            f"{v} {k}"for k, v in topic_counter.items() if v > 0
        ) or "no alerts"
        logger.info(f"recieve {summary_str}")
        return latest_alerts


    def read_simulated_alerts(self):
        simulated_alerts_dir = Path(simulated_alerts)
        simulated_alerts = [path for path in simulated_alerts_dir.glob("*.json")]
        latest_alerts = []
        for simulated_alert_path in sorted(simulated_alerts):
            with open(simulated_alert_path, "r") as f:
                alert = json.load(f)
            latest_alerts.append( (None, alert, None) ) # topic, alert, key.
        for alert_path in simulated_alerts:
            os.remove(alert_path)
        return latest_alerts
        
             
    def process_alerts(
        self, latest_alerts, dump_alerts=True, simulated_alerts=False, t_ref=None, delta_t=None
    ):
        """
        TODO add docs
        """
        t_ref = t_ref or Time.now()

        logger.info(f"process {len(latest_alerts)} new alerts!")
        new_targets = []
        updated_targets = []
        saved_cutouts = []
        for topic, alert, key in latest_alerts:
            if not simulated_alerts:
                if dump_alerts:
                    self.dump_alert(topic, alert, key)
                new_alert = alert["candidate"]

                extra_keys = [
                    'candid', 'objectId', 'timestamp', 'cdsxmatch', 
                    'rf_snia_vs_nonia', 'snn_snia_vs_nonia', 'snn_sn_vs_all', 
                    'mulens', 'roid', 'nalerthist', 'rf_kn_vs_nonkn'
                ]
                new_alert.update({k: alert[k] for k in extra_keys} )
                new_alert["topic"] = topic

                alert_history = pd.DataFrame(alert["prv_candidates"])
                if len(alert_history) == 0:
                    logger.info(f"skip {objectId}, no alert_history")
                    continue
            else:
                new_alert = alert
                new_alert["topic"] = topic

            objectId = new_alert["objectId"]


            # Do we already know about this object?
            target = self.target_lookup.get(objectId, None)

            fink_lightcurve_path = get_fink_lightcurve_path(objectId)
            dt_update = get_file_update_interval(fink_lightcurve_path, t_ref=t_ref)

            if (target is None) or (target.fink_data.lightcurve is None):
                # if first (expr) is True, then it ignores the second (expr)...
                try:
                    # always query for new lc on alert - never try to read existing.
                    logger.info(f"{objectId} query lc...")
                    alert_history = FinkQuery.query_objects(
                        objectId=alert["objectId"], 
                        withupperlim=True,
                        return_df=True, 
                        fix_column_names=True,
                    )
                except Exception as e:
                    logger.warning(f"{objectId} fink query failed")
                    logger.info(f"{objectId} use history from alert (len={len(alert_history)})")
                    print(e)
                delta_t = new_alert.get("delta_t", None)
                if delta_t is not None:
                    # This is useful for simulating alert streams...
                    logger.info(f"modify jd: jd->jd + {delta_t}")
                    alert_history["jd"] = alert_history["jd"] + delta_t
                    alert_history.query("jd < @t_ref.jd", inplace=True)
                alert_history.sort_values("jd", inplace=True)
                if len(alert_history) < 3:
                    logger.info(f"skip {objectId}, len history={len(alert_history)}")
                    continue
                
                target = Target.from_fink_lightcurve(objectId, fink_lightcurve=alert_history)
                if target is not None:                    
                    self.target_lookup[objectId] = target
                else:
                    logger.warning(f"{objectId} Target.from_fink_lightcurve() failed")
                    continue
                if target.fink_data.lightcurve is None:
                    raise ValueError(
                        f"{objectId} fink_data.lightcurve is None after Target.from_fink_lightcurve()"
                    )
                new_targets.append(objectId)
            else:
                logger.info(f"{objectId} target already exists")
                updated_targets.append(objectId) # keep track of who we've updated.

            assert objectId in self.target_lookup # we've just added it - it should be there...!
            logger.info(f"{objectId} update target")
            target = self.target_lookup[objectId]
            if "tag" in target.fink_data.lightcurve and "tag" not in new_alert:
                new_alert["tag"] = "valid" # Fink will not broadcast 'badquality' or 'upperlim'??

            new_alert_df = pd.DataFrame( 
                new_alert, 
                index=[len(target.fink_data.lightcurve)], # needs an index - assume it's going at the end.
            )
            alert_time = Time(new_alert['jd'], format="jd")
            alert_time_str = alert_time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"alert jd = {alert_time.jd:.5f}")
            logger.info(f"         = {alert_time_str}")

            # preferable to keep old data as it has more "value added" data [eg classifications.]
            self.update_target_lightcurve(target, new_alert_df, keep_existing=True)

            if not simulated_alerts:
                if not fink_lightcurve_path.exists():
                    save_fink_lightcurve(target.fink_data.lightcurve, fink_lightcurve_path)
                else:
                    old_lightcurve = pd.read_csv(fink_lightcurve_path)
                    if old_lightcurve["jd"].max() < target.fink_data.lightcurve["jd"].max():
                        save_fink_lightcurve(target.fink_data.lightcurve, fink_lightcurve_path)
            else:
                logger.info("simulated alerts! skip save lightcurve...")

            ## Do we need to update the cutouts?
            cutout_update_time = target.fink_data.meta.get("cutout_update_time", None)
            if cutout_update_time is None or alert_time.jd > cutout_update_time.jd:
                logger.info(f"{target.objectId} keep alert cutouts")
                for imtype in FinkQuery.imtypes:
                    target.fink_data.cutouts[imtype] = readstamp(
                        alert.get('cutout'+imtype, {}).get('stampData', None)
                    )
                target_cutout_path = get_fink_cutout_path(target.objectId)
                write_cutouts(target.fink_data.cutouts, target_cutout_path)
                saved_cutouts.append(objectId)
                target.fink_data.meta["cutout_update_time"] = t_ref
            else:                
                logger.info(f"{target.objectId} ignore alert cutouts")
            print()

        logger.info(f"added {len(new_targets)}, updated {len(updated_targets)}")
        logger.info(f"saved {len(saved_cutouts)} cutouts")
        return None


    def dump_alert(self, topic, alert, key, outdir=None):
        """
        method directly from the fink-client scripts
        
        """
        _parsed_schema = _get_alert_schema(key=key) # ??? - copied from fink-client scripts...
        if outdir is None:
            classification = topic
            date_str = Time.now().datetime.strftime("%Y%m%d")
            outdir = paths.alertDB_path / classification / date_str
            outdir.mkdir(exist_ok=True, parents=True)
        write_alert(alert, _parsed_schema, outdir, overwrite=True)
        return None

    def update_lightcurves(self, new_objectIds=None, t_ref: Time=None):
        t_ref = t_ref or Time.now()

        existing_objectIds = set([objectId for objectId in self.target_lookup.keys()])
        all_objectIds = set(new_objectIds).union(existing_objectIds)

        #for objectId, target in self.target_lookup.items():

        for objectId in all_objectIds:
            object_lightcurve_path = get_fink_lightcurve_path(objectId)
            dt_update = get_file_update_interval(object_lightcurve_path)
            lc_file_is_old = (dt_update > self.cutout_update_interval)

            target = self.target_lookup.get(objectId, None)

            if target is not None:
                if (target.fink_data.lightcurve is not None) and (not lc_file_is_old):
                    continue

            if target.fink_data.lightcurve is None:
                if object_lightcurve_path.exists():
                    fink_lightcurve = pd.read_csv(object_lightcurve_path)
                    if not fink_lightcurve.empty:
                        target.fink_data.lightcurve = fink_lightcurve

            if target.fink_data.lightcurve is None or (dt_update > self.cutout_update_interval):
                fink_lightcurve = FinkQuery.query_objects(objectId=objectId)


    def query_new_targets(self, t_ref: Time=None):
        t_ref = t_ref or Time.now()
               
        t0 = Time(t_ref.jd - self.query_lookback_time, format="jd")
        
        interval = 6 * u.hour
        n_intervals = int(self.query_lookback_time * 24 * u.hour / interval)

        query_df_list = []
        for class_ in self.query_classes:
            query_name = class_.replace(" ", "_")
            query_results_path = get_query_results_path(query_name)

            logger.info(f"queries for {class_}")
            query_results_age = get_file_update_interval(query_results_path, t_ref)
            if query_results_path.exists():
                if query_results_age < self.query_update_interval:
                    if query_results_path.stat().st_size > 1: # some small number...
                        query_df = pd.read_csv(query_results_path)
                        logger.info(f"read existing: (age={query_results_age:.2f}d)")
                        query_df_list.append(query_df)
                    else:
                        logger.info(f"{query_results_path} empty!")
                    continue

            df_list = []
            for ii in range(n_intervals):
                start_time = t0 + ii * interval
                start_str = start_time.datetime.strftime("%Y-%m-%d %H:%M:%S")
                end_time = t0 + (ii + 1) * interval
                end_str = end_time.datetime.strftime("%Y-%m-%d %H:%M:%S")

                if ii % 10 == 0:
                    logger.info(f"{ii+1}/{n_intervals}: {start_str}")
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
            query_df = pd.concat(df_list)
            query_df["query_name"] = query_name
            query_df.to_csv(query_results_path, index=False)
            query_df_list.append(query_df)

        result_df = pd.concat(query_df_list)
        self.last_query_update = t_ref

        return result_df

    def update_target_lightcurve(
        self, target: Target, updates: pd.DataFrame, keep_existing=True,
    ):
        update_target_data_lightcurve(
            target.fink_data, updates, keep_existing=keep_existing
        )

    def update_cutouts(self, objectId_list=None, t_ref: Time=None):
        logger.info("updating cutouts")
        t_ref = t_ref or Time.now()
        updated = []
        counter = 0
        objectId_list = objectId_list or list(self.target_lookup.keys())
        for objectId in objectId_list:
            target = self.target_lookup[objectId]

            target_cutout_path = get_fink_cutout_path(target.objectId)
            cutouts_are_None = any(
                [target.fink_data.cutouts.get(im, None) is None for im in FinkQuery.imtypes]
            )

            # If there are existing cutouts, and we're missing some... read them!
            if target_cutout_path.exists() and cutouts_are_None:
                existing_cutouts = read_existing_cutouts(target_cutout_path)
                for imtype in FinkQuery.imtypes:
                    if target.fink_data.cutouts.get(imtype, None) is None:
                        target.fink_data.cutouts[imtype] = existing_cutouts.get(imtype, None)

            if target.get_last_score(obs_name="no_observatory") is None:
                # We might have accidentally downloaded a target that's junk.
                logger.debug(f"{objectId} no score: skip cutouts")
                continue

            dt_update = get_file_update_interval(target_cutout_path, t_ref=t_ref)
            
            # Do we need to update them?
            if dt_update > self.cutout_update_interval:
                logger.info(f"{objectId} updating cutouts")
                for imtype in FinkQuery.imtypes:
                    im = FinkQuery.get_cutout(imtype, objectId=objectId)
                    target.fink_data.cutouts[imtype] = im
                write_cutouts(target.fink_data.cutouts, target_cutout_path)


    def perform_all_tasks(self, simulated_alerts=False, dump_alerts=True, t_ref: Time=None):
        logger.info("begin fink tasks")
        t_ref = t_ref or Time.now()
        
        
        if simulated_alerts:
            self.read_simulated_alerts()
        else:
            new_alerts = self.listen_for_alerts()
        self.process_alerts(new_alerts, simulated_alerts=simulated_alerts, dump_alerts=dump_alerts)
        if t_ref - self.last_query_update > self.query_update_interval:
            fink_objects = self.query_new_targets()
            new_objectIds = np.unique(fink_objects["objectId"])
        else:
            new_objectIds = []

        self.update_cutouts()
        pass
        