import json
import logging
import os
import requests
import time
from pathlib import Path

import numpy as np
import pandas as pd

from astropy.time import Time

from fink_client.avroUtils import write_alert, _get_alert_schema, AlertReader
from fink_client.consumer import AlertConsumer

from dk154_targets import paths
from dk154_targets.target import Target
from dk154_targets.queries import FinkQuery

from .generic_query_manager import GenericQueryManager

from dk154_targets.utils import readstamp

logger = logging.getLogger(__name__.split(".")[-1])

class FinkQueryManager(GenericQueryManager):
    
    name = "fink"
    default_num_alerts = 5
    default_timeout = 10

    def __init__(self, consumer_config, target_lookup):
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

        self.target_lookup = target_lookup


    def listen_for_alerts(self, fake_alerts: str=False):
        if not fake_alerts:
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
                        topic, alert, key = consumer.poll(timeout=timeout)
                        if any([x is None for x in [topic, alert, key]]):
                            logger.info(f"break after {len(latest_alerts)} alerts")
                            break
                        topic_counter[topic] = topic_counter[topic] + 1
                        latest_alerts.append( (topic, alert, key,) )
            summary_str = ", ".join(
                f"{v} {k}"for k, v in topic_counter.items() if v > 0
            ) or "no alerts"
            logger.info(f"recieve {summary_str}")
        else:
            fake_alerts_dir = Path(fake_alerts)
            fake_alerts = [path for path in fake_alerts_dir.glob("*.json")]
            latest_alerts = []
            for fake_alert_path in sorted(fake_alerts):
                with open(fake_alert_path, "r") as f:
                    alert = json.load(f)
                latest_alerts.append( (None, alert, None) ) # topic, alert, key.
            for alert_path in fake_alerts:
                os.remove(alert_path)
        return latest_alerts
        
             
    def process_alerts(
        self, latest_alerts, dump_alerts=True, fake_alerts=False, t_ref=None, delta_t=None
    ):
        """
        TODO add docs
        """
        t_ref = t_ref or Time.now()

        logger.info(f"process {len(latest_alerts)} new alerts!")
        new_targets = []
        updated_targets = []
        for topic, alert, key in latest_alerts:
            if not fake_alerts:
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

            if objectId not in self.target_lookup:
                try:
                    alert_history = FinkQuery.query_objects(
                        objectId=alert["objectId"], 
                        withupperlim=True,
                        return_df=True, 
                        fix_column_names=True
                    )
                    logger.info(f"{objectId} query lc ")
                except Exception as e:
                    logger.warning(f"{objectId} fink query failed")
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

                logger.info(f"{objectId} initialise target")
                """
                if alert_history is not None:
                    new_alert_df = pd.DataFrame(new_alert, index=[len(alert_history)])
                    target_history = pd.concat(
                        [alert_history, new_alert_df], 
                        ignore_index=True
                    )
                else:
                    logger.info("set target history only as alert")
                    target_history = pd.DataFrame(new_alert, index=0)
                """

                target = Target(
                    objectId, new_alert["ra"], new_alert["dec"], target_history=alert_history
                )
                self.target_lookup[objectId] = target
                new_targets.append(objectId)
            else:
                updated_targets.append(objectId) # keep track of who we've updated.
            assert objectId in self.target_lookup # we've just added it - it should be there...!
            logger.info(f"{objectId} update target")
            target = self.target_lookup[objectId]
            if "tag" in target.target_history:
                new_alert["tag"] = "valid"

            new_alert_df = pd.DataFrame(new_alert, index=[len(target.target_history)])
            alert_time = Time(new_alert['jd'], format="jd")
            alert_time_str = alert_time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"alert jd = {alert_time.jd:.5f}")
            logger.info(f"         = {alert_time_str}")
            target.update_target_history(new_alert_df, keep_old=True)
            print()
            for imtype in FinkQuery.imtypes:
                target.cutouts[imtype] = readstamp(
                    alert.get('cutout'+imtype, {}).get('stampData', None)
                )
        logger.info(f"manager added {len(new_targets)}, updated {len(updated_targets)}")
        return None


    def dump_alert(self, topic, alert, key, outdir=None):
        """
        method directly from the fink-
        
        """
        _parsed_schema = _get_alert_schema(key=key) # ??? - copied from fink-client scripts...
        if outdir is None:
            classification = topic
            date_str = Time.now().datetime.strftime("%Y%m%d")
            outdir = paths.alertDB_path / classification / date_str
            outdir.mkdir(exist_ok=True, parents=True)
        write_alert(alert, _parsed_schema, outdir, overwrite=True)
        return None


    def update_cutouts(self, objectId_list=None):

        updated = []
        counter = 0
        objectId_list = objectId_list or list(self.target_lookup.keys())
        logger.info("updating cutouts...")
        for objectId in objectId_list:
            target = self.target_lookup[objectId]
            cutouts_are_None = any([im is None for im in target.cutouts.values()])
            no_cutouts = len(target.cutouts) == 0
            if no_cutouts or cutouts_are_None:
                #target.update_cutouts()
                for imtype in FinkQuery.imtypes:
                    im = FinkQuery.get_cutout(imtype, objectId=objectId)
                    target.cutouts[imtype] = im
                updated.append(objectId)
            if len(updated) > (counter + 1) * 10:
                counter = counter + 1
                logger.info(f"updated {len(updated)} fink cutouts...")
        if len(updated) > 0:
            logger.info(f"updated {len(updated)} fink cutouts")


    def perform_all_tasks(self, fake_alerts=False, dump_alerts=True):
        new_alerts = self.listen_for_alerts(fake_alerts=fake_alerts)
        self.process_alerts(new_alerts, dump_alerts=dump_alerts, fake_alerts=fake_alerts)
        #self.update_cutouts()
        