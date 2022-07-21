import logging
import requests
import time

import numpy as np
import pandas as pd

from astropy.time import Time

from fink_client.avroUtils import write_alert, _get_alert_schema
from fink_client.consumer import AlertConsumer

from dk154_targets import paths
from dk154_targets.target import Target
from dk154_targets.queries import FinkQuery

from dk154_targets.utils import readstamp

logger = logging.getLogger(__name__.split(".")[-1])

class FinkQueryManager:
    
    name = "fink"
    default_num_alerts = 5
    default_timeout = 10

    def __init__(self, consumer_config, target_lookup):
        self.consumer_config = consumer_config
        self.credential_config = {
            x: self.consumer_config.get(x) for x in ["username", "group_id", "server"]
        }
        print("credential is", self.credential_config)
        if any([x is None for x in self.credential_config.values()]):
            msg = (
                "your selector_config should contain fink:\n"
                "query_managers:\n  fink:\n    "
                "username: <username>\n    group_id: <group-id>\n    servers: <server>\n"
            )
            raise ValueError(msg)

        topics = self.consumer_config.get("topics", None)
        self.topics = topics or ["fink_sso_ztf_candidates_ztf"]

        self.target_lookup = target_lookup


    def listen_for_alerts(self):
        num_alerts = self.consumer_config.get("num_alerts", self.default_num_alerts)
        timeout = self.consumer_config.get("timeout", self.default_timeout)
        latest_alerts = []
        logger.info(f"listen for {round(timeout)} sec, for {num_alerts} alerts")
        topic_counter = {topic: 0 for topic in self.topics}
        for current_topic in self.topics:
            with AlertConsumer([current_topic], self.credential_config) as consumer:
                for ii in range(num_alerts):
                    topic, alert, key = consumer.poll(timeout=timeout)
                    if any([x is None for x in [topic, alert, key]]):
                        logger.info(f"break after {len(latest_alerts)}")
                        break
                    topic_counter[topic] = topic_counter[topic] + 1
                    latest_alerts.append( (topic, alert, key,) )
        summary_str = ", ".join(
            f"{v} {k}"for k, v in topic_counter.items() if v > 0
        ) or "no alerts"
        logger.info(f"recieve {summary_str}")
        return latest_alerts
        
             
    def process_alerts(self, latest_alerts, dump_alerts=True):
        """
        TODO add docs
        """
        logger.info(f"process {len(latest_alerts)} new alerts!")
        new_targets = []
        updated_targets = []
        for topic, alert, key in latest_alerts:
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
                continue
            if all([x is None for x in alert_history["magpsf"]]):
                prv_candidates = pd.DataFrame(alert["prv_candidates"])
                logger.info("launch query")
                alert_history = FinkQuery.query_objects(
                    objectId=alert["objectId"], return_df=True, fix_column_names=True
                )
            alert_history.sort_values("jd", inplace=True)

            objectId = new_alert["objectId"]

            if objectId not in self.target_lookup:
                logger.info(f"initialise target {objectId}")
                target_history = pd.concat(
                    [alert_history, pd.DataFrame(new_alert, index=[len(alert_history)])], 
                    ignore_index=True
                )

                target = Target(
                    objectId, new_alert["ra"], new_alert["dec"], target_history=target_history
                )
                self.target_lookup[objectId] = target
                new_targets.append(objectId)
            else:
                assert objectId in self.target_lookup
                logger.info(f"update target {objectId}")
                target = self.target_lookup[objectId]
                target.update_target(pd.DataFrame(new_alert))
                updated_targets.append(objectId)

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


    def update_cutouts(self,):
        updated = []
        for objectId, target in self.target_lookup.items():
            cutouts_are_None = any([im is None for im in target.cutouts.values()])
            no_cutouts = len(target.cutouts) == 0
            if no_cutouts or cutouts_are_None:
                target.update_cutouts()
                updated.append(objectId)
        if len(updated) > 0:
            logger.info(f"updated {len(cutouts)} fink cutouts")


    def perform_all_tasks(self):
        new_alerts = self.listen_for_alerts()
        self.process_alerts(new_alerts)
        self.update_cutouts()
        