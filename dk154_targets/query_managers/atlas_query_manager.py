import io
import logging
import re
import requests
import yaml
from pathlib import Path

import pandas as pd

from astropy import units as u
from astropy.time import Time

from dk154_targets.queries import AtlasQuery, AtlasQueryError
from dk154_targets.target import Target

from .generic_query_manager import GenericQueryManager

from dk154_targets import paths

logger = logging.getLogger(__name__.split(".")[-1])

atlas_data_path = paths.data_path / "atlas"

def get_atlas_df_path(objectId):
    return paths.atlas_data_path / f"{objectId}.csv"

class AtlasQueryManager:

    atlas_base_url = "https://fallingstar-data.com/forcedphot"

    # these are the normal http response codes...
    QUERY_EXISTS = 200
    QUERY_SUBMITTED = 201
    QUERY_THROTTLED = 429

    def __init__(self, atlas_config: dict , target_lookup: dict):
        self.atlas_config = atlas_config
        token = self.atlas_config["token"]
        self.atlas_headers = dict(
            Authorization=f"Token {token}", Accept="application/json"
        )

        self.lookback_time = self.atlas_config.get("lookback_time", 30.)
        self.update_interval = self.atlas_config.get("update_interval", 3.)
        self.query_interval = self.atlas_config.get("query_interval", 10) * u.min
        self.max_submitted = self.atlas_config.get("max_submitted", 5)

        self.last_query_update = Time("2000-01-01 12:00:00") # never.

        self.target_lookup = target_lookup

        self.submitted_queries = {}
        self.throttled_objectIds = []


    def read_existing_atlas_data(self,):
        for objectId, target in self.target_lookup.items():
            atlas_df_path = get_atlas_df_path(target.objectId)
            atlas_data = getattr(target, "atlas_data", None)
            if atlas_data is not None:
                # don't overwrite what's already there!
                continue
            if atlas_data is None and atlas_df_path.exists():
                logger.info(f"{objectId} read existing csv")
                atlas_df = pd.read_csv(atlas_df_path)
                assert target.atlas_data is None
                target.atlas_data = atlas_df
                continue
            atlas_txt_path = atlas_df_path.with_suffix(".txt")
            if atlas_txt_path.exists():
                logger.info(f"{objectId} process existing txt")
                atlas_df = pd.read_csv(atlas_txt_path, delim_whitespace=True)
                col_rename = {c: c.replace("###", "") for c in atlas_df.columns if "###" in c}
                atlas_df.rename(col_rename, axis=1, inplace=True)
                assert not atlas_df_path.exists()
                atlas_df.to_csv(atlas_df_path, index=False)
                target.atlas_data = atlas_df


    def submit_query(self, target: Target, mjd_min=None, t_ref=None):
        t_ref = t_ref or Time.now()
        mjd_min = t_ref.mjd - self.lookback_time

        objectId = target.objectId

        if len(self.submitted_queries) > self.max_submitted:
            self.throttled_objectIds.append(objectId)
            return self.QUERY_THROTTLED # 429

        query_data = dict(
            ra = target.ra,
            dec=target.dec,
            mjd_min=mjd_min,
            mjd_max=t_ref.mjd - 1e-3, # about 1 minute before t_ref.
            send_email=False,
            comment=target.objectId
        )
        res = AtlasQuery.atlas_query(self.atlas_headers, query_data)
        if res.status_code == self.QUERY_SUBMITTED:
            task_url = res.json()['url']                
            self.submitted_queries[objectId] = task_url
            return res.status_code
        elif res.status_code == self.QUERY_THROTTLED:
            self.throttled_objectIds.append(objectId)
            return res.status_code
        else:
            logger.error(f"{objectId} query returned status {res.status_code} ?!")
            return res.status_code

    def retrieve_finished_queries(self,):
        finished, ongoing = [], []
        with requests.Session() as s:
            for objectId, task_url in self.submitted_queries.items():
                res = s.get(task_url, headers=self.atlas_headers)
                if res.status_code == self.QUERY_EXISTS:
                    finishtimestamp = res.json().get("finishtimestamp", None)
                    if finishtimestamp is None:
                        logger.info(f"{objectId} query not finished")
                        ongoing.append(objectId)
                        continue
                    result_url = res.json().get("result_url", None)
                    logger.info(f"{objectId} finished")
                    if result_url is None:
                        raise AtlasQueryError(
                            f"there should be a result_url if finished at {finishtimestamp}"
                        )
                    photom_res = s.get(result_url, headers=self.atlas_headers)
                    df = AtlasQuery.process_df(photom_res) # convert the result into a pandas...
                    finished.append(objectId)
                    if len(df) == 0:
                        logger.info(f"{objectId} has no new data!")
                        continue
                    atlas_df_path = get_atlas_df_path(objectId)
                    if atlas_df_path.exists():
                        old_df = pd.read_csv(atlas_df)
                        assert old_df["MJD"].max() < df["MJD"].min()
                        atlas_df = pd.concat([old_df, df])
                        atlas_df.reset_index(drop=True, inplace=True)
                    else:
                        atlas_df = df
                    atlas_df.to_csv(atlas_df_path, index=False)

                    target = self.target_lookup[objectId]

                    target.atlas_data = atlas_df
                    target.last_atlas_query = finishtimestamp
                    target.updated = True
                else:
                    logger.error()
            for objectId in finished:
                finished_task_url = self.submitted_queries.pop(objectId)
                s.delete(finished_task_url, headers=self.atlas_headers)

        logger.info(f"{len(finished)} finished, {len(ongoing)} ongoing")
        return finished, ongoing

    def retry_throttled_queries(self,):
        old_throttled_list = self.throttled_objectIds
        self.throttled_objectIds = [] # reset! then re-add things if needed.
        N_success = 0
        N_throttled = 0
        for objectId in old_throttled_list:
            target = self.target_lookup[objectId]
            status_code = self.submit_query(target)
            if status_code == self.QUERY_SUBMITTED:
                N_success = N_success + 1
                assert objectId in self.submitted_queries
            elif status_code == self.QUERY_THROTTLED:
                N_throttled = N_throttled + 1
                assert objectId in self.throttled_objectIds
            else:
                pass

        assert N_success + N_throttled == len(old_throttled_list)

        return None

    def submit_new_queries(self, t_ref=None):
        t_ref = t_ref or Time.now()

        for objectId, target in self.target_lookup.items():
            if objectId in self.submitted_queries:
                # We're waiting for the results.
                continue

            if objectId in self.throttled_objectIds:
                # It's already waiting to be submitted
                continue

            atlas_data = getattr(target, "atlas_data", None)
            if atlas_data is not None:
                last_atlas_mjd = target.atlas_data["MJD"].max()
                #last_update = target.last_atlas_query
                obj_update_interval = t_ref.mjd - last_atlas_mjd
                if obj_update_interval < self.update_interval:
                    logger.info(f"{objectId}: phot {obj_update_interval:.1f}d old")
                    # The last photometry point was quite recently
                    continue
                else:
                    logger.info(f"{objectId} phot interval {obj_update_interval:.1f}d")

            atlas_df_path = get_atlas_df_path(objectId)
            if atlas_data is not None and atlas_df_path.exists():
                try:
                    last_df_update = Time(atlas_df_path.stat().st_mtime, format="unix")
                    obj_update_interval = t_ref.jd - last_df_update.jd
                    if obj_update_interval < self.update_interval:
                        logger.info(f"{objectId}: file {obj_update_interval:.1f}d old")
                        continue
                    else:
                        logger.info(f"{objectId} file interval {obj_update_interval:.1f}d")

                except Exception as e:
                    logger.error(f"accessing {objectId} file mod time")
            if target.get_last_score(obs_name="no_observatory") is None:
                logger.info(f"{objectId} not yet scored")
                continue

            #if len(self.submitted_queries) >= self.max_submitted:
            #    self.throttled_objectIds.append(objectId)
            #    logger.info(f"{objectId} waiting ({len(self.submitted_queries)} submitted already)")
            status_code = self.submit_query(target)
            if status_code == self.QUERY_SUBMITTED:
                logger.info(f"{objectId} query submitted")



    def perform_all_tasks(self, t_ref=None):
        t_ref = Time.now()
        dt_query = t_ref - self.last_query_update
        if dt_query < self.query_interval:
            dt_query_mins = dt_query.to(u.min).value
            query_interval_mins = self.query_interval.to(u.min).value
            logger.info(f"dt_query={dt_query_mins:.2f} min < req. {query_interval_mins:.2f} min")
            return

        self.read_existing_atlas_data()
        finished, ongoing = self.retrieve_finished_queries()
        self.retry_throttled_queries()
        self.submit_new_queries()
        logger.info(f"{len(self.submitted_queries)} submitted, {len(self.throttled_objectIds)} throttled")
        self.last_query_update = t_ref
        return 


            
