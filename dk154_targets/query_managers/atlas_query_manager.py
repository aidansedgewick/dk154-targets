import io
import logging
import re
import requests
import traceback
import yaml
from pathlib import Path

import pandas as pd

from astropy import units as u
from astropy.time import Time

from dk154_targets.queries import AtlasQuery, AtlasQueryError
from dk154_targets.target import Target

from .generic_query_manager import GenericQueryManager
from .query_manager_utils import get_file_update_interval

from dk154_targets import paths

logger = logging.getLogger(__name__.split(".")[-1])

atlas_data_path = paths.data_path / "atlas"

def get_atlas_df_path(objectId):
    return paths.atlas_data_path / f"{objectId}.csv"

def write_empty_atlas_df(df_path):
    cols = "MJD,m,dm,uJy,duJy,F,err,chi/N,RA,Dec,x,y,maj,min,phi,apfit,mag5sig,Sky,Obs".split(",")
    df = pd.DataFrame([], columns=cols)
    df.to_csv(df_path, index=False)


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


        recovered_queries = self.recover_existing_queries() # gets only the first page...
        while len(recovered_queries) > 0:
            self.retrieve_finished_queries(task_url_lookup=recovered_queries)
            old_recovered_queries = recovered_queries
            recovered_queries = self.recover_existing_queries() # get the next page.
            if set(old_recovered_queries) == set(recovered_queries):
                break

    def recover_existing_queries(self):
        logger.info("recover existing queries...")
        results_response = AtlasQuery.get_existing_queries(headers=self.atlas_headers)
        results_data = results_response.json()

        task_results = results_data.get("results", None)
        if task_results is None:
            logger.error("no 'results' in response.json()")
            return {}
        taskcount = results_data["taskcount"]
        logger.info(f"recovered {taskcount} existing queries")

        submitted_queries = {}
        for result in task_results[::-1]:
            objectId = result.get("comment", None)
            if objectId is None:
                logger.warning(f"existing query has no comment=objectId")
                continue
            task_url = result.get("url", None)
            if task_url is None:
                logger.warning(f"existing query {objectId} has NO URL")
                continue
            submitted_queries[objectId] = task_url
        return submitted_queries
            

    def read_existing_atlas_lightcurves(self,):

        sorted_objectIds = sorted(self.target_lookup.keys())
        #for objectId, target in self.target_lookup.items():
        for objectId in sorted_objectIds:
            target = self.target_lookup.get(objectId)
            atlas_df_path = get_atlas_df_path(target.objectId)
            
            if not atlas_df_path.exists():
                logger.info(f"no existing {atlas_df_path}")
                continue
            if target.atlas_data.lightcurve is not None:
                # don't overwrite what's already there!
                logger.info(f"{objectId} don't overwrite!")
                continue
            if target.atlas_data.lightcurve is None and atlas_df_path.exists():
                logger.info(f"{objectId} read existing csv")
                atlas_df = pd.read_csv(atlas_df_path)
                if not atlas_df.empty:
                    assert target.atlas_data.lightcurve is None
                    target.atlas_data.lightcurve = atlas_df
                continue
            atlas_txt_path = atlas_df_path.with_suffix(".txt")
            if atlas_txt_path.exists():
                logger.info(f"{objectId} process existing txt")
                atlas_df = pd.read_csv(atlas_txt_path, delim_whitespace=True)
                col_rename = {c: c.replace("###", "") for c in atlas_df.columns if "###" in c}
                atlas_df.rename(col_rename, axis=1, inplace=True)
                assert not atlas_df_path.exists()
                atlas_df.to_csv(atlas_df_path, index=False)
                target.atlas_data.lightcurve = atlas_df


    def submit_query(self, target: Target, mjd_min=None, t_ref=None):
        t_ref = t_ref or Time.now()
        if target.atlas_data.lightcurve is not None:
            mjd_min = target.atlas_data.lightcurve["MJD"].max() + 1e-3
            mjd_min_str = Time(mjd_min, format="mjd").datetime.strftime("%Y-%m-%d %X")
            logger.info(f"{target.objectId} mjd_min={mjd_min:.4f}")
            logger.info(f"    ={mjd_min_str}")
        else:
            mjd_min = t_ref.mjd - self.lookback_time

        objectId = target.objectId

        if len(self.submitted_queries) >= self.max_submitted:
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
            logger.error(f"{objectId} query returned status \033[33;1m{res.status_code}\033[0m?!")
            return res.status_code


    def retrieve_finished_queries(self, task_url_lookup: dict=None):
        """
            Retrieve query data from finished.
        """
        
        finished, ongoing = [], []

        if task_url_lookup is None:
            task_url_lookup = self.submitted_queries
        
        with requests.Session() as s:
            for objectId, task_url in task_url_lookup.items():
                res = s.get(task_url, headers=self.atlas_headers)
                if res.status_code == self.QUERY_EXISTS:
                    finishtimestamp = res.json().get("finishtimestamp", None)
                    if finishtimestamp is None:
                        logger.info(f"{objectId} query not finished")
                        ongoing.append(objectId)
                        continue
                    result_data = res.json()
                    result_url = result_data.get("result_url", None)
                    logger.info(f"{objectId} finished")
                    atlas_df_path = get_atlas_df_path(objectId)
                    if result_url is None:
                        error_msg = result_data.get("error_msg", None)
                        if (error_msg is not None) and (error_msg == "No data returned"):
                            logger.info(f"{objectId} returned NO DATA")
                            if not atlas_df_path.exists():
                                write_empty_atlas_df(atlas_df_path)
                            else:
                                # rewrite existing atlas data to update timestamp, to avoid requery.
                                atlas_df = pd.read_csv(atlas_df_path)
                                atlas_df.to_csv(atlas_df_path, index=False)
                            finished.append(objectId)
                            continue
                        logger.error(f"query finished, but error_msg=={error_msg}")

                    photom_res = s.get(result_url, headers=self.atlas_headers)
                    df = AtlasQuery.process_df(photom_res) # convert the result into a pandas...
                    if len(df) == 0:
                        logger.info(f"{objectId} has no new data!")
                        continue
                    if atlas_df_path.exists():
                        old_df = pd.read_csv(atlas_df_path)
                        if not old_df.empty:
                            assert old_df["MJD"].max() < df["MJD"].min()
                            atlas_df = pd.concat([old_df, df])
                            atlas_df.reset_index(drop=True, inplace=True)
                        else:
                            atlas_df = df
                    else:
                        atlas_df = df
                    atlas_df.to_csv(atlas_df_path, index=False)
                    finished.append(objectId) # wait until data saved to delete!

                    target = self.target_lookup.get(objectId, None)
                    if target is None:
                        logger.info("")
                    else:
                        target.atlas_data.lightcurve = atlas_df
                        #target.last_atlas_query = finishtimestamp
                        target.updated = True
                else:
                    logger.error()
            for objectId in finished:
                finished_task_url = task_url_lookup.pop(objectId)
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

        N_skipped = 0
        N_submitted = 0

        # which targets to prioritise?

        score_lookup = {}
        for objectId, target in self.target_lookup.items():
            last_score = target.get_last_score(obs_name="no_observatory")
            if last_score is None:
                last_score = 0.
            score_lookup[objectId] = last_score
        sorted_targets = sorted(score_lookup, key=lambda k: score_lookup[k], reverse=True)
        if len(sorted_targets) != len(self.target_lookup):
            raise ValueError(f"sorted len != lookup len: {len(sorted_targets)}!={len(self.target_lookup)}")

        #for objectId, target in self.target_lookup.items():
        for objectId in sorted_targets:
            target = self.target_lookup[objectId]
            if objectId in self.submitted_queries:
                # We're waiting for the results.
                continue

            if objectId in self.throttled_objectIds:
                # It's already waiting to be submitted
                continue

            submit_reasons = []

            #atlas_data = getattr(target, "atlas_data", None)
            if target.atlas_data.lightcurve is not None:
                last_atlas_mjd = target.atlas_data.lightcurve["MJD"].max()
                #last_update = target.last_atlas_query
                obj_update_interval = t_ref.mjd - last_atlas_mjd
                if obj_update_interval < self.update_interval:
                    logger.debug(f"{objectId}: phot {obj_update_interval:.1f}d old")
                    # The last photometry point was quite recently
                    continue
                else:
                    logger.debug(f"{objectId} phot interval {obj_update_interval:.1f}d")
                    submit_reasons.append(f"phot {obj_update_interval:.1f}d old")
            else:
                submit_reasons.append("no photom")
            

            atlas_df_path = get_atlas_df_path(objectId)
            dt_update = get_file_update_interval(atlas_df_path, t_ref)
            if atlas_df_path.exists():
                if dt_update > self.update_interval:
                    # need to update the photometry
                    submit_reasons.append(f"csv {dt_update:.1f}d old")
                else:
                    continue
            else:
                submit_reasons.append("no csv exists")

            if target.get_last_score(obs_name="no_observatory") is None:
                N_skipped = N_skipped + 1
                logger.debug(f"{objectId} not yet scored")
                continue

            #if len(self.submitted_queries) >= self.max_submitted:
            #    self.throttled_objectIds.append(objectId)
            #    logger.info(f"{objectId} waiting ({len(self.submitted_queries)} submitted already)")
            logger.info(f"{target.objectId} " + ",".join(submit_reasons))
            status_code = self.submit_query(target)
            N_submitted = N_submitted + 1
            if status_code == self.QUERY_SUBMITTED:
                logger.info(f"{objectId} query submitted")
            elif status_code == self.QUERY_THROTTLED:
                logger.info(f"{objectId} query throttled BY ATLAS!")
            else:
                logger.info("query_status?!")
        logger.info(f"{N_skipped} skipped")
        logger.info(f"{N_submitted} new queries submitted")


    def perform_all_tasks(self, t_ref: Time=None):
        logger.info("begin atlas tasks")
        t_ref = Time.now()
        dt_query = t_ref - self.last_query_update
        if dt_query < self.query_interval:
            dt_query_mins = dt_query.to(u.min).value
            query_interval_mins = self.query_interval.to(u.min).value
            logger.info(f"dt_query={dt_query_mins:.2f} min < req. {query_interval_mins:.2f} min")
            return

        self.read_existing_atlas_lightcurves()
        try:
            finished, ongoing = self.retrieve_finished_queries()
            logger.info("during retrieve_finished ...")
        except Exception as e:
            logger.error(traceback.format_exc())

        try:
            self.retry_throttled_queries()
        except Exception as e:
            logger.info("during retry_throttled ...")
            logger.error(traceback.format_exc())

        try:
            self.submit_new_queries()
        except Exception as e:
            logger.info("during submit_new ...")
            logger.error(traceback.format_exc())
        logger.info(f"total: {len(self.submitted_queries)} submitted, {len(self.throttled_objectIds)} throttled")
        self.last_query_update = t_ref
        return 


            
