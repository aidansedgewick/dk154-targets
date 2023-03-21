import logging
from pathlib import Path

import numpy as np

import pandas as pd

from astropy.time import Time, TimeDelta

from alerce.core import Alerce

from dk154_targets.target import Target

from .generic_query_manager import GenericQueryManager
from .query_manager_utils import get_file_update_interval

from dk154_targets import paths

logger = logging.getLogger(__name__.split(".")[-1])

def get_query_results_path(query_name: str, t_ref: Time=None):
    #t_ref = t_ref or Time.now()
    date_str = t_ref.strftime("%Y%m%d_%H%M%S")
    #df_stem = f"{query_name}" #_{date_str}"
    df_stem = f"{query_name}"
    query_path = AlerceQueryManager.alerce_query_results_path / f"{df_stem}.csv"
    return query_path

def get_alerce_lightcurve_path(objectId):
    return AlerceQueryManager.alerce_lightcurve_path / f"{objectId}.csv"

def get_alerce_magstats_path(objectId):
    return AlerceQueryManager.alerce_magstats_path / f"{objectId}.csv"

def get_alerce_probabilities_path(objectId):
    return AlerceQueryManager.alerce_probabilities_path / f"{objectId}.csv"

def process_alerce_lightcurve(detections, non_detections):
    if non_detections is not None:
        detections["tag"] = "valid"
        is_dubious = detections["dubious"]
        detections.loc[is_dubious,"tag"] = "badquality"
        non_detections.loc[:,"tag"] = "upperlim"
        alerce_lightcurve = pd.concat([detections, non_detections])
    else:
        alerce_lightcurve = detections

    alerce_lightcurve.insert(
        2, "jd", Time(alerce_lightcurve["mjd"], format="mjd").jd
    )
    alerce_lightcurve.sort_values("jd", inplace=True)
    return alerce_lightcurve
    

class AlerceQueryManager(GenericQueryManager):

    name = "alerce"
    alerce_data_path = paths.alerce_data_path
    alerce_query_results_path = alerce_data_path / "query_results"
    alerce_lightcurve_path = alerce_data_path / "lightcurves"
    alerce_magstats_path = alerce_data_path / "magstats"
    alerce_probabilities_path = alerce_data_path / "probabilities"

    def __init__(self, alerce_config: dict, target_lookup: dict):
        self.alerce_config = alerce_config
        self.object_queries = self.alerce_config.get("object_queries", {})
        self.alerce_broker = Alerce()
        self.last_query_update = Time(0., format="unix")

        self.last_obs_lookback_time = self.alerce_config.get("last_obs_lookback_time", 30.)
        self.first_obs_lookback_time = self.alerce_config.get("first_obs_lookback_time", 70.)
        self.n_objects = int(self.alerce_config.get("n_objects", 25))
        self.magmin = self.alerce_config.get("magmin", 19.0)
        self.data_update_interval = self.alerce_config.get("data_update_interval", 2.)
        self.query_update_interval = self.alerce_config.get("query_update_interval", 0.25)

        # make sure paths are created
        self.alerce_data_path.mkdir(exist_ok=True, parents=True)
        self.alerce_query_results_path.mkdir(exist_ok=True, parents=True)
        self.alerce_lightcurve_path.mkdir(exist_ok=True, parents=True)
        self.alerce_magstats_path.mkdir(exist_ok=True, parents=True)
        self.alerce_probabilities_path.mkdir(exist_ok=True, parents=True)


        # so that we don't bother re-reading things we reject again and again.
        # reset every time we make a new object query.
        self.already_processed_targets = [] 

        ## the target_lookup!
        self.target_lookup = target_lookup


    def query_new_targets(self, t_ref: Time=None):
        t_ref = t_ref or Time.now()
        logger.info(f"use last_obs_lookback_time {self.last_obs_lookback_time:.1f}d")
        logger.info(f"use first_obs_lookback_time {self.first_obs_lookback_time:.1f}d")
        lastmjd = (t_ref.mjd - self.last_obs_lookback_time, t_ref.mjd)
        firstmjd = (t_ref.mjd - self.first_obs_lookback_time, t_ref.mjd)

        query_df_list = []

        for query_name, query_pattern in self.object_queries.items():
            query_results_path = get_query_results_path(query_name, t_ref=t_ref)
            query_results_update_interval = get_file_update_interval(query_results_path, t_ref)
            query_results_are_old = (query_results_update_interval > self.query_update_interval)

            logger.info(f"try {query_name} query ")
            if query_results_path.exists():
                if query_results_are_old:
                    logger.info("objects results are old - re-query!")
                else:
                    logger.info(f"read existing: (age={query_results_update_interval:.2f})")
                    if query_results_path.stat().st_size > 1: # some small number...
                        query_df = pd.read_csv(query_results_path)
                        query_df_list.append(query_df)
                    else:
                        logger.info(f"{query_results_path} empty!")
                    continue
            df_list = []
            page = 1
            logger.info(f"{self.n_objects} per query")
            while True:
                query_data = dict(
                    **query_pattern, lastmjd=lastmjd, firstmjd=firstmjd, 
                    page=page, page_size=self.alerce_config.get("n_objects", 25)
                )
                try:
                    new_targets_df = self.alerce_broker.query_objects(**query_data)
                    df_list.append(new_targets_df)
                    logger.info(f"finished query {page}")
                except Exception as e:
                    logger.warning("query error...")
                    print(e)
                    logger.info(f"query {page} not successful.")
                if len(new_targets_df) < self.n_objects:
                    logger.info(f"break after {page} queries")
                    break
                page = page + 1
            query_df = pd.concat(df_list)
            logger.info(f"query returned {len(query_df)}")
            #if len(query_df) > 0:
            query_df["query_name"] = query_name
            query_df.to_csv(query_results_path, index=False)
            query_df_list.append(query_df)
        result_df = pd.concat(query_df_list)
        result_df.reset_index(drop=True, inplace=True)
        self.last_query_update = t_ref

        return result_df


    def perform_lightcurve_query(self, objectId):
        alerce_detections = self.alerce_broker.query_detections(objectId, format="pandas")
        alerce_non_detections = self.alerce_broker.query_non_detections(objectId, format="pandas")
        if alerce_non_detections.empty:
            alerce_non_detections = None # skips the append in process_alerce_lc.
        alerce_lightcurve = process_alerce_lightcurve(
            alerce_detections, alerce_non_detections
        )
        return alerce_lightcurve

        
    def update_lightcurves(self, new_objectIds=None, t_ref: Time=None):
        t_ref = t_ref or Time.now()
        if new_objectIds is None:
            new_objectIds = []

        logger.info(f"reject obj fainter than {self.magmin:.2f}")
        n_magstats_queries = 0
        n_existing_updated = 0
        n_new_added = 0
        n_lc_queries = 0

        existing_objectIds = set([objectId for objectId in self.target_lookup.keys()])
        all_objectIds = set(new_objectIds).union(existing_objectIds)

        for objectId in all_objectIds:

            object_lightcurve_path = get_alerce_lightcurve_path(objectId)
            lc_file_update_interval = get_file_update_interval(object_lightcurve_path, t_ref)
            lc_file_is_old = (lc_file_update_interval > self.data_update_interval)
            logger.debug(f"process {objectId}")

            target = self.target_lookup.get(objectId, None)

            if target is not None:
                if (target.alerce_data.lightcurve is not None) and (not lc_file_is_old):
                    continue # No need to update!

            if object_lightcurve_path.exists() and not lc_file_is_old:
                logger.debug("read existsing lightcurve")
                alerce_lightcurve = pd.read_csv(object_lightcurve_path)
                alerce_magmin = alerce_lightcurve["magpsf"].min()
                if alerce_magmin > self.magmin:
                    continue # too faint.
            else:
                # Do we have magstats? (small summary?)
                obj_magstats_path = get_alerce_magstats_path(objectId)
                magstats_update_interval = get_file_update_interval(obj_magstats_path, t_ref)
                magstats_file_is_old = magstats_update_interval > self.data_update_interval
                if obj_magstats_path.exists() and (not magstats_file_is_old):
                    logger.info(f"{objectId} read ex. magstats")
                    magstats = pd.read_csv(obj_magstats_path)
                else:
                    n_magstats_queries = n_magstats_queries + 1
                    magstats = self.alerce_broker.query_magstats(objectId, format="pandas")
                    if not magstats.empty:
                        magstats.to_csv(obj_magstats_path, index=False)
                        logger.info(f"{objectId} save magstats")
                    else:
                        continue
                    
                alerce_magmin = magstats["magmin"].min() # one row for each of g,r
                if alerce_magmin is None or not np.isfinite(alerce_magmin):
                    continue
                logger.info(f"{objectId} lc query")
                alerce_lightcurve = self.perform_lightcurve_query(objectId)
                # function merges detections and non-detections nicely.
                alerce_lightcurve.to_csv(object_lightcurve_path, index=False)
                n_lc_queries = n_lc_queries + 1

            if target is None:
                target = Target.from_alerce_lightcurve(
                    objectId, alerce_lightcurve=alerce_lightcurve
                )
                if target is not None:
                    self.target_lookup[objectId] = target
                    n_new_added = n_new_added + 1
            else:
                target.alerce_data.lightcurve = alerce_lightcurve
                n_existing_updated = n_existing_updated + 1
        logger.info(f"{n_magstats_queries} magstats queries")
        logger.info(f"{n_new_added} new targets added")
        logger.info(f"{n_existing_updated} existing updated")

    def update_target_probabilities(self, t_ref: Time=None):
        t_ref = t_ref or Time.now()
        for objectId, target in self.target_lookup.items():
            obj_prob_data_path = get_alerce_probabilities_path(objectId)
            obj_prob_update_interval = get_file_update_interval(obj_prob_data_path, t_ref)
            prob_data_is_old = (obj_prob_update_interval > self.data_update_interval)
            #updated_probabilities = False
            if (target.alerce_data.probabilities is None) or prob_data_is_old:
                if obj_prob_data_path.exists() and (not prob_data_is_old):
                    probabilities = pd.read_csv(obj_prob_data_path)
                    target.alerce_data.probabilities = probabilities
                else:
                    probabilities = self.alerce_broker.query_probabilities(
                        objectId, format="pandas"
                    )
                    probabilities.to_csv(obj_prob_data_path, index=False)
                    target.alerce_data.probabilities = probabilities
                target.updated = True
                target.alerce_data.probabilities.set_index(
                    ["classifier_name", "class_name"], inplace=True
                )

                #updated_probabilities = True
            #print(objectId, updated_probabilities)
            #print(probabilities)
            #if updated_probabilities:

    def perform_all_tasks(self, t_ref: Time=None):
        t_ref = t_ref or Time.now()

        if t_ref - self.last_query_update > self.query_update_interval:
            alerce_objects = self.query_new_targets()
            new_objectIds = np.unique(alerce_objects["oid"])
        else:
            new_objectIds = []
        self.update_lightcurves(new_objectIds, t_ref=t_ref)
        self.update_target_probabilities(t_ref=t_ref)
        return