import io
import logging
import re
import requests
import yaml
from pathlib import Path

import pandas as pd

from astropy.time import Time

from dk154_targets.queries import AtlasQuery, AtlasQueryError

from dk154_targets import paths # TODO fix this as relative import?

logger = logging.getLogger(__name__.split(".")[-1])

atlas_data_path = paths.data_path / "atlas"

class AtlasQueryManager:

    name = "atlas"
    atlas_base_url = "https://fallingstar-data.com/forcedphot"

    def __init__(self, atlas_config, target_lookup):
        self.atlas_config = atlas_config
        self.credential_config = self.atlas_config.get("credential")
        if self.credential_config is None:
            raise ValueError("config atlas must contain `credential` with `token`, `username`, `password`")
        self.headers = dict(
            Authorization=f"Token {self.credential_config['token']}", Accept="application/json"
        )
        self.submitted_queries = {}
        self.throttled_queries = []
        self.target_lookup = target_lookup





    def queue_forced_photometry(self, objectId_list=None):
        new_submissions = []
        objectId_list = objectId_list or list(self.target_lookup.keys())
        for objectId in objectId_list:
            target = self.target_lookup[objectId]
            if (objectId in self.submitted_queries) or (objectId in self.throttled_queries):
                continue
            atlas_data = target.data.get("atlas", None)
            if atlas_data is None or (not atlas_data.candidate):
                continue
            data = dict(
                ra=target.ra,
                dec=target.dec,
                mjd_min=Time.now().mjd-30., 
                mjd_max=Time.now().mjd,
                send_email=False, 
                comment=objectId,
            )
            logger.info(f"queue {objectId} {data['ra']:.4f} {data['dec']:.5f}")
            res = AtlasQuery.atlas_query(headers=self.headers, data=data)
            if res.status_code == 201:
                task_url = res.json()['url']
                self.submitted_queries[objectId] = task_url
                new_submissions.append(new_submissions)
            elif res.status_code == 429:
                self.throttled_queries.append(objectId) # ready to resubmit.
            logger.info(f"{objectId} status {res.status_code}")

        logger.info(f"{len(new_submissions)} new queries submitted")
        logger.info(f"{len(self.throttled_queries)} throttled")


    def retry_throttled_queries(self,):
        if len(self.throttled_queries) == 0:
            logger.info("no throttled queries")
            return
        resubmit_list = [
            objectId for objectId in self.throttled_queries if objectId not in self.submitted_queries
        ]
        # remove anything that might have been submitted twice.
        self.throttled_queries = [] # reset before retrying.
        self.queue_forced_photometry(objectId_list=resubmit_list)


    def retreive_forced_photometry(self, return_df=True):
        results = {}
        with requests.Session() as s:
            for objectId, task_url in self.submitted_queries.items():
                res = s.get(task_url, headers=self.headers)
                if res.status_code == 200:
                    finishtimestamp = res.json().get('finishtimestamp', None)
                    if finishtimestamp is None:
                        continue
                    
                    result_url = res.json().get('result_url', None)
                    if result_url is None:
                        raise AtlasQueryError(
                            f"there should be a result_url if finished at {finishtimestamp}"
                        )
                    textdata = s.get(result_url, headers=self.headers).text
                    df = pd.read_csv(
                        io.StringIO(
                            textdata.replace("###", "")
                        ), 
                        delim_whitespace=True
                    )
                    atlas_df_path = paths.atlas_data_path / f"{objectId}.csv"
                    ## concat to existing data.
                    if atlas_df_path.exists():
                        old_df = pd.read_csv(atlas_df_path)
                        df = pd.concat([old_df, df])
                        df.drop_duplicates(subset="Obs", inplace=True)

                    df.to_csv(atlas_df_path, index=False)
                    results[objectId] = df
                else:
                    logger.error(f"Atlas result_url gives status_code {res.status_code}")
                s.delete(task_url, headers=self.headers)

        for objectId in results.keys():
            self.submitted_queries.pop(objectId) # remove this from the data to keep checking for.
        return results

    def perform_all_tasks(self):
        self.retry_throttled_queries()
        self.queue_forced_photometry()
        self.retreive_forced_photometry()
        
if __name__ == "__main__":
    aqm = AtlasQueryManager.from_config_file(force_token=True)
