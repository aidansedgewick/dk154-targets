import copy
import json
import logging
import re
import requests
import time
import yaml

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table
from astropy.time import Time

from dk154_targets.query_managers import GenericQueryManager
from dk154_targets.query_managers.query_manager_utils import (
    get_file_update_interval
)


from dk154_targets import paths

logger = logging.getLogger(__name__.split(".")[-1])



allowed_tns_parameters = [
    "discovered_period_value", "discovered_period_units", "unclassified_at", 
    "classified_sne", "include_frb",
    "name", "name_like", "isTNS_AT", "public", "ra", "decl", "radius", 
    "coords_unit", "reporting_groupid[]",
    "groupid[]", "classifier_groupid[]", "objtype[]", "at_type[]", "date_start[date]",
    "date_end[date]",  "discovery_mag_min", "discovery_mag_max", 
    "internal_name", "discoverer", "classifier",
    "spectra_count", "redshift_min", "redshift_max", "hostname", 
    "ext_catid", "ra_range_min", "ra_range_max",
    "decl_range_min", "decl_range_max", "discovery_instrument[]", 
    "classification_instrument[]",
    "associated_groups[]", "official_discovery", "official_classification", 
    "at_rep_remarks", "class_rep_remarks",
    "frb_repeat", "frb_repeater_of_objid", "frb_measured_redshift", 
    "frb_dm_range_min", "frb_dm_range_max",
    "frb_rm_range_min", "frb_rm_range_max", "frb_snr_range_min", 
    "frb_snr_range_max", "frb_flux_range_min",
    "frb_flux_range_max", "format", "num_page"
]

# the literal square brackets are included in the TNS keywords...


class TNSQueryManager(GenericQueryManager):

    name = "tns"

    allowed_parameters = allowed_tns_parameters
    tns_base_url = "https://www.wis-tns.org/search"

    tns_data_path = paths.tns_data_path


    def __init__(self, tns_config: dict, target_lookup: dict):
        self.tns_config = tns_config

        self.target_lookup = target_lookup

        tns_user = self.tns_config.get("user")
        tns_uid = self.tns_config.get("uid")
        
        marker_dict = dict(tns_id=str(tns_uid), type="user", name=tns_user)
        marker_str = json.dumps(marker_dict)
        # a literal string: '{"tns_id": "1234", "type": "user", "name": "your_name"}'

        self.tns_marker = f"tns_marker{marker_str}"
        # literal braces from marker_str will be included!
        # see https://docs.python.org/3/library/string.html#formatstrings
        self.tns_parameters = self.tns_config.get("tns_parameters", {})

        self.recently_checked_targets = []

        self.tns_headers = {"User-Agent": self.tns_marker}

        self.tns_data_path.mkdir(exist_ok=True, parents=True)


    @classmethod
    def from_default_config(cls, target_lookup: dict):
        config_path = paths.config_path / "selector_config.yaml"
        with open(config_path, "r") as f:
            selector_config = yaml.load(f, Loader=yaml.FullLoader)

        tns_config = selector_config.get("query_managers", {}).get(cls.name, None)

        if tns_config is None:
            raise KeyError(f"no {cls.name} in \"query_managers\" in config")
        return cls(tns_config, target_lookup)

        



    def perform_query(self, tns_search_params: dict=None, sleep_time=None, param_log=False):

        tns_search_params = tns_search_params or {}
        search_params = copy.deepcopy(self.tns_parameters)
        search_params.update(tns_search_params)

        url_components = []
        for k, v in search_params.items():
            if not k in self.allowed_parameters:
                logger.warning(f"unexpected keyword {k}")

            if isinstance(v, bool):
                v = int(v)               
            url_components.append( f"&{k}={str(v)}")
        param_url = "".join(url_components)
        # param_url = "".join(f"&{k}={str(v)}" for k, v in search_params.items())

        # Now do the query for each page of 'num_page' results...

        num_page = search_params.pop("num_page", 50)

        if int(num_page) > 50:
            logger.warning("num_page max is 50...")
            num_page = 50

        page = 0

        df_list = []
        while True:
            t1 = time.clock()

            url = f"{self.tns_base_url}?{param_url}&page={page}&num_page={num_page}"
            response = requests.post(url, headers=self.tns_headers)
            response_data = response.text.splitlines()

            columns = response_data[0].replace('"','').split(",")
            quoted = re.compile('"[^"]*"') # strip all quotes eg. "SN 2022sai" -> SN 2022sai
            # comma split not useful here as some entries are eg. "ASAS-SN, ALeRCE"
            df_data = [quoted.findall(row) for row in response_data[1:]]

            df = pd.DataFrame(df_data, columns=columns)
            for ii, col in enumerate(df.columns):
                df.iloc[:, ii] = df.iloc[:, ii].str.replace('"', '')
                if col in ["Redshift", "Discovery Mag/Flux", "Host Redshift"]:
                    df.iloc[:, ii] = pd.to_numeric(df.iloc[:, ii], errors="coerce")
            N_results = len(df)

            status = response.status_code
            req_limit = response.headers.get("x-rate-limit-limit")
            req_remaining = response.headers.get("x-rate-limit-remaining")
            req_reset = response.headers.get("x-rate-limit-reset")

            if (response.status_code != 200) or N_results==0:
                logger.info(f"break: page {page} status={status}, len={N_results}")
                logger.info(f"{req_remaining}/{req_limit} requests left (reset {req_reset}s)")
                break            

            df_list.append(df)
            
            logger.info(f"{N_results} results from page {page}")
            logger.info(f"{req_remaining}/{req_limit} requests left (reset {req_reset}s)")

            if int(req_remaining) < 2:
                logger.info(f"waiting {req_reset}s for reset...")
                time.sleep(int(req_reset)+1.0)

            t2 = time.clock()

            if N_results < int(num_page):
                logger.info(f"break after results < {num_page}")

            if sleep_time is not None:
                query_time = (t2-t1)
                if sleep_time - query_time > 0:
                    time.sleep(sleep_time-query_time)

            page = page + 1

        if not df_list:
            return pd.DataFrame() # empty dataframe.

        results_df = pd.concat(df_list, ignore_index=True)
        return results_df


    def perform_match(
        self, tns_results: pd.DataFrame, seplimit=4*u.arcsec, ignore_recently_checked=False
    ):

        tns_results.set_index("ID", verify_integrity=True, inplace=True)

        logger.info("match TNS data to targets")

        if ignore_recently_checked:
            raise NotImplementedError
            # TODO: make a new dict of targets which are not in self.recently_checked_targets

        tns_candidate_data = []
        tns_candidate_coords = []

        matching_name = 0 # how many can we do "easily" with the internal name?
        for ii, row in tns_results.iterrows():
            tns_internal_name = row["Disc. Internal Name"]

            # can immediately match the targets whose internal name matches ours.
            if tns_internal_name in self.target_lookup:
                target = self.target_lookup[tns_internal_name]
                tns_data = row.to_dict()
                if not target.tns_data.parameters:
                    target.tns_data.parameters = tns_data
                    target.updated = True
                    matching_name = matching_name + 1
                else:
                    logger.warning(f"{target.objectId} already has TNS data")

            tns_candidate_data.append(row.to_dict())
            coord = SkyCoord(row.RA, row.DEC, unit=(u.hourangle, u.deg))
            tns_candidate_coords.append(coord)

        tns_candidate_coords = SkyCoord(tns_candidate_coords) # need this format for match.

        if matching_name > 0:
            logger.info(f"{matching_name} matched on objectId")

        target_candidate_objectIds = []
        target_candidate_coords = []
        for objectId, target in self.target_lookup.items():
            if target.tns_data.parameters:
                continue
            target_candidate_objectIds.append(objectId)
            target_candidate_coords.append(target.coord)
        target_candidate_coords = SkyCoord(target_candidate_coords)
        
        target_match_idx, tns_match_idx, skysep, _ =  search_around_sky(
            target_candidate_coords, tns_candidate_coords, seplimit
        )
        # tns_match_idx are indices of coords1 (targ_cand)

        for ii, (idx1, idx2) in enumerate(zip(target_match_idx, tns_match_idx)):
            objectId = target_candidate_objectIds[idx1]
            target = self.target_lookup[objectId]

            tns_data = tns_candidate_data[idx2]

            target.tns_data.parameters = tns_data
            target.updated = True

        logger.info(f"{len(skysep)} matched on crossmatch (<{seplimit})")

        self.recently_checked_targets.extend(target_candidate_objectIds)
            


    def perform_all_tasks(self, t_ref: Time=None):
        t_ref = t_ref or Time.now()


        tns_results_path = self.tns_data_path / "tns_results.csv"

        tns_update_interval = get_file_update_interval(tns_results_path, t_ref=t_ref)


        tomorrow = Time(t_ref.mjd+1, format="mjd").iso.split()[0]
        yesterday = Time(t_ref.mjd-1, format="mjd").iso.split()[0]
        lookback_start = Time(t_ref.mjd-60, format="mjd").iso.split()[0]

        if tns_update_interval * u.day > 1 * u.day:
            logger.info("perform longer query")

            tomorrow = Time(t_ref.mjd+1, format="mjd").iso.split()[0]

            tns_parameters = {
                "date_start[date]": lookback_start,
                "date_end[date]": tomorrow,
            }
            results = self.perform_query(tns_parameters)
            results.to_csv(tns_results_path, index=False)
            self.recently_checked_targets = [] # reset recently checked targets, to try again.

        elif tns_update_interval * u.day > 60. * u.min:
            logger.info("check updates today")
            tns_parameters = {
                "date_start[date]": yesterday,
                "date_end[date]": tomorrow,
            }
        
            new_results = self.perform_query(tns_parameters)
            if not tns_results_path.exists():
                logger.error("should not be in elif block if tns_results.csv does not exist!")
                raise FileExistsError

            if not new_results.empty:
                existing_results = pd.read_csv(tns_results_path)
                results = pd.concat([existing_results, new_results])
                results.to_csv(tns_results_path, index=False)
                self.recently_checked_targets = [] # reset recently checked, to try again.
        
        results = pd.read_csv(tns_results_path)
        self.perform_match(results, ignore_recently_checked=False)

if __name__ == "__main__":

    config_path = paths.config_path / "selector_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tns_config = config.get("query_managers", {}).get("tns", {})

    print(tns_config)

    qm = TNSQueryManager(tns_config, {})
    qm.perform_query()