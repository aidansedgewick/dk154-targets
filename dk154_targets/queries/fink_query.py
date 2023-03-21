import logging
import requests
import time

import numpy as np
import pandas as pd

from astropy.time import Time

from fink_client.avroUtils import write_alert, _get_alert_schema
from fink_client.consumer import AlertConsumer

from dk154_targets import paths

from dk154_targets.utils import readstamp

logger = logging.getLogger(__name__.split(".")[-1])

class FinkQueryError(Exception):
    pass

class FinkQuery:
    """See https://fink-portal.org/api"""


    fink_latests_url = 'https://fink-portal.org/api/v1/latests'
    fink_objects_url = 'https://fink-portal.org/api/v1/objects'
    fink_explorer_url = 'https://fink-portal.org/api/v1/explorer'
    fink_cutout_url = 'https://fink-portal.org/api/v1/cutouts'

    imtypes = ("Science", "Template", "Difference")

    def __init__(self):
        pass

    @classmethod
    def fix_column_names(cls, df):
        column_lookup = {
            col: col.split(":")[1] if ":" in col else col for col in df.columns
        }
        return df.rename(column_lookup, axis=1)

    @classmethod
    def process_response(cls, res, return_df=True, fix_column_names=True):
        if res.status_code in [404, 500, 504]:
            logger.error("\033[31;1merror rasied\033[0m")
            if res.elapsed.total_seconds() > 58.:
                logger.error("likely a timeout error")
            raise FinkQueryError(res.content.decode())
        if not return_df:
            return res
        df = pd.read_json(res.content)
        if fix_column_names:
            return cls.fix_column_names(df)
        return df


    @classmethod
    def query_latest_alerts(cls, return_df=True, fix_column_names=True, **kwargs):
        res = requests.post(cls.fink_latests_url, json=kwargs)
        if res.status_code != 200:
            logger.warning(f"query_latest_alerts status {res.status_code}")
        return cls.process_response(res, return_df=True, fix_column_names=True)


    @classmethod
    def query_objects(cls, return_df=True, fix_column_names=True, **kwargs):
        res = requests.post(cls.fink_objects_url, json=kwargs)
        if res.status_code != 200:
            logger.warning(f"query_objects status {res.status_code}")
        return cls.process_response(res, return_df=True, fix_column_names=True)


    @classmethod
    def query_database(cls, return_df=True, fix_column_names=True, **kwargs):
        res = requests.post(cls.fink_explorer_url, json=kwargs)
        if res.status_code != 200:
            logger.warning(f"query_database status {res.status_code}")
        return cls.process_response(res, return_df=True, fix_column_names=True)


    @classmethod
    def get_cutout(cls, imtype, **kwargs):
        if imtype not in cls.imtypes:
            raise ValueError(f"choose imtype from {cls.imtypes}")
        imtype_key = 'b:cutout'+imtype+'_stampData' # gives eg 'b:cutoutScience_stampData'

        json_data = {
            'kind': imtype,
            'output-format': 'array',
        }
        json_data.update(kwargs)
        im_req = requests.post(
            cls.fink_cutout_url, json=json_data
        )
        try:
            im_df = pd.read_json(im_req.content)
        except Exception as e:
            logger.warning(f"on request for {imtype} stamp: {e}")
            return None

        try:
            im = np.array(im_df[imtype_key].values[0], dtype=float)
        except:
            logger.warning(f"on {imtype} stamp np conversion: {e}")
            return None

        return im
            