import getpass
import io
import requests

import pandas as pd

class AtlasQueryError(Exception):
    pass

class AtlasQuery:

    atlas_base_url = "https://fallingstar-data.com/forcedphot"


    @staticmethod
    def get_atlas_token():
        usr = input("Username: ")
        pwd = getpass.getpass()
        url = f"{AtlasQuery.atlas_base_url}/api-token-auth/"
        res = requests.post(url, data=dict(username=usr, password=pwd))
        token = res.json()["token"]
        print(f"your token is: {token}")

    @classmethod
    def atlas_query(cls, headers, data):
        res = requests.post(
            url=f"{cls.atlas_base_url}/queue/", headers=headers, data=data
        )
        return res

    @staticmethod
    def process_df(photom_data, text_processed=False):
        if not text_processed:
            textdata = photom_data.text
        else:
            textdata = photom_data
        df = pd.read_csv(
            io.StringIO(
                textdata.replace("###", "")
            ), 
            delim_whitespace=True
        )
        return df
        