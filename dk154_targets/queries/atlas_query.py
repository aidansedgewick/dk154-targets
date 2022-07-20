import requests

class AtlasQueryError(Exception):
    pass

class AtlasQuery:
    @classmethod
    def atlas_query(cls, headers, data):
        res = requests.post(
            url=f"{cls.atlas_base_url}/queue/", headers=headers, data=data
        )
        return res