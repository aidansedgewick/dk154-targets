import gzip
import io
import logging
from pathlib import Path

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.io import fits

logger = logging.getLogger(__name__.split(".")[-1])

def chunk_list(l, size=50):
    n = int(len(l) / size) + 1
    logger.info(f"{n} chunks size <={size}")
    for start in range(0, len(l), size):
        yield l[start:start+size]


def readstamp(stamp: str, return_type='array') -> np.array:
    """ 
    copied and pasted directly from 
    https://github.com/astrolabsoftware/fink-science-portal/blob/master/apps/utils.py#L201 ...
    
    Read the stamp data inside an alert.
    Parameters
    ----------
    alert: dictionary
        dictionary containing alert data
    field: string
        Name of the stamps: cutoutScience, cutoutTemplate, cutoutDifference
    return_type: str
        Data block of HDU 0 (`array`) or original FITS uncompressed (`FITS`) as file-object.
        Default is `array`.
    Returns
    ----------
    data: np.array
        2D array containing image data (`array`) or FITS file uncompressed as file-object (`FITS`)
    """
    if stamp is None:
        logger.warn("postage stamp is none")
        return None
    with gzip.open(io.BytesIO(stamp), 'rb') as f:
        with fits.open(io.BytesIO(f.read())) as hdul:
            if return_type == 'array':
                data = hdul[0].data
            elif return_type == 'FITS':
                data = io.BytesIO()
                hdul.writeto(data)
                data.seek(0)
            else:
                raise ValueError("choose return_type from 'array', 'FITS'")
    return data


def dustmaps_setup()
    from dustmaps.config import config
    if not Path(config.fname).exists():
        logger.info("reset dustmap config")
        config.reset()

    from dustmaps import sfd
    logger.info("check for sfd map")
    try:
        sfd.query(SkyCoord(ra=0., dec=0., unit="deg"))
    except:
    sfd.fetch()