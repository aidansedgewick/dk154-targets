import logging
import traceback
import warnings

import numpy as np

from astropy.table import Table

try:
    import sncosmo
except:
    sncosmo = None

from dustmaps import sfd

sfdq = sfd.SFDQuery()

logger = logging.getLogger("default_modelling")

def default_sncosmo_model(target):
    band_lookup = {1: "ztfg", 2: "ztfr"}

    sncosmo_data = Table(
        dict(
            time=target.target_history["jd"].values, # .values is an np array...
            band=target.target_history["fid"].map(band_lookup).values,
            mag=target.target_history["magpsf"].values,
            magerr=target.target_history["sigmapsf"].values,
        )
    )


    sncosmo_data["flux"] = 10 ** (0.4 * (8.9 - sncosmo_data["mag"]))
    sncosmo_data["fluxerr"] = sncosmo_data["flux"] * sncosmo_data["magerr"] * np.log(10.) / 2.5
    sncosmo_data["zp"] = np.full(len(sncosmo_data), 25.0)
    sncosmo_data["zpsys"] = np.full(len(sncosmo_data), "ab")

    dust = sncosmo.F99Dust()
    model = sncosmo.Model(
        source="salt2", effects=[dust], effect_names=["mw"], effect_frames=["obs"]
    )
    mwebv = sfdq(target.coord)
    model.set(mwebv=mwebv)
    fitting_params = model.param_names
    fitting_params.remove("mwebv")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, fitted_model = sncosmo.fit_lc(
                sncosmo_data, model,
                fitting_params,
                bounds={'z':(0.005, 0.5)}
            )
        fitted_model.res = result
        logger.info(f"model fit for {target.objectId}")
    except Exception as e:
        logger.warning(f"model fitting failed for {target.objectId}")
        fitted_model = None
        tr = traceback.format_exc()
        print(tr)
        #    raise Exception(e)
    return fitted_model
