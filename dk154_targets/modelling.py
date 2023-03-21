import logging
import traceback
import warnings
import yaml

import numpy as np

from astropy.table import Table

try:
    import sncosmo
except:
    sncosmo = None

from dustmaps import sfd

sfdq = sfd.SFDQuery()

logger = logging.getLogger("default_modelling")



def default_sncosmo_model(target, **kwargs):
    ztf_band_lookup = {1: "ztfg", 2: "ztfr"}

    use_mcmc = kwargs.get("use_mcmc", None)
    if use_mcmc is None:
        logger.info("'use_mcmc' not found in selector_config.modelling: set to False ")
        use_mcmc = False

    logger.info(f"{target.objectId} fit sncosmo model")

    if "tag" in target.target_history.columns:
        tag_query = "(tag=='valid') or (tag=='badquality')"
        detections = target.target_history.query(tag_query)
        logger.info(f"use data {tag_query}")
    else:
        detections = target.target_history

    sncosmo_data = Table(
        dict(
            time=detections["jd"].values, # .values is an np array...
            #band=detections["band"].map(ztf_band_lookup).values,
            band=detections["band"].values,
            mag=detections["mag"].values,
            magerr=detections["magerr"].values,
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

    # for known TNS redshifts...
    known_redshift = target.tns_data.parameters.get("Redshift", None)
    if known_redshift is not None:
        logger.info(f"use known redshift z={known_redshift:.3f}")
        model.set(z=known_redshift)
        fitting_params.remove("z")
        bounds = {}
    else:
        bounds = {'z':(0.005, 0.5)}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lsq_result, lsq_fitted_model = sncosmo.fit_lc(
                sncosmo_data, model,
                fitting_params,
                bounds=bounds
            )
            if use_mcmc:
                result, fitted_model = sncosmo.mcmc_lc(
                    sncosmo_data, lsq_fitted_model,
                    fitting_params,
                    nsamples=5000,
                    nwalkers=32,
                    bounds=bounds
                )
            else:
                fitted_model = lsq_fitted_model
                result = lsq_result

        fitted_model.res = result
        logger.info(f"sncosmo model fitting done")
    except Exception as e:
        logger.warning(f"model fitting FAILED")
        fitted_model = None
        tr = traceback.format_exc()
        print(tr)
        #    raise Exception(e)
    return fitted_model
