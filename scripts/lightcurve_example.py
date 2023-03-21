import numpy as np

import matplotlib.pyplot as plt

from astropy.time import Time

from dk154_targets import TargetSelector
from dk154_targets.target import Target, plot_lightcurve
from dk154_targets.modelling import default_sncosmo_model
from dk154_targets import paths

ts = TargetSelector.from_config() # use default config file!

objectId = "ZTF22abqkzuq"

t_ref0 = Time("2022-11-04 00:00")
t_ref1 = Time("2022-12-11 00:00")


target_lc_path = paths.fink_data_path / f"lightcurves/{objectId}.csv"
target = Target.from_fink_query(objectId, withupperlim=True) #, target_lc_path)
print(target.fink_data.lightcurve[["jd","ra"]])
ts.add_target(target)
ts.fink_query_manager.update_cutouts()

chains = []
chain_labels = []

ts.compile_all_target_histories(t_ref=t_ref0)
ts.model_targets(default_sncosmo_model)
t_peak = Time(ts.target_lookup[objectId].models[-1]["t0"], format="jd")
fig = plot_lightcurve(ts.target_lookup[objectId], t_ref=t_peak, forecast_days=15., xlabel="time from predicted peak [days]")
chains.append(  ts.target_lookup[objectId].models[-1].res.get("samples", None) )
chain_labels.append("early target, ZTF only")
fig.savefig("/home/aidan/early_ztf_only.pdf")

ts.compile_all_target_histories(t_ref=t_ref1)
ts.model_targets(default_sncosmo_model)
t_peak = Time(ts.target_lookup[objectId].models[-1]["t0"], format="jd")
fig = plot_lightcurve(ts.target_lookup[objectId], t_ref=t_peak, forecast_days=35., xlabel="time from predicted peak [days]")
chains.append( ts.target_lookup[objectId].models[-1].res.get("samples", None) )
chain_labels.append("older target, ZTF only")
fig.savefig("/home/aidan/older_ztf_only.pdf")


ts.atlas_query_manager.read_existing_atlas_lightcurves()                      

ts.compile_all_target_histories(t_ref=t_ref0)
ts.model_targets(default_sncosmo_model)
t_peak = Time(ts.target_lookup[objectId].models[-1]["t0"], format="jd")
fig = plot_lightcurve(ts.target_lookup[objectId], t_ref=t_peak, forecast_days=15., xlabel="time from predicted peak [days]")
chains.append(  ts.target_lookup[objectId].models[-1].res.get("samples", None) )
chain_labels.append("early target, ZTF+ATLAS")
fig.savefig("/home/aidan/young_ztf_atlas.pdf")

ts.compile_all_target_histories(t_ref=t_ref1)
ts.model_targets(default_sncosmo_model)
t_peak = Time(ts.target_lookup[objectId].models[-1]["t0"], format="jd")
fig = plot_lightcurve(ts.target_lookup[objectId], t_ref=t_peak, forecast_days=35., xlabel="time from predicted peak [days]")
chains.append(  ts.target_lookup[objectId].models[-1].res.get("samples", None) )
chain_labels.append("older target, ZTF+ATLAS")
fig.savefig("/home/aidan/older_ztf_atlas.pdf")

import pygtc

if not any([c is None for c in chains]):

    param_names = ts.target_lookup[objectId].models[-1].res.get("samples", None)

    pygtc.plotGTC(chains=chains, chainLabels=chain_labels, filledPlots=False)



plt.show()

