import numpy as np

import pandas as pd


import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

import sncosmo

from dustmaps import sfd

from dk154_targets.target import Target
from dk154_targets.scoring import default_score

from dk154_targets import paths

data_path = paths.data_path / "fink_SN_candidate_archive.csv"



archive_data = pd.read_csv(data_path)
rename_dict = {col:col.split(":")[-1] for col in archive_data.columns}
archive_data.rename(columns=rename_dict, inplace=True)

group_sizes = archive_data.groupby("objectId").size()
archive_data["n_det"] = archive_data.groupby("objectId")["magpsf"].transform("size")

archive_data.query("(n_det>=10) & (n_det < 50)", inplace=True)

#hist = plt.hist(group_sizes, bins=np.arange(0, 20))
#plt.show()


sfdq = sfd.SFDQuery()

print(list(archive_data.columns))


band_lookup = {1: "ztfg", 2: "ztfr"}

peak_error = []
time_since_best_peak = []
n_points = []
plotted = []
objectId_list = []

n_objects = len(np.unique(archive_data["objectId"]))

fig, ax = plt.subplots()

for ii, (objectId, target_history) in enumerate(archive_data.groupby("objectId")):

    if "tag" in target_history.columns:
        detections = target_history.query("tag=='valid'")
    else:
        detections = target_history

    #if len(detections) < 8:
    #    print(ii, n_objects, "ndet < 8")
    #    continue

    N_obs = {}
    for fid, fid_detections in detections.groupby("fid"):
        N_obs[fid] = len(fid_detections)
    if N_obs.get(1, 0) < 2 or N_obs.get(2, 0) < 2:
        print(ii, n_objects, "not enough detections per band")
        continue



    target_history.sort_values("jd")

    target = Target.from_target_history(objectId, target_history)
    t_ref = Time(target_history.iloc[-1]["jd"], format="jd")

    #score, score_comments, reject_comments = target.evaluate_target(
    #    default_score, None, t_ref=t_ref
    #)

    interval = detections["jd"].max() - detections["jd"].min()

    if interval > 50:
        print(ii, n_objects, f"too long interval {interval:.2f}" )
        continue

    #target_coord = SkyCoord(
    #    ra=detections.iloc[0]["ra"], dec=detections.iloc[0]["dec"], unit="deg"
    #)

    print(ii, n_objects)
    detections = target_history

    mwebv = sfdq(target.coord)

    sncosmo_data = Table(
        dict(
            time=detections["jd"].values, # .values is an np array...
            band=detections["fid"].map(band_lookup).values,
            mag=detections["magpsf"].values,
            magerr=detections["sigmapsf"].values,
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
    model.set(mwebv=mwebv)
    fitting_params = model.param_names
    fitting_params.remove("mwebv")

    try:
        full_result, full_fitted_model = sncosmo.fit_lc(
            sncosmo_data, model,
            fitting_params,
            bounds={'z':(0.005, 0.5)}
        )
        full_parameters = {}
    except Exception as e:
        print(f"{objectId} failed!")
        continue

    target.models.append(full_fitted_model)

    plotted.append(objectId)
    try:
        lc_fig = target.plot_lightcurve(t_ref=t_ref)
        lc_fig.savefig(paths.base_path / f"scripts/lc_plots/{objectId}.png")
        plt.close(lc_fig)
    except Exception as e:
        pass


    for cut_ii in range(4, len(target_history)-1):
        detections = target_history[:cut_ii]

        t_latest = Time(detections.iloc[-1]["jd"], format="jd")

        sncosmo_data = Table(
            dict(
                time=detections["jd"].values, # .values is an np array...
                band=detections["fid"].map(band_lookup).values,
                mag=detections["magpsf"].values,
                magerr=detections["sigmapsf"].values,
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
        model.set(mwebv=mwebv)
        fitting_params = model.param_names
        fitting_params.remove("mwebv")

        try:
            result, fitted_model = sncosmo.fit_lc(
                sncosmo_data, model,
                fitting_params,
                bounds={'z':(0.005, 0.5)}
            )
        except Exception as e:
            print(f"{objectId} failed!")

        dt = full_fitted_model["t0"] - fitted_model["t0"]

        peak_error.append(dt)
        time_since_best_peak.append(t_latest.jd - full_fitted_model["t0"])
        objectId_list.append(objectId)
        n_points.append(len(detections))


        #print(dt)

        #fig = target.plot_lightcurve(t_ref=t_ref)

        #target_cut = Target.from_target_history(objectId, detections)
        #target_cut.models.append(fitted_model)
        #target_cut.plot_lightcurve(fig=fig, t_ref=t_ref)


peak_error = np.array(peak_error)
time_since_best_peak = np.array(time_since_best_peak)
n_points = np.array(n_points)
objectId_list = np.array(objectId_list)
df = pd.DataFrame({
    "peak_error": peak_error, "time_since_best_peak": time_since_best_peak, 
    "n_points": n_points, "objectId": objectId_list
})

tr_fig, tr_ax = plt.subplots()
for objectId, obj_df in df.groupby("objectId"):
    tr_ax.plot(obj_df["time_since_best_peak"], obj_df["peak_error"], color="k")

fig, axes = plt.subplots(3, 2)
axes = axes.flatten()
plot_ranges = [[4], [5], [6], [7, 8], [9, 10, 11], [x for x in range(12, max(n_points))]]
for ii, plot_range in enumerate(plot_ranges):
    mask = np.in1d(n_points, plot_range)
    err_ii = peak_error[ mask ]
    time_ii = time_since_best_peak[ mask ]
    
    ax = axes[ii]
    ax.scatter(time_ii, err_ii, s=4, color="k", alpha=0.2)
    ax.set_ylim(-5, 5)
    ax.set_xlim(-30, 50)

timing_fig, timing_ax = plt.subplots()
timing_ax.scatter(time_since_best_peak, peak_error, s=5, fc="grey", ec="none", marker=".")
timing_ax.set_ylim(-5, 5)
timing_ax.set_xlim(-30, 50)


quantiles = [(0.023, 0.977), (0.159, 0.841)]
t_bins = np.arange(-30, 50, 1.0)
t_mids = 0.5 * (t_bins[:-1] + t_bins[1:])
sigma1_low = np.zeros(len(t_bins)-1)
sigma1_high = np.zeros(len(t_bins)-1)
sigma2_low = np.zeros(len(t_bins)-1)
sigma2_high = np.zeros(len(t_bins)-1)

for ii, (low, high) in enumerate(zip(t_bins[:-1], t_bins[1:])):
    dat = peak_error[ (low < time_since_best_peak) & (time_since_best_peak < high) ]
    if len(dat) < 10:
        continue
    s1_l, s1_h, s2_l, s2_h = np.quantile(dat, [0.159, 0.841, 0.023, 0.977])
    sigma1_low[ii] = s1_l
    sigma1_high[ii] = s1_h
    sigma2_low[ii] = s2_l
    sigma2_high[ii] = s2_h

timing_ax.plot(t_mids, sigma1_high, ls="-", color="k")
timing_ax.plot(t_mids, sigma1_low, ls="-", color="k")
timing_ax.plot(t_mids, sigma2_high, ls=":", color="k")
timing_ax.plot(t_mids, sigma2_low, ls=":", color="k")

before_peak_mask = (time_since_best_peak < 0)
n_points_bins = np.arange(3.0, 15.0, 1.0) - 0.5
n_points_mids = 0.5 * (n_points_bins[:-1] + n_points_bins[1:])


df_before_peak = df.query("time_since_best_peak < 0")
n_alerts_before_peak = df_before_peak.groupby("objectId")["n_points"].max()
print(n_alerts_before_peak)
n_points_hist, _ = np.histogram(n_alerts_before_peak.values, bins=n_points_bins)

print(len(n_points_mids), len(n_points_hist))

fig, ax = plt.subplots()
ax.bar(n_points_mids, n_points_hist, width=1, )

print(f"includes {len(plotted)} objects")

plt.show()