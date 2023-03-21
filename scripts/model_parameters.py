import logging
import os

import numpy as np
import pandas as pd
import tqdm

import matplotlib.pyplot as plt

from astropy.time import Time

from dk154_targets.target import Target
from dk154_targets.target_selector import TargetSelector
from dk154_targets.modelling import default_sncosmo_model
from dk154_targets.scoring import default_score

from dk154_targets import paths

logger = logging.getLogger(__name__.split(".")[-1])

fink_archive_path = paths.archive_path / "fink"
lc_archive_path = fink_archive_path / "light_curves"

alert_df_path = fink_archive_path / "alert_archive.csv"
alert_df = pd.read_csv(alert_df_path)
alert_df.sort_values("jd", inplace=True)


### plot fraction of alerts which have lc data downloaded.

jd_bins = np.arange(int(alert_df["jd"].min()), int(alert_df["jd"].max())+2.0, 1.0)
#alert_count = np.zeros(len(jd_bins)-1)
#lc_count = np.zeros(len(jd_bins)-1)

# for ii, alert in alert_df.iterrows():
#     if ii % 100 == 0:
#         print(ii, len(alert_df))
#     objectId = alert["objectId"]
#     alert_jd = alert["jd"]
#     bin_idx = np.digitize(alert_jd, jd_bins) - 1
#     assert bin_idx > -1
#     alert_count[bin_idx] = alert_count[bin_idx] + 1
#     lc_path = lc_archive_path / f"{objectId}.csv"
#     if lc_path.exists():
#         lc_count[bin_idx] = lc_count[bin_idx] + 1


lc_paths = [lc_archive_path / f"{objectId}.csv" for objectId in alert_df["objectId"]]
exist_mask = np.array([p.exists() for p in lc_paths])

alert_count, _ = np.histogram(alert_df["jd"], bins=jd_bins)
lc_count, _ = np.histogram(alert_df["jd"][ exist_mask ], bins=jd_bins)

frac = lc_count / alert_count


ticks = np.linspace(int(alert_df["jd"].min()), int(alert_df["jd"].max())+2.0, 10)
ticklabels = [
    Time(x, format="jd").strftime("%y-%m-%d") for x in ticks
]

fig, ax = plt.subplots()
ax.plot(jd_bins[:-1] - jd_bins[0], frac)
#ax.set_xticks(ticks)
#ax.set_xticklabels(ticklabels, rotation=45)
plt.show()




matched_summary_df_path = paths.archive_path / "lc_summary_tns_matched.csv"
matched_summary_df = pd.read_csv(matched_summary_df_path)

summary_df_path = fink_archive_path / "light_curve_summaries.csv"
summary_df = pd.read_csv(summary_df_path)

matched_summary_df.set_index("objectId", verify_integrity=True)

valid_objects = summary_df.query(
    "(peak_mag < 19.5) & (timespan < 60.) & (n_detections > 10) "
    " & (N_obs_1 > 2) & (N_obs_2 > 2) " #& (Redshift > 0)"
)
logger.info(f"{len(valid_objects)} valid objects" )





model_summaries_path = fink_archive_path / "model_summaries.csv"
redshift_model_summaries_path = fink_archive_path / "redshift_model_summaries.csv"
if not model_summaries_path.exists():
    logger.info("compute models and sumaries")


    model_data_list = []
    redshift_model_data_list = []

    for ii, object_row in valid_objects.iterrows(): #["objectId"]
        objectId = object_row["objectId"]

        lc_path = lc_archive_path / f"{objectId}.csv"

        target_history = pd.read_csv(lc_path)
        target_history.sort_values("jd", inplace=True) # avoid settingcopywarning...!
        detections = target_history.query("tag=='valid'")
        full_t_ref = Time(detections["jd"].values[-1], format="jd")


        for ii in range(2):
            target = Target.from_fink_lightcurve(objectId, target_history)
            target.compile_target_history(t_ref=full_t_ref)
            if ii == 1:

                if objectId in matched_summary_df:
                    redshift_val = matched_summary_df.loc[objectId]["Redshift"]
                    target.tns_data.parameters["Redshift"] = redshift_val
                else:
                    continue

            full_model = default_sncosmo_model(target) #.model_target()
            if full_model is None:
                logger.info(f"{objectId} no full model!!")
                continue
            full_model_t0 = full_model["t0"]
            target.models.append(full_model_t0)

            pre_peak_data = detections.query("jd < @full_model_t0")
            if len(pre_peak_data) < 4:
                logger.info(f"{objectId} skip model")
                continue

            for N in range(4, len(detections)):
                detections_to_N = detections[:N]
                t_ref = Time(detections_to_N["jd"].values[-1], format="jd")

                time_since_true_peak = t_ref.jd - full_model_t0
                if time_since_true_peak > 10:
                    logger.info(f"{objectId} N={N} skip: dt={time_since_true_peak:.2f}")
                    continue

                target.compile_target_history(t_ref=t_ref)
                target_to_N_model = default_sncosmo_model(target)
                if target_to_N_model is None:
                    logger.info(f"{objectId} N={N} model failed, skip")
                    continue
                target.models.append(target_to_N_model)

                #fig = target.plot_lightcurve(t_ref=full_t_ref)
                model_t0 = target_to_N_model["t0"]
                peak_time_error = model_t0 - full_model_t0

                model_to_N_data = dict(
                    objectId=objectId, N_data_points=N, N_full=len(detections),
                    peak_time_error=peak_time_error, time_since_true_peak=time_since_true_peak
                )
                if ii==0:
                    model_data_list.append(model_to_N_data)
                elif ii==1:
                    model_to_N_data["redshift"] = redshift_val
                    redshift_model_data_list.append(model_to_N_data)


        #plt.show()
        

    model_summaries = pd.DataFrame(model_data_list)
    model_summaries.to_csv(model_summaries_path, index=False)

    redshift_model_summaries = pd.DataFrame(model_data_list)
    redshift_model_summaries.to_csv(redshift_model_summaries_path, index=False)

    #fig = target.plot_lightcurve(t_ref=t_ref)
    #plt.show()
else:
    logger.info("read existing model summaries")
    model_summaries = pd.read_csv(model_summaries_path)

peak_mag_lookup = {}
for objectId in np.unique(model_summaries["objectId"].values):
    row = summary_df[ summary_df["objectId"] == objectId]
    peak_mag_lookup[objectId] = row["peak_mag"].values[0]

model_summaries["peak_mag"] = model_summaries["objectId"].map(peak_mag_lookup)

mag_limits = [0.0, 18, 18.5, 19.5]

timing_fig, timing_axes = plt.subplots(3, 1, figsize=(5.4, 8))
#timing_ax.axhline(0., color="k")

dt_bins = np.arange(-12, 10, 1.0)
dt_mids = 0.5 * (dt_bins[:-1] + dt_bins[1:])

for jj, (mag_bright, mag_faint) in enumerate(zip(mag_limits[:-1], mag_limits[1:])):

    timing_ax = timing_axes[jj]
    timing_ax.axhline(0., color="grey")
    timing_ax.axvline(0., color="grey", ls="--")

    model_summaries_ii = model_summaries.query("(@mag_bright<peak_mag) & (peak_mag<@mag_faint)")

    #timing_fig.suptitle(f"peak mag < {mag_lim:.1f}")

    peak_error = model_summaries_ii["peak_time_error"].values
    peak_dt = model_summaries_ii["time_since_true_peak"].values
    timing_ax.scatter(peak_dt, peak_error, color=f"C{jj}", s=1, zorder=2, alpha=0.2)
    timing_ax.text(
        0.05, 0.95, r"$" + f"{mag_bright} < m < {mag_faint}" + r"$", 
        ha="left", va="top", transform=timing_ax.transAxes
    )

    sigma = np.zeros((2, len(dt_mids)))
    median = np.zeros(len(dt_mids))

    for ii, (low, high) in enumerate(zip(dt_bins[:-1], dt_bins[1:])):
        dat = peak_error[ (low < peak_dt) & (peak_dt < high)]
        if len(dat) == 0:
            continue

        median[ii] = np.median(dat)
        sigma[:, ii] = np.quantile(dat, [0.159, 0.841])

    if jj < 2:
        timing_ax.set_xticks([])

    timing_ax.plot(dt_mids, median, color=f"C{jj}", ls="-", zorder=4)
    timing_ax.plot(dt_mids, sigma[0,:], color=f"C{jj}", ls="--", zorder=4)
    timing_ax.plot(dt_mids, sigma[1,:], color=f"C{jj}", ls="--", zorder=4)

    timing_ax.set_xlim(-12, 10)
    timing_ax.set_ylim(-8, 8)

timing_axes[1].set_ylabel(r"$t_0$ error (days) ", fontsize=14)
timing_axes[2].set_xlabel(r"Alert time from best $t_0$ (days)", fontsize=14)

timing_fig.tight_layout()
timing_fig.subplots_adjust(hspace=0.)

print(f"fig includes {len(np.unique(model_summaries['objectId']))} obj")

plt.show()




### model parameters


