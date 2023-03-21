from code import interact
import logging
import os

import numpy as np
import pandas as pd
import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm


from astropy import units as u
from astropy.time import Time, TimeDelta

from dk154_targets.target import Target
from dk154_targets.target_selector import TargetSelector
from dk154_targets.modelling import default_sncosmo_model
from dk154_targets.scoring import default_score

from dk154_targets import paths

logger = logging.getLogger(__name__.split(".")[-1])

archive_path = paths.data_path / "archive"
lc_archive_path = archive_path / "light_curves"

alert_df_path = archive_path / "alert_archive.csv"
alert_df = pd.read_csv(alert_df_path)
alert_df.sort_values("jd", inplace=True)

alert_df.query("magpsf < 18.5", inplace=True)
alert_df.reset_index(inplace=True, drop=True)

lc_summary_df_path = archive_path / "light_curve_summaries.csv"
lc_summary_df = pd.read_csv(lc_summary_df_path)
lc_summary_df.set_index("objectId", inplace=True, verify_integrity=True)

print(alert_df["jd"].min(), alert_df.iloc[0]["jd"])
palomar_midday_frac = 0.29167 # This fraction is 19:00:00 UT

alert_jd = alert_df["jd"].min()
less_than_frac = (alert_jd % 1) < palomar_midday_frac
start_time = Time(int(alert_jd) + int(less_than_frac) + palomar_midday_frac, format="jd") - 1.0

interval = 12. / 24. # TimeDelta(1 * u.day)

interval_grid = start_time.jd + np.arange(0, 180 + interval, interval) + 0.01
day_grid = start_time.jd + np.arange(0, interval_grid.max() + 2., 1.)

# keep track...
n_alert_per_day = []
new_targets_per_day = []
removed_targets_per_day = []
jd_vals = []

# things to be reset at the end of each ""day""
alerts_today = 0
yesterdays_target_list = set()

selector = TargetSelector.from_config()

useful_lightcurves = {}

rfig, rax = plt.subplots()

print(alert_df.index.values)

rank_histories = {}
first_alert = {}

next_day = day_grid[1]


for jj, (low_interval, high_interval) in enumerate(zip(interval_grid[:-1], interval_grid[1:])):

    new_alerts = alert_df.query("(@low_interval <= jd) & (jd < @high_interval)")
    t_ref = Time(high_interval, format="jd")

    print(low_interval, high_interval)

    if new_alerts is not None:
        for ii, alert in new_alerts.iterrows():
            objectId = alert["objectId"]
            alert_jd = alert["jd"]
            alert_time = Time(alert_jd, format="jd")

            alerts_today = alerts_today + 1

            if objectId not in lc_summary_df.index.values:
                print(objectId, " not in index!!!" )
                continue

            object_summary = lc_summary_df.loc[objectId]
            if object_summary["timespan"] > 60:
                continue
            if object_summary["N_obs_1"] < 2 or object_summary["N_obs_2"] < 2:
                continue

            target = selector.target_lookup.get(objectId, None)

            if target is None:
                lc_path = lc_archive_path / f"{objectId}.csv"
                full_lc = pd.read_csv(lc_path)
                useful_lightcurves[objectId] = full_lc

                target_history = full_lc.query("jd <= @alert_jd")
                assert np.isclose(target_history["jd"].max(), alert_jd)
                target = Target.from_target_history(objectId, target_history)
                selector.add_target(target)

            else:
                full_lc = useful_lightcurves[objectId]
                max_current_jd = target.target_history["jd"].max()
                new_data = full_lc.query(f"(@max_current_jd < jd) & (jd <={alert_jd+1e-5})")
                target.update_target_history(new_data)

    #selector.model_targets(default_sncosmo_model)
    selector.evaluate_all_targets(default_score, t_ref=t_ref)
    selector.remove_bad_targets()
    logger.info(f"{len(selector.target_lookup)} objects after removing bad targets")
    ranked_list = selector.build_ranked_target_list(
        observatory=None, plots=False, save_list=False, t_ref=t_ref
    )

    if ranked_list is not None:
        for rank, (ii, row) in enumerate(ranked_list.iterrows(), 1):
            row_objId = row["objectId"]
            if row_objId not in rank_histories:
                rank_histories[row_objId] = []
            rank_histories[row_objId].append((rank, t_ref.jd))

    if high_interval > next_day:
        current_day = next_day # ie, that's now.
        jd_vals.append(current_day)

        n_alert_per_day.append(alerts_today)

        current_target_list = set(selector.target_lookup.keys())
        
        new_targets_today = current_target_list - yesterdays_target_list
        logger.info(f"{len(new_targets_today)} new targets today ({alerts_today} alerts today)")
        new_targets_per_day.append(len(new_targets_today))

        removed_targets_today = yesterdays_target_list - current_target_list
        logger.info(f"{len(removed_targets_today)} removed targets today")
        removed_targets_per_day.append(len(removed_targets_today))

        next_day = day_grid[ day_grid > high_interval ][0]

        alerts_today = 0
        yesterdays_target_list = current_target_list
        logger.info(f"end of day {current_day - start_time.jd}")


print(ranked_list)


#for objectId, target in selector.target_lookup.items():
#    rank_history = target.rank_history["no_observatory"]
#    if len(rank_history) > 0:
#        rank_df = pd.DataFrame(rank_history, columns="rank date".split())
#        #rank_df["jd"] = rank_df.apply(lambda x: x.jd)
#        rank_histories[objectId] = rank_df


first_alerts = alert_df.groupby("objectId")["jd"].min()
alert_mapper = {objId: x for objId, x in first_alerts.items()}

cmapper = cm.ScalarMappable(cmap='viridis_r') # initialise object
cmapper.set_clim(vmin=np.min(first_alerts),vmax=np.max(first_alerts)) #set normalisation

for objId, rh in rank_histories.items():
    rank_df = pd.DataFrame(rh, columns="rank date".split())
    c = cmapper.to_rgba(alert_mapper[objId])
    rax.plot(rank_df["date"], rank_df["rank"], c=c)
    


ticks = rax.get_xticks()
labels = [Time(x, format="jd").strftime("%Y-%m-%d") for x in ticks]
rax.set_xticks(ticks)
rax.set_xticklabels(labels, rotation=45)
rax.set_ylim(25, 0)

rfig.tight_layout()

jd_vals = np.array(jd_vals)
n_alert_per_day = np.array(n_alert_per_day)
new_targets_per_day = np.array(new_targets_per_day)
removed_targets_per_day = np.array(removed_targets_per_day)

print(len(n_alert_per_day), len(new_targets_per_day))

new_frac = new_targets_per_day / n_alert_per_day * 100
new_frac[ n_alert_per_day==0 ] = 0
removed_frac = removed_targets_per_day

ave = np.average(new_frac[ n_alert_per_day > 0 ])

print(f"select an average of {ave} new targets per 100 alerts")

fig, ax = plt.subplots()
ax.bar(jd_vals, new_frac)
ax.scatter(jd_vals[ new_frac == 0 ], np.full(sum(new_frac == 0), 1.), marker="x", s=40, color="k")
ax.set_ylabel("New targets per 100 alerts")
ticks = ax.get_xticks()
labels = [Time(x, format="jd").strftime("Y-%m-%d") for x in ticks]
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

mod_frac = new_frac.copy()
mod_frac[ n_alert_per_day == 0 ] = -2

mod_bins = np.arange(-2.5, 12.5, 1.0)

mod_hist, _ = np.histogram(mod_frac, bins=mod_bins)
print(mod_hist)

mids = 0.5 * (mod_bins[:-1] + mod_bins[1:])
fig, ax = plt.subplots()
ax.bar(mids, mod_hist, width=1.0)
ticks = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels = ["no alerts", ""] + ticks[2:]
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.axvline(ave, color="k", ls="--", lw=2)

ax.set_ylabel("Frequency", fontsize=14)
ax.set_xlabel(f"Number of new targets per night (per 100 alerts)", fontsize=14)

plt.show()