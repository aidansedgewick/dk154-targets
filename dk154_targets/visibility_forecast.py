import logging

import numpy as np

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.coordinates import get_moon, get_sun
from astropy.table import Table
from astropy.time import Time

from astroplan import Observer


logger = logging.getLogger(__name__.split(".")[-1])

def plot_observing_chart(observer: Observer, target: "Target"=None, t_ref=None):
    t_ref = t_ref or Time.now()

    fig, ax = plt.subplots()

    time_grid = t_ref + np.linspace(0, 24, 24 * 4) * u.hour

    timestamps = np.array([x.mjd for x in time_grid])

    moon_altaz = observer.moon_altaz(time_grid)
    sun_altaz = observer.sun_altaz(time_grid)

    print(moon_altaz.alt)

    civil_night = observer.tonight(horizon=0*u.deg)
    astro_night = observer.tonight(horizon=-18*u.deg)


    ax.fill_between( # civil night
        timestamps, -90*u.deg, 90*u.deg, (sun_altaz.alt < 0*u.deg), color="0.9", 
    )
    ax.fill_between( # civil night
        timestamps, -90*u.deg, 90*u.deg, (sun_altaz.alt < -6*u.deg), color="0.7", 
    )
    ax.fill_between( # civil night
        timestamps, -90*u.deg, 90*u.deg, (sun_altaz.alt < -12*u.deg), color="0.4", 
    )
    ax.fill_between( # astronomical night
        timestamps, -90*u.deg, 90*u.deg, sun_altaz.alt < -18*u.deg, color="0.3", 
    )


    ax.plot(timestamps, moon_altaz.alt.deg, color="0.5", ls="--", label="moon")
    ax.plot(timestamps, sun_altaz.alt.deg, color="0.5", ls=":", label="sun")
    ax.set_ylim(0, 90)
    ax.set_ylabel("Altitude [deg]", fontsize=16)


    if target is not None:
        target_altz = observer.altaz(time_grid, target.coord)
        ax.plot(timestamps, target_altaz.alt.deg, color="b", label="target")

        if all(target_altaz.alt < 30*u.deg):
            ax.text(
                0.5, 0.5, f"target alt never >30 deg", color="red", rotation=45,
                ha="center", va="center", transform=ax.transAxes, fontsize=18
            )

    #obs_name = getattr(vf.observatory.info, "name", vf.obs_str) or vf.obs_str
    obs_name = observer.name
    title = f"Observing from {obs_name}"
    title = title + f"\n starting at {t_ref.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    ax.text(
        0.5, 1.0, title, fontsize=14,
        ha="center", va="bottom", transform=ax.transAxes
    )

    iv = 3 # tick marker interval.
    fiv = 24 / iv # interval fraction of day.

    xticks = round(timestamps[0] * fiv, 0) / fiv + np.arange(0, 1, 1. / fiv)
    hourmarks = [Time(x, format="mjd").datetime for x in xticks]
    xticklabels = [hm.strftime("%H:%M") for hm in hourmarks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_xlim(timestamps[0], timestamps[-1])

    if target is not None:
        ax2 = ax.twinx()
        mask = target_altaz.alt > 10. * u.deg
        airmass_time = timestamps[ mask ]
        airmass = 1. / np.cos(target_altaz.zen[ mask ]).value
        ax2.plot(airmass_time, airmass, color="red")
        ax2.set_ylim(1.0, 4.0)
        ax2.set_ylabel("Airmass", color="red", fontsize=14)
        ax2.tick_params(axis='y', colors='red')
        ax2.set_xlim(ax.get_xlim())

    ax.legend()

    return fig
