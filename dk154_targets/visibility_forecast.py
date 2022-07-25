import logging

import numpy as np

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.coordinates import get_moon, get_sun
from astropy.table import Table
from astropy.time import Time


logger = logging.getLogger(__name__.split(".")[-1])

class VisibilityForecast:
    def __init__(self, target: "Target", observatory: EarthLocation, t0: Time=None):
        # target typehint is a str here to avoid circular import.
        self.target = target
        self.observatory = observatory
        self.obs_str = self.get_observatory_location_str()
        self.t0 = t0 or Time.now()
        self.update(t0=self.t0)


    def update(self, t0: Time):
        """
        Recompute the target position, distance to sun and moon for 24hrs after t0.
        """
        self.t0 = t0

        self.time_grid = t0 + np.linspace(0, 24, 24 * 4) * u.hour


        altaz_transform = AltAz(obstime=self.time_grid, location=self.observatory)

        moon_pos = get_moon(self.time_grid)
        self.moon_altaz = moon_pos.transform_to(altaz_transform)

        sun_pos = get_sun(self.time_grid)
        self.sun_altaz = sun_pos.transform_to(altaz_transform)

        if self.target is not None:
            self.target_altaz = self.target.coord.transform_to(altaz_transform)
            self.moon_dist = self.target.coord.separation(SkyCoord(ra=moon_pos.ra, dec=moon_pos.dec))
            self.sun_dist = self.target.coord.separation(SkyCoord(ra=sun_pos.ra, dec=sun_pos.dec))


    def immediate_night(self, t_ref: Time=None):
        t_ref = t_ref or Time.now()

        if t_ref > self.time_grid[-1]:
            self.update(t0=t_ref)

        obs_night_mask = self.sun_altaz.alt < -18.0 * u.deg # when is the sun down?
        interval = (self.time_grid[1:] - self.time_grid[:-1])[0]
        night_grid = self.time_grid[ obs_night_mask ] # Time values during the night.
        # Are we near a timegrid point? Do this instead of night[0] < t < night[-1]
        is_currently_night = any(abs(night_grid - t_ref) < 1.1 * interval)
        #if not is_currently_night:
        #    logger.info(f"not dark at obs {self.observatory.info.name}")
        t_ref = t_ref if is_currently_night else night_grid[0]
        return t_ref

    def get_immediate_altitude(self, t_ref: Time=None):
        t_ref = t_ref or Time.now()
        immediate_night = self.get_immediate_night()
        return self.target.
        

    def target_validity_check(self,):
        if self.target is None:
            raise ValueError("Target is None!")


    def ever_visible(self, min_alt=20.*u.deg):
        self.check_forecast_validity()
        return self.target_altaz.alt.max() > min_alt


    def plot_observing_chart(self,):
        return plot_observing_chart(self)


    def check_forecast_validity(self,):
        # TODO: check is before next night.
        if self.time_grid.max() < Time.now() + 12. * u.hour:
            self.update(Time.now())


    def get_max_alt(self,):
        self.target_validity_check()
        self.check_forecast_validity()
        max_idx = np.argmax(self.target_altaz.deg)
        return self.time_grid[ max_idx ], self.target_altaz[ max_idx ].deg


    def get_observatory_location_str(self,):  
        obs_lat = self.observatory.lat.signed_dms
        obs_lon = self.observatory.lon.signed_dms
        lat_str = f"{round(obs_lat.d)} {round(obs_lat.m)} {round(obs_lat.s)}"
        lat_card = "W" if obs_lat.sign < 0 else "E"
        lon_str = f"{round(obs_lon.d)} {round(obs_lon.m)} {round(obs_lon.s)}"
        lon_card = "S" if obs_lon.sign < 0 else "N"
        return f"{lat_str} {lat_card} {lon_str} {lon_card}"


def plot_observing_chart(vf: VisibilityForecast):
    fig, ax = plt.subplots()

    timestamps = np.array([x.mjd for x in vf.time_grid])

    ax.fill_between(
        timestamps, -90*u.deg, 90*u.deg, vf.sun_altaz.alt < 0*u.deg, color="0.7", 
    )
    ax.fill_between(
        timestamps, -90*u.deg, 90*u.deg, vf.sun_altaz.alt < -18*u.deg, color="0.2", 
    )

    if vf.target is not None:
        ax.plot(timestamps, vf.target_altaz.alt.deg, color="b", label="target")
    ax.plot(timestamps, vf.moon_altaz.alt.deg, color="0.4", ls="--", label="moon")
    ax.plot(timestamps, vf.sun_altaz.alt.deg, color="0.4", ls=":", label="sun")
    ax.set_ylim(0, 90)
    ax.set_ylabel("Altitude [deg]", fontsize=16)

    if all(vf.target_altaz.alt < 30*u.deg):
        ax.text(
            0.5, 0.5, f"target alt never >30 deg", color="red", rotation=45,
            ha="center", va="center", transform=ax.transAxes, fontsize=18
        )

    obs_name = getattr(vf.observatory.info, "name", vf.obs_str) or vf.obs_str
    title = f"Observing from {obs_name}"
    title = title + f"\n starting at {vf.t0.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    ax.text(
        0.5, 1.0, title, fontsize=14,
        ha="center", va="bottom", transform=ax.transAxes
    )

    iv = 3
    fiv = 24 / iv

    xticks = round(timestamps[0] * fiv, 0) / fiv + np.arange(0, 1, 1. / fiv)
    hourmarks = [Time(x, format="mjd").datetime for x in xticks]
    xticklabels = [hm.strftime("%H:%M") for hm in hourmarks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_xlim(timestamps[0], timestamps[-1])


    ax2 = ax.twinx()

    if vf.target is not None:
        mask = vf.target_altaz.alt > 10. * u.deg
        airmass_time = timestamps[ mask ]
        airmass = 1. / np.cos(vf.target_altaz.zen[ mask ]).value
        
        ax2.plot(airmass_time, airmass, color="red")
        ax2.set_ylim(1.0, 4.0)
        ax2.set_ylabel("Airmass", color="red", fontsize=14)
        ax2.tick_params(axis='y', colors='red')
        ax2.set_xlim(ax.get_xlim())

    ax.legend()

    return fig
