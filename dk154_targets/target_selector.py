import copy
import io
import json
import logging
import os
import pickle
import shutil
import time
import yaml
from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from dk154_targets.modelling import default_sncosmo_model
from dk154_targets.queries import FinkQuery
from dk154_targets.fink_query_manager import FinkQueryManager
#from AtlasQueryManager, FinkQueryManager
from dk154_targets.scoring import default_score
from dk154_targets.target import Target
from dk154_targets.utils import chunk_list, readstamp
from dk154_targets.visibility_forecast import VisibilityForecast

from dk154_targets import paths


logger = logging.getLogger(__name__.split(".")[-1])

default_num_alerts = 5

class TargetSelector:

    default_num_alerts = 5
    default_timeout = 10
    default_sleeptime = 60

    default_selector_config_path = paths.config_path / "selector_config.yaml"

    default_full_target_history_path = paths.base_path / "full_target_history.csv"
    default_selector_pickle_path = paths.base_path / "latest_selector.pkl"

    def __init__(self, selector_config):

        ###=========== config ===========###
        self.selector_config = selector_config or {}

        ###======== target "storage" ========###
        self.target_lookup = {}

        self.query_managers = {}
        self.query_mananger_config = self.selector_config.get("query_managers", {})
        self.initialise_query_managers()

        ###========= set observatories ========###
        self.observatory_config = self.selector_config.get("observatories", {})
        self.observatories = [None]
        self.initialise_observatories()

        ###========== modelling details =======###
        self.lazy_modelling = self.selector_config.get("lazy_modelling", True)
        assert isinstance(self.lazy_modelling, bool)

        ###======== target of opportunity =====###
        self.targets_of_opportunity_path = Path(
            self.selector_config.get(
                "targets_of_opportunity_path", 
                paths.base_path / "targets_of_opportunity"
            )
        ).absolute()

        ###========== telegram_messenger ==========###
        # todo add telegram messenger??

        ###=============== plotting ===============###
        plot_dir = self.selector_config.get("plotting_dir", None)
        if plot_dir is None:
            self.plot_dir = paths.base_path / "plots"
        else:
            self.plot_dir = Path(plot_dir).absolute()


    @classmethod
    def from_config(cls, config_file=None):
        config_file = config_file or cls.default_selector_config_path
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        selector = cls(config)
        return selector


    def initialise_query_managers(self,):
        fink_config = self.query_mananger_config.get("fink", None)
        #if fink_config is not None:
        #    self.query_managers["fink"] = FinkQueryManager(fink_config, self.target_lookup)
        if fink_config is None:
            raise ValueError("fink_config shouldn't be None!")
        self.fink_query_manager = FinkQueryManager(fink_config, self.target_lookup)

        #atlas_config = self.query_mananger_config.get("atlas", None)
        #if atlas_config is not None:
        #    self.query_managers["atlas"] = AtlasQueryManager(atlas_config, self.target_lookup)


    def initialise_observatories(self,):
        for obs_name, obs_id in self.observatory_config.items():
            if isinstance(obs_id, str):
                observatory = EarthLocation.of_site(obs_id)
            else:
                observatory = EarthLocation(**obs_id)
            logger.info(f"initalise obs {obs_name}")
            observatory.name = obs_name
            self.observatories.append(observatory)


    def add_target(self, target: Target):
        objectId = target.objectId
        if objectId in self.target_lookup:
            raise ValueError(f"{objectId} already in target_lookup")
        self.target_lookup[target.objectId] = target


    def perform_query_manager_tasks(self):
        logger.info("perform all queries")
        #for name, query_manager in self.query_managers.items():
        #    query_manager.perform_all_tasks()
        self.fink_query_manager.perform_all_tasks()


    def evaluate_all_targets(
        self, scoring_function: Callable, observatory: EarthLocation = None
    ):
        for objectId, target in self.target_lookup.items():
            target.evaluate_target(scoring_function, observatory)


    def remove_bad_targets(self,):
        to_remove = []
        for objectId, target in self.target_lookup.items():
            raw_score = target.get_last_score("no_observatory")
            if not np.isfinite(raw_score):
                to_remove.append(objectId)
        for objectId in to_remove:
            target = self.target_lookup.pop(objectId)
            logger.info(f"rm {objectId}")
            assert objectId not in self.target_lookup
        if len(to_remove) > 0:
            logger.info(f"remove {len(to_remove)} targets")
        return to_remove
        

    def model_targets(self, modelling_function: Callable):
        if modelling_function is None:
            return None
        logger.info("model all targets")
        modelled = []
        for objectId, target in self.target_lookup.items():
            if not target.updated and self.lazy_modelling:
                continue
            target.model_target(modelling_function)
            target.updated = False
        

    def check_for_targets_of_opportunity(self,):
        logger.info("look for targets of opportunity")
        t_of_opp_files = list(self.targets_of_opportunity_path.glob("*.yaml"))
        targets_of_opportunity = []
        for t_of_opp_file in t_of_opp_files:
            with open(t_of_opp_file, "r") as f:
                target_config = yaml.load(f, Loader=yaml.FullLoader)
                objectId = target_config.get("objectId", None)
                if objectId is None:
                    msg = (
                        f"New target of opportunity in file {t_of_opp_file.name}"
                        "could not be added: there is no `objectId`!"
                    )
                    target = None
                else:
                    target = Target.from_fink_query(objectId)
                    ra = target_config.get("ra", None)
                    dec = target_config.get("dec", None)
                    if target is None:
                        if (ra is None) or (dec is None):
                            msg = (
                                f"New target of opportunity {objectId} could not be added:"
                                "fink_query failed, and no ra/dec provided."
                            )
                            target = None
                        else:
                            target = Target(objectId, ra, dec)
                            msg = f"Target {objectId} added but no fink data found."
                    else:
                        msg = f"New target of opportunity {objectId} added with fink data!"
                    if target is not None:
                        target.target_of_opportunity = True
                        self.target_lookup[objectId] = target
                        targets_of_opportunity.append(objectId)
                #TODO send message
                #TODO delete ToO file!
        if len(targets_of_opportunity) > 0:
            logger.info(f"add {len(targets_of_opportunity)} targets of opportunity")


    def build_ranked_target_list(self, observatory, plot_figures=True):
        obs_name = getattr(observatory, "name", "no_observatory")
        if obs_name == "no_observatory":
            assert observatory is None

        data_list = []
        for objectId, target in self.target_lookup.items():
            score = target.get_last_score(obs_name)
            if score < 0:
                target.rank_history[obs_name]
                continue
            target_data = dict(
                objectId=objectId, ra=target.ra, dec=target.dec, score=score
            )
            data_list.append(target_data)
        target_list = pd.DataFrame(data_list)
        target_list.sort_values("score", inplace=True)
        target_list.reset_index(inplace=True)

        if plot_figures:
            obs_plot_dir = self.plot_dir / obs_name
            obs_plot_dir.mkdir(exist_ok=True, parents=True)
            ext = ".png"
            existing_fig_list = obs_plot_dir.glob(f"*{ext}")
            for fig_path in existing_fig_list:
                assert fig_path.suffix == ext
                fig_path.unlink()
                assert not fig_path.exists()
            for ii, row in target_list.iterrows():
                objectId = row["objectId"]
                target = self.target_lookup[objectId]
                lc_fig = target.plot_lightcurve()
                lc_fig_path = obs_plot_dir / f"{ii:04d}_{objectId}_lc{ext}"
                lc_fig.savefig(lc_fig_path)
                plt.close(lc_fig)
                if observatory is None:
                    continue
                oc_fig = target.plot_observing_chart(observatory)
                oc_fig_path = obs_plot_dir / f"{ii:04d}_{objectId}_oc{ext}"
                oc_fig.savefig(oc_fig_path)
                plt.close(oc_fig)

        

    def start(self, scoring_function: Callable, modelling_function: Callable):
        self.perform_query_manager_tasks()
        self.check_for_targets_of_opportunity()
        self.model_targets(modelling_function)
        for observatory in self.observatories:
            self.evaluate_all_targets(scoring_function, observatory=observatory)
        logger.info(f"{len(self.target_lookup)} targets before removing bad targets")
        self.remove_bad_targets()
        for observatory in self.observatories:
            self.build_ranked_target_list(observatory)

        sleep_time = self.selector_config.get("sleep_time", 5.)
        logger.info(f"sleep for {sleep_time} sec")
        time.sleep(sleep_time)