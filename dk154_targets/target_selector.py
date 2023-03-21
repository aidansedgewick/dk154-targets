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

from astroplan import Observer



from dk154_targets.modelling import default_sncosmo_model
from dk154_targets.queries import FinkQuery
from dk154_targets.query_managers import (
    FinkQueryManager, AlerceQueryManager, AtlasQueryManager, TNSQueryManager, GenericQueryManager
)
from dk154_targets.scoring import default_score
from dk154_targets.target import Target
from dk154_targets.telegram_bot import TelegramManager
from dk154_targets.slack_bot import SlackManager
from dk154_targets.utils import chunk_list, readstamp

from dk154_targets import paths


logger = logging.getLogger(__name__.split(".")[-1])

default_num_alerts = 5

class TargetSelector:

    default_num_alerts = 5
    default_timeout = 10
    default_sleeptime = 60

    default_selector_config_path = paths.config_path / "selector_config.yaml"

    default_objectId_list_path = paths.outputs_path / "latest_objectIds.csv"
    default_selector_pickle_path = paths.outputs_path / "latest_selector.pkl"

    default_targets_of_opportunity_path = paths.targets_of_opportunity_path

    default_ranked_target_lists_path = paths.ranked_target_lists_path

    default_unranked_value = 9999

    def __init__(self, selector_config):

        ###=========== config ===========###
        self.selector_config = selector_config or {}

        ###======== target "storage" ========###
        self.target_lookup = {}

        ###======= ranked target lists ========###
        ranked_list_path = self.selector_config.get("ranked_target_lists_path", None)
        if ranked_list_path is None:
            logger.warn("no 'ranked_target_lists_path' in config.")
            logger.warn(f"default to {self.default_ranked_target_lists_path}")
            self.ranked_target_lists_path = self.default_ranked_target_lists_path
        else:
            self.ranked_target_lists_path = Path(ranked_list_path).absolute()
        self.ranked_target_lists_path.mkdir(exist_ok=True, parents=True)

        ###======== query managers ========###
        self.query_managers = {}
        self.query_mananger_config = self.selector_config.get("query_managers", {})
        self.initialise_query_managers()

        ###========= set observatories ========###
        self.observatory_config = self.selector_config.get("observatories", {})
        self.observatories = [None]
        self.initialise_observatories()

        ###========== modelling details =======###
        modelling_config = self.selector_config.get("modelling", None)
        if modelling_config is None:
            logger.warning("no 'modelling' in config!")
        self.modelling_config = modelling_config or {}

        self.lazy_modelling = self.modelling_config.get("lazy_modelling", True)
        assert isinstance(self.lazy_modelling, bool)

        ###======== target of opportunity =====###
        topp_path = self.selector_config.get("targets_of_opportunity_path", None)
        if topp_path is None:
            logger.warn("no 'targets_of_opportunity_path' in config.")
            logger.warn(f"default to {self.default_targets_of_opportunity_path}")
            self.targets_of_opportunity_path = self.default_targets_of_opportunity_path
        else:
            self.targets_of_opportunity_path = Path(topp_path).absolute()
        self.targets_of_opportunity_path.mkdir(exist_ok=True, parents=True)


        ###========== telegram_messenger ==========###
        self.initialise_bots()

        ###=============== plotting ===============###
        plot_dir = self.selector_config.get("plotting_dir", None)
        if plot_dir is None:
            self.plot_dir = paths.outputs_path / "plots"
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
        """
        If you add new query_managers...
        remember to call the tasks of your query manager in perform_query_manager_tasks()
        """

        ### FINK ###
        fink_config = self.query_mananger_config.get("fink", None)
        if fink_config is None:
            logger.warning("no fink_config! fink_query_manager=None")
            self.fink_query_manager = None
        else:
            logger.info("init FINK query manager...")
            self.fink_query_manager = FinkQueryManager(fink_config, self.target_lookup)
        self.query_managers["fink"] = self.fink_query_manager

        # ### ALERCE ###
        alerce_config = self.query_mananger_config.get("alerce", None)
        if alerce_config is None:
            logger.warning("no alerce_config! alerce_query_manager=None")
            self.alerce_query_manager = None
        else:
            logger.info("init ALERCE query manager...")
            self.alerce_query_manager = AlerceQueryManager(alerce_config, self.target_lookup)
        self.query_managers["alerce"] = self.alerce_query_manager

        ### ATLAS ###
        atlas_config = self.query_mananger_config.get("atlas", None)
        if atlas_config is None:
            logger.info("no atlas config. atlas_query_manager=None")
            self.atlas_query_manager = None
        else:
            logger.info("init ATLAS query manager...")
            self.atlas_query_manager = AtlasQueryManager(atlas_config, self.target_lookup)
        self.query_managers["atlas"] = self.atlas_query_manager

        ### TNS ###
        tns_config = self.query_mananger_config.get("tns", None)
        if tns_config is None:
            logger.info("no tns config. tns_query_manager=None")
            self.tns_query_manager = None
        else:
            logger.info("init TNS query manager...")
            self.tns_query_manager = TNSQueryManager(tns_config, self.target_lookup)

        self.query_managers["tns"] = self.tns_query_manager


        ##########################################################
        ## Remember to update perform_query_manager_tasks() !!! ##
        ##########################################################


    def initialise_observatories(self,):
        for obs_name, obs_id in self.observatory_config.items():
            if isinstance(obs_id, str):
                #observatory = EarthLocation.of_site(obs_id)
                location = EarthLocation.of_site(obs_id)
            else:
                #observatory = EarthLocation(**obs_id)
                location = EarthLocation(**obs_id)
            observatory = Observer(location=location, name=obs_name)
            logger.info(f"initalise obs {observatory.name}")
            #observatory.name = obs_name
            self.observatories.append(observatory)
        logger.info(f"{len(self.observatories)} observatories")
        logger.info(f"    including `no_observatory`")


    def initialise_bots(self,):
        self.initialise_telegram_bot()
        self.initialise_slack_bot()


    def initialise_telegram_bot(self,):
        telegram_config = self.selector_config.get("telegram", None)
        self.telegram_manager = TelegramManager.from_config(telegram_config)

    
    def initialise_slack_bot(self,):
        slack_config = self.selector_config.get("slack", None)
        self.slack_manager = SlackManager.from_config(slack_config)


    def add_target(self, target: Target):
        objectId = target.objectId
        if objectId in self.target_lookup:
            raise ValueError(f"{objectId} already in target_lookup")
        self.target_lookup[target.objectId] = target


    #def add_targets_from_objectId_list(self, objectId_list):


    def perform_query_manager_tasks(self, simulated_alerts=False, dump_alerts=True, t_ref=None):
        t_ref = t_ref

        logger.info("perform all query manager tasks")

        # if self.fink_query_manager is None:
        #     logger.warning("no fink query manager!")
        # else:
        #     self.fink_query_manager.perform_all_tasks(
        #         simulated_alerts=simulated_alerts, dump_alerts=dump_alerts
        #     )
        # if self.alerce_query_manager is None:
        #     logger.warning("no ALERCE query manager")
        # else:
        #     self.alerce_query_manager.perform_all_tasks()
        # if self.atlas_query_manager is None:
        #     pass
        # else:
        #     self.atlas_query_manager.perform_all_tasks()
        # if self.tns_query_manager is None:
        #     pass
        # else:
        #     self.tns_query_manager.perform_all_tasks()

        for name, query_manager in self.query_managers.items():
            if query_manager is None:
                logger.info(f"no {name} query_manager")
            else:
                query_manager.perform_all_tasks(t_ref=t_ref)


    def compile_all_target_histories(self, t_ref=None):
        t_ref = t_ref or Time.now()
        logger.info("compile photometric data...")
        for objectId, target in self.target_lookup.items():
            target.compile_target_history(t_ref=t_ref)
        logger.info("done!")


    def initial_new_target_check(self, scoring_function: Callable, t_ref: Time=None):
        t_ref = t_ref or Time.now()
        to_remove = []
        new_targets = []
        for objectId, target in self.target_lookup.items():
            last_score = target.get_last_score("no_observatory")
            if last_score is not None:
                continue
            target.evaluate_target(scoring_function, None, t_ref=t_ref)
            new_targets.append(objectId)
            initial_score = target.get_last_score("no_observatory")
        return new_targets


    def evaluate_all_targets(
        self, 
        scoring_function: Callable, 
        observatory: Observer=None, 
        t_ref: Time=None, 
        **kwargs
    ):
        obs_name = getattr(observatory, "name", "no_observatory")
        if obs_name == "no_observatory":
            assert observatory is None
        t_ref = t_ref or Time.now()
        if not isinstance(t_ref, Time):
            raise ValueError(f"t_ref should be astropy.time.Time, not {type(t_ref)}")

        if "t_ref" not in kwargs:
            kwargs["t_ref"] = t_ref
        if observatory is not None:
            kwargs["tonight"] = observatory.tonight(time=t_ref)

        logger.info(f"eval targets for {obs_name}")
        for objectId, target in self.target_lookup.items():
            target.evaluate_target(scoring_function, observatory, **kwargs)
            assert obs_name in target.score_history


    def remove_bad_targets(self, verbose=True):
        to_remove = []
        for objectId, target in self.target_lookup.items():
            raw_score = target.get_last_score("no_observatory")
            if target.target_of_opportunity:
                logger.info(f"{objectId} is Opp target - keep!")
            if not np.isfinite(raw_score):
                to_remove.append(objectId)
        removed_targets = []
        for objectId in to_remove:
            target = self.target_lookup.pop(objectId)
            if verbose:
                logger.info(f"rm {objectId}")
                print(target.reject_comments)
            removed_targets.append(target)
            assert objectId not in self.target_lookup
        if len(to_remove) > 0:
            logger.info(f"remove {len(to_remove)} targets")
            assert len(removed_targets) == len(to_remove)
        return removed_targets
        

    def model_targets(
        self, modelling_function: Callable, lazy_modelling: bool=True
    ):
        if modelling_function is None:
            logger.warn("modelling function is None!")
            return None
        logger.info(f"model {len(self.target_lookup)} targets")
        modelled = []
        for ii, (objectId, target) in enumerate(self.target_lookup.items()):
            if not target.updated and lazy_modelling:
                logger.debug(f"{objectId} no updates: skip")
                continue
                
            logger.info(f"modelling {ii+1} of {len(self.target_lookup)}")
            model = modelling_function(target, **self.modelling_config)
            if model is not None:
                target.models.append(model)
            #target.model_target(modelling_function)

        
    def check_for_targets_of_opportunity(self, try_fink=True):
        logger.info("look for targets of opportunity")
        opp_target_path_list = list(self.targets_of_opportunity_path.glob("*.yaml"))
        targets_of_opportunity = []
        logger.info(f"found {len(opp_target_path_list)} opp target yamls.")
        for opp_target_path in opp_target_path_list:
            with open(opp_target_path, "r") as f:
                target_config = yaml.load(f, Loader=yaml.FullLoader)
                objectId = target_config.get("objectId", None)
                base_score = target_config.get("base_score", Target.default_base_score)
                if objectId is None:
                    msg = (
                        f"New target of opportunity in file {opp_target_path.name}"
                        "could not be added: there is no `objectId`!"
                    )
                    target = None
                else:
                    if self.fink_query_manager is not None:
                        target = Target.from_fink_query(objectId, base_score=base_score)
                    else:
                        target = None
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
                            target = Target(objectId, ra, dec, base_score=base_score)
                            msg = f"Target {objectId} added but no fink data found."
                    else:
                        msg = f"New target of opportunity {objectId} added with fink data!"
                    if target is not None:
                        target.target_of_opportunity = True
                        self.target_lookup[objectId] = target
                        targets_of_opportunity.append(objectId)
                #TODO send message
                os.remove(opp_target_path)
                assert not opp_target_path.exists()
        if len(targets_of_opportunity) > 0:
            logger.info(f"add {len(targets_of_opportunity)} targets of opportunity")


    def reset_updated_targets(self,):
        for objectId, target in self.target_lookup.items():
            target.updated = False


    def reset_all_target_figures(self,):
        for objectId, target in self.target_lookup.items():
            target.reset_target_figures()


    def build_ranked_target_list(
        self, observatory=None, plots=True, save_list=True, output_dir=None, t_ref: Time=None
    ):
        obs_name = getattr(observatory, "name", "no_observatory")
        if obs_name == "no_observatory":
            assert observatory is None

        logger.info(f"build ranked list for {obs_name}")

        t_ref = t_ref or Time.now()
        if not isinstance(t_ref, Time):
            raise ValueError(f"t_ref should be astropy.time.Time, not {type(t_ref)}")

        if output_dir is None:
            output_dir = self.ranked_target_lists_path #/ obs_name
        output_dir.mkdir(exist_ok=True, parents=True)

        data_list = []
        for objectId, target in self.target_lookup.items():
            if obs_name not in target.rank_history:
                target.rank_history[obs_name] = []
            score = target.get_last_score(obs_name)
            if score < 0 or not np.isfinite(score):
                target.rank_history[obs_name].append((self.default_unranked_value, t_ref.jd))
                continue
            target_data = dict(
                objectId=objectId, ra=target.ra, dec=target.dec, score=score
            )
            data_list.append(target_data)

        if len(data_list) == 0:
            logger.info(f"No targets for {obs_name}!")
            return 
        target_list = pd.DataFrame(data_list)
        target_list.sort_values("score", inplace=True, ascending=False)
        target_list.reset_index(inplace=True, drop=True)

        for ii, row in target_list.iterrows():
            target = self.target_lookup[row.objectId]
            ranking = ii + 1
            target.rank_history[obs_name].append((ranking, t_ref.jd))

        if save_list:
            target_list_path = output_dir / f"{obs_name}_ranked_list.csv"
            target_list.to_csv(target_list_path, index=True)

        if plots:
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
                if target.latest_lc_fig is None:
                    lc_fig = target.plot_lightcurve(t_ref=t_ref)
                else:
                    lc_fig = target.latest_lc_fig
                if lc_fig is not None:
                    lc_fig_path = obs_plot_dir / f"{ii:04d}_{objectId}_lc{ext}"
                    lc_fig.savefig(lc_fig_path)
                    plt.close(lc_fig)
                if observatory is None:
                    continue
                try:
                    oc_fig = target.plot_observing_chart(observatory, t_ref=t_ref)
                except Exception as e:
                    oc_fig = None
                    logger.warning(e)
                if oc_fig is not None:
                    oc_fig_path = obs_plot_dir / f"{ii:04d}_{objectId}_oc{ext}"
                    oc_fig.savefig(oc_fig_path)
                    plt.close(oc_fig)

        return target_list

    def send_target_updates(self,):
        logger.info("starting updates")
        for objectId, target in self.target_lookup.items():
            if not target.updated:
                continue
            description = target.get_description()
            if self.slack_manager is not None:                
                scratch_lc_path = paths.scratch_path / f"{objectId}_lc.png"
                if target.latest_lc_fig is not None:
                    target.latest_lc_fig.savefig(scratch_lc_path)
                    self.slack_manager.update_channels(text=description, files=[scratch_lc_path])



    def dump_objectId_list(self, output_path=None):
        if output_path is None:
            output_path = self.default_objectId_list_path

        objectId_list = []
        for objectId, target in self.target_lookup.items():
            objectId_list.append(objectId)
        df = pd.DataFrame({"objectId": objectId_list})
        df.to_csv(output_path, index=False)
        return None
        
    def start(
        self, 
        scoring_function: Callable=default_score, 
        modelling_function: Callable=default_sncosmo_model, 
        plots=True,
        break_after_one=False, # ONLY USED FOR TESTING! to break from infinte loop.
    ):
        successful_loops = 0

        while True:
            ## get new data
            self.perform_query_manager_tasks()

            ## are there any manually added targets?
            self.check_for_targets_of_opportunity()

            ## concatenate all the useful photometric data from eg. FINK/ATLAS.
            self.compile_all_target_histories()
            
            ## save all the targets currently care about
            self.dump_objectId_list()

            ## are there any targets that aren't worth modelling?
            logger.info(f"{len(self.target_lookup)} targets before modelling")
            self.initial_new_target_check(scoring_function)
            removed_before_modelling = self.remove_bad_targets(verbose=False)
            logger.info(f"{len(removed_before_modelling)} not worth modelling")

            ## build target models
            self.model_targets(modelling_function, lazy_modelling=self.lazy_modelling)

            ## compute the score for each observatory, remove bad targets.
            for observatory in self.observatories:
                self.evaluate_all_targets(scoring_function, observatory=observatory)
            logger.info(f"{len(self.target_lookup)} targets before removing bad targets")
            self.remove_bad_targets()
            logger.info(f"{len(self.target_lookup)} targets after removing bad targets")

            ## build the ranked list
            for observatory in self.observatories:
                self.build_ranked_target_list(observatory, plots=plots)

            if successful_loops > 0:
                self.send_target_updates()

            ## set all target.updated = False
            self.reset_updated_targets()
            self.reset_all_target_figures()

            sleep_time = self.selector_config.get("sleep_time", 600.)
            logger.info(f"sleep for {sleep_time} sec")
            time.sleep(sleep_time)
            if break_after_one:
                break
            successful_loops = successful_loops + 1