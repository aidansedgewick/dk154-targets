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
from dk154_targets.query_managers import FinkQueryManager, AtlasQueryManager
from dk154_targets.scoring import default_score
from dk154_targets.target import Target
from dk154_targets.utils import chunk_list, readstamp

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

    default_target_list_dir = paths.base_path / "ranked_target_lists"

    default_unranked_value = 9999

    def __init__(self, selector_config):

        ###=========== config ===========###
        self.selector_config = selector_config or {}

        ###======== target "storage" ========###
        self.target_lookup = {}

        ###======== target_list_path =========###
        self.target_list_dir = Path(
            self.selector_config.get(
                "target_list_dir",
                self.default_target_list_dir
            )
        )
        self.target_list_dir.mkdir(exist_ok=True, parents=True)

        ###======== query managers ========###
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
        self.targets_of_opportunity_path.mkdir(exist_ok=True, parents=True)

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
        """
        Remember to call the tasks of your query manager in perform_query_manager_tasks()
        """

        
        fink_config = self.query_mananger_config.get("fink", None)
        if fink_config is None:
            logger.warning("no fink_config! Set fink_query_manager to None.")
            self.fink_query_manager = None
        else:
            logger.info("init FINK query manager...")
            self.fink_query_manager = FinkQueryManager(fink_config, self.target_lookup)

        atlas_config = self.query_mananger_config.get("atlas", None)
        if atlas_config is None:
            logger.info("no atlas config")
            self.atlas_query_manager = None
        else:
            logger.info("init ATLAS query manager...")
            self.atlas_query_manager = AtlasQueryManager(atlas_config, self.target_lookup)

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
        logger.info(f"{len(self.observatories)} obs (inc. `no_observatory`:None)")


    def add_target(self, target: Target):
        objectId = target.objectId
        if objectId in self.target_lookup:
            raise ValueError(f"{objectId} already in target_lookup")
        self.target_lookup[target.objectId] = target


    def add_targets_from_df(self, target_list: pd.DataFrame, key="objectId", validate=True):
        n_groups = len(np.unique(target_list[key]))
        for ii, (objectId, target_history) in enumerate(target_list.groupby(key)):
            if ii % 10 == 0:
                logger.info(f"target {ii+1} of {n_groups}")
            target = Target.from_target_history(
                objectId, target_history
            )
            if validate and objectId in self.target_lookup:
                # Already know about this one...
                raise ValueError(f"validate=True, and target {objectId} already in targets.")
            self.target_lookup[objectId] = target
        return


    def add_targets_from_objectId_list(self, objectId_list, chunk_size=20, **fink_kwargs):
        if isinstance(objectId_list, str):
            objectId_list = [objectId_list]
        logger.info(f"initialise {len(objectId_list)} objects")
        df_list = []
        for ii, objectId_chunk in enumerate(chunk_list(objectId_list, size=chunk_size)):
            objectId_str = ",".join(objId for objId in objectId_chunk)
            df = FinkQuery.query_objects(return_df=True, objectId=objectId_str, **fink_kwargs)
            
            logger.info(f"chunk {ii+1} of {int(len(objectId_list)/chunk_size)+1} (n_rows={len(df)})")
            df_list.append(df)
        df = pd.concat(df_list)
        self.add_targets_from_df(df)


    def perform_query_manager_tasks(self, fake_alerts=False, dump_alerts=True):
        logger.info("perform all query manager tasks")
        if self.fink_query_manager is None:
            logger.warning("no fink query manager!")
        else:
            self.fink_query_manager.perform_all_tasks(
                fake_alerts=fake_alerts, dump_alerts=dump_alerts
            )
        if self.atlas_query_manager is None:
            pass
        else:
            self.atlas_query_manager.perform_all_tasks()

    #def initial

    def initial_new_target_check(self, scoring_function: Callable, t_ref: Time=None):

        t_ref = t_ref or Time.now()
        to_remove = []
        for objectId, target in self.target_lookup.items():
            last_score = target.get_last_score("no_observatory")
            if last_score is not None:
                continue
            target.evaluate_target(scoring_function, None, t_ref=t_ref)
            initial_score = target.get_last_score("no_observatory")

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

    def remove_bad_targets(self,):
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
            return None
        logger.info(f"model {len(self.target_lookup)} targets")
        modelled = []
        for objectId, target in self.target_lookup.items():
            if not target.updated and lazy_modelling:
                logger.info(f"{objectId} not updated: skip")
                continue
            target.model_target(modelling_function)
            target.updated = False
        

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
            output_dir = self.target_list_dir #/ obs_name
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
                if ii < 20:
                    target.update_cutouts()
                lc_fig = target.plot_lightcurve(t_ref=t_ref)
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

    def dump_full_target_list(self, output_path=None):
        if output_path is None:
            output_path = self.default_full_target_history_path
        df_list = []
        for objectId, target in self.target_lookup.items():
            df_list.append(target.target_history)
        if len(df_list) == 0:
            return
        output = pd.concat(df_list)
        output.to_csv(output_path, index=False)
        logger.info("save all target_history")
        return None
        
    def start(
        self, 
        scoring_function: Callable=default_score, 
        modelling_function: Callable=default_sncosmo_model, 
        plots=True,
        save_target_history=True,
        target_history_path=None,
        break_after_one=False, # ONLY USED FOR TESTING! to break from infinte loop.
    ):
        while True:
            ## get new data
            self.perform_query_manager_tasks()

            ## are there any manually added targets?
            self.check_for_targets_of_opportunity()

            ## are there any targets that aren't worth modelling?
            logger.info(f"{len(self.target_lookup)} targets before modelling")
            self.initial_new_target_check(scoring_function)
            removed_before_modelling = self.remove_bad_targets()
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

            ## save all the targets currently care about
            if save_target_history:
                self.dump_full_target_list(output_path=target_history_path)
            sleep_time = self.selector_config.get("sleep_time", 5.)
            logger.info(f"sleep for {sleep_time} sec")
            time.sleep(sleep_time)
            if break_after_one:
                break