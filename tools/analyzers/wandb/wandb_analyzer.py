# Use this script to analyze data from wandb logs.
import wandb
import pandas as pd
from omegaconf import OmegaConf
from typing import Optional, Tuple, List
from omegaconf import MISSING, II

import utils
from dataclasses import dataclass, field

from loguru import logger


############################### Analyzer Configuration ########################################
@dataclass
class AnalyzerConfig:
    # Wandb entity and project
    entity: str = MISSING
    project: str = MISSING
    # Select a particular run ID to run the analyzer on
    run_id: str = MISSING
    # Run validation analyzer only up to the last checkpoint specified
    last_ckpt: Optional[int] = None
    # Metric tags to ignore in the analysis
    ignore_tags: List = field(default_factory=lambda: [])
    # Additionally group by metric tags during the ranking process
    group_by: List = field(default_factory=lambda: [])
    # Once the metrics are ranked, this determines how the ranks are aggregated
    aggregate_ranks_by: str = "mean"
    # Sampling frequency applied to total number of iterations
    # If iters_sampling_freq = 10, every 10 iterations are considered for ranking
    iters_sampling_freq: int = 1
    # Metric tags to include in the analysis
    # in descending or ascending format for ranking
    rank_descending_keys: List = field(default_factory=lambda: ["psnr", "ssim"])
    rank_ascending_keys: List = field(default_factory=lambda: ["mae", "mse", "nmse"])


# Example: python tools/analyzers/wandb/wandb_analyzer.py entity=maastro-clinic project="Media_Experiments" run_id="348tusn"
# group_by=[phantom,BODY] ignore_tags=['cycle']
#################################################################################################


def main(conf):
    api = wandb.Api()
    api.entity = conf.entity

    for run in api.runs(f"{conf.project}"):
        if run.id == conf.run_id:
            logger.info(f"Loading {run.name} ...")
            df = utils.get_wandb_history(run, conf)

            # Overall all the checkpoints in the run history, get ranks for the metrics
            # by going over each metric and ranking the series in ascending or descending order
            for label, series in df.items():
                if utils.list_of_strings_has_substring(conf.rank_descending_keys, label):
                    df[label] = series.rank(ascending=False)
                elif utils.list_of_strings_has_substring(conf.rank_ascending_keys, label):
                    df[label] = series.rank(ascending=True)
                else:
                    logger.warning(f'{label} not in ascending or descending set of keys')

            # Aggregate 'all' metrics based on rank and selected method of ordering
            df[f'{conf.aggregate_ranks_by}_rank_all_metrics'] = utils.get_aggregate_ranks(
                df, conf.aggregate_ranks_by)
            sort_by = [f'{conf.aggregate_ranks_by}_rank_all_metrics']

            # Check for any provided groups to be inspected among the metrics
            for group_key in conf.group_by:
                # If group key is present in df columns add it to
                group_metric_cols = [col for col in df.columns if group_key.lower() in col.lower()]
                group_df = df[group_metric_cols]
                # Aggregate 'group_key' metrics based on rank and selected method
                df[f'{conf.aggregate_ranks_by}_rank_{group_key}_metrics'] = utils.get_aggregate_ranks(
                    group_df, conf.aggregate_ranks_by)
                sort_by += [f'{conf.aggregate_ranks_by}_rank_{group_key}_metrics']

            # For 'all' metrics and grouped metrics, sort by the aggregate rank
            for val in sort_by:
                df = df.sort_values(by=val)
                df[val].to_csv(f"{run.name}_{val}.csv")
                logger.info(f"Top 5 iterations for {val}: \n {df[val].head()}\n")


if __name__ == "__main__":
    cli = OmegaConf.from_cli()
    conf = AnalyzerConfig()
    conf = OmegaConf.merge(conf, cli)
    main(conf)
