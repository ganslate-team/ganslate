# Use this script to analyze data from wandb logs.
import wandb
import pandas as pd
from omegaconf import OmegaConf
from typing import Optional, Tuple, List
from omegaconf import MISSING, II

from dataclasses import dataclass, field


############################### Analyzer Configuration ########################################
@dataclass
class AnalyzerConfig:
    # Wandb entity and project 
    entity: str = MISSING
    project: str = MISSING
    # Metric tags to ignore in the analysis
    ignore_tags: List = field(default_factory=lambda: [])
    # Additionally group by metric tags during the ranking process
    group_by: List = field(default_factory= lambda: [])
    # Metric tags to include in the analysis 
    # in descending or ascending format for ranking
    rank_descending_keys: List = field(default_factory= lambda: ["psnr", "ssim"])
    rank_ascending_keys: List = field(default_factory= lambda: ["mae", "mse", "nmse"])


# Example: python tools/wandb/wandb_analyzer.py entity=maastro-clinic project="Media_Experiments" 
# group_by=[phantom,cbcttoct] ignore_tags=[plate,'phantom psnr','phantom ssim','phantom nmse','phantom mse','cycle']

# The rank is displayed for all metrics, grouped by all metrics that contain "phantom" and "cbcttoct" tag 
# All the tags in ignore_tags are not considered for the ranking

#################################################################################################


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def list_of_strings_has_substring(list_of_strings, string):
    """Check if any of the strings in `list_of_strings` is  a substrings of `string`"""
    return any([elem.lower() in string.lower() for elem in list_of_strings])

def filter_columns(df, columns):
    """Filters a list of columns from the dataframe"""
    return df[df.columns[~df.columns.isin(columns)]]



def main(conf):
    api = wandb.Api()
    api.entity = conf.entity

    all_keys = conf.rank_descending_keys + conf.rank_ascending_keys
    # Add Train key to list of keys to ignore
    ignore_tags = conf.ignore_tags + ['Train']

    for run in api.runs(f"{conf.project}"):
        if run.state == "finished":
            print(f"Loading {run.name} ...")
            df = pd.DataFrame()

            # Get total number of samples from the total number of iteration
            samples = run.summary._json_dict['_step']

            for _, row in run.history(samples=samples).iterrows():
                row_dict = {'iteration': row['_step']}

                for metric_label in row.keys():
                    if list_of_strings_has_substring(all_keys, metric_label):
                        if not list_of_strings_has_substring(ignore_tags, metric_label):
                            row_dict[metric_label] = row[metric_label]

                df = df.append(row_dict, ignore_index=True)

            # Drop NaN values
            df = df.dropna()
            df = df.set_index('iteration')
            print("Analyzing columns - ", list(df.columns))

            # Rank values based on ascending or descending
            for label, series in df.items():
                if list_of_strings_has_substring(conf.rank_descending_keys, label):
                    df[label] = series.rank(ascending=False)

                elif list_of_strings_has_substring(conf.rank_ascending_keys, label):
                    df[label] = series.rank(ascending=True)
                else:
                    print(f'{label} not in ascending or descending set of keys')

            # Get mode of values across all metrics
            df['mode_rank_across_all_metrics'] = df.mode(axis=1)[0]
            sort_by = ['mode_rank_across_all_metrics']

            # Check for any provided groupings in the metrics
            for group_key in conf.group_by:
                group_metrics = []
                for col in df.columns:
                    # Check if grouping key is a substring of the column
                    if group_key.lower() in col.lower():    
                        group_metrics.append(col)
                group_df = df[group_metrics]
                df[f'mode_rank_across_{group_key}'] = group_df.mode(axis=1)[0]
                sort_by += [f'mode_rank_across_{group_key}']
            
            for val in sort_by:
                df = df.sort_values(by=val)
                df[[val]].to_csv(f"{run.name}_{val}.csv")
                print(f"Top 5 iterations for {val}: \n {df[[val]].head()}\n")


if __name__ == "__main__":
    cli = OmegaConf.from_cli()
    conf = AnalyzerConfig()
    conf = OmegaConf.merge(conf, cli)
    main(conf)
