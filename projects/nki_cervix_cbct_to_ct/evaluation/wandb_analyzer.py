# Use this script to analyze data from wandb logs.
import wandb
import pandas as pd
import copy
# Ascending means for the metric, lower values are better, and the first value in the ascending sort
# will be the best value.
DEFAULT_KEYS = {"descending": ["psnr", "ssim"], "ascending": ["mae", "mse", "nmse"]}


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def list_of_strings_has_substring(list_of_strings, string):
    """Check if any of the strings in `list_of_strings` is  a substrings of `string`"""
    return any([elem in string for elem in list_of_strings])


def filter_columns(df, columns):
    """Filters a list of columns from the dataframe"""
    return df[df.columns[~df.columns.isin(columns)]]


def main(args):
    api = wandb.Api()
    api.entity = args.entity

    all_keys = flatten(DEFAULT_KEYS.values())

    # Add Train key to list of keys to ignore
    ignore_tags = args.ignore_tags
    ignore_tags.extend(['Train'])

    for run in api.runs(f"{args.project}"):
        if run.state == "finished":
            print(f"Loading {run.name} ...")
            df = pd.DataFrame()

            for _, row in run.history().iterrows():
                row_dict = {'iteration': row['_step']}

                for metric_label in row.keys():
                    # Filter to get keys that contain strings that are 
                    # defined in DEFAULT_KEYS and ignore 'Train' keys
                    if list_of_strings_has_substring(all_keys, metric_label):
                        if not list_of_strings_has_substring(ignore_tags, metric_label):
                            row_dict[metric_label] = row[metric_label]

                df = df.append(row_dict, ignore_index=True)

            # Drop NaN values
            df = df.dropna()

            for label, series in filter_columns(df, ['iteration']).items():
                for key, value in DEFAULT_KEYS.items():
                    if list_of_strings_has_substring(value, label):
                        sort = key
                        break

                # Depending on if min-max is specified, rank the
                # list of values
                ascending = sort == 'ascending'
                df[label] = series.rank(ascending=ascending)

            # Get average rank across all the metrics, if any metrics need
            # to be filtered, they can be added in the list
            filter_list = ['iteration']
            df['average_rank'] = filter_columns(df, filter_list).mean(axis=1)
            df = df.sort_values(by='average_rank').reset_index()

            df.to_csv(f"{run.name}.csv")
            print(f"Top 5 iterations: \n {df[['iteration', 'average_rank']].head()}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--entity", help="Wandb entity", type=str, default=None)
    parser.add_argument("--project", help="Wandb project", type=str)
    parser.add_argument("--ignore_tags", help="List of regex tags to ignore in the metric columns", nargs='+', default=[])
    args = parser.parse_args()

    main(args)
