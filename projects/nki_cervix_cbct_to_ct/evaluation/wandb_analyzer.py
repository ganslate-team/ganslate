# Use this script to analyze data from wandb logs.
import wandb
import pandas as pd

# Ascending means for the metric, lower values are better, and the first value in the ascending sort
# will be the best value.
DEFAULT_KEYS = {"descending": ["psnr", "ssim"], "ascending": ["mae", "mse", "nmse"]}

# Lambda fns for some easy ops
flatten = lambda t: [item for sublist in t for item in sublist]
# Check if any of the items of list1 are substrings of string
is_list_element_substr = lambda list1, string: any([key in string for key in list1])
# Filters a list of columns from the dataframe
filter_cols = lambda df, cols: df[df.columns[~df.columns.isin(cols)]]


def main(args):
    api = wandb.Api()
    all_keys = flatten([v for v in DEFAULT_KEYS.values()])

    runs = api.runs(f"{args.entity}/{args.project}")

    for run in runs:
        if run.state == "finished":
            print(f"Loading {run.name} ...")
            df = pd.DataFrame()

            for _, row in run.history().iterrows():
                row_dict = {'iteration': row['_step']}

                for metric_label in row.keys():
                    # Filter to get keys that contain strings that are defined in DEFAULT_KEYS
                    # and ignore 'Train' keys
                    if is_list_element_substr(all_keys, metric_label) and \
                        not 'Train' in metric_label:

                        row_dict[metric_label] = row[metric_label]
                df = df.append(row_dict, ignore_index=True)

            # Drop NaN values
            df = df.dropna()

            for label, series in filter_cols(df, ['iteration']).items():
                for key, value in DEFAULT_KEYS.items():
                    if is_list_element_substr(value, label):
                        sort = key
                        break

                # Depending on if min-max is specified, rank the
                # list of values
                ascending = True if sort == 'ascending' else False
                df[label] = series.rank(ascending=ascending)

            # Get average rank across all the metrics, if any metrics need
            # to be filtered, they can be added in the list
            filter_list = ['iteration']
            df['average_rank'] = filter_cols(df, filter_list).mean(axis=1)
            df = df.sort_values(by='average_rank').reset_index()

            df.to_csv(f"{run.name}.csv")
            print(f"Top 5 iterations: \n {df[['iteration', 'average_rank']].head()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--entity", help="Wandb entity", type=str, default="maastro-clinic")
    parser.add_argument("--project", help="Wandb project", type=str, default="Media_Experiments_V1")

    args = parser.parse_args()

    main(args)
