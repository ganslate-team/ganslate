import pandas as pd
import wandb


def get_wandb_history(run, conf):
    """
    Get wandb history from a particular run as a dataframe
    """
    df = pd.DataFrame()

    # Get total number of samples from the total number of iteration
    samples = run.summary._json_dict['_step']

    for _, row in run.history(samples=samples).iterrows():
        row_dict = {'iteration': row['_step']}

        if row_dict['iteration'] % conf.iters_sampling_freq != 0:
            continue

        if conf.last_ckpt and row_dict['iteration'] > conf.last_ckpt:
            logger.info(f"Stopped collecting samples @{row_dict['iteration']}")
            break

        for metric_label in row.keys():
            if list_of_strings_has_substring(conf.rank_descending_keys + conf.rank_ascending_keys,
                                             metric_label):
                if not list_of_strings_has_substring(conf.ignore_tags + ['Train'], metric_label):
                    row_dict[metric_label] = row[metric_label]

        df = df.append(row_dict, ignore_index=True)

    # Drop NaN values
    df = df.dropna()
    df = df.set_index('iteration')
    return df


def get_aggregate_ranks(df, metric):
    if metric == "mean":
        return df.mean(axis=1) / len(df)
    elif metric == "mode":
        return df.mode(axis=1)[0] / len(df)


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def list_of_strings_has_substring(list_of_strings, string):
    """Check if any of the strings in `list_of_strings` is  a substrings of `string`"""
    return any([elem.lower() in string.lower() for elem in list_of_strings])


def filter_columns(df, columns):
    """Filters a list of columns from the dataframe"""
    return df[df.columns[~df.columns.isin(columns)]]
