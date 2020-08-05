import os
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from midaGAN.utils.normalization import z_score_normalize
from midaGAN.utils import sitk_utils


class SliceBasedDataset(Dataset):
    def __init__(self, conf):
        self.dir_root = os.path.join(conf.dataset.root)

        dataset_summary = pd.read_csv(os.path.join(conf.dataset.root, 'dataset_summary.csv'))
        # Filter out rows by their domain
        self.domain_A_summary = dataset_summary[dataset_summary["volume_filename"].str.startswith('A')]
        self.domain_B_summary = dataset_summary[dataset_summary["volume_filename"].str.startswith('B')]

        self.num_datapoints_A = len(self.domain_A_summary)
        self.num_datapoints_B = len(self.domain_B_summary)

    def __getitem__(self, index):
        index_A = int(index % self.num_datapoints_A)
        index_B = random.randint(0, self.num_datapoints_B - 1)

        summary_A = self.domain_A_summary.iloc[index_A]
        summary_B = self.domain_B_summary.iloc[index_B]

        path_A = os.path.join(self.dir_root, summary_A["volume_filename"])
        path_B = os.path.join(self.dir_root, summary_B["volume_filename"])
        
        # load volume as SimpleITK object
        A = sitk_utils.load(path_A)
        B = sitk_utils.load(path_B)

        A = sitk_utils.get_tensor(A)
        B = sitk_utils.get_tensor(B)

        # Take the slice
        A = A[summary_A["slice"]]
        B = B[summary_B["slice"]]

        # Z-score normalization per volume
        mean_std_A = (summary_A["volume_mean"], summary_A["volume_std"])
        mean_std_B = (summary_B["volume_mean"], summary_B["volume_std"])
        min_max_A = (summary_A["volume_min"], summary_A["volume_max"])
        min_max_B = (summary_B["volume_min"], summary_B["volume_max"])
        A = z_score_normalize(A, scale_to_range=(-1,1), mean_std=mean_std_A, original_scale=min_max_A)
        B = z_score_normalize(B, scale_to_range=(-1,1), mean_std=mean_std_B, original_scale=min_max_B)
        
        # Add channel dimension (1 = grayscale)
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.num_datapoints_A, self.num_datapoints_B)
