# coding=utf-8
"""
Copyright (c) DIRECT Contributors

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# Taken from Detectron 2, licensed under Apache 2.0.
# Changes:
# - Docstring to match the rest of the library
# - Calls to other subroutines which do not exist in DIRECT.

import torch
import itertools

from typing import Optional
from torch.utils.data.sampler import Sampler
from util.distributed import multi_gpu


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Parameters
        ----------
        size : int
            Length of the underlying dataset.
        shuffle : bool
            If true, the indices will be shuffled
        seed : int
            Initial seed of the shuffle, must be the same across all workers!
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = 1837237 #multi_gpu.shared_random_seed() TODO
        self._seed = int(seed)

        self._rank = multi_gpu.get_rank()
        self._world_size = multi_gpu.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)
