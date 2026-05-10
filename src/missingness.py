# missingness simulation for the enriched feature set
# applied as DataLoader transform (per epoch, stochastic), not pre_transform

import random

import torch


def make_missingness_transform(group_specs, drop_prob=0.5, strategy="zero", seed=None):
    # group_specs: dict like {"structural": (0, 11), "semantic": (11, 20)}
    # drop_prob: per-group prob of being dropped each call
    # strategy: "zero" zeros out the columns, "prune" keeps only the structural part
    # returns a callable that takes a Data object and returns the mutated copy
    rng = random.Random(seed)
    group_names = list(group_specs.keys())

    def transform(data):
        x = data.x.clone()
        mask = torch.ones(len(group_names), dtype=torch.float)

        for i, name in enumerate(group_names):
            if rng.random() < drop_prob:
                start, end = group_specs[name]
                if strategy == "zero":
                    x[:, start:end] = 0.0
                    mask[i] = 0.0
                elif strategy == "prune":
                    # keep structural intact, only zero non-structural groups
                    # mask reflects what was actually zeroed so gating sees consistent signal
                    if name != "structural":
                        x[:, start:end] = 0.0
                        mask[i] = 0.0
                else:
                    raise ValueError(f"unknown strategy: {strategy}")

        data.x = x
        # store as 1xK so pyg batches it correctly to [batch_size, num_groups]
        data.missingness_mask = mask.unsqueeze(0)
        return data

    return transform


def attach_full_mask(data, num_groups):
    # for eval where we want no missingness, attach an all ones mask
    data.missingness_mask = torch.ones(1, num_groups)
    return data
