"""Prefetch deterministic step-addressed batches without changing sample identities."""

import torch
from torch.utils.data import DataLoader

from .training_contracts import batch_indices


class StepBatchSampler:
    def __init__(self, length, *, seed, start_step, stop_step, accumulation, batch_size):
        if start_step < 0 or stop_step < start_step or min(length, accumulation, batch_size) < 1:
            raise ValueError("invalid batch plan")
        self.length, self.seed = length, seed
        self.start_step, self.stop_step = start_step, stop_step
        self.accumulation, self.batch_size = accumulation, batch_size

    def __iter__(self):
        for step in range(self.start_step, self.stop_step):
            for micro in range(self.accumulation):
                yield batch_indices(self.length, self.seed, step, micro, self.batch_size)

    def __len__(self):
        return (self.stop_step - self.start_step) * self.accumulation


def make_loader(dataset, sampler, *, workers, pin_memory):
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    extra = (
        dict(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
        if workers
        else {}
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        # Worker seeding must not advance the restored policy's global RNG.
        # Dataset transforms are deterministic; random augmentation belongs in the main process.
        generator=torch.Generator().manual_seed(sampler.seed),
        **extra,
    )
