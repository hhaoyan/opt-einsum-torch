import torch

from opt_einsum_torch.planner import EinsumPlanner


def einsum(formula, *tensors,
           cuda_device='cuda:0', cuda_mem_limit=0.9,
           async_computation=True):

    planner = EinsumPlanner(
        torch.device(cuda_device), cuda_mem_limit)
    return planner.einsum(
        formula, *tensors, async_computation=async_computation)
