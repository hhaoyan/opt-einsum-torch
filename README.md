# opt-einsum-torch

There have been many implementations of Einstein's summation. numpy's 
`numpy.einsum` is the least efficient one as it only runs in single thread on 
CPU. PyTorch's `torch.einsum` works for both CPU and CUDA tensors. However,
since there is no virtual CUDA memory, `torch.einsum` will run out of CUDA 
memory for large tensors. 

This code aims at implementing a memory-efficient `einsum` function using
PyTorch as the backend. This code also uses the `opt_einsum` package to 
optimizes the contraction path to achieve the minimal FLOPS.

### Usage

```python
from opt_einsum_torch.planner import EinsumPlanner
import torch

# Some huge tensors
arr1, arr2 = ..., ...
ee = EinsumPlanner(torch.device('cuda:0'), cuda_mem_limit=0.9)
result = ee.einsum('ijk,jkl->il', arr1, arr2)

```

The resulting tensor `result` will be a PyTorch CPU tensor. You could convert
it into numpy array by simply calling `result.numpy()`.

### Future works

- Support multiple GPUs.
- Memory efficient einsum kernels.
- CUDA data transfer profilers.