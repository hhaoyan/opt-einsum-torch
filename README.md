# Installation
```
pip install git+https://github.com/hhaoyan/opt-einsum-torch
 ```

# opt-einsum-torch

There have been many implementations of Einstein's summation. Numpy's 
`numpy.einsum` is the least efficient one as it only runs in single threads on 
CPU. PyTorch's `torch.einsum` works for both CPU and CUDA tensors. However,
since there is no virtual CUDA memory, `torch.einsum` will run out of CUDA 
memory for large tensors:

```python
import torch

torch.einsum(
    'jrl,ijr,ijrk,ijrl,ikmn->imn',
    torch.empty((100, 3, 1024)).cuda(),
    torch.empty((500, 100, 3)).cuda(),
    torch.empty((500, 100, 3, 1024)).cuda(),
    torch.empty((500, 100, 3, 1024)).cuda(),
    torch.empty((500, 1024, 9, 9)).cuda(),
)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/.../torch/functional.py", line 299, in einsum
    return _VF.einsum(equation, operands)  # type: ignore[attr-defined]
RuntimeError: CUDA out of memory. Tried to allocate 585.94 GiB (GPU 0; 10.76 GiB total capacity; 1.87 GiB already allocated; 7.87 GiB free; 1.89 GiB reserved in total by PyTorch)
```

This code aims at implementing a memory-efficient `einsum` function using
PyTorch as the backend. This code also uses the `opt_einsum` package to 
optimize the contraction path to achieve the minimal FLOPS.

### Usage

```python
import logging
import torch
from opt_einsum_torch import einsum

logging.basicConfig(level=logging.DEBUG)

result = einsum(
    'jrl,ijr,ijrk,ijrl,ikmn->imn',
    torch.empty((1000, 3, 1024)),
    torch.empty((500, 1000, 3)),
    torch.empty((500, 1000, 3, 1024)),
    torch.empty((500, 1000, 3, 1024)),
    torch.empty((500, 1024, 9, 9)),
)
print('Result tensor:', result.shape)
```

Since the input tensors cannot be fit into a single GPU card, `opt-einsum-torch`
will try to split the computation into blocks, and accumulate the results.
```
DEBUG:OptimalEinsum:Print CUDA memory info for device(type='cuda', index=0): total 11554717696 (10.8 GiB), reserved 0 (0 Bytes), allocated 0 (0 Bytes)
DEBUG:OptimalEinsum:Einsum summary: formula: jrl,ijr,ijrk,ijrl,ikmn->imn, shapes: [(1000, 3, 1024), (500, 1000, 3), (500, 1000, 3, 1024), (500, 1000, 3, 1024), (500, 1024, 9, 9)]
DEBUG:OptimalEinsum:Using PyTorch to speed up einsum: naive FLOPs 6.370e+14, optimized FLOPs 6.228e+09, largest intermediate: 1.4 MiB
DEBUG:OptimalEinsum:Plan for performing einsum: storage for input tensors: ['gpu', 'gpu', 'cpu', 'cpu', 'gpu'], tensor split info: ((2, 'j', 829),), maximal CUDA memory usage: 9.7 GiB
DEBUG:OptimalEinsum:Preparing tensors...
DEBUG:OptimalEinsum:Transferred 175.6 MiB from CPU to GPU.
DEBUG:OptimalEinsum:Transferred 9.5 GiB from CPU to GPU.
DEBUG:OptimalEinsum:Transferred 2.0 GiB from CPU to GPU.
Result tensor: torch.Size([500, 9, 9])
```

The resulting tensor `result` will be a PyTorch CPU tensor. You could convert
it into numpy array by simply calling `result.numpy()`.

### Documentation

`opt_einsum_torch.einsum` is the drop-in replacement for `torch.einsum`.

```python
opt_einsum_torch.einsum(
    formula,   # Einsum formula. Ellipsis and broadcasting are not supported. 
    *tensors,  # List of tensors. Could be np.ndarray or torch.tensor.
    cuda_device='cuda:0',  # The device to use for performing einsum.
    cuda_mem_limit=0.9,    # Maximum CUDA memory to use. Values 0.8-0.9 work best.
    async_computation=True # Enable async computation.
)
```

A few things to be noted:
- Not all CUDA memory can be utilized for computing. Usually you may only use up 
  to 85% - 90% of the total CUDA memory. Therefore, set the `cuda_mem_limit`
  that works best for your GPU card.
- Enabling `async_computation` will force all tensors to be pinned memory. This
  increases the overall CPU memory footprint. You may read more info at 
  https://pytorch.org/docs/stable/notes/cuda.html. If you can, passing in 
  tensors that are already pinned memory works best.
- If you use PyTorch in other parts of your code, make sure to delete unused 
  tensors before entering `opt_einsum_torch` or it will run out of CUDA memory.
  Passing in CUDA tensors could speed up things in some cases, but it interferes
  with the tensor planning algorithm and is not recommended.

### Future works

- Support multiple GPUs.
- Memory efficient einsum kernels.
- CUDA data transfer profilers.
