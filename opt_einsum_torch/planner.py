import itertools
import logging
from typing import Union, List, Tuple, Iterable

import humanize
import numpy as np
import opt_einsum as oe
import torch
from opt_einsum.contract import PathInfo
from torch import Tensor

from opt_einsum_torch.utils import DummyArray, find_largest_intermediates, \
    parse_einsum_formula, generate_array_selector, tensor_size

logger = logging.getLogger('OptimalEinsum')

TENSOR_LOCATION_CPU = 'cpu'
TENSOR_LOCATION_CUDA = 'gpu'


class EinsumPlanner:
    """An efficient implementation of einsum."""

    def __init__(self,
                 cuda_device: torch.device,
                 cuda_mem_limit: Union[float, int] = 0.9):
        """
        Setup the einsum environment by specifying CUDA device and memory limit.

        :param cuda_device: The CUDA device to use.
        :param cuda_mem_limit: Maximum CUDA memory size to use. Defaults to 90%
            of the total CUDA memory size (to allow some memory mis-alignments).
        """
        self.cuda_device = cuda_device

        if isinstance(cuda_mem_limit, float):
            cuda_total_mem = torch.cuda.get_device_properties(
                cuda_device).total_memory
            self.cuda_mem_limit = int(cuda_mem_limit * cuda_total_mem)
        elif isinstance(cuda_mem_limit, int):
            self.cuda_mem_limit = cuda_mem_limit
        else:
            raise ValueError("Unknown CUDA memory limit %r", cuda_mem_limit)
        # Align 1K bytes
        self.cuda_mem_limit = (self.cuda_mem_limit >> 10) << 10

    def _print_memory_info(self):
        total = torch.cuda.get_device_properties(self.cuda_device).total_memory
        memory = torch.cuda.memory_stats(device=self.cuda_device)
        allocated = memory.get("allocated_bytes.all.current")
        reserved = memory.get("reserved_bytes.all.current")
        logger.debug(
            "Print CUDA memory info for %r: total %d (%s), "
            "reserved %d (%s), allocated %d (%s)",
            self.cuda_device,
            total, humanize.naturalsize(total, binary=True),
            reserved, humanize.naturalsize(reserved, binary=True),
            allocated, humanize.naturalsize(allocated, binary=True),
        )
        if total - allocated < self.cuda_mem_limit:
            logger.warning(
                "CUDA device %r has less free memory (%s) than expected (%s), "
                "did you have other tensors in your application?",
                self.cuda_device,
                humanize.naturalsize(total - allocated, binary=True),
                humanize.naturalsize(self.cuda_mem_limit, binary=True))

    def find_optimal_divide(
            self,
            subscripts: List[str],
            tensors: List[Union[Tensor, np.ndarray, DummyArray]],
            path_info: PathInfo,
            target_size: int) -> Tuple[bool, Tuple[Tuple[int, str, int]], int]:
        """
        Find the best split of tensors along some index such that each split
        occupies no more than `target_size` bytes.

        For example, for einsum `ijk,mkl->ijml` where the first array has shape
        `(10, 10, 20)` and the second array has shape `(10, 20, 20)`, if the
        memory limit is 25000 bytes and all arrays are double-floating numbers,
        then the best split is to divide tensors along the `k` axis into 2
        sub-arrays.

        :param subscripts: List of strings representing the subscripts of
            tensors, e.g., 'ijk'.
        :param tensors: List of tensors to split. The shapes must be consistent
            with `subscripts`.
        :param path_info: :class:`PathInfo` object from `opt_einsum`.
        :param target_size: Target memory size limit.
        :return: Tuple of (needs_divide, list of splits, maximal memory size).
            The list of splits are (n_split, index, block size)
        """
        if len(subscripts) != len(tensors):
            raise ValueError("The length of the tensor subscripts does not "
                             "equal to the length of tensors")
        # Size of all tensors
        tensor_sizes = [tensor_size(x) for x in tensors]
        total_tensor_size = sum(tensor_sizes)

        if total_tensor_size <= target_size:
            return False, (), total_tensor_size

        split_plans = []

        for i in path_info.size_dict:
            # How much size is each one slice?
            slice_size = sum([x // path_info.size_dict[i]
                              for x, index in zip(tensor_sizes, subscripts)
                              if i in index])
            # Need to reduce (total_size - target_size)
            reduction_goal = total_tensor_size - target_size
            n_slice_reduction = (reduction_goal + slice_size - 1) // slice_size
            block_size = path_info.size_dict[i] - n_slice_reduction
            if block_size <= 0:
                # Cannot reduce this index
                continue

            n_splits = (path_info.size_dict[i] + block_size - 1) // block_size
            split_size = total_tensor_size - slice_size * n_slice_reduction
            split_plans.append((n_splits, i, block_size, split_size))

        if not split_plans:
            raise ValueError(
                "Could not find a valid split that "
                "satisfies the memory requirements (< %s)." % (
                    humanize.naturalsize(self.cuda_mem_limit, binary=True)
                ))

        # First minimize number of splits, since each CPU/CUDA transfer is very
        # expensive.
        # Then maximize the size of each split to achieve maximal FLOPS.
        best_split = min(split_plans, key=lambda x: (x[0], -x[2]))
        return True, (best_split[:3],), best_split[3]

    def _prepare_input_arrays(
            self, arrays: Iterable[Union[Tensor, np.ndarray]],
            plans: Iterable[str], async_transfer=True) -> List[Tensor]:
        """
        Prepare the list of arrays into GPU/GPU-ready tensors.

        If `async_transfer=True`, then the arrays marked "partial" will be
        converted into pinned memory tensors, which supports async CPU/CUDA data
        transfer. But this increases the amount of CPU memory used.

        :param arrays: List of arrays.
        :param plans: 'gpu' or 'cpu'. 'gpu' tensors will be completely CUDA
            tensors, and 'cpu' tensors will be CPU tensors.
        :param async_transfer: If set to true, will pin memory for CPU tensors 
            to enable async data transfer.
        :returns: List of PyTorch tensors.
        """
        tensor_by_id = {}
        transfer_size = 0
        for array, plan in zip(arrays, plans):
            assert plan in {TENSOR_LOCATION_CUDA, TENSOR_LOCATION_CPU}, \
                "plan must be '%s' or '%s'" % (
                    TENSOR_LOCATION_CUDA, TENSOR_LOCATION_CPU
                )
            if id(array) in tensor_by_id:
                continue

            if isinstance(array, np.ndarray):
                torch_array = torch.from_numpy(array)
            elif isinstance(array, torch.Tensor):
                torch_array = array
            else:
                raise TypeError(
                    "Cannot convert %r into PyTorch array." % type(array))

            if (not torch_array.is_cuda or
                    torch_array.device != self.cuda_device):
                if plan == TENSOR_LOCATION_CUDA:
                    transfer_size += tensor_size(torch_array)
                    tensor_by_id[id(array)] = torch_array.to(
                        self.cuda_device, non_blocking=async_transfer)
                elif plan == TENSOR_LOCATION_CPU:
                    if not async_transfer:
                        tensor_by_id[id(array)] = torch_array
                    else:
                        tensor_by_id[id(array)] = torch_array.pin_memory()
            else:
                tensor_by_id[id(array)] = torch_array
                if plan == TENSOR_LOCATION_CUDA:
                    logger.debug(
                        "You provided a tensor(%r, dtype=%s) on device %r. "
                        "This is not recommended as it may impact how we "
                        "allocate computations.",
                        tuple(torch_array.shape), str(torch_array.dtype),
                        self.cuda_device)
                elif plan == TENSOR_LOCATION_CPU:
                    logger.warning(
                        "You provided a tensor(%r, dtype=%s) on device %r, "
                        "which is not expected to be on GPU. This "
                        "will likely overflow CUDA memory.",
                        tuple(torch_array.shape), str(torch_array.dtype),
                        self.cuda_device)

        if transfer_size:
            logger.debug('Transferred %s from CPU to GPU.',
                         humanize.naturalsize(transfer_size, binary=True))
        gpu_tensors = [tensor_by_id[id(x)] for x in arrays]
        return gpu_tensors

    def _plan_einsum(self, formula, *tensor_shapes,
                     tensor_ids=None, dtype_sz=4):
        """
        Make a plan for performing einsum.

        :param formula: einsum formula.
        :param tensor_shapes: Shapes of the input tensors.
        :param tensor_ids: List of ids of the input tensors for de-duplication.
        :param dtype_sz: Byte size of the data type.
        :param async_computation: Whether to use async computation.
        :return:
        """
        if tensor_ids is None:
            tensor_ids = list(range(len(tensor_shapes)))

        path, path_info = oe.contract_path(
            formula, *tensor_shapes, optimize='optimal', shapes=True)
        logger.debug('Einsum summary: formula: %s, shapes: %r',
                     formula, [tuple(x) for x in tensor_shapes])
        logger.debug(
            'Using PyTorch to speed up einsum: '
            'naive FLOPs %.3e, optimized FLOPs %.3e, largest intermediate: %s',
            path_info.naive_cost, path_info.opt_cost,
            humanize.naturalsize(path_info.largest_intermediate, binary=True))

        # Look for the contraction with the largest intermediates
        intermediate_subscripts, _ = find_largest_intermediates(
            path_info, dtype_sz)
        intermediates_shape = [
            [path_info.size_dict[i] for i in x]
            for x in intermediate_subscripts]

        input_subscripts, _ = parse_einsum_formula(formula)
        tensor_sizes = [np.prod(x) * dtype_sz for x in tensor_shapes]

        # Remove duplicates
        cuda_tensor_ids = set()
        cuda_tensor_subscripts = []
        cuda_tensor_shapes = []
        for tensor_id, subscript, shape in zip(
                tensor_ids, input_subscripts, tensor_shapes):
            if (tensor_id, subscript) in cuda_tensor_ids:
                continue
            cuda_tensor_ids.add((tensor_id, subscript))
            cuda_tensor_subscripts.append(subscript)
            cuda_tensor_shapes.append(shape)

        # Figure out splits.
        # Sometimes einsum will copy operands, so we use n_computation=2
        # https://github.com/pytorch/pytorch/issues/60295
        n_computation = 2
        all_subscripts = (cuda_tensor_subscripts +
                          intermediate_subscripts * n_computation)
        all_tensors = (
                [DummyArray(x, dtype_sz) for x in cuda_tensor_shapes] +
                [DummyArray(x, dtype_sz) for x in intermediates_shape] *
                n_computation)
        need_split, splits, total_size = self.find_optimal_divide(
            subscripts=all_subscripts, tensors=all_tensors,
            path_info=path_info, target_size=self.cuda_mem_limit
        )
        if not need_split:
            logger.debug(
                'All tensors fit CUDA memory, going ahead with torch.einsum.')
            return (
                path, path_info,
                [TENSOR_LOCATION_CUDA] * len(tensor_shapes),
                None, total_size)

        assert len(splits) == 1, "Currently we only support one split."
        n_split, split_index, block_size = splits[0]

        # Optimize: can we push input tensors all to CUDA memory?
        plan_by_id = {}
        for tensor_id, sz, subscript in sorted(
                zip(tensor_ids, tensor_sizes, input_subscripts)):
            if tensor_id in plan_by_id:
                continue
            if split_index in subscript:
                delta_sz = sz // path_info.size_dict[split_index] * (
                        path_info.size_dict[split_index] - block_size)
                if total_size + delta_sz <= self.cuda_mem_limit:
                    total_size += delta_sz
                    plan_by_id[tensor_id] = TENSOR_LOCATION_CUDA
                else:
                    plan_by_id[tensor_id] = TENSOR_LOCATION_CPU
            else:
                plan_by_id[tensor_id] = TENSOR_LOCATION_CUDA

        input_tensors_plan = [plan_by_id[x] for x in tensor_ids]

        logger.debug(
            'Plan for performing einsum: storage for input tensors: %r, '
            'tensor split info: %r, maximal CUDA memory usage: %s',
            input_tensors_plan, splits,
            humanize.naturalsize(total_size, binary=True))
        return path, path_info, input_tensors_plan, splits, total_size

    @staticmethod
    def check_dtypes(
            tensors: Iterable[Union[Tensor, np.ndarray]]
    ) -> Tuple[int, torch.dtype]:
        """
        Check all tensors are floating numbers and return size of the floating
        numbers.

        :param tensors: Tensors to check.
        :return: Size of the floating number dtype.
        """
        dtypes = set()
        for i, tensor in enumerate(tensors):
            if isinstance(tensor, np.ndarray):
                if not np.issubdtype(tensor.dtype, np.floating):
                    raise TypeError(
                        "%d-th tensor (np.ndarray, %r) is not "
                        "a floating number type" % (i, tensor.dtype))
                size = tensor.dtype.itemsize
                if len(dtypes) > 0 and size not in dtypes:
                    raise TypeError(
                        "%d-th tensor (np.ndarray, %r) has floating number of "
                        "size %d, while previous tensors have size %d" % (
                            i, tensor.dtype, size, list(dtypes)[0]
                        ))
                dtypes.add(size)
            elif isinstance(tensor, Tensor):
                if not tensor.dtype.is_floating_point:
                    raise TypeError(
                        "%d-th tensor (torch.Tensor, %r) is not a floating "
                        "number type" % (i, tensor.dtype))
                size = torch.finfo(tensor.dtype).bits // 8
                if len(dtypes) > 0 and size not in dtypes:
                    raise TypeError(
                        "%d-th tensor (np.ndarray, %r) has floating number of "
                        "size %d, while previous tensors have size %d" % (
                            i, tensor.dtype, size, list(dtypes)[0]
                        ))
                dtypes.add(size)

        dtype_sz = list(dtypes)[0]
        if dtype_sz == 2:
            return 2, torch.float16
        elif dtype_sz == 4:
            return 4, torch.float32
        elif dtype_sz == 8:
            return 8, torch.float64
        else:
            raise TypeError("Unknown floating number of size %d" % dtype_sz)

    def _batch_einsum(
            self, formula, tensors, dtype, async_computation,
            path, path_info, input_tensors_plan, splits):
        # Create the result CPU tensor
        result_shape = tuple(
            path_info.size_dict[x] for x in path_info.output_subscript)
        result = torch.zeros(result_shape, dtype=dtype,
                             pin_memory=async_computation)
        result_buffer = []

        transfer_stream = torch.cuda.Stream(self.cuda_device, priority=0)

        # Split product
        split_indices = [x[1] for x in splits]
        split_blocksizes = [x[2] for x in splits]
        split_ranges = [range(0, path_info.size_dict[x[1]], x[2]) for x in
                        splits]

        def handle_last_results():
            transfer_stream.synchronize()
            for buf, result_indexer in result_buffer:
                result[result_indexer] += buf
            del result_buffer[:]

        for block_start in itertools.product(*split_ranges):
            arg_by_id = {}
            args = []
            transfer_size = 0
            for index, tensor, plan in zip(
                    path_info.input_subscripts.split(','),
                    tensors,
                    input_tensors_plan):
                if (id(tensor), index) in arg_by_id:
                    args.append(arg_by_id[(id(tensor), index)])
                    continue

                indexer = generate_array_selector(
                    index, split_indices, block_start,
                    split_blocksizes, path_info.size_dict)

                # Create the sliced tensor, transfer to GPU if necessary.
                arg = tensor[indexer]
                if plan == TENSOR_LOCATION_CPU:
                    arg = arg.to(
                        self.cuda_device, non_blocking=async_computation)
                    transfer_size += tensor_size(arg)

                arg_by_id[(id(tensor), index)] = arg
                args.append(arg)

            logger.debug("Transferred %s from CPU to GPU.",
                         humanize.naturalsize(transfer_size, binary=True))

            partial_result = oe.contract(
                formula, *args,
                optimize=path, use_blas=True, backend='torch')
            handle_last_results()

            # Aggregate the result
            indexer = generate_array_selector(
                path_info.output_subscript, split_indices,
                block_start, split_blocksizes, path_info.size_dict)
            if not async_computation:
                result[indexer] += partial_result.cpu()
            else:
                # Wait for einsum to complete, then start data transfer.
                torch.cuda.synchronize(self.cuda_device)
                with torch.cuda.stream(transfer_stream):
                    result_buffer.append((
                        partial_result.to('cpu', non_blocking=True),
                        indexer))

        if async_computation:
            handle_last_results()

        return result

    def einsum(
            self, formula, *tensors: Union[np.array, Tensor],
            async_computation=True):
        # Empty reserved CUDA memory.
        torch.cuda.empty_cache()
        self._print_memory_info()

        dtype_sz, result_dtype = self.check_dtypes(tensors)
        einsum_plan = self._plan_einsum(
            formula, *[x.shape for x in tensors],
            tensor_ids=[id(x) for x in tensors],
            dtype_sz=dtype_sz)
        path, path_info, input_tensors_plan, splits, total_size = einsum_plan

        logger.debug('Preparing tensors...')
        tensors = self._prepare_input_arrays(
            tensors, input_tensors_plan, async_transfer=async_computation)

        if not splits:
            # If no split is needed, it implies all input tensors are CUDA
            # tensors.
            with torch.no_grad():
                result = oe.contract(
                    formula, *tensors,
                    optimize=path, use_blas=True, backend='torch').cpu()
        else:
            with torch.no_grad():
                result = self._batch_einsum(
                    formula, tensors, result_dtype, async_computation,
                    path, path_info, input_tensors_plan, splits
                )
        return result
