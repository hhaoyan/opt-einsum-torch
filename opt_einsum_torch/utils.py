from typing import List, Tuple, Iterable, Dict, Union

import numpy as np
from opt_einsum.contract import PathInfo
from torch import Tensor


class DummyArray:
    """A dummy array for planning einsum memory allocation."""

    def __init__(self, shape, dtype_sz):
        self.shape = shape
        self.dtype_sz = dtype_sz


def tensor_size(tensor: Union[Tensor, np.ndarray, DummyArray]) -> int:
    """
    Compute tensor size in bytes.

    :param tensor: The tensor to compute size for.
    :returns: Tensor size in bytes.
    """
    if isinstance(tensor, Tensor):
        return tensor.element_size() * tensor.nelement()
    elif isinstance(tensor, np.ndarray):
        return tensor.dtype.itemsize * tensor.size
    elif isinstance(tensor, DummyArray):
        return tensor.dtype_sz * np.prod(tensor.shape, dtype=int)


def parse_einsum_formula(formula: str) -> Tuple[List[str], str]:
    """
    Parse an einsum formula into input tensor subscripts and output tensor
    subscript.

    Note that we don't support ellipsis.

    :param formula: The einsum formula to parse.
    :return:
    """
    operands, output_subscript = formula.split('->')
    input_subscripts = [x.strip() for x in operands.split(',')]
    return input_subscripts, output_subscript


def find_largest_intermediates(
        path_info: PathInfo, dtype_sz: int) -> Tuple[List[str], int]:
    """
    During einsum contraction, there are some intermediate arrays. This function
    looks for the largest possible list of intermediates while contracting an
    einsum formula.

    :param path_info: The :class:`PathInfo` object from
        `opt_einsum.contract_path`.
    :param dtype_sz: Data type size.
    :return: tuple(list of intermediate subscripts, largest intermediate size)
    """
    operands = path_info.input_subscripts.split(',')
    operand_is_intermediate = [False] * len(operands)

    def subscript2size(subscript):
        shape = [path_info.size_dict[x] for x in subscript]
        return dtype_sz * np.prod(shape, dtype=int)

    intermediate_size = 0
    largest_intermediates_size = 0
    intermediates_subscripts = []
    for contraction in path_info.contraction_list:
        produce_subscript = contraction[2].split('->')[1]
        operands.append(produce_subscript)
        operand_is_intermediate.append(True)
        intermediate_size += subscript2size(produce_subscript)

        if intermediate_size > largest_intermediates_size:
            largest_intermediates_size = intermediate_size
            intermediates_subscripts = [
                x for x, y in zip(operands, operand_is_intermediate) if y]

        for operand_ind in sorted(contraction[0], reverse=True):
            op_subscript = operands.pop(operand_ind)
            is_intermediate = operand_is_intermediate.pop(operand_ind)
            if is_intermediate:
                intermediate_size -= subscript2size(op_subscript)

    return intermediates_subscripts, largest_intermediates_size


def generate_array_selector(
        subscript: str,
        split_indices: Iterable[str],
        split_starts: Iterable[int],
        split_block_sizes: Iterable[int],
        indices_size_dict: Dict[str, int]) -> List[slice]:
    """
    Generate the list of selectors `obj` to use with `array[obj]` when the array
    is split along some indices.

    :param subscript: The subscript of the array.
    :param split_indices: List of indices to split.
    :param split_starts: List of starting points of the splits.
    :param split_block_sizes: List of block sizes of the splits.
    :param indices_size_dict: Dictionary containing sizes of each index.
    :return: The selector `obj` used to select subarray `array[obj]`.
    """
    selector = []
    splits = {
        ind: (start, block_size)
        for ind, start, block_size in
        zip(split_indices, split_starts, split_block_sizes)}
    for i in subscript:
        if i in splits:
            end = min(splits[i][0] + splits[i][1], indices_size_dict[i])
            selector.append(slice(splits[i][0], end))
        else:
            selector.append(slice(None, None, None))
    return selector
