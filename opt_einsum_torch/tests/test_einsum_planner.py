import unittest

import numpy as np
import torch
from opt_einsum.contract import PathInfo

from opt_einsum_torch.planner import EinsumPlanner


class TestEinsumPlanner(unittest.TestCase):
    def test_not_divide(self):
        einsum = EinsumPlanner(torch.device('cuda:0'), 99760)
        einsum_plan = einsum._plan_einsum(
            'ijk,jkl,l->i',
            (10, 10, 20),
            (10, 20, 50),
            (50,),
            dtype_sz=8
        )
        self.assertListEqual(einsum_plan.tensor_storage, ['full'] * 3)
        self.assertIsNone(einsum_plan.split_info)
        self.assertEqual(einsum_plan.mem_required, 99760)

    def test_divide_full_input(self):
        einsum = EinsumPlanner(torch.device('cuda:0'), 67000)
        einsum_plan = einsum._plan_einsum(
            'ik,jkl,l->ij',
            (10, 20),
            (10, 20, 50),
            (50,),
            dtype_sz=8
        )
        self.assertListEqual(einsum_plan.tensor_storage,
                             ['full', 'partial', 'full'])
        self.assertTupleEqual(einsum_plan.split_info, ((2, 'l', 37),))
        # 10*20*8+10*20*50*8/50*37+50*8+2*(10*20*8+10*10*8)
        self.assertEqual(einsum_plan.mem_required, 66000)


class TestTensorDivider(unittest.TestCase):
    def test_not_divide(self):
        path_info = PathInfo(
            None, None, None, None, None, None, 1.,
            1., [0], {'i': 10, 'j': 10, 'k': 20, 'l': 20})
        einsum = EinsumPlanner(torch.device('cuda:0'), 1024 * 1024)
        divide = einsum.find_optimal_divide(
            subscripts=['ijk', 'jkl'],
            tensors=[
                np.empty((10, 10, 20), dtype=np.float64),
                np.empty((10, 20, 20), dtype=np.float64)
            ],
            path_info=path_info,
            target_size=50000
        )
        self.assertTupleEqual(
            divide, (False, (), 48000))

    def test_single_divide(self):
        path_info = PathInfo(
            None, None, None, None, None, None, 1.,
            1., [0], {'i': 10, 'j': 10, 'm': 10, 'k': 20, 'l': 20})
        einsum = EinsumPlanner(torch.device('cuda:0'), 1024 * 1024)

        divide = einsum.find_optimal_divide(
            subscripts=['ijk', 'mkl'],
            tensors=[
                np.empty((10, 10, 20), dtype=np.float64),
                np.empty((10, 20, 20), dtype=np.float64)
            ],
            path_info=path_info,
            target_size=25000
        )
        self.assertTupleEqual(
            divide, (True, ((2, 'k', 10),), 24000))
