import unittest

import numpy as np
import torch

from opt_einsum_torch.planner import EinsumPlanner


class TestEinsumPlanner(unittest.TestCase):
    def test_not_divide(self):
        einsum = EinsumPlanner(torch.device('cuda:0'), 99760)
        input_tensors_plan, split_info, max_mem_size = einsum._plan_einsum(
            'ijk,jkl,l->i',
            (10, 10, 20),
            (10, 20, 50),
            (50,),
            dtype_sz=8
        )
        self.assertListEqual(input_tensors_plan, ['full'] * 3)
        self.assertIsNone(split_info)
        self.assertEqual(max_mem_size, 99760)

    def test_divide_full_input(self):
        einsum = EinsumPlanner(torch.device('cuda:0'), 67000)
        input_tensors_plan, split_info, max_mem_size = einsum._plan_einsum(
            'ik,jkl,l->ij',
            (10, 20),
            (10, 20, 50),
            (50,),
            dtype_sz=8
        )
        self.assertListEqual(input_tensors_plan, ['full', 'partial', 'full'])
        self.assertTupleEqual(split_info, ((2, 'l', 37),))
        # 10*20*8+10*20*50*8/50*37+50*8+2*(10*20*8+10*10*8)
        self.assertEqual(max_mem_size, 66000)


class TestTensorDivider(unittest.TestCase):
    def test_not_divide(self):
        divide = EinsumPlanner.find_optimal_divide(
            ['ijk', 'jkl'],
            [
                np.empty((10, 10, 20), dtype=np.float64),
                np.empty((10, 20, 20), dtype=np.float64)
            ],
            50000
        )
        self.assertTupleEqual(
            divide, (False, (), 48000))

    def test_single_divide(self):
        divide = EinsumPlanner.find_optimal_divide(
            ['ijk', 'mkl'],
            [
                np.empty((10, 10, 20), dtype=np.float64),
                np.empty((10, 20, 20), dtype=np.float64)
            ],
            25000
        )
        self.assertTupleEqual(
            divide, (True, ((2, 'k', 10),), 24000))
