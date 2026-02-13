import sys
import os
import unittest
# from unittest.mock import MagicMock, patch

# Mock vbcsr before importing dptb to avoid crash if vbcsr is not installed or broken
# sys.modules["vbcsr"] = MagicMock()

import torch
from dptb.postprocess.unified.system import TBSystem
from dptb.postprocess.unified.calculator import HamiltonianCalculator, DeePTBAdapter
from dptb.data import AtomicDataDict

# @patch('dptb.nn.hr2hR.ImageContainer')
# @patch('dptb.nn.hr2hR.AtomicData_vbcsr')
class TestPostProcessHR(unittest.TestCase):
    def setUp(self):
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(self.root, "tests/data/e3_band/ref_model/nnenv.ep1474.pth")
        self.struc_path = os.path.join(self.root, "tests/data/e3_band/data/Si64.vasp")

    def test_get_hR_real_model(self):
        # Initialize System with real model
        system = TBSystem(
            data=self.struc_path,
            calculator=self.model_path,
            device='cpu'
        )
        
        # Call get_hR
        h, s = system.get_hR()

        sK = s.sample_k([0,0,0], sym=True)
        print(sK.to_dense())
        
        # Verification
        # The model in test_to_pardiso might or might not have overlap. 
        # Based on name "nnsk.iter_ovp0.000.pth", it likely does? Or maybe not.
        # Let's check if s is None or not based on model config.
        # But at least the code ran and returned what Hr2HR produced.
        
        

if __name__ == '__main__':
    unittest.main()
