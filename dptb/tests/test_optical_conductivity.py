import unittest
import os
import torch
import numpy as np
from dptb.postprocess.unified.system import TBSystem

class TestOpticalConductivity(unittest.TestCase):
    def setUp(self):
        # paths relative to project root
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../examples/ToW90_PythTB"))
        self.model_path = os.path.join(self.root_dir, "models", "nnsk.ep20.pth")
        self.struct_path = os.path.join(self.root_dir, "silicon.vasp")
        
        # simple check if files exist
        if not os.path.exists(self.model_path) or not os.path.exists(self.struct_path):
            self.skipTest(f"Test files not found at {self.root_dir}")

        self.device = 'cpu'
        
    def test_silicon_optical_conductivity_orth(self):
        """Test optical conductivity calculation for Silicon model."""
        # Initialize System
        system = TBSystem(
            data=self.struct_path,
            calculator=self.model_path,
            device=self.device
        )
        system.set_electrons({'Si': 4})
        
        # 1. Calc Fermi level
        kmesh_efermi = [4, 4, 4] 
        efermi = system.get_efermi(kmesh=kmesh_efermi)
        # Check reasonable Fermi level for Silicon (around -8.5 eV in this model)
        assert abs(efermi - -8.5588) < 1e-5
        
        # 2. Calc Optical Conductivity
        omegas = np.linspace(0.1, 5.0, 50) # Coarse grid for speed
        kmesh_opt = [6, 6, 6] # Coarse mesh for speed
        
        # Use JIT method as in notebook example
        sigma = system.accond.compute(
            omegas=omegas,
            kmesh=kmesh_opt,
            eta=0.05,
            direction='xx',
            broadening='lorentzian',
            method='jit'
        )
        ref_sig_real =  torch.tensor([0.00837816, 0.00843790, 0.00854087, 0.00868832, 0.00888322, 0.00912982,
        0.00943376, 0.00980245, 0.01024545, 0.01077522, 0.01140800, 0.01216525,
        0.01307577, 0.01417897, 0.01553008, 0.01720899, 0.01933576, 0.02209974,
        0.02581912, 0.03107581, 0.03906935, 0.05276508, 0.08213381, 0.18777525,
        0.45900596, 0.87531792, 0.51921803, 1.36088354, 0.70162997, 1.04471288,
        0.73618954, 0.77278873, 0.45280920, 1.03011782, 1.73190348, 1.72394500,
        2.84396866, 2.52134255, 3.99499332, 2.15141878, 0.81073068, 0.37168581,
        0.37063105, 0.36194618, 0.45951775, 0.37643697, 0.24729162, 0.35520490,
        0.11137389, 0.06938815], dtype=torch.float64)
        ref_sig_imag = torch.tensor([-0.01671449, -0.03351582, -0.05048011, -0.06769407, -0.08524915,
        -0.10324409, -0.12178744, -0.14100061, -0.16102185, -0.18201113,
        -0.20415677, -0.22768423, -0.25286827, -0.28005068, -0.30966546,
        -0.34227807, -0.37864700, -0.41982665, -0.46735068, -0.52358439,
        -0.59248141, -0.68146015, -0.80703235, -1.00943765, -0.96654951,
        -1.14575997, -0.83195691, -1.24833063, -0.29735416, -0.96977719,
        -0.18409819, -0.42162556, -0.86429959, -1.24271126, -1.41426199,
        -1.05522313, -0.69645477, -0.96722652,  1.56216140,  1.88118809,
         2.23282337,  1.46657767,  0.98069451,  1.01382815,  0.63370519,
         1.02075540,  0.69233259,  0.87053525,  0.71759076,  0.56351753],
       dtype=torch.float64)
        # Checks

        self.assertEqual(len(sigma), len(omegas))
        self.assertTrue(torch.is_complex(sigma))
        # Compare real part to reference
        diff_real = torch.abs(sigma.real - ref_sig_real).max()
        self.assertLess(diff_real, 1e-5)
        # Compare imaginary part to reference
        diff_imag = torch.abs(sigma.imag - ref_sig_imag).max()
        self.assertLess(diff_imag, 1e-5)
        
        # Real part (absorption) should be non-negative
        # allowing for small numerical noise, but generally >= 0
        real_part = sigma.real
        self.assertTrue(torch.all(real_part >= -1e-9))
        
        # Check loop method consistency
        sigma_loop = system.accond.compute(
            omegas=omegas,
            kmesh=kmesh_opt,
            eta=0.05,
            direction='xx',
            broadening='lorentzian',
            method='loop'
        )
        
        # Should be close
        diff = torch.abs(sigma - sigma_loop).max()
        self.assertLess(diff, 1e-6)


        sigma = system.accond.compute(
            omegas=omegas,
            kmesh=kmesh_opt,
            eta=0.05,
            direction='xx',
            broadening='gaussian',
            method='jit'
        )
        sigma_loop = system.accond.compute(
            omegas=omegas,
            kmesh=kmesh_opt,
            eta=0.05,
            direction='xx',
            broadening='gaussian',
            method='loop'
        )

        ref_sig_real_gauss = torch.tensor([    0.00000171,     0.00000000,     0.00000000,     0.00000000,
            0.00000000,     0.00000000,     0.00000000,     0.00000000,
            0.00000000,     0.00000000,     0.00000000,     0.00000000,
            0.00000000,     0.00000000,     0.00000000,     0.00000000,
            0.00000000,     0.00000000,     0.00000000,     0.00000000,
            0.00000000,     0.00000009,     0.00080163,     0.12637314,
            0.42220475,     0.97778191,     0.37357975,     1.65309522,
            0.64873583,     1.20423554,     0.80707162,     0.63649152,
            0.16754380,     0.85413168,     1.88318838,     1.80086702,
            2.92473283,     2.87270292,     4.87694003,     2.21466861,
            0.60885423,     0.06020150,     0.27886417,     0.29327497,
            0.50874240,     0.41114855,     0.18927311,     0.40395584,
            0.02510403,     0.00047163], dtype=torch.float64)
        diff_gauss = torch.abs(sigma - sigma_loop).max()
        self.assertLess(diff_gauss, 1e-6)
        diff_real_gauss = torch.abs(sigma.real - ref_sig_real_gauss).max()
        self.assertLess(diff_real_gauss, 1e-5)

if __name__ == '__main__':
    unittest.main()
