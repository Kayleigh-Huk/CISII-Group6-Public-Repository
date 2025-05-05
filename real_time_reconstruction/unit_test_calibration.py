import numpy as np
import unittest
import random
import time
from calibration import CalibrationExperiment
from reconstruction import ShapeSensingStylet

class TestCalibration(unittest.TestCase):
    def setUp(self):
        # wave shift = 0.22 (photoelastic coeff) * R (dist from center of rod) * kappa (curvature)
        R = 1e-4
        pe = 0.22
        self.kappas = np.array([[0.5, 0], [1.0, 0], [1.5, 0], [2, 0]])
        self.angles = np.array([0])
        self.weights = np.diag([1, 1, 1, 0.05])
        self.aa_loc = np.arange(20, 300, 20) *1e-3
        self.del_lambdas = np.zeros((4, 7, 14))
        self.del_lambdas_noisy = np.zeros((4, 7, 14))
        for i in range(len(self.kappas)):
            corr_shift = pe * self.kappas[i, 0] * R
            shifts = np.ones((7,14)) * corr_shift
            self.del_lambdas[i, :, :] = shifts

            sign = random.choice([-1, 1])
            noisy_shifts = np.ones((7,14)) * corr_shift
            for j in range(7):
                for k in range(14):
                    noise = random.randint(0, 100) * 1e-7
                    noisy_shifts[j, k] = noisy_shifts[j, k] + (sign*noise)
            self.del_lambdas_noisy[i, :, :] = noisy_shifts

        self.sty = ShapeSensingStylet(14, 7, self.aa_loc, 0.00145, 0.3, 0.33, 83*1e9)
        self.exp = CalibrationExperiment(self.kappas, self.angles, 4, 1, self.sty)
            
    def testCalcCalMatrixNoNoise(self):
        Cs, weights = self.exp.get_calibration_matrix(self.kappas, self.weights, self.del_lambdas)
        
        for i in range(len(Cs)):
            k_recon = Cs[i] @ self.del_lambdas[:, :, i].T

            for i in range(4):
                self.assertAlmostEqual(k_recon.T[i, 0], self.kappas[i, 0], 14)
                self.assertAlmostEqual(k_recon.T[i, 1], self.kappas[i, 1], 14)

    def testCalcCalMatrixNoisy(self):
        Cs, weights = self.exp.get_calibration_matrix(self.kappas, self.weights, self.del_lambdas_noisy)
        
        for i in range(len(Cs)):
            k_recon = Cs[i] @ self.del_lambdas_noisy[:, :, i].T
            for i in range(4):
                self.assertAlmostEqual(k_recon.T[i, 0], self.kappas[i, 0], 12)
                self.assertAlmostEqual(k_recon.T[i, 1], self.kappas[i, 1], 12)

if __name__ == '__main__':
    unittest.main()