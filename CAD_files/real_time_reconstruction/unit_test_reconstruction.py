import unittest
import numpy as np
import random
from calibration import CalibrationExperiment
from reconstruction import ShapeSensingStylet
from time import time

class TestReconstruction(unittest.TestCase):
    def setUp(self):
        # wave shift = 0.22 (photoelastic coeff) * R (dist from center of rod) * kappa (curvature)
        R = 1e-4
        pe = 0.22
        self.kappas = np.array([[0.5, 0], [1.0, 0], [1.5, 0], [2, 0]])
        self.angles = np.array([0])
        self.weights = np.diag([1, 1, 1, 0.05])
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
        
        self.aa_loc = np.arange(20, 300, 20) *1e-3

        self.stylet = ShapeSensingStylet(14, 7, self.aa_loc, 0.00145, 0.3, 0.33, 83*1e9)
        self.stylet.insertion_depth = .250

        self.cal = CalibrationExperiment(self.kappas, self.angles, 4, 1, self.stylet)

        self.Cs_no_noise, self.weights_no_noise = self.calcCalMatrixNoNoise()
        self.Cs_noisy, self.weights_noisy = self.calcCalMatrixNoisy()
            
    def calcCalMatrixNoNoise(self):
        Cs, weights = self.cal.get_calibration_matrix(self.kappas, self.weights, self.del_lambdas)
        #print('test\n', Cs[0] @ self.del_lambdas[0, :, 0], '\n')
        return Cs, weights

    def calcCalMatrixNoisy(self):
        Cs, weights = self.cal.get_calibration_matrix(self.kappas, self.weights, self.del_lambdas_noisy)
        
        return Cs, weights

    def testGetMeasuredCurvatures(self):
        w = self.stylet.get_measured_curvatures(self.del_lambdas[0], self.Cs_no_noise)

        w_truth = np.zeros((14,3))
        w_truth[:, 0] = 0.5

        result = np.allclose(w_truth, w)

        self.assertTrue(result)

    def testGetWaveData(self):
        s = time()
        w = self.stylet.get_wave_data('5.1.25 test/validate/0.33-0-1', 6, 8, reference=True)
        e = time()
    
    def testGetOptimParams(self):
        pass
        #self.stylet.get_opt_params(0.5)
    
    def testGetShape(self):
        w = self.stylet.get_measured_curvatures(self.del_lambdas[0], self.Cs_no_noise)

        self.stylet.get_shape(w)

if __name__ == '__main__':
    unittest.main()