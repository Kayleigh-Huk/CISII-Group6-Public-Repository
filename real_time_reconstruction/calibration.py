import numpy as np
import csv
import numpy.linalg as la
from math import sin, cos, pi
from reconstruction import ShapeSensingStylet
import matplotlib.pyplot as plt

class CalibrationExperiment:
    def __init__(self, gt_curvatures : np.ndarray, gt_angles : np.ndarray, stylet : ShapeSensingStylet, clinical_threshold : float = 1.5, num_trials : int = 5):
        self.gt_curvatures = gt_curvatures
        self.gt_angles = gt_angles
        self.num_curvatures = gt_curvatures.shape[0]
        self.num_angles = gt_angles.shape[0]
        self.stylet = stylet
        self.clinical_threshold = clinical_threshold
        self.num_trials = num_trials

        self.cal_mats = None
        self.weightings = None

    def perform_calibration(self, fbg_data_filepath : str, insertion_depth : float):
        self.stylet.set_insertion_depth(insertion_depth)
        reference_wavelengths = self.get_trial_data(fbg_data_filepath, 0, 0, reference=True)

        self.stylet.set_reference(reference_wavelengths)

        calibration_shifts = self.get_wave_shifts(fbg_data_filepath)
        kappa_mat = self.get_k_matrix()
        
        weight_mat = self.get_w_matrix()

        C_mats, aa_weight_mat = self.get_calibration_matrix(kappa_mat, weight_mat, calibration_shifts)

        self.cal_mats = C_mats
        self.weightings = aa_weight_mat

        return C_mats, aa_weight_mat

    # get kappa matrix
    # column 0: rotation in xz plane
    # column 1: rotation in yz plane
    def get_k_matrix(self):
        k = np.zeros((self.num_curvatures*self.num_angles+1, 2), dtype=np.float64)
        for i in range(self.num_curvatures):
            for j in range(self.num_angles):
                k[i*self.num_angles+j, 0] = self.gt_curvatures[i] * cos(self.gt_angles[j])
                k[i*self.num_angles+j, 1] = self.gt_curvatures[i] * sin(self.gt_angles[j])
        k[-1] = [0,0]

        return k

    # get square weights matrix for weighted least squares
    def get_w_matrix(self):
        trials = np.ones((self.num_angles*self.num_curvatures+1,))
        weights = np.ones((self.num_angles*self.num_curvatures+1,))
        for i in range(self.num_curvatures):
            for j in range(self.num_angles):
                trials[i*self.num_angles+j] = self.gt_curvatures[i]
        trials[-1] = 0
        inds = np.where(trials > self.clinical_threshold)
        weights[inds] = 0.05
        w = np.diag(weights)

        return w

    def get_trial_data(self, path : str, curve : float, angle : int, reference : bool = False) -> np.ndarray:
        trial = np.zeros((self.num_trials, self.stylet.num_channels, self.stylet.num_inserted))
        for k in range(self.num_trials):
            fname = f'{path}{curve}-{angle}-{k+1}'
            trial[k, :, :] = self.stylet.get_wave_data(fname, reference=reference, temp_comp=(not reference)) 

        return np.mean(trial, 0) 

    def get_wave_shifts(self, fbg_data_filepath : str):
        wvs = np.zeros((self.num_angles*self.num_curvatures+1, self.stylet.num_channels, self.stylet.num_inserted), dtype=np.float64)
        
        for i in range(self.num_curvatures):
            for j in range(self.num_angles):    
                wvs[i*self.num_angles+j,:] = self.get_trial_data(fbg_data_filepath, self.gt_curvatures[i], round(self.gt_angles[j]*180/pi))
        wvs[-1,:] = self.get_trial_data(fbg_data_filepath, 0, 0)
        shifts = np.zeros((self.num_curvatures*self.num_angles+1, self.stylet.num_channels, self.stylet.num_inserted), dtype=np.float64)
        for i in range(self.num_angles*self.num_curvatures+1):
            shifts[i, :, :] = wvs[i].reshape((self.stylet.num_channels, self.stylet.num_inserted))

        return shifts

    def get_calibration_matrix(self, k : np.array, W : np.array, shifts : np.array):
        Cs = np.zeros((self.stylet.num_inserted, 2, self.stylet.num_channels))
        k_ests = np.zeros((self.stylet.num_inserted, self.num_curvatures*self.num_angles+1, 2))
        MSE = np.zeros((self.stylet.num_inserted,))

        for i in range(self.stylet.num_inserted):
            # calibration for each AA       
            aa = shifts[:,:,i]
            
            A = (aa.T @ W @ aa)
            b = (aa.T @ W @ k)
            C = la.lstsq(A, b, rcond=None)[0]
            Cs[i,:,:] = C.T
            k_est = aa @ C
            k_ests[i,:,:] = k_est
            err = abs(k-k_est)
            MSE[i] = np.mean(err**2)

        weights = 1 / MSE
        weights = weights/sum(weights)

        return Cs, weights
    
    def calc_shape_from_data(self, fbg_filepath : str):
        kxz_val = 0
        kyz_val = 0
        for i in range(self.stylet.num_inserted):
            # calibration for each AA       
            aa = self.stylet.get_wave_data(fbg_filepath)
            aa = aa[:, i]
            k_est = self.cal_mats[i] @ aa

            kxz_val += k_est[0] * self.weightings[i]
            kyz_val += k_est[1] * self.weightings[i]
                
            w_init_val = np.array([kyz_val, kxz_val, 0])
            pmat_val, rmat_val = self.stylet.get_shape(w_init_val)

        return pmat_val, rmat_val

if __name__ == '__main__':
    gt_curvatures = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    gt_angles = np.array([0, pi/2])

    stylet = ShapeSensingStylet(14, 5, np.arange(20, 300, 20)*1e-3, 0.00145, 0.3, 0.33, 83e9)
    
    cal_path = 'cal_dataset/'
    depth = 0.25

    cal = CalibrationExperiment(gt_curvatures, gt_angles, stylet)
    
    Cs, weights = cal.perform_calibration(cal_path, depth)
    
    

