import numpy as np
import numpy.linalg as la
from math import sin, cos, pi

from stylet import ShapeSensingStylet

def main():
    pass

def get_trial_data(self, path : str, curve : float, angle : int, reference : bool = False) -> np.ndarray:
    trial = np.zeros((self.num_trials, self.stylet.num_channels, self.stylet.num_inserted))
    for k in range(self.num_trials):
        fname = f'{path}{curve}-{angle}-{k+1}'
        trial[k, :, :] = self.stylet.get_wave_data(fname, reference=reference, temp_comp=(not reference)) 

    return np.mean(trial, 0) 

def append_cal_matrices_to_parameter_json(filename : str, cal_matrices : np.ndarray):
    pass

def write_validation_results(te_tot, rmse_tot):
    print('Tip Error')
    print('Values: \n', te_tot)
    print('Mean TE: \n', np.mean(te_tot)*1e3, 'mm')
    print('TE STD: \n', np.std(te_tot)*1e3, 'mm')
        
    print('\nRoot-mean-square Error')
    print('Values: \n', rmse_tot)
    print('Mean RMSE: \n', np.mean(rmse_tot)*1e3, 'mm')
    print('RMSE STD: \n', np.std(rmse_tot)*1e3, 'mm')

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
    
def perform_validation(self, fbg_filepath, val_gt_curves, val_gt_angles):
    curves = val_gt_curves.shape[0]
    angles = val_gt_angles.shape[0]

    te_tot = np.zeros((curves*angles,self.num_trials))
    rmse_tot = np.zeros((curves*angles,self.num_trials))
    errs = np.zeros((self.stylet.num_inserted,curves*angles,self.num_trials))
    for k in range(curves):
        for a in range(angles):
            kcx_truth = cos(val_gt_angles[a]) * val_gt_curves[k] 
            kcy_truth = sin(val_gt_angles[a]) * val_gt_curves[k] 
            w_truth = np.array([kcy_truth, kcx_truth, 0])
            for t in range(self.num_trials):
                p_gt, r_gt = self.stylet.get_shape(w_truth)
                p_val, r_val = self.calc_shape_from_data(f'{fbg_filepath}{val_gt_curves[k]}-{round(val_gt_angles[a]*180/pi)}-{t+1}')

                te = p_gt[-1, :] - p_val[-1, :]
                te = la.norm(te)
                te_tot[k*angles+a, t] = te
                perr = la.norm(p_gt[:, :] - p_val[:, :], axis=1)
                sqerr = np.square(perr)
                mse = np.mean(sqerr)
                rmse = np.sqrt(mse)
                rmse_tot[k*angles+a, t] = rmse
    return te_tot, rmse_tot

if __name__ == '__main__':
    gt_curvatures = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    gt_angles = np.array([0, pi/2])

    stylet = ShapeSensingStylet(14, 5, np.arange(20, 300, 20)*1e-3, 0.00145, 0.3, 0.33, 83e9)
    
    cal_path = 'cal_dataset/'
    depth = 0.25

    cal = main(gt_curvatures, gt_angles, stylet)
    
    Cs, weights = cal.perform_calibration(cal_path, depth)

    te, rmse = cal.perform_validation('cal_dataset/', np.array([0.33, 0.66, 1.33]), gt_angles)

    cal.display_validation_results(te, rmse)

    
    

