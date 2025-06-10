import numpy as np
import numpy.linalg as la
from math import sin, cos, pi

import json
import csv

from stylet import Stylet
from io_methods import read_trial_data, read_parameter_json

def main(parameter_file, cal_path, val_path, insertion_depth, gt_cal_curvatures, gt_cal_angles, gt_val_curvatures, gt_val_angles, clinical_thresh : float = 2):
    stylet = Stylet(parameter_file, False)

    Cs, weights = perform_calibration(stylet, cal_path, insertion_depth, gt_cal_curvatures, gt_cal_angles, clinical_thresh)
    
    ref_file = f'{cal_path}0-0-1'
    append_cal_results_to_parameter_json(parameter_file, ref_file, Cs, weights)
    stylet.cal_matrices = Cs
    stylet.aa_weights = weights

    te, rmse = perform_validation(stylet, val_path, gt_val_curvatures, gt_val_angles)
    write_validation_results(cal_path, te, rmse, gt_val_curvatures, gt_val_angles)

    return

def append_cal_results_to_parameter_json(filename : str, reference_file : str, cal_matrices : np.ndarray, weightings : np.ndarray):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    
    data["Reference filepath"] = reference_file
    data["Calibration matrices"] = cal_matrices.tolist()
    data["Active area weights"] = weightings.tolist()

    with open(filename, 'w' ) as outfile:
        json.dump(data, outfile, indent=4)

    return

def write_validation_results(filepath, te_tot, rmse_tot, gt_val_curvatures, gt_val_angles, num_insertions = 5):
    with open(f'{filepath}validation_results.csv', 'w') as results:
        writer = csv.writer(results, dialect='excel-tab')

        te_mean = np.mean(te_tot)*1e3
        rmse_mean = np.mean(rmse_tot)*1e3
        te_outcome = 'Failed' if te_mean > 0.5 else 'Passed'
        rmse_outcome = 'Failed' if rmse_mean > 0.5 else 'Passed'

        writer.writerow(['Metric', 'Tip Error Results', 'Root-Mean-Square Error Results'])
        writer.writerow(['Overall Outcome', te_outcome, rmse_outcome])
        writer.writerow(['Mean (mm)', f'{te_mean:.4f}', f'{rmse_mean:.4f}'])
        writer.writerow(['Standard Deviation (mm):', f'{np.std(te_tot)*1e3:.4f}', f'{np.std(rmse_tot)*1e3:.4f}'])
        for i in range(gt_val_curvatures.shape[0]):
            for j in range(gt_val_angles.shape[0]):
                writer.writerow([f'Insertion {gt_val_curvatures[i]}-{round(gt_val_angles[j]*180/pi)}:', f'{te_tot[i*gt_val_angles.shape[0]+j]*1e3:.4f}', f'{rmse_tot[i*gt_val_angles.shape[0]+j]*1e3:.4f}'])

    return

def perform_calibration(stylet : Stylet, fbg_data_filepath : str, insertion_depth : float, curves : np.ndarray, angles : np.ndarray, clinical_thresh : float):
    reference_wavelengths = read_trial_data(fbg_data_filepath, 0, 0, stylet.num_aa, stylet.num_channels)
    stylet.set_reference(reference_wavelengths)

    calibration_shifts = get_wave_shifts(stylet, fbg_data_filepath, curves, angles)
    stylet.update_insertion_depth(insertion_depth)
    
    kappa_mat = get_k_matrix(curves, angles)
        
    weight_mat = get_w_matrix(curves, angles, clinical_thresh)

    C_mats, aa_weight_mat = get_calibration_matrix(stylet, kappa_mat, weight_mat, calibration_shifts)

    return C_mats, aa_weight_mat

def get_wave_shifts(stylet : Stylet, fbg_data_filepath : str, gt_curvatures : np.ndarray, gt_angles : np.ndarray):
    num_curvatures = gt_curvatures.shape[0]
    num_angles = gt_angles.shape[0]
    wvs = np.zeros((num_angles*num_curvatures+1, stylet.num_channels, stylet.num_aa), dtype=np.float64)
        
    for i in range(num_curvatures):
        for j in range(num_angles):    
            wvs[i*num_angles+j,:] = read_trial_data(fbg_data_filepath, gt_curvatures[i], round(gt_angles[j]*180/pi), stylet.num_aa, stylet.num_channels, stylet.reference)
    wvs[-1,:] = stylet.reference
    #wvs = wvs[[1,2,3,4,5], 2:]

    return wvs

    # get kappa matrix
    # column 0: rotation in xz plane
    # column 1: rotation in yz plane
def get_k_matrix(gt_curvatures : np.ndarray, gt_angles : np.ndarray):
    num_curvatures = gt_curvatures.shape[0]
    num_angles = gt_angles.shape[0]

    k = np.zeros((num_curvatures*num_angles+1, 2), dtype=np.float64)
    for i in range(num_curvatures):
        for j in range(num_angles):
            k[i*num_angles+j, 0] = gt_curvatures[i] * cos(gt_angles[j])
            k[i*num_angles+j, 1] = gt_curvatures[i] * sin(gt_angles[j])
    k[-1] = [0,0]

    return k

# get square weights matrix for weighted least squares
def get_w_matrix(gt_curvatures : np.ndarray, gt_angles : np.ndarray, clinical_threshold : float):
    num_curvatures = gt_curvatures.shape[0]
    num_angles = gt_angles.shape[0]

    trials = np.ones((num_angles*num_curvatures+1,))
    weights = np.ones((num_angles*num_curvatures+1,))
    for i in range(num_curvatures):
        for j in range(num_angles):
            trials[i*num_angles+j] = gt_curvatures[i]
    trials[-1] = 0
    inds = np.where(trials > clinical_threshold)
    weights[inds] = 0.05
    w = np.diag(weights)

    return w

def get_calibration_matrix(stylet : Stylet, k : np.array, W : np.array, shifts : np.array):
    Cs = np.zeros((stylet.num_inserted, 2, stylet.num_channels))
    k_ests = np.zeros((stylet.num_inserted, k.shape[0], 2))
    MSE = np.zeros((stylet.num_inserted,))

    for i in range(stylet.num_inserted):
        # calibration for each AA       
        aa = shifts[:,:,stylet.inserted_aa_inds[i]]
            
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
    
def perform_validation(stylet : Stylet, fbg_filepath : str, val_gt_curves : np.ndarray, val_gt_angles : np.ndarray, num_trials : int = 5):
    curves = val_gt_curves.shape[0]
    angles = val_gt_angles.shape[0]

    te_tot = np.zeros((curves*angles,))
    rmse_tot = np.zeros((curves*angles,))
    errs = np.zeros((stylet.num_inserted,curves*angles))
    waveshifts = get_wave_shifts(stylet, fbg_filepath, val_gt_curves, val_gt_angles)
    for k in range(curves):
        for a in range(angles):
            kcx_truth = cos(val_gt_angles[a]) * val_gt_curves[k] 
            kcy_truth = sin(val_gt_angles[a]) * val_gt_curves[k] 
            w_truth = np.array([kcy_truth, kcx_truth, 0])

            p_gt, r_gt = stylet.integrate_shape_from_w_init(w_truth)
            p_val, r_val = stylet.get_constant_curvature_shape(waveshifts[k*angles+a])

            te = p_gt[-1, :] - p_val[-1, :]
            te = la.norm(te)
            te_tot[k*angles+a] = te
            perr = la.norm(p_gt[:, :] - p_val[:, :], axis=1)
            sqerr = np.square(perr)
            mse = np.mean(sqerr)
            rmse = np.sqrt(mse)
            rmse_tot[k*angles+a] = rmse
    return te_tot, rmse_tot

if __name__ == '__main__':
    gt_cal_curvatures = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    gt_val_curvatures = np.array([0.33, 0.66, 1.33])
    gt_angles = np.array([0, pi/2])

    params = 'stylet_params/7CH_14AA_300.json'
    cal_path = 'cal_dataset/'
    depth = 0.25

    main(params, cal_path, cal_path, depth, gt_cal_curvatures, gt_angles, gt_val_curvatures, gt_angles)
    


    
    

