from calibration import CalibrationExperiment
from reconstruction import ShapeSensingStylet
from math import pi
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def plot_3d(shape1, shape2):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    ax.plot(shape1[:, 0], shape1[:, 1], shape1[:, 2])
    ax.plot(shape2[:, 0], shape2[:, 1], shape2[:, 2])
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    return fig, ax

def axisEqual3D( ax ):
    """ taken from online """
    extents = np.array( [ getattr( ax, 'get_{}lim'.format( dim ) )() for dim in 'xyz' ] )
    sz = extents[ :, 1 ] - extents[ :, 0 ]
    centers = np.mean( extents, axis=1 )
    maxsize = max( abs( sz ) )
    r = maxsize / 2
    for ctr, dim in zip( centers, 'xyz' ):
        getattr( ax, 'set_{}lim'.format( dim ) )( ctr - r, ctr + r )

    
    
if __name__ == '__main__':
    gt_curvatures = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    gt_angles = np.array([0, pi/2])

    stylet = ShapeSensingStylet(14, 5, np.arange(20, 300, 20)*1e-3, 0.00145, 0.3, 0.33, 83e9)
    
    cal_path = 'cal_dataset/'
    depth = 0.25

    cal = CalibrationExperiment(gt_curvatures, gt_angles, stylet)
    
    Cs, weights = cal.perform_calibration(cal_path, depth)  
    
    path = input("Enter curvature file: ")
    gtx = float(input("Enter ground truth XZ bending: "))
    gty = float(input("Enter ground truth YZ bending: "))

    p, r = cal.calc_shape_from_data(path)
    gt_w = np.array([gty, gtx, 0])
    pmat_gt, rmat_gt = stylet.get_shape(gt_w)

    err = la.norm(pmat_gt[:, :] - p[:, :], axis=1)
    sqerr = np.square(err)
    mse = np.mean(sqerr)
    rmse = np.sqrt(mse)

    print('Tip Error:', round(la.norm(pmat_gt[-1] - p[-1])*1e3, 3), 'mm')
    print('RMSE:', round(rmse*1e3, 3), 'mm')

    fig, ax = plot_3d(pmat_gt*1e3, p*1e3)
    ax.plot(
        0,
        0,
        50,
        'r*',
        label="insertion_point"
    )
    axisEqual3D(ax)
    plt.title('0.75 Curve Reconstruction')

    plt.show()