import numpy as np
from math import pi, pow
from spatialmath.base import exp2r

from io_methods import read_parameter_json, read_trial_data

class Stylet:
    def __init__(self, parameter_file : str, calibrated : bool) -> None:
        """
        Read in stylet parameters from JSON file and set as attributes.

        Parameters
        ----------
        parameter_file : string
            The name of the JSON file containing the stylet parameters.
        calibrated : bool
            True if the file contains the calibration matrices for the stylet.

        Raises
        ------
        ValueError
            Raises value error if a required parameter is missing from the file.

        Returns
        -------
        None
        """
        # read in all attributes from parameter file based on calibration status
        self.reference = None
        if calibrated:
            self.stylet_length, self.stylet_diameter, self.emod, \
            self.pratio, self.num_aa, self.num_channels, self.aa_locations, \
            self.ref_filepath, self.cal_matrices, self.aa_weights = read_parameter_json(parameter_file, calibrated=calibrated)
            # set the reference wavelength 
            self.reference = read_trial_data(self.ref_filepath, 0, 0, self.num_aa, self.num_channels)
            # convert to numpy arrays
            self.cal_matrices = np.array(self.cal_matrices)
            self.aa_weights = np.array(self.aa_weights)
        else:
            self.stylet_length, self.stylet_diameter, self.emod, \
            self.pratio, self.num_aa, self.num_channels, \
            self.aa_locations = read_parameter_json(parameter_file, calibrated=calibrated)
            self.ref_filepath, self.cal_matrices, self.aa_weights = (None, None, None)

        # get active area locations measured from stylet tip
        self.aa_locations_from_tip = np.ones_like(self.aa_locations)*self.stylet_length - self.aa_locations
        # compute the stiffness matrix
        self.B_matrix = self.get_B_matrix()
        # initialize the current depth attributes to no insertion
        self.insertion_depth = 0
        self.inserted_aa_inds = None
        self.num_inserted = 0
    
    def get_B_matrix(self) -> np.ndarray:
        """
        Original Author (MATLAB): Jacynthe Francoeur
        Translated Author: Kayleigh Huk

        Calculate the stiffness matrix based on the material attributes.

        Parameters
        ----------
        None

        Returns
        -------
        B_mat : np.ndarray of shape (3,3)
            The stiffness matrix of the material defined by the Young's modulus value
            and the Poisson ratio provided.
        """
        # compute the bending moment of inertia for a circular cross-section
        Ibend = pow((pi*self.stylet_diameter), 4) / 64

        # compute the shear modulus
        Gmod = self.emod / (2*(1+self.pratio))

        # compute the polar moment of inertia for a circular cross-section
        Jtorsion = pow((pi * self.stylet_diameter), 4) / 32

        # stiffness and torsional values
        Bstiff = self.emod*Ibend
        Btorsion = Gmod * Jtorsion

        # diagonal stiffness matrix
        B_mat = np.diag([Bstiff, Bstiff, Btorsion])

        return B_mat

    def set_reference(self, ref_wave_data):
        """
        Setter method for the reference wavelengths.

        Parameters
        ----------
        ref_wave_data : np.ndarray of shape (self.num_channels, self.num_aa)
            The reference wavelengths for the stylet.

        Returns
        ----------
        None
        """
        self.reference = ref_wave_data

    def update_insertion_depth(self, insertion_depth):
        """
        Update the current insertion depth and re-compute the inserted active areas using
        the active area locations from the tip of the needle.
        
        Parameters
        ----------
        insertion_depth : float
            The current depth of the needle measured from the tip in meters.

        Returns
        ----------
        None
        """
        self.insertion_depth = insertion_depth
        self.inserted_aa_inds = np.where(self.aa_locations_from_tip < self.insertion_depth)[0]
        self.num_inserted = len(self.inserted_aa_inds)

    def get_constant_curvature_shape(self, waveshifts : np.ndarray) -> np.ndarray:
        """
        Compute the shape for a constant C-shape stylet curvature.

        Parameters
        ----------
        waveshifts : np.ndarray of shape (self.num_channels, self.num_aa)
            The waveshifts of the stylet to compute the shape for.

        Returns
        ---------- 
        P_mat : np.ndarray
            The position matrix which defines the shape of the stylet with respect to
            the needle base.
        R_mat : np.ndarray
            The rotation matrix which defined the rotation from the needle base frame
            along the stylet.
        """
        # use the calibration matrices to estimate the curvature at each active area using
        # weighted sum of each
        kxz_val = 0
        kyz_val = 0
        for i in range(self.num_inserted):
            # matrix multiplication for each active area       
            aa = waveshifts[:, i]
            k_est = self.cal_matrices[i] @ aa

            # weighted sum of curvature for each active area
            kxz_val += k_est[0] * self.aa_weights[i]
            kyz_val += k_est[1] * self.aa_weights[i]

        # final curvature value estimate       
        w_init_val = np.array([kyz_val, kxz_val, 0])

        # perform integration along the stylet length
        
        P_mat, R_mat = self.integrate_shape_from_w_init(w_init_val)
        
        return P_mat, R_mat

    def integrate_shape_from_w_init(self, w_init_val, R_init: np.ndarray = np.eye( 3 ) ):
        """ 
        Author: Dimitri Lezcano
        Integrate angular deformation to get the pose of the needle along it's arclengths

            :param wv: N x 3 angular deformation vector
            :param s: numpy array of arclengths to integrate
            :param ds: (Default = None) the arclength increments desired
            :param s0: (Default = 0) the initial length to start off with
            :param R_init: (Default = numpy.eye(3)) Rotation matrix of the inital pose

            :returns: pmat, Rmat
                - pmat: N x 3 position for the needle shape points in-tissue
                -Rmat: N x 3 x 3 SO(3) rotation matrices for
        """
        # set-up the containers
        ds = 0.0005
        N = int(self.insertion_depth / ds) + 1
        wv = w_init_val.reshape((-1,3)).repeat(N, axis=0)
        s = np.arange(N) * ds

        N = wv.shape[ 0 ]
        pmat = np.zeros( (N, 3) )
        Rmat = np.expand_dims( np.eye( 3 ), axis=0 ).repeat( N, axis=0 )
        Rmat[ 0 ] = R_init

        # integrate angular deviation vector in order to get the pose
        for i in range( 1, N ):
            Rmat[ i ] = Rmat[ i - 1 ] @ exp2r( ds * np.mean( wv[ i - 1:i ], axis=0 ) )
            e3vec = Rmat[ :i + 1, :, 2 ].T  # grab z-direction coordinates

            if i == 1:
                pmat[ i ] = pmat[ i - 1 ] + Rmat[ i, :, 2 ] * ds
            else:
                pmat[ i ] = self.simpson_vec_int( e3vec, ds )

        return pmat, Rmat
    
    def rotz(self, t: float ) -> np.ndarray:
        """ 
        Author: Dimitri Lezcano
        Rotation matrix about z-axis
        """
        return np.array(
                [ [ np.cos( t ), -np.sin( t ), 0 ], [ np.sin( t ), np.cos( t ), 0 ], [ 0, 0, 1 ] ] )

    def simpson_vec_int(self, f: np.ndarray, dx: float ) -> np.ndarray:
        """ 
        Implementation of Simpson vector integration

        Original Author (MATLAB): Jin Seob Kim
        Translated Author: Dimitri Lezcano

        Args:
            f:  m x n numpy array where m is the dimension of the vector and n is the dimension of the parameter ( n > 2 )
                    Integration intervals
            dx: float of the step size

        Return:
            numpy vector of shape (m,)
        """
        num_intervals = f.shape[ 1 ] - 1
        assert (num_intervals > 1)  # need as least a single interval


        # perform the integration
        if num_intervals == 2:  # base case 1
            int_res = dx / 3 * np.sum( f[ :, 0:3 ] * [ [ 1, 4, 1 ] ], axis=1 )
            return int_res

        elif num_intervals == 3:  # base case 2
            int_res = 3 / 8 * dx * np.sum( f[ :, 0:4 ] * [ [ 1, 3, 3, 1 ] ], axis=1 )
            return int_res

        else:
            int_res = np.zeros( (f.shape[ 0 ]) )

            if num_intervals % 2 != 0:
                int_res += 3 / 8 * dx * np.sum( f[ :, -4: ] * [ [ 1, 3, 3, 1 ] ], axis=1 )
                m = num_intervals - 3

            else:
                m = num_intervals

            int_res += dx / 3 * (f[ :, 0 ] + 4 * np.sum( f[ :, 1:m:2 ], axis=1 ) + f[ :, m ])

            if m > 2:
                int_res += dx / 3 * 2 * np.sum( f[ :, 2:m:2 ], axis=1 )

        return int_res


            
        