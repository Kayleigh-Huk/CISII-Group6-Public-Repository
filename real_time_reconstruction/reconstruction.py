import numpy as np
from math import pi, pow
import csv
from spatialmath.base import exp2r
# sensors are defined from the tip of the needle

class ShapeSensingStylet:
    def __init__(self, num_aa : int, num_channels : int, aa_locations : np.ndarray, stylet_diameter : float, stylet_length : float, P_ratio : float, E_modulus : int) -> None:
        self.num_aa = num_aa
        self.num_channels = num_channels
        self.stylet_diameter = stylet_diameter
        self.stylet_length = stylet_length
        self.aa_locations = aa_locations
        self.aa_locations_from_tip = np.ones_like(self.aa_locations)*self.stylet_length - self.aa_locations
        self.E_mod = E_modulus
        self.P_rat = P_ratio
        self.B_matrix = self.get_B_matrix(P_ratio, E_modulus)
        self.insertion_depth = self.stylet_length
        self.ds = 0.0005
        self.reference = None
        self.inserted_aa_inds = np.arange(self.num_aa)
        self.num_inserted = self.num_aa

    def set_reference(self, ref_wave_data):
        self.reference = ref_wave_data

    def set_insertion_depth(self, insertion_depth):
        self.insertion_depth = insertion_depth
        self.update_inserted_aa()
    
    def update_inserted_aa(self):
        self.inserted_aa_inds = np.where(self.aa_locations_from_tip < self.insertion_depth)[0]
        self.num_inserted = len(self.inserted_aa_inds)

    def get_B_matrix(self, P_ratio : float, E_modulus : int) -> np.ndarray:
        Ibend = pow((pi*self.stylet_diameter), 4) / 64
        Gmod = E_modulus / (2*(1+P_ratio))
        Jtorsion = pow((pi * self.stylet_diameter), 4) / 32

        Bstiff = E_modulus*Ibend
        Btorsion = Gmod * Jtorsion
        B_mat = np.diag([Bstiff, Bstiff, Btorsion])

        return B_mat

    def get_wave_data(self, filename : str, col_skip : int = 6, header_skip : int = 8, reference : bool = False, temp_comp : bool = True) -> np.ndarray:
        loc_inds = np.arange(header_skip, (self.num_channels+2)*self.num_aa*col_skip+header_skip, col_skip)

        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, 'excel-tab')
            c = 0
            waves = np.zeros((200, (self.num_channels+2)*self.num_aa), dtype=np.float128)
            for line in reader:
                skip = False
                if c >= 200:
                    break
                line_waves = np.array(line)[loc_inds].astype(np.float128)
                for i in range(loc_inds.shape[0]):
                    if (line_waves[i] == '0.000000000000E+0') and (i != 0) and (i != 1):
                        skip = True
                
                if not skip:
                    waves[c, :] = line_waves
                    c += 1
        
        waves = np.mean(waves, axis=0)
        waves = waves.reshape((self.num_channels+2, self.num_aa))
   
        inserted_waves = np.zeros((self.num_channels+2, self.num_inserted))
        inserted_waves = waves[:, self.inserted_aa_inds]
        inserted_waves = inserted_waves[[1,2,3,4,5], :]
        
        if not reference:
            inserted_waves = inserted_waves - self.reference
        
        if temp_comp:
            for i in range(self.num_inserted):
                inserted_waves[:, i] = inserted_waves[:, i] - np.mean(inserted_waves[:, i])
      
        return inserted_waves

    def get_measured_curvatures(self, waveshifts : np.ndarray, Cs : np.ndarray) -> np.ndarray:
        # use calibration matrices to compute reconstructed wavelength at each active area
        k_recons = Cs @ waveshifts.T.reshape((14, 7, -1))
        k_recons = k_recons.reshape((14,2))
        # add 0 to create z axis curvature
        k_recons = np.append(k_recons, np.zeros((14,1)) , axis=1)
        
        return k_recons
    
    def get_shape(self, curvatures):
        N = int(self.insertion_depth / self.ds) + 1
        curvatures = curvatures.reshape((-1,3)).repeat(N, axis=0)
        s = np.arange(N) * self.ds
        P_mat, R_mat = self.integratePose_wv(curvatures, s, self.stylet_length - self.insertion_depth)
        
        return P_mat, R_mat

    def integratePose_wv(self,
        wv, s: np.ndarray = None, s0: float = 0, ds: float = None,
        R_init: np.ndarray = np.eye( 3 ) ):
        """ Integrate angular deformation to get the pose of the needle along it's arclengths

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
        N = wv.shape[ 0 ]
        pmat = np.zeros( (N, 3) )
        Rmat = np.expand_dims( np.eye( 3 ), axis=0 ).repeat( N, axis=0 )
        Rmat[ 0 ] = R_init

        # get the arclengths
        if (s is None) and (ds is not None):
            s = s0 + np.arange( N ) * ds
        elif s is not None:
            pass
        else:
            raise ValueError( "Either 's' or 'ds' must be used, not both." )

        # else

        # integrate angular deviation vector in order to get the pose
        for i in range( 1, N ):
            Rmat[ i ] = Rmat[ i - 1 ] @ exp2r( self.ds * np.mean( wv[ i - 1:i ], axis=0 ) )
            #print(exp2r(self.ds * np.mean( wv[ i - 1:i ], axis=0 )))
            e3vec = Rmat[ :i + 1, :, 2 ].T  # grab z-direction coordinates

            if i == 1:
                pmat[ i ] = pmat[ i - 1 ] + Rmat[ i, :, 2 ] * self.ds
                #print(Rmat[ i, :, 2 ])
            else:
                pmat[ i ] = self.simpson_vec_int( e3vec, self.ds )
            #print(pmat[i])
            
        # for

        return pmat, Rmat
    
    def rotz(self, t: float ) -> np.ndarray:
        """ Rotation matrix about z-axis"""
        return np.array(
                [ [ np.cos( t ), -np.sin( t ), 0 ], [ np.sin( t ), np.cos( t ), 0 ], [ 0, 0, 1 ] ] )

    def simpson_vec_int(self, f: np.ndarray, dx: float ) -> np.ndarray:
        """ Implementation of Simpson vector integration

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

        # TODO: non-uniform dx integration

        # perform the integration
        if num_intervals == 2:  # base case 1
            int_res = dx / 3 * np.sum( f[ :, 0:3 ] * [ [ 1, 4, 1 ] ], axis=1 )
            return int_res

        # if
        elif num_intervals == 3:  # base case 2
            int_res = 3 / 8 * dx * np.sum( f[ :, 0:4 ] * [ [ 1, 3, 3, 1 ] ], axis=1 )
            return int_res

        # elif

        else:
            int_res = np.zeros( (f.shape[ 0 ]) )

            if num_intervals % 2 != 0:
                int_res += 3 / 8 * dx * np.sum( f[ :, -4: ] * [ [ 1, 3, 3, 1 ] ], axis=1 )
                m = num_intervals - 3

            # if
            else:
                m = num_intervals

            # else

            int_res += dx / 3 * (f[ :, 0 ] + 4 * np.sum( f[ :, 1:m:2 ], axis=1 ) + f[ :, m ])

            if m > 2:
                int_res += dx / 3 * 2 * np.sum( f[ :, 2:m:2 ], axis=1 )

            # if

        # else

        return int_res


            
        