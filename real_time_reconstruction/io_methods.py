import csv
import json
import numpy as np

def read_parameter_json(filename : str, calibrated : bool) -> list:
    """
    Read in stylet parameters from a JSON file.

    Parameters
    ----------
    filename : string
        The name of the JSON file containing the stylet parameters.
    calibrated : bool
        True if the file contains the calibration matrices for the stylet.

    Raises
    ------
    ValueError
        Raises value error if a required parameter is missing from the file.

    Returns
    -------
    attributes : list
        The parameters parsed from the JSON file in the following order:
            [length (float), diameter (float), emod (float), pratio (int), 
             num_aa (int), num_channels (int), aa_locations (list of floats)].
    """
    # parse provided JSON file
    with open(filename, 'r') as json_file:
        data = json.load(json_file)

    # keys for a correct parameter file
    corr_keys = ["Length (m)", "Stylet diameter (m)", "Young's modulus", \
            "Poisson ratio", "Active areas per core", "Total channels", \
            "Active area locations from base (m)"]

    # check all correct keys are present and add all values to attributes list
    attributes = []
    for key in corr_keys:
        if key in data.keys():
            attributes.append(data[key])
        else:
            raise ValueError(f"Required parameter not found in provided JSON file: {key}")
    
    # if stylet is calibrated, add the calibration matrices to the attributes list
    if calibrated:
        cal_key = "Calibration matrices"
        if cal_key in data.keys():
            attributes.append(data[cal_key])
        else:
            raise ValueError(f"{cal_key} not found. Please calibrate stylet using \'calibration.py\' first.")
    
    return attributes

def read_interrogator_csv(filename : str, num_aa : int, num_channels : int, reference_wavelengths : np.ndarray = None) -> np.ndarray:
    """
    Read in the average wavelength data from the csv file produced by the FemtoSense GUI.

    Parameters
    ----------
    filename : string
        The name of the csv file containing the raw interrogator data.
    num_aa : int
        The number of active areas along each channel in the stylet.
    num_channels : int
        The number of channels in the stylet.
    reference_wavelengths : np.ndarray
        Default value = None
        The reference wavelengths to use as the baseline to compute the wavelength shifts.
        Temperature compensation will only be performed if a reference is provided.

    Raises
    ------
    ValueError
        Raises value error if the file provided does not contain 200 valid readings.

    Returns
    -------
    waves : np.ndarray of shape (num_channels, num_aa)
        The average wavelengths (if reference_wavelengths = None) or wavelength shifts from
        the first 200 valid readings from the file provided.
    """
    # csv file format values from FemtoSenseGUI
    header_skip = 8
    col_skip = 6

    # csv indices where wavelength data is stored
    wave_inds = np.arange(header_skip, num_channels*num_aa*col_skip+header_skip, col_skip)

    # get wavelength data into an array of shape: (num_channels, num_aa)
    waves = get_wavelength_readings(filename, wave_inds)
    waves = np.mean(waves, axis=0)
    waves = waves.reshape((num_channels, num_aa))
    
    # if the reference wavelengths are provided, compute wavelength shifts and then compensate for temperature
    if type(reference_wavelengths) != type(None):
        waves = waves - reference_wavelengths

        # for each active area, subtract the mean wavelength shifts from each channel at that active area
        # to remove effect of temperature
        for i in range(num_aa):
            waves[:, i] = waves[:, i] - np.mean(waves[:, i])
      
    return waves

def get_wavelength_readings(filename : str, wave_data_indices : np.ndarray) -> np.ndarray:
    """
    Helper method for read_interrogator_csv.
    Parse the interrogator file for the raw wavelength data until 200 valid readings are
    found.

    Parameters
    ----------
    filename : string
        The name of the csv file containing the raw interrogator data.
    wave_data_indices : np.ndarray of shape (num_channels*num_aa) 
        The indices of the raw wavelength data within a line of the csv file.
   
    Raises
    ------
    ValueError
        Raises value error if the file provided does not contain 200 valid readings.

    Returns
    -------
    waves : np.ndarray of shape (200, num_channels*num_aa)
        The wavelength data for the first 200 valid readings in the provided csv file.
    """
    # parse provided csv file from interrogator
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, 'excel-tab')
        c = 0

        # initialize array to hold 200 wavelength readings
        waves = np.zeros((200, wave_data_indices.shape[0]), dtype=np.float128)
        for line in reader:
            # stop once 200 valid readings found
            skip = False
            if c >= 200:
                break

            # read in only the wavelength data stored at provided indices
            line_waves = np.array(line)[wave_data_indices].astype(np.float128)

            # check if any of the wavelength data was invalid - indicated by a reading of 0
            if np.where(line_waves == 0)[0].shape[0] != 0:
                continue

            # add to waves array if wavelength data is valid
            waves[c, :] = line_waves
            c += 1
        
        if c != 200:
            raise ValueError(f"File provided ({filename}) does not contain 200 valid trials.")
    
    return waves

def write_parameter_json(filename, attributes):
    """
    Write the stylet parameters to a JSON file.

    Parameters
    ----------
    filename : string
        The name of the json file to write the parameters.
    attributes : list
        The parameters to write to the JSON file in the following order:
            [length (float), diameter (float), emod (float), pratio (int), 
             num_aa (int), num_channels (int), aa_locations (list of floats)].
    Raises
    ------
    ValueError
        Raises value error if the attributes list provided is of the wrong size.

    Returns
    -------
    None
    """
    # define keys for parameter JSON file
    keys = ["Length (m)", "Stylet diameter (m)", "Young's modulus", \
            "Poisson ratio", "Active areas per core", "Total channels", \
            "Active area locations from base (m)"]
    
    # check that correct number of attributes is provided 
    corr_num_keys = 7
    act_num_keys = len(attributes)
    if act_num_keys != corr_num_keys:
        raise ValueError(f"{act_num_keys} items found in attributes list, but only {corr_num_keys} expected.")
    
    # create a dictionary for the attributes
    data = {}
    for i in range(corr_num_keys):
        data[keys[i]] = attributes[i]

    # write the dictionary to the json file provided
    with open(filename, 'w' ) as outfile:
        json.dump(data, outfile, indent=4)