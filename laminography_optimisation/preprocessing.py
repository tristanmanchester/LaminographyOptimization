"""Data preprocessing functions."""

import numpy as np
from cil.framework import DataContainer
from cil.processors import TransmissionAbsorptionConverter, Normaliser, Binner

def preprocess_data(data, image_key, angles):
    """
    Preprocess data by separating image types, normalizing and converting.
    
    Parameters:
    -----------
    data : ndarray
        The combined dataset containing flats, darks and projections
    image_key : ndarray
        Array indicating image types (0=tomography, 1=flat, 2=dark)
    angles : ndarray
        Array of rotation angles for each image
        
    Returns:
    --------
    absorption_data : DataContainer
        Preprocessed data ready for reconstruction
    projection_angles : ndarray
        Angles corresponding to the projections
    """
    # Find indices for each image type
    flat_indices = np.where(image_key == 1)[0]
    dark_indices = np.where(image_key == 2)[0]
    proj_indices = np.where(image_key == 0)[0]
    
    # Extract data for each image type
    flat_fields = data[flat_indices]
    dark_fields = data[dark_indices]
    projections = data[proj_indices]
    projection_angles = angles[proj_indices]
    
    # Create average flats and darks
    flat_mean = DataContainer(np.mean(flat_fields, axis=0), dtype=np.float32)
    dark_mean = DataContainer(np.mean(dark_fields, axis=0), dtype=np.float32)
    
    # Create and normalize projection data container
    proj_data = DataContainer(projections, dtype=np.float32)
    normalizer = Normaliser(flat_field=flat_mean, dark_field=dark_mean)
    normalized_data = normalizer(proj_data)
    
    # Convert transmission to absorption
    converter = TransmissionAbsorptionConverter()
    absorption_data = converter(normalized_data)
    
    return absorption_data, projection_angles

def apply_binning(data, binning_factor):
    """
    Apply binning to data using CIL's Binner processor.
    
    Parameters:
    -----------
    data : DataContainer
        Data to be binned
    binning_factor : int
        Factor by which to bin the data
        
    Returns:
    --------
    binned_data : DataContainer
        Binned data
    """
    if binning_factor <= 1:
        return data
    
    # Create binning ROI
    roi = {
        'horizontal': (None, None, binning_factor),
        'vertical': (None, None, binning_factor)
    }
    
    # Create and apply binner
    binner = Binner(roi)
    binner.set_input(data)
    binned_data = binner.get_output()
    
    return binned_data
