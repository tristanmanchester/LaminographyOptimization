"""Data loading and saving functions."""

import os
import numpy as np
from cil.framework import DataContainer
from cil.io import TIFFWriter
import cil.io.utilities

def load_data(file_path):
    """Load data, image keys, and angles from an NXS file."""
    # Load the raw data
    data = cil.io.utilities.HDF5_utilities.read(file_path, '/entry/imaging/data')
    
    # Load the image keys (0=tomography, 1=flat, 2=dark)
    image_key = cil.io.utilities.HDF5_utilities.read(file_path, '/entry/instrument/EtherCAT/image_key')
    
    # Load the angles (includes angles for flats/darks)
    angles = cil.io.utilities.HDF5_utilities.read(file_path, '/entry/imaging_sum/smaract_zrot')
    
    return data, image_key, angles

def save_reconstruction(reconstruction, output_path, as_16bit=True):
    """
    Save reconstruction as 16-bit TIFF files.
    
    Parameters:
    -----------
    reconstruction : ImageData
        Reconstructed volume
    output_path : str
        Path to save the reconstruction
    as_16bit : bool, optional
        Whether to save as 16-bit (True) or 32-bit (False), default is True
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to 16-bit if requested
    if as_16bit:
        # Scale to 16-bit range
        min_val = reconstruction.as_array().min()
        max_val = reconstruction.as_array().max()
        scaled_data = ((reconstruction.as_array() - min_val) /
                       (max_val - min_val) * 65535).astype(np.uint16)
        
        # Create scaled data container
        scaled_reconstruction = type(reconstruction)(scaled_data,
                                                  geometry=reconstruction.geometry,
                                                  dimension_labels=reconstruction.dimension_labels)
        
        # Save as TIFF
        TIFFWriter(scaled_reconstruction, output_path).write()
    else:
        # Save as is
        TIFFWriter(reconstruction, output_path).write()
