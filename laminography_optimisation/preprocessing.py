"""Data preprocessing functions."""

import numpy as np
from cil.framework import DataContainer
from cil.processors import Binner

def normalise_projection(projection, flat, dark, tolerance=1e-5):
    """
    Normalise a projection using flat and dark field images.
    
    Parameters:
    -----------
    projection : ndarray
        Projection data
    flat : ndarray
        Flat field data
    dark : ndarray
        Dark field data
    tolerance : float, optional
        Tolerance for division by zero
        
    Returns:
    --------
    normalized : ndarray
        Normalized projection
    """
    a = (projection - dark)
    b = (flat - dark)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = tolerance  # set to tolerance if 0/0 or inf
        
    return c.astype(np.float32)

def normalize_data(projections, flat, dark, tolerance=1e-5, verbose=True):
    """
    Normalize projections using flat and dark field images.
    
    Parameters:
    -----------
    projections : ndarray
        Array of projections
    flat : ndarray
        Flat field (mean)
    dark : ndarray
        Dark field (mean)
    tolerance : float, optional
        Tolerance for division by zero
    verbose : bool, optional
        Print verbose information
        
    Returns:
    --------
    normalized : ndarray
        Normalized projections
    """
    if verbose:
        print(f"Manual normalization of {projections.shape[0]} projections")
        print(f"Flat shape: {flat.shape}, Dark shape: {dark.shape}")
    
    # Normalize each projection
    normalized = np.zeros_like(projections, dtype=np.float32)
    
    for i in range(projections.shape[0]):
        normalized[i] = normalise_projection(projections[i], flat, dark, tolerance)
        
        # Print progress every 10%
        if verbose and i % (projections.shape[0] // 10) == 0:
            print(f"  Normalized {i}/{projections.shape[0]} projections")
    
    return normalized

def transmission_to_absorption(data, min_intensity=0.001, verbose=True):
    """
    Convert transmission data to absorption data using Beer-Lambert law.
    
    Parameters:
    -----------
    data : ndarray
        Transmission data
    min_intensity : float, optional
        Minimum intensity to prevent log(0)
    verbose : bool, optional
        Print verbose information
        
    Returns:
    --------
    absorption : ndarray
        Absorption data
    """
    if verbose:
        print(f"Converting transmission to absorption (min_intensity={min_intensity})")
    
    # Clip values to avoid issues with log(0)
    clipped = np.clip(data, min_intensity, None)
    
    # Apply -log to convert to absorption (Beer-Lambert law)
    absorption = -np.log(clipped)
    
    if verbose:
        print(f"  Transmission range: [{data.min():.6f}, {data.max():.6f}]")
        print(f"  Absorption range: [{absorption.min():.6f}, {absorption.max():.6f}]")
    
    return absorption

def preprocess_data(data, image_key, angles, verbose=True):
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
    verbose : bool, optional
        Whether to print detailed information, default is True
        
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
    
    if verbose:
        print(f"Flat field indices: {len(flat_indices)} found")
        print(f"  First few indices: {flat_indices[:5]}")
        print(f"Dark field indices: {len(dark_indices)} found")
        print(f"  First few indices: {dark_indices[:5]}")
        print(f"Projection indices: {len(proj_indices)} found")
        print(f"  First few indices: {proj_indices[:5]}")
    
    # Extract data for each image type
    flat_fields = data[flat_indices]
    dark_fields = data[dark_indices]
    projections = data[proj_indices]
    projection_angles = angles[proj_indices]
    
    if verbose:
        print(f"Flat fields shape: {flat_fields.shape}")
        print(f"Dark fields shape: {dark_fields.shape}")
        print(f"Projections shape: {projections.shape}")
        print(f"Projection angles shape: {projection_angles.shape}")
    
    # Check if we have flats and darks
    if len(flat_indices) == 0:
        raise ValueError("No flat field images found! Check the image key values.")
    
    if len(dark_indices) == 0:
        raise ValueError("No dark field images found! Check the image key values.")
    
    # Calculate mean of flat and dark fields
    flat_mean = np.mean(flat_fields, axis=0)
    dark_mean = np.mean(dark_fields, axis=0)
    
    if verbose:
        print(f"Flat mean shape: {flat_mean.shape}, dtype: {flat_mean.dtype}")
        print(f"Dark mean shape: {dark_mean.shape}, dtype: {dark_mean.dtype}")
    
    # Normalize the projections manually
    print("Normalizing data...")
    normalized_data = normalize_data(projections, flat_mean, dark_mean, verbose=verbose)
    
    if verbose:
        print(f"Normalized data shape: {normalized_data.shape}")
    
    # Convert transmission to absorption manually
    absorption_data = transmission_to_absorption(normalized_data, verbose=verbose)
    
    # Create DataContainer from absorption data
    absorption_container = DataContainer(absorption_data, dtype=np.float32)
    
    return absorption_container, projection_angles

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
