"""Acquisition geometry functions."""

import numpy as np
from cil.framework import AcquisitionGeometry

def create_geometry(projection_angles, tilt_angle, num_pixels_x, num_pixels_y,
                   detector_pixel_size, detector_binning=1):
    """
    Create a laminography acquisition geometry with tilt.
    
    Parameters:
    -----------
    projection_angles : ndarray
        Array of angles for each projection
    tilt_angle : float
        Tilt angle of the rotation axis in degrees
    num_pixels_x : int
        Number of detector pixels in x-direction
    num_pixels_y : int
        Number of detector pixels in y-direction
    detector_pixel_size : float
        Size of detector pixels in microns
    detector_binning : int, optional
        Detector binning factor, default is 1 (no binning)
        
    Returns:
    --------
    ag : AcquisitionGeometry
        Acquisition geometry object for reconstruction
    """
    # Convert tilt angle to radians
    tilt = np.radians(tilt_angle)
    
    # Calculate effective pixel size
    effective_pixel_size = detector_pixel_size * detector_binning
    
    # Create parallel beam geometry with tilted rotation axis
    ag = AcquisitionGeometry.create_Parallel3D(
        rotation_axis_direction=[0, np.sin(tilt), -np.cos(tilt)],
        units="microns"
    )
    
    # Set detector panel properties
    ag.set_panel(
        num_pixels=[num_pixels_x, num_pixels_y],
        origin='top-left',
        pixel_size=effective_pixel_size
    )
    
    # Set projection angles
    ag.set_angles(projection_angles)
    
    return ag
