"""Main script for laminography reconstruction."""

import os
import numpy as np
from cil.framework import AcquisitionData

from .data_io import load_data, save_reconstruction
from .preprocessing import preprocess_data, apply_binning
from .geometry import create_geometry
from .reconstruction import reconstruct

def run_reconstruction(file_path, output_path, tilt_angle, cor_offset=0,
                      binning=1, detector_pixel_size=0.54, detector_binning=4):
    """
    Main function to run laminography reconstruction pipeline.
    
    Parameters:
    -----------
    file_path : str
        Path to the NXS data file
    output_path : str
        Path to save the reconstruction
    tilt_angle : float
        Tilt angle of the rotation axis in degrees
    cor_offset : float, optional
        Center of rotation offset in pixels, default is 0
    binning : int, optional
        Additional binning factor to apply during processing, default is 1 (no binning)
    detector_pixel_size : float, optional
        Size of detector pixels in microns, default is 0.54
    detector_binning : int, optional
        Detector binning factor used during acquisition, default is 4
        
    Returns:
    --------
    reconstruction : ImageData
        Reconstructed volume
    """
    print(f"Loading data from {file_path}")
    # Load data
    data, image_key, angles = load_data(file_path)
    
    print("Preprocessing data")
    # Preprocess data
    processed_data, projection_angles = preprocess_data(data, image_key, angles)
    
    # Apply additional binning if requested
    if binning > 1:
        print(f"Applying {binning}x binning")
        processed_data = apply_binning(processed_data, binning)
    
    print(f"Creating geometry (tilt angle: {tilt_angle}°)")
    # Create geometry
    num_pixels_x = processed_data.shape[2]
    num_pixels_y = processed_data.shape[1]
    geometry = create_geometry(
        projection_angles,
        tilt_angle,
        num_pixels_x,
        num_pixels_y,
        detector_pixel_size,
        detector_binning
    )
    
    print("Preparing acquisition data")
    # Create AcquisitionData with geometry
    acquisition_data = AcquisitionData(processed_data, geometry=geometry)
    
    print(f"Reconstructing with COR offset: {cor_offset}")
    # Reconstruct
    reconstruction = reconstruct(acquisition_data, cor_offset)
    
    print(f"Saving reconstruction to {output_path}")
    # Save reconstruction
    save_reconstruction(reconstruction, output_path, as_16bit=True)
    
    print("Reconstruction complete")
    return reconstruction

if __name__ == "__main__":
    # Example usage that can be edited for testing
    
    # Path configuration
    file_path = "/path/to/your/file.nxs"  
    output_dir = "/path/to/output"       
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reconstruction.tiff")
    
    # Reconstruction parameters
    tilt_angle = 35.0  
    cor_offset = 0.0  
    binning = 1      
    
    # Detector parameters
    detector_pixel_size = 0.54  # Default detector pixel size in microns
    detector_binning = 4        # Hardware binning during acquisition
    
    # Run the reconstruction
    reconstruction = run_reconstruction(
        file_path=file_path,
        output_path=output_path,
        tilt_angle=tilt_angle,
        cor_offset=cor_offset,
        binning=binning,
        detector_pixel_size=detector_pixel_size,
        detector_binning=detector_binning
    )
