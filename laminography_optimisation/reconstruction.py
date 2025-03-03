"""Reconstruction algorithms."""

from cil.recon import FBP

def reconstruct(acquisition_data, cor_offset=0):
    """
    Reconstruct data using FBP with ASTRA backend.
    
    Parameters:
    -----------
    acquisition_data : AcquisitionData
        Preprocessed projection data with geometry
    cor_offset : float, optional
        Center of rotation offset in pixels, default is 0
        
    Returns:
    --------
    reconstruction : ImageData
        Reconstructed volume
    """
    # Set center of rotation if needed
    if cor_offset != 0:
        acquisition_data.geometry.set_centre_of_rotation(cor_offset, distance_units='pixels')
    
    # Ensure data is in ASTRA-compatible order
    acquisition_data.reorder('astra')
    
    # Create FBP reconstructor with ASTRA backend
    reconstructor = FBP(acquisition_data, backend='astra')
    
    # Run reconstruction
    reconstruction = reconstructor.run()
    
    return reconstruction
