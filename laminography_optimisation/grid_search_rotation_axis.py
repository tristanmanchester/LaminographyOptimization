"""
Grid search for optimal rotation axis orientation in laminography reconstruction.
This script generates reconstructions across a range of rotation axis orientations,
keeping the COR at 0 and the primary tilt angle fixed at 35 degrees.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
from skimage.filters import sobel
from scipy.ndimage import gaussian_gradient_magnitude, laplace
from scipy.stats import kurtosis, skew
from scipy.signal import correlate2d
import gc  # For garbage collection
import hashlib

# Import functions from our package
from data_io import load_data, save_reconstruction
from preprocessing import preprocess_data
from reconstruction import reconstruct
from main import run_reconstruction

# Global flag for handling graceful shutdown
SHUTDOWN_REQUESTED = False

def signal_handler(sig, frame):
    """Handle Ctrl+C signal for graceful shutdown"""
    global SHUTDOWN_REQUESTED
    print("\n\nShutdown requested. Finishing current batch and saving results...")
    SHUTDOWN_REQUESTED = True

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Import the metric functions from grid_search_loss_fn.py
from grid_search_loss_fn import (
    forward_projection_metric, gradient_magnitude_metric, negative_entropy_metric,
    total_variation_metric, contrast_metric, kurtosis_metric, skewness_metric,
    focus_metric, frequency_metric, gradient_weighted_contrast, 
    directional_gradient_metric, autocorrelation_metric, is_failed_result
)

def create_geometry_with_axis_tilt(projection_angles, lam_tilt_angle, 
                                 axis_tilt_x, axis_tilt_z,
                                 num_pixels_x, num_pixels_y,
                                 detector_pixel_size, detector_binning=1):
    """
    Create a laminography acquisition geometry with full control over rotation axis.
    
    Parameters:
    -----------
    projection_angles : ndarray
        Array of angles for each projection
    lam_tilt_angle : float
        Main laminography tilt angle of the rotation axis in degrees (tilt toward/away from detector)
    axis_tilt_x : float
        Additional tilt of rotation axis in x-direction (degrees)
    axis_tilt_z : float
        Additional tilt of rotation axis in z-direction (degrees)
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
    from cil.framework import AcquisitionGeometry
    
    # Convert angles to radians
    lam_tilt = np.radians(lam_tilt_angle)
    x_tilt = np.radians(axis_tilt_x)
    z_tilt = np.radians(axis_tilt_z)
    
    # Calculate effective pixel size
    effective_pixel_size = detector_pixel_size * detector_binning
    
    # Calculate rotation axis direction with all tilts
    # Start with standard laminography tilt [0, sin(tilt), -cos(tilt)]
    # Then apply additional rotations for x and z tilts
    
    # Base direction for laminography tilt
    axis_dir = [0, np.sin(lam_tilt), -np.cos(lam_tilt)]
    
    # Apply rotation around x-axis (affects y and z components)
    if x_tilt != 0:
        y_component = axis_dir[1]
        z_component = axis_dir[2]
        axis_dir[1] = y_component * np.cos(x_tilt) - z_component * np.sin(x_tilt)
        axis_dir[2] = y_component * np.sin(x_tilt) + z_component * np.cos(x_tilt)
    
    # Apply rotation around z-axis (affects x and y components)
    if z_tilt != 0:
        x_component = axis_dir[0]
        y_component = axis_dir[1]
        axis_dir[0] = x_component * np.cos(z_tilt) - y_component * np.sin(z_tilt)
        axis_dir[1] = x_component * np.sin(z_tilt) + y_component * np.cos(z_tilt)
    
    # Normalize to unit vector
    norm = np.sqrt(axis_dir[0]**2 + axis_dir[1]**2 + axis_dir[2]**2)
    axis_dir = [component / norm for component in axis_dir]
    
    # Create parallel beam geometry with calculated rotation axis
    ag = AcquisitionGeometry.create_Parallel3D(
        rotation_axis_direction=axis_dir,
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

def process_job(job_params):
    """
    Process a single reconstruction job.
    
    Parameters:
    -----------
    job_params : dict
        Contains all parameters needed for the reconstruction job
        
    Returns:
    --------
    dict
        Metrics calculated for this reconstruction
    """
    # Unpack parameters
    lam_tilt = job_params['lam_tilt']
    axis_tilt_x = job_params['axis_tilt_x']
    axis_tilt_z = job_params['axis_tilt_z']
    cor = job_params['cor']
    processed_data = job_params['processed_data']
    projection_angles = job_params['projection_angles']
    reference_data = job_params['reference_data']
    detector_pixel_size = job_params['detector_pixel_size']
    detector_binning = job_params['detector_binning']
    job_id = job_params['job_id']
    total_jobs = job_params['total_jobs']
    
    # Create a unique name for this reconstruction
    recon_name = f"lam_tilt_{lam_tilt:.2f}_x_tilt_{axis_tilt_x:.2f}_z_tilt_{axis_tilt_z:.2f}_cor_{cor:.1f}"
    
    print(f"\nReconstruction {job_id}/{total_jobs}: lam_tilt={lam_tilt:.2f}°, x_tilt={axis_tilt_x:.2f}°, z_tilt={axis_tilt_z:.2f}°, COR={cor:.1f}")
    
    # Create geometry
    print("Creating geometry...")
    geometry = create_geometry_with_axis_tilt(
        projection_angles, 
        lam_tilt, 
        axis_tilt_x,
        axis_tilt_z,
        processed_data.shape[2], 
        processed_data.shape[1], 
        detector_pixel_size, 
        detector_binning
    )
    
    # Set center of rotation
    if cor != 0:
        geometry.set_centre_of_rotation(cor, distance_units='pixels')
    
    # Create acquisition data
    print("Preparing acquisition data...")
    from cil.framework import AcquisitionData
    acquisition_data = AcquisitionData(processed_data, geometry=geometry)
    
    # Reorder for reconstruction
    acquisition_data.reorder('astra')
    
    # Reconstruct
    print("Running reconstruction...")
    reconstruction = reconstruct(acquisition_data)  # COR already in geometry
    
    # Extract central slice for metrics (no need to save the reconstruction)
    print("Extracting central slice...")
    central_slice = reconstruction.as_array()[reconstruction.shape[0]//2]
    
    print("Calculating metrics...")
    # Calculate all metrics
    metrics = {
        'lam_tilt': lam_tilt,
        'axis_tilt_x': axis_tilt_x,
        'axis_tilt_z': axis_tilt_z,
        'cor': cor,
    }
    
    # Calculate metrics
    try:
        metrics['gradient_magnitude'] = gradient_magnitude_metric(central_slice)
        metrics['negative_entropy'] = negative_entropy_metric(central_slice)
        metrics['total_variation'] = total_variation_metric(central_slice)
        metrics['contrast'] = contrast_metric(central_slice)
        
        # Additional metrics
        metrics['kurtosis'] = kurtosis_metric(central_slice)
        metrics['skewness'] = skewness_metric(central_slice)
        metrics['focus'] = focus_metric(central_slice)
        metrics['frequency'] = frequency_metric(central_slice)
        metrics['gradient_weighted_contrast'] = gradient_weighted_contrast(central_slice)
        metrics['directional_gradient'] = directional_gradient_metric(central_slice)
        metrics['autocorrelation'] = autocorrelation_metric(central_slice)
        
        # Forward projection metric (physics-based comparison)
        print("Calculating forward projection metric...")
        metrics['projection_consistency'] = forward_projection_metric(
            reconstruction, 
            reference_data,
            lam_tilt, 
            cor, 
            detector_pixel_size, 
            detector_binning
        )
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        # Add NaN values for failed metrics
        for metric in ['gradient_magnitude', 'negative_entropy', 'total_variation', 
                     'contrast', 'kurtosis', 'skewness', 'focus', 'frequency',
                     'gradient_weighted_contrast', 'directional_gradient', 
                     'autocorrelation', 'projection_consistency']:
            if metric not in metrics:
                metrics[metric] = np.nan
    
    # Cleanup to free memory
    del reconstruction
    del acquisition_data
    gc.collect()
    
    return metrics

def generate_folder_name(lam_tilt, x_tilt_range, z_tilt_range, cor, file_path):
    """Generate a unique folder name based on the parameters"""
    # Create a hash of the file path to make it part of the folder name
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    
    # Create folder name based on parameter ranges
    # Handle the case where x_tilt_range is a single value
    if isinstance(x_tilt_range, (int, float)) or len(x_tilt_range) == 1:
        x_tilt_value = x_tilt_range[0] if isinstance(x_tilt_range, tuple) else x_tilt_range
        x_tilt_part = f"x_tilt_{x_tilt_value:.2f}"
    else:
        x_tilt_part = f"x_tilt_{x_tilt_range[0]:.2f}-{x_tilt_range[1]:.2f}-{x_tilt_range[2]:.3f}"
    
    # Handle the case where z_tilt_range is a single value
    if isinstance(z_tilt_range, (int, float)) or len(z_tilt_range) == 1:
        z_tilt_value = z_tilt_range[0] if isinstance(z_tilt_range, tuple) else z_tilt_range
        z_tilt_part = f"z_tilt_{z_tilt_value:.2f}"
    else:
        z_tilt_part = f"z_tilt_{z_tilt_range[0]:.2f}-{z_tilt_range[1]:.2f}-{z_tilt_range[2]:.3f}"
    
    folder_name = f"axis_search_lam_tilt_{lam_tilt:.2f}_{x_tilt_part}_{z_tilt_part}_cor_{cor:.2f}_{file_hash}"
    
    # Add timestamp for uniqueness
    folder_name = os.path.join("/dls/science/users/qps56811/environments/LaminographyOptimization", 
                             folder_name)
    
    return folder_name

def run_axis_search(file_path, output_dir=None, 
                   lam_tilt=35.0, cor=0.0,
                   x_tilt_range=(-1, 1.25, 0.25), z_tilt_range=(-1, 1.25, 0.25),
                   detector_pixel_size=0.54, detector_binning=4,
                   binning=1, batch_size=20, resume=True, max_retries=3):
    """
    Run grid search over rotation axis orientation angles.
    
    Parameters:
    -----------
    file_path : str
        Path to the NXS data file
    output_dir : str
        Directory to save results. If None, will generate based on parameters.
    lam_tilt : float
        Fixed laminography tilt angle in degrees (default 35.0)
    cor : float
        Fixed center of rotation offset in pixels (default 0.0)
    x_tilt_range : tuple or float
        (start, stop, step) for x-axis tilt in degrees, or a single value
    z_tilt_range : tuple or float
        (start, stop, step) for z-axis tilt in degrees, or a single value
    detector_pixel_size : float
        Size of detector pixels in microns
    detector_binning : int
        Detector hardware binning factor
    binning : int
        Additional software binning to apply
    batch_size : int
        Number of reconstructions to process in parallel
    resume : bool
        Whether to resume from previous run if output_dir exists
    max_retries : int
        Maximum number of retries for failed jobs
    """
    # Generate output directory name based on parameters if not provided
    if output_dir is None:
        output_dir = generate_folder_name(lam_tilt, x_tilt_range, z_tilt_range, cor, file_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate parameter grid
    # Handle the case where x_tilt_range is a single value
    if isinstance(x_tilt_range, (int, float)):
        x_tilt_values = np.array([float(x_tilt_range)])
    elif len(x_tilt_range) == 1:
        x_tilt_values = np.array([float(x_tilt_range[0])])
    else:
        x_tilt_values = np.arange(*x_tilt_range)
    
    # Handle the case where z_tilt_range is a single value
    if isinstance(z_tilt_range, (int, float)):
        z_tilt_values = np.array([float(z_tilt_range)])
    elif len(z_tilt_range) == 1:
        z_tilt_values = np.array([float(z_tilt_range[0])])
    else:
        z_tilt_values = np.arange(*z_tilt_range)
    
    print(f"Grid search with {len(x_tilt_values)} x-axis tilt values and {len(z_tilt_values)} z-axis tilt values")
    print(f"Fixed laminography tilt: {lam_tilt:.2f}°, Fixed COR: {cor:.2f} pixels")
    total_recons = len(x_tilt_values) * len(z_tilt_values)
    print(f"Total reconstructions: {total_recons}")
    
    # Track completed and failed parameters
    completed_params = set()
    failed_params = {}  # (lam_tilt, axis_tilt_x, axis_tilt_z, cor) -> retry_count
    
    # Check for existing results if resume is enabled
    if resume and os.path.exists(os.path.join(output_dir, 'axis_search_results.csv')):
        try:
            print("Found existing results, loading to resume...")
            existing_df = pd.read_csv(os.path.join(output_dir, 'axis_search_results.csv'))
            
            # Check each row for successful completion
            for _, row in existing_df.iterrows():
                lam_tilt_val = row['lam_tilt']
                x_tilt = row['axis_tilt_x']
                z_tilt = row['axis_tilt_z']
                cor_val = row['cor']
                
                # Create metrics dict from row to check if it's a failed result
                metrics = {col: row[col] for col in row.index if pd.notna(row[col])}
                
                if not is_failed_result(metrics):
                    # Good result, mark as completed
                    completed_params.add((lam_tilt_val, x_tilt, z_tilt, cor_val))
                else:
                    # Failed result, add to retry list with count 0
                    failed_params[(lam_tilt_val, x_tilt, z_tilt, cor_val)] = 0
                    print(f"Found failed result to retry: lam_tilt={lam_tilt_val:.2f}°, x_tilt={x_tilt:.2f}°, z_tilt={z_tilt:.2f}°, COR={cor_val:.1f}")
            
            print(f"Found {len(completed_params)} successful and {len(failed_params)} failed reconstructions.")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            print("Starting fresh.")
            completed_params = set()
            failed_params = {}
    
    # Load and preprocess data (do this once for efficiency)
    print("Loading and preprocessing data...")
    data, image_key, angles = load_data(file_path)
    processed_data, projection_angles = preprocess_data(data, image_key, angles, verbose=False)
    
    if binning > 1:
        # Apply binning if needed
        from preprocessing import apply_binning
        print(f"Applying {binning}x binning...")
        processed_data = apply_binning(processed_data, binning)
    
    # Create reference acquisition data (for forward projection comparison)
    print("Creating reference acquisition data...")
    from cil.framework import AcquisitionData
    
    # Create reference geometry for the fixed laminography tilt
    ref_geometry = create_geometry_with_axis_tilt(
        projection_angles, 
        lam_tilt,
        0.0,  # No additional x tilt for reference
        0.0,  # No additional z tilt for reference
        processed_data.shape[2], 
        processed_data.shape[1], 
        detector_pixel_size, 
        detector_binning
    )
    
    # Set COR if not zero
    if cor != 0:
        ref_geometry.set_centre_of_rotation(cor, distance_units='pixels')
    
    # Create reference acquisition data 
    reference_data = AcquisitionData(processed_data, geometry=ref_geometry)
    
    # Create results list
    all_results = []
    
    # Load any existing results from file
    if resume and os.path.exists(os.path.join(output_dir, 'axis_search_results.csv')):
        try:
            existing_df = pd.read_csv(os.path.join(output_dir, 'axis_search_results.csv'))
            
            # Add only successful results to all_results
            for _, row in existing_df.iterrows():
                metrics = row.to_dict()
                if not is_failed_result(metrics):
                    all_results.append(metrics)
            
            print(f"Loaded {len(all_results)} successful results.")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    
    # Create job list with all parameters that need processing
    job_list = []
    job_id = 1
    
    for x_tilt in x_tilt_values:
        for z_tilt in z_tilt_values:
            # Skip jobs that have already been completed successfully
            if (lam_tilt, x_tilt, z_tilt, cor) in completed_params:
                print(f"Skipping already completed: lam_tilt={lam_tilt:.2f}°, x_tilt={x_tilt:.2f}°, z_tilt={z_tilt:.2f}°, COR={cor:.1f}")
                continue
            
            # Check if this is a previously failed job that we should retry
            retry_count = failed_params.get((lam_tilt, x_tilt, z_tilt, cor), 0)
            if retry_count >= max_retries:
                print(f"Skipping failed job that reached max retries: lam_tilt={lam_tilt:.2f}°, x_tilt={x_tilt:.2f}°, z_tilt={z_tilt:.2f}°, COR={cor:.1f}")
                continue
                
            # Add this job to the processing list
            job = {
                'lam_tilt': lam_tilt,
                'axis_tilt_x': x_tilt,
                'axis_tilt_z': z_tilt,
                'cor': cor,
                'processed_data': processed_data,
                'projection_angles': projection_angles,
                'reference_data': reference_data,
                'detector_pixel_size': detector_pixel_size,
                'detector_binning': detector_binning,
                'job_id': job_id,
                'total_jobs': total_recons - len(completed_params),
                'retry_count': retry_count
            }
            
            job_list.append(job)
            job_id += 1
    
    print(f"Prepared {len(job_list)} jobs to process")
    
    # Start timing
    start_time = time.time()
    
    # Process jobs in batches
    if job_list:
        # Create executor
        print(f"Processing in batches of {batch_size}...")
        
        # Process all jobs
        batch_counter = 0
        completed_in_current_run = 0
        retry_jobs = []  # Jobs that need to be retried due to failures
        
        while (job_list or retry_jobs) and not SHUTDOWN_REQUESTED:
            # Prioritize main jobs over retry jobs
            if job_list:
                current_batch = job_list[:batch_size]
                job_list = job_list[batch_size:]
            else:
                current_batch = retry_jobs[:batch_size]
                retry_jobs = retry_jobs[batch_size:]
            
            batch_counter += 1
            print(f"\nProcessing batch {batch_counter}, {len(current_batch)} jobs...")
            
            # Process the batch in parallel
            with ProcessPoolExecutor(max_workers=batch_size) as executor:
                future_to_job = {executor.submit(process_job, job): job for job in current_batch}
                
                for future in as_completed(future_to_job):
                    if SHUTDOWN_REQUESTED:
                        print("Shutdown requested. Waiting for current jobs to complete...")
                        
                    job = future_to_job[future]
                    lam_tilt_val = job['lam_tilt']
                    x_tilt = job['axis_tilt_x']
                    z_tilt = job['axis_tilt_z']
                    cor_val = job['cor']
                    
                    try:
                        metrics = future.result()
                        
                        # Check if metrics indicate a failed result
                        if is_failed_result(metrics):
                            retry_count = job.get('retry_count', 0) + 1
                            print(f"Job failed with zeros/NaNs: lam_tilt={lam_tilt_val:.2f}°, x_tilt={x_tilt:.2f}°, z_tilt={z_tilt:.2f}°, COR={cor_val:.1f}, retry {retry_count}/{max_retries}")
                            
                            if retry_count < max_retries:
                                # Queue for retry with incremented retry counter
                                job['retry_count'] = retry_count
                                failed_params[(lam_tilt_val, x_tilt, z_tilt, cor_val)] = retry_count
                                retry_jobs.append(job)
                            else:
                                print(f"Max retries reached for job: lam_tilt={lam_tilt_val:.2f}°, x_tilt={x_tilt:.2f}°, z_tilt={z_tilt:.2f}°, COR={cor_val:.1f}")
                                # Still add to results so we don't lose the parameter values
                                all_results.append(metrics)
                        else:
                            # Successful job
                            all_results.append(metrics)
                            completed_params.add((lam_tilt_val, x_tilt, z_tilt, cor_val))
                            # Remove from failed_params if it was there
                            if (lam_tilt_val, x_tilt, z_tilt, cor_val) in failed_params:
                                del failed_params[(lam_tilt_val, x_tilt, z_tilt, cor_val)]
                                
                        completed_in_current_run += 1
                        
                        # Progress update
                        elapsed = time.time() - start_time
                        remaining_jobs = len(job_list) + len(retry_jobs)
                        total_current_run = completed_in_current_run + remaining_jobs
                        
                        if total_current_run > 0:
                            progress_pct = completed_in_current_run / total_current_run * 100
                            remaining_time = (elapsed / completed_in_current_run) * remaining_jobs
                            print(f"Progress: {completed_in_current_run}/{total_current_run} ({progress_pct:.1f}%)")
                            print(f"Elapsed time: {elapsed/60:.1f} minutes, Estimated remaining: {remaining_time/60:.1f} minutes")
                            if retry_jobs:
                                print(f"Queued for retry: {len(retry_jobs)} jobs")
                        
                    except Exception as e:
                        print(f"Error processing job lam_tilt={lam_tilt_val:.2f}°, x_tilt={x_tilt:.2f}°, z_tilt={z_tilt:.2f}°, COR={cor_val:.1f}: {e}")
                        # Add to retry queue if under max retries
                        retry_count = job.get('retry_count', 0) + 1
                        if retry_count < max_retries:
                            job['retry_count'] = retry_count
                            failed_params[(lam_tilt_val, x_tilt, z_tilt, cor_val)] = retry_count
                            retry_jobs.append(job)
                            print(f"Queued for retry: lam_tilt={lam_tilt_val:.2f}°, x_tilt={x_tilt:.2f}°, z_tilt={z_tilt:.2f}°, COR={cor_val:.1f}, retry {retry_count}/{max_retries}")
            
            # Save intermediate results after each batch
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(output_dir, 'axis_search_results.csv'), index=False)
            
            # Create intermediate plots after every few batches
            if batch_counter % 3 == 0 and not SHUTDOWN_REQUESTED:
                try:
                    plot_metric_heatmaps(df, output_dir, x_tilt_values, z_tilt_values)
                    plot_combined_heatmap(df, output_dir, x_tilt_values, z_tilt_values)
                except Exception as e:
                    print(f"Warning: Error creating intermediate plots: {e}")
            
            # Break if shutdown requested
            if SHUTDOWN_REQUESTED:
                print("Shutdown requested. Saving results and stopping...")
                break
    
    # Create final dataframe and save
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(output_dir, 'axis_search_results.csv'), index=False)
    
    # Plot the metrics as 2D heatmaps
    print("Creating final plots...")
    try:
        plot_metric_heatmaps(final_df, output_dir, x_tilt_values, z_tilt_values)
        plot_combined_heatmap(final_df, output_dir, x_tilt_values, z_tilt_values)
    except Exception as e:
        print(f"Error creating final plots: {e}")
    
    # Report completion
    if SHUTDOWN_REQUESTED:
        print(f"\nAxis search interrupted. Partial results saved to {output_dir}")
    else:
        print(f"\nAxis search completed. Results saved to {output_dir}")
    
    return final_df

def plot_metric_heatmaps(df, output_dir, x_tilt_values, z_tilt_values):
    """Plot heatmaps or line plots for each metric across the parameter grid."""
    metrics = [
        'gradient_magnitude', 'negative_entropy', 'total_variation', 'contrast',
        'kurtosis', 'skewness', 'focus', 'frequency', 
        'gradient_weighted_contrast', 'directional_gradient', 'autocorrelation',
        'projection_consistency'
    ]
    
    for metric in metrics:
        try:
            print(f"Plotting {metric}...")
            
            # Handle different cases based on number of x_tilt and z_tilt values
            if len(x_tilt_values) == 1 and len(z_tilt_values) == 1:
                # Single point - just print the value
                plt.figure(figsize=(6, 4))
                mask = (df['axis_tilt_x'] == x_tilt_values[0]) & (df['axis_tilt_z'] == z_tilt_values[0])
                if mask.any():
                    value = df.loc[mask, metric].values[0]
                    plt.text(0.5, 0.5, f"{metric}: {value:.4f}", 
                           horizontalalignment='center', verticalalignment='center',
                           fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_value.png'), dpi=300)
                plt.close()
                
            elif len(x_tilt_values) == 1:
                # Line plot along z_tilt axis
                plt.figure(figsize=(10, 6))
                data = []
                for z_tilt in z_tilt_values:
                    mask = (df['axis_tilt_x'] == x_tilt_values[0]) & (df['axis_tilt_z'] == z_tilt)
                    if mask.any():
                        data.append(df.loc[mask, metric].values[0])
                    else:
                        data.append(np.nan)
                
                plt.plot(z_tilt_values, data, 'o-', linewidth=2)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xlabel('Z-Axis Tilt (degrees)')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.title(f'{metric.replace("_", " ").title()} vs Z-Axis Tilt (X-Tilt={x_tilt_values[0]:.2f}°)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_vs_z_tilt.png'), dpi=300)
                plt.close()
                
            elif len(z_tilt_values) == 1:
                # Line plot along x_tilt axis
                plt.figure(figsize=(10, 6))
                data = []
                for x_tilt in x_tilt_values:
                    mask = (df['axis_tilt_x'] == x_tilt) & (df['axis_tilt_z'] == z_tilt_values[0])
                    if mask.any():
                        data.append(df.loc[mask, metric].values[0])
                    else:
                        data.append(np.nan)
                
                plt.plot(x_tilt_values, data, 'o-', linewidth=2)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xlabel('X-Axis Tilt (degrees)')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.title(f'{metric.replace("_", " ").title()} vs X-Axis Tilt (Z-Tilt={z_tilt_values[0]:.2f}°)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_vs_x_tilt.png'), dpi=300)
                plt.close()
                
            else:
                # 2D heatmap (original behavior)
                plt.figure(figsize=(10, 8))
                
                # Reshape the metric data into a 2D grid
                grid_data = np.zeros((len(x_tilt_values), len(z_tilt_values)))
                grid_data.fill(np.nan)  # Fill with NaN to account for missing values
                
                for i, x_tilt in enumerate(x_tilt_values):
                    for j, z_tilt in enumerate(z_tilt_values):
                        mask = (df['axis_tilt_x'] == x_tilt) & (df['axis_tilt_z'] == z_tilt)
                        if mask.any():
                            grid_data[i, j] = df.loc[mask, metric].values[0]
                
                # Plot heatmap
                plt.imshow(grid_data, cmap='viridis', aspect='auto', 
                          extent=[z_tilt_values[0], z_tilt_values[-1], x_tilt_values[-1], x_tilt_values[0]])
                
                # Add colorbar and labels
                plt.colorbar(label=metric.replace('_', ' ').title())
                plt.xlabel('Z-Axis Tilt (degrees)')
                plt.ylabel('X-Axis Tilt (degrees)')
                plt.title(f'{metric.replace("_", " ").title()} across Parameter Grid')
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_heatmap.png'), dpi=300)
                plt.close()
                
        except Exception as e:
            print(f"Error plotting {metric}: {e}")

def plot_combined_heatmap(df, output_dir, x_tilt_values, z_tilt_values):
    """Plot all metrics in a grid for comparison, adapting to the parameter dimensions."""
    metrics = [
        'gradient_magnitude', 'negative_entropy', 'total_variation', 'contrast',
        'kurtosis', 'skewness', 'focus', 'frequency', 
        'gradient_weighted_contrast', 'directional_gradient', 'autocorrelation',
        'projection_consistency'
    ]
    
    # Handle different cases based on number of x_tilt and z_tilt values
    if len(x_tilt_values) == 1 and len(z_tilt_values) == 1:
        # For a single point, create a bar chart of all metrics
        print("Creating combined metrics bar chart...")
        
        plt.figure(figsize=(12, 8))
        metric_values = []
        metric_names = []
        
        mask = (df['axis_tilt_x'] == x_tilt_values[0]) & (df['axis_tilt_z'] == z_tilt_values[0])
        if mask.any():
            for metric in metrics:
                if metric in df.columns:
                    value = df.loc[mask, metric].values[0]
                    if not np.isnan(value):
                        metric_values.append(value)
                        metric_names.append(metric.replace('_', ' ').title())
        
        # Create bar chart of metrics
        plt.bar(range(len(metric_names)), metric_values)
        plt.xticks(range(len(metric_names)), metric_names, rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.title(f'Metrics for X-Tilt={x_tilt_values[0]:.2f}°, Z-Tilt={z_tilt_values[0]:.2f}°')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_metrics_bar.png'), dpi=300)
        plt.close()
        
    elif len(x_tilt_values) == 1:
        # For a fixed x_tilt angle, plot multiple metrics against z_tilt
        print("Creating combined metrics line plot (vs Z-Tilt)...")
        
        plt.figure(figsize=(15, 10))
        
        # Normalize each metric for better comparison
        for i, metric in enumerate(metrics):
            data = []
            for z_tilt in z_tilt_values:
                mask = (df['axis_tilt_x'] == x_tilt_values[0]) & (df['axis_tilt_z'] == z_tilt)
                if mask.any() and metric in df.columns:
                    data.append(df.loc[mask, metric].values[0])
                else:
                    data.append(np.nan)
            
            # Skip if all values are NaN
            if all(np.isnan(d) for d in data):
                continue
                
            # Normalize data for comparison (min-max scaling)
            valid_data = np.array(data)[~np.isnan(data)]
            if len(valid_data) > 0:
                data_min = np.min(valid_data)
                data_max = np.max(valid_data)
                data_range = data_max - data_min
                
                if data_range > 0:
                    normalized_data = [(d - data_min) / data_range if not np.isnan(d) else np.nan for d in data]
                    plt.plot(z_tilt_values, normalized_data, 'o-', linewidth=2, label=metric.replace('_', ' ').title())
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Z-Axis Tilt (degrees)')
        plt.ylabel('Normalized Metric Value')
        plt.title(f'Normalized Metrics vs Z-Axis Tilt (X-Tilt={x_tilt_values[0]:.2f}°)')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_metrics_vs_z_tilt.png'), dpi=300)
        plt.close()
        
    elif len(z_tilt_values) == 1:
        # For a fixed z_tilt, plot multiple metrics against x_tilt
        print("Creating combined metrics line plot (vs X-Tilt)...")
        
        plt.figure(figsize=(15, 10))
        
        # Normalize each metric for better comparison
        for i, metric in enumerate(metrics):
            data = []
            for x_tilt in x_tilt_values:
                mask = (df['axis_tilt_x'] == x_tilt) & (df['axis_tilt_z'] == z_tilt_values[0])
                if mask.any() and metric in df.columns:
                    data.append(df.loc[mask, metric].values[0])
                else:
                    data.append(np.nan)
            
            # Skip if all values are NaN
            if all(np.isnan(d) for d in data):
                continue
                
            # Normalize data for comparison (min-max scaling)
            valid_data = np.array(data)[~np.isnan(data)]
            if len(valid_data) > 0:
                data_min = np.min(valid_data)
                data_max = np.max(valid_data)
                data_range = data_max - data_min
                
                if data_range > 0:
                    normalized_data = [(d - data_min) / data_range if not np.isnan(d) else np.nan for d in data]
                    plt.plot(x_tilt_values, normalized_data, 'o-', linewidth=2, label=metric.replace('_', ' ').title())
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('X-Axis Tilt (degrees)')
        plt.ylabel('Normalized Metric Value')
        plt.title(f'Normalized Metrics vs X-Axis Tilt (Z-Tilt={z_tilt_values[0]:.2f}°)')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_metrics_vs_x_tilt.png'), dpi=300)
        plt.close()
        
    else:
        # Original 2D heatmap behavior for both parameters varying
        print("Creating combined metrics heatmap...")
        
        # Set up the figure
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        # Process each metric
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            try:
                # Reshape the metric data into a 2D grid
                grid_data = np.zeros((len(x_tilt_values), len(z_tilt_values)))
                grid_data.fill(np.nan)  # Fill with NaN for missing values
                
                for j, x_tilt in enumerate(x_tilt_values):
                    for k, z_tilt in enumerate(z_tilt_values):
                        mask = (df['axis_tilt_x'] == x_tilt) & (df['axis_tilt_z'] == z_tilt)
                        if mask.any():
                            grid_data[j, k] = df.loc[mask, metric].values[0]
                
                # Plot heatmap
                im = ax.imshow(grid_data, cmap='viridis', aspect='auto',
                               extent=[z_tilt_values[0], z_tilt_values[-1], x_tilt_values[-1], x_tilt_values[0]])
                
                # Add labels
                ax.set_xlabel('Z-Axis Tilt (degrees)')
                ax.set_ylabel('X-Axis Tilt (degrees)')
                ax.set_title(metric.replace('_', ' ').title())

            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)[:50]}...",
                       horizontalalignment='center', verticalalignment='center')
        
        # Hide any unused subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'combined_metrics_heatmap.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    # Example usage
    file_path = "/dls/science/users/qps56811/environments/LaminographyOptimization/data/k11-54013.nxs"
    
    # Fixed parameters
    lam_tilt = 37.5  # Fixed laminography tilt angle
    cor = 0.0  # Fixed center of rotation
    
    # Parameter ranges for grid search of rotation axis orientation
    # Format: (start, stop, step)
    x_tilt_range = (-1, 1.05, 0.05)  # X-axis tilt range
    z_tilt_range = (-1, 1.05, 0.05)  # Z-axis tilt range
    
    # Run grid search with optimized parameters
    results = run_axis_search(
        file_path=file_path,
        # Output dir will be automatically generated based on parameters
        lam_tilt=lam_tilt,
        cor=cor,
        x_tilt_range=x_tilt_range,
        z_tilt_range=z_tilt_range,
        detector_pixel_size=0.54,
        detector_binning=4,
        binning=1,
        batch_size=15,    # Process in parallel
        resume=True,      # Resume from previous run if available
        max_retries=3     # Retry failed jobs up to 3 times
    )