"""
Grid search for optimal laminography reconstruction parameters.
This script generates reconstructions across a range of tilt angles and COR values,
then evaluates them using various image quality metrics.
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
from geometry import create_geometry
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

def forward_projection_metric(reconstruction, original_data, tilt, cor, 
                             detector_pixel_size, detector_binning):
    """
    Compare original projections with forward projections of the reconstruction.
    
    Parameters:
    -----------
    reconstruction : ImageData
        Reconstructed volume
    original_data : AcquisitionData
        Original projection data
    tilt : float
        Tilt angle used for reconstruction
    cor : float
        Center of rotation offset used for reconstruction
    
    Returns:
    --------
    float
        Similarity metric (higher is better)
    """
    print("  Calculating forward projection metric...")
    
    try:
        from cil.plugins.astra import ProjectionOperator
        
        # Create a copy of original data
        original_data_astra = original_data.copy()
        
        # Reorder to ASTRA format
        print("    Reordering data to ASTRA format...")
        original_data_astra.reorder('astra')
        
        # Create projection operator
        projector = ProjectionOperator(reconstruction.geometry, original_data_astra.geometry)
        
        # Forward project the reconstruction
        print("    Performing forward projection...")
        forward_proj = projector.direct(reconstruction)
        
        # Sample projections for efficiency (every 10th)
        sample_step = 10
        num_projections = min(original_data_astra.shape[0], forward_proj.shape[0])
        indices = np.arange(0, num_projections, sample_step)
        
        # Calculate metrics
        print("    Calculating projection similarity...")
        
        # Calculate RMSE (lower is better, so we negate it)
        orig_arr = original_data_astra.as_array()
        fproj_arr = forward_proj.as_array()
        
        # Ensure arrays are the same shape before comparison
        if orig_arr.shape != fproj_arr.shape:
            print(f"    Warning: Shape mismatch - Original: {orig_arr.shape}, Forward: {fproj_arr.shape}")
            # Use minimum dimensions
            min_dims = [min(o, f) for o, f in zip(orig_arr.shape, fproj_arr.shape)]
            orig_arr = orig_arr[:min_dims[0], :min_dims[1], :min_dims[2]]
            fproj_arr = fproj_arr[:min_dims[0], :min_dims[1], :min_dims[2]]
        
        # Calculate RMSE
        rmse = -np.sqrt(np.mean(
            (fproj_arr[indices] - orig_arr[indices])**2
        ))
        
        # Calculate normalized cross-correlation (higher is better)
        def ncc(img1, img2):
            img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-10)
            img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-10)
            return np.mean(img1_norm * img2_norm)
        
        ncc_values = []
        for i in indices:
            ncc_val = ncc(fproj_arr[i], orig_arr[i])
            ncc_values.append(ncc_val)
        
        # Return combined metric (weighted average of normalized metrics)
        mean_ncc = np.mean(ncc_values)
        
        # Combined metric - simple average
        combined = mean_ncc
        
        print(f"    RMSE: {rmse:.6f}, NCC: {mean_ncc:.6f}, Combined: {combined:.6f}")
        
        # Clean up
        del forward_proj
        del original_data_astra
        gc.collect()
        
        return combined
        
    except Exception as e:
        print(f"    Error in forward projection: {e}")
        return np.nan

def gradient_magnitude_metric(img):
    """Higher values indicate sharper features."""
    print("  Calculating gradient magnitude...")
    return np.mean(sobel(img))

def negative_entropy_metric(img):
    """Lower entropy often indicates better reconstruction (more structure)."""
    print("  Calculating negative entropy...")
    return -shannon_entropy(img)

def total_variation_metric(img):
    """Lower values indicate smoother regions with preserved edges."""
    print("  Calculating total variation...")
    # Calculate gradients
    dx = np.diff(img, axis=0)  # Shape is (height-1, width)
    dy = np.diff(img, axis=1)  # Shape is (height, width-1)
    
    # Simply sum the absolute values of gradients (standard total variation definition)
    tv = np.sum(np.abs(dx)) + np.sum(np.abs(dy))
    
    return -tv  # Negative because we want to maximize during optimization

def contrast_metric(img):
    """Higher values indicate better contrast."""
    print("  Calculating contrast...")
    p2, p98 = np.percentile(img, (2, 98))
    return p98 - p2

def kurtosis_metric(img):
    """Higher values often indicate sharper, more detailed images."""
    print("  Calculating kurtosis...")
    return kurtosis(img.flatten())

def skewness_metric(img):
    """Measures asymmetry of pixel distribution; deviations can indicate artifacts."""
    print("  Calculating skewness...")
    return -abs(skew(img.flatten()))  # Negative absolute value to maximize

def focus_metric(img):
    """Higher values indicate sharper focus/less blur (variance of Laplacian)."""
    print("  Calculating focus metric...")
    return np.var(laplace(img))

def frequency_metric(img):
    """Measures how energy is distributed in frequency domain."""
    print("  Calculating frequency metric...")
    # Apply 2D FFT - downsample if image is large
    if img.shape[0] > 1000 or img.shape[1] > 1000:
        # Downsample to speed up FFT
        downsampled = img[::2, ::2]
        f_transform = np.fft.fft2(downsampled)
    else:
        f_transform = np.fft.fft2(img)
    
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_transform_shifted)
    
    # Create distance from center matrix
    rows, cols = magnitude.shape
    center_row, center_col = rows//2, cols//2
    y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
    distance = np.sqrt(x*x + y*y)
    
    # Calculate ratio of high to low frequencies
    low_freq = np.sum(magnitude[distance < min(rows, cols)/4])
    high_freq = np.sum(magnitude[distance >= min(rows, cols)/4])
    return high_freq / (low_freq + 1e-10)  # Avoid division by zero

def gradient_weighted_contrast(img):
    """Combines contrast with gradient information."""
    print("  Calculating gradient weighted contrast...")
    # Calculate gradient magnitude
    gradient = sobel(img)
    
    # Calculate contrast
    p2, p98 = np.percentile(img, (2, 98))
    contrast = p98 - p2
    
    # Weight contrast by gradient information
    return contrast * np.mean(gradient)

def directional_gradient_metric(img):
    """Measures asymmetry in horizontal vs vertical gradients."""
    print("  Calculating directional gradient...")
    grad_x = np.abs(np.diff(img, axis=1, append=img[:,-1:]))
    grad_y = np.abs(np.diff(img, axis=0, append=img[-1:,:]))
    
    # Calculate ratio of vertical to horizontal gradients
    ratio = np.sum(grad_y) / (np.sum(grad_x) + 1e-10)
    
    # Return negative absolute difference from 1.0 (balanced gradients)
    return -abs(ratio - 1.0)

def autocorrelation_metric(img, max_size=256):
    """Detects repeating patterns (like double images). Optimized version."""
    print("  Calculating autocorrelation...")
    
    # Downsample image if it's large (for performance)
    if img.shape[0] > max_size or img.shape[1] > max_size:
        # Calculate downsample factors
        h_factor = max(1, img.shape[0] // max_size)
        w_factor = max(1, img.shape[1] // max_size)
        
        # Downsample by taking every nth pixel
        img_small = img[::h_factor, ::w_factor]
        print(f"    Downsampled from {img.shape} to {img_small.shape} for autocorrelation")
    else:
        img_small = img
    
    # Normalize image
    img_norm = (img_small - np.mean(img_small)) / (np.std(img_small) + 1e-10)
    
    # Compute autocorrelation (this is the expensive part)
    correlation = correlate2d(img_norm, img_norm, mode='same')
    
    # Normalize center peak to 1
    correlation = correlation / np.max(correlation)
    
    # Mask out the central peak
    center_y, center_x = correlation.shape[0]//2, correlation.shape[1]//2
    mask_radius = min(correlation.shape) // 20
    y, x = np.ogrid[-center_y:correlation.shape[0]-center_y, -center_x:correlation.shape[1]-center_x]
    mask = (x*x + y*y <= mask_radius*mask_radius)
    correlation[mask] = 0
    
    # Return negative of maximum correlation (excluding center)
    # Higher secondary peaks indicate double images
    return -np.max(correlation)

def is_failed_result(metrics):
    """
    Check if a result contains mostly zeros or NaNs, indicating a failed run.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from a job
        
    Returns:
    --------
    bool
        True if the result appears to be a failed run
    """
    # Keys to check (excluding tilt and cor which are parameters, not results)
    metric_keys = [
        'gradient_magnitude', 'negative_entropy', 'total_variation', 'contrast',
        'kurtosis', 'skewness', 'focus', 'frequency',
        'gradient_weighted_contrast', 'directional_gradient', 
        'autocorrelation', 'projection_consistency'
    ]
    
    # Count zeros and NaNs
    zero_count = 0
    nan_count = 0
    total_metrics = 0
    
    for key in metric_keys:
        if key in metrics:
            total_metrics += 1
            if metrics[key] == 0:
                zero_count += 1
            elif np.isnan(metrics[key]):
                nan_count += 1
    
    # If we have no metrics, consider it a failure
    if total_metrics == 0:
        return True
    
    # If more than 50% of the metrics are zeros or NaNs, consider it a failure
    failure_ratio = (zero_count + nan_count) / total_metrics
    return failure_ratio > 0.5

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
    tilt = job_params['tilt']
    cor = job_params['cor']
    processed_data = job_params['processed_data']
    projection_angles = job_params['projection_angles']
    reference_data = job_params['reference_data']
    detector_pixel_size = job_params['detector_pixel_size']
    detector_binning = job_params['detector_binning']
    job_id = job_params['job_id']
    total_jobs = job_params['total_jobs']
    
    # Create a unique name for this reconstruction
    recon_name = f"tilt_{tilt:.2f}_cor_{cor:.1f}"
    
    print(f"\nReconstruction {job_id}/{total_jobs}: tilt={tilt:.2f}°, COR={cor:.1f}")
    
    # Create geometry
    print("Creating geometry...")
    geometry = create_geometry(
        projection_angles, 
        tilt, 
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
        'tilt': tilt,
        'cor': cor,
    }
    
    # Calculate metrics
    try:
        metrics['gradient_magnitude'] = gradient_magnitude_metric(central_slice)
        metrics['negative_entropy'] = negative_entropy_metric(central_slice)
        metrics['total_variation'] = total_variation_metric(central_slice)
        metrics['contrast'] = contrast_metric(central_slice)
        
        # New metrics
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
            tilt, 
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

def generate_folder_name(tilt_range, cor_range, file_path):
    """Generate a unique folder name based on the parameters"""
    # Create a hash of the file path to make it part of the folder name
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    
    # Create folder name based on parameter ranges
    # Handle the case where tilt_range is a single value
    if isinstance(tilt_range, (int, float)) or len(tilt_range) == 1:
        tilt_value = tilt_range[0] if isinstance(tilt_range, tuple) else tilt_range
        tilt_part = f"tilt_{tilt_value:.2f}"
    else:
        tilt_part = f"tilt_{tilt_range[0]:.2f}-{tilt_range[1]:.2f}-{tilt_range[2]:.3f}"
    
    # Handle the case where cor_range is a single value
    if isinstance(cor_range, (int, float)) or len(cor_range) == 1:
        cor_value = cor_range[0] if isinstance(cor_range, tuple) else cor_range
        cor_part = f"cor_{cor_value:.2f}"
    else:
        cor_part = f"cor_{cor_range[0]:.2f}-{cor_range[1]:.2f}-{cor_range[2]:.3f}"
    
    folder_name = f"grid_search_{tilt_part}_{cor_part}_{file_hash}"
    
    # Add timestamp for uniqueness
    folder_name = os.path.join("/dls/science/users/qps56811/environments/LaminographyOptimization", 
                             folder_name)
    
    return folder_name

def run_grid_search(file_path, output_dir=None, 
                   tilt_range=(30, 40.125, 0.125), cor_range=(-10, 10.25, 0.25),
                   detector_pixel_size=0.54, detector_binning=4,
                   binning=1, batch_size=20, resume=True, max_retries=3):
    """
    Run grid search over tilt angles and COR values.
    
    Parameters:
    -----------
    file_path : str
        Path to the NXS data file
    output_dir : str
        Directory to save results. If None, will generate based on parameters.
    tilt_range : tuple or float
        (start, stop, step) for tilt angles in degrees, or a single value
    cor_range : tuple or float
        (start, stop, step) for center of rotation offsets in pixels, or a single value
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
        output_dir = generate_folder_name(tilt_range, cor_range, file_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate parameter grid
    # Handle the case where tilt_range is a single value
    if isinstance(tilt_range, (int, float)):
        tilt_values = np.array([float(tilt_range)])
    elif len(tilt_range) == 1:
        tilt_values = np.array([float(tilt_range[0])])
    else:
        tilt_values = np.arange(*tilt_range)
    
    # Handle the case where cor_range is a single value
    if isinstance(cor_range, (int, float)):
        cor_values = np.array([float(cor_range)])
    elif len(cor_range) == 1:
        cor_values = np.array([float(cor_range[0])])
    else:
        cor_values = np.arange(*cor_range)
    
    print(f"Grid search with {len(tilt_values)} tilt values and {len(cor_values)} COR values")
    total_recons = len(tilt_values) * len(cor_values)
    print(f"Total reconstructions: {total_recons}")
    
    # Track completed and failed parameters
    completed_params = set()
    failed_params = {}  # (tilt, cor) -> retry_count
    
    # Check for existing results if resume is enabled
    if resume and os.path.exists(os.path.join(output_dir, 'grid_search_results.csv')):
        try:
            print("Found existing results, loading to resume...")
            existing_df = pd.read_csv(os.path.join(output_dir, 'grid_search_results.csv'))
            
            # Check each row for successful completion
            for _, row in existing_df.iterrows():
                tilt = row['tilt']
                cor = row['cor']
                
                # Create metrics dict from row to check if it's a failed result
                metrics = {col: row[col] for col in row.index if pd.notna(row[col])}
                
                if not is_failed_result(metrics):
                    # Good result, mark as completed
                    completed_params.add((tilt, cor))
                else:
                    # Failed result, add to retry list with count 0
                    failed_params[(tilt, cor)] = 0
                    print(f"Found failed result to retry: tilt={tilt:.2f}°, COR={cor:.1f}")
            
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
    
    # Create a dictionary to hold reference data for each tilt angle
    reference_data_dict = {}
    
    # Create all reference data objects for each tilt angle upfront
    # This is needed for the metrics calculation and prevents recreation for each COR value
    for tilt in tilt_values:
        # Create geometry for this tilt
        ref_geometry = create_geometry(
            projection_angles, 
            tilt,
            processed_data.shape[2], 
            processed_data.shape[1], 
            detector_pixel_size, 
            detector_binning
        )
        
        # Create reference acquisition data 
        reference_data_dict[tilt] = AcquisitionData(processed_data, geometry=ref_geometry)
    
    # Create results list
    all_results = []
    
    # Load any existing results from file
    if resume and os.path.exists(os.path.join(output_dir, 'grid_search_results.csv')):
        try:
            existing_df = pd.read_csv(os.path.join(output_dir, 'grid_search_results.csv'))
            
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
    
    for tilt in tilt_values:
        reference_data = reference_data_dict[tilt]
        
        for cor in cor_values:
            # Skip jobs that have already been completed successfully
            if (tilt, cor) in completed_params:
                print(f"Skipping already completed: tilt={tilt:.2f}°, COR={cor:.1f}")
                continue
            
            # Check if this is a previously failed job that we should retry
            retry_count = failed_params.get((tilt, cor), 0)
            if retry_count >= max_retries:
                print(f"Skipping failed job that reached max retries: tilt={tilt:.2f}°, COR={cor:.1f}")
                continue
                
            # Add this job to the processing list
            job = {
                'tilt': tilt,
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
                    tilt = job['tilt']
                    cor = job['cor']
                    
                    try:
                        metrics = future.result()
                        
                        # Check if metrics indicate a failed result
                        if is_failed_result(metrics):
                            retry_count = job.get('retry_count', 0) + 1
                            print(f"Job failed with zeros/NaNs: tilt={tilt:.2f}°, COR={cor:.1f}, retry {retry_count}/{max_retries}")
                            
                            if retry_count < max_retries:
                                # Queue for retry with incremented retry counter
                                job['retry_count'] = retry_count
                                failed_params[(tilt, cor)] = retry_count
                                retry_jobs.append(job)
                            else:
                                print(f"Max retries reached for job: tilt={tilt:.2f}°, COR={cor:.1f}")
                                # Still add to results so we don't lose the parameter values
                                all_results.append(metrics)
                        else:
                            # Successful job
                            all_results.append(metrics)
                            completed_params.add((tilt, cor))
                            # Remove from failed_params if it was there
                            if (tilt, cor) in failed_params:
                                del failed_params[(tilt, cor)]
                                
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
                        print(f"Error processing job tilt={tilt:.2f}°, COR={cor:.1f}: {e}")
                        # Add to retry queue if under max retries
                        retry_count = job.get('retry_count', 0) + 1
                        if retry_count < max_retries:
                            job['retry_count'] = retry_count
                            failed_params[(tilt, cor)] = retry_count
                            retry_jobs.append(job)
                            print(f"Queued for retry: tilt={tilt:.2f}°, COR={cor:.1f}, retry {retry_count}/{max_retries}")
            
            # Save intermediate results after each batch
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(output_dir, 'grid_search_results.csv'), index=False)
            
            # Create intermediate plots after every few batches
            if batch_counter % 3 == 0 and not SHUTDOWN_REQUESTED:
                try:
                    plot_metric_heatmaps(df, output_dir, tilt_values, cor_values)
                    plot_combined_heatmap(df, output_dir, tilt_values, cor_values)
                except Exception as e:
                    print(f"Warning: Error creating intermediate plots: {e}")
            
            # Break if shutdown requested
            if SHUTDOWN_REQUESTED:
                print("Shutdown requested. Saving results and stopping...")
                break
    
    # Create final dataframe and save
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(output_dir, 'grid_search_results.csv'), index=False)
    
    # Plot the metrics as 2D heatmaps
    print("Creating final plots...")
    try:
        plot_metric_heatmaps(final_df, output_dir, tilt_values, cor_values)
        plot_combined_heatmap(final_df, output_dir, tilt_values, cor_values)
    except Exception as e:
        print(f"Error creating final plots: {e}")
    
    # Report completion
    if SHUTDOWN_REQUESTED:
        print(f"\nGrid search interrupted. Partial results saved to {output_dir}")
    else:
        print(f"\nGrid search completed. Results saved to {output_dir}")
    
    return final_df

def plot_metric_heatmaps(df, output_dir, tilt_values, cor_values):
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
            
            # Handle different cases based on number of tilt and cor values
            if len(tilt_values) == 1 and len(cor_values) == 1:
                # Single point - just print the value
                plt.figure(figsize=(6, 4))
                mask = (df['tilt'] == tilt_values[0]) & (df['cor'] == cor_values[0])
                if mask.any():
                    value = df.loc[mask, metric].values[0]
                    plt.text(0.5, 0.5, f"{metric}: {value:.4f}", 
                           horizontalalignment='center', verticalalignment='center',
                           fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_value.png'), dpi=300)
                plt.close()
                
            elif len(tilt_values) == 1:
                # Line plot along COR axis
                plt.figure(figsize=(10, 6))
                data = []
                for cor in cor_values:
                    mask = (df['tilt'] == tilt_values[0]) & (df['cor'] == cor)
                    if mask.any():
                        data.append(df.loc[mask, metric].values[0])
                    else:
                        data.append(np.nan)
                
                plt.plot(cor_values, data, 'o-', linewidth=2)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xlabel('Center of Rotation Offset (pixels)')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.title(f'{metric.replace("_", " ").title()} vs COR (Tilt={tilt_values[0]:.2f}°)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_vs_cor.png'), dpi=300)
                plt.close()
                
            elif len(cor_values) == 1:
                # Line plot along Tilt axis
                plt.figure(figsize=(10, 6))
                data = []
                for tilt in tilt_values:
                    mask = (df['tilt'] == tilt) & (df['cor'] == cor_values[0])
                    if mask.any():
                        data.append(df.loc[mask, metric].values[0])
                    else:
                        data.append(np.nan)
                
                plt.plot(tilt_values, data, 'o-', linewidth=2)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xlabel('Tilt Angle (degrees)')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.title(f'{metric.replace("_", " ").title()} vs Tilt (COR={cor_values[0]:.2f})')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_vs_tilt.png'), dpi=300)
                plt.close()
                
            else:
                # 2D heatmap (original behavior)
                plt.figure(figsize=(10, 8))
                
                # Reshape the metric data into a 2D grid
                grid_data = np.zeros((len(tilt_values), len(cor_values)))
                grid_data.fill(np.nan)  # Fill with NaN to account for missing values
                
                for i, tilt in enumerate(tilt_values):
                    for j, cor in enumerate(cor_values):
                        mask = (df['tilt'] == tilt) & (df['cor'] == cor)
                        if mask.any():
                            grid_data[i, j] = df.loc[mask, metric].values[0]
                
                # Plot heatmap
                plt.imshow(grid_data, cmap='viridis', aspect='auto', 
                          extent=[cor_values[0], cor_values[-1], tilt_values[-1], tilt_values[0]])
                
                # Add colorbar and labels
                plt.colorbar(label=metric.replace('_', ' ').title())
                plt.xlabel('Center of Rotation Offset (pixels)')
                plt.ylabel('Tilt Angle (degrees)')
                plt.title(f'{metric.replace("_", " ").title()} across Parameter Grid')
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_heatmap.png'), dpi=300)
                plt.close()
                
        except Exception as e:
            print(f"Error plotting {metric}: {e}")

def plot_combined_heatmap(df, output_dir, tilt_values, cor_values):
    """Plot all metrics in a grid for comparison, adapting to the parameter dimensions."""
    metrics = [
        'gradient_magnitude', 'negative_entropy', 'total_variation', 'contrast',
        'kurtosis', 'skewness', 'focus', 'frequency', 
        'gradient_weighted_contrast', 'directional_gradient', 'autocorrelation',
        'projection_consistency'
    ]
    
    # Handle different cases based on number of tilt and cor values
    if len(tilt_values) == 1 and len(cor_values) == 1:
        # For a single point, create a bar chart of all metrics
        print("Creating combined metrics bar chart...")
        
        plt.figure(figsize=(12, 8))
        metric_values = []
        metric_names = []
        
        mask = (df['tilt'] == tilt_values[0]) & (df['cor'] == cor_values[0])
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
        plt.title(f'Metrics for Tilt={tilt_values[0]:.2f}°, COR={cor_values[0]:.2f}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_metrics_bar.png'), dpi=300)
        plt.close()
        
    elif len(tilt_values) == 1:
        # For a fixed tilt angle, plot multiple metrics against COR
        print("Creating combined metrics line plot (vs COR)...")
        
        plt.figure(figsize=(15, 10))
        
        # Normalize each metric for better comparison
        for i, metric in enumerate(metrics):
            data = []
            for cor in cor_values:
                mask = (df['tilt'] == tilt_values[0]) & (df['cor'] == cor)
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
                    plt.plot(cor_values, normalized_data, 'o-', linewidth=2, label=metric.replace('_', ' ').title())
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Center of Rotation Offset (pixels)')
        plt.ylabel('Normalized Metric Value')
        plt.title(f'Normalized Metrics vs COR (Tilt={tilt_values[0]:.2f}°)')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_metrics_vs_cor.png'), dpi=300)
        plt.close()
        
    elif len(cor_values) == 1:
        # For a fixed COR, plot multiple metrics against tilt
        print("Creating combined metrics line plot (vs Tilt)...")
        
        plt.figure(figsize=(15, 10))
        
        # Normalize each metric for better comparison
        for i, metric in enumerate(metrics):
            data = []
            for tilt in tilt_values:
                mask = (df['tilt'] == tilt) & (df['cor'] == cor_values[0])
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
                    plt.plot(tilt_values, normalized_data, 'o-', linewidth=2, label=metric.replace('_', ' ').title())
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Tilt Angle (degrees)')
        plt.ylabel('Normalized Metric Value')
        plt.title(f'Normalized Metrics vs Tilt (COR={cor_values[0]:.2f})')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_metrics_vs_tilt.png'), dpi=300)
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
                grid_data = np.zeros((len(tilt_values), len(cor_values)))
                grid_data.fill(np.nan)  # Fill with NaN for missing values
                
                for j, tilt in enumerate(tilt_values):
                    for k, cor in enumerate(cor_values):
                        mask = (df['tilt'] == tilt) & (df['cor'] == cor)
                        if mask.any():
                            grid_data[j, k] = df.loc[mask, metric].values[0]
                
                # Plot heatmap
                im = ax.imshow(grid_data, cmap='viridis', aspect='auto',
                               extent=[cor_values[0], cor_values[-1], tilt_values[-1], tilt_values[0]])
                
                # Add labels
                ax.set_xlabel('COR (pixels)')
                ax.set_ylabel('Tilt Angle (degrees)')
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
    
    # Example 1: Grid search with both parameters varying
    # Format: (start, stop, step)
    tilt_range = (30,40,0.1)  
    cor_range = (0)
    
    # Example 2: Constant tilt angle with varying COR
    # tilt_range = 35.0  # Fixed tilt angle
    # cor_range = (-1, 1, 0.1)  # Range of COR values
    
    # Example 3: Varying tilt angle with constant COR
    # tilt_range = (35, 36, 0.1)  # Range of tilt angles
    # cor_range = 0  # Fixed COR value
    
    # Example 4: Single parameter combination
    # tilt_range = 35.0  # Fixed tilt angle
    # cor_range = 0  # Fixed COR value
    
    # Run grid search with optimized parameters
    results = run_grid_search(
        file_path=file_path,
        # Output dir will be automatically generated based on parameters
        tilt_range=tilt_range,
        cor_range=cor_range,
        detector_pixel_size=0.54,
        detector_binning=4,
        binning=1,
        batch_size=15,    # Process in parallel
        resume=True,      # Resume from previous run if available
        max_retries=3     # Retry failed jobs up to 3 times
    )