# %% [markdown]
# PhD team 
# Presenting Code for PhD Thesis

# %% info [markdown]
# Here, we are presenting a completed routine, a thorough connected blocks, to perform the very idea of my Ph.D. thesis, conducting a comprehensively automatic and longitudinal analyses of a given MS patient.
# All of the rights of this routine are reserved for the developer(s).
# 
# 
# Mahdi Bashiri Bawil
# Developer

# %% [markdown]
# 

# %% [markdown]
# # Attempt : Comprehensive Analyses 

# %% [markdown]
# %% [markdown]
# ## Phase 0: Dependencies & Functions

# %% [markdown]
# ### Packages

# %% Packages
import os
import cv2
import shutil
import skimage
import warnings
import numpy as np
import tensorflow as tf
from scipy.ndimage import binary_dilation, label


# %% Model Inferences and Mask Productions 
def fitter(image, brn_cnt, brn_axs, output_shape=(256, 256)):

    new_width = np.max(brn_axs)
    new_height = np.max(brn_axs)

    half_width = np.round(new_width // 2).astype(np.uint16)
    half_height = np.round(new_height // 2).astype(np.uint16)
    
    # Calculate cropping box coordinates
    top = int(max(brn_cnt[1] - half_height, 0))
    bottom = int(min(brn_cnt[1] + half_height, image.shape[0]))
    left = int(max(brn_cnt[0] - half_width, 0))
    right = int(min(brn_cnt[0] + half_width, image.shape[1]))
    
    # Crop the image using slicing
    cropped_image = image[top:bottom, left:right]
    # Resize the cropped image to the desired output shape (256x256)
    resized_image = np.uint16(np.round(skimage.transform.resize(cropped_image / 65535.0, output_shape, anti_aliasing=True) * 65535))   
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    return resized_image

#
def load_image_stack(image_stack):
    
    """
    Convert a 3D NumPy array of shape (256, 256, 20) to a 4D array 
    suitable for TensorFlow Pix2Pix model input.
    """
    # Ensure the input is in the expected format
    if image_stack.ndim != 3:
        raise ValueError("Input must be a 3D NumPy array of shape (height, width, num_slices).")

    # Normalizing the input to [-1, 1] 
    image_stack = (image_stack / (65535.0 / 2)) - 1

    # Rearrange dimensions: (256, 256, 20) -> (20, 256, 256, 1)
    # Add a channel dimension for grayscale images
    image_stack_4d = np.expand_dims(np.transpose(image_stack, (2, 0, 1)), axis=-1)
    
    # Convert to TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image_stack_4d, dtype=tf.float32)

    return image_tensor

#
def generate_images(generator, input_image):
    """Generate an image using the trained Pix2Pix generator."""

    prediction = generator(input_image, training=True)
    
    return prediction.numpy()

#
def de_fitter_v1(image, brn_cnt, brn_axs):
    
    new_height = np.max(brn_axs)
    new_width = np.max(brn_axs)

    # Resize the image to the previous bounding box

    resized_image = np.uint16(np.round(skimage.transform.resize(image / 65535.0, (new_height, new_width), anti_aliasing=True) * 65535))   

    # Translate the resized image to meet the centers of main brain image

    translated_image = np.zeros_like(image)

    main_height, main_width = translated_image.shape
    small_height, small_width = resized_image.shape

    # Calculate top-left corner for placing the small image
    top_left_x = int(brn_cnt[0] - small_width / 2)
    top_left_y = int(brn_cnt[1] - small_height / 2)

    # Ensure the coordinates don't exceed the boundaries of the main image
    top_left_x = max(0, min(top_left_x, main_width - small_width))
    top_left_y = max(0, min(top_left_y, main_height - small_height))

    # Place the small image on the main image
    translated_image[top_left_y:top_left_y + small_height, top_left_x:top_left_x + small_width] = resized_image
    
    return translated_image

def de_fitter(image, brn_cnt, brn_axs):
    """
    Resize and translate image with robust handling of NaN and infinite values.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image array
    brn_cnt : tuple/array
        Brain center coordinates (x, y)
    brn_axs : tuple/array
        Brain axis dimensions
        
    Returns:
    --------
    numpy.ndarray
        Processed image with same shape as input
    """
    
    new_height = int(np.max(brn_axs))
    new_width = int(np.max(brn_axs))
    
    # Check input image for NaN/inf values
    has_nan = np.isnan(image).any()
    has_inf = np.isinf(image).any()
    
    if has_nan or has_inf:
        warnings.warn(f"Input image contains {'NaN' if has_nan else ''}{' and ' if has_nan and has_inf else ''}{'infinite' if has_inf else ''} values.")
        # Clean the input image
        clean_input = np.nan_to_num(image, nan=0, posinf=65535, neginf=0)
    else:
        clean_input = image
    
    # Ensure input is within valid range
    clean_input = np.clip(clean_input, 0, 65535)
    
    # Normalize to [0, 1] for skimage processing
    normalized_image = clean_input / 65535.0
    
    # Resize the image with error handling
    try:
        resized_normalized = skimage.transform.resize(
            normalized_image, 
            (new_height, new_width), 
            anti_aliasing=True,
            preserve_range=False
        )
        
        # Convert back to uint16 range with robust handling
        resized_scaled = resized_normalized * 65535.0
        
        # Handle any NaN/inf values that might have been introduced during resize
        if np.isnan(resized_scaled).any() or np.isinf(resized_scaled).any():
            warnings.warn("NaN/inf values detected after resize operation. Cleaning...")
            resized_scaled = np.nan_to_num(resized_scaled, nan=0, posinf=65535, neginf=0)
        
        # Robust conversion to uint16
        resized_scaled_clipped = np.clip(resized_scaled, 0, 65535)
        
        # Use np.rint for better rounding behavior with edge cases
        resized_rounded = np.rint(resized_scaled_clipped)
        
        # Final conversion to uint16 with additional safety check
        resized_image = resized_rounded.astype(np.uint16)
        
        # Verify the result doesn't contain invalid values
        if np.isnan(resized_image).any() or np.isinf(resized_image).any():
            warnings.warn("Invalid values persist after conversion. Applying final cleanup.")
            resized_image = np.nan_to_num(resized_image, nan=0, posinf=65535, neginf=0).astype(np.uint16)
            
    except Exception as e:
        warnings.warn(f"Error during resize operation: {e}. Using fallback method.")
        # Fallback: create a zero array of target size
        resized_image = np.zeros((new_height, new_width), dtype=np.uint16)
    
    # Translate the resized image to meet the centers of main brain image
    translated_image = np.zeros_like(image, dtype=np.uint16)
    
    main_height, main_width = translated_image.shape
    small_height, small_width = resized_image.shape
    
    # Ensure brn_cnt values are finite and valid
    if not (np.isfinite(brn_cnt[0]) and np.isfinite(brn_cnt[1])):
        warnings.warn("Brain center coordinates contain non-finite values. Using image center as fallback.")
        brn_cnt = [main_width // 2, main_height // 2]
    
    # Calculate top-left corner for placing the small image
    top_left_x = int(brn_cnt[0] - small_width / 2)
    top_left_y = int(brn_cnt[1] - small_height / 2)
    
    # Ensure the coordinates don't exceed the boundaries of the main image
    top_left_x = max(0, min(top_left_x, main_width - small_width))
    top_left_y = max(0, min(top_left_y, main_height - small_height))
    
    # Additional bounds checking
    if top_left_x < 0 or top_left_y < 0 or \
       top_left_x + small_width > main_width or \
       top_left_y + small_height > main_height:
        warnings.warn("Image placement would exceed boundaries. Adjusting placement.")
        top_left_x = max(0, min(top_left_x, main_width - small_width))
        top_left_y = max(0, min(top_left_y, main_height - small_height))
    
    # Place the small image on the main image with bounds checking
    try:
        end_y = min(top_left_y + small_height, main_height)
        end_x = min(top_left_x + small_width, main_width)
        
        actual_height = end_y - top_left_y
        actual_width = end_x - top_left_x
        
        translated_image[top_left_y:end_y, top_left_x:end_x] = resized_image[:actual_height, :actual_width]
        
    except Exception as e:
        warnings.warn(f"Error during image placement: {e}")
        # Return the original image if placement fails
        return image.astype(np.uint16)
    
    return translated_image

#
def transform_back(pred_masks, b_cnts, b_axes):

    de_fit_data = np.zeros_like(pred_masks)

    # Go in loop
    for i in range(pred_masks.shape[-1]):

        im_mask = pred_masks[..., i]

        # Perform the fit function
        de_fit_mask = de_fitter(im_mask, b_cnts[..., i][0], b_axes[..., i][0])

        # Save the fit image
        de_fit_data[..., i] = de_fit_mask

    return de_fit_data

#
def midpoint_clustering(predictions, midpoints):
    """
    Apply midpoint-based thresholding to map probabilities to discrete classes.
    :param predictions: Normalized prediction array (0-65535).
    :return: Discrete label array.
    """
    # Create an empty array for discrete labels
    labels = np.zeros_like(predictions, dtype=np.uint16)

    # Apply thresholds based on midpoints
    labels[predictions <= midpoints[0]] = 0  # Background
    labels[(predictions > midpoints[0]) & (predictions <= midpoints[1])] = 1  # Ventricles
    labels[(predictions > midpoints[1]) & (predictions <= midpoints[2])] = 2  # Juxtaventricular WMH
    labels[predictions > midpoints[2]] = 3  # Other WMH

    return labels

#
def labels_to_rgb(labels):
    """
    Convert label array to an RGB image.
    :param labels: Discrete label array with values 0 (background), 1 (ventricles), 
                   2 (juxtaventricular WMH), 3 (other WMH).
    :return: RGB image with colors assigned.
    """
    # Create an empty RGB image
    rgb_image = np.zeros((*labels.shape, 3), dtype=np.uint8)

    # Assign colors
    rgb_image[labels == 1] = [0, 0, 255]  # Blue for ventricles
    rgb_image[labels == 2] = [0, 255, 0]  # Green for juxtaventricular WMH
    rgb_image[labels == 3] = [255, 0, 0]  # Red for other WMH
    # Background (labels == 0) remains black: [0, 0, 0]

    return rgb_image

# Neuro-anatimical Mask Processing
def adjacency_finder_v1(image1, image2, min_area=5):
    # Step 1: Remove small objects from image1
    labeled_image1, num_objects1 = label(image1)
    processed_image1 = np.zeros_like(image1, dtype=bool)

    for i in range(num_objects1):
        object_area = np.sum(labeled_image1 == (i + 1))
        if object_area >= min_area:
            processed_image1[labeled_image1 == (i + 1)] = True

    # Step 2: Dilate the processed image1 by 1 pixel
    dilated_image1 = binary_dilation(processed_image1, structure=np.ones((3, 3)))

    # Step 3: Perform AND operation with image2 to get the seed image
    seed_image = dilated_image1 & image2

    # Step 4: Keep only objects in image2 that overlap with the seed image
    labeled_image2, num_objects2 = label(image2)
    filtered_image2 = np.zeros_like(image2, dtype=bool)

    for j in range(num_objects2):
        if np.any(seed_image[labeled_image2 == (j + 1)]):  # Check for overlap
            filtered_image2[labeled_image2 == (j + 1)] = True

    return filtered_image2

def enhanced_adjacency_finder_with_ventricles(abnormal_wmh, normal_wmh, ventricle_mask, 
                                            min_area=5, adjacency_threshold=1, alignment_threshold=2,
                                            voxel_size=(1.0, 1.0)):
    """
    Enhanced adjacency finder that considers ventricle context for WMH reclassification.
    
    Parameters:
    -----------
    abnormal_wmh : numpy.ndarray
        Binary mask of abnormal WMH objects
    normal_wmh : numpy.ndarray  
        Binary mask of normal WMH objects (juxtaventricular)
    ventricle_mask : numpy.ndarray
        Binary mask of ventricle regions
    min_area : int
        Minimum area threshold for objects (in pixels)
    adjacency_threshold : float
        Distance threshold for adjacency (in mm, will be converted to pixels)
    voxel_size : tuple
        Voxel spacing in mm (height, width)
    
    Returns:
    --------
    pixels_to_move : numpy.ndarray
        Binary mask of pixels to move from normal_wmh to abnormal_wmh
    """
    
    # Convert adjacency threshold from mm to pixels
    pixel_threshold = adjacency_threshold / min(voxel_size)
    
    # Step 1: Remove small objects from abnormal_wmh
    labeled_abnormal, num_abnormal = label(abnormal_wmh)
    processed_abnormal = np.zeros_like(abnormal_wmh, dtype=bool)
    
    for i in range(1, num_abnormal + 1):
        object_area = np.sum(labeled_abnormal == i)
        if object_area >= min_area:
            processed_abnormal[labeled_abnormal == i] = True
    
    # Step 2: Find normal WMH objects adjacent to abnormal WMH
    dilated_abnormal = binary_dilation(processed_abnormal, 
                                     structure=np.ones((int(2*adjacency_threshold+1), int(2*adjacency_threshold+1))))
    
    # Get normal WMH objects that are adjacent to abnormal WMH
    adjacent_seed = dilated_abnormal & normal_wmh
    labeled_normal, num_normal = label(normal_wmh)
    
    adjacent_normal_objects = np.zeros_like(normal_wmh, dtype=bool)
    for j in range(1, num_normal + 1):
        if np.any(adjacent_seed[labeled_normal == j]):
            adjacent_normal_objects[labeled_normal == j] = True
    
    # Step 3: For each adjacent normal object, determine which pixels are 
    # positioned between abnormal WMH and ventricles
    pixels_to_move = np.zeros_like(normal_wmh, dtype=bool)
    
    # Re-label the adjacent normal objects for individual processing
    labeled_adjacent, num_adjacent = label(adjacent_normal_objects)
    
    for obj_id in range(1, num_adjacent + 1):
        current_object = (labeled_adjacent == obj_id)
        
        # Find pixels in this object that are between abnormal WMH and ventricles
        between_pixels = find_pixels_between_regions(
            current_object, processed_abnormal, ventricle_mask, voxel_size, alignment_threshold
        )
        
        pixels_to_move |= between_pixels
    
    return pixels_to_move

def find_pixels_between_regions(normal_object, abnormal_wmh, ventricle_mask, voxel_size, alignment_tolerance):
    """
    Find pixels in normal_object that are spatially between abnormal_wmh and ventricles.
    
    Parameters:
    -----------
    normal_object : numpy.ndarray
        Binary mask of a single normal WMH object
    abnormal_wmh : numpy.ndarray
        Binary mask of abnormal WMH regions
    ventricle_mask : numpy.ndarray
        Binary mask of ventricle regions
    voxel_size : tuple
        Voxel spacing in mm
        
    Returns:
    --------
    between_pixels : numpy.ndarray
        Binary mask of pixels that are between abnormal WMH and ventricles
    """
    
    between_pixels = np.zeros_like(normal_object, dtype=bool)
    
    # Get coordinates of pixels in the normal object
    normal_coords = np.where(normal_object)
    if len(normal_coords[0]) == 0:
        return between_pixels
    
    normal_points = np.column_stack(normal_coords)
    
    # Get coordinates of abnormal WMH and ventricle boundaries
    abnormal_coords = np.where(abnormal_wmh)
    ventricle_coords = np.where(ventricle_mask)
    
    if len(abnormal_coords[0]) == 0 or len(ventricle_coords[0]) == 0:
        return between_pixels
    
    abnormal_points = np.column_stack(abnormal_coords)
    ventricle_points = np.column_stack(ventricle_coords)
    
    # Scale coordinates by voxel size for accurate distance calculation
    normal_scaled = normal_points * np.array(voxel_size)
    abnormal_scaled = abnormal_points * np.array(voxel_size)
    ventricle_scaled = ventricle_points * np.array(voxel_size)
    
    # For each pixel in normal object, check if it's between abnormal WMH and ventricle
    for i, normal_pixel in enumerate(normal_scaled):
        # Find closest points in abnormal WMH and ventricle
        dist_to_abnormal = np.min(np.linalg.norm(abnormal_scaled - normal_pixel, axis=1))
        dist_to_ventricle = np.min(np.linalg.norm(ventricle_scaled - normal_pixel, axis=1))
        
        # Find the closest abnormal and ventricle points
        closest_abnormal_idx = np.argmin(np.linalg.norm(abnormal_scaled - normal_pixel, axis=1))
        closest_ventricle_idx = np.argmin(np.linalg.norm(ventricle_scaled - normal_pixel, axis=1))
        
        closest_abnormal = abnormal_scaled[closest_abnormal_idx]
        closest_ventricle = ventricle_scaled[closest_ventricle_idx]
        
        # Check if the normal pixel lies approximately on the line between 
        # the closest abnormal and ventricle points
        if is_point_between(normal_pixel, closest_abnormal, closest_ventricle, tolerance=alignment_tolerance):
            between_pixels[normal_coords[0][i], normal_coords[1][i]] = True
    
    return between_pixels

def is_point_between(point, point1, point2, tolerance=2.0):
    """
    Check if a point lies approximately between two other points.
    
    Parameters:
    -----------
    point : numpy.ndarray
        The point to check
    point1, point2 : numpy.ndarray
        The two reference points
    tolerance : float
        Distance tolerance in mm
        
    Returns:
    --------
    bool : True if point is between point1 and point2
    """
    
    # Calculate distances
    dist_1_to_point = np.linalg.norm(point1 - point)
    dist_point_to_2 = np.linalg.norm(point - point2)
    dist_1_to_2 = np.linalg.norm(point1 - point2)
    
    # Check if point lies approximately on the line between point1 and point2
    # The sum of distances should approximately equal the direct distance
    return abs((dist_1_to_point + dist_point_to_2) - dist_1_to_2) < tolerance

# Enhanced main processing function
def process_wmh_masks_with_ventricles(wmh, v_wmh, ventricle_masks, voxel_size, min_area_pixels=10, adjacency_threshold=2, alignment_threshold=3):
    """
    Process WMH masks considering ventricle context for each slice.
    
    Parameters:
    -----------
    wmh : numpy.ndarray
        3D abnormal WMH mask (H, W, D)
    v_wmh : numpy.ndarray  
        3D normal/juxtaventricular WMH mask (H, W, D)
    ventricle_masks : numpy.ndarray
        3D ventricle mask (H, W, D)
    voxel_size : tuple
        Voxel spacing (height, width, depth) in mm
        
    Returns:
    --------
    updated_wmh : numpy.ndarray
        Updated abnormal WMH mask
    updated_v_wmh : numpy.ndarray
        Updated normal WMH mask
    """
    
    updated_wmh = wmh.copy()
    updated_v_wmh = v_wmh.copy()
        
    # Process each slice
    for slice_idx in range(wmh.shape[-1]):
        print(f"Processing slice {slice_idx + 1}/{wmh.shape[-1]}")
        
        wmh_slice = wmh[..., slice_idx]
        v_wmh_slice = v_wmh[..., slice_idx]
        ventricle_slice = ventricle_masks[..., slice_idx]
        
        # Find pixels to move using enhanced algorithm
        pixels_to_move = enhanced_adjacency_finder_with_ventricles(
            wmh_slice, v_wmh_slice, ventricle_slice,
            min_area=min_area_pixels,
            adjacency_threshold=adjacency_threshold,  # 1mm threshold
            alignment_threshold=alignment_threshold,  # 
            voxel_size=voxel_size[:2]  # Only height and width for 2D processing
        )
        
        # Update masks
        updated_v_wmh[..., slice_idx] = updated_v_wmh[..., slice_idx] & ~pixels_to_move
        updated_wmh[..., slice_idx] = updated_wmh[..., slice_idx] | pixels_to_move
        
        # Print statistics for this slice
        moved_pixels = np.sum(pixels_to_move)
        if moved_pixels > 0:
            print(f"  Slice {slice_idx}: Moved {moved_pixels} pixels from normal to abnormal WMH")
    
    return updated_wmh, updated_v_wmh

