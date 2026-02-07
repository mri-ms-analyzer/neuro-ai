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
import skimage
import numpy as np
from scipy import ndimage


# %% Mask Processing and Feature Extraction

def wmh_vent_distance(
    obj_contour: 'np.ndarray',
    vent_image: 'np.ndarray',
    pixel_size: list
) -> float:
    """
    Calculate the minimum distance between a single WMH object contour and the ventricle contours in a given slice.

    Parameters:
        obj_contour (np.ndarray): Contour of the WMH object (single contour).
        vent_image (np.ndarray): Binary image containing all ventricle contours in the slice.
        pixel_size (list): A two-element list or array specifying the physical size of each pixel (e.g., [width, height] in mm).
    
    Returns:
        float: The minimum distance between the WMH contour and the nearest ventricle contour, in physical units.
    """
    vent_contour, _ = cv2.findContours(vent_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(vent_contour) != 0:

        # obj_contour is an obj of WMHs in a slice.
        # vent_contour is ventricles in a corresponding slice.

        # Measuring minimum distance between the object and ventricular system
        # Initialize the minimum distance as a large value
        min_distance = float('inf')
        min_distance_cnt = float('inf')

        # Calculate the centroid of the obj
        M = cv2.moments(obj_contour)


        # Calculate the centroid coordinates (center of mass)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # there might be only one or two points in the obj_contour:
            cx = obj_contour[0][0][0]
            cy = obj_contour[0][0][1]

        # Print the centroid coordinates
        # print(f"Centroid coordinates (x, y): ({cx}, {cy})")

        # First, min distance due to the center of obj:
        # Iterate through all pairs of points in the two or more contours:
        for vent_cont in vent_contour:
            for point1 in vent_cont:

                distance = np.linalg.norm(point1 - [cx, cy])
                # print(distance)

                # Update the minimum distance if a smaller distance is found
                if distance < min_distance_cnt:
                    min_distance_cnt = distance

        # Second, min distance due to boundaries:
        # Iterate through all pairs of points in the two or more contours:
        for vent_cont in vent_contour:
            for point1 in vent_cont:
                for point2 in obj_contour:
                    # Calculate the Euclidean distance between the two points
                    distance = np.linalg.norm(point1 - point2)
                    # print(distance)

                    # Update the minimum distance if a smaller distance is found
                    if distance < min_distance:
                        min_distance = distance

        min_distance = np.round(min_distance * pixel_size, 3)
        min_distance_cnt = np.round(min_distance_cnt * pixel_size, 3)
        # Print the minimum distance
        # print(f"Minimum distance between the edges of the vents and the center of the object: {min_distance_cnt}")
        # print(f"Minimum distance between the edges of the vents and the object: {min_distance}")
    else:
        min_distance = min_distance_cnt = 1000   # a large number!
        # Print the minimum distance
        print(f'\nThere is no seen ventricle in the slice, so min_distance cannot be defined.'
              f'  (it is set to a large value : {min_distance})')

    return min_distance_cnt, min_distance, vent_contour

#
def wmh_gm_distance(
    obj_contour: 'np.ndarray',
    gm_image: 'np.ndarray',
    pixel_size: list
) -> float:
    """
    Calculate the minimum distance between a single WMH object contour and the gray matter (GM) contours in a given slice.

    Parameters:
        obj_contour (np.ndarray): Contour of the WMH object (single contour).
        gm_image (np.ndarray): Binary image containing all gray matter (GM) regions in the slice.
        pixel_size (list): A two-element list or array specifying the physical size of each pixel (e.g., [width, height] in mm).
    
    Returns:
        float: The minimum distance between the WMH contour and the nearest GM contour, in physical units.
    """
    # First of all, we should check whether there is an overlapping between obj (our its contour) and GM map or not!
    check = 1
    gm_cont = []
    for point in obj_contour:

        if gm_image[point[0][0], point[0][1]] == 1:
            # object is in the gm map:
            # check = 0
            break
    if check == 1:
        min_gm_distance_cnt, min_gm_distance, gm_cont = wmh_vent_distance(obj_contour, gm_image, pixel_size)

    else:
        # Print the minimum distance
        min_gm_distance_cnt = min_gm_distance = 1000
        print(f'\nThere is no distinction between the object and GrayMatter tissue, so min_distance cannot be defined.'
              f'  (it is set to a zero value : {min_gm_distance})')

    return min_gm_distance_cnt, min_gm_distance, gm_cont

#
def wmh_csf_distance(
    obj_contour: 'np.ndarray',
    csf_image: 'np.ndarray',
    pixel_size: list
) -> float:
    """
    Calculate the minimum distance between a single WMH object contour and the cerebrospinal fluids (CSF) contours in a given slice.

    Parameters:
        obj_contour (np.ndarray): Contour of the WMH object (single contour).
        csf_image (np.ndarray): Binary image containing all cerebrospinal fluids (CSF) regions in the slice.
        pixel_size (list): A two-element list or array specifying the physical size of each pixel (e.g., [width, height] in mm).
    
    Returns:
        float: The minimum distance between the WMH contour and the nearest CSF contour, in physical units.
    """
    # First of all, we should check whether there is an overlapping between obj (our its contour) and GM map or not!
    check = 1
    csf_cont = []
    for point in obj_contour:

        if csf_image[point[0][0], point[0][1]] == 1:
            # object is in the csf map:
            # check = 0
            break
    if check == 1:
        min_csf_distance_cnt, min_csf_distance, csf_cont = wmh_vent_distance(obj_contour, csf_image, pixel_size)

    else:
        # Print the minimum distance
        min_csf_distance_cnt = min_csf_distance = 1000
        print(f'\nThere is no distinction between the object and CSF region, so min_distance cannot be defined.'
              f'  (it is set to a zero value : {min_csf_distance})')

    return min_csf_distance_cnt, min_csf_distance, csf_cont

#
def obj_categorize(obj_area, _v, _v_cnt, _g, _g_cnt, vent_rule=10, gm_rule=(2, 5)):
    # based on the rules:
    # decide the category:

    code = 0
    if _v <= .5 * vent_rule:
        # the WMH is "Periventricular"
        # print('peri')
        code = 1

    elif _v_cnt <= vent_rule:
        # the WMH is "Periventricular"
        # print('peri')
        code = 1

    elif _g <= gm_rule[0] and obj_area <= np.pi * (gm_rule[1] ** 2) / 4:
        # the WMH is "Juxtacortical"
        # print('juxt')
        code = 3

    else:
        # the WMH is "Paraventricular"
        # print('para')
        code = 2

    return code

#
def depth_distance(obj_contour, binary_mask):

    # binary_mask = dilation(binary_mask, disk(10))

    # Visualization
    def visualize_contours(binary_mask, obj_contour, cx, cy, mask_contour):
        # Create a BGR version of the binary mask for visualization
        visualization = cv2.cvtColor(slice_flair, cv2.COLOR_GRAY2BGR)

        # Draw the big contour (mask contour)
        cv2.drawContours(visualization, [mask_contour], -1, (0, 255, 0), 2)  # Green for mask contour

        # Draw the object contour
        cv2.drawContours(visualization, [obj_contour], -1, (255, 0, 0), 2)  # Blue for object contour

        # Draw the object center
        cv2.circle(visualization, (cx, cy), 1, (0, 0, 255), -1)  # Red for the center of the object

        # Show the visualization
        cv2.imshow("Contours and Object Center", visualization)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        return

    # Calculate the centroid of the object contour
    M = cv2.moments(obj_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
    else:
        # Handle cases with very few points
        cx = obj_contour[0][0][0]
        cy = obj_contour[0][0][1]

    # Find contours of the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_distances = []
    for mask_contour in contours:

        # There should be only one or two contours representing the brain area(s), hemisphares.
        mask_contour = mask_contour.reshape((-1, 2))  # Ensure it's in (N, 2) format

        # Calculate the minimum distance from the centroid to the mask contour
        distances = [cv2.pointPolygonTest(mask_contour, (float(cx), float(cy)), True)]
        # Convert from signed distance to a positive value if needed
        positive_distances = [abs(d) for d in distances]   
        min_distances.append(min(positive_distances))

        # Visualize results
        # visualize_contours(binary_mask, obj_contour, cx, cy, mask_contour)
        # print(f"Minimum distance to mask edge: {min_distances}, Centroid: ({cx}, {cy})")

    min_distance = min(min_distances)

    return [min_distance, cx, cy]

#
def create_hemispheric_mask(b_mask, line_width=1, use_major_axis=True):
    """
    Create hemispheric brain mask by dividing elliptical mask along its appropriate axis.

    Parameters:
    - b_mask: 2D binary numpy array (brain mask with white ellipse on black background)
    - line_width: width of the separation line (default: 1)
    - use_major_axis: if True, divide along the minor axis (perpendicular to major axis)
                     if False, always use vertical division

    Returns:
    - hemispharic_b_mask: binary mask with separation line
    - left_hemisphere: left hemisphere mask only
    - right_hemisphere: right hemisphere mask only
    - center_info: dictionary with ellipse center and axis information
    """

    # Ensure the mask is binary
    binary_mask = (b_mask > 0).astype(np.uint8)

    # Find contours to get the ellipse boundary
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError("No contours found in the binary mask")

    # Get the largest contour (should be the brain ellipse)
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit ellipse to the contour
    if len(largest_contour) >= 5:  # Need at least 5 points to fit ellipse
        ellipse = cv2.fitEllipse(largest_contour)
        center, axes, angle = ellipse
        center_x, center_y = center[0], center[1]
        major_axis, minor_axis = max(axes), min(axes)

        # Convert angle to radians and adjust for OpenCV convention
        angle_rad = np.deg2rad(angle)

        # Determine the orientation of the major axis
        if axes[0] > axes[1]:  # Major axis is along the first axis
            major_axis_angle = angle_rad
        else:  # Major axis is along the second axis
            major_axis_angle = angle_rad + np.pi / 2

    else:
        # Fallback: use center of mass and assume no rotation
        center_y, center_x = ndimage.center_of_mass(binary_mask)
        angle = 0
        angle_rad = 0
        major_axis_angle = 0
        axes = (0, 0)
        major_axis, minor_axis = 0, 0

    height, width = b_mask.shape

    if use_major_axis and len(largest_contour) >= 5:
        # Create separation line perpendicular to major axis (along minor axis)
        # This properly divides the ellipse into left and right hemispheres

        # Calculate the perpendicular angle (minor axis direction)
        perp_angle = major_axis_angle + np.pi / 2

        # Create line equation: we want to divide along the minor axis
        # Line passes through center and has direction perpendicular to major axis
        cos_perp = np.cos(perp_angle)
        sin_perp = np.sin(perp_angle)

        # Create masks
        hemispharic_b_mask = binary_mask.copy()
        left_hemisphere = np.zeros_like(binary_mask)
        right_hemisphere = np.zeros_like(binary_mask)

        # For each pixel, determine which side of the separation line it's on
        y_coords, x_coords = np.mgrid[0:height, 0:width]

        # Vector from center to each pixel
        dx = x_coords - center_x
        dy = y_coords - center_y

        # Cross product to determine which side of the line each pixel is on
        # Using the normal vector to the separation line (major axis direction)
        cos_major = np.cos(major_axis_angle)
        sin_major = np.sin(major_axis_angle)

        # Distance from line (signed)
        signed_distance = dx * sin_major - dy * cos_major

        # Create separation line mask
        line_mask = np.abs(signed_distance) <= line_width / 2

        # Apply separation line
        hemispharic_b_mask[line_mask & (binary_mask > 0)] = 0

        # Create hemisphere masks
        left_side = signed_distance < -line_width / 2
        right_side = signed_distance > line_width / 2

        left_hemisphere[left_side & (binary_mask > 0)] = binary_mask[left_side & (binary_mask > 0)]
        right_hemisphere[right_side & (binary_mask > 0)] = binary_mask[right_side & (binary_mask > 0)]

    else:
        # Fallback to vertical division
        center_x_int, center_y_int = int(center_x), int(center_y)
        hemispharic_b_mask = binary_mask.copy()

        # Create vertical separation line
        line_start = max(0, center_x_int - line_width // 2)
        line_end = min(width, center_x_int + line_width // 2 + 1)

        # Apply the separation line only where the original mask was white
        for x in range(line_start, line_end):
            for y in range(height):
                if binary_mask[y, x] > 0:
                    hemispharic_b_mask[y, x] = 0

        # Create individual hemisphere masks
        left_hemisphere = binary_mask.copy()
        right_hemisphere = binary_mask.copy()

        # Left hemisphere: keep only pixels to the left of center
        left_hemisphere[:, center_x_int:] = 0

        # Right hemisphere: keep only pixels to the right of center
        right_hemisphere[:, :center_x_int] = 0

    # Store center information
    center_info = {
        'center_x': center_x,
        'center_y': center_y,
        'angle': angle if len(largest_contour) >= 5 else 0,
        'major_axis_angle_deg': np.rad2deg(major_axis_angle) if len(largest_contour) >= 5 else 0,
        'axes': axes if len(largest_contour) >= 5 else (0, 0),
        'major_axis_length': major_axis if len(largest_contour) >= 5 else 0,
        'minor_axis_length': minor_axis if len(largest_contour) >= 5 else 0
    }

    return hemispharic_b_mask, left_hemisphere, right_hemisphere, center_info

#
def obj_analyzer(
    flair_image: 'np.ndarray', 
    bw_image: 'np.ndarray', 
    vent_image: 'np.ndarray', 
    gm_image: 'np.ndarray', 
    jv_image: 'np.ndarray', 
    b_mask: 'np.ndarray',
    c_mask: 'np.ndarray',
    voxel_size: float, 
    vent_rule: str, 
    gm_rule: str
    ) -> dict:
    """
    Analyze objects in medical images based on given rules and parameters.
    
    Parameters:
        flair_image (np.ndarray): Input FLAIR image.
        bw_image (np.ndarray): Binary white matter hyperintensity (WMH) slice image.
        vent_image (np.ndarray): Binary ventricle slice image.
        gm_image (np.ndarray): Binary gray matter slice image.
        jv_image (np.ndarray): Binary juxtaventricle WMH slice image.
        b_mask (np.ndarray): Binary brain slice image.
        c_mask (np.ndarray): Binary CSF slice mask.
        voxel_size (float): Size of a single voxel in the image (in mm³).
        vent_rule (str): Rule for ventricle analysis.
        gm_rule (str): Rule for gray matter analysis (e.g., "distance" or "diameter").
    
    Returns:
        dict: A dictionary containing analysis results.
    """

    ## Analyzing the jv_WMH
    contours, _ = cv2.findContours(jv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Number of found objects
    jv_number = len(contours)

    # List to store areas of detected objects
    jv_area = []

    if jv_number == 0:
        print('\nThere is no object in the given JV image/slice!\n')
    else:
        for contour in contours:
            # Create a blank mask for the current contour
            blank = np.zeros((flair_image.shape[0], flair_image.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(blank, [contour], -1, 1, -1)  # Fill the contour with value 1
            blank = np.where(blank > 0, 1, 0).astype(np.uint8)[..., 0]  # Create binary mask

            # Calculate the area in real-world units (e.g., mm²)
            area_j = np.round(np.sum(blank) * voxel_size[0] * voxel_size[1], 3)
            jv_area.append(area_j)

    ## Analyzing the WMH
    contours, _ = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # the number of found objects
    number = len(contours)
    print(f"                                          Number of Objects: {number}")

    # Prepare to go in a loop through each contour
    areas = []
    areas_code = []
    areas_int = []
    axes = []
    v_dist = []
    gm_dist = []
    csf_dist = []
    depth_dist = []

    peri_mask = np.zeros(bw_image.shape)
    para_mask = np.zeros(bw_image.shape)
    juxt_mask = np.zeros(bw_image.shape)

    # c_image = np.zeros((flair_image.shape[0], flair_image.shape[1], 3)).astype(np.uint8)
    # c_image[..., 0] = flair_image
    # c_image[..., 1] = flair_image
    # c_image[..., 2] = flair_image

    # vent_im = np.copy(c_image)
    # vent_im[..., 0] = 0 * vent_image
    # vent_im[..., 1] = 200 * vent_image
    # vent_im[..., 2] = 255 * vent_image
    # c_image = cv2.addWeighted(c_image, 1 - 0.1, vent_im, 0.5, 0)

    # gm_im = np.copy(c_image)
    # gm_im[..., 0] = 100 * gm_image
    # gm_im[..., 1] = 100 * gm_image
    # gm_im[..., 2] = 255 * gm_image

    if len(contours) == 0:
        print('\nThere is no object in the given image/slice!\n')
        pass

    else:
        for contour in contours:

            print("---------------------------------------------------------------------------------------------------")
            # Calculate the area & intensity index of the contour

            # Create a blank mask for the current contour
            blank = np.zeros((flair_image.shape[0], flair_image.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(blank, [contour], -1, 1, -1)  # Fill the contour with value 1
            blank = np.where(blank > 0, 1, 0).astype(np.uint8)[..., 0]  # Create binary mask

            # Calculate the area in real-world units (e.g., mm²)
            area = np.round(np.sum(blank) * voxel_size[0] * voxel_size[1], 3)
            areas.append(area)

            # Intensity Index of th Object
            obj_only = blank * (flair_image / 255.0)
            obj_int = obj_only                      # results an image-sized the normalized wmh image for that object

            obj_int = np.round(np.average(obj_int[obj_int > 0]), 5)  # results just a number for a given specific wmh.
            areas_int.append(obj_int)
            print(f"\nobject's intensity index : {obj_int}")

            # Fit an ellipse to the contour

            if len(contour) < 5:
                if len(contour) == 1:
                    major_axis = np.round(1 * voxel_size[0], 3)
                    minor_axis = np.round(1 * voxel_size[0], 3)
                    axes.append([major_axis, minor_axis])

                else:
                    j = 0
                    distance = []
                    for point in contour:
                        if j == 0:
                            j += 1
                            point0 = point
                            continue

                        # Calculate the distances from the center to each vertex
                        distance.append(
                            np.sqrt((point[..., 0] - point0[..., 0]) ** 2 + (point[..., 1] - point0[..., 1]) ** 2))
                    distance = np.array(distance)
                    major_axis = np.round(np.max(distance) * voxel_size[0], 3)
                    minor_axis = np.round(np.min(distance) * voxel_size[0], 3)
                    axes.append([major_axis, minor_axis])

            else:
                ellipse = cv2.fitEllipse(contour)
                # Get the major and minor axes of the fitted ellipse
                major_axis, minor_axis = ellipse[1]
                major_axis = np.round(max(major_axis, minor_axis) * voxel_size[0], 3)
                minor_axis = np.round(min(major_axis, minor_axis) * voxel_size[0], 3)
                axes.append([major_axis, minor_axis])

            # Compute Ventricular distance for the object of study:
            min_v_distance_cnt, min_v_distance, vent_cont = wmh_vent_distance(contour, vent_image, voxel_size[0])

            # Compute GrayMatter distance for the object of study:
            min_gm_distance_cnt, min_gm_distance, gm_cont = wmh_gm_distance(contour, gm_image, voxel_size[0])

            # Compute CSF distance for the object of study:
            min_csf_distance_cnt, min_csf_distance, csf_cont = wmh_csf_distance(contour, c_mask, voxel_size[0])

            # # Compute depth distance for the object of study:
            # First, make hemispharic brain masks:
            hemispharic_b_mask, _, _, _ = create_hemispheric_mask(
                b_mask, line_width=2, use_major_axis=True
            )
            depth_list = depth_distance(contour, hemispharic_b_mask)
            depth_list[0] = np.round(depth_list[0] * voxel_size[0], 2)

            # Print the area, axes, vent_distance, and GM_distance for each object
            print(f"Object Area: {area}")
            print(f"Major Axis Length: {max(major_axis, minor_axis)}")
            print(f"Minor Axis Length: {min(major_axis, minor_axis)}")
            print(f"Distance from the Ventricles: shortest ==> {min_v_distance} , centroid ==> {min_v_distance_cnt}")
            print(f"Distance from the GrayMatter: shortest ==> {min_gm_distance} , centroid ==> {min_gm_distance_cnt}")
            print(f"Distance from the CSF: shortest        ==> {min_csf_distance} , centroid ==> {min_csf_distance_cnt}")
            print(f"Distance from the Brain's Hemispharic Edges: {depth_list[0]}")

            # v_dist.append(min_v_distance)
            v_dist.append(min_v_distance_cnt)
            gm_dist.append(min_gm_distance)
            csf_dist.append(min_csf_distance_cnt)
            # gm_dist.append(min_gm_distance_cnt)
            depth_dist.append(depth_list[0])

            # Producing new image
            # color decision (type of the WMH):
            print("\n   min_v_distance_cnt: ", min_v_distance_cnt, "\n   min_gm_distance: ", min_gm_distance, "\n   min_csf_distance: ", min_csf_distance)
            code = obj_categorize(
                area, min_v_distance, min_v_distance_cnt, min_gm_distance, min_gm_distance_cnt, vent_rule, gm_rule
            )
            areas_code.append(code)
            if code > 0:
                if code == 1:
                    # it is a periventricular WMH
                    color = (255, 0, 0)
                    # adding the corresponding object to the premier blank mask
                    cv2.drawContours(peri_mask, [contour], -1, 1, -1)
                elif code == 2:
                    # it is a paraventricular WMH
                    color = (0, 255, 0)
                    # adding the corresponding object to the premier blank mask
                    cv2.drawContours(para_mask, [contour], -1, 1, -1)
                elif code == 3:
                    # it is a juxtacortical WMH
                    color = (255, 165, 0)
                    # adding the corresponding object to the premier blank mask
                    cv2.drawContours(juxt_mask, [contour], -1, 1, -1)

            # hull = cv2.convexHull(contour)
            # # cv2.drawContours(c_image, [hull], -1, color, 1)  # (0, 255, 0) is the color, 2 is the thickness
            # epsilon = 0.01 * cv2.arcLength(contour, True)
            # approx = cv2.approxPolyDP(contour, epsilon, True)
            # cv2.drawContours(c_image, [approx], -1, color,
            #                  -1)  # (0, 255, 0) is the color, 2 is the thickness

            # drawing vents on image:
            """for contour in vent_cont:
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(c_image, [approx], -1, (0, 200, 255),
                                 1)  # (0, 255, 0) is the color, 2 is the thickness"""

        # display flair after
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # skimage.io.imshow(c_image)
        # plt.subplot(1, 2, 2)
        # skimage.io.imshow(c_image_label)
        # skimage.io.imsave(save_dir + id + '_' + str(slc) + '_label.png', c_image_label)
        # skimage.io.imsave(save_dir + id + '_' + str(slc) + '_classi.png', c_image)
        # plt.show()

    return (peri_mask, para_mask, juxt_mask,
            areas_code, contours, 
            number, areas, areas_int, axes,
            v_dist, gm_dist, depth_dist, csf_dist,
            jv_number, jv_area)

