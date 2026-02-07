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
import numpy as np
import pydicom as dc
import nibabel as nib
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# %% Specific Utility Functions

#
def dicom_to_nifti(dicom_dir, output_path):
    # Read all DICOM files
    dicom_files = []
    for file in Path(dicom_dir).glob("*.dcm"):
        ds = dc.dcmread(file)
        dicom_files.append((ds.ImagePositionPatient[2], ds))  # Sort by Z position
    
    # Sort by slice position
    dicom_files.sort(key=lambda x: x[0])
    
    # Extract pixel arrays
    slices = [ds.pixel_array for _, ds in dicom_files]
    volume = np.stack(slices, axis=-1)
    
    # Get voxel spacing and create affine matrix
    ds = dicom_files[0][1]  # First slice for metadata
    pixel_spacing = ds.PixelSpacing
    slice_thickness = ds.SliceThickness
    
    # Create basic affine (you may need to adjust for proper orientation)
    affine = np.eye(4)
    affine[0, 0] = pixel_spacing[0]
    affine[1, 1] = pixel_spacing[1]
    affine[2, 2] = slice_thickness
    
    # Create and save NIfTI
    nii_img = nib.Nifti1Image(volume, affine)
    nib.save(nii_img, output_path)

#
def concatenate_images_horizontally(img1_path, img2_path, output_path):
    """
    Concatenate two images horizontally and save the result.
    
    Parameters:
    - img1_path: path to the first image (left)
    - img2_path: path to the second image (right)
    - output_path: path to save the concatenated image
    
    Returns:
    - output_path if successful, None otherwise
    """
    try:
        # Load images
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # Get dimensions
        width1, height1 = img1.size
        width2, height2 = img2.size
        
        # Create new image with combined width and max height
        max_height = max(height1, height2)
        total_width = width1 + width2
        
        # Create new image with white background
        combined_img = Image.new('RGB', (total_width, max_height), 'white')
        
        # Paste images (center vertically if heights differ)
        y_offset1 = (max_height - height1) // 2
        y_offset2 = (max_height - height2) // 2
        
        combined_img.paste(img1, (0, y_offset1))
        combined_img.paste(img2, (width1, y_offset2))
        
        # Save the result
        combined_img.save(output_path, quality=95)
        print(f"Created concatenated image: {output_path}")
        
        # Close images
        img1.close()
        img2.close()
        
        return output_path
        
    except Exception as e:
        print(f"Error concatenating {img1_path} and {img2_path}: {str(e)}")
        return None

#    
def create_2x2_layout(top_left_path, top_right_path, bottom_left_path, bottom_right_path, output_path):
    """
    Create a 2x2 layout of images.
    
    Parameters:
    - top_left_path: path to top-left image
    - top_right_path: path to top-right image  
    - bottom_left_path: path to bottom-left image
    - bottom_right_path: path to bottom-right image
    - output_path: path to save the combined image
    """
    try:
        # Load all images
        img_tl = Image.open(top_left_path)      # All_Plaque_*_tp0.png (1200x2400)
        img_tr = Image.open(top_right_path)     # BoxPlot_Plaque_*_tp0.png (1200x1200)
        img_bl = Image.open(bottom_left_path)   # Category_Plaque_*_tp0.png (1200x2400)
        img_br = Image.open(bottom_right_path)  # Pie_Plaque_*_tp0.png (1200x1200)
        
        # Get dimensions
        w_tl, h_tl = img_tl.size  # 1200x2400
        w_tr, h_tr = img_tr.size  # 1200x1200
        w_bl, h_bl = img_bl.size  # 1200x2400
        w_br, h_br = img_br.size  # 1200x1200
        
        # Calculate total dimensions
        # Top row: max height of All_Plaque (2400) and BoxPlot (1200) = 2400
        # Bottom row: max height of Category (2400) and Pie (1200) = 2400
        # Total width: 1200 + 1200 = 2400
        # Total height: 2400 + 2400 = 4800
        
        top_row_height = max(h_tl, h_tr)  # 2400
        bottom_row_height = max(h_bl, h_br)  # 2400
        total_width = w_tl + w_tr  # 2400
        total_height = top_row_height + bottom_row_height  # 4800
        
        # Create new image
        combined_img = Image.new('RGB', (total_width, total_height), 'white')
        
        # Calculate positions for centering images vertically within their rows
        # Top row
        y_tl = 0  # All_Plaque is full height, so it starts at 0
        y_tr = (top_row_height - h_tr) // 2  # Center BoxPlot vertically
        
        # Bottom row  
        y_bl = top_row_height  # Category starts at the beginning of bottom row
        y_br = top_row_height + (bottom_row_height - h_br) // 2  # Center Pie vertically
        
        # Paste images
        combined_img.paste(img_tl, (0, y_tl))           # Top-left
        combined_img.paste(img_tr, (w_tl, y_tr))        # Top-right
        combined_img.paste(img_bl, (0, y_bl))           # Bottom-left  
        combined_img.paste(img_br, (w_bl, y_br))        # Bottom-right
        
        # Save the result
        combined_img.save(output_path, quality=95)
        print(f"Created: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error creating 2x2 layout for {output_path}: {str(e)}")
        return False

#
def create_2x2_layout_triple(top_path, bottom_left_path, bottom_right_path, output_path):
    """
    Create a 2x2 layout of images.
    
    Parameters:
    - top_path: path to top-left image
    - bottom_left_path: path to bottom-left image
    - bottom_right_path: path to bottom-right image
    - output_path: path to save the combined image
    """
    try:
        # Load all images
        img_t = Image.open(top_path)            # Category_*_Area_tp0.png (1200x2400)
        img_bl = Image.open(bottom_left_path)   # Category_*_Area_tp0.png (1200x2400)
        img_br = Image.open(bottom_right_path)  # Category_*_Area_tp0.png (1200x2400)
        
        # Get dimensions
        w_t, h_t = img_t.size  # 1200x2400
        w_bl, h_bl = img_bl.size  # 1200x2400
        w_br, h_br = img_br.size  # 1200x1200
        
        # Calculate total dimensions
        # Top row: max height of All_Plaque (2400) and BoxPlot (1200) = 2400
        # Bottom row: max height of Category (2400) and Pie (1200) = 2400
        # Total width: 1200 + 1200 = 2400
        # Total height: 2400 + 2400 = 4800
        
        top_row_height = max(h_t, h_t)  # 2400
        bottom_row_height = max(h_bl, h_br)  # 2400
        total_width = w_t + w_t  # 2400
        total_height = top_row_height + bottom_row_height  # 4800
        
        # Create new image
        combined_img = Image.new('RGB', (total_width, total_height), 'white')
        
        # Calculate positions for centering images vertically within their rows
        # Top row
        y_tl = 0  # All_Plaque is full height, so it starts at 0
        y_tr = (top_row_height - h_t) // 2  # Center BoxPlot vertically
        
        # Bottom row  
        y_bl = top_row_height  # Category starts at the beginning of bottom row
        y_br = top_row_height + (bottom_row_height - h_br) // 2  # Center Pie vertically
        
        # Paste images
        combined_img.paste(img_t, (w_t//2, y_tl))      # Top-left
        combined_img.paste(img_bl, (0, y_bl))           # Bottom-left  
        combined_img.paste(img_br, (w_bl, y_br))        # Bottom-right
        
        # Save the result
        combined_img.save(output_path, quality=95)
        print(f"Created: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error creating 2x2 layout for {output_path}: {str(e)}")
        return False

#
def concat_resulting_images(directory_path):
    """
    Process all medical images according to the specified requirements.
    
    Parameters:
    - directory_path: path to the directory containing the images
    """
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist!")
        return
    
    # Create output directory for combined images
    # output_dir = os.path.join(directory_path, "combined_images")
    # os.makedirs(output_dir, exist_ok=True)
    output_dir = directory_path
    
    print("Processing medical images...")
    print("=" * 50)
    
    # 1. Handle Number feature (simple horizontal concatenation)
    print("1. Processing Number feature...")
    category_number = os.path.join(directory_path, "Category_Plaque_Number_tp0.png")
    pie_number = os.path.join(directory_path, "Pie_Plaque_Number_tp0.png")
    output_number = os.path.join(output_dir, "Combined_Number.png")
    
    if os.path.exists(category_number) and os.path.exists(pie_number):
        concatenate_images_horizontally(category_number, pie_number, output_number)
    else:
        print("   Warning: Number images not found!")
    
    # 2. Handle other features (Area, Intensity, Penetration, Depth) with 2x2 layout
    features = ["Area", "Intensity", "Penetration", "Depth"]
    
    for feature in features:
        print(f"\n2. Processing {feature} feature...")
        
        # Define file paths
        all_plaque = os.path.join(directory_path, f"All_Plaque_{feature}_tp0.png")
        boxplot = os.path.join(directory_path, f"BoxPlot_Plaque_{feature}_tp0.png")
        category = os.path.join(directory_path, f"Category_Plaque_{feature}_tp0.png")
        pie = os.path.join(directory_path, f"Pie_Plaque_{feature}_tp0.png")
        
        output_combined = os.path.join(output_dir, f"Combined_{feature}.png")
        
        # Check if all required files exist
        files_to_check = [all_plaque, boxplot, category, pie]
        missing_files = [f for f in files_to_check if not os.path.exists(f)]
        
        if missing_files:
            print(f"   Warning: Missing files for {feature}:")
            for missing in missing_files:
                print(f"      - {os.path.basename(missing)}")
            continue
        
        # Create 2x2 layout
        # Top row: All_Plaque (left) + BoxPlot (right)
        # Bottom row: Category (left) + Pie (right)
        create_2x2_layout(all_plaque, boxplot, category, pie, output_combined)
    
    # 3. Handle other features (Ventricles, CSF, Normal WMH)
    features = ["Ventricles", "CSF", "Normal Hyperintensities"]

    print(f"\n3. Processing Triple Image...")
    
    # Define file paths
    category1 = os.path.join(directory_path, f"Category_{features[0]}_Area_tp0.png")
    category2 = os.path.join(directory_path, f"Category_{features[1]}_Area_tp0.png")
    category3 = os.path.join(directory_path, f"Category_{features[2]}_Area_tp0.png")
    
    output_combined = os.path.join(output_dir, f"Combined_Triplet.png")
    
    # Check if all required files exist
    files_to_check = [category1, category2, category3]
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    
    if missing_files:
        print(f"   Warning: Missing files:")
        for missing in missing_files:
            print(f"      - {os.path.basename(missing)}")
    
    else:
        # Create 2x2 layout
        # Top row: category1
        # Bottom row: category2 (left) + category3 (right)
        create_2x2_layout_triple(category1, category2, category3, output_combined)
        
        print("\n" + "=" * 50)
        print("Processing completed!")
        print(f"Combined images saved in: {output_dir}")
    
    # List the created files
    created_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    if created_files:
        print(f"\nCreated combined images. The all images in result folder are:")
        for file in sorted(created_files):
            print(f"  - {file}")
  
#  
def display_results(directory_path):
    """
    Display the created combined images for verification.
    """
    output_dir = os.path.join(directory_path, "combined_images")
    
    if not os.path.exists(output_dir):
        print("No combined images directory found!")
        return
    
    # Get all PNG files in the output directory
    combined_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    combined_files.sort()
    
    if not combined_files:
        print("No combined images found!")
        return
    
    # Display images
    n_images = len(combined_files)
    cols = 2
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8 * rows))
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, filename in enumerate(combined_files):
        row = i // cols
        col = i % cols
        
        img_path = os.path.join(output_dir, filename)
        img = Image.open(img_path)
        
        if rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
            
        ax.imshow(img)
        ax.set_title(filename, fontsize=12)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].axis('off')
        else:
            axes[col].axis('off')
    
    plt.tight_layout()
    plt.show()
