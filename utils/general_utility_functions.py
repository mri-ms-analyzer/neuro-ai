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
import sys
import ctypes
import PyPDF2
import zipfile
import webbrowser
import numpy as np
import matplotlib.pyplot as plt


# %% General Utility Functions

def set_console_position_and_size(x, y, width, height):
    """Set the position and size of the console window."""
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if hwnd:
        # Set position
        ctypes.windll.user32.SetWindowPos(hwnd, 0, x, y, width, height, 0)

# 
def display_nii(data, step=1, timeout=1, start=0):
    print('there')

    for i in np.arange(start, data.shape[2], step):
        print(i)
        test = data[:, :, i]
        # if np.sum(test) == 0:
        #     continue

        # test = (test - np.max(test)) / (np.max(test) - np.min(test))
        # g = (g - np.max(g)) / (np.max(g) - np.min(g))

        # test = skimage.transform.rotate(test, -90)
        # g = skimage.transform.rotate(g, -90)

        plt.figure(i + 1)
        plt.imshow(test, cmap='gray')
        plt.axis('off')
        plt.pause(timeout)
        plt.show(block=False)
        plt.close()
        print('in for', test.dtype)
        print(np.max(test), np.min(test))
        print(test.shape)
    return

# 
def display_nii_1(data, step=1, timeout=1, start=0):
    print('there')
    test = data[:, :]
    # test = (test - np.max(test)) / (np.max(test) - np.min(test))
    # g = (g - np.max(g)) / (np.max(g) - np.min(g))

    # test = skimage.transform.rotate(test, -90)
    # g = skimage.transform.rotate(g, -90)

    plt.figure(1)
    plt.imshow(test, cmap='gray')
    plt.axis('off')
    plt.pause(timeout)
    plt.show(block=False)
    plt.close()
    print('in for', test.dtype)
    print(np.max(test), np.min(test))
    print(test.shape)
    return

# Function to get the correct path for bundled resources
def get_resource_path(relative_path):

    if getattr(sys, 'frozen', False):  # Running as a bundled executable
        base_path = sys._MEIPASS

    else:  # Running as a script

        try:
            # Try to use __file__ for script-based execution
            base_path = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Fallback for environments where __file__ is not available
            base_path = os.getcwd()

    return os.path.join(base_path, relative_path)

# Finding the understudy folder or directory
def find_most_recent_folder(directory):
    """Find the most recently created folder in the given directory."""
    # Get all items in the directory with their full paths
    entries = [os.path.join(directory, entry) for entry in os.listdir(directory)]
    
    # Filter out folders only
    folders = [entry for entry in entries if os.path.isdir(entry)]
    
    if not folders:
        return None  # No folders found in the directory
    
    # Sort folders by creation time (newest first)
    most_recent_folder = max(folders, key=os.path.getctime)
    return most_recent_folder

#
def final_pdf_report(report_path):    

    def merge_pdfs(pdf_list, output_path):
        # Create a PDF merger object
        pdf_merger = PyPDF2.PdfMerger()
        
        # Append each PDF to the merger
        for pdf in pdf_list:
            pdf_merger.append(pdf)
        
        # Write the merged PDF to the output file
        with open(output_path, 'wb') as output_pdf:
            pdf_merger.write(output_pdf)

    # List of PDF files to merge (add the file paths here)

    pdf_files = [os.path.join(report_path, file) for file in os.listdir(report_path) if (file.startswith('p') and file.endswith('.pdf'))]
    pdf_files = sorted(pdf_files)

    # Output merged PDF file
    output_pdf = os.path.join(os.path.dirname(pdf_files[0]), f"{len(pdf_files)}_PagesReport.pdf")

    # Call the function to merge PDFs
    merge_pdfs(pdf_files, output_pdf)

    print(f'Merged PDF saved to {output_pdf}')
    return output_pdf

#
def open_pdf_browser(report_path):    

    def in_browser(pdf_files):
        for pdf_file in pdf_files:
            """Open the generated PDF in the default browser."""
            webbrowser.open(pdf_file)

    # List of PDF files to merge (add the file paths here)

    pdf_files = [os.path.join(report_path, file) for file in os.listdir(report_path) if file.endswith('PagesReport.pdf')]
    pdf_files = sorted(pdf_files)

    # Call the function to open PDF(s)
    in_browser(pdf_files)

#
def retrieve_folders_by_national_id(directory, national_id):
    """
    Retrieve unique patient folders for a given national ID. 
    For repeated patient IDs, pick the folder with the most recent datetime.
    Sort the results descendingly by the datetime component.
    
    Args:
        directory (str): Path to the directory containing folders.
        national_id (str): The national ID to filter folders.
    
    Returns:
        list: A list of folder paths sorted by ascending datetime.
    """
    # Filter folders matching the national ID pattern
    matching_folders = [
        folder for folder in os.listdir(directory)
        if folder.startswith(f"{national_id}_") and os.path.isdir(os.path.join(directory, folder))
    ]
    
    if not matching_folders:
        print(f"No folders found for national ID: {national_id}")
        return []
    
    # Parse folder details into a list of tuples: (folder_path, patient_id, datetime)
    folder_details = []
    for folder in matching_folders:
        try:
            parts = folder.split("_")
            patient_id = parts[-2]
            datetime_str = parts[-1]
            folder_details.append((os.path.join(directory, folder), patient_id, int(datetime_str)))
        except (IndexError, ValueError):
            print(f"Skipping malformed folder name: {folder}")
    
    # Group folders by patient ID and select the newest one by datetime
    selected_folders = {}
    for folder_path, patient_id, datetime_value in folder_details:
        if patient_id not in selected_folders or datetime_value > selected_folders[patient_id][1]:
            selected_folders[patient_id] = (folder_path, datetime_value)
    
    # Extract folder paths and sort descendingly by datetime
    sorted_folders = sorted(
        [data[0] for data in selected_folders.values()],
        key=lambda x: int(x.split('_')[-1]),  # Extract datetime from the folder name
        reverse=False  # Ascending order
    )
    
    return sorted_folders

#
def create_zip_file(directory_path, zip_file_path):
    """Create a ZIP file from a directory."""
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"The directory '{directory_path}' does not exist.")

    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, start=directory_path)
                zipf.write(full_path, arcname)

#
def get_default_history_directory():
    """Get the default directory for saving history."""
    try:
        exe_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    except NameError:
        exe_dir = os.getcwd()
    history_dir = os.path.join(exe_dir, "MRI_MS_Analyzer_History")
    os.makedirs(history_dir, exist_ok=True)
    return history_dir

#
def error_window(error_msg):
    raise ValueError(error_msg)
    status_msg ="Analysis Failed"

    return status_msg, error_msg

#
def completion_window():
    status_msg = "Analyses Completed Successfully!"

    return status_msg
