import os
import re
import shutil
import numpy as np

# from tqdm import trange
from cellpose_functions import *

# Configuration
input_folder = "/mnt/bigdisk1/AllieSpangaro/Morphology_Replicative_Age_Project/MRC-5_MAX_SUM_PROJ/20250328_rep05"
output_base = "/mnt/bigdisk1/AllieSpangaro/Morphology_Replicative_Age_Project/MRC-5_MAX_SUM_PROJ/20250328_rep05"

field_pattern = r"(f\d{2})"  # Matches "FXX" where XX are digits
block_1_numbers = [1] + list(range(22, 41))
block_2_numbers = list(range(2, 22))
# Remember you can reuse the file opening on cellpose if you move folders around - recurse with os.walk

file_list = load_sorted_directory_list(input_folder)
nchannels = get_nchannels(file_list)
grouped_files_by_channel = group_files_by_channel(file_list, nchannels)


def group_files_by_well(file_list, nfields=40, nchannels=4):
    """_summary_

    Args:
        file_list (_type_): _description_
        nfields (int, optional): _description_. Defaults to 40.

    Returns:
        _type_: _description_
    """
    grouped_files_by_well = []
    for i in range(0, len(file_list), nchannels * nfields):
        grouped_files_by_well.append(file_list[i : i + nchannels * nfields])
    return grouped_files_by_well


def get_well_stitching_block_paths(grouped_files_by_well):
    block_1_paths, block_2_paths = []
    # for i in range(grouped_files_by_well)

    return block_1_paths, block_2_paths


grouped_files_by_well = group_files_by_well(file_list, 40, nchannels)
print_grouped_files(grouped_files_by_well)


"""
for i in range(len(grouped_files_by_channel)):
      file_group = grouped_files_by_channel[i]
      img_set_name = get_image_set_name(file_group)
      

for fname in os.listdir(input_folder):
    match = re.search(pattern, fname)
    if match:
        set_id = match.group(1)
        images_by_set.setdefault(set_id, []).append(fname)

# Process each set
for set_id, files in images_by_set.items():
    files.sort()  # Ensure consistent order
    for i in range(0, len(files), images_per_new_folder):
        chunk = files[i:i+images_per_new_folder]
        folder_name = f"{set_id}_part{i//images_per_new_folder+1}"
        out_folder = os.path.join(output_base, folder_name)
        os.makedirs(out_folder, exist_ok=True)
        scheme_path = os.path.join(out_folder, 'scheme.txt')
        with open(scheme_path, 'w') as scheme_file:
            for fname in chunk:
                src = os.path.join(input_folder, fname)
                dst = os.path.join(out_folder, fname)
                shutil.copy2(src, dst)
                scheme_file.write(f"{fname}\n")"""

# Define the names for your two new folders
folder_a_name = "Folder_A"
folder_b_name = "Folder_B"

# Create the full paths for the new folders
folder_a_path = os.path.join(output_base, folder_a_name)
folder_b_path = os.path.join(output_base, folder_b_name)

# Create the new folders if they don't exist
os.makedirs(folder_a_path, exist_ok=True)
os.makedirs(folder_b_path, exist_ok=True)

# List to hold file names for Folder A (F02-F21)
files_for_folder_a = []
for i in range(2, 22):  # Range is exclusive, so 2 to 21
    files_for_folder_a.append(
        f"F{i:02d}.jpg"
    )  # Assuming .jpg extension, adjust if different

# List to hold file names for Folder B (F01, F22-F40)
files_for_folder_b = []
files_for_folder_b.append("F01.jpg")  # F01 explicitly
for i in range(22, 41):  # Range is exclusive, so 22 to 40
    files_for_folder_b.append(
        f"F{i:02d}.jpg"
    )  # Assuming .jpg extension, adjust if different

# Move files to Folder A
print(f"Moving files to {folder_a_name}...")
for filename in files_for_folder_a:
    source_file_path = os.path.join(source_directory, filename)
    destination_file_path = os.path.join(folder_a_path, filename)
    if os.path.exists(source_file_path):
        shutil.move(source_file_path, destination_file_path)
        print(f"Moved {filename} to {folder_a_name}")
    else:
        print(f"Warning: {filename} not found in {source_directory}")

# Move files to Folder B
print(f"\nMoving files to {folder_b_name}...")
for filename in files_for_folder_b:
    source_file_path = os.path.join(source_directory, filename)
    destination_file_path = os.path.join(folder_b_path, filename)
    if os.path.exists(source_file_path):
        shutil.move(source_file_path, destination_file_path)
        print(f"Moved {filename} to {folder_b_name}")
    else:
        print(f"Warning: {filename} not found in {source_directory}")

print("\nFile organization complete!")

from stitching import Stitcher


def stitch_image_block(file_list, output_path):
    """Stitches a list of images and saves the result."""
    print(f"Stitching {len(file_list)} images to {output_path}...")
    stitcher = Stitcher()  # Add your settings here
    panorama = stitcher.stitch(file_list)
    # Save the image using a library like OpenCV
    # cv2.imwrite(output_path, panorama)
