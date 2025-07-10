import os
import re
import shutil
import numpy as np
from pathlib import Path

# from tqdm import trange
# Configuration

def file_sort_key(filename):
  '''
    Generate a key to sort a list of image files in the 'MAX_chN-rXXcYYfZZ.tif' filename nomenclature by their plate location and channel.
    > e.g: MAX_ch1-r02c02f01.tif, MAX_ch2-r02c02f01.tif, MAX_ch3-r02c02f01.tif, MAX_ch3-r02c02f01.tif, MAX_ch1-r02c02f02.tif, MAX_ch2-r02c02f02.tif, MAX_ch3-r02c02f02.tif, MAX_ch3-r02c02f02.tif
    Parameters:
          filename (str): the filename from the image path
    Returns:
          (location,channel) (tuple of str): the list of files sorted by location and channel. Images will be ordered by location first, and then by channel to match the order seen in CellProfiler  
  '''
  parts = filename.split("-")
  channel = parts[0][-1:] # get the last character of the first part for the channel number
  location = parts[1] #get the rXXcYYfZZ.tif portion
  return (location,channel)

def sort_files(dir, image_ext):
  '''
    Sort the list of directories sorted by their plate location and channel 
    Parameters:
          dir (Path object or str): the directory containing the images
          image_ext (str, optional): the image extension, tif by default
    Returns:
          files (list of Path objects): the list of files sorted by location and channel  
  '''
  if not dir.exists():
    raise FileNotFoundError("directory does not exist")
  files = sorted([f for f in dir.glob("*"+image_ext) if "_masks" not in f.name and "_flows" not in f.name and "SUM" not in f.name],
                           key=lambda x: file_sort_key(x.name))
 # sort by number in filename
  if(len(files)==0):
    raise FileNotFoundError("no image files found, did you specify the correct folder and extension?")
  else:
    return files
  
def print_files(files):
  '''
    Print a list of filenames from a list of Path objects
    Parameters:
            grouped_files_by_channel (2D list of Path objects): list of files grouped into image sets by their channels
  '''
  for f in files:
    print(f.name)

def load_sorted_directory_list(dir, image_ext=".tif"):
    '''
    Load the list of directories sorted by their location and channel to allow grouping into a 2D list and force the objects to be Path objects
    Parameters:
          dir (Path object or str): the directory containing the images
          image_ext (str, optional): the image extension, tif by default
    Returns:
          file_list (list of Path objects): the list of files sorted by location and channel  
    '''
    dir = Path(dir) 
    file_list = sort_files(dir, image_ext)
    return file_list
  
def get_nchannels(ordered_files):
  '''
    Get the number of channels based on the last element list element in a list of image files ordered by channels
    Parameters:
          ordered_files (1D list of Path objects): list of files grouped into image sets by their channels
    Returns:
          nchannels (int): the number of channels based on the length of the first element
  '''
  lastitem = ordered_files[-1].name
  parts = lastitem.split("-")
  ch_char = parts[0][-1:]
  nchannels = int(ch_char)
  return nchannels
      
def group_files_by_channel(files, nchannels=None):
  '''
    Load the list of directories
    Parameters:
          dir (Path object or str): the directory containing the images
          nchannels (int,optionsl): the number of image channels to use for grouping
    Returns:
          grouped_files_by_channel (2D list of Path objects): list of files grouped into image sets by their channels
  '''
  grouped_files_by_channel = []
  if nchannels==None:
    nchannels = get_nchannels(files)
    
  for i in range(0,len(files),nchannels):
    grouped_files_by_channel.append(files[i:i+nchannels])
  return grouped_files_by_channel

def print_grouped_files(grouped_files_by_channel):
  '''
    Print a list of files grouped by channel to confirm that the images are loaded in the correct order
    Parameters:
            grouped_files_by_channel (2D list of Path objects): list of files grouped into image sets by their channels
  '''
  for i in range(len(grouped_files_by_channel)):
    print(f"\n Group {i+1} of {len(grouped_files_by_channel)}")
    for j in range(len(grouped_files_by_channel[i])):
      item = grouped_files_by_channel[i][j]
      print(" "+ item.name)

def get_plate_location(filepath, rowcolonly=False):
    field_pattern = r"f(\d{2})"  # Matches "FXX" where XX are digits
    filename = os.path.basename(filepath)
    parts = filename.split("-")
    channel = parts[0][-1:] # get the last character of the first part for the channel number
    location_withextension = parts[1]
    location = location_withextension.split(".")[0]
    if rowcolonly:
      return location[:6]
    else:
      return location

def block_a_order_key(filepath):
    location = get_plate_location(filepath)
    field_pattern = r"f(\d{2})"  # Matches "FXX" where XX are digits
    match = re.search(field_pattern, location)
    if match:
        field = int(match.group(1))
        # Place f01 after f25 but before f26
        if field == 1:
            field_order = 25.5
        else:
            field_order = float(field)
    else:
        # If no match, put at the end
        field_order = float('inf')
    #print (filename, field_order)
    return field_order


# Remember you can reuse the file opening on cellpose if you move folders around - recurse with os.walk
""" input_folder = "/mnt/bigdisk1/AllieSpangaro/Morphology_Replicative_Age_Project/MRC-5_MAX_SUM_PROJ/20250328_rep05"
output_base = "/mnt/bigdisk1/AllieSpangaro/Morphology_Replicative_Age_Project/MRC-5_MAX_SUM_PROJ/20250328_rep05"
block_a_numbers = [1] + list(range(22, 41))
block_b_numbers = list(range(2, 22))
file_list = load_sorted_directory_list(input_folder)
nchannels = get_nchannels(file_list)
grouped_files_by_channel = group_files_by_channel(file_list, nchannels)
print_grouped_files(grouped_files_by_channel) """

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

def well_namer(row, col):
    well_name = str(chr(ord('@')+ row)) + str(col).rjust(2, '0')  #make the number have a left align, adding a zero
    return well_name

def get_well_stitching_block_paths(grouped_files_by_well, debug = True):
    """_summary_

    Args:
        grouped_files_by_well (_type_): _description_

    Returns:
        list: 2D lists for block A and B
    """    
    field_pattern = r"f(\d{2})"  # Matches "FXX" where XX are digits
    block_a_numbers = [1] + list(range(22, 41))
    block_b_numbers = list(range(2, 22))

    files_for_folder_a = []
    files_for_folder_b = []
    for well_idx, well in enumerate(grouped_files_by_well, start=1):
        well_list_a = []
        well_list_b = []
        for i, curr_file in enumerate(well):
            match = re.search(field_pattern, os.path.basename(curr_file))
            if match:
                field_number = int(match.group(1))
                if field_number in block_a_numbers:
                    well_list_a.append(curr_file)
                elif field_number in block_b_numbers:
                    well_list_b.append(curr_file)
            else:
                print(f"Error: filename doesn't have a field number: {curr_file}")
        
        # Sort block a and append per-well lists into the main list after collecting all for this well
        sorted_a_list = sorted(well_list_a, key=block_a_order_key)
        if debug:
            print(f"Well {well_idx}: Block A sorted order:")
            for f in sorted_a_list:
                print(f"    {os.path.basename(f)}")
                
            print(f"Well {well_idx}: Block B sorted order:")
            for f in well_list_b:
                print(f"    {os.path.basename(f)}")
        files_for_folder_a.append(sorted_a_list)
        files_for_folder_b.append(well_list_b)
        print(len(files_for_folder_a),len(files_for_folder_b))
    return files_for_folder_a, files_for_folder_b

def make_folders_and_move_files_by_scheme(grouped_files_by_well, block_a_files, block_b_files): 
    
    for groups in grouped_files_by_well:
        plate_location = get_plate_location(groups[0])
        location_base_folder = os.path.join(os.path.basename(groups), plate_location)
        os.makedirs(location_base_folder, exist_ok=True)
        
        folder_a_path = os.path.join(location_base_folder, "Block_A")
        folder_b_path = os.path.join(location_base_folder, "Block_B")
        # Create the new folders if they don't exist
        os.makedirs(folder_a_path, exist_ok=True)
        os.makedirs(folder_b_path, exist_ok=True)

        #loop thru all files in grouped by well
        for file in groups:
            filename = os.path.basename(file)
            if os.path.exists(file):
                if file in block_a_files:
                    shutil.move(file, folder_a_path)
                    print(f"Moved {filename} to {folder_a_path}")
                elif file in block_b_files:
                    shutil.move(file, folder_b_path)
                    print(f"Moved {filename} to {folder_b_path}")
            else:
                print(f"Warning: {filename} not found in {os.path.basename(groups)}")

    
    print("\nFile organization complete!")
    