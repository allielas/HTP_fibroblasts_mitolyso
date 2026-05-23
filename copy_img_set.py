import os
import shutil
import pandas as pd
import re


def get_location_code(file_name):
    match = re.search(r"(r\d{2}c\d{2}f\d{2})", file_name)
    if match:
        # print(match.group(0))  # Print the full match for debugging
        return match.group(1)
    else:
        return None


def get_plate_number(file_name):
    match = re.search(r"rep(\d{1,2})", file_name)
    if match:
        return match.group(1)
    else:
        return None


def copy_img_set(img_names, in_dir, out_dir):
    log = open("copy_img_set_log.txt", "w")  # Open a log file for writing
    img_location_codes = {
        get_location_code(name)
        for name in img_names
        if get_location_code(name) is not None
    }
    os.makedirs(out_dir, exist_ok=True)

    # Create a mapping dict from location codes to the indices of the images in the image set that have that location code
    code_to_indices = {}
    for i, name in enumerate(img_names):
        code = get_location_code(name)
        if code is not None:
            code_to_indices.setdefault(code, []).append(i)

    for root, _, files in os.walk(in_dir):
        for file_name in files:
            location_code = get_location_code(file_name)
            full_path = os.path.abspath(root)
            parent_dir = os.path.basename(full_path)
            # log.write(
            #     f"Processing file: {file_name} in folder: {parent_dir} with location code: {location_code} and parent directory: {parent_dir} \n"
            # )  # Print the file name, location code, and parent directory for debugging

            # compare the plate number in the file name to the plate number in the image set
            plate_number = get_plate_number(parent_dir)
            if plate_number is None:
                plate_number = get_plate_number(full_path)
                log.write(
                    f"Could not extract plate number from parent directory {parent_dir} for file {file_name}, trying full path {full_path} instead, got plate number {plate_number} \n"
                )
                if plate_number is None:
                    log.write(
                        f"Could not extract plate number from parent directory {parent_dir} or full path {full_path} for file {file_name} \n"
                    )

            matching_plate_flag = False

            if location_code in code_to_indices:
                matched_indices = code_to_indices[location_code]  # e.g. [3, 17, 42]

                for idx in matched_indices:
                    if get_plate_number(img_names[idx]) == plate_number:
                        matching_plate_flag = True
                    log.write(
                        f"Location code {location_code} found in {parent_dir}/{file_name}, plate {plate_number} at indices {matched_indices}; comparing plate number {plate_number} to {img_names[idx]}; matching_plate_flag = {matching_plate_flag} \n"
                    )
                    if matching_plate_flag:
                        break
            if location_code in img_location_codes and matching_plate_flag:
                src_image = os.path.join(root, file_name)
                dest_image = os.path.join(out_dir, file_name)
                log.write(f"Copying {src_image} to {dest_image} \n")
                # Print the source and destination paths for debugging
                shutil.copy2(src_image, dest_image)


if __name__ == "__main__":
    in_dir = "/mnt/bigdisk1/AllieSpangaro/Morphology_Replicative_Age_Project/MRC-5_MAX_SUM_PROJ"
    out_dir = "/mnt/bigdisk1/AllieSpangaro/MitoSegSubset"

    # Load the image sets from the CSV files
    df = pd.read_csv("/mnt/bigdisk1/AllieSpangaro/importing_mitomarkercsv.csv")
    image_colname = "Image_FileName_MitoTracker_MAX"
    plate_colname = "Plate_Number"
    df["plate_image_colname"] = (
        df[image_colname] + "_" + "rep0" + df[plate_colname].astype(str)
    )
    # Get the image file names from the dataframes
    img_names = df["plate_image_colname"].tolist()
    print(img_names)  # Print the list of image names for debugging
    print(
        f"plate_number {get_plate_number(img_names[1])} at index 1"
    )  # Print the plate number for debugging
    # Copy the images to the output directory
    copy_img_set(img_names, in_dir, out_dir)
