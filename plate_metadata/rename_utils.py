import numpy as np
import pandas as pd

import os
from pathlib import Path
import shutil

# Set your folder path here
folder = Path(
    "/mnt/bigdisk1/AllieSpangaro/Senesence_Markers_Classification/Pilot_RawImages/20251010_4ChannelsSeperate/stack"
)
replacements_pilot_sheet = Path(
    "/mnt/bigdisk1/AllieSpangaro/HTP_fibroblasts_mitolyso/plate_metadata/20251001_pilot_metadata/Pilotplate_map.csv"
)
# Define the string(s) to find and their replacements
replacements = {
    # "T": "AG",
    # "R": "SPB",
    "_Ctrl_": "_",
    # "20250710_MRC5_Ki67+LMNB1-488_P12_Ctrl_": "20250710_MRC5_Ki67+LMNB1-488_P28_Ctrl_",
    # Add more as needed
}


def load_replacements_from_csv(csv_path, old_col="old", new_col="new"):
    """
    Load string replacements from a CSV file into a dictionary.

    Args:
        csv_path (str or Path): Path to CSV file with replacement mappings.
        old_col (str): Column name containing strings to find (default "old").
        new_col (str): Column name containing replacement strings (default "new").

    Returns:
        dict: Dictionary mapping old strings to new strings.

    Example CSV format:
        old,new
        _Ctrl_,_
        T,AG
        R,SPB
    """
    df = pd.read_csv(csv_path)

    if old_col not in df.columns or new_col not in df.columns:
        raise ValueError(f"CSV must contain '{old_col}' and '{new_col}' columns")

    # Convert to dictionary, handling NaN values
    replacements = df.set_index(old_col)[new_col].dropna().to_dict()

    # Convert all keys and values to strings
    replacements = {str(k): str(v) for k, v in replacements.items()}

    return replacements


def apply_replacements(df, mapping_dict):
    if not mapping_dict:
        return df.copy()
    # For performance: compile mapping items once
    items = list(mapping_dict.items())

    def replace_value(x):
        """
        Docstring for replace_value. Just a simple string replace

        :str x: the string to replace based on the dict
        """
        if isinstance(x, str):
            for old, new in items:
                if old:
                    x = x.replace(old, new)
            return x
        return x

    return df.applymap(replace_value)


def save_modified_csv(df, out_path):
    """_summary_

    Args:
        df (_type_): _description_
        out_path (_type_): _description_
    """
    out_df = df.copy()

    out_df.to_csv(out_path, index=False)
    print(f"Wrote replaced data to: {out_path}")


def rename_csv(folder, replacements):
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".csv") and "map" not in filename:
                path = os.path.join(root, filename)
                shutil.copy2(path, path + ".bak")
                with open(path, "r") as f:
                    content = f.read()

                for old, new in replacements.items():
                    content = content.replace(old, new)

                with open(path, "w") as f:
                    f.write(content)
                print(f"Modified contents of: {path}")


def replace_strings_and_save_new_csv(in_file, replacements_path, out_path):
    df = pd.read_csv(in_file)
    replacements = load_replacements_from_csv(replacements_path)
    new_df = apply_replacements(df, replacements)
    save_modified_csv(new_df, out_path)


def rename_path_in_list(path, replacements):
    """_summary_

    Args:
        path (_type_): _description_
        replacements (_type_): _description_
    """
    path_to_rename = Path(path)
    for old, new in replacements.items():
        name = path_to_rename.name
        if old in name and new not in name:
            oldpath = Path.joinpath(path_to_rename.parent, name)
            newname = name.replace(old, new)
            newpath = Path.joinpath(path_to_rename.parent, newname)
            if os.path.exists(oldpath):
                try:
                    Path.rename(oldpath, newpath)
                    print(f"Modified contents of: {oldpath}")
                except Exception as e:
                    print(f"Error:{e}; cannot rename {oldpath} to {path_to_rename}")


def rename_image_folders(folder, replacements, ext=".vsi", identifier=""):
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            if identifier in dir:
                dir_path = os.path.join(root, dir)
                rename_path_in_list(dir_path, replacements=replacements)

        for filename in files:
            if filename.endswith(ext) and identifier in filename:
                file_path = os.path.join(root, filename)
                shutil.copy2(file_path, file_path + ".bak")
                rename_path_in_list(file_path, replacements=replacements)


def rm_baks(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            print(file)
            if file.endswith(".bak"):
                filepath = os.path.join(root, file)
                os.remove(filepath)
                print(f"removed: {filepath} ")


def rename_files_based_on_metadata(
    root_folder,
    replacements_file,
    ext,
    match_col,
    prefix_cols,
    match_mode="contains",  # "contains" or "equals"
    recursive=True,
    backup=True,
    dry_run=True,
):
    """
    Add a prefix to a filename given a metadata spreadsheet with matching file coordinates

    Args:
        root_folder (str or Path): root folder to search.
        replacements_file (str): Metadata file with match and prefix columns.
        match_col (str): column name containing strings to match against filenames.
        prefix_col (str): column name containing prefixes to add.
        ext (str, optional): filter files by extension (e.g. ".vsi"). If None, all files considered.
        match_mode (str): "contains" (default) or "equals".
        case_sensitive (bool): whether matching is case sensitive (default False).
        recursive (bool): walk directories recursively (default True).
        backup (bool): create a .bak copy before renaming (default True).
        dry_run (bool): if True, only print planned actions, do not rename.
    """
    folder = Path(root_folder)
    df = pd.read_csv(replacements_file)
    print(df.columns)
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(ext):
                df["match_col"] = df[match_col].astype(str)
                df["prefix"] = df[prefix_cols].astype(str).agg("_".join, axis=1) + "_"

                for idx, row in df.iterrows():
                    match_val = row["match_col"]
                    prefix = row["prefix"]
                    matched = False
                    if match_mode == "contains" and match_val in filename:
                        matched = True
                    elif match_mode == "equals" and match_val == filename:
                        matched = True

                    if matched:
                        if filename.startswith(prefix):
                            break
                        oldpath = Path(root) / filename
                        newname = prefix + filename
                        newpath = oldpath.parent / newname

                        # avoid clobbering existing files
                        counter = 1
                        while newpath.exists():
                            stem = newpath.stem
                            suffix = newpath.suffix
                            newpath = (
                                oldpath.parent / f"{prefix}{stem}_{counter}{suffix}"
                            )
                            counter += 1

                        print(f"Rename: {oldpath} -> {newpath}")
                        if dry_run:
                            break

                        if backup:
                            shutil.copy2(str(oldpath), str(oldpath) + ".bak")
                        try:
                            oldpath.rename(newpath)
                        except Exception as e:
                            print(f"Failed to rename {oldpath}: {e}")
                        break  # move to next file after first matching row
            if not recursive:
                break
        # cleanup helper column
        if "match_col" in df.columns:
            df.drop(columns=["match_col"], inplace=True, errors="ignore")


coords = "Metadata_RowColFieldCode"
prefixes = ["TreatmentGroup", "ShortStaining"]
# rename_files_based_on_metadata(
#     root_folder=folder,
#     replacements_file=replacements_pilot_sheet,
#     ext=".tiff",
#     match_col=coords,
#     prefix_cols=prefixes,
#     match_mode="contains",
#     recursive=True,
#     backup=False,
#     dry_run=False,
# )

# rename_image_folders(
#     folder=folder, replacements=replacements, ext=".vsi", identifier="20250710"
# )
# rm_baks(folder=folder)

old_plate = "plate_metadata/20250501_rep07_metadata/may01plate.csv"
new_plate = "plate_metadata/20250501_rep07_metadata/new_may01plate.csv"
replace_strings_and_save_new_csv(
    old_plate, "plate_metadata/replacements.csv", new_plate
)
