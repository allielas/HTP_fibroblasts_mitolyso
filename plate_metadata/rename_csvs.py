import numpy as np
import pandas as pd

import os
from pathlib import Path
import shutil

# Set your folder path here
folder = Path(
    "/media/mattiazzilab/AllieS/Microscopy Images/20250710_MRC5_IF_Ki67+LMNB1/P28"
)

# Define the string(s) to find and their replacements
replacements = {
    # "T": "AG",
    # "R": "SPB",
    "_Ctrl_": "_",
    # "20250710_MRC5_Ki67+LMNB1-488_P12_Ctrl_": "20250710_MRC5_Ki67+LMNB1-488_P28_Ctrl_",
    # Add more as needed
}


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


rename_image_folders(
    folder=folder, replacements=replacements, ext=".vsi", identifier="20250710"
)
rm_baks(folder=folder)
