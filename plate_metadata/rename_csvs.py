import numpy as np
import pandas as pd

import os
import shutil

# Set your folder path here
folder = "/Users/allielas/HTP_fibroblasts_mitolyso/plate_metadata"

# Define the string(s) to find and their replacements
replacements = {
    #"T": "AG",
    #"R": "SPB",
    #"MitoSPBed" : "MitoRed",
    "CF650SPB" : "CF640R",
    #"AGfn" : "Tfn"
    # Add more as needed
}

for root, dirs, files in os.walk(folder):
    for filename in files:
        if filename.endswith(".csv") and "map" not in filename:
            path = os.path.join(root,filename)
            shutil.copy2(path, path + ".bak")
            with open(path,'r') as f:
                content = f.read()
                
            for old, new in replacements.items():
                content = content.replace(old, new)
                
            with open(path, "w") as f:
                f.write(content)
            print(f"Modified contents of: {path}")