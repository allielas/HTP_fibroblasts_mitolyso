import os
import numpy as np
import pandas as pd
import sqlite3

#Debugging functions
def add_drug_to_group(init_df, group, drug):
    '''
        Add the name of a drug treatment from the "Drug" column to the main "group" column
        
        Returns
            Series object: A series containing the column with the drug added to the group
    '''
    if drug is not None:
        # Replace values in 'col1' with values from 'col2' only if 'col2' is not None or NaN
        df = init_df.copy()
        df[group] = np.where(df[drug].notna(), df[drug], df[group])
        newcol = df[group]

    return newcol

#display(cell_df[["Cell_Mean_Nuclei_AreaShape_Area", "Cell_AreaShape_Area"]])

def cell_filters(df):
    not_empty_df = df[df["Metadata_EmptyImage_Cell"] == 0] 
    normal_cells = not_empty_df[not_empty_df["Cell_Classify_Normal"] ==1] #one nucleus only
    size_filtered_cells = normal_cells[df["Cell_AreaShape_Area"] > df["Cell_Mean_Nuclei_AreaShape_Area"]] #cell area bigger than nuclear area
    final_df = size_filtered_cells.reset_index(drop=True)
    return final_df

def multinucleate_cells(df):
    multinuc_df = df[df["Cell_Classify_multinucleate"] ==1]
    return multinuc_df

def calculate_median_object_features(parent_df, object_df, feature, parent_key, child_key="Cell_Number_Object_Number"):
    """
    Calculate the median of specified features grouped by a parent key.
    
    Parameters:
    df (DataFrame): The DataFrame containing the features.
    object_df (DataFrame): The DataFrame containing the object features.
    feature (str): The feature column to calculate the median for.
    child_key (str): The column name that serves as the child key.
    parent_key (str): The column name that serves as the parent key.

    Returns:
    modified_df (DataFrame): The DataFrame with the new median feature column added.
    
    """
    median_values = object_df.groupby(parent_key)[feature].median()
    col_name = f"Cell_Median_{feature}"
    
    modified_df = parent_df.copy()
    modified_df[col_name] = parent_df[child_key].map(median_values)
    
    return modified_df
#group by parent key to find median  

def add_median_object_features_to_parent(parent_df, object_df, object_name, child_key="Cell_Number_Object_Number"):
    """
    Add median features from object_df to parent_df based on the specified feature and keys.
    
    Parameters:
    parent_df (DataFrame): The DataFrame containing the parent features.
    object_df (DataFrame): The DataFrame containing the object features.
    object_name (str): The object of interest.
    child_key (str): The column name that serves as the child key.

    Returns:
    modified_df (DataFrame): The DataFrame with the new median feature column added.
    
    """
    modified_df = parent_df.copy()
    feature_types = ["AreaShape", "Distance", "Intensity", "Location"]
    other_channels = ["DAPI", "LAMP1", "MitoTracker", "Phalloidin"]
    
    if "Lysosomes" in object_name:
        parent_key = "Lysosomes_Parent_Cell"
        other_channels.remove("LAMP1")
        
    elif "Mitochondria" in object_name:
        parent_key = "Mitochondria_Parent_Cell"
        other_channels.remove("MitoTracker")
    elif "Nuclei" in object_name:
        parent_key = "Nuclei_Parent_Cell"
        other_channels.remove("DAPI")
    else:
        raise ValueError(f"Unknown object name: {object_name}. Expected 'Lysosomes', 'Mitochondria', or 'Nuclei'.")
        
    
    for feature in object_df.columns:
        print("checking feature:", feature)
        # Exclude if matches any in other_channels, unless it also matches feature_types
        if (
            object_name not in feature
            or (any(ch in feature for ch in other_channels))
        ):
            print("oops, skipping:", feature)
            continue  # Skip the parent key column or non-feature columns
        if any(ft in feature for ft in feature_types):
            print("FEATURE TYPE MATCH:", feature)
            modified_df = calculate_median_object_features(modified_df, object_df, feature, parent_key, child_key)

    return modified_df

def exclude_borders(df, min_x=0.0, min_y=0.0, max_x=2160.0, max_y=2160.0, prefix="Cell_"):
    """
    Exclude rows in the DataFrame that have bounding box coordinates outside the specified limits.
    
    Parameters:
    df (DataFrame): The DataFrame containing the bounding box coordinates.
    min_x (float): Minimum x-coordinate for exclusion.
    min_y (float): Minimum y-coordinate for exclusion.
    max_x (float): Maximum x-coordinate for exclusion.
    max_y (float): Maximum y-coordinate for exclusion.

    Returns:
    DataFrame: Filtered DataFrame with rows excluded based on bounding box coordinates.
    """
    filtered_df = df[
        (df[f"{prefix}AreaShape_BoundingBoxMaximum_X"] < max_x) &
        (df[f"{prefix}AreaShape_BoundingBoxMaximum_Y"] < max_y) &
        (df[f"{prefix}AreaShape_BoundingBoxMinimum_X"] > min_x) &
        (df[f"{prefix}AreaShape_BoundingBoxMinimum_Y"] > min_y)
    ]
    return filtered_df

def well_namer(row, col):
    '''
        Convert row and column numbers to a well name in the format A01, B02, etc.
        
        Args:
            row (int): The row number (1-8)
            col (int): The column number (1-12)
        
        Returns:
            str: Well name in the format A01, B02, etc.
    '''
    well_name = str(chr(ord('@')+ row)) + str(col).rjust(2, '0')  #make the number have a left align, adding a zero
    return well_name

def add_well_metadata(image_df):
    '''
        Add well metadata to the image DataFrame.
        
        Args:
            image_df (DataFrame): DataFrame containing image metadata
        
        Returns:
            DataFrame: Updated DataFrame with well metadata
    '''
    image_df.columns = image_df.columns.str.replace(r'^Image_Metadata_', 'Metadata_', regex=True)
    image_df[["Metadata_WellRow","Metadata_WellColumn","Metadata_Field"]] = image_df["Image_URL_DAPI"].str.extract(r'r(\d{2})c(\d{2})f(\d{2}).tif')
    # Convert extracted columns to int
    image_df["Metadata_WellRow"] = image_df["Metadata_WellRow"].astype(int)
    image_df["Metadata_WellColumn"] = image_df["Metadata_WellColumn"].astype(int)
    image_df["Metadata_Field"] = image_df["Metadata_Field"].astype(int)
    # apply well namer function
    image_df["Metadata_Well"] = image_df.apply(lambda x: well_namer(x["Metadata_WellRow"], x["Metadata_WellColumn"]), axis=1)
        
    return image_df

def update_database_with_well_metadata(db_path):
    '''
        Update the database with well metadata.
        
        Args:
            db_path (str): Path to the database file
    '''
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Read Per_Image table
    image_df = pd.read_sql_query("SELECT * FROM Per_Image", conn)
    
    # Add well metadata
    updated_image_df = add_well_metadata(image_df)
    
    # Write updated DataFrame back to the database
    try:
        updated_image_df.to_sql('Per_Image', conn, if_exists='replace', index=False)
        print("Database updated successfully with well metadata.")
    except Exception as e:
        print(f"Error updating database: {e}")
    #cursor.execute("SELECT Metadata_Well FROM Per_Image LIMIT 5;")
    #cursor.fetchall()
    conn.close()



