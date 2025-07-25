import os
import numpy as np
import pandas as pd
import sqlite3


# Debugging functions
def passage_group(passage_num):
    """
    Group passages into bins for plotting
    returns string of the group that the passage number belongs to
    """
    # use this function to group passages into groups for plotting
    passage = int(passage_num)
    if 6 <= passage <= 10:
        return "P6-10"
    elif 11 <= passage <= 13:
        return "P11-13"
    elif 14 <= passage <= 16:
        return "P14-16"
    elif 17 <= passage <= 19:
        return "P17-19"
    elif 20 <= passage <= 22:
        return "P20-22"
    elif 23 <= passage <= 25:
        return "P23-25"
    elif 26 <= passage <= 28:
        return "P26-28"
    elif passage >= 29:
        return "P29+"
    else:
        return "Unknown"


def add_drug_to_group(init_df, group, drug):
    """
    Add the name of a drug treatment from the "Drug" column to the main "group" column

    Returns
        Series object: A series containing the column with the drug added to the group
    """
    if drug is not None:
        # Replace values in 'col1' with values from 'col2' only if 'col2' is not None or NaN
        df = init_df.copy()
        df[group] = np.where(df[drug].notna(), df[drug], df[group])
        newcol = df[group]

    return newcol


def get_all_group_order():
    """
    Get the order of the passage groups for plotting
    Returns a list of the passage groups in order
    Returns:
        list: A list of strings representing the group order
    """
    order = [
        "P6-10",
        "P11-13",
        "P14-16",
        "P17-19",
        "P20-22",
        "P23-25",
        "P26-28",
        "P29+",
        "Doxo",
    ]
    return order


def enforce_objects_one_to_one(df, parent_obj="Cell", child_obj="Nuclei"):
    """apply filters based on the number of nuclei to remove:
    - Cells that have less or more than one nucleus
    - Poorly segmented cells where the cell area is smaller than the nuclei area
    Can also use other objects instead of nuclei

    Args:
        df (DataFrame): The parent dataframe

    Returns:
        _type_: _description_
    """
    if child_obj == "Nuclei" and parent_obj == "Cell":
        not_empty_df = df[df["Metadata_EmptyImage_Cell"] == 0]
        normal_cells = not_empty_df[
            not_empty_df["Cell_Classify_Normal"] == 1
        ]  # one nucleus only
        size_filtered_cells = normal_cells[
            normal_cells["Cell_AreaShape_Area"]
            > normal_cells["Cell_Mean_Nuclei_AreaShape_Area"]
        ]  # cell area bigger than nuclear area

        # convenience column for area
        normal_cells["Nuclei_AreaShape_Area"] = normal_cells[
            "Cell_Mean_Nuclei_AreaShape_Area"
        ]
        final_df = size_filtered_cells.reset_index(drop=True)
    else:
        try:
            not_empty_df = df[df[f"Metadata_EmptyImage_{parent_obj}"] == 0]
            normal_cells = not_empty_df[
                not_empty_df[f"{parent_obj}_Children_{child_obj}_Count"] < 2
            ]
        except:
            not_empty_df_child = df[df[f"Metadata_EmptyImage_{child_obj}"] == 0]
            not_empty_df_parent = df[df[f"Metadata_EmptyImage_{parent_obj}"] == 0]
            if len(not_empty_df_child.rows()) == len(not_empty_df_parent.rows()):
                final_df = normal_cells.reset_index(drop=True)
                return final_df
    return final_df


def multinucleate_cells(df):
    multinuc_df = df[df["Cell_Classify_multinucleate"] == 1]
    return multinuc_df


def plate_df_setup_fromcsv(
    curr_plates,
    curr_plate_datafolders,
    parent_dir,
    csv_names=[
        "Cell.csv",
        "Nuclei.csv",
        "MergedMitoPerCell.csv",
        "MergedLysoPerCell.csv",
    ],
):
    """
    Combine the cellprofiler feature data from different plates into a single DataFrame
    Returns a DataFrame with the combined data
    """
    # Initialize a list to store the combined DataFrames
    plate_dfs = {}

    for i, plate in enumerate(curr_plates):
        # Construct the full path to the folder
        folder_path = os.path.join(parent_dir, plate)

        # Construct the full path to the metadata file and CSV file
        map_file = os.path.join(folder_path, "metadata/map.csv")
        csv_folder_path = os.path.join(folder_path, curr_plate_datafolders[i])

        # Make a list of the csv file paths for each compartment
        compartment_paths = []

        for file in csv_names:
            cp_file = os.path.join(csv_folder_path, file)
            if os.path.exists(cp_file) and file in csv_names:
                compartment_paths.append(cp_file)

        # Join the file dataframes
        if "Cell.csv" in compartment_paths[0]:
            pre_cell_df = pd.read_csv(compartment_paths[0])
        else:
            return FileNotFoundError("Cell.csv not found in the folder")

        for j, compartment in enumerate(compartment_paths):
            if j == 0 and "Cell.csv" in compartment:
                continue

            compartment_df = pd.read_csv(compartment)
            excluded_columns = ["ImageNumber", "ObjectNumber"]

            prefix = csv_names[j].replace(".csv", "") + "_"

            keys_df = compartment_df[excluded_columns]
            excluded_keys_df = compartment_df.drop(columns=excluded_columns)

            prefixed_compartment_df = excluded_keys_df.add_prefix(prefix)
            combined_prefixed_compartment_df = pd.concat(
                [keys_df, prefixed_compartment_df], axis=1
            )

            pre_cell_df = pre_cell_df.merge(
                combined_prefixed_compartment_df,
                on=["ImageNumber", "ObjectNumber"],
                how="left",
            )

        # Join the metadata with the data
        if os.path.exists(cp_file) and os.path.exists(map_file):
            # Read the metadata file and merge with dataframes (map.csv)
            platemap_df = pd.read_csv(map_file)
            cell_df = pre_cell_df.merge(
                platemap_df,
                on=[
                    "Metadata_Well",
                    "Metadata_WellRow",
                    "Metadata_WellColumn",
                    "Metadata_Field",
                ],
                how="left",
            )

            # Add a column to the cell_df to group passages and identify the plate replicate
            cell_df["Passage Group"] = cell_df["PassageNumber"].apply(passage_group)
            cell_df["Metadata_Plate"] = plate
            cell_df["Replicate_Number"] = i + 1
            # Append the merged DataFrame to the list
            plate_dfs[plate] = cell_df

    # Combine all the different replicate DataFrames into a single DataFrame
    combined_replicates_df = pd.concat(plate_dfs.values(), ignore_index=True)

    # Filter DataFrames to only include cells that were stained with LAMP1-488 and MitoRed
    combined_replicates_df_mitolyso = combined_replicates_df[
        combined_replicates_df["Staining"].str.startswith("LAMP1-488 + MitoRed")
    ]
    return combined_replicates_df_mitolyso


def calculate_median_object_features(
    parent_df, object_df, feature, parent_key, child_key="Cell_Number_Object_Number"
):
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


# group by parent key to find median
def add_median_object_features_to_parent(
    parent_df, object_df, object_name, child_key="Cell_Number_Object_Number"
):
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
        raise ValueError(
            f"Unknown object name: {object_name}. Expected 'Lysosomes', 'Mitochondria', or 'Nuclei'."
        )

    for feature in object_df.columns:
        print("checking feature:", feature)
        # Exclude if matches any in other_channels, unless it also matches feature_types
        if object_name not in feature or (any(ch in feature for ch in other_channels)):
            print("oops, skipping:", feature)
            continue  # Skip the parent key column or non-feature columns
        if any(ft in feature for ft in feature_types):
            print("FEATURE TYPE MATCH:", feature)
            modified_df = calculate_median_object_features(
                modified_df, object_df, feature, parent_key, child_key
            )

    return modified_df


def exclude_borders(
    df, min_x=0.0, min_y=0.0, max_x=2160.0, max_y=2160.0, prefix="Cell_"
):
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
        (df[f"{prefix}AreaShape_BoundingBoxMaximum_X"] < max_x)
        & (df[f"{prefix}AreaShape_BoundingBoxMaximum_Y"] < max_y)
        & (df[f"{prefix}AreaShape_BoundingBoxMinimum_X"] > min_x)
        & (df[f"{prefix}AreaShape_BoundingBoxMinimum_Y"] > min_y)
    ]
    return filtered_df


def well_namer(row, col):
    """
    Convert row and column numbers to a well name in the format A01, B02, etc.

    Args:
        row (int): The row number (1-8)
        col (int): The column number (1-12)

    Returns:
        str: Well name in the format A01, B02, etc.
    """
    well_name = str(chr(ord("@") + row)) + str(col).rjust(
        2, "0"
    )  # make the number have a left align, adding a zero
    return well_name


def add_well_metadata(image_df):
    """
    Add well metadata to the image DataFrame.

    Args:
        image_df (DataFrame): DataFrame containing image metadata

    Returns:
        DataFrame: Updated DataFrame with well metadata
    """
    image_df.columns = image_df.columns.str.replace(
        r"^Image_Metadata_", "Metadata_", regex=True
    )
    image_df[["Metadata_WellRow", "Metadata_WellColumn", "Metadata_Field"]] = image_df[
        "Image_URL_DAPI"
    ].str.extract(r"r(\d{2})c(\d{2})f(\d{2}).tif")
    # Convert extracted columns to int
    image_df["Metadata_WellRow"] = image_df["Metadata_WellRow"].astype(int)
    image_df["Metadata_WellColumn"] = image_df["Metadata_WellColumn"].astype(int)
    image_df["Metadata_Field"] = image_df["Metadata_Field"].astype(int)
    # apply well namer function
    image_df["Metadata_Well"] = image_df.apply(
        lambda x: well_namer(x["Metadata_WellRow"], x["Metadata_WellColumn"]), axis=1
    )

    return image_df


def update_database_with_well_metadata(db_path):
    """
    Update the database with well metadata.

    Args:
        db_path (str): Path to the database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Read Per_Image table
    image_df = pd.read_sql_query("SELECT * FROM Per_Image", conn)

    # Add well metadata
    updated_image_df = add_well_metadata(image_df)

    # Write updated DataFrame back to the database
    try:
        updated_image_df.to_sql("Per_Image", conn, if_exists="replace", index=False)
        print("Database updated successfully with well metadata.")
    except Exception as e:
        print(f"Error updating database: {e}")
    # cursor.execute("SELECT Metadata_Well FROM Per_Image LIMIT 5;")
    # cursor.fetchall()
    conn.close()


def standardize_group(df, columns):
    """_summary_

    Args:
        df (_type_): _description_
        columns (_type_): _description_

    Returns:
        _type_: _description_
    """
    from sklearn.preprocessing import StandardScaler

    # Import the scaler and transform all time values to that of a standard distribution - only use for ML, not very desceiptive
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df[columns])
    return scaled_df


def group_by_condition(df, feature_list, groupby_column="AgeGroup"):
    """Group by a condition

    Args:
        df (_type_): _description_
        feature_list (_type_): _description_
        groupby_column (str, optional): _description_. Defaults to "AgeGroup".

    Returns:
        _type_: _description_
    """
    # Group columns by age group and apply groupby function to the DF
    df_groupby = df.groupby(groupby_column).apply(
        lambda x: standardize_group(x, feature_list)
    )
    return df_groupby
