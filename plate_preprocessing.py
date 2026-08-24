"""
Helper functions for plate preprocessing and data analysis / visualization
Allie Spangaro, Toronto Metropolitan University
"""

import operator
import os
import re
import sqlite3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import matplotlib.pyplot as plt

####
# Functions for grouping passages and creating/remnaming columns
###


def well_namer(row, col, pad_zeros=False):
    """
    Convert row and column numbers to a well name in the format A01, B02, etc.

    Args:
        row (int): The row number (1-8)
        col (int): The column number (1-12)

    Returns:
        str: Well name in the format A01, B02, etc.
    """
    if pad_zeros:  # make the number have a left align, adding a zero
        well_name = str(chr(ord("@") + row)) + str(col).rjust(2, "0")
    else:
        well_name = str(chr(ord("@") + row)) + str(col)

    return well_name


def get_all_group_order():
    """
    Get the order of the passage groups for plotting
    Returns a list of the passage groups in order
    Returns:
        list: A list of strings representing the group order
    """
    order = [
        "P6-12",
        "P13-15",
        "P16-18",
        "P19-21",
        "P22-24",
        "P25-27",
        "P28-31",
        "P32-35",
        "Doxo",
    ]
    return order


def passage_group(passage_num):
    """
    Group passages into bins for plotting (based on new groupings after quality control)
    returns string of the group that the passage number belongs to
    """
    # use this function to group passages into groups for plotting
    passage = int(passage_num)
    if 6 <= passage <= 12:
        return "P6-12"
    elif 13 <= passage <= 15:
        return "P13-15"
    elif 16 <= passage <= 18:
        return "P16-18"
    elif 19 <= passage <= 21:
        return "P19-21"
    elif 22 <= passage <= 24:
        return "P22-24"
    elif 25 <= passage <= 27:
        return "P25-27"
    elif 28 <= passage <= 31:
        return "P28-31"
    elif passage >= 32:
        return "P32-35"
    else:
        return "Unknown"


def normalize_well_string(s):
    # A01, A1, a-01, " A 01 " -> A01
    s = s.astype(str).str.upper().str.strip()
    s = s.str.replace(r"[^A-Z0-9]", "", regex=True)  # remove separators/spaces
    s = s.str.replace(r"^([A-H])([0-9])$", r"\10\2", regex=True)  # A1 -> A01
    return s


def rename_mismatched_well_names(
    df, well_col="Metadata_Well", suffix_l="_x", suffix_r="_y"
):
    """
    Rename mismatched well names in a dataframe to match the expected format (A01, B02, etc.)

    Args:
        df (DataFrame): The dataframe containing the well names.
        well_col (str): The name of the column containing the well names.

    Returns:
        DataFrame: The dataframe with renamed well names.
    """
    # df[well_col] = normalize_well_string(df[f"{well_col}{suffix_l}"])
    df[well_col] = df[f"{well_col}{suffix_l}"].combine_first(
        df[f"{well_col}{suffix_r}"]
    )
    df.drop(columns=[f"{well_col}{suffix_l}", f"{well_col}{suffix_r}"], inplace=True)
    return df


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


def take_drug_from_condition(init_df, group_variable_col, drug_metadata_col, drug_name):
    """Grab a string "drug name" from a column with the drug metadata and add it to the grouping variable column

    Args:
        init_df (DataFrame): The dataframe to add to
        group_variable_col (str): _description_
        drug_metadata_col (str): The column to take from
        drug_name (str): The drug used

    Returns:
        Series: The modified grouping variable column with the added drug
    """
    if drug_metadata_col is not None:
        # Replace values in 'col1' with values from 'col2' only if 'col2' is not None or NaN
        df = init_df.copy()

        df[group_variable_col] = np.where(
            df[drug_metadata_col].str.contains(drug_name, na=False),
            df[drug_metadata_col],
            df[group_variable_col],
        )
        newcol = df[group_variable_col]

    return newcol


####
# Column search functions
####
def find_plate(path):
    """
    Find the plate number from a given path string.

    Args:
        path (str): The path string to search for the plate number.

    Returns:
        int: The plate number, or None if not found.
    """
    plate_pattern = (
        r"R(\d{1})"  # Matches "RX" where X is the plate number (placeholder for now)
    )
    match = re.search(plate_pattern, path)
    if match:
        plate = int(match.group(1))
    else:
        plate = None
    return plate


def search_column_name(
    df, query="", case_sensitive=False, inclusive_or=True, verbose=True
):
    """_summary_

    Args:
        df (DataFrame): _description_
        query (str or list, optional): your search query. Defaults to "".
        verbose (bool, optional): whether to print the results. Defaults to True.
    Return:

    """
    if inclusive_or:
        if isinstance(query, list):
            query_cols = []
            for q in query:
                if case_sensitive:
                    query_cols.extend([col for col in df.columns if q in col])
                else:
                    query_cols.extend(
                        [col for col in df.columns if q.lower() in col.lower()]
                    )
        else:
            if case_sensitive:
                query_cols = [col for col in df.columns if query in col]
            else:
                query_cols = [col for col in df.columns if query.lower() in col.lower()]
    else:
        if isinstance(query, list):
            query_cols = []
            for col in df.columns:
                if all(
                    q in col if case_sensitive else q.lower() in col.lower()
                    for q in query
                ):
                    query_cols.append(col)
                else:
                    continue
        else:
            if case_sensitive:
                query_cols = [col for col in df.columns if query not in col]
            else:
                query_cols = [
                    col for col in df.columns if query.lower() not in col.lower()
                ]
    if verbose:
        print(f"Query: {query}")
        for col in query_cols:
            print(f"    {col}")
    return query_cols


def find_row_col(well_code):
    """Find the row and column of a well based on its code.

    Args:
        well_code (str): The well code (e.g., "A01", "B02", etc.)

    Returns:
        tuple: A tuple containing the row and column of the well.
    """
    import re

    rowcol_pattern = r"r(\d{1,2})c(\d{1,2})"  # Matches "RX" where X is the plate number (placeholder for now)
    match = re.search(rowcol_pattern, well_code)
    if match:
        row_metadata = int(match.group(1))
        col_metadata = int(match.group(2))
    else:
        row_metadata = None
        col_metadata = None
    return row_metadata, col_metadata


def find_plate_cp_output_folder(path):
    import re

    plate_pattern = r"_rep0(\d{1})_"  # Matches "RX" where X is the plate number (placeholder for now)
    match = re.search(plate_pattern, path)
    if match:
        plate = int(match.group(1))
    else:
        plate = None
    return plate


def query_group_plate_condition(
    df, group, plate_number=0, condition_col="", op=operator.eq, value=None
):
    """
    Filter df by group, plate_number, and a condition using a passed operator.
    Example: op=operator.lt for '<', op=operator.gt for '>', op=operator.eq for '=='
    """
    mask = (
        (df["AllGroups"] == group)
        & (df["PlateNumber"] == plate_number)
        & (op(df[condition_col], value))
    )
    return df[mask]


def get_unique_cols_to_use(df):
    base_cols = [
        "Metadata_PlateNumber",
        "Metadata_RowColFieldCode",
        "AllGroups",
        "AreaShape_Area",
    ]
    areashape_features = [
        "AreaShape_Area",
        "AreaShape_Perimeter",
        "AreaShape_EquivalentDiameter",
        "AreaShape_Eccentricity",
        "AreaShape_FormFactor",
        "AreaShape_Solidity",
        "AreaShape_Extent",
        "AreaShape_MaxFeretDiameter",
        "AreaShape_MinFeretDiameter",
        "AreaShape_MeanRadius",
    ]

    colnames_mitoskel_nuc = search_column_name(df, "Nuclei_ObjectSkeleton")
    colnames_mitocount = search_column_name(
        df, ["Children_Mitochondria", "Count"], inclusive_or=False
    )
    colnames_mitoarea = search_column_name(
        df, ["Mito", "AreaShape_Area"], inclusive_or=False
    )

    colnames_lysocount = search_column_name(
        df, ["Children_Lysosomes", "Count"], inclusive_or=False
    )
    colnames_lysoarea = search_column_name(df, areashape_features, inclusive_or=True)
    for col in colnames_lysoarea.copy():
        if "Lyso" in col and "AreaShape" in col:
            continue
        else:
            colnames_lysoarea.remove(col)

    colnames_intesnity_distribution = search_column_name(
        df, ["Radial", "MitoTracker"], inclusive_or=False
    )
    colnames_intesnity_distribution += search_column_name(
        df, ["Radial", "LAMP1"], inclusive_or=False
    )
    for col in colnames_intesnity_distribution.copy():
        if "Nuclei" in col or "Closing" in col:
            colnames_intesnity_distribution.remove(col)
        else:
            continue

    colnames_overlap = search_column_name(
        df, ["Overlap", "Correlation"], inclusive_or=True
    )
    for col in colnames_overlap.copy():
        if ("Mito" not in col and "Lyso" not in col and "LAMP1" not in col) or (
            "Texture" in col or "DAPI" in col
        ):
            colnames_overlap.remove(col)
        else:
            continue
    print(colnames_overlap)

    colnames_ij = search_column_name(df, "IJ_Mitochondria")

    use_cols = (
        base_cols
        + colnames_mitoskel_nuc
        + colnames_ij
        + colnames_mitocount
        + colnames_mitoarea
        + colnames_lysocount
        + colnames_lysoarea
        + colnames_overlap
        + colnames_intesnity_distribution
    )
    use_cols_unique = list(dict.fromkeys(use_cols))
    use_cols_unique_copy = use_cols_unique.copy()
    for col in use_cols_unique_copy:
        if col in base_cols:
            continue
        elif "Metadata" in col or "Title" in col or "FileName" in col:
            use_cols_unique.remove(col)
    return use_cols_unique


def get_object_skeleton_length_cols(df):
    colnames_object_skeleton_length = search_column_name(df, "SkeletonLength")
    for col in colnames_object_skeleton_length.copy():
        if (
            "Mean" in col
            or "Median" in col
            or "Threshold" in col
            or "Stdev" in col
            or "FromBranches" in col
        ):
            colnames_object_skeleton_length.remove(col)

    print(colnames_object_skeleton_length)
    return colnames_object_skeleton_length


####
# Sort functions
####


def passage_groups_sort_key(group_name):
    """
    Key function for natural sorting of strings containing numbers.
    Extract numeric parts and convert to int .
    """
    digit_pattern = (
        r"([0-9]+)"  # Matches "RX" where X is the plate number (placeholder for now)
    )
    match = re.search(digit_pattern, group_name)
    if match:
        first_digit = int(match.group(1))
        return first_digit
    else:
        text = group_name.lower()
        if text == "doxo":
            return 999
        else:
            return ValueError


def allgroups_sort_key(value):
    """Custom sort key function for 'AllGroups' column."""
    import re

    match = re.match(r"P(\d+)", value)
    if match:
        first_number = match.group(1)
        if first_number.isdigit():
            return int(first_number)
        else:
            return 99999
    return 99999


print(allgroups_sort_key("P6-10"))


def sort_df_by_plate_number(df, x_value):
    """Sort a pandas dataframe of experimental data that has been grouped by an x_value by the integer representation of the plate number using a sort key
    Ideally used before plotting so that your plots have the same pallete and are easily comparable
    Args:
        df (DataFrame): your df to be sorted
        x_value (str): the column containing your grouping variable to be sorted by plate number

    Returns:
        DataFrame: The dataframe sorted by plate number
    """
    df_sorted = df.sort_values(
        by=[x_value], key=lambda x: x.map(passage_groups_sort_key)
    ).reset_index(drop=True)
    return df_sorted


####
# Pairing functions
####
def getpairs(df, group, order=None):
    from itertools import combinations

    """
    Get pairs of unique values from a specified column in the DataFrame.
    Args:
        df (DataFrame): The DataFrame containing the data.
        group (str): The name of the column to get unique values from.
        order (list): A list of values to order the unique values by. If empty, uses the unique values as is.
    Returns:
        list: A list of tuples containing pairs of unique values from the specified column."""
    # Get the unique values of the categorical column, Order the unique values according to the specified order
    if order is None:
        order = df[group].dropna().unique().tolist()
    unique_values = df[group].dropna().unique()

    ordered_values = [value for value in order if value in unique_values]

    pairs = list(combinations(ordered_values, 2))
    return pairs


####
# Feature selection functions
####


def define_cell_features(df):
    # Get the columns of the dataframe
    columns_list = df.columns.tolist()
    columns_list = [
        col
        for col in columns_list
        if "Metadata" not in col
        and "FileName" not in col
        and "PathName" not in col
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    old_columns_list = columns_list = [
        col
        for col in columns_list
        if "Metadata" not in col and "FileName" not in col and "PathName" not in col
    ]
    print(
        "Original columns:",
        len(old_columns_list),
        "Filtered columns:",
        len(columns_list),
    )
    return columns_list


def make_feature_dict(columns_list):
    """
    Create a dictionary of features from the columns list.
    Args:
        columns_list (list): List of column names from the DataFrame.
    Returns:
        dict: A dictionary with keys as feature types and values as lists of corresponding column names."""
    # Add the different types of features to a dictionary
    feature_dict = {
        "intensity": [],
        "texture": [],
        "areashape": [],
        "granularity": [],
        "radialdistribution": [],
        "totals": [],
        "count": [],
        "distance": [],
        "per_cell_area": [],
        "coloc": [],
        "other": [],
    }
    for col in columns_list:
        if "Texture" in col:
            feature_dict["texture"].append(col)
        elif "Intensity" in col:
            feature_dict["intensity"].append(col)
        elif "Math_" in col or "Corr_" in col:
            feature_dict["totals"].append(col)
        elif "Count" in col:
            feature_dict["count"].append(col)
        elif "AreaShape" in col:
            feature_dict["areashape"].append(col)
        elif "Distance" in col:
            feature_dict["distance"].append(col)
        elif "PerCell" in col:
            feature_dict["per_cell_area"].append(col)
        elif "Granularity" in col:
            feature_dict["granularity"].append(col)
        elif "RadialDistribution" in col:
            feature_dict["radialdistribution"].append(col)
        elif "Lysosomes_Mitochondria_Ratio" in col:
            feature_dict["coloc"].append(col)
        else:
            feature_dict["other"].append(col)

    return feature_dict


def get_feature_dicts(df):
    columns_list = define_cell_features(df)
    mito_features = make_feature_dict(
        [
            col
            for col in columns_list
            if ("Mito" in col or "Mitochondria" in col)
            and ("DAPI" not in col and "LAMP1" not in col and "Frame" not in col)
            and not (col.startswith("Nuclei_"))
        ]
    )
    lyso_features = make_feature_dict(
        [
            col
            for col in columns_list
            if ("Lysosome" in col or "LAMP1" in col or "Lyso" in col)
            and ("DAPI" not in col and "Mito" not in col and "Frame" not in col)
            and not (col.startswith("Nuclei_"))
        ]
    )
    nuc_features = make_feature_dict(
        [
            col
            for col in columns_list
            if ("Nuc" in col or "DAPI" in col)
            and ("MitoTracker" not in col and "LAMP1" not in col and "Frame" not in col)
        ]
    )
    cell_features = make_feature_dict(
        [
            col
            for col in columns_list
            if "AreaShape" in col
            and "Mito" not in col
            and "Lyso" not in col
            and "LAMP1" not in col
            and "Nuc" not in col
            and "DAPI" not in col
            and "Metadata" not in col
            and "FileName" not in col
            and "PathName" not in col
        ]
    )
    return [mito_features, lyso_features, nuc_features, cell_features]


####
# Feature derivation functions
####
def mean_intensity_per_compartment_per_cell(df, compartment, name, tag, math=None):
    colname = f"Mean_Intensity_Per_{compartment} Per_Cell"
    integrated = "Intensity_IntegratedIntensity_" + tag + "_MAX"
    total_organelle_area = name + "_AreaShape_Area"
    total_organelle_area = math if math is not None else total_organelle_area

    df[colname] = df.apply(lambda x: x[integrated] / x[total_organelle_area], axis=1)

    return df[colname]


def cell_nuc_area_ratio(
    df,
    cell_area_col="Cell_AreaShape_Area",
    nuc_area_col="Nuclei_AreaShape_Area",
    ratio_col_name="Cell_Nuclei_Area_Ratio",
):
    """
    Calculate the ratio of cell area to nuclear area.

    Args:
        df (Series): A DataFrame containing 'Cell_AreaShape_Area' and 'Nuclei_AreaShape_Area' cols.

    Returns:
        Series: The column with the cell/nuc area ratio column to be added
    """
    df_overzero = df[df[nuc_area_col] > 0]
    final_df = df_overzero.dropna(subset=[nuc_area_col]).reset_index(drop=True)
    final_df[ratio_col_name] = final_df[cell_area_col] / final_df[nuc_area_col]
    return final_df[ratio_col_name]


def multinucleate_cells(df):
    multinuc_df = df[df["Cell_Classify_multinucleate"] == 1]
    return multinuc_df


def make_per_cell_area_column_names(
    df,
    use_cols,
    area_col="AreaShape_Area",
    colnames_mitoskel_seeds=None,
    number_of_seeds_col="Children_MitoSkel_Seeds_Count",
    areashape_cols=None,
    calculate_totals=False,
    base_cols=None,
    exclude_per_area=None,
):
    if base_cols is None:
        base_cols = [
            "Metadata_PlateNumber",
            "Metadata_RowColField",
            "AllGroups",
            "AreaShape_Area",
        ]
    if areashape_cols is None:
        areashape_cols = [
            "AreaShape_Area",
            "AreaShape_Perimeter",
            "AreaShape_EquivalentDiameter",
        ]
    df = df.copy()
    new_use_cols = use_cols.copy()
    for col in base_cols:
        if col in new_use_cols:
            new_use_cols.remove(col)

    new_columns = {}

    count_flag = 0
    for col in areashape_cols:
        if col in new_use_cols:
            new_use_cols.remove(col)

        df[col] = pd.to_numeric(df[col], errors="coerce")

        if calculate_totals and "Mean" in col:
            col_without_mean = col.replace("Mean_", "")
            new_columns["Math_Total_" + col_without_mean] = (
                df[col] * df[count_cols[count_flag]]
            )
            new_columns["Per_Area_AreaOccupied_" + col_without_mean] = (
                df["Math_Total_" + col_without_mean] / df[area_col]
            )

        elif "Mean" in col:
            continue
        elif "Total" in col:
            col_without_total = col.replace("Total_", "")
            new_columns["Per_Area_AreaOccupied_" + col_without_total] = (
                df[col] / df[area_col]
            )
        elif "RelabeledMito" in col:
            new_columns["Per_Area_AreaOccupied_" + col] = df[col] / df[area_col]
        else:
            new_columns["Per_Area_" + col] = df[col] / df[area_col]

    if colnames_mitoskel_seeds:
        for col in colnames_mitoskel_seeds:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            col_without_mean = col.replace("Mean_", "")
            new_columns["Math_Total_" + col_without_mean] = (
                df[col] * df[number_of_seeds_col]
            )
            new_columns["Per_Area_" + col_without_mean] = (
                new_columns["Math_Total_" + col_without_mean] / df[area_col]
            )
            if col in new_use_cols:
                new_use_cols.remove(col)

    if exclude_per_area is None:
        exclude_per_area = [
            "Mean",
            "Median",
            "Std",
            "Distribution",
            "Metadata",
            "Title",
            "Per_Area",
            "Location",
            "Correlation",
            "Overlap",
            "Eccentricity",
            "FormFactor",
            "Solidity",
            "Extent",
            "BranchLength",
        ]
    for col in new_use_cols:
        if any(exclude in col for exclude in exclude_per_area):
            continue
        else:
            print("making per area column for:", col)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            new_colname = "Per_Area_" + col
            if "Count" in col:
                new_colname = new_colname.replace("Children_", "Number_")

            new_columns[new_colname] = df[col] / df[area_col]

    new_columns_df = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, new_columns_df], axis=1)
    return df


def make_per_skeleton_length_column_names(
    df,
    use_cols,
    skeleton_length_cols,
    base_cols=None,
    colnames_mitoskel_seeds=None,
    sets=None,
    feature_types=None,
):
    if sets is None:
        sets = []
    if feature_types is None:
        feature_types = [
            "Nuclei_ObjectSkeleton",
            "MitoSkel_Seeds_ObjectSkeleton",
            "IJ_Mitochondria",
        ]
    df = df.copy()
    new_use_cols = use_cols.copy()
    if base_cols is None:
        base_cols = [
            "Metadata_PlateNumber",
            "Metadata_RowColField",
            "AllGroups",
            "AreaShape_Area",
        ]
    for col in use_cols:
        if (
            col in base_cols
            or col in skeleton_length_cols
            or "Metadata" in col
            or "Title" in col
            or "Mean" in col
            or "Median" in col
            or "Stdev" in col
            or "Footprint" in col
            or "Per_Area" in col
            or "Correlation" in col
            or "Overlap" in col
            or "Distribution" in col
            or "AreaShape" in col
            or "Children" in col
        ):
            new_use_cols.remove(col)

    new_columns = {}

    for col in new_use_cols:
        this_feature_type = None
        for feature_type in feature_types:
            if feature_type in col:
                this_feature_type = feature_type
                break

        df[col] = pd.to_numeric(df[col], errors="coerce")

        for length_col in skeleton_length_cols:
            if this_feature_type and this_feature_type not in length_col:
                continue
            elif this_feature_type == "IJ_Mitochondria":
                subtypes = ["_LargestStructure", "_TotalAcrossAllStructures"]
                if any(subtype in col for subtype in subtypes) and any(
                    (subtype in col and subtype not in length_col)
                    or (subtype not in col and subtype in length_col)
                    for subtype in subtypes
                ):
                    continue
            print(f"Calculating Per_SkeletonLength for {col} using {length_col}")
            if sets and any(set_name in length_col for set_name in sets):
                for set_name in sets:
                    if set_name in length_col and set_name in col:
                        print(f"Using set {set_name} for {col} and {length_col}")
                        new_columns["Per_SkeletonLength_" + col] = (
                            df[col] / df[length_col]
                        )
                        break
            else:
                new_columns["Per_SkeletonLength_" + col] = df[col] / df[length_col]

    new_columns_df = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, new_columns_df], axis=1)
    return df


def calculate_extra_features(full_df, organelles=["Mitochondria", "Lysosomes"]):
    df = full_df.copy()
    for organelle in organelles:
        if organelle == "Mitochondria":
            tag = "MitoTracker"
            name = "TotalMitochondria"
            math = None
        elif organelle == "Lysosomes":
            tag = "LAMP1"
            name = "TotalLysosomes"
            math = None
        else:
            continue
        df[f"Mean_Intensity_Per_{organelle}_PerCell_Area"] = (
            mean_intensity_per_compartment_per_cell(df, organelle, name, tag, math=math)
        )
        df[f"Ratio_Mean_{organelle}_MaxMinFeret_DiameterRatio"] = (
            df[f"Mean_{organelle}_AreaShape_MaxFeretDiameter"]
            / df[f"Mean_{organelle}_AreaShape_MinFeretDiameter"]
        )
        df[f"Ratio_Median_{organelle}_DiameterRatio_PerCell"] = (
            df[f"{organelle}_Median_{organelle}_AreaShape_MaxFeretDiameter"]
            / df[f"{organelle}_Median_{organelle}_AreaShape_MinFeretDiameter"]
        )

        df[f"Ratio_Mean_{organelle}_Distance_Centroid_Cell_Minimum_Cell_QuinRatio"] = (
            df[f"Mean_{organelle}_Distance_Centroid_Cell"]
            / df[f"Mean_{organelle}_Distance_Minimum_Cell"]
        )

        df[f"Per_Area_Mean_{organelle}_Distance_Centroid_Cell"] = (
            df[f"Mean_{organelle}_Distance_Centroid_Cell"] / df["AreaShape_Area"]
        )
        df[f"Per_Area_Mean_{organelle}_Distance_Minimum_Cell"] = (
            df[f"Mean_{organelle}_Distance_Minimum_Cell"] / df["AreaShape_Area"]
        )

    df["Ratio_Number_Lysosomes_To_Mitochondria"] = (
        df["Children_Lysosomes_Count"] / df["Children_Mitochondria_Count"]
    )
    df["Ratio_Area_Lysosomes_To_Mitochondria"] = (
        df["TotalLysosomes_AreaShape_Area"] / df["TotalMitochondria_AreaShape_Area"]
    )

    return df


def calculate_aggregated_object_features(
    parent_df,
    object_df,
    feature,
    parent_key,
    object_name="Object",
    child_key="Cell_Number_Object_Number",
    aggregation="Median",
    prefix="Cell",
):
    """
    Calculate the aggregated function (typically median) of specified features grouped by a parent key.

    Parameters:
    df (DataFrame): The DataFrame containing the features.
    object_df (DataFrame): The DataFrame containing the object features.
    feature (str): The feature column to calculate the median for.
    child_key (str): The column name in the PARENT table that identifies the child obj.
    parent_key (str): The column name in the CHILD table that identifies the parent key.
    aggregation (str): the type of aggregation, can be "Mean","Median","Mode", or "Std"
    child_name (str): The name of the child object type.

    Returns:
    modified_df (DataFrame): The DataFrame with the new median feature column added.

    """
    agg_title = aggregation.title()  # make it title case for the column syntax
    if agg_title == "Median":
        agg_values = object_df.groupby([parent_key, "ImageNumber"])[feature].median()
    elif agg_title == "Mean" or agg_title == "Avg" or agg_title == "Average":
        agg_title = "Averaged"
        agg_values = object_df.groupby([parent_key, "ImageNumber"])[feature].mean()
    elif agg_title == "Sum":
        agg_title = "Total"
        agg_values = object_df.groupby([parent_key, "ImageNumber"])[feature].sum()
    elif agg_title == "Max":
        agg_values = object_df.groupby([parent_key, "ImageNumber"])[feature].max()
    elif agg_title == "Min":
        agg_values = object_df.groupby([parent_key, "ImageNumber"])[feature].min()
    elif agg_title == "Std":
        agg_values = object_df.groupby([parent_key, "ImageNumber"])[feature].std()
    else:
        raise ValueError('aggregation (str) not in "Mean","Median","Mode", or "Std"')
        return pd.DataFrame()

    # create the new column name for the aggregated feature
    col_name = f"{prefix}_{object_name}_{agg_title}_{feature}"

    # reformat the agg_values dataframe for merging
    agg_values = agg_values.reset_index()
    agg_values.columns = [child_key, "ImageNumber", col_name]
    agg_values["CellNumber_ImageNumber_Index"] = (
        agg_values[child_key].astype(str) + "_" + agg_values["ImageNumber"].astype(str)
    )

    # Merge the aggregated values back into the parent dataframe
    modified_df = parent_df.copy()
    # add a column to parent_df to merge on
    modified_df["CellNumber_ImageNumber_Index"] = (
        modified_df[child_key].astype(str)
        + "_"
        + modified_df["ImageNumber"].astype(str)
    )

    modified_merged_df = modified_df.merge(
        agg_values[["CellNumber_ImageNumber_Index", col_name]],
        how="left",
        left_on="CellNumber_ImageNumber_Index",
        right_on="CellNumber_ImageNumber_Index",
        suffixes=("", ""),
    )

    return modified_merged_df


# group by parent key to find median


def add_median_object_features_to_parent(
    parent_df,
    object_df,
    object_name,
    child_key="Cell_Number_Object_Number",
    prefix="Cell",
):
    """
    Add median features from object_df to parent_df based on the specified feature and keys.

    Parameters:
    parent_df (DataFrame): The DataFrame containing the parent features (e.g cell).
    object_df (DataFrame): The DataFrame containing the object features (e.g. mitochondria).
    object_name (str): The object of interest to add to the parent table.
    child_key (str): The column name in the PARENT table that identifies the child object.
    prefix (str): The prefix for the new feature column names.

    Returns:
    modified_df (DataFrame): The DataFrame with the new median feature column added.

    """
    # print(parent_df)
    modified_df = parent_df.copy()
    feature_types = ["AreaShape", "Distance", "Intensity", "Location"]
    features_to_exclude = [
        "Zernike",
        "Maximum_X",
        "Maximum_Y",
        "Minimum_X",
        "Mimimum_Y",
        "Centroid_X",
        "Centroid_Y",
    ]
    other_channels = ["DAPI", "LAMP1", "MitoTracker", "Phalloidin"]
    # use the other channels list as a check to discard irrelavent features

    if "Lysosomes" in object_name:
        parent_key = "Lysosomes_Parent_Cell"
        other_channels.remove("LAMP1")
    elif "Mitochondria" in object_name:
        parent_key = "Mitochondria_Parent_Cell"
        other_channels.remove("MitoTracker")
    elif "Nuclei" in object_name:
        parent_key = "Nuclei_Parent_Cell"
        other_channels.remove("DAPI")
    elif "MitoEnds" in object_name:
        parent_key = "MitoEnds_Parent_Cell"
        other_channels.remove("MitoTracker")
    else:
        raise ValueError(
            f"Unknown object name: {object_name}. Expected 'Lysosomes', 'Mitochondria', 'MitoEnds', or 'Nuclei'."
        )

    for feature in object_df.columns:
        # print("checking feature:", feature)
        # Exclude if matches any in other_channels, unless it also matches feature_types
        if object_name not in feature or (
            any(ch in feature for ch in other_channels)
            or any(exclusion in feature for exclusion in features_to_exclude)
        ):
            # print("oops, skipping:", feature)
            continue  # Skip the parent key column or non-feature columns
        if any(ft in feature for ft in feature_types):
            modified_df = calculate_aggregated_object_features(
                modified_df,
                object_df,
                feature,
                parent_key,
                object_name,
                child_key,
                aggregation="Median",
                prefix=prefix,
            )

    return modified_df


def load_organelle_stats(db_path, df, organelles=None, stats_list=None):
    """
    Load organelle median features from the database.

    Parameters:
    db_path (str): Path to the database file containing the extra features
    df (DataFrame): the Pandas dataframe to add the medians to.
    organelle (str): Name of the organelle (e.g., 'Lysosomes', 'Mitochondria', 'Nuclei').

    Returns:
    DataFrame: DataFrame containing the median features for the specified organelle.
    """
    import gc
    import sqlite3

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if organelles is None:
        organelles = ["Lysosomes", "Mitochondria"]
    if stats_list is None:
        stats_list = ["Median", "Std"]
    cell_df = df.copy()

    conn = sqlite3.connect(db_path)
    # open each one individually to not explode the ram
    for organelle in organelles:
        query = f"SELECT * FROM Per_{organelle}"
        org_df = pd.read_sql_query(query, conn)
        if "Median" in stats_list:
            cell_df = add_median_object_features_to_parent(
                parent_df=cell_df, object_df=org_df, object_name=organelle
            )
        if "Std" in stats_list:
            cell_df = add_standard_deviation_features_to_parent(
                parent_df=cell_df, object_df=org_df, object_name=organelle
            )
        del org_df  # Free memory
        gc.collect()

    # conn.close()
    return cell_df


def add_standard_deviation_features_to_parent(
    parent_df,
    object_df,
    object_name,
    child_key="Cell_Number_Object_Number",
    prefix="Cell",
    feature_types=None,
):
    """
    Calculate the standard deviation of specified features in a dataframe, grouped by certain metadata columns.

    Parameters:
    parent_df (pd.DataFrame): The input dataframe containing the data.
    object_df (pd.DataFrame): A dataframe containing object-level data.
    features (list): A list of feature column names for which to calculate standard deviations.
    parent_key (str): The name of the parent key column in the dataframe.
    child_key (str): The name of the child key column in the dataframe.

    Returns:
    pd.DataFrame: A dataframe containing the standard deviations of the specified features, grouped by metadata.
    """
    # open the csv file with extra info
    modified_df = parent_df.copy()
    other_channels = ["DAPI", "LAMP1", "MitoTracker", "Phalloidin"]
    # use the other channels list as a check to discard irrelavent features
    if "Lysosomes" in object_name:
        parent_key = "Lysosomes_Parent_Cell"
        other_channels.remove("LAMP1")
    elif "Mitochondria" in object_name:
        parent_key = "Mitochondria_Parent_Cell"
        other_channels.remove("MitoTracker")
    elif "Nuclei" in object_name:
        parent_key = "Nuclei_Parent_Cell"
        other_channels.remove("DAPI")
    elif "MitoEnds" in object_name:
        parent_key = "MitoEnds_Parent_Cell"
        other_channels.remove("MitoTracker")
    else:
        raise ValueError(
            f"Unknown object name: {object_name}. Expected 'Lysosomes', 'Mitochondria', 'MitoEnds', or 'Nuclei'."
        )

    if feature_types is None:
        feature_types = [
            "AreaShape_Area",
            "AreaShape_Perimeter",
            "AreaShape_EquivalentDiameter",
            "AreaShape_Extent",
            "AreaShape_Solidity",
            "AreaShape_Compactness",
            "AreaShape_Eccentricity",
        ]
    features_to_exclude = [
        "Zernike",
        "Maximum_X",
        "Maximum_Y",
        "Minimum_X",
        "Mimimum_Y",
        "Centroid_X",
        "Centroid_Y",
    ]
    modified_df = parent_df.copy()
    for feature in object_df.columns:
        # print("checking feature:", feature)
        # Exclude if matches any in other_channels, unless it also matches feature_types
        if object_name not in feature or (
            any(ch in feature for ch in other_channels)
            or any(exclusion in feature for exclusion in features_to_exclude)
        ):
            continue  # Skip the parent key column or non-feature columns
        if any(ft in feature for ft in feature_types):
            modified_df = calculate_aggregated_object_features(
                modified_df,
                object_df,
                feature,
                parent_key=parent_key,
                object_name=object_name,
                child_key=child_key,
                aggregation="Std",
                prefix=prefix,
            )

    # modified_df = modified_df.rename(columns=lambda x: re.sub(r"Std", f"Std_{object_df_name_noext}", x))
    return modified_df.copy()


####
# Plate setup functions functions
####
def plate_df_setup_fromcsv(
    curr_plates,
    curr_plate_datafolders,
    parent_dir,
    csv_names=None,
):
    """
    Combine the cellprofiler feature data from different plates into a single DataFrame
    Returns a DataFrame with the combined data
    """
    if csv_names is None:
        csv_names = [
            "Cell.csv",
            "Nuclei.csv",
            "MergedMitoPerCell.csv",
            "MergedLysoPerCell.csv",
        ]
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

            # Add a column to the cell_df to group passages and identify the plate plate
            cell_df["Passage Group"] = cell_df["PassageNumber"].apply(passage_group)
            cell_df["Metadata_Plate"] = plate
            cell_df["PlateNumber"] = i + 1
            # Append the merged DataFrame to the list
            plate_dfs[plate] = cell_df

    # Combine all the different plate DataFrames into a single DataFrame
    combined_plates_df = pd.concat(plate_dfs.values(), ignore_index=True)

    # Filter DataFrames to only include cells that were stained with LAMP1-488 and MitoRed
    combined_plates_df_mitolyso = combined_plates_df[
        combined_plates_df["Staining"].str.startswith("LAMP1-488 + MitoRed")
    ]
    return combined_plates_df_mitolyso


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

    # Read Per_Image table
    image_df = pd.read_sql_query("SELECT * FROM Per_Image", conn)

    # Add well metadata
    updated_image_df = add_well_metadata(image_df)

    # Write updated DataFrame back to the database
    try:
        updated_image_df.to_sql("Per_Image", conn, if_exists="replace", index=False)
        print("Database updated successfully with well metadata.")
    except sqlite3.DatabaseError as e:
        print(f"Error updating database: {e}")
    conn.close()


def make_single_feature_df(data, group, feature, plates):
    """Make a dataframe for a single feature from a larger dataframe in "tidy" format

    Args:
        data (DataFrame): your dataframe
        group (string): the grouping variable (x value)
        feature (string): the quantitavie feature to measure (y value)
        plates (string): the variable representing experimental plates for grouping

    Returns:
        _type_: _description_
    """
    pd.options.mode.copy_on_write = True

    subset = [group, feature, plates]

    df = data.dropna(subset=subset).reset_index(drop=True)
    df = df[df[feature] != 0]

    df_subset = df[subset]
    df_subset[group] = df[group].astype("category")
    df_subset.reset_index(drop=True, inplace=True)

    return df_subset


def relate_objects(
    obj_df_1,
    obj_df_2,
    obj1_name="",
    obj2_name="",
    feature_cols=None,
    ratio_colname="Cell_Nuclei_Area_Ratio",
    metadata_cols=None,
    max_x=2160.0,
    max_y=2160.0,
):
    """Relates two tables of segmented images
    Based off of ImageNumber and where objects in obj_df_2 are contained within larger objects in obj_df_1.
    NOTE: Bounding box (min_row, min_col, max_row, max_col). Pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col)
    Args:
        obj_df_1 (pd.DataFrame): DataFrame for the first (larger) objects e.g. cells.
                                 Expected columns: "ImageNumber", "label", "bbox-0", "bbox-1", "bbox-2", "bbox-3".
                                 "bbox-0", "bbox-1" typically represent min_x, max_x.
                                 "bbox-2", "bbox-3" typically represent max_y, max_y.
        obj_df_2 (pd.DataFrame): DataFrame for the second (smaller, contained) objects e.g. nuclei.
                                 Expected columns: "ImageNumber", "label", "bbox-0", "bbox-1", "bbox-2", "bbox-3".
                                 "bbox-0", "bbox-1" typically represent min_x, max_x.
                                 "bbox-2", "bbox-3" typically represent max_y, max_y.
        obj1_name (str, optional): Name of the first object (e.g., "Cell"). Defaults to "".
        obj2_name (str, optional): Name of the second object (e.g., "Nucleus"). Defaults to "".

    Returns:
        pd.DataFrame: DataFrame with the related objects, including a "Parent_Obj1Name_Number_Object_Number"
                      column in the obj_df_2 data, and a combined DataFrame with aggregated means
                      and a calculated ratio.
    """
    if feature_cols is None:
        feature_cols = []
    if metadata_cols is None:
        metadata_cols = [
            "ImageNumber",
            "PlateNumber",
            "Metadata_WellRow",
            "Metadata_WellColumn",
            "Metadata_Field",
            "SerialPassage_BatchNumber",
            "AgeGroup",
            "Drug",
            "slice",
            "Filename",
            "Parent_Folder",
            "Path",
            "Metadata_Well_ID",
            "Metadata_Well_x",
            "Block",
            "Metadata_Well_y",
            "TimepointName",
            "Staining",
            "Passage Group",
            "AllGroups",
        ]
    obj_df_1 = obj_df_1.sort_values(by="ImageNumber").copy()
    obj_df_2 = obj_df_2.sort_values(by="ImageNumber").copy()

    related_second_objects = []
    for i, obj1_row in obj_df_1.iterrows():
        obj1_min_x, obj1_min_y, obj1_max_x, obj1_max_y = (
            # Bounding box (min_row, min_col, max_row, max_col)
            obj1_row["bbox-0"],
            obj1_row["bbox-1"],
            obj1_row["bbox-2"],
            obj1_row["bbox-3"],
        )
        parent_obj_label = obj1_row["label"]
        img_number = obj1_row["ImageNumber"]

        # Filter obj_df_2 to only use the current ImageNumber
        current_img_obj2_df = obj_df_2[obj_df_2["ImageNumber"] == img_number]

        for j, obj2_row in current_img_obj2_df.iterrows():
            if obj_df_2["ImageNumber"][j] > i:
                break
            # relate if the second object is contained within the parent object  (min_row, min_col, max_row, max_col)
            obj2_min_x, obj2_min_y, obj2_max_x, obj2_max_y = (
                obj2_row["bbox-0"],
                obj2_row["bbox-1"],
                obj2_row["bbox-2"],
                obj2_row["bbox-3"],
            )

            # assign conditions to booleans
            # Pixels belonging to the bounding box: [min_row; max_row) and [min_col; max_col)
            is_contained_x = (obj2_min_x >= obj1_min_x) and (obj2_max_x <= obj1_max_x)
            is_contained_y = (obj2_min_y >= obj1_min_y) and (obj2_max_y <= obj1_max_y)

            # also make a flag to check if second object is toucjing border
            touching_border_nuc = bool(
                (obj2_min_x == 0)
                or (
                    obj2_max_x == (max_x * 0.25)
                )  # multiply by in the origial downlampling factor from the masks (0.25)
                or (obj2_min_y == 0)
                or (obj2_max_y == (max_y * 0.25))
            )

            if is_contained_x and is_contained_y:
                # print(f"second object bbox: {obj2_row["slice"]}, touching border? {touching_border_nuc}")
                # Create a copy to avoid SettingWithCopyWarning
                second_obj_with_parent = obj2_row.copy()
                second_obj_with_parent[f"Parent_{obj1_name}_Number_Object_Number"] = (
                    parent_obj_label
                )
                second_obj_with_parent[f"{obj2_name}_Touching_Border"] = (
                    touching_border_nuc
                )
                related_second_objects.append(second_obj_with_parent)

    # case when you don't have any relationships
    if not related_second_objects:
        return pd.DataFrame  # empty df

    second_objs_related_df = pd.DataFrame(related_second_objects)
    # Aggregate means of the second objects by their assigned parent and ImageNumber - group by the parent label and ImageNumber for aggregation
    boolean_cols = second_objs_related_df.select_dtypes(include=bool).columns

    if not feature_cols:
        feature_cols = second_objs_related_df.select_dtypes(
            include=np.number
        ).columns.tolist()  # list of all numeric cols only

    # make a dictionary to tell pandas what aggregations to do
    aggregations = {col: "mean" for col in feature_cols}

    for col in boolean_cols:
        aggregations[col] = "mean"  # Proportion of True values

    for col in metadata_cols:
        aggregations[col] = "first"  # Assuming metadata is consistent within a group

    # Get a count column to see # of nuclei
    second_obj_counts = second_objs_related_df.groupby(
        ["ImageNumber", f"Parent_{obj1_name}_Number_Object_Number"]
    ).count()["area"]  # arbitrary col

    # aggregate using the dictionary above
    second_obj_means = second_objs_related_df.groupby(
        ["ImageNumber", f"Parent_{obj1_name}_Number_Object_Number"]
    ).agg(aggregations)  # Ensure only numeric columns are averaged

    second_obj_means[f"Children_{obj2_name}_Count"] = second_obj_counts

    # second_obj_means = second_objs_related_df.groupby(
    #     ["ImageNumber", f"Parent_{obj1_name}_Number_Object_Number"]
    # ).mean(numeric_only=True)  # Ensure only numeric columns are averaged
    second_obj_means = second_obj_means.rename_axis(  # change index of df to "label"
        index={f"Parent_{obj1_name}_Number_Object_Number".format(obj1_name): "label"}
    )

    # second_obj_means = second_obj_means[second_obj_means["label"] +1] #one-indexed
    # Join on ImageNumber and the label of the first object (which is the parent label in second_obj_means)
    joined_obj_df = obj_df_1.join(
        second_obj_means,
        on=["ImageNumber", "label"],
        how="left",
        rsuffix=f"_mean_{obj2_name}",
    )
    # display(joined_obj_df)
    # display(joined_obj_df.head(10), joined_obj_df.shape)
    # Calculate the ratio, ensuring the columns exist and handling potential NaNs
    if (
        f"{obj1_name}_AreaShape_Area" in joined_obj_df.columns
        and f"{obj2_name}_AreaShape_Area" in joined_obj_df.columns
    ):
        joined_obj_df[ratio_colname] = (
            joined_obj_df[f"{obj1_name}_AreaShape_Area"]
            / joined_obj_df[f"{obj2_name}_AreaShape_Area"]
        )
    else:
        print(
            f"Warning: 'AreaShape_Area' columns not found for ratio calculation. Expected: {obj1_name}_AreaShape_Area and {obj2_name}_AreaShape_Area_mean"
        )
        joined_obj_df[ratio_colname] = float("nan")

    # Drop rows where the ratio could not be calculated (due to missing second object data)
    relate_objects_df = joined_obj_df.dropna(subset=[ratio_colname])
    # second_obj_means = second_objs_df.groupby("ImageNumber").agg("mean")
    # joined_df = obj_df_1.join(second_obj_means, on=["ImageNumber","label"], x=obj1_name,y=obj2_name, how="left")
    # joined_df["CellNucRatio"] = joined_df.apply(lambda x: x[f"{obj1_name}_AreaShape_Area"]/x[f"{obj2_name}_AreaShape_Area"])

    return relate_objects_df


####
# Normalization functions
####
def normalize_quantities_to_control_group_average(
    df,
    quantitiy_cols,
    group_col,
    control_value,
    plate_number_col="PlateNumber",
    overwrite=False,
    drop_avg_cols=False,
):
    # Create a new DataFrame to store the normalized values
    normalized_df = df.copy()

    # find average of the control group for each plate
    ctrl_averages = (
        df[df[group_col] == control_value]
        .groupby([plate_number_col])[quantitiy_cols]
        .mean()
        .reset_index()
    )
    # Merge the control averages back to the original DataFrame
    normalized_df = normalized_df.merge(
        ctrl_averages, on=[plate_number_col], suffixes=("", "_CtrlAvg")
    )

    # Normalize the intensity columns by dividing by the well average
    for col in quantitiy_cols:
        new_name = f"{col}_CtrlNormalized"
        if overwrite:
            normalized_df[col] = normalized_df[col] / normalized_df[f"{col}_CtrlAvg"]
        else:
            if new_name not in normalized_df.columns:
                normalized_df[new_name] = (
                    normalized_df[col] / normalized_df[f"{col}_CtrlAvg"]
                )

    # Drop the well average columns
    if drop_avg_cols:
        normalized_df.drop(
            columns=[f"{col}_CtrlAvg" for col in quantitiy_cols], inplace=True
        )

    return normalized_df


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


# Normalize featues row by row (SLOW)
def normalize_features(df, feature_list):
    """
    Normalize the features in the DataFrame to the control (age group 0) for each plate.
    Args:
        df (DataFrame): The DataFrame containing the features to be normalized.
        feature_list (list): A list of feature column names to normalize.
    Returns:
        DataFrame: A DataFrame with normalized features for each plate.
    """
    # Normalize the features to the control (age group 0) for each plate
    norm_df = df.copy()
    for feature in feature_list:
        # print('Normalizing feature: ', feature, '...', norm_df[feature].values[0])
        norm_df[feature] = normalize_to_control(df, feature)
        # print('Normalized feature: ', feature, '...', norm_df[feature].values[0])
    return norm_df


def normalize_to_control(df, feature, norm_column="AgeGroup"):
    """
    Normalize a feature to the control group (AgeGroup = 0) for each plate.
    Args:
        df (DataFrame): The DataFrame containing the feature to be normalized.
        feature (str): The name of the feature column to normalize.
        norm_column (str): The column used to identify the control group (default is 'AgeGroup').'`
    Returns:
        Series: A Series containing the normalized feature values.
    """
    # Take the t0 df - lowest passage data point
    t0_df = df[df[norm_column] == 0]
    treatment_df = df[[feature, norm_column]].copy()

    # calculate the mean
    mean_zero = t0_df[feature].mean()
    # Check for non-numeric values
    if not pd.api.types.is_numeric_dtype(treatment_df[feature]):
        print(f"[normalize_to_control] WARNING: {feature} is not numeric!")
    # now update the column to have all rows dividied by the mean of group 0
    treatment_df["norm_" + feature] = treatment_df[feature] / mean_zero
    # return the normalized feature columnn
    return treatment_df["norm_" + feature]


def apply_feature_normalization(df, feature_dict, curr_plates):
    """
    Apply feature normalization to the DataFrame for each plate in a list of plates.
    Args:
        df (DataFrame): The DataFrame containing the features to be normalized.
        feature_dict (dict): A dictionary containing lists of feature columns to normalize.
        curr_plates (list): A list of plate names to apply normalization to.
    Returns:
        DataFrame: A DataFrame with normalized features for each plate.
    """
    # Normalize the features to the control (age group 0) for each plate
    norm_cell_df = df.copy()
    for plate in curr_plates:
        curr_plate_df = norm_cell_df[norm_cell_df["Metadata_Plate"] == plate].copy()
        for feature_type in feature_dict:
            # get the normalized features, locate the corresponding features on the plate, and replace them on that plate to the plate
            curr_plate_features_df = normalize_features(
                curr_plate_df, feature_dict[feature_type]
            )
            curr_plate_df.loc[:, feature_dict[feature_type]] = curr_plate_features_df[
                feature_dict[feature_type]
            ].astype(float)
        norm_cell_df.loc[norm_cell_df["Metadata_Plate"] == plate] = curr_plate_df
    return norm_cell_df


def get_valid_numeric_features(df, feature_dicts):
    all_valid_features = []
    for feat_dict in feature_dicts:
        all_cols = [col for cols in feat_dict.values() for col in cols]
        for col in set(all_cols):
            if pd.api.types.is_numeric_dtype(df[col]):
                all_valid_features.append(col)
    return list(
        dict.fromkeys(all_valid_features)
    )  # remove duplicates while preserving order


####
# Merging functions
####


def combine_one_to_one_dfs(df, db_conn, tables_to_add=None, main_df_prefix="Cell"):
    """Merge table outputs from cellprofiler where the tables are objects in a one-to-one relationship e.g. cells and nuclei
    Args:
        df (DataFrame): _description_
        db_conn (sqlite3 Connection object): _description_
        tables_to_add (list of str, optional): _description_. Defaults to ["Nuclei","Cytoplasm"].

    Returns:
        DataFrame: _description_
    """
    if tables_to_add is None:
        tables_to_add = ["Nuclei", "Cytoplasm"]
    # add all these to a list
    main_df = df.copy()
    dfs_to_add = []
    for table in tables_to_add:
        new_df = pd.read_sql_query(f"SELECT * FROM Per_{table};", db_conn)
        dfs_to_add.append(new_df)

    # now merge to the cell df
    for i, object_df in enumerate(dfs_to_add):
        compartment = tables_to_add[i]
        # special case for nuclei; where we want cell to be the primary table but nuclei are the parent of a cell
        if compartment == "Nuclei":
            main_df = pd.merge(
                main_df,
                object_df,
                how="left",  # left on the object index, image-by-imageCell_Parent_Nuclei
                left_on=[f"{main_df_prefix}_Parent_{compartment}", "ImageNumber"],
                right_on=[f"{compartment}_Number_Object_Number", "ImageNumber"],
            )
        else:
            main_df = pd.merge(
                main_df,
                object_df,
                how="left",  # left on the object index, image-by-imageCell_Parent_Nuclei
                left_on=[f"{main_df_prefix}_Number_Object_Number", "ImageNumber"],
                right_on=[f"{compartment}_Parent_{main_df_prefix}", "ImageNumber"],
            )

    merged_df_final = main_df.reset_index(drop=True)
    return merged_df_final


def merge_totalobject_df_into_parent_df(
    parent_df,
    object_df,
    key_col="ImageNumber_Object_Number",
    rename_key=True,
    verbose=False,
):
    """
    Merges two DataFrames with unequal rows based on a common key created from the ImageNumber and Object_Number columns. The function performs a left join, keeping all rows from the parent DataFrame and adding matching rows from the object DataFrame. If there are no matches, NaN values will be filled in for the object DataFrame's columns.
    """
    parent_name = parent_df.columns[1].split("_")[
        0
    ]  # Extract the table name from the second column
    object_name = object_df.columns[1].split("_")[
        0
    ]  # Extract the table name from the second column

    if "Total" in object_name:
        short_object_name = object_name.replace("Total", "")
    else:
        short_object_name = object_name
    parent_df[f"{parent_name}_Children_{short_object_name}_Count"] = parent_df[
        f"{parent_name}_Children_{short_object_name}_Count"
    ].astype(int)

    # first drop rows from the parent_df where the count of the child object is less than 1
    mask_df = parent_df[
        parent_df[f"{parent_name}_Children_{short_object_name}_Count"] < 1
    ]
    images_with_dropped_objects = mask_df["ImageNumber"].unique()
    dropped_parent_df = parent_df.drop(mask_df.index)

    # Add a new column for the new object numbers after dropping rows to match the other dataframe
    # copy the original object number column to a new column
    if f"{parent_name}_Number_Object_Number" in dropped_parent_df.columns:
        dropped_parent_df["New_Object_Number"] = dropped_parent_df[
            f"{parent_name}_Number_Object_Number"
        ]
    else:
        print(
            f"Column {parent_name}_Number_Object_Number not found in dropped_parent_df. Please check the column names."
        )
        dropped_parent_df["New_Object_Number"] = None
    for img in images_with_dropped_objects:
        # for the dropped images, we need to reassign the object numbers to be sequential starting from 1
        if img in dropped_parent_df["ImageNumber"].values:
            df_at_img = dropped_parent_df.loc[
                dropped_parent_df["ImageNumber"] == img
            ].copy()
            if verbose:
                print(
                    f"Number of objects in Image {img}: {len(df_at_img)} with objects dropped from the DataFrame subset"
                )
            if not df_at_img.empty:
                # reassign the object numbers to be sequential starting from 1 ONLY for that image
                print(
                    f"Reassigning object numbers for Image {img} in dropped_parent_df."
                )
                df_at_img.loc[:, "New_Object_Number"] = range(1, len(df_at_img) + 1)
            dropped_parent_df.update(df_at_img)
    # make the key columns for the object_df to merge on
    dropped_parent_df[key_col] = (
        "Img"
        + dropped_parent_df["ImageNumber"].astype(str)
        + "_"
        + dropped_parent_df["New_Object_Number"].astype(str)
    )
    object_df[key_col] = (
        "Img"
        + object_df["ImageNumber"].astype(str)
        + "_"
        + object_df[f"{object_name}_Number_Object_Number"].astype(str)
    )

    if verbose:
        print(
            f"dropped_parent_df shape: {dropped_parent_df.shape}, object_df shape: {object_df.shape}, original parent_df shape: {parent_df.shape}"
        )
    merged_parent_df = dropped_parent_df.merge(
        object_df, on=key_col, how="left", suffixes=("", f"_{short_object_name}")
    )
    if rename_key:
        merged_parent_df.rename(
            columns={key_col: f"ImageNumber_Object_Number_{short_object_name}"},
            inplace=True,
        )
    if verbose:
        print(f"merged_parent_df shape: {merged_parent_df.shape}")

    return merged_parent_df


#####
# Groupby funcitons
#####


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


def average_groups_pivot(group_avg_df, x_value, y_value, plate_col_name):
    """Make a pivot table from the averaged dataframe

    Args:
        df (DataFrame): your dataframe output from average_groups_by_plate()
        x_value (string): the grouping variable (x value)
        y_value (string): the quantitavie feature to measure (y value)
        plates (string): the variable representing experimental plates for grouping

    Returns:
        DataFrame: a pivot table
    """
    group_avg_pivot = group_avg_df.pivot_table(
        columns=x_value, values=y_value, index=plate_col_name
    )
    return group_avg_pivot


#####
# Summary statistics functions
#####


def make_summary_stats_for_df_and_feature(
    df,
    x_value,
    feature,
    summary_outpath,
    df_tag="original",
    plate_col_name="PlateNumber",
    feature_name="area",
    group_name="passage_group",
    include_cols=[],
):
    from pathlib import Path

    try:
        table_csvname = f"{df_tag}_total_combined_{feature_name}_stats.csv"
        feature_csvname = f"{df_tag}_{feature_name}_by_{group_name}_stats.csv"
        agg_feature_csvname = f"{df_tag}_agg_{feature_name}_by_{group_name}_stats.csv"

        subfolder_name = f"{df_tag}_{feature_name}_summary_stats"
        parent_folder = Path(summary_outpath, subfolder_name)
        parent_folder.mkdir(exist_ok=True)

        if not include_cols:
            df_to_summarize = df
        else:
            df_to_summarize = df[include_cols]
        df_to_summarize.describe().to_csv(
            os.path.join(summary_outpath, subfolder_name, table_csvname)
        )
        group_averages = df.groupby(
            [x_value, plate_col_name], as_index=False, observed=True
        )[feature]
        # Reset the index to get a clean DataFrame
        # average_df = group_averages.reset_index()
        avg_summary = group_averages.describe()
        avg_summary_sorted = avg_summary.sort_values(
            by=[x_value], key=lambda x: x.map(passage_groups_sort_key)
        ).reset_index(drop=True)
        avg_summary_sorted.to_csv(
            os.path.join(summary_outpath, subfolder_name, feature_csvname)
        )

        # do the agg by passage group only
        group_averages_agg = df.groupby([x_value], as_index=False, observed=True)[
            feature
        ]
        avg_agg_summary = group_averages_agg.describe()
        avg_agg_summary_sorted = avg_agg_summary.sort_values(
            by=[x_value], key=lambda x: x.map(passage_groups_sort_key)
        )
        avg_agg_summary_sorted.to_csv(
            os.path.join(summary_outpath, subfolder_name, agg_feature_csvname)
        )
        print(
            f"saved files {(table_csvname, feature_csvname, agg_feature_csvname)} to {summary_outpath}"
        )
        return True
    except ValueError as e:
        print(f"Could not make summary stats: {e}")
        return False
