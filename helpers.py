"""
Helper functions for data analysis and visualization
Allie Spangaro, Toronto Metropolitan University
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


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


def define_cell_features(df):
    """_summary_

    Args:
        df (DataFrame): _description_

    Returns:
        list: A list of columns that are numerical features
    """    
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
    # old_columns_list = columns_list = [col for col in columns_list if 'Metadata' not in col and 'FileName' not in col and 'PathName' not in col]
    # print("Original columns:", len(old_columns_list), "Filtered columns:", len(columns_list))
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
        "arearatios": [],
        "count": [],
        "distance": [],
        "metadata": [],
    }
    for col in columns_list:
        if "Texture" in col:
            feature_dict["texture"].append(col)
        elif "Intensity" in col:
            feature_dict["intensity"].append(col)
        elif "Math_" in col or "Corr_" in col:
            feature_dict["arearatios"].append(col)
        elif "Count" in col:
            feature_dict["count"].append(col)
        elif "AreaShape" in col:
            feature_dict["areashape"].append(col)
        elif "Distance" in col:
            feature_dict["distance"].append(col)
        elif "Granularity" in col:
            feature_dict["granularity"].append(col)
        elif "RadialDistribution" in col:
            feature_dict["radialdistribution"].append(col)
        else:
            feature_dict["metadata"].append(col)

    return feature_dict


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


# Function to find the ratio between two columns in the two dataframes and return the ratio as a column
def ratioCalc(df1, df2, col1, col2):
    # Deprecate this function
    int1 = df1[col1]
    int2 = df2[col2]

    temp_copy1 = outlier_removal(df1, int1)
    temp_copy2 = outlier_removal(df2, int2)

    intensity_ratio = temp_copy1[int1] / temp_copy2[int2]
    return df1[intensity_ratio]


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


def outlier_removal(df, nuclei_df, column):
    """OLD FUNCTION - DO NOT USE

    Args:
        df (_type_): _description_
        nuclei_df (_type_): _description_
        column (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # Create a copy of the column and the 'group' column, along with parent nuclei
    mini_df = pd.DataFrame(
        {
            column: df[column].copy(),
            "Time": df["Time"].copy(),
            "Parent_Nuclei": df["Parent_Nuclei"].copy(),
            "ImageNumber": df["ImageNumber"].copy(),
        }
    )

    # remove stuff within 1 SD above of the mean of the oldest passage
    nuc_oldest_mean = []
    nuc_oldest_std_dev = []
    try:
        nuc_oldest_mean = mini_df[mini_df["Time"] == 6][column].mean()
        nuc_oldest_std_dev = mini_df[mini_df["Time"] == 6][column].std()
    except:
        nuc_oldest_mean = mini_df[mini_df["Time"] == 4][column].mean()
        nuc_oldest_std_dev = mini_df[mini_df["Time"] == 4][column].std()
    nuc_threshold = nuc_oldest_mean + (nuc_oldest_std_dev)

    # print('threshold ', nuc_threshold, 'stdev=',nuc_t4_std_dev, 'mean=', nuc_t4_mean)
    # Filter the nuclei dataframe to remove rows with AreaShape_Area above the threshold
    filtered_nuclei_df = nuclei_df[nuclei_df["AreaShape_MeanRadius"] <= nuc_threshold]

    # Filter the cell dataframe to keep only rows where Parent_Nuclei is in the filtered DF
    filtered_cell_df = mini_df.merge(
        filtered_nuclei_df[["Number_Object_Number", "ImageNumber"]],
        left_on=["Parent_Nuclei", "ImageNumber"],
        right_on=["Number_Object_Number", "ImageNumber"],
        how="inner",
    )

    # Filter out values less than 0
    final_filtered_df = filtered_cell_df[filtered_cell_df[column] > 0].dropna()
    return final_filtered_df


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


def mean_intesity_per_compartment_per_cell(df, compartment, tag):
    """_summary_

    Args:
        df (_type_): _description_
        compartment (_type_): _description_
        tag (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # Calculate the mean intensity of each compartment per cell
    # mean_intesity_per_compartment = integrated / (children*mean_area)
    colname = "MeanIntensity_Per_" + compartment + "_Per_Cell"
    integrated = "Intensity_IntegratedIntensity_" + tag
    children = "Children_" + compartment + "_Count"
    mean_area = "Mean_" + compartment + "_AreaShape_Area"
    df[colname] = df.apply(
        lambda x: x[integrated] / (x[children] * x[mean_area]), axis=1
    )
    return df[colname]


def proportion_area_occupied_per_cell(df, compartment):
    """_summary_

    Args:
        df (_type_): _description_
        compartment (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # proportion of area occupied = children * mean organelle area / cell area
    colname = "Total_Area_Proportion_" + compartment + "_Per_Cell"

    # children = 'Children_' + compartment + '_Count'
    # mean_organelle_area = 'Mean_'+ compartment + '_AreaShape_Area'
    organelle_area = compartment + "_AreaShape_Area"
    cell_area = "AreaShape_Area"
    # df[colname] = df.apply(lambda x: (x[children] * x[mean_organelle_area]) / x[cell_area], axis=1)
    df[colname] = df.apply(lambda x: (x[organelle_area]) / x[cell_area], axis=1)
    return df[colname]


def proportion_area_occupied_per_cell_fromtotal(df, compartment):
    """_summary_

    Args:
        df (DataFrame): _description_
        compartment (string): _description_

    Returns:
        Series: The column to add
    """    
    # proportion of area occupied = children * mean organelle area / cell area
    colname = "Total_Area_Proportion_" + compartment + "_Per_Cell"

    # children = 'Children_' + compartment + '_Count'
    # mean_organelle_area = 'Mean_'+ compartment + '_AreaShape_Area'
    organelle_area = compartment + "_AreaShape_Area"
    cell_area = "AreaShape_Area"
    # df[colname] = df.apply(lambda x: (x[children] * x[mean_organelle_area]) / x[cell_area], axis=1)
    df[colname] = df.apply(lambda x: (x[organelle_area]) / x[cell_area], axis=1)
    return df[colname]


def mean_intesity_per_compartment_per_cell_fromtotal(df, compartment, name, tag):
    """_summary_

    Args:
        df (DataFrame): _description_
        compartment (_type_): _description_
        name (_type_): _description_
        tag (_type_): _description_

    Returns:
        Series: the column to add
    """    
    # Calculate the mean intensity of each compartment per cell
    # mean_intesity_per_compartment = integrated / (children*mean_area)
    colname = "MeanIntensity_Per_" + compartment + "_Per_Cell"
    integrated = "Intensity_IntegratedIntensity_" + tag
    # children = 'Children_' + compartment + '_Count'
    # mean_area = 'Mean_'+ compartment + '_AreaShape_Area'
    total_organelle_area = name + "_AreaShape_Area"
    df[colname] = df.apply(lambda x: x[integrated] / x[total_organelle_area], axis=1)
    return df[colname]


def average_groups_by_plate(df, x_value, y_value, replicates):
    """
    Group the DataFrame by the specified columns and calculate the mean of the y_value column.
    Returns the averaged dataframe for plotting
    
    Args:
        df (DataFrame): your dataframe
        x_value (string): the grouping variable (x value)
        y_value (string): the quantitavie feature to measure (y value)
        replicates (string): the variable representing experimental replicates for grouping

    Returns:
        DataFrame: your data grouped by replicate
    """
    df = df.dropna(subset=[x_value, y_value, replicates])
    df = df[df[y_value] != 0]

    df.reset_index(drop=True, inplace=True)
    
    group_averages = df.groupby([x_value, replicates], as_index=False, observed=True).agg({y_value: "mean"})
    
    # Reset the index to get a clean DataFrame
    average_df = group_averages.reset_index()
   
    return average_df


def make_single_feature_df(data, group, feature, replicates):
    """Make a dataframe for a single feature from a larger dataframe in "tidy" format

    Args:
        data (DataFrame): your dataframe
        group (string): the grouping variable (x value)
        feature (string): the quantitavie feature to measure (y value)
        replicates (string): the variable representing experimental replicates for grouping

    Returns:
        _type_: _description_
    """    
    pd.options.mode.copy_on_write = True

    subset = [group, feature, replicates]

    df = data.dropna(subset=subset).reset_index(drop=True)
    df = df[df[feature] != 0]

    df_subset = df[subset]
    df_subset[group] = df[group].astype("category")
    df_subset.reset_index(drop=True, inplace=True)

    return df_subset


def average_groups_by_plate_v0(df, x_value, y_value, replicates):
    """
    Group the DataFrame by the specified columns and calculate the mean of the y_value column.
    Returns the averaged dataframe for plotting
    
    Args:
        df (DataFrame): your dataframe
        x_value (string): the grouping variable (x value)
        y_value (string): the quantitavie feature to measure (y value)
        replicates (string): the variable representing experimental replicates for grouping

    Returns:
        DataFrame: your data grouped by replicate
    """
    df = df.dropna(subset=[x_value, y_value, replicates])
    # df.reset_index(drop=True, inplace=True) - don't need this?

    group_averages = df.groupby(
        [x_value, replicates], as_index=False, observed=True
    ).agg({y_value: "mean"})

    # Reset the index to get a clean DataFrame
    average_df = group_averages.reset_index()

    return average_df


def average_groups_pivot(df, x_value, y_value, replicates):
    """Make a pivot table from the averaged dataframe

    Args:
        df (DataFrame): your dataframe output from average_groups_by_plate()
        x_value (string): the grouping variable (x value)
        y_value (string): the quantitavie feature to measure (y value)
        replicates (string): the variable representing experimental replicates for grouping

    Returns:
        DataFrame: a pivot table
    """    
    group_ave_pivot = df.pivot_table(columns=x_value, values=y_value, index=replicates)
    return group_ave_pivot
