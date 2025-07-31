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


def find_replicate(path):
    import re

    replicate_pattern = r"R(\d{1})"  # Matches "RX" where X is the replicate number (placeholder for now)
    match = re.search(replicate_pattern, path)
    if match:
        replicate = int(match.group(1))
    else:
        replicate = None
    return replicate


def find_row_col(well_code):
    import re

    rowcol_pattern = r"r(\d{1,2})c(\d{1,2})"  # Matches "RX" where X is the replicate number (placeholder for now)
    match = re.search(rowcol_pattern, well_code)
    if match:
        row_metadata = int(match.group(1))
        col_metadata = int(match.group(2))
    else:
        row_metadata = None
        col_metadata = None
    return row_metadata, col_metadata


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


def average_groups_pivot(group_avg_df, x_value, y_value, replicate_col_name):
    """Make a pivot table from the averaged dataframe

    Args:
        df (DataFrame): your dataframe output from average_groups_by_plate()
        x_value (string): the grouping variable (x value)
        y_value (string): the quantitavie feature to measure (y value)
        replicates (string): the variable representing experimental replicates for grouping

    Returns:
        DataFrame: a pivot table
    """
    group_avg_pivot = group_avg_df.pivot_table(
        columns=x_value, values=y_value, index=replicate_col_name
    )
    return group_avg_pivot


def passage_groups_sort_key(group_name):
    import re

    """
    Key function for natural sorting of strings containing numbers.
    Extract numeric parts and convert to int .
    """
    digit_pattern = r"([0-9]+)"  # Matches "RX" where X is the replicate number (placeholder for now)
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
