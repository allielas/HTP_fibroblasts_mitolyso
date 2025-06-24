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
    from sklearn.preprocessing import StandardScaler

    # Import the scaler and transform all time values to that of a standard distribution - only use for ML, not very desceiptive
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df[columns])
    return scaled_df


def group_by_condition(df, feature_list, groupby_column="AgeGroup"):
    # Group columns by age group and apply groupby function to the DF
    df_groupby = df.groupby(groupby_column).apply(
        lambda x: standardize_group(x, feature_list)
    )
    return df_groupby


def outlier_removal(df, nuclei_df, column):
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
    # proportion of area occupied = children * mean organelle area / cell area
    colname = "Total_Area_Proportion_" + compartment + "_Per_Cell"

    # children = 'Children_' + compartment + '_Count'
    # mean_organelle_area = 'Mean_'+ compartment + '_AreaShape_Area'
    organelle_area = compartment + "_AreaShape_Area"
    cell_area = "AreaShape_Area"
    # df[colname] = df.apply(lambda x: (x[children] * x[mean_organelle_area]) / x[cell_area], axis=1)
    df[colname] = df.apply(lambda x: (x[organelle_area]) / x[cell_area], axis=1)
    return df[colname]


def remove_outliers_iqr(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(
            "number"
        ).columns  # limits to a (float), b (int) and e (timedelta)
    df_sub = df.loc[:, cols]

    iqr = df_sub.quantile(0.75, numeric_only=False) - df_sub.quantile(
        0.25, numeric_only=False
    )

    # calculate  extreme outlisers by dividing median by iqr
    lim = np.abs((df_sub - df_sub.median()) / iqr) < 2.22

    # replace outliers with nan
    df.loc[:, cols] = df_sub.where(lim, np.nan)
    df.dropna(subset=cols, inplace=True)  # drop rows with NaN in numerical columns
    return df


def make_superviolinplot_with_kruskal(
    data, group, feature_meas, replicates, ytitle=None, pallete="bright", ylim=None
):
    order = get_all_group_order()

    if ytitle is None:
        ytitle = feature_meas.replace("_", " ")
    if ylim is None:
        ylim = (-1, 12)

    feature_df = make_single_feature_df(
        data, group=group, feature=feature_meas, replicates=replicates
    )
    pairs = getpairs(feature_df, group, order)

    # Remove the n=1 replicate
    feature_df = feature_df[feature_df[group] != "P22-24"]

    group_avg_df = average_groups_by_plate(
        feature_df, x_value=group, y_value=feature_meas, replicates=replicates
    )
    group_avg_df_pivot = average_groups_pivot(
        group_avg_df, x_value=group, y_value=feature_meas, replicates=replicates
    )

    sns.set_theme(style="ticks")
    sns.set_context("talk", font_scale=0.6)

    plt.figure(dpi=300)

    sns.violinplot(
        data=feature_df,
        x=group,
        y=feature_meas,
        order=order,
        fill=False,
        color="gainsboro",
        cut=2,
        native_scale=True,
        linecolor="k",
        inner=None,
        # inner_kws=dict(box_width = 5)
    )

    ax = sns.swarmplot(
        data=group_avg_df,
        x=group,
        y=feature_meas,
        hue=replicates,
        order=order,
        palette=pallete,
        size=10,
        edgecolor="k",
        linewidth=1,
        dodge=0.5,
    )

    sns.pointplot(
        data=group_avg_df,
        x=group,
        y=feature_meas,
        color="dimgray",
        order=order,
        dodge=False,
        markers="_",
        linestyle=None,
        errorbar=None,
        ax=ax,
    )

    ax.legend_.remove()

    sns.despine()
    plt.gcf()  # .set_size_inches(10, 6)
    plt.xlabel(group)
    plt.ylabel(ytitle)
    plt.ylim(ylim)

    from statannotations.Annotator import Annotator

    annotator = Annotator(ax, pairs, data=group_avg_df_pivot, order=order)
    annotator.configure(
        test="Kruskal",
        text_format="star",
        loc="inside",
        hide_non_significant=True,
        color="black",
        verbose=2,
    )
    annotator.apply_and_annotate()

    plt.savefig(feature_meas + "_superviolinplot.png", dpi=300)
    plt.show()


def proportion_area_occupied_per_cell_fromtotal(df, compartment):
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
    """
    df = df.dropna(subset=[x_value, y_value, replicates])
    df = df[df[y_value] != 0]

    df.reset_index(drop=True, inplace=True)

    group_averages = df.groupby(
        [x_value, replicates], as_index=False, observed=True
    ).agg({y_value: "mean"})

    # Reset the index to get a clean DataFrame
    average_df = group_averages.reset_index()

    return average_df


def make_single_feature_df(data, group, feature, replicates):
    pd.options.mode.copy_on_write = True

    subset = [group, feature, replicates]

    df = data.dropna(subset=subset).reset_index(drop=True)
    df = df[df[feature] != 0]

    df_subset = df[subset]
    df_subset[group] = df[group].astype("category")
    df_subset.reset_index(drop=True, inplace=True)

    return df_subset


def oneway_anova(data, group_name, feature_meas):
    from scipy.stats import f_oneway

    data = data.dropna(subset=[group_name, feature_meas])
    data = data[data[feature_meas] != 0]

    groups = data[group_name].unique()
    data = [data[data[group_name] == group][feature_meas].dropna() for group in groups]
    anova_result = f_oneway(*data)

    print(
        f"ANOVA F-statistic: {anova_result.statistic}, ANOVA p-value: {anova_result.pvalue}"
    )
    return anova_result


def tukey_test(data, test_groups, feature):
    """
    Perform a oneway anova test and a pairwise tukey post hoc test using averaged values per replicate
    Returns a dataframe
    """
    from scipy.stats import f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    df = data.copy()

    # groups = getpairs(temp_copy, 'Passage Group')
    # calculate tukey HSD

    tukey = pairwise_tukeyhsd(endog=df[feature], groups=df[test_groups], alpha=0.05)

    # Extract relevant results
    results = np.array(tukey.summary().data)[:, [0, 1, 3, 6]]
    df_results = pd.DataFrame(
        results, columns=["Group 1", "Group 2", "p-value", "Reject"]
    ).drop([0])
    df_results.reset_index(drop=True, inplace=True)
    df_results[["Group 1", "Group 2"]] = df_results[["Group 1", "Group 2"]]
    df_results["p-value"] = df_results["p-value"].astype(float)

    return df_results


def average_groups_by_plate_v0(df, x_value, y_value, replicates):
    """
    Group the DataFrame by the specified columns and calculate the mean of the y_value column.
    Returns the averaged dataframe for plotting
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
    group_ave_pivot = df.pivot_table(columns=x_value, values=y_value, index=replicates)
    return group_ave_pivot
