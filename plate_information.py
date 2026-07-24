"""
Helper functions for data analysis and visualization
Allie Spangaro, Toronto Metropolitan University
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import operator


def old_get_all_group_order():
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


def find_plate(path):
    import re

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


def passage_groups_sort_key(group_name):
    import re

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


def make_summary_stats_for_df_and_feature(
    df,
    x_value,
    feature,
    summary_outpath,
    df_tag="original",
    plate_col_name="Plate_Number",
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


def get_mini_filtered_df(
    final_filtered_df,
    condition_col="",
    valueslist=[
        "Cell_Unique_ID",
        "ImageNumber",
        "TimepointName",
        "Metadata_WellRow",
        "Metadata_WellColumn",
        "Metadata_Field",
        "AllGroups",
        "Plate_Number",
        "SerialPassage_BatchNumber",
        "AgeGroup",
        "PassageNumber",
        "Number_Object_Number",
        "AreaShape_Area",
        "Nuclei_AreaShape_Area",
        "Cell_Nuclei_Area_Ratio",
        "Children_Mitochondria_Count",
        "Children_Lysosomes_Count",
        "Image_Width_DAPI",
        "Image_URL_MitoTracker_MAX",
        "Image_FileName_MitoTracker_MAX",
        "Image_FileName_LAMP1_MAX",
    ],
    op=operator.le,
    condition_value=10,
):
    mini_df = final_filtered_df[valueslist]

    mini_df["Metadata_Rep_RowColField"] = (
        "R"
        + mini_df["Plate_Number"].astype(str)
        + "_r"
        + mini_df["Metadata_WellRow"].astype(str)
        + "c"
        + mini_df["Metadata_WellColumn"].astype(str)
        + "f"
        + mini_df["Metadata_Field"].astype(str)
        + ""
    )

    # Now add the filter
    filter_mini_df = mini_df[op(mini_df[condition_col], condition_value)]
    filter_mini_df_sorted = filter_mini_df.sort_values(
        by=["AllGroups"], key=lambda x: x.map(passage_groups_sort_key)
    ).reset_index(drop=True)

    # reduce cols for readability
    filter_mini_df_display = filter_mini_df_sorted[
        [
            "AllGroups",
            "Plate_Number",
            "Cell_Unique_ID",
            "Image_FileName_MitoTracker_MAX",
            condition_col,
        ]
    ]
    print(filter_mini_df_display)
    return filter_mini_df_sorted


def find_plate_cp_output_folder(path):
    import re

    plate_pattern = r"_rep0(\d{1})_"  # Matches "RX" where X is the plate number (placeholder for now)
    match = re.search(plate_pattern, path)
    if match:
        plate = int(match.group(1))
    else:
        plate = None
    return plate


def pull_up_cp_segmentation_image_fromID(
    df,
    object_key,
    parent_dir="",
    plate_col_name="Plate_Number",
    feature="",
    image_channel="LAMP1",
    save=False,
    savepath="",
):
    """Function to pull up an image with cellprofiler segmentation outlines that has a matching image in the dataframe
    and also highlight the sepecific object with a bounding box

    Args:
        object_key (int): the uniqueobject key in the dataframe
        parent_dir (str, optional): _description_. Defaults to "~/".
        img_filename (str): _description_.
        plate (int, optional): _description_. Defaults to 0.
        group (str, optional): _description_. Defaults to "".
    """
    from matplotlib import image as mpimg
    from PIL import Image

    # get the filename, plate and coords from the unique ID if the ID exists
    if object_key is not None and df is not None:
        rect = get_object_bbox_coordinates_as_rectangle(df, object_key)

        unique_row = df[df["Cell_Unique_ID"] == object_key]
        img_filename = unique_row[f"Image_FileName_{image_channel}_MAX"].values[0]
        plate = unique_row[plate_col_name].values[0]
        passage = unique_row["PassageNumber"].values[0]
        group = unique_row["AllGroups"].values[0]
    else:
        print(f"Object number {object_key} not found in dataframe.")
        return False
    # loop over to find the file in the directory
    img_filename_noext = img_filename.split(".")[0]
    for root, dirs, files in os.walk(parent_dir):
        for filename in files:
            if (
                img_filename_noext.split("_")[1] in filename
                and filename.endswith(".png")
                and "active" in root  # active cp output folder
                and plate == find_plate_cp_output_folder(root)
            ):
                print(filename)
                img_path = os.path.join(
                    root,
                    filename,  # img_filename_noext + ".png"
                )  # make the path
                # Now we see if the image exists and try to open it
                try:
                    img = Image.open(img_path)
                    fig, ax = plt.subplots(figsize=(12, 12))
                    plt.imshow(img)
                    plt.axis("off")  # Turn off axis labels for a cleaner image display
                    plt.title(
                        f"ID:{object_key}, R{plate} P{passage}, {filename}, {group}"
                    )
                    # Lets add a rectangle if the oject key is in the dataframe
                    ax.add_patch(rect)
                    # add a label with the feature value if it exists
                    if feature in df.columns:
                        feature_value = unique_row[feature].values[0]
                        plt.text(
                            rect.get_x(),
                            rect.get_y() - 10,
                            f"{feature}: {feature_value:.2f}",
                            color="lime",
                            fontsize=12,
                            weight="bold",
                            bbox=dict(facecolor="black", alpha=0.5, pad=2),
                        )
                    if save:
                        plt.savefig(
                            f"{savepath}/R{plate}_P{passage}_{filename}_{object_key}.png",
                            bbox_inches="tight",
                        )
                    plt.show()
                    print(f"Segmented image url: {img_path}")
                    # img.show()
                    return True
                except FileNotFoundError:
                    print(
                        f"Image file {img_path} not found. Please ensure 'your_image.png' exists."
                    )
    return False


def get_object_bbox_coordinates_as_rectangle(
    df,
    unique_ID,
    coord_cols=[
        "AreaShape_BoundingBoxMaximum_X",
        "AreaShape_BoundingBoxMaximum_Y",
        "AreaShape_BoundingBoxMinimum_X",
        "AreaShape_BoundingBoxMinimum_Y",
    ],
):
    from matplotlib import patches

    row = df[df["Cell_Unique_ID"] == unique_ID]
    if not row.empty:
        x_min = row["AreaShape_BoundingBoxMinimum_X"].values[0]
        y_min = row["AreaShape_BoundingBoxMinimum_Y"].values[0]
        x_max = row["AreaShape_BoundingBoxMaximum_X"].values[0]
        y_max = row["AreaShape_BoundingBoxMaximum_Y"].values[0]
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
    return rect


def query_group_plate_condition(
    df, group, plate_number=0, condition_col="", op=operator.eq, value=None
):
    """
    Filter df by group, plate_number, and a condition using a passed operator.
    Example: op=operator.lt for '<', op=operator.gt for '>', op=operator.eq for '=='
    """
    mask = (
        (df["AllGroups"] == group)
        & (df["Plate_Number"] == plate_number)
        & (op(df[condition_col], value))
    )
    return df[mask]


def get_unique_cols_to_use(df):
    base_cols = [
        # "FileName_MitoTracker_MAX",
        "Metadata_PlateNumber",
        "Metadata_RowColField",
        "AllGroups",
        "AreaShape_Area",
    ]
    areashape_features = [
        "AreaShape_Area",
        "AreaShape_Perimeter",
        "AreaShape_EquivalentDiameter",
    ]

    colnames_mitoskel_nuc = search_column_name(df, "Nuclei_ObjectSkeleton")
    colnames_mitocount = search_column_name(
        df, ["Children_Mito", "Count"], inclusive_or=False
    )
    colnames_mitoarea = search_column_name(
        df, ["Mito", "AreaShape_Area"], inclusive_or=False
    )

    colnames_lysocount = search_column_name(
        df, ["Children_Lyso", "Count"], inclusive_or=False
    )
    colnames_lysoarea_initial = search_column_name(
        df, areashape_features, inclusive_or=True
    )
    colnames_lysoarea = colnames_lysoarea_initial.copy()

    for col in colnames_lysoarea_initial:
        if "Lyso" in col and "AreaShape" in col:
            continue
        else:
            colnames_lysoarea.remove(col)

    colnames_ij = search_column_name(df, "IJ_Mitochondria")

    use_cols = (
        base_cols
        + colnames_mitoskel_nuc
        + colnames_ij
        + colnames_mitocount
        + colnames_mitoarea
        + colnames_lysocount
        + colnames_lysoarea
    )
    use_cols_unique = list(dict.fromkeys(use_cols))
    # scrub out any non-numeric columns that we aren't going to use for analysis
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
        if "Mean" in col or "Median" in col or "Stdev" in col or "FromBranches" in col:
            colnames_object_skeleton_length.remove(col)

    print(colnames_object_skeleton_length)
    return colnames_object_skeleton_length
