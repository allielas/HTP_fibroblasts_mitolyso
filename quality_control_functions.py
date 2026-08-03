import operator
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
import scikit_posthocs as sp
import seaborn as sns
from matplotlib import image as mpimg
from matplotlib import patches
from PIL import Image
from scipy import stats
from skimage import color

from mitolyso_plot_functions import *
from plate_information import *
from plate_preprocessing import *

try:
    from IPython.display import display
except ImportError:

    def display(*args, **kwargs):
        return None


try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("matplotlib", "inline")
except Exception:
    pass


def fix_column_names(df):
    # remove "Cells_" from the beginning of column names
    df.columns = df.columns.str.replace(r"^Cell_", "", regex=True)
    # also rename nuclei cols in the filtered df when you made the df one-to-one
    df.columns = df.columns.str.replace(r"^Mean_Nuclei_", "Nuclei_", regex=True)
    df["Cell_Nuclei_Area_Ratio"] = df[
        "Nuclei_Area_Ratio"
    ]  # dupe column with the old name back
    unique_ID_df = df.reset_index().rename(
        columns={"index": "Cell_Unique_ID"}  # add a unique ID
    )
    return unique_ID_df


def normalize_intensity_to_well_average(df, intensity_cols, drop_well_avg_cols=False):
    # Create a new DataFrame to store the normalized values
    normalized_df = df.copy()

    # Group by 'Plate_Number' and 'Metadata_Well' to calculate the average intensity for each well
    well_averages = (
        df.groupby(["Plate_Number", "Metadata_Well"])[intensity_cols]
        .mean()
        .reset_index()
    )

    # Merge the well averages back to the original DataFrame
    normalized_df = normalized_df.merge(
        well_averages, on=["Plate_Number", "Metadata_Well"], suffixes=("", "_WellAvg")
    )

    # Normalize the intensity columns by dividing by the well average
    for col in intensity_cols:
        new_name = f"{col}_WellNormalized"
        normalized_df[new_name] = normalized_df[col] / normalized_df[f"{col}_WellAvg"]

    # Drop the well average columns
    if drop_well_avg_cols:
        normalized_df.drop(
            columns=[f"{col}_WellAvg" for col in intensity_cols], inplace=True
        )

    return normalized_df


def get_intensity_cols(df, channels=None):
    intensity_cols = []
    if channels is None:
        channels = ["DAPI", "LAMP1", "MitoTracker"]
    for channel in channels:
        channel = re.escape(
            channel
        )  # escape any special characters in the channel name
        if channel == "DAPI":
            pattern = rf"^(?=.*Nuclei_Intensity)(?=.*(?:MaxIntensity_{channel}|MeanIntensity_{channel}|MinIntensity_{channel}|MedianIntensity_{channel}))(?=.*_MAX)"
        else:
            pattern = rf"^Intensity_(?=.*(?:MaxIntensity_{channel}|MeanIntensity_{channel}|MinIntensity_{channel}|MedianIntensity_{channel}))(?=.*_MAX)"
        intensity_cols += [
            col
            for col in df.columns
            # regex to match columns that contain our desired intensity cols
            if re.search(pattern, col)
        ]
    return intensity_cols


def enforce_objects_one_to_one(
    df,
    parent_obj="Cell",
    child_obj="Nuclei",
    parent_colname="Cell_AreaShape_Area",
    child_colname="Cell_Mean_Nuclei_AreaShape_Area",
):
    """apply filters based on the number of nuclei to remove to enforce a 1:1 cell-nucleus relationship:
    - Cells that have less or more than one nucleus
    - Poorly segmented cells where the cell area is smaller than the nuclei area
    - Image is not flagged as empty by CellProfiler
    Can also use other objects instead of nuclei

    Args:
        df (DataFrame): The parent dataframe
        child_obj (str, optional): _description_. Defaults to "Nuclei".
        parent_colname (str, optional): _description_. Defaults to "Cell_AreaShape_Area".
        child_colname (str, optional): _description_. Defaults to "Cell_Mean_Nuclei_AreaShape_Area".

    Returns:
        DataFrame: the filtered dataframe with the removed objects
    """
    if child_obj == "Nuclei" and parent_obj == "Cell":
        try:
            not_empty_df = df[df["Metadata_EmptyImage_Cells"] == 0]
            normal_cells = not_empty_df[
                not_empty_df["Cell_Classify_one_nuc"] == 1
            ]  # one nucleus only
        except KeyError as e:
            print(f"KeyError {e}; skipping emptyimage")
            normal_cells = df[df["Cell_Classify_one_nuc"] == 1]
        # remove if my cell area is bigger than nuclear area
        size_filtered_cells = normal_cells[
            normal_cells[parent_colname] > normal_cells[child_colname]
        ]
        final_df = size_filtered_cells.reset_index(drop=True)
        return final_df
    else:
        # just do the size excludion
        not_empty_df = df[df["Metadata_EmptyImage_Cells"] == 0]
        size_filtered_cells = not_empty_df[
            not_empty_df[parent_colname] > not_empty_df[child_colname]
        ]
        return size_filtered_cells


def filter_out_empty_compartment_from_cells(
    df, organelle, prefix="", min_compartments=1
):
    """Remove rows from a dataframe where a parent "cell" object doesn't have any child objects of {organelle}

    Args:
        df (DataFrame): _description_
        organelle (str): the organelle in plural. Typically "Mitochondria or Lysosomes (or Nuclei)
        prefix (str, optional): _description_. Defaults to "".

    Returns:
        Dataframe: _description_
    """
    organelle = organelle.title()
    this_df = df.copy()
    atleast_one_df = this_df[
        this_df[f"{prefix}Children_{organelle}_Count"] > min_compartments
    ].reset_index(drop=True)
    return atleast_one_df


def hard_size_shape_filter_rows_in_df(
    df,
    area_col="Nuclei_AreaShape_Area",
    prefix="",
    min_threshold=0,
    max_threshold=np.inf,
):
    """Remove rows from a dataframe that are above a size theshold
    Args:
        df (DataFrame): _description_.
        area_col (str, optional): _description_. Defaults to "Nuclei_AreaShape_Area".
        organelle (str): the organelle in plural. Typically "Mitochondria or Lysosomes (or Nuclei)
        prefix (str, optional): _description_. Defaults to "".

    Returns:
        Dataframe: _description_
        min_rows_removed: number of rows removed by the min threshold
        max_rows_removed: number of rows removed by the max threshold
    """
    data_df = df.copy()
    data_df = data_df[data_df[prefix + area_col] > min_threshold]
    min_rows_removed = df.shape[0] - data_df.shape[0]
    data_df_2 = data_df[data_df[prefix + area_col] < max_threshold]
    max_rows_removed = data_df.shape[0] - data_df_2.shape[0]
    final_df = data_df_2.reset_index(drop=True)
    return final_df, min_rows_removed, max_rows_removed


def filter_out_images_with_n_cells(
    df, n, image_count_col="Image_Count_Cell", prefix=""
):
    filter_df = df.copy()
    filter_df = filter_df[filter_df[image_count_col] > n]
    final_df = filter_df.reset_index(drop=True)
    return final_df


def get_min_max_percentile_thesholds(
    df,
    column,
    min_percentile=0.025,
    max_percentile=0.975,
    subset_col="",
    min_subset_group=None,
    max_subset_group=None,
):
    if subset_col:
        if subset_col not in df.columns:
            raise ValueError(f"Subset column {subset_col} not found in dataframe")
        if min_subset_group is not None:
            df_min = df[df[subset_col] == min_subset_group]
            min_threshold = df_min[column].quantile(min_percentile)
            # print(f"Min {column} threshold for {min_subset_group}: {min_threshold}")
        else:
            min_threshold = df[column].min()

        if max_subset_group is not None:
            df_max = df[df[subset_col] == max_subset_group]
            max_threshold = df_max[column].quantile(max_percentile)
            # print(f"Max {column} threshold for {max_subset_group}: {max_threshold}")
        else:
            max_threshold = df[column].max()

    else:
        min_threshold = df[column].quantile(min_percentile)
        max_threshold = df[column].quantile(max_percentile)

    return min_threshold, max_threshold


def add_filtering_summary_to_dict(
    filtering_summary_dict,
    filter_id,
    filter_name,
    filtered_df,
    previous_df,
    threshold_info,
    rows_removed=None,
):
    """_summary_

    Args:
        filtering_summary_dict (_type_): _description_
        filter_id (_type_): _description_
        filter_name (_type_): _description_
        filtered_df (_type_): _description_
        previous_df (_type_): _description_
        threshold_info (_type_): _description_
        rows_removed (_type_, optional): manual override for rows removed. Defaults to None.

    Returns:
        _type_: _description_
    """
    if rows_removed is not None:
        print(f"Manually overriding rows removed for {filter_name} to {rows_removed}")
    else:
        rows_removed = previous_df.shape[0] - filtered_df.shape[0]

    filtering_summary_dict[filter_id] = {
        "filter_type": filter_name,
        "columns": filtered_df.shape[1],
        "rows_initial": previous_df.shape[0],
        "rows_removed": rows_removed,
        "rows": filtered_df.shape[0],
        "threshold": threshold_info,
    }
    return filtering_summary_dict


def apply_all_filters(
    df,
    reference_df=None,  # optional df to use for calculating the thresholds to avoid biasing the thresholds by the filters
    nuc_size_min_threshold_percentile=0.025,
    nuc_size_max_threshold_percentile=0.975,
    cell_size_min_threshold_percentile=0.025,
    cell_size_max_threshold_percentile=0.975,
    cell_nuc_area_minthreshold_percentile=0.025,
    cell_nuc_area_maxthreshold_percentile=0.975,
    cell_eccentricity_min_threshold_percentile=0.025,
    cell_eccentricity_max_threshold_percentile=0.975,
    nuc_eccentricity_min_threshold_percentile=0.025,
    nuc_eccentricity_max_threshold_percentile=0.975,
    min_cell_count=0,
    min_nuc_intensity_percentile="",
    max_nuc_intensity_percentile="",
    max_nuc_intensity_hardthreshold=2.5,
    max_neighbours_percenttouching_percentile=0.975,
    max_neighbours_numberof_percentile=0.975,
    max_nuc_neighbours_numberof_percentile=0.975,
    cell_nuc_ratio_col="Cell_Nuclei_Area_Ratio",
    area_col="AreaShape_Area",
    nuc_area_col="Nuclei_AreaShape_Area",
    cell_eccentricity_col="",
    nuc_eccentricity_col="",
    neighbours_col_percenttouching="Neighbors_PercentTouching_5",
    neighbours_col_numberof="Neighbors_NumberOfNeighbors_5",
    nuc_neighbours_col_numberof="Nuclei_Neighbors_NumberOfNeighbors_1",
    image_count_col="Image_Count_Cell",
    norm_nuc_intensity_col="Nuclei_Intensity_MedianIntensity_DAPI_MAX_WellNormalized",
    norm_mito_intensity_col="Intensity_MeanIntensity_MitoTracker_MAX_WellNormalized",
    norm_lamp1_intensity_col="Intensity_MeanIntensity_LAMP1_MAX_WellNormalized",
    min_compartments=0,
    filtering_summary_dict=None,
):
    prev_df = (
        df  # enforce_objects_one_to_one(df)  # , child_colname="Nuclei_AreaShape_Area")
    )
    if reference_df is not None:
        reference_df = reference_df.copy()
    else:
        reference_df = prev_df.copy()

    if filtering_summary_dict is None:
        filtering_summary_dict = {}
        filtering_summary_dict["Initial_Dataset"] = {
            "filter_type": "Initial Dataset",
            "columns": prev_df.shape[1],
            "rows_initial": prev_df.shape[0],
            "rows_removed": 0,
            "threshold": f"{None},{None}",
        }

    # filter out the cells wo compartments
    filtered_df = filter_out_empty_compartment_from_cells(
        prev_df, "mitochondria", min_compartments=min_compartments
    )
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        "Number_of_Mitochondria_Filter",
        "Min Number of Mitochondria Filter",
        filtered_df,
        prev_df,
        f"min_compartments={min_compartments}",
    )
    prev_df = filtered_df.copy()

    filtered_df = filter_out_empty_compartment_from_cells(
        prev_df, "lysosomes", min_compartments=min_compartments
    )
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        "Number_of_Lysosomes_Filter",
        "Min Number of Lysosomes Filter",
        filtered_df,
        prev_df,
        f"min_compartments={min_compartments}",
    )
    prev_df = filtered_df.copy()

    # cell size filter
    min_threshold_cell_size, max_threshold_cell_size = get_min_max_percentile_thesholds(
        reference_df,
        area_col,
        min_percentile=cell_size_min_threshold_percentile,
        max_percentile=cell_size_max_threshold_percentile,
        subset_col="AllGroups",
        min_subset_group="P6-12",
        max_subset_group="Doxo",
    )
    filtered_df, min_cells_removed, max_cells_removed = (
        hard_size_shape_filter_rows_in_df(
            prev_df,
            min_threshold=min_threshold_cell_size,
            max_threshold=max_threshold_cell_size,
            area_col=area_col,
            prefix="",
        )
    )
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        area_col + "_Filter_Bottom",
        "Cell Size Filter Bottom Band",
        filtered_df,
        prev_df,
        f"Bottom threshold: {min_threshold_cell_size}",
        rows_removed=min_cells_removed,
    )
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        area_col + "_Filter_Top",
        "Cell Size Filter Top Band",
        filtered_df,
        prev_df,
        f"Top threshold: {max_threshold_cell_size}",
        rows_removed=max_cells_removed,
    )
    prev_df = filtered_df.copy()

    # nuc size filter (using untrimmed df to get the thresholds to avoid biasing the thresholds by the cell size filter)
    min_threshold_nuc_size, max_threshold_nuc_size = get_min_max_percentile_thesholds(
        reference_df,
        nuc_area_col,
        min_percentile=nuc_size_min_threshold_percentile,
        max_percentile=nuc_size_max_threshold_percentile,
        subset_col="AllGroups",
        min_subset_group="P6-12",
        max_subset_group="Doxo",
    )
    filtered_df, min_nuc_removed, max_nuc_removed = hard_size_shape_filter_rows_in_df(
        filtered_df,
        min_threshold=min_threshold_nuc_size,
        max_threshold=max_threshold_nuc_size,
        area_col=nuc_area_col,
        prefix="",
    )
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        nuc_area_col + "_Filter_Bottom",
        "Nuclear Size Filter Bottom Band",
        filtered_df,
        prev_df,
        f"Bottom threshold: {min_threshold_nuc_size}",
        rows_removed=min_nuc_removed,
    )
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        nuc_area_col + "_Filter_Top",
        "Nuclear Size Filter Top Band",
        filtered_df,
        prev_df,
        f"Top threshold: {max_threshold_nuc_size}",
        rows_removed=max_nuc_removed,
    )
    prev_df = filtered_df.copy()

    # cell:nuc ratio filter
    # bottom band
    min_cell_nuc_ratio_threshold, max_cell_nuc_ratio_threshold = (
        get_min_max_percentile_thesholds(
            reference_df,
            cell_nuc_ratio_col,
            min_percentile=cell_nuc_area_minthreshold_percentile,
            max_percentile=cell_nuc_area_maxthreshold_percentile,
            subset_col="AllGroups",
            min_subset_group="P6-12",
            max_subset_group="Doxo",
        )
    )
    filtered_df = filtered_df[
        filtered_df[cell_nuc_ratio_col] > min_cell_nuc_ratio_threshold
    ]
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        cell_nuc_ratio_col + "_Filter_Bottom",
        "Cell:Nuclear Area Ratio Filter Bottom Band",
        filtered_df,
        prev_df,
        f"Bottom threshold: {min_cell_nuc_ratio_threshold}",
    )
    prev_df = filtered_df.copy()

    # top band
    filtered_df = filtered_df[
        filtered_df[cell_nuc_ratio_col] < max_cell_nuc_ratio_threshold
    ]
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        cell_nuc_ratio_col + "_Filter_Top",
        "Cell:Nuclear Area Ratio Filter Top Band",
        filtered_df,
        prev_df,
        f"Top threshold: {max_cell_nuc_ratio_threshold}",
    )
    prev_df = filtered_df.copy()

    if cell_eccentricity_col and nuc_eccentricity_col:
        min_cell_eccentricity_threshold, max_cell_eccentricity_threshold = (
            get_min_max_percentile_thesholds(
                reference_df,
                cell_eccentricity_col,
                min_percentile=cell_eccentricity_min_threshold_percentile,
                max_percentile=cell_eccentricity_max_threshold_percentile,
                subset_col="AllGroups",
                min_subset_group="P6-12",
                max_subset_group="Doxo",
            )
        )
        min_nuc_eccentricity_threshold, max_nuc_eccentricity_threshold = (
            get_min_max_percentile_thesholds(
                reference_df,
                nuc_eccentricity_col,
                min_percentile=nuc_eccentricity_min_threshold_percentile,
                max_percentile=nuc_eccentricity_max_threshold_percentile,
                subset_col="AllGroups",
                min_subset_group="P6-12",
                max_subset_group="Doxo",
            )
        )
        filtered_df, min_cell_eccentricity_removed, max_cell_eccentricity_removed = (
            hard_size_shape_filter_rows_in_df(
                filtered_df,
                area_col=cell_eccentricity_col,
                min_threshold=min_cell_eccentricity_threshold,
                max_threshold=max_cell_eccentricity_threshold,
            )
        )
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            cell_eccentricity_col + "_Filter_Bottom",
            "Cell Eccentricity Filter Bottom Band",
            filtered_df,
            prev_df,
            f"Bottom threshold: {min_cell_eccentricity_threshold}",
            rows_removed=min_cell_eccentricity_removed,
        )
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            cell_eccentricity_col + "_Filter_Top",
            "Cell Eccentricity Filter",
            filtered_df,
            prev_df,
            f"Top threshold: {max_cell_eccentricity_threshold}",
            rows_removed=max_cell_eccentricity_removed,
        )
        prev_df = filtered_df.copy()

        filtered_df, min_nuc_eccentricity_removed, max_nuc_eccentricity_removed = (
            hard_size_shape_filter_rows_in_df(
                filtered_df,
                area_col=nuc_eccentricity_col,
                min_threshold=min_nuc_eccentricity_threshold,
                max_threshold=max_nuc_eccentricity_threshold,
            )
        )
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            nuc_eccentricity_col + "_Filter_Bottom",
            "Nuclear Eccentricity Filter Bottom Band",
            filtered_df,
            prev_df,
            f"Bottom threshold: {min_nuc_eccentricity_threshold}",
            rows_removed=min_nuc_eccentricity_removed,
        )
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            nuc_eccentricity_col + "_Filter_Top",
            "Nuclear Eccentricity Filter Top Band",
            filtered_df,
            prev_df,
            f"Top threshold: {max_nuc_eccentricity_threshold}",
            rows_removed=max_nuc_eccentricity_removed,
        )
        prev_df = filtered_df.copy()

    # else:
    #     # if not using this filter, just set the min and max intensity filtered dfs to be the same as the previous filter so that the rest of the filters can run without error
    #     filter_df_cell_eccentricity = filter_df_maxratio.copy()
    #     filter_df_nuc_eccentricity = filter_df_maxratio.copy()
    #     min_cell_eccentricity_threshold = None
    #     max_cell_eccentricity_threshold = None
    #     min_nuc_eccentricity_threshold = None
    #     max_nuc_eccentricity_threshold = None

    # nuc intensity filter
    if norm_nuc_intensity_col:
        if min_nuc_intensity_percentile == "" or max_nuc_intensity_percentile == "":
            min_nuc_intensity_threshold = filtered_df[norm_nuc_intensity_col].min()
            max_nuc_intensity_threshold = max_nuc_intensity_hardthreshold
        else:
            min_nuc_intensity_threshold = 0
            max_nuc_intensity_threshold = max_nuc_intensity_hardthreshold
        print(f"Max {norm_nuc_intensity_col} threshold: {max_nuc_intensity_threshold}")
        # filter the bottom and add to the summary dict
        filtered_df = filtered_df[
            filtered_df[norm_nuc_intensity_col] > min_nuc_intensity_threshold
        ]
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            norm_nuc_intensity_col + "_Filter_Bottom",
            norm_nuc_intensity_col.replace("_", " ") + " Filter Bottom Band",
            filtered_df,
            prev_df,
            f"Bottom threshold: {min_nuc_intensity_threshold}",
            # rows_removed=len(filtered_df) - len(prev_df),
        )
        prev_df = filtered_df.copy()

        # now filter the top and add to the summary dict
        filtered_df = filtered_df[
            filtered_df[norm_nuc_intensity_col] < max_nuc_intensity_threshold
        ]
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            norm_nuc_intensity_col + "_Filter_Top",
            norm_nuc_intensity_col.replace("_", " ") + " Filter Top Band",
            filtered_df,
            prev_df,
            f"Top threshold: {max_nuc_intensity_threshold}",
            # rows_removed=len(filtered_df) - len(prev_df),
        )
        prev_df = filtered_df.copy()
    else:
        prev_df = filtered_df.copy()

    # more than one cell filter (not using)
    if min_cell_count > 0:
        filtered_df = filtered_df[filtered_df[image_count_col] > min_cell_count]
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            image_count_col + "_Filter",
            "Min Cells Per Image Filter",
            filtered_df,
            prev_df,
            f"Min threshold: {min_cell_count}",
        )
        prev_df = filtered_df.copy()

    # neighbors filters
    max_threshold_neighbours_percenttouching = get_min_max_percentile_thesholds(
        reference_df,
        neighbours_col_percenttouching,
        min_percentile=0,
        max_percentile=max_neighbours_percenttouching_percentile,
        subset_col="AllGroups",
        max_subset_group="P6-12",
    )[1]
    max_threshold_neighbours_numberof = get_min_max_percentile_thesholds(
        reference_df,
        neighbours_col_numberof,
        min_percentile=0,
        max_percentile=max_neighbours_numberof_percentile,
        subset_col="AllGroups",
        max_subset_group="P6-12",
    )[1]
    if (
        max_threshold_neighbours_percenttouching == 0
        or max_threshold_neighbours_numberof == 0
    ):
        print("One or more max neighbor thresholds are 0. Trying with closest distance")
        # this filter would be the inverse to remove cells with too many neigbours
        min_neighbour_distance_theshold = get_min_max_percentile_thesholds(
            reference_df[
                reference_df["Neighbors_FirstClosestDistance_5"] > 0
            ],  # use only non-zero distances to avoid biasing the threshold by the zero distances
            "Neighbors_FirstClosestDistance_5",
            min_percentile=0.025,
            max_percentile=1,
        )[0]
        filtered_df = filtered_df[
            (
                filtered_df["Neighbors_FirstClosestDistance_5"]
                > min_neighbour_distance_theshold
            )
            | (filtered_df["Neighbors_FirstClosestDistance_5"] == 0)
        ]  # optional filter for touching neighbors AND number of neighbors
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            "Cell_Neighbours_Filter",
            "First Closest Neighbour distance < 0.025th percentile (97.5th closest distance) or == 0 (no neighbours)",
            filtered_df,
            prev_df,
            f"Min closest neighbour distance: {min_neighbour_distance_theshold}",
        )
    else:
        filtered_df = filtered_df[
            (
                filtered_df[neighbours_col_percenttouching]
                < max_threshold_neighbours_percenttouching
            )
            & (filtered_df[neighbours_col_numberof] < max_threshold_neighbours_numberof)
        ]  # optional filter for touching neighbors AND number of neighbors
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            "Cell_Neighbours_Filter",
            "Percent of Cell Touching Neighbours AND Number Of Neighbours < 97.5th Percentile",
            filtered_df,
            prev_df,
            f"Max Percent Touching: {max_threshold_neighbours_percenttouching} and Max Number Of Neighbours: {max_threshold_neighbours_numberof}",
        )
    prev_df = filtered_df.copy()

    max_threshold_nuc_neighbours_numberof = get_min_max_percentile_thesholds(
        reference_df,
        nuc_neighbours_col_numberof,
        min_percentile=0,
        max_percentile=max_nuc_neighbours_numberof_percentile,
        subset_col="AllGroups",
        max_subset_group="P6-12",
    )[1]
    filtered_df = filtered_df[
        filtered_df[nuc_neighbours_col_numberof]
        <= max_threshold_nuc_neighbours_numberof
    ]  # optional filter for number of neighbouring nuclei
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        "Nuc_Neighbours_Filter",
        "Number Of Nuclei Neighbours < 97.5th Percentile",
        filtered_df,
        prev_df,
        f"Max Number Of Nuclei Neighbours: {max_threshold_nuc_neighbours_numberof}",
    )
    prev_df = filtered_df.copy()

    final_filtered_df = (
        filtered_df.copy()
    )  # or filter_df_size if you want the size filter

    # return both to export as CSV
    return final_filtered_df, filtering_summary_dict


def apply_filters_individually(
    df,
    reference_df=None,  # optional df to use for calculating the thresholds to avoid biasing the thresholds by the filters
    nuc_size_min_threshold_percentile=0.025,
    nuc_size_max_threshold_percentile=0.975,
    cell_size_min_threshold_percentile=0.025,
    cell_size_max_threshold_percentile=0.975,
    cell_nuc_area_minthreshold_percentile=0.025,
    cell_nuc_area_maxthreshold_percentile=0.975,
    cell_eccentricity_min_threshold_percentile=0.025,
    cell_eccentricity_max_threshold_percentile=0.975,
    nuc_eccentricity_min_threshold_percentile=0.025,
    nuc_eccentricity_max_threshold_percentile=0.975,
    min_cell_count=0,
    min_nuc_intensity_percentile="",
    max_nuc_intensity_percentile="",
    max_nuc_intensity_hardthreshold=2.5,
    max_neighbours_percenttouching_percentile=0.975,
    max_neighbours_numberof_percentile=0.975,
    max_nuc_neighbours_numberof_percentile=0.975,
    cell_nuc_ratio_col="Cell_Nuclei_Area_Ratio",
    area_col="AreaShape_Area",
    nuc_area_col="Nuclei_AreaShape_Area",
    cell_eccentricity_col="",
    nuc_eccentricity_col="",
    neighbours_col_percenttouching="Neighbors_PercentTouching_5",
    neighbours_col_numberof="Neighbors_NumberOfNeighbors_5",
    nuc_neighbours_col_numberof="Nuclei_Neighbors_NumberOfNeighbors_1",
    image_count_col="Image_Count_Cell",
    norm_nuc_intensity_col="Nuclei_Intensity_MedianIntensity_DAPI_MAX_WellNormalized",
    norm_mito_intensity_col="Intensity_MeanIntensity_MitoTracker_MAX_WellNormalized",
    norm_lamp1_intensity_col="Intensity_MeanIntensity_LAMP1_MAX_WellNormalized",
    min_compartments=0,
    filtering_summary_dict=None,
):
    prev_df = df.copy()
    if reference_df is not None:
        reference_df = reference_df.copy()
    else:
        reference_df = prev_df.copy()

    if filtering_summary_dict is None:
        filtering_summary_dict = {}
        filtering_summary_dict["Initial_Dataset"] = {
            "filter_type": "Initial Dataset",
            "columns": prev_df.shape[1],
            "rows_initial": prev_df.shape[0],
            "rows_removed": 0,
            "threshold": f"{None},{None}",
        }

    # filter out the cells wo compartments
    filtered_df = filter_out_empty_compartment_from_cells(
        prev_df, "mitochondria", min_compartments=min_compartments
    )
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        "Number_of_Mitochondria_Filter",
        "Min Number of Mitochondria Filter",
        filtered_df,
        prev_df,
        f"min_compartments={min_compartments}",
    )
    filtered_df = prev_df.copy()
    prev_df = filtered_df.copy()

    filtered_df = filter_out_empty_compartment_from_cells(
        prev_df, "lysosomes", min_compartments=min_compartments
    )
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        "Number_of_Lysosomes_Filter",
        "Min Number of Lysosomes Filter",
        filtered_df,
        prev_df,
        f"min_compartments={min_compartments}",
    )
    filtered_df = prev_df.copy()
    prev_df = filtered_df.copy()

    # cell size filter
    min_threshold_cell_size, max_threshold_cell_size = get_min_max_percentile_thesholds(
        reference_df,
        area_col,
        min_percentile=cell_size_min_threshold_percentile,
        max_percentile=cell_size_max_threshold_percentile,
        subset_col="AllGroups",
        min_subset_group="P6-12",
        max_subset_group="Doxo",
    )
    filtered_df, min_cells_removed, max_cells_removed = (
        hard_size_shape_filter_rows_in_df(
            prev_df,
            min_threshold=min_threshold_cell_size,
            max_threshold=max_threshold_cell_size,
            area_col=area_col,
            prefix="",
        )
    )
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        area_col + "_Filter_Bottom",
        "Cell Size Filter Bottom Band",
        filtered_df,
        prev_df,
        f"Bottom threshold: {min_threshold_cell_size}",
        rows_removed=min_cells_removed,
    )
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        area_col + "_Filter_Top",
        "Cell Size Filter Top Band",
        filtered_df,
        prev_df,
        f"Top threshold: {max_threshold_cell_size}",
        rows_removed=max_cells_removed,
    )
    filtered_df = prev_df.copy()
    prev_df = filtered_df.copy()

    # nuc size filter (using untrimmed df to get the thresholds to avoid biasing the thresholds by the cell size filter)
    min_threshold_nuc_size, max_threshold_nuc_size = get_min_max_percentile_thesholds(
        reference_df,
        nuc_area_col,
        min_percentile=nuc_size_min_threshold_percentile,
        max_percentile=nuc_size_max_threshold_percentile,
        subset_col="AllGroups",
        min_subset_group="P6-12",
        max_subset_group="Doxo",
    )
    filtered_df, min_nuc_removed, max_nuc_removed = hard_size_shape_filter_rows_in_df(
        filtered_df,
        min_threshold=min_threshold_nuc_size,
        max_threshold=max_threshold_nuc_size,
        area_col=nuc_area_col,
        prefix="",
    )
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        nuc_area_col + "_Filter_Bottom",
        "Nuclear Size Filter Bottom Band",
        filtered_df,
        prev_df,
        f"Bottom threshold: {min_threshold_nuc_size}",
        rows_removed=min_nuc_removed,
    )
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        nuc_area_col + "_Filter_Top",
        "Nuclear Size Filter Top Band",
        filtered_df,
        prev_df,
        f"Top threshold: {max_threshold_nuc_size}",
        rows_removed=max_nuc_removed,
    )
    filtered_df = prev_df.copy()
    prev_df = filtered_df.copy()

    # cell:nuc ratio filter
    # bottom band
    min_cell_nuc_ratio_threshold, max_cell_nuc_ratio_threshold = (
        get_min_max_percentile_thesholds(
            reference_df,
            cell_nuc_ratio_col,
            min_percentile=cell_nuc_area_minthreshold_percentile,
            max_percentile=cell_nuc_area_maxthreshold_percentile,
            subset_col="AllGroups",
            min_subset_group="P6-12",
            max_subset_group="Doxo",
        )
    )
    filtered_df = filtered_df[
        filtered_df[cell_nuc_ratio_col] > min_cell_nuc_ratio_threshold
    ]
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        cell_nuc_ratio_col + "_Filter_Bottom",
        "Cell:Nuclear Area Ratio Filter Bottom Band",
        filtered_df,
        prev_df,
        f"Bottom threshold: {min_cell_nuc_ratio_threshold}",
    )
    # top band
    filtered_df = filtered_df[
        filtered_df[cell_nuc_ratio_col] < max_cell_nuc_ratio_threshold
    ]
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        cell_nuc_ratio_col + "_Filter_Top",
        "Cell:Nuclear Area Ratio Filter Top Band",
        filtered_df,
        prev_df,
        f"Top threshold: {max_cell_nuc_ratio_threshold}",
    )
    filtered_df = prev_df.copy()
    prev_df = filtered_df.copy()

    if cell_eccentricity_col and nuc_eccentricity_col:
        min_cell_eccentricity_threshold, max_cell_eccentricity_threshold = (
            get_min_max_percentile_thesholds(
                reference_df,
                cell_eccentricity_col,
                min_percentile=cell_eccentricity_min_threshold_percentile,
                max_percentile=cell_eccentricity_max_threshold_percentile,
                subset_col="AllGroups",
                min_subset_group="P6-12",
                max_subset_group="Doxo",
            )
        )
        min_nuc_eccentricity_threshold, max_nuc_eccentricity_threshold = (
            get_min_max_percentile_thesholds(
                reference_df,
                nuc_eccentricity_col,
                min_percentile=nuc_eccentricity_min_threshold_percentile,
                max_percentile=nuc_eccentricity_max_threshold_percentile,
                subset_col="AllGroups",
                min_subset_group="P6-12",
                max_subset_group="Doxo",
            )
        )
        filtered_df, min_cell_eccentricity_removed, max_cell_eccentricity_removed = (
            hard_size_shape_filter_rows_in_df(
                filtered_df,
                area_col=cell_eccentricity_col,
                min_threshold=min_cell_eccentricity_threshold,
                max_threshold=max_cell_eccentricity_threshold,
            )
        )
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            cell_eccentricity_col + "_Filter_Bottom",
            "Cell Eccentricity Filter Bottom Band",
            filtered_df,
            prev_df,
            f"Bottom threshold: {min_cell_eccentricity_threshold}",
            rows_removed=min_cell_eccentricity_removed,
        )
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            cell_eccentricity_col + "_Filter_Top",
            "Cell Eccentricity Filter",
            filtered_df,
            prev_df,
            f"Top threshold: {max_cell_eccentricity_threshold}",
            rows_removed=max_cell_eccentricity_removed,
        )
        filtered_df = prev_df.copy()
        prev_df = filtered_df.copy()

        filtered_df, min_nuc_eccentricity_removed, max_nuc_eccentricity_removed = (
            hard_size_shape_filter_rows_in_df(
                filtered_df,
                area_col=nuc_eccentricity_col,
                min_threshold=min_nuc_eccentricity_threshold,
                max_threshold=max_nuc_eccentricity_threshold,
            )
        )
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            nuc_eccentricity_col + "_Filter_Bottom",
            "Nuclear Eccentricity Filter Bottom Band",
            filtered_df,
            prev_df,
            f"Bottom threshold: {min_nuc_eccentricity_threshold}",
            rows_removed=min_nuc_eccentricity_removed,
        )
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            nuc_eccentricity_col + "_Filter_Top",
            "Nuclear Eccentricity Filter Top Band",
            filtered_df,
            prev_df,
            f"Top threshold: {max_nuc_eccentricity_threshold}",
            rows_removed=max_nuc_eccentricity_removed,
        )
        filtered_df = prev_df.copy()
        prev_df = filtered_df.copy()

    # else:
    #     # if not using this filter, just set the min and max intensity filtered dfs to be the same as the previous filter so that the rest of the filters can run without error
    #     filter_df_cell_eccentricity = filter_df_maxratio.copy()
    #     filter_df_nuc_eccentricity = filter_df_maxratio.copy()
    #     min_cell_eccentricity_threshold = None
    #     max_cell_eccentricity_threshold = None
    #     min_nuc_eccentricity_threshold = None
    #     max_nuc_eccentricity_threshold = None

    # nuc intensity filter
    if norm_nuc_intensity_col:
        if min_nuc_intensity_percentile == "" or max_nuc_intensity_percentile == "":
            min_nuc_intensity_threshold = filtered_df[norm_nuc_intensity_col].min()
            max_nuc_intensity_threshold = max_nuc_intensity_hardthreshold
        else:
            min_nuc_intensity_threshold = 0
            max_nuc_intensity_threshold = max_nuc_intensity_hardthreshold
        print(f"Max {norm_nuc_intensity_col} threshold: {max_nuc_intensity_threshold}")
        # filter the bottom and add to the summary dict
        filtered_df = filtered_df[
            filtered_df[norm_nuc_intensity_col] > min_nuc_intensity_threshold
        ]
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            norm_nuc_intensity_col + "_Filter_Bottom",
            norm_nuc_intensity_col.replace("_", " ") + " Filter Bottom Band",
            filtered_df,
            prev_df,
            f"Bottom threshold: {min_nuc_intensity_threshold}",
            # rows_removed=len(filtered_df) - len(prev_df),
        )
        # now filter the top and add to the summary dict
        filtered_df = filtered_df[
            filtered_df[norm_nuc_intensity_col] < max_nuc_intensity_threshold
        ]
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            norm_nuc_intensity_col + "_Filter_Top",
            norm_nuc_intensity_col.replace("_", " ") + " Filter Top Band",
            filtered_df,
            prev_df,
            f"Top threshold: {max_nuc_intensity_threshold}",
            # rows_removed=len(filtered_df) - len(prev_df),
        )
        filtered_df = prev_df.copy()
        prev_df = filtered_df.copy()
    else:
        filtered_df = prev_df.copy()
        prev_df = filtered_df.copy()

    # more than one cell filter (not using)
    if min_cell_count > 0:
        filtered_df = filtered_df[filtered_df[image_count_col] > min_cell_count]
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            image_count_col + "_Filter",
            "Min Cells Per Image Filter",
            filtered_df,
            prev_df,
            f"Min threshold: {min_cell_count}",
        )
        filtered_df = prev_df.copy()
        prev_df = filtered_df.copy()

    # neighbors filters
    max_threshold_neighbours_percenttouching = get_min_max_percentile_thesholds(
        reference_df,
        neighbours_col_percenttouching,
        min_percentile=0,
        max_percentile=max_neighbours_percenttouching_percentile,
        subset_col="AllGroups",
        max_subset_group="P6-12",
    )[1]
    max_threshold_neighbours_numberof = get_min_max_percentile_thesholds(
        reference_df,
        neighbours_col_numberof,
        min_percentile=0,
        max_percentile=max_neighbours_numberof_percentile,
        subset_col="AllGroups",
        max_subset_group="P6-12",
    )[1]
    if (
        max_threshold_neighbours_percenttouching == 0
        or max_threshold_neighbours_numberof == 0
    ):
        print("One or more max neighbor thresholds are 0. Trying with closest distance")
        # this filter would be the inverse to remove cells with too many neigbours
        min_neighbour_distance_theshold = get_min_max_percentile_thesholds(
            reference_df[
                reference_df["Neighbors_FirstClosestDistance_5"] > 0
            ],  # use only non-zero distances to avoid biasing the threshold by the zero distances
            "Neighbors_FirstClosestDistance_5",
            min_percentile=0.025,
            max_percentile=1,
        )[0]
        filtered_df = filtered_df[
            (
                filtered_df["Neighbors_FirstClosestDistance_5"]
                > min_neighbour_distance_theshold
            )
            | (filtered_df["Neighbors_FirstClosestDistance_5"] == 0)
        ]  # optional filter for touching neighbors AND number of neighbors
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            "Cell_Neighbours_Filter",
            "First Closest Neighbour distance < 0.025th percentile (97.5th closest distance) or == 0 (no neighbours)",
            filtered_df,
            prev_df,
            f"Min closest neighbour distance: {min_neighbour_distance_theshold}",
        )
    else:
        filtered_df = filtered_df[
            (
                filtered_df[neighbours_col_percenttouching]
                < max_threshold_neighbours_percenttouching
            )
            & (filtered_df[neighbours_col_numberof] < max_threshold_neighbours_numberof)
        ]  # optional filter for touching neighbors AND number of neighbors
        filtering_summary_dict = add_filtering_summary_to_dict(
            filtering_summary_dict,
            "Cell_Neighbours_Filter",
            "Percent of Cell Touching Neighbours AND Number Of Neighbours < 97.5th Percentile",
            filtered_df,
            prev_df,
            f"Max Percent Touching: {max_threshold_neighbours_percenttouching} and Max Number Of Neighbours: {max_threshold_neighbours_numberof}",
        )
    filtered_df = prev_df.copy()
    prev_df = filtered_df.copy()

    # also do the individual filters for neighbouring cells (not just the combined filter)
    filtered_df = filtered_df[
        (
            filtered_df[neighbours_col_percenttouching]
            < max_threshold_neighbours_percenttouching
        )
    ]  # optional filter for touching neighbors AND number of neighbors
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        "Cell_Neighbours_PercentTouchingOnlyFilter",
        "Percent of Cell Touching Neighbours < 97.5th Percentile",
        filtered_df,
        prev_df,
        f"Max Percent Touching: {max_threshold_neighbours_percenttouching}",
    )
    filtered_df = prev_df.copy()
    prev_df = filtered_df.copy()

    filtered_df = filtered_df[
        (filtered_df[neighbours_col_numberof] < max_threshold_neighbours_numberof)
    ]  # optional filter for touching neighbors AND number of neighbors
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        "Cell_Neighbours_NumberOfOnlyFilter",
        "Number Of Neighbours < 97.5th Percentile",
        filtered_df,
        prev_df,
        f"Max Number Of Neighbours: {max_threshold_neighbours_numberof}",
    )
    filtered_df = prev_df.copy()
    prev_df = filtered_df.copy()

    max_threshold_nuc_neighbours_numberof = get_min_max_percentile_thesholds(
        reference_df,
        nuc_neighbours_col_numberof,
        min_percentile=0,
        max_percentile=max_nuc_neighbours_numberof_percentile,
        subset_col="AllGroups",
        max_subset_group="P6-12",
    )[1]
    filtered_df = filtered_df[
        filtered_df[nuc_neighbours_col_numberof]
        <= max_threshold_nuc_neighbours_numberof
    ]  # optional filter for number of neighbouring nuclei
    filtering_summary_dict = add_filtering_summary_to_dict(
        filtering_summary_dict,
        "Nuc_Neighbours_Filter",
        "Number Of Nuclei Neighbours < 97.5th Percentile",
        filtered_df,
        prev_df,
        f"Max Number Of Nuclei Neighbours: {max_threshold_nuc_neighbours_numberof}",
    )
    filtered_df = prev_df.copy()
    prev_df = filtered_df.copy()

    # return the df and dict
    return filtered_df, filtering_summary_dict


def get_filtered_df_and_export_summary_to_csv(
    combined_df,
    groups=None,
    reference_df=None,
    group_col="Plate_Number",
    savepath="",
    prefix="",
    individual_filters=False,
):

    if reference_df is None:
        reference_df = combined_df.copy()

    if groups is not None:
        # get a per-group breakdown of the filtering summary to see if some groups are more affected by the filters than others
        # loop through the groups and apply the filters to each group separately and then combine the summary dicts into a summary df with a column for the group number
        summary_dfs = []
        group_dfs = []
        for group in groups:
            group_df = combined_df[combined_df[group_col] == group]
            if individual_filters:
                group_filtered_df, group_filtering_summary = apply_filters_individually(
                    group_df,
                    reference_df=reference_df,  # , norm_nuc_intensity_cols=norm_intensity_cols
                    # cell_eccentricity_col="AreaShape_Eccentricity",
                    # nuc_eccentricity_col="Nuclei_AreaShape_Eccentricity",
                )
                prefix = "individual_filters_"
            else:
                group_filtered_df, group_filtering_summary = apply_all_filters(
                    group_df,
                    reference_df=reference_df,  # , norm_nuc_intensity_cols=norm_intensity_cols
                    # cell_eccentricity_col="AreaShape_Eccentricity",
                    # nuc_eccentricity_col="Nuclei_AreaShape_Eccentricity",
                )

            group_summary_df = pd.DataFrame.from_dict(
                group_filtering_summary, orient="index"
            )
            if group_col == "Plate_Number":
                group_summary_df[group_col] = (
                    f"Plate {int(group)}"  # add the group number back to the summary df
                )
            else:
                group_summary_df[group_col] = (
                    group  # add the group number back to the summary df
                )
            summary_dfs.append(group_summary_df)
            group_dfs.append(group_filtered_df)

        final_filtered_df = pd.concat(group_dfs, axis=0)

        final_filtered_summary_df = pd.concat(summary_dfs, axis=0)

        if group_col == "AllGroups":
            desired_order = get_all_group_order()
        elif group_col == "Plate_Number":
            desired_order = [f"Plate {int(g)}" for g in sorted(groups)]
        else:
            desired_order = sorted(groups)

        final_filtered_summary_pivot = final_filtered_summary_df.pivot_table(
            index="filter_type",
            columns=group_col,
            values="rows_removed",
            aggfunc="first",
            fill_value=0,
        ).reindex(columns=desired_order, fill_value=0)

        final_filtered_summary_df.to_csv(
            Path(savepath, f"{prefix}filtering_summary_all_{group_col}.csv")
        )
        final_filtered_summary_pivot.to_csv(
            Path(savepath, f"{prefix}filtering_pivot_summary_all_{group_col}.csv")
        )
    else:
        final_filtered_df = combined_df.copy()
        # get the overall filtering summary for the whole dataset
        if individual_filters:
            final_filtered_df, filtering_summary = apply_filters_individually(
                final_filtered_df,
                reference_df=reference_df,  # , norm_nuc_intensity_cols=norm_intensity_cols
                # cell_eccentricity_col="AreaShape_Eccentricity",
                # nuc_eccentricity_col="Nuclei_AreaShape_Eccentricity",
            )
            prefix = "individual_filters_"
        else:
            final_filtered_df, filtering_summary = apply_all_filters(
                final_filtered_df,
                reference_df=reference_df,
                # norm_nuc_intensity_cols=norm_intensity_cols,
                # cell_eccentricity_col="AreaShape_Eccentricity",
                # nuc_eccentricity_col="Nuclei_AreaShape_Eccentricity",
            )
        final_filtered_summary_df = pd.DataFrame.from_dict(
            filtering_summary, orient="index"
        )
        final_filtered_summary_df.to_csv(
            Path(savepath, f"{prefix}filtering_summary.csv")
        )
        # filtered_df.to_csv(f"{savepath}{prefix}filtered_cell_data.csv", index=False)
        # summary_df = pd.DataFrame.from_dict(filtering_summary_dict, orient="index")
        # summary_df.to_csv(f"{prefix}filtering_summary.csv", index=False)
    return final_filtered_df, final_filtered_summary_df


def get_per_group_per_plate_and_export_summary_to_csv(
    combined_df,
    groups=None,
    plates=None,
    reference_df=None,
    plate_col="Plate_Number",
    group_col="AllGroups",
    savepath="",
    prefix="individual_filters_",
):

    if reference_df is None:
        reference_df = combined_df.copy()

    if groups is not None and plates is not None:
        # get a per-group breakdown per plate of the filtering summary to see if some groups are more affected by the filters than others
        # loop through the groups and apply the filters to each group separately and then combine the summary dicts into a summary df with a column for the group number
        summary_dfs = []
        group_dfs = []
        for group in groups:
            group_df = combined_df[combined_df[group_col] == group]
            for plate in plates:
                group_plate_df = group_df[group_df[plate_col] == plate]
                group_filtered_df, group_filtering_summary = apply_filters_individually(
                    group_plate_df,
                    reference_df=reference_df,  # , norm_nuc_intensity_cols=norm_intensity_cols
                    # cell_eccentricity_col="AreaShape_Eccentricity",
                    # nuc_eccentricity_col="Nuclei_AreaShape_Eccentricity",
                )
                group_summary_df = pd.DataFrame.from_dict(
                    group_filtering_summary, orient="index"
                )

                group_summary_df[plate_col] = (
                    f"Plate {int(plate)}"  # add the plate number back to the summary df
                )
                group_summary_df[group_col] = (
                    group  # add the group number back to the summary df
                )
                summary_dfs.append(group_summary_df)
                group_dfs.append(group_filtered_df)

        final_filtered_df = pd.concat(group_dfs, axis=0)
        final_filtered_summary_df = pd.concat(summary_dfs, axis=0)

        if group_col == "AllGroups":
            desired_order = get_all_group_order()
        elif group_col == "Plate_Number":
            desired_order = [f"Plate {int(g)}" for g in sorted(groups)]
        else:
            desired_order = sorted(groups)

        final_filtered_summary_pivot = final_filtered_summary_df.pivot_table(
            index=["filter_type"],
            columns=[plate_col, group_col],
            values=["rows_removed"],
            aggfunc="sum",
            fill_value=0,
        )  # .reindex(desired_order, axis=1, level=1, fill_value="missing") #note level means the level of the multindex
        final_filtered_summary_pivot_v2 = (
            final_filtered_summary_df.pivot_table(
                columns=["filter_type"],
                index=[plate_col, group_col],
                values=["rows_removed"],
                aggfunc="sum",
                fill_value=0,
            )
            .sort_index(axis=1, level=[0, 1], sort_remaining=False)
            .reindex(index=desired_order, level=1, fill_value="missing")
        )
        display(final_filtered_summary_df)
        display(final_filtered_summary_pivot)
        display(final_filtered_summary_pivot_v2)
        final_filtered_summary_df.to_csv(
            Path(savepath, f"{prefix}filtering_summary_{group_col}_per_plate.csv")
        )
        final_filtered_summary_pivot.to_csv(
            Path(savepath, f"{prefix}filtering_pivot_summary_{group_col}_per_plate.csv")
        )
        final_filtered_summary_pivot_v2.to_csv(
            Path(
                savepath,
                f"{prefix}filtering_pivot_summary_v2_{group_col}_per_plate.csv",
            )
        )
    else:
        final_filtered_df = combined_df.copy()
        # get the overall filtering summary for the whole dataset
        final_filtered_df, filtering_summary = apply_filters_individually(
            final_filtered_df,
            reference_df=reference_df,
            # norm_nuc_intensity_cols=norm_intensity_cols,
            # cell_eccentricity_col="AreaShape_Eccentricity",
            # nuc_eccentricity_col="Nuclei_AreaShape_Eccentricity",
        )
        final_filtered_summary_df = pd.DataFrame.from_dict(
            filtering_summary, orient="index"
        )
        final_filtered_summary_df = final_filtered_summary_df.reset_index().rename(
            columns={"index": "filter_type"}
        )
        final_filtered_summary_df["Percent_Removed"] = (
            final_filtered_summary_df["rows_removed"]
            / final_filtered_summary_df["rows_initial"]
            * 100
        )
        final_filtered_summary_df.to_csv(
            Path(savepath, f"{prefix}filtering_summary_individually.csv")
        )
        # filtered_df.to_csv(f"{savepath}{prefix}filtered_cell_data.csv", index=False)
        # summary_df = pd.DataFrame.from_dict(filtering_summary_dict, orient="index")
        # summary_df.to_csv(f"{prefix}filtering_summary.csv", index=False)
    return final_filtered_df, final_filtered_summary_df


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


def mean_intensity_per_compartment_per_cell(df, compartment, name, tag, math=None):
    # Calculate the mean intensity of each compartment per cell
    # mean_intesity_per_compartment = integrated / (children*mean_area)
    colname = f"Mean_Intensity_Per_{compartment} Per_Cell"
    integrated = "Intensity_IntegratedIntensity_" + tag
    # children = 'Children_' + compartment + '_Count'
    # mean_area = 'Mean_'+ compartment + '_AreaShape_Area'
    total_organelle_area = name + "_AreaShape_Area"
    total_organelle_area = math if math is not None else total_organelle_area

    df[colname] = df.apply(lambda x: x[integrated] / x[total_organelle_area], axis=1)

    return df[colname]


def calculate_corrected_features(full_df):
    """_summary_

    Args:
        full_df (DataFrame): _description_

    Returns:
        df (DataFrame): the df with all the feature calcs
    """
    df = full_df.copy()
    df["Math_Total_Mitochondria_AreaShape_Area_PerCell"] = (
        df["Children_Mitochondria_Count"] * df["Mean_Mitochondria_AreaShape_Area"]
    )
    df["Math_Total_Lysosomes_AreaShape_Area_PerCell"] = (
        df["Children_Lysosomes_Count"] * df["Mean_Lysosomes_AreaShape_Area"]
    )
    df["Math_Total_Mitochondria_Puncta_AreaShape_Area_PerCell"] = (
        df["Children_Mitochondria_Puncta_Count"]
        * df["Mean_Mitochondria_Puncta_AreaShape_Area"]
    )

    # Total intensity per cell based on integrated instensity if I don't already have the merged area
    df["Mean_Intensity_Per_Lysosomes_PerCell_Area"] = (
        mean_intensity_per_compartment_per_cell(
            df,
            "Lysosomes",
            "MergedLysoPerCell",
            "LAMP1",
            math="Math_Total_Lysosomes_AreaShape_Area_PerCell",
        )
    )
    df["Mean_Intensity_Per_Mitochondria_PerCell_Area"] = (
        mean_intensity_per_compartment_per_cell(
            df,
            "Mitochondria",
            "MergedMitoPerCell",
            "MitoTracker",
            math="Math_Total_Mitochondria_AreaShape_Area_PerCell",
        )
    )
    df["Mean_Intensity_Per_Mitochondria_Puncta_PerCell_Area"] = (
        mean_intensity_per_compartment_per_cell(
            df,
            "Mitochondria_Puncta",
            "MergedMitoPunctaPerCell",
            "MitoTracker",
            math="Math_Total_Mitochondria_Puncta_AreaShape_Area_PerCell",
        )
    )
    # #same thing but for medians
    # df["Median_Intensity_Per_Lysosomes_PerCell_Area"] = (
    #     df["Children_Lysosomes_Count"]
    #     * df["Mean_Lysosomes_Intensity_MeanIntensity_LAMP1"]
    # )
    # df["Median_Intensity_Per_Mitochondria_PerCell_Area"] = (
    #     df["Children_Mitochondria_Count"]
    #     * df["Mean_Mitochondria_Intensity_MeanIntensity_MitoTracker"]
    # )

    # Corrected mitochondria and lysosomes counts per cell area (density)
    df["Density_Children_Mitochondria_Count_PerCell_Area"] = (
        df["Children_Mitochondria_Count"] / df["AreaShape_Area"]
    )
    df["Density_Children_Lysosomes_Count_PerCell_Area"] = (
        df["Children_Lysosomes_Count"] / df["AreaShape_Area"]
    )
    df["Density_Children_Mitochondria_Puncta_Count_PerCell_Area"] = (
        df["Children_Mitochondria_Puncta_Count"] / df["AreaShape_Area"]
    )

    # organelle area fractions per cell ratio
    df["OccupiedAreaFraction_Mitochondria_PerCell_Area"] = (
        df["Math_Total_Mitochondria_AreaShape_Area_PerCell"] / df["AreaShape_Area"]
    )
    df["OccupiedAreaFraction_Mitochondria_Puncta_PerCell_Area"] = (
        df["Math_Total_Mitochondria_Puncta_AreaShape_Area_PerCell"]
        / df["AreaShape_Area"]
    )
    df["OccupiedAreaFraction_Lysosomes_PerCell_Area"] = (
        df["Math_Total_Lysosomes_AreaShape_Area_PerCell"] / df["AreaShape_Area"]
    )

    # Compartment diameter ratios
    df["Mean_Lysosomes_MaxMinFeret_DiameterRatio_PerCell"] = (
        df["Mean_Lysosomes_AreaShape_MaxFeretDiameter"]
        / df["Mean_Lysosomes_AreaShape_MinFeretDiameter"]
    )
    df["Mean_Mitochondria_MaxMinFeret_DiameterRatio_PerCell"] = (
        df["Mean_Mitochondria_AreaShape_MaxFeretDiameter"]
        / df["Mean_Mitochondria_AreaShape_MinFeretDiameter"]
    )
    df["Median_Mitochondria_DiameterRatio_PerCell"] = (
        df["Median_Mitochondria_AreaShape_MaxFeretDiameter"]
        / df["Median_Mitochondria_AreaShape_MinFeretDiameter"]
    )
    df["Median_Lysosomes_DiameterRatio_PerCell"] = (
        df["Median_Lysosomes_AreaShape_MaxFeretDiameter"]
        / df["Median_Lysosomes_AreaShape_MinFeretDiameter"]
    )
    df["Mean_Mitochondria_Puncta_MaxMinFeret_DiameterRatio_PerCell"] = (
        df["Mean_Mitochondria_Puncta_AreaShape_MaxFeretDiameter"]
        / df["Mean_Mitochondria_Puncta_AreaShape_MinFeretDiameter"]
    )
    # df["Median_Mitochondria_Puncta_DiameterRatio_PerCell"] = (
    #     df["Median_Mitochondria_Puncta_AreaShape_MaxFeretDiameter"]
    #     / df["Median_Mitochondria_Puncta_AreaShape_MinFeretDiameter"]
    # )

    # mean and median area per organelle per cell area
    df["Mean_Mitochondria_Area_PerCell_Area"] = (
        df["Mean_Mitochondria_AreaShape_Area"] / df["AreaShape_Area"]
    )
    df["Mean_Lysosomes_Area_PerCell_Area"] = (
        df["Mean_Lysosomes_AreaShape_Area"] / df["AreaShape_Area"]
    )
    df["Mean_Mitochondria_Puncta)Area_PerCell_Area"] = (
        df["Mean_Mitochondria_Puncta_AreaShape_Area"] / df["AreaShape_Area"]
    )
    df["Median_Mitochondria_Area_PerCell_Area"] = (
        df["Median_Mitochondria_AreaShape_Area"] / df["AreaShape_Area"]
    )
    df["Median_Lysosomes_Area_PerCell_Area"] = (
        df["Median_Lysosomes_AreaShape_Area"] / df["AreaShape_Area"]
    )

    # mitolyso related
    df["Children_Lysosomes_Mitochondria_Ratio"] = (
        df["Children_Lysosomes_Count"] / df["Children_Mitochondria_Count"]
    )
    df["Density_Lysosomes_Mitochondria_Ratio"] = (
        df["OccupiedAreaFraction_Lysosomes_PerCell_Area"]
        / df["OccupiedAreaFraction_Mitochondria_PerCell_Area"]
    )
    df["Area_Lysosomes_Mitochondria_Ratio"] = (
        df["Math_Total_Lysosomes_AreaShape_Area_PerCell"]
        / df["Math_Total_Mitochondria_AreaShape_Area_PerCell"]
    )

    # Quin's Ratio: Ratio of centroid distance to minimum distance for mitochondria and lysosomes
    # increase = more peripheral, decrease = more nuclear
    df["Mean_Mitochondria_Distance_Centroid_Nuclei_Minimum_Nuclei_QuinRatio"] = (
        df["Mean_Mitochondria_Distance_Centroid_Nuclei"]
        / df["Mean_Mitochondria_Distance_Minimum_Nuclei"]
    )
    df["Mean_Lysosomes_Distance_Centroid_Nuclei_Minimum_Nuclei_QuinRatio"] = (
        df["Mean_Lysosomes_Distance_Centroid_Nuclei"]
        / df["Mean_Lysosomes_Distance_Minimum_Nuclei"]
    )
    df["Mean_Mitochondria_Distance_Centroid_Nuclei_Minimum_Cell_QuinRatio"] = (
        df["Mean_Mitochondria_Distance_Centroid_Nuclei"]
        / df["Mean_Mitochondria_Distance_Minimum_Cell"]
    )
    df["Mean_Lysosomes_Distance_Centroid_Nuclei_Minimum_Cell_QuinRatio"] = (
        df["Mean_Lysosomes_Distance_Centroid_Nuclei"]
        / df["Mean_Lysosomes_Distance_Minimum_Cell"]
    )

    df["Mean_Mitochondria_Puncta_Distance_Centroid_Nuclei_Minimum_Nuclei_QuinRatio"] = (
        df["Mean_Mitochondria_Puncta_Distance_Centroid_Nuclei"]
        / df["Mean_Mitochondria_Puncta_Distance_Minimum_Nuclei"]
    )
    df["Mean_Mitochondria_Puncta_Distance_Centroid_Nuclei_Minimum_Cell_QuinRatio"] = (
        df["Mean_Mitochondria_Puncta_Distance_Centroid_Nuclei"]
        / df["Mean_Mitochondria_Puncta_Distance_Minimum_Cell"]
    )
    # Distance to parents percellarea
    df["Mean_Lysosomes_Distance_Centroid_Nuclei_PerCell_Area"] = (
        df["Mean_Lysosomes_Distance_Centroid_Nuclei"] / df["AreaShape_Area"]
    )
    df["Mean_Mitochondria_Distance_Centroid_Nuclei_PerCell_Area"] = (
        df["Mean_Mitochondria_Distance_Centroid_Nuclei"] / df["AreaShape_Area"]
    )
    df["Mean_Lysosomes_Distance_Centroid_Cell_PerCell_Area"] = (
        df["Mean_Lysosomes_Distance_Centroid_Cell"] / df["AreaShape_Area"]
    )
    df["Mean_Mitochondria_Distance_Centroid_Cell_PerCell_Area"] = (
        df["Mean_Mitochondria_Distance_Centroid_Cell"] / df["AreaShape_Area"]
    )
    df["Mean_Lysosomes_Distance_Minimum_Nuclei_PerCell_Area"] = (
        df["Mean_Lysosomes_Distance_Minimum_Nuclei"] / df["AreaShape_Area"]
    )
    df["Mean_Mitochondria_Distance_Minimum_Nuclei_PerCell_Area"] = (
        df["Mean_Mitochondria_Distance_Minimum_Nuclei"] / df["AreaShape_Area"]
    )
    df["Mean_Lysosomes_Distance_Minimum_Cell_PerCell_Area"] = (
        df["Mean_Lysosomes_Distance_Minimum_Cell"] / df["AreaShape_Area"]
    )
    df["Mean_Mitochondria_Distance_Minimum_Cell_PerCell_Area"] = (
        df["Mean_Mitochondria_Distance_Minimum_Cell"] / df["AreaShape_Area"]
    )

    df["Mean_Mitochondria_Puncta_Distance_Centroid_Nuclei_PerCell_Area"] = (
        df["Mean_Mitochondria_Distance_Centroid_Nuclei"] / df["AreaShape_Area"]
    )
    df["Mean_Mitochondria_Puncta_Distance_Centroid_Cell_PerCell_Area"] = (
        df["Mean_Mitochondria_Puncta_Distance_Centroid_Cell"] / df["AreaShape_Area"]
    )
    df["Mean_Mitochondria_Puncta_Distance_Minimum_Nuclei_PerCell_Area"] = (
        df["Mean_Mitochondria_Puncta_Distance_Minimum_Nuclei"] / df["AreaShape_Area"]
    )
    df["Mean_Mitochondria_Puncta_Distance_Minimum_Cell_PerCell_Area"] = (
        df["Mean_Mitochondria_Puncta_Distance_Minimum_Cell"] / df["AreaShape_Area"]
    )

    # transform the mitoends - number of ends times the mean to get total per cell
    df["MitoEnds_Math_Total_NumberBranchEnds_MitoSkeleton"] = (
        df["Children_MitoEnds_Count"]
        * df["Mean_MitoEnds_ObjectSkeleton_NumberBranchEnds_MitoSkeleton"]
    )
    df["MitoEnds_Math_Total_NumberNonTrunkBranches_MitoSkeleton"] = (
        df["Children_MitoEnds_Count"]
        * df["Mean_MitoEnds_ObjectSkeleton_NumberNonTrunkBranches_MtSkltn"]
    )
    df["MitoEnds_Math_Total_NumberTrunks_MitoSkeleton"] = (
        df["Children_MitoEnds_Count"]
        * df["Mean_MitoEnds_ObjectSkeleton_NumberTrunks_MitoSkeleton"]
    )
    df["MitoEnds_Math_TotalObjectSkeltnLngth_MitoSkeleton"] = (
        df["Children_MitoEnds_Count"]
        * df["Mean_MitoEnds_ObjectSkeleton_TotalObjectSkeltnLngth_MtSkltn"]
    )
    # now divide these by cell area
    df["MitoEnds_NumberBranchEnds_PerCell_Area"] = (
        df["MitoEnds_Math_Total_NumberBranchEnds_MitoSkeleton"] / df["AreaShape_Area"]
    )
    df["MitoEnds_NumberNonTrunkBranches_PerCell_Area"] = (
        df["MitoEnds_Math_Total_NumberNonTrunkBranches_MitoSkeleton"]
        / df["AreaShape_Area"]
    )
    df["MitoEnds_NumberTrunks_PerCell_Area"] = (
        df["MitoEnds_Math_Total_NumberTrunks_MitoSkeleton"] / df["AreaShape_Area"]
    )
    df["MitoEnds_TotalObjectSkeltnLngth_PerCell_Area"] = (
        df["MitoEnds_Math_TotalObjectSkeltnLngth_MitoSkeleton"] / df["AreaShape_Area"]
    )

    # also the nuclear mito skeleton features per cell area
    df["Nuclei_ObjectSkeleton_NumberBranchEnds_PerCell_Area"] = (
        df["Nuclei_ObjectSkeleton_NumberBranchEnds_MitoSkeleton"] / df["AreaShape_Area"]
    )
    df["Nuclei_ObjectSkeleton_NumberNonTrunkBranches_PerCell_Area"] = (
        df["Nuclei_ObjectSkeleton_NumberNonTrunkBranches_MitoSkeleton"]
        / df["AreaShape_Area"]
    )
    df["Nuclei_ObjectSkeleton_NumberTrunks_PerCell_Area"] = (
        df["Nuclei_ObjectSkeleton_NumberTrunks_MitoSkeleton"] / df["AreaShape_Area"]
    )
    df["Nuclei_ObjectSkeleton_TotalObjectSkeletonLength_PerCell_Area"] = (
        df["Nuclei_ObjectSkeleton_TotalObjectSkeletonLength_MitoSkeleton"]
        / df["AreaShape_Area"]
    )

    return df


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


def flag_outliers_by_group_mad(df, group_col, feature_col):
    """
    Adds a boolean 'Outlier' column to df, True if the value is an outlier within its group.
    """
    df = df.copy()
    df["Outlier"] = df.groupby(group_col)[feature_col].transform(
        lambda x: pg.madmedianrule(x)
    )
    return df


def flag_outliers_by_group_gesd(df, group_col, feature_col, noutliers=20, report=True):
    """
    Adds a boolean 'Outlier' column to df, True if the value is an outlier within its group.
    """

    df = df.copy()
    df["Outlier_GESD"] = df.groupby(group_col)[feature_col].transform(
        lambda x: sp.outliers_gesd(x, outliers=noutliers, hypo=True, report=report)
    )
    return df


def flag_outliers_by_group_tietjen(df, group_col, feature_col, noutliers=5):
    """
    Adds a boolean 'Outlier_Grubbs' column to df, True if you reject the null hypothesis that the extreme value is an outlier.
    """

    df = df.copy()
    df["Outlier_Tietjen"] = df.groupby(group_col)[feature_col].transform(
        lambda x: sp.outliers_tietjen(x, k=noutliers, hypo=True)
    )
    return df


def remove_outliers_by_group_gesd(df, group_col, feature_col, noutliers=5):
    """
    Filters out outliers in feature_col within each group of group_col using the GEST test.
    Returns a DataFrame with outliers removed.
    """

    df = df.copy()
    filtered_groups = []
    for group_val, group_df in df.groupby(group_col):
        mask = sp.outliers_gesd(group_df[feature_col], outliers=noutliers, hypo=False)
        filtered_group = group_df[group_df[feature_col].isin(mask)]
        filtered_groups.append(filtered_group)
    final_df = pd.concat(filtered_groups, axis=0)
    return final_df


def remove_outliers_by_group_tietjen(df, group_col, feature_col, noutliers=5):
    """
    Filters out outliers in feature_col within each group of group_col using Tietjen's test.
    Returns a DataFrame with outliers removed.
    """

    df = df.copy()
    filtered_groups = []
    for group_val, group_df in df.groupby(group_col):
        mask = sp.outliers_tietjen(group_df[feature_col], k=noutliers, hypo=False)
        filtered_group = group_df[group_df[feature_col].isin(mask)]
        filtered_groups.append(filtered_group)
    final_df = pd.concat(filtered_groups, axis=0)
    return final_df


def get_mini_filtered_df(
    final_filtered_df,
    condition_col="",
    valueslist=None,
    op=operator.le,
    condition_value=10,
    group="",
    group_col="AllGroups",
    plates=[],
    plate_col="Plate_Number",
):
    if valueslist is None:
        valueslist = [
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
        ]
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
    # Filter by plate of you specify to
    if plates:
        mini_df = mini_df[mini_df[plate_col].isin(plates)]
    if group:
        mini_df = mini_df[mini_df[group_col] == group]

    # Now add the filter
    filter_mini_df = mini_df[op(mini_df[condition_col], condition_value)]
    filter_mini_df_sorted = filter_mini_df.sort_values(
        by=[group_col], key=lambda x: x.map(passage_groups_sort_key)
    ).reset_index(drop=True)

    # reduce cols for readability
    filter_mini_df_display = filter_mini_df_sorted[
        [
            group_col,
            "Plate_Number",
            "Cell_Unique_ID",
            "Number_Object_Number",
            "Image_FileName_MitoTracker_MAX",
            condition_col,
        ]
    ]

    display(filter_mini_df_display)

    return filter_mini_df_sorted


def find_plate_cp_output_folder(path):
    import re

    plate_pattern = r"_rep0(\d{1})"  # Matches "RX" where X is the plate number (placeholder for now)
    match = re.search(plate_pattern, path)
    if match:
        plate = int(match.group(1))
    else:
        plate = None
    return plate


def pull_up_cp_segmentation_image(
    img_filename, parent_dir="~/", plate=0, group="", object_key=None, df=None
):
    """Function to pull up an image with cellprofiler segmentation outlines that has a matching image in the dataframe
    Can also highlight the sepecific object with a bounding box

    Args:
        parent_dir (str, optional): _description_. Defaults to "~/".
        img_filename (str): _description_.
        plate (int, optional): _description_. Defaults to 0.
        group (str, optional): _description_. Defaults to "".
    """
    # Loop over the plates
    # make sure the filename in the format of: "Image_FileName_MitoTracker_MAX"

    img_filename_noext = img_filename.split(".")[0]
    for root, dirs, files in os.walk(parent_dir):
        for filename in files:
            if (
                img_filename_noext in filename
                and filename.endswith(".png")
                and "active" in root
                and plate == find_plate_cp_output_folder(root)
            ):
                img_path = os.path.join(
                    root, img_filename_noext + ".png"
                )  # make the path
                try:
                    img = Image.open(img_path)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    plt.imshow(img)
                    plt.axis("off")  # Turn off axis labels for a cleaner image display
                    plt.title(f"R{plate}, {filename}, {group}")

                    if object_key is not None and df is not None:
                        row = df[df["Cell_Unique_ID"] == object_key]
                        if not row.empty:
                            rect, filename_2 = get_image_and_object_coordinates(
                                df, object_key
                            )
                            ax.add_patch(rect)
                        else:
                            print(f"Object number {object_key} not found in dataframe.")
                    plt.show()
                    print(f"Segmented image url: {img_path}")
                    # img.show()

                except FileNotFoundError:
                    print(
                        f"Image file {img_path} not found. Please ensure 'your_image.png' exists."
                    )


def pull_up_cp_segmentation_image_fromID(
    df,
    object_key,
    parent_dir="",
    plate_col_name="Plate_Number",
    feature="",
    image_channel="LAMP1",
    extension=".png",
    colour=None,
    only_active=True,
    save=False,
    savepath="",
    show=True,
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
    if only_active:
        active_string = "active"
    else:
        active_string = ""
    # loop over to find the file in the directory
    img_filename_noext = img_filename.split(".")[0]
    for root, dirs, files in os.walk(parent_dir):
        for filename in files:
            if (
                img_filename_noext.split("_")[1] in filename
                and "MAX" in filename
                and filename.endswith(extension)
                and active_string in root  # active cp output folder
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
                    if colour is not None:
                        plt.imshow(img, cmap=colour)
                    else:
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
                            bbox={"facecolor": "black", "alpha": 0.5, "pad": 2},
                        )
                    if save:
                        savepath_full = f"{savepath}/{object_key}_R{plate}_P{passage}_{filename.split('.')[0]}.png"
                        plt.savefig(
                            savepath_full,
                            bbox_inches="tight",
                        )
                        print(
                            "Saved image to:",
                            savepath_full,
                        )
                    if show:
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
    coord_cols=None,
):
    if coord_cols is None:
        coord_cols = [
            "AreaShape_BoundingBoxMinimum_X",
            "AreaShape_BoundingBoxMinimum_Y",
            "AreaShape_BoundingBoxMaximum_X",
            "AreaShape_BoundingBoxMaximum_Y",
        ]
    row = df[df["Cell_Unique_ID"] == unique_ID]
    if not row.empty:
        x_min = row[coord_cols[0]].values[0]
        y_min = row[coord_cols[1]].values[0]
        x_max = row[coord_cols[2]].values[0]
        y_max = row[coord_cols[3]].values[0]
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


def annotate_cp_segmentation_image_with_feature_values(
    df,
    img_filename,
    plate=0,
    group="",
    parent_dir="~/",
    plate_col_name="Plate_Number",
    feature="",
    save=False,
    savepath="",
    decimals=2,
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
    # get the filename, plate and coords from the unique ID if the ID exists
    if "ch2" in img_filename:
        image_col_name = "Image_FileName_MitoTracker_MAX"
    elif "ch1" in img_filename:
        image_col_name = "Image_FileName_LAMP1_MAX"

    # image_rows = df[(df[image_col_name] == img_filename) & (df[plate_col_name]== plate)]
    image_rows = df.query(
        f"{image_col_name} == @img_filename and {plate_col_name} == @plate"
    )

    passage = image_rows["PassageNumber"].values[0]
    group = image_rows["AllGroups"].values[0]

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
                        f"Feature: {feature}, R{plate} P{passage}, {filename}, {group}"
                    )

                    # add a label with the feature value if it exists
                    if feature in df.columns:
                        for i in range(len(image_rows[plate_col_name])):
                            object_key = image_rows["Cell_Unique_ID"].values[i]
                            rect = get_object_bbox_coordinates_as_rectangle(
                                df, object_key
                            )
                            feature_value = image_rows[feature].values[i]
                            rect_x = rect.get_x()
                            rect_y = rect.get_y()
                            centre_x = image_rows["AreaShape_Center_X"].values[i]
                            centre_y = image_rows["AreaShape_Center_Y"].values[i]

                            plt.text(
                                centre_x,
                                centre_y,
                                f"{feature_value:.{decimals}f}",
                                color="lime",
                                fontsize=16,
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


def pull_up_multichannel_image_fromID(
    df,
    object_key,
    parent_dir="",
    plate_col_name="Plate_Number",
    feature="",
    image_channels=None,
    extension=".tif",
    colour_map=None,
    save=False,
    save_prefix="merged",
    savepath="",
    show=True,
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

    # get the filename, plate and coords from the unique ID if the ID exists
    if image_channels is None:
        image_channels = ["MitoTracker", "LAMP1", "DAPI"]
    if object_key is not None and df is not None:
        rect = get_object_bbox_coordinates_as_rectangle(df, object_key)

        unique_row = df[df["Cell_Unique_ID"] == object_key]
        img_filenames = [
            unique_row[f"Image_FileName_{channel}_MAX"].values[0]
            for channel in image_channels
        ]
        plate = unique_row[plate_col_name].values[0]
        passage = unique_row["PassageNumber"].values[0]
        group = unique_row["AllGroups"].values[0]
    else:
        print(f"Object number {object_key} not found in dataframe.")
        return False
    # loop over to find the file in the directory
    img_filename_noext = img_filenames[0].split(".")[0]
    for root, dirs, files in os.walk(parent_dir):
        for filename in files:
            if (
                img_filename_noext.split("_")[1] in filename
                and "MAX" in filename
                and filename.endswith(extension)
                and plate == find_plate_cp_output_folder(root)
            ):
                print(filename)
                img_path = os.path.join(
                    root,
                    filename,  # img_filename_noext + ".png"
                )  # make the path
                # Now we see if the image exists and try to open it
                try:
                    imgs = []
                    for img_filename in img_filenames:
                        img_path = os.path.join(
                            root,
                            img_filename,  # img_filename_noext + ".png"
                        )  # make the path
                        img = plt.imread(img_path)
                        imgs.append(img)
                    img_stack = np.stack(imgs, axis=-1)

                    print(img_stack.shape)
                    img_rgb = color.xyz2rgb(
                        img_stack, channel_axis=-1
                    )  # Merge the first three channels into an RGB image
                    fig, ax = plt.subplots(figsize=(12, 12))
                    if colour_map is not None:
                        plt.imshow(img_rgb, cmap=colour_map)
                    else:
                        plt.imshow(img_rgb)
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
                        new_filename = re.sub(r"ch\d+", save_prefix, filename)
                        save_filename = f"{savepath}/{object_key}_R{plate}_P{passage}_{new_filename.split('.')[0]}.png"
                        plt.savefig(
                            save_filename,
                            bbox_inches="tight",
                        )
                        print(f"Saved image to: {save_filename}")
                    if show:
                        plt.show()
                    print(f"Segmented image url: {img_path}")
                    # img.show()
                    return True
                except FileNotFoundError:
                    print(
                        f"Image file {img_path} not found. Please ensure 'your_image.png' exists."
                    )

    return False


def pull_up_singlechannel_image_fromID(
    df,
    object_key,
    parent_dir="",
    plate_col_name="Plate_Number",
    feature="",
    image_channel="DAPI",
    extension=".tif",
    colour_map=None,
    save=False,
    save_prefix="merged",
    savepath="",
    show=True,
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

    img_filename_noext = img_filename.split(".")[0]
    for root, dirs, files in os.walk(parent_dir):
        for filename in files:
            if (
                img_filename_noext.split("_")[1] in filename
                and "MAX" in filename
                and filename.endswith(extension)
                and plate == find_plate_cp_output_folder(root)
            ):
                img_path = os.path.join(
                    root, img_filename
                )  # img_filename_noext + ".png")  # make the path
                img = plt.imread(img_path)
                print(filename)
                # Now we see if the image exists and try to open it
                fig, ax = plt.subplots(figsize=(12, 12))
                if colour_map is not None:
                    plt.imshow(img, cmap=colour_map)
                else:
                    plt.imshow(img)
                plt.axis("off")  # Turn off axis labels for a cleaner image display
                plt.title(f"ID:{object_key}, R{plate} P{passage}, {filename}, {group}")
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
                        bbox={"facecolor": "black", "alpha": 0.5, "pad": 2},
                    )
                if save:
                    new_filename = re.sub(r"ch\d+", save_prefix, filename)
                    save_filename = f"{savepath}/{object_key}_R{plate}_P{passage}_{new_filename.split('.')[0]}.png"
                    plt.savefig(
                        save_filename,
                        bbox_inches="tight",
                    )
                    print(f"Saved image to: {save_filename}")
                if show:
                    plt.show()
                print(f"Segmented image url: {img_path}")
                # img.show()
                return True

    return False


def get_object_keys_from_filenames(path, target_folder_name):
    """Get the unique object keys from the dataframe that match a given filename in a directory"""
    object_keys = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if target_folder_name in root and file.endswith(".png"):
                # Extract the unique object key from the filename
                # Assuming the filename format is something like "R5.0_P11_MAX_ch1-r06c10f21.png_51559.png"
                key_part = file.split("_")[-1].split(".")[
                    0
                ]  # Get the last part before .png
                try:
                    object_key = int(key_part)  # Convert to integer
                    if object_key not in object_keys:  # Avoid duplicates
                        object_keys.append(object_key)
                except ValueError:
                    print(f"Could not convert {key_part} to an integer.")

    return object_keys


def ridge_label(x, color, label):
    ax = plt.gca()
    ax.text(
        -0.1,
        -0.2,
        label,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )


def seaborn_ridgeplot(
    df,
    value_col,
    group_col,
    palette=None,
    bw_adjust=1,
    xlabel=None,
    xlim=(None, None),
    title=None,
    fill_alpha=1,
    linewidth=1.5,
    figsize=(20, 30),
    save=True,
    out_dir="",
    show_percentiles=True,
    truncate_outliers=True,
    norm=False,
):
    """
    Make a ridgeline (joyplot) using seaborn FacetGrid and kdeplot
    Args:
        df: DataFrame
        value_col: str, column with numeric values
        group_col: str, column with group/category
        palette: seaborn palette or list/dict of colors
        bw_adjust: float, KDE bandwidth adjust
        xlabel: str or None
        title: str or None
        fill_alpha: float, alpha for fill
        linewidth: float, line width for outline
        figsize: tuple, figure size
    """

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    unique_groups = df[group_col].unique()
    if palette is None:
        palette = sns.cubehelix_palette(len(unique_groups), rot=-0.25, light=0.7)
    else:
        palette = palette

    if truncate_outliers and xlim == (None, None):
        try:
            if norm:
                top_fence = df[value_col].mean() + 10 * df[value_col].std()
                bottom_fence = None  # np.percentile(data_df[y_value], 0.000001)
            else:
                top_fence = np.percentile(df[value_col], 99.9)
                bottom_fence = np.percentile(df[value_col], 0.01)
                # handle errors where the data is very skewed and the percentile is inf or nan
                if top_fence == 0 or np.isnan(top_fence) or np.isinf(top_fence):
                    top_fence = None
                if (
                    bottom_fence == 0
                    or np.isnan(bottom_fence)
                    or np.isinf(bottom_fence)
                ):
                    bottom_fence = None
            xlim = (bottom_fence, top_fence)
        except ValueError as e:
            print(e)
            top_fence = None
            bottom_fence = None
            xlim = (None, None)
        print(f"Truncating outliers at: {xlim}")
        xmin, xmax = xlim
        plot_df = df.copy()
        if xmin is not None:
            plot_df = plot_df[plot_df[value_col] >= xmin]
        if xmax is not None:
            plot_df = plot_df[plot_df[value_col] <= xmax]
    else:
        plot_df = df.copy()
    # Initialize the FacetGrid object
    g = sns.FacetGrid(
        plot_df,
        row=group_col,
        hue=group_col,
        aspect=15,
        height=0.5,
        palette=palette,
        xlim=xlim,
    )
    # Draw the densities in a few steps
    g.map(
        sns.kdeplot,
        value_col,
        bw_adjust=bw_adjust,
        clip_on=False,
        fill=True,
        alpha=fill_alpha,
        linewidth=linewidth,
    )
    g.map(sns.kdeplot, value_col, clip_on=False, color="w", lw=2, bw_adjust=bw_adjust)

    if show_percentiles:
        percentiles = [5, 12.5, 25, 50, 75, 87.5, 95]
        for ax in g.axes.flatten():
            # group label text (FacetGrid puts "group_col = <value>" in the title)
            title_text = ax.get_title()
            if " = " in title_text:
                group_val = title_text.split(" = ", 1)[1]
            else:
                group_val = title_text

            group_data = df[df[group_col] == group_val][value_col].dropna()
            if group_data.empty:
                continue

            # pick the most representative line on the axis (the KDE line)
            lines = ax.get_lines()
            if not lines:
                continue
            # choose the line with the largest x-range (robust against multiple lines)
            kde_line = max(lines, key=lambda l: np.ptp(l.get_xdata()))
            xs = kde_line.get_xdata()
            ys = kde_line.get_ydata()
            # line_colour = kde_line.get
            # compute median and interpolate its KDE height
            median = np.median(group_data)
            height = np.interp(median, xs, ys, left=0.0, right=0.0)

            # draw a solid thicker median line from y=0 up to the KDE height
            ax.vlines(
                median, 0, height, color="black", linewidth=3, linestyle=":", zorder=4
            )
            ax.set_xlim(xlim)
            # draw lighter dashed lines for the percentiles (optional)
            group_percentiles = np.percentile(group_data, percentiles)
            for p in group_percentiles:
                ax.vlines(p, 0, height, color="dimgray", ls=":", alpha=0.6, linewidth=2)
    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    g.map(ridge_label, value_col)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)
    g.figure.set_size_inches(figsize)
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlim=xlim)
    g.despine(bottom=True, left=True)
    if xlabel:
        plt.xlabel(xlabel, fontweight="bold", fontsize=14)
    else:
        plt.xlabel(value_col, fontweight="bold", fontsize=14)
    if title:
        g.figure.suptitle(title, ha="right", fontsize=18, fontweight="bold")
    plt.xlim(xlim)
    plt.tight_layout()
    if save:
        plt.savefig(f"{Path(out_dir, f'{value_col}_{group_col}_joyplot')}.png")
    plt.show()


def plotly_histogram(df, y_value, group_var, save=False, out_dir=""):

    df_sorted = df.sort_values(
        by=[group_var], key=lambda x: x.map(passage_groups_sort_key)
    ).reset_index(drop=True)
    hist2 = px.histogram(
        df_sorted,
        x=y_value,
        color=group_var,
        marginal="box",
        # histnorm='probability density',
        # range_x=(0, top_fence),
    )
    hist2.write_image(Path(out_dir, f"{y_value}_histogram.png"), scale=1.5)
    hist2.show()


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
    inculded_percentiles=None,
):
    from pathlib import Path

    if inculded_percentiles is None:
        inculded_percentiles = [
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            0.95,
            0.975,
            0.99,
        ]
    try:
        table_csvname = f"{df_tag}_total_combined_stats.csv"
        feature_csvname = f"{df_tag}_{feature_name}_by_{group_name}_stats.csv"
        agg_feature_csvname = f"{df_tag}_agg_{feature_name}_by_{group_name}_stats.csv"

        subfolder_name = f"{df_tag}_{feature_name}_summary_stats"
        parent_folder = Path(summary_outpath, subfolder_name)
        parent_folder.mkdir(exist_ok=True)

        if not include_cols:
            df_to_summarize = df
        else:
            df_to_summarize = df[include_cols]
        df_to_summarize.describe(percentiles=inculded_percentiles).to_csv(
            os.path.join(summary_outpath, table_csvname)
        )
        group_averages = df.groupby(
            [x_value, plate_col_name], as_index=False, observed=True
        )[feature]
        # Reset the index to get a clean DataFrame
        # average_df = group_averages.reset_index()
        avg_summary = group_averages.describe(percentiles=inculded_percentiles)
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
        avg_agg_summary = group_averages_agg.describe(percentiles=inculded_percentiles)
        avg_agg_summary_sorted = avg_agg_summary.sort_values(
            by=[x_value], key=lambda x: x.map(passage_groups_sort_key)
        ).reset_index(drop=True)
        avg_agg_summary_sorted.T.to_csv(
            os.path.join(summary_outpath, subfolder_name, agg_feature_csvname)
        )
        print(
            f"saved files {(table_csvname, feature_csvname, agg_feature_csvname)} to {summary_outpath}"
        )
        return True
    except ValueError as e:
        print(f"Could not make summary stats: {e}")
        return False
