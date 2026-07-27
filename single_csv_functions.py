import os
import pandas as pd
import numpy as np
from mitolyso_plot_functions import *
from plate_preprocessing import *
from plate_information import *


def merge_csvs_mitolyso(
    cell_df_mitolyso,
    csvpath,
    csvname,
    image_number_col="ImageNumber",
    object_number_col="ObjectNumber",
):
    """Merge two cellprofiler csv files based on ImageNumber and ObjectNumber columns. CSV files MUST be the same number of rows and be in a 1:1 relationship.
    Args:
        cell_df_mitolyso (pd.DataFrame): the input dataframe to merge with the new csv file.
        csvpath (str): The path to the directory containing the csv files.
        csvname (str): The name of the csv file to merge.
        image_number_col (str, optional): The column name for the image number. Defaults to "ImageNumber".
        object_number_col (str, optional): The column name for the object number. Defaults to "ObjectNumber".

    Returns:
        pd.DataFrame: The dataframe with the merged data.
    """
    new_df = pd.read_csv(os.path.join(csvpath, csvname))
    prefix = csvname.split(".")[0].strip()

    for col in cell_df_mitolyso.columns:
        # remove columns that are identical in both dataframes, except for the image and object number keys
        if (
            col in new_df.columns
            and new_df[col].equals(cell_df_mitolyso[col])
            and col not in {image_number_col, object_number_col}
        ):
            new_df = new_df.drop(col, axis=1)

    new_df = new_df.add_prefix(prefix + "_")
    if prefix == "Nuclei":
        cell_df_mitolyso = pd.merge(
            cell_df_mitolyso,
            new_df,
            left_on=[image_number_col, f"Parent_{prefix}"],
            right_on=[f"{prefix}_{image_number_col}", f"{prefix}_{object_number_col}"],
            how="left",
        )
    else:
        cell_df_mitolyso = pd.merge(
            cell_df_mitolyso,
            new_df,
            left_on=[image_number_col, object_number_col],
            right_on=[f"{prefix}_{image_number_col}", f"{prefix}_{object_number_col}"],
            how="left",
        )
    return cell_df_mitolyso


def make_per_lysosome_area_column_names(
    df,
    area_col="AreaShape_Area",
    use_cols=[],
    base_cols=[
        "Metadata_PlateNumber",
        "Metadata_RowColField",
        "AllGroups",
        "AreaShape_Area",
    ],
    area_cols=[],
    count_cols=[],
    calculate_totals=False,
):
    df = df.copy()
    new_use_cols = use_cols.copy()
    for col in base_cols:
        if col in new_use_cols:
            new_use_cols.remove(col)

    for col in area_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if calculate_totals and "Mean" in col:
            col_without_mean = col.replace("Mean_", "")
            if "IJ" in col:
                df["Total_" + col_without_mean] = (
                    df[col] * df["Children_Lysosomes_IJ_Count"]
                )
            else:
                df["Total_" + col_without_mean] = (
                    df[col] * df["Children_Lysosomes_Count"]
                )

            df["Per_Area_AreaOccupied_" + col_without_mean] = (
                df["Total_" + col_without_mean] / df[area_col]
            )
        elif "Total" in col:
            col_without_total = col.replace("Total_", "")
            df["Per_Area_AreaOccupied_" + col_without_total] = df[col] / df[area_col]

        # remove the col from new_use_cols so that it doesn't get processed again in the final loop
        if col in new_use_cols:
            new_use_cols.remove(col)

    for col in count_cols:
        df["Per_Area_Number_" + col] = df[col] / df[area_col]
        if col in new_use_cols:
            new_use_cols.remove(col)

    # for everything else in use_cols, just divide by area
    for col in new_use_cols:
        print("making per area column for:", col)
        df["Per_Area_" + col] = df[col] / df["AreaShape_Area"]

    return df


def get_duplicate_rows(df, subset_cols=None, name="df", keep=False, show=True):
    """
    Return duplicate rows and optionally print a short report.
    Args:
        df (pd.DataFrame): The dataframe to check for duplicates.
        subset_cols (list, optional): The list of columns to check for duplicates. Defaults to None

    subset=[...] checks duplicates only on those columns.
    """
    dup_mask = df.duplicated(subset=subset_cols, keep=keep)
    dup_rows = df.loc[dup_mask].copy()

    if show:
        if subset_cols is None:
            print(f"{name}: {dup_rows.shape[0]} duplicate rows found")
        else:
            print(
                f"{name}: {dup_rows.shape[0]} duplicate rows found for keys {subset_cols}"
            )

    return dup_rows


def merge_ij_skeleton_features_into_combined_dataframe(
    combined_cell_df_mitolyso,
    ij_csvpath,
    ij_csv,
    ij_keys=None,
    parent_keys=None,
    check_duplicates=True,
    show=False,
):
    """
    Merge the combined cell dataframe with the ImageJ skeleton features dataframe based on plate number
    Args:
        combined_cell_df_mitolyso (pd.DataFrame): The combined cell dataframe.
        ij_csvpath (str): The path to the directory containing the ImageJ skeleton features csv file.
        ij_csv (str): The name of the ImageJ skeleton features csv file.
        ij_keys (list): The list of columns to merge on from the ImageJ dataframe.
        parent_keys (list): The list of columns to merge on from the parent dataframe.
    """
    if ij_keys is None:
        ij_keys = ["Metadata_PlateNumber", "Metadata_RowColField", "ObjectNumber"]
    if parent_keys is None:
        parent_keys = ["Metadata_PlateNumber", "Metadata_RowColField", "ObjectNumber"]

    ij_skeleton_df = pd.read_csv(os.path.join(ij_csvpath, ij_csv))

    csv_name = (
        ij_csv.split(".")[0].strip().title()
    )  # convert to title to match conventions
    csv_suffix = "IJ_Mitochondria_" + csv_name.split("_")[-1].strip()

    # check for dupe keys in the ij_skeleton_df before merging
    if check_duplicates:
        dup_keys = get_duplicate_rows(
            ij_skeleton_df,
            subset_cols=["ImageTitle"] + ij_keys,
            name=f"{ij_csv} merge keys",
            show=show,
        )
        if not dup_keys.empty:
            ij_skeleton_df = ij_skeleton_df.drop_duplicates(
                subset=["ImageTitle"] + ij_keys, keep="first"
            )
            if show:
                display(dup_keys.sort_values(ij_keys))

    for col in ij_skeleton_df.columns:
        new_col = col
        if "+AF8-" in col:
            new_col = col.replace("+AF8-", "_")
        if new_col not in ij_keys + [
            "ImageTitle",
        ]:  # rename the column to easily idefntify these features
            new_col = f"{csv_suffix}_{new_col}"
            ij_skeleton_df.rename(columns={col: new_col}, inplace=True)

        ij_skeleton_df["Metadata_PlateNumber"] = ij_skeleton_df[
            "Metadata_PlateNumber"
        ].astype(int)
    if show:
        # display(ij_skeleton_df)
        display(f"shape before merge: {combined_cell_df_mitolyso.shape}")

    combined_cell_df_mitolyso = pd.merge(
        combined_cell_df_mitolyso,
        ij_skeleton_df,
        left_on=parent_keys,
        right_on=ij_keys,
        how="left",
        suffixes=("", f"_{csv_suffix}"),
    )
    if show:
        display(f"shape after merge: {combined_cell_df_mitolyso.shape}")
    return combined_cell_df_mitolyso


def merge_ij_skeleton_features_into_combined_dataframe_from_folder(
    df,
    ij_csvpath,
    ij_csvs=None,
    ij_keys=None,
    parent_keys=None,
    check_duplicates=True,
    show=False,
):
    """
    Merge the combined cell dataframe with the ImageJ skeleton features dataframes from a folder based on plate number
    Args:
        df (pd.DataFrame): The combined cell dataframe.
        ij_csvpath (str): The path to the directory containing the ImageJ skeleton features csv files.
        ij_csvs (list): The list of ImageJ skeleton features csv files.
        ij_keys (list): The list of columns to merge on from the ImageJ dataframe.
        parent_keys (list): The list of columns to merge on from the parent dataframe.
    """
    if parent_keys is None:
        parent_keys = ["Metadata_PlateNumber", "Metadata_RowColField", "ObjectNumber"]
    if ij_keys is None:
        ij_keys = ["Metadata_PlateNumber", "Metadata_RowColField", "ObjectNumber"]

    combined_cell_df_mitolyso = df.copy()
    if ij_csvs is None:
        ij_csvs = os.listdir(ij_csvpath)
    for item in ij_csvs:
        if os.path.isdir(os.path.join(ij_csvpath, item)) and item not in [
            "AllSkeletons",
            "test",
        ]:
            ij_csv_folder = os.path.join(ij_csvpath, item)
            ij_csv_folder_items = os.listdir(ij_csv_folder)
            for file in ij_csv_folder_items:
                if file.endswith(".csv"):
                    ij_csv = str(file)
                    print(f"Processing {ij_csv} in folder {item}...")
                    combined_cell_df_mitolyso = (
                        merge_ij_skeleton_features_into_combined_dataframe(
                            combined_cell_df_mitolyso,
                            ij_csv_folder,
                            ij_csv,
                            show=show,
                            ij_keys=ij_keys,
                            parent_keys=parent_keys,
                            check_duplicates=check_duplicates,
                        )
                    )
        else:
            if item.endswith(".csv"):
                ij_csv = str(item)
                print(f"Processing {ij_csv} in folder {item}...")
            else:
                print(f"Skipping {item}, not a CSV file.")
                continue
            combined_cell_df_mitolyso = (
                merge_ij_skeleton_features_into_combined_dataframe(
                    combined_cell_df_mitolyso,
                    ij_csvpath,
                    ij_csv,
                    show=show,
                    ij_keys=ij_keys,
                    parent_keys=parent_keys,
                    check_duplicates=check_duplicates,
                )
            )
    return combined_cell_df_mitolyso


def get_standard_deviations_from_large_df(
    df,
    object_df_name,
    parent_key,
    child_key,
    features=[
        "AreaShape_Area",
        "AreaShape_Perimeter",
        "AreaShape_EquivalentDiameter",
        "AreaShape_Extent",
        "AreaShape_Solidity",
        "AreaShape_Compactness",
        "AreaShape_Eccentricity",
    ],
    prefix="Cell",
):
    """
    Calculate the standard deviation of specified features in a dataframe, grouped by certain metadata columns.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data.
    object_df (pd.DataFrame): A dataframe containing object-level data.
    features (list): A list of feature column names for which to calculate standard deviations.
    parent_key (str): The name of the parent key column in the dataframe.
    child_key (str): The name of the child key column in the dataframe.

    Returns:
    pd.DataFrame: A dataframe containing the standard deviations of the specified features, grouped by metadata.
    """
    # open the csv file with extra info
    object_df = pd.read_csv(object_df_name)
    modified_df = df.copy()
    object_df_name_noext = os.path.splitext(os.path.basename(object_df_name))[0]

    for feature in object_df.columns:
        # print("checking feature:", feature)
        # Exclude if matches any in other_channels, unless it also matches feature_types
        if feature not in features:
            print("oops, skipping:", feature)
            continue  # Skip the parent key column or non-feature columns
        if any(ft in feature for ft in features):
            modified_df = calculate_aggregated_object_features(
                modified_df,
                object_df,
                feature,
                parent_key=parent_key,
                object_name=object_df_name_noext,
                child_key=child_key,
                aggregation="Std",
                prefix=prefix,
            )

    # modified_df = modified_df.rename(columns=lambda x: re.sub(r"Std", f"Std_{object_df_name_noext}", x))
    return modified_df.copy()


def plot_boxplots_with_swarm(
    display_df,
    display_df_pivot_plates,
    feature_cols,
    colour_dict={},
    size=(6, 10),
    outpath="lyso_segmentation_test/plots",
):

    os.makedirs(outpath, exist_ok=True)
    for col in feature_cols:
        plt.figure(figsize=size)
        sns.set_style("whitegrid")
        sns.set_context("notebook")
        sns.boxplot(
            display_df,
            x="AllGroups",
            y=col,
            palette="pastel",
            fill=True,
            hue="AllGroups",
            legend=False,
        )
        sns.swarmplot(
            display_df_pivot_plates,
            x="AllGroups",
            y=col,
            palette=colour_dict,
            hue="Metadata_PlateNumber",
            dodge=False,
            size=9,
            legend=False,
            linewidth=1,
            edgecolor="k",
        )
        sns.despine()
        plt.ylim(0, np.percentile(display_df[col].dropna(), 99) * 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, f"{col}_boxplot.png"))
        plt.show()
