import papermill as pm

area_features = [
    "Cell_AreaShape_Area",
    "Nuclei_AreaShape_Area",
    "Cell_Nuclei_Area_Ratio",
]
feature_shortnames = ["cell_area", "nucleus_area", "cell_nuc_ratio"]

for i in range(len(area_features)):
    feat_index = i
    pm.execute_notebook(
        input_path="/mnt/bigdisk1/AllieSpangaro/HTP_fibroblasts_mitolyso/cell_area_size_comparison.ipynb",
        output_path="/mnt/bigdisk1/AllieSpangaro/HTP_fibroblasts_mitolyso/cell_area_size_comparison_output_{}.ipynb".format(
            i
        ),
        parameters={
            "feat_index": feat_index,
            "area_features": area_features,
            "feature_shortnames": feature_shortnames,
            "csvpath": "/mnt/bigdisk1/AllieSpangaro/Morphology_Replicative_Age_Project/CP_Output/postprocessed_csvs/",
            "stitched_path": "/mnt/bigdisk1/AllieSpangaro/Morphology_Replicative_Age_Project/stitching/segmentation_testset",
            "filename": "total_combined_cell.csv",
            "stitched_csv": "stitched_test_data_v3.csv",
            "stitched_csv_borders_excluded": "stitched_test_data_v3_borders_excluded.csv",
            "stitched_nuc_csv": "nuclei_stitched_test_data_v3.csv",
            "stitched_nuc_csv_borders_excluded": "nuclei_stitched_test_data_v3_borders_excluded.csv",
            "csv_outpath": "/mnt/bigdisk1/AllieSpangaro/Morphology_Replicative_Age_Project/CP_Output/postprocessed_csvs/",
            "summary_outpath": "/mnt/bigdisk1/AllieSpangaro/Morphology_Replicative_Age_Project/Cell_Size_Data/summary_stats/",
        },
    )
