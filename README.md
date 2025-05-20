# HTP fibroblasts mitolyso
Stats and image analysis codes for my Masters project using high-throughput microscopy to quantify changes in mitochondrial and lysosomal morpholgy in serially-passaged fibroblasts across their lifespan.

## Scripts / notebooks
- 96Well_PlateMap - make a metadata sheet from a csv representation of 96-well plate containing the serial passage batch, replicate, passage number, and fluorescence staining used
- Proliferation_GrowthCurves - make growth curves from proliferation assays aquired on an Incuctye live cell imager
- CP_Data_NoDB - analyze cell morphology from cellprofiler segmentation results in csv files. Generate graphs and calculate p values via one way ANOVA. Will also incorportate t-SNE.
  - helpers.py contains helper functions for this notebook (along with feat_stat.py)
- BetaGal - visualize and analyze the amount of senescence-associated beta-galactoside staining in cells
- run_Cellpose-SAM - adapted from Stringer et al. to run Cellpose in a jupyter notebook and output segmentation masks as images
  - Helper functions save in cellpose_functions.py
- 

## CellProfiler pipelines
- MitoLyso 
- BetaGal
