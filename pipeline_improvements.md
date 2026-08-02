---
Date_Created: 2026-08-02
Title: Pipeline Improvements Suggestions 
---
# Pipeline Improvements Suggestions

## For macro used by cellprofiler

- Remove the skeletonizing, I don't use that one in the final pipeline

## For cp itself

- update pipeline to match IJ macro
- use the new cellpose masks (after validating them) and reduce the amount of dilation I do
- make sure the plate 5 does not remove the first row with doxo
- export closed_mitochondria skeletons as single-object binary mask images
- maybe export them to the same folder for all the pipelines so I can save time moving them around after
- for object neighbours, increase the distance between parents
- double check intensity distribution
- make a little version of the pipeline to make figures
- Maybe remove lysosomes/mito that are super bright?
- 

## For ImageJ post-processing script

- Grab skeletons produced by cellprofiler (along with masks) to save time
- Rename some features to be more intuitive
  - Replace the feature names as per the logic; \_Max = LargestStructure, \_Total = TotalAcrossAllStructures
  - IJ_Mitochondria_Masks_BranchLength_Max and IJ_Mitochondria_Masks_LargestStructure_MaximumBranchLength are the same, drop the former so its consistent
  - same with IJ_Mitochondria_Masks_BranchesPerStructure_Max and IJ_Mitochondria_Masks_LargestStructure_NumberOfBranches
  - "IJ_Mitochondria_Masks_SkeletonLength_Max" should be renamed to "IJ_Mitochondria_Masks_LargestStructure_SkeletonLength"
  - "IJ_Mitochondria_Masks_TotalAcrossAllStructures_SkeletonLength" and "IJ_Mitochondria_Masks_TotalAcrossAllStructures_SkeletonLength_FromBranches" are basically the exact same, remove the latter for simplicity and to save processing time
