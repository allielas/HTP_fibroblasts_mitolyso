---
Date: 2025-02-19
---
# Features
## From [Jang et al.(2022) Science](https://www.researchgate.net/publication/366353742_Endosomal_lipid_signaling_reshapes_the_endoplasmic_reticulum_to_control_mitochondrial_function?enrichId=rgreq-b6059081215fe8d97c387c9e48528e32-XXX&enrichSource=Y292ZXJQYWdlOzM2NjM1Mzc0MjtBUzoxMTQzMTI4MTIwNTA3MTIzOUAxNzAwMDg4MTE4Mzkw&el=1_x_3&_esc=publicationCoverPdf) 
IF protocol
- 4% PFA fixation
- Wash in PBST 0.3% Triton
- 3% BSA blocking in PBST (0.3% Trition)
- 2 hr primary, 1 hr secondary
- 3x wash in PBST 
Mitochondria
- Gaussian filter
- Enhance tubness
- Robust backgorund segmentation
- Meaure number and area per ROI
  - Averaged per cell in an ROI
Lysosomes
- Mask within cell
- Enhance speckles (~5 px)
  - Then gaussian
- Identify lysosomes via robust background and shrink to a point
  - Then expand by 1 px
  - 2 SD and 0.05 bounds
- Relate objects to calculate centroid distance from lysosome to cell centroid
  - stdev describes the extent of lysosomal dispersion
- “Fraction at distance" from intensity distirbution