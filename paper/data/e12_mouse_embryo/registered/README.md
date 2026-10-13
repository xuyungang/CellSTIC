# E12.5 Mouse Embryo PLA Registration Data

Five 16-bit PLA images from sections adjacent to A1 and D1 are anatomically
aligned to a 23,216 × 13,108-pixel H&E reference grid. The derived data
include registered PLA intensity, a binary mask marking sampled versus padded
pixels, affine transforms between source and reference coordinates, and tissue
outline quality-control views. Visium HD spot coordinates are full-resolution
H&E pixel centres in `(x, y)` order; image arrays use `[y, x]` indexing.

The alignment uses tissue anatomy rather than PLA intensity. Tissue-outline
overlap assesses only coarse outer-boundary correspondence. Serial sections can
differ locally even after affine alignment. These registered images therefore
support whole-tissue comparison and visually reviewed regions of interest;
they are not validated 16-µm-bin PLA ground truth without further local
registration and independent error assessment.
