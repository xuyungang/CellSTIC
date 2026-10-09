# External example data: SPOTS mouse spleen

This directory contains the complete processed input for mouse spleen
replicate 1 from the public SPOTS spatial transcriptomics--proteomics study
([GEO GSE198353](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353)).
SPOTS measures whole-transcriptome gene expression and antibody-derived tags
(ADTs) at the same spatial barcodes.

The two files are the unmodified GEO downloads:

| File | Contents | Size |
|---|---|---:|
| `GSE198353_spleen_rep_1_filtered_feature_bc_matrix.h5` | 10x HDF5 matrix containing 32,285 RNA and 21 ADT features at 2,653 filtered spatial barcodes | 11.6 MiB |
| `GSE198353_spleen_replicate_1_spatial.tar.gz` | Visium coordinates, scale factors, and tissue images | 4.1 MiB |

No spots or molecular features were sampled from these files. All 2,653
matrix barcodes are marked as tissue spots and have matching coordinates.
The tutorial separates the two feature types into aligned AnnData objects but
retains every barcode and feature. CellSTIC subsequently selects highly
variable RNA features only for constructing its PCA representation; the full
RNA matrix remains in the exported result.

Exact source URLs, SHA-256 checksums, dimensions, and the retrieval date are
recorded in `manifest.json`. The associated article is Ben-Chetrit et al.,
*Nature Biotechnology* 41, 788--793 (2023),
[doi:10.1038/s41587-022-01536-3](https://doi.org/10.1038/s41587-022-01536-3).
Please cite both the article and GEO accession when reusing the data.
