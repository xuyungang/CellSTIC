# E12.5 Mouse Embryo Visium HD and PLA

The paired E12.5 mouse embryo data are from Liu et al., *Nature Genetics*
(2026), [doi:10.1038/s41588-026-02624-9](https://doi.org/10.1038/s41588-026-02624-9).
Visium HD count data are deposited as
[GEO GSE291088](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE291088),
and the five raw 16-bit proximity ligation assay (PLA) images are in
[Zenodo record 18625864](https://zenodo.org/records/18625864). A1
(GSM8829020) and D1 (GSM8829021) are two distinct Visium HD sections;
each was paired with spatially adjacent PLA sections.

| Sample | Genes | 16-µm bins | Nonzero RNA counts | PLA pairs |
|---|---:|---:|---:|---|
| A1 | 19,059 | 85,017 | 78,617,738 | Fgf8–Fgfr1; Dll1–Notch1 |
| D1 | 19,059 | 93,248 | 100,125,764 | Spp1–Cd44; Igf2–Igf2r; Efna3–Epha5 |

The five PLA images match the checksums in the Zenodo record. Their native
sizes are 9,832 × 11,776 or 11,264 × 13,108 pixels for A1, and
9,422 × 13,108 pixels for D1.

## A1 barcode anomaly

The GEO item labeled as the A1 16-µm barcode list duplicates the gene-feature
list and has 19,059 rows, while the A1 count matrix requires 85,017 barcodes.
The 85,017 unique barcode values used for A1 were recovered, in matrix order,
from the corresponding Space Ranger graph-clustering table. The ordering rule
was checked against D1: 93,236 clustered barcodes preserve the order of the
93,248-barcode D1 matrix; 12 D1 barcodes were unclustered.
