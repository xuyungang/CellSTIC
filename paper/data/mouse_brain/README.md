# Mouse Brain Spatial Multi-omics

Source: Guo, P., Mao, L., Chen, Y. et al. Multiplexed spatial mapping of
chromatin features, transcriptome and proteins in tissues. *Nature Methods*
22, 520–529 (2025).
[doi:10.1038/s41592-024-02576-0](https://doi.org/10.1038/s41592-024-02576-0)

The five aligned assays cover 9,994 mouse-brain spatial observations.

| Modality | Features |
|---|---:|
| RNA | 48,440 genes |
| ADT | 131 proteins |
| ATAC | 61,366 peaks |
| H3K27ac | 49,550 peaks |
| H3K27me3 | 68,943 broad regions |

The current chromatin matrices use integer fragment-overlap counts for MACS2
BEDPE peak calls: narrow peaks for ATAC and H3K27ac, and broad regions for
H3K27me3. Peak intervals retain zero-based, half-open genomic boundaries.
Primary chromosomes (chr1–19 and chrX) and the ArchR mm10 blacklist were used.
Peak calling did not use RNA cell-type labels or spatial smoothing. Because
there was no matched input control, the calls are not independent biological
validation.
