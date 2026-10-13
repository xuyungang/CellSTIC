# CellTypist reference models

This directory contains five pretrained CellTypist cell-type annotation models.
The mouse brain and liver models are used by
[`paper/preprocess/mouse_embryo.py`](../../paper/preprocess/mouse_embryo.py).
These are third-party model weights, with provenance and license scope recorded
below.

## Official sources

- Software project: [Teichlab/celltypist](https://github.com/Teichlab/celltypist).
- Model catalog and citation guidance: [CellTypist models](https://www.celltypist.org/models).
- Machine-readable catalog: [official model index](https://celltypist.cog.sanger.ac.uk/models/models.json).
- Software license: [upstream LICENSE](https://github.com/Teichlab/celltypist/blob/main/LICENSE).
- Website terms linked from the model catalog: [Wellcome Sanger Institute legal terms](https://www.sanger.ac.uk/legal/).

These sources were reviewed on **2026-10-12**. The official model index reported
`last_update: 2026-03-16 15:58:39.812071` at that review.

## Local inventory

The version, date, download URL, and training-source DOI below were read from
the local pickle `description` metadata using static `pickletools` inspection,
without loading or executing the serialized objects. The five corresponding
entries in the official model index matched those metadata fields when
reviewed. The remote model binaries were not downloaded or compared with the
local files; a matching catalog entry does not establish byte-for-byte identity.

The dates below are upstream model metadata dates, with no timezone specified.
The original dates on which these files were obtained for this repository are
**unknown**; model metadata dates and filesystem timestamps are not retrieval
records.

| Local file | Size (bytes) | Local model version | Embedded model date | Annotation scope |
|---|---:|---|---|---|
| `Adult_Human_Skin.pkl` | 556,651 | v1 | 2023-09-22 18:13:10.902604 | Healthy adult human skin |
| `Cells_Human_Tonsil.pkl` | 4,852,528 | v1 | 2024-04-21 17:50:57.165795 | Human tonsil, ages 3–65 years |
| `Developing_Mouse_Brain.pkl` | 5,410,811 | v1 | 2022-01-25 09:00:56.249989 | Embryonic mouse brain, gastrulation to birth |
| `Healthy_Mouse_Liver.pkl` | 337,777 | v1 | 2023-07-28 12:16:49.675179 | Healthy mouse liver, scRNA-seq and snRNA-seq |
| `Immune_All_Low.pkl` | 2,824,990 | v2 | 2022-07-16 00:20:42.927778 | Immune subpopulations from 20 tissues and 18 studies |

The following original download links and training-source papers are recorded
in each local file and in the official model index:

| Local file | Original model URL | Training-source paper |
|---|---|---|
| `Adult_Human_Skin.pkl` | [Human_Skin_Reynolds/v1/Adult_Human_Skin.pkl](https://celltypist.cog.sanger.ac.uk/models/Human_Skin_Reynolds/v1/Adult_Human_Skin.pkl) | [doi:10.1126/science.aba6500](https://doi.org/10.1126/science.aba6500) |
| `Cells_Human_Tonsil.pkl` | [Human_Tonsil_Badosa/v1/Cells_Human_Tonsil.pkl](https://celltypist.cog.sanger.ac.uk/models/Human_Tonsil_Badosa/v1/Cells_Human_Tonsil.pkl) | [doi:10.1016/j.immuni.2024.01.006](https://doi.org/10.1016/j.immuni.2024.01.006) |
| `Developing_Mouse_Brain.pkl` | [Mouse_Devbrain_Manno/v1/Developing_Mouse_Brain.pkl](https://celltypist.cog.sanger.ac.uk/models/Mouse_Devbrain_Manno/v1/Developing_Mouse_Brain.pkl) | [doi:10.1038/s41586-021-03775-x](https://doi.org/10.1038/s41586-021-03775-x) |
| `Healthy_Mouse_Liver.pkl` | [Mouse_Liver_Guilliams/v1/Healthy_Mouse_Liver.pkl](https://celltypist.cog.sanger.ac.uk/models/Mouse_Liver_Guilliams/v1/Healthy_Mouse_Liver.pkl) | [doi:10.1016/j.cell.2021.12.018](https://doi.org/10.1016/j.cell.2021.12.018) |
| `Immune_All_Low.pkl` | [Pan_Immune_CellTypist/v2/Immune_All_Low.pkl](https://celltypist.cog.sanger.ac.uk/models/Pan_Immune_CellTypist/v2/Immune_All_Low.pkl) | [doi:10.1126/science.abl5197](https://doi.org/10.1126/science.abl5197) |

## Local file integrity

These SHA-256 digests were computed from the local files on 2026-10-12. They
identify the exact bytes included here; they are not upstream-published checksums
and do not establish a redistribution license.

```text
65bbd15b3776f95412257c6ccdec540d5ce1574e0955ed3a349a166205860879  Adult_Human_Skin.pkl
ad6b6d70d8a63550d57d969f112e079f7a13f279d142237b7ce209875dd04953  Cells_Human_Tonsil.pkl
52d94194410d6ffeb3cfb750dff1654a3b3d612154c5a1b095f1ec859caa20a6  Developing_Mouse_Brain.pkl
0136f4a030cbdabc046f7414ba0eb30b8b0ff5459321a805299f43beb6564a15  Healthy_Mouse_Liver.pkl
290874d35dac039d4c9218c343fde4aac1077709b72a331ce7266f6828c36502  Immune_All_Low.pkl
```

To check the local bytes, run `sha256sum resources/celltypist/*.pkl` from the
repository root and compare with this list.

## License scope and attribution

The **CellTypist software** is distributed under the MIT license, with copyright
attributed to the Teichmann Lab (2020) in the
[upstream LICENSE](https://github.com/Teichlab/celltypist/blob/main/LICENSE).
That license requires preservation of its copyright and permission notices when
redistributing covered software. This software license alone does not establish
the license of separately downloaded model weights derived from individual
studies.

The model catalog links to the
[Sanger legal terms](https://www.sanger.ac.uk/legal/). Those terms state a general
default of **CC BY-NC-ND 4.0** for online content created and hosted by the
Institute unless another license is stated, and call for attribution to Genome
Research Limited with a source link. They also provide general guidance on
acknowledging data in publications. The terms do not identify these five
CellTypist pickle files or explicitly resolve how that website-content default
applies to their downloadable model weights.

**The redistribution license for all five model-weight files is unconfirmed.**
No explicit model-weight license was found in their embedded metadata or their
official index entries during this review. This document therefore does not
assign MIT, the CellSTIC GPLv3 license, or the source papers' licenses to these
weights. Before redistributing the weights, obtain and record upstream
confirmation of the applicable terms, including whether the Sanger default
applies and any model-specific conditions. The catalog lists
`celltypist@sanger.ac.uk` as a contact for upstream questions.

For scientific reuse, cite the applicable training-source papers above and
follow the [official CellTypist citation guidance](https://www.celltypist.org/models).
This provenance record and the acknowledgment of source studies do not replace
permission or the applicable license terms.
