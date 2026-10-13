# Bundled CellChatDB interaction tables

These CSV files contain species-specific CellChatDB interaction-table snapshots
used by [CellSTIC's loader](../../utils/tools/cellchat_utils.py). CellChatDB is
curated by the CellChat authors; its canonical upstream project is
[jinworks/CellChat](https://github.com/jinworks/CellChat). The official
[database documentation](https://github.com/jinworks/CellChat/blob/main/R/data.R)
also links to the earlier `sqjin/CellChat` repository.

## Source and local versions

| Local file | Canonical upstream data object | Interaction rows | Version labels present in this CSV |
|---|---|---:|---|
| `CellChatDB.human.csv` | [CellChatDB.human.rda](https://github.com/jinworks/CellChat/blob/main/data/CellChatDB.human.rda) | 3,233 | `CellChatDB v1`: 1,920; `CellChatDB v2`: 1,313 |
| `CellChatDB.mouse.csv` | [CellChatDB.mouse.rda](https://github.com/jinworks/CellChat/blob/main/data/CellChatDB.mouse.rda) | 3,379 | `CellChatDB v1`: 1,993; `CellChatDB v2`: 1,386 |
| `CellChatDB.zebrafish.csv` | [CellChatDB.zebrafish.rda](https://github.com/jinworks/CellChat/blob/main/data/CellChatDB.zebrafish.rda) | 2,774 | No `version` column; database version unknown |

Counts exclude the header. Human and mouse have 28 named columns; zebrafish has
11. Each CSV also has an unnamed first column containing exported row labels.
The human and mouse `version` fields label individual interactions; they do not
identify the CellChat package release or the exact upstream revision.

The original export script, acquisition/export date, CellChat package version,
and upstream commit were not recorded in the available CellSTIC files or Git
history. The upstream links above identify official source locations, not a
verified original revision for these local files. No claim of an exact match
to the current upstream data objects is made.

The official
[CellChatDB export tutorial](https://github.com/jinworks/CellChat/blob/main/tutorial/Update-CellChatDB.Rmd)
describes the main table as `CellChatDB$interaction` and demonstrates CSV
export. These local files have the corresponding interaction-table schema;
they do not include the separate `complex`, `cofactor`, or `geneInfo` tables
from a complete CellChatDB R object. Complex names in ligand/receptor fields
may therefore refer to definitions outside these snapshots.

## License and attribution

The upstream CellChat package declares `License: GPL-3` in its
[DESCRIPTION](https://github.com/jinworks/CellChat/blob/main/DESCRIPTION), and
the repository includes the
[GNU GPL version 3 license](https://github.com/jinworks/CellChat/blob/main/LICENSE).
That complete license text is available locally in [../../LICENSE](../../LICENSE);
it was verified to be byte-identical to the upstream license on 2026-10-12.
These snapshots retain attribution to CellChat and this notice of its upstream
GPL-3 terms.

The upstream sources reviewed do not provide a separate CellChatDB data
license. The tables also contain evidence references to external resources,
including KEGG. Their independent terms and the precise scope of the package
license over externally sourced annotations were not established by the
recorded local provenance. This notice reports the upstream package/repository
license and the remaining provenance limits; it does not assign additional
rights to third-party material.

The [official CellChat citation guidance](https://github.com/jinworks/CellChat#about-cellchat-and-cellchatdb)
requests citations when using CellChat or CellChatDB:

- Jin et al. (2021). *Inference and analysis of cell-cell communication using
  CellChat*. Nature Communications 12, 1088.
  [doi:10.1038/s41467-021-21246-9](https://doi.org/10.1038/s41467-021-21246-9).
- Jin et al. (2024). *CellChat for systematic analysis of cell–cell communication
  from single-cell transcriptomics*. Nature Protocols.
  [doi:10.1038/s41596-024-01045-4](https://doi.org/10.1038/s41596-024-01045-4).

## File integrity

SHA-256 identifies the exact bundled CSV bytes. These checksums do not verify
the unknown upstream export revision.

```text
38f1ec2cfdc8a0feed42cb8b835f697727f70e19e60322d9dbfb0ce20e6dbb3d  CellChatDB.human.csv
e3d9372f3323113045b580137e854cd79905ed1e0965ce68fb2921079b11f9cf  CellChatDB.mouse.csv
3cb2f2df35df90a75e180ede00aa18ab1e49ad39aa2fe8267ea6cce79900ad3b  CellChatDB.zebrafish.csv
```

When replacing these files, record the upstream commit/package version,
retrieval date, export command, and any filtering or field transformations,
then update the counts and checksums above.
