# CellSTIC pipeline and I/O contract

This document is the canonical entry point for the CellSTIC workflow. Dataset
READMEs describe only dataset acquisition and assembly differences; the model
card describes intended use and limitations. The interfaces, dimensions, files,
and hand-offs shared by all datasets are defined here.

## Pipeline at a glance

```text
Raw dataset files
    -> dataset-specific assembly
Aligned AnnData objects, one per modality
    -> prepare_modalities
Model-ready features and sparse graphs
    -> feature integration and clustering
Integrated cell representation and spatial domains
    -> optional CCC training and leaf inference
Leaf ligand-receptor score matrices
    -> optional leaf calibration, then hierarchy aggregation
out/out.h5ad + checkpoint/*.pt
    -> Analysis.from_result
Tables and requested figures
```

The public end-to-end entry point is `pipeline.run_cellstic`. It performs
preprocessing, graph construction, dataset-specific unsupervised optimization,
inference, and result export in one run.

## 1. Assemble one AnnData object per modality

Dataset-specific scripts under `paper/preprocess/` convert raw files into
assembled `.h5ad` files. For the standard layout, each file is named
`<modality>.h5ad` and loaded with:

```python
from paper.preprocess import read_modalities

rna, atac = read_modalities("paper/data/ccc_sim/h5ad/domains_11", ["rna", "atac"])
```

### Input AnnData contract

Let `n` denote cells or spots, `p_i` the raw feature count of modality `i`, and
`m` the number of modalities.

| Location | Format and shape | Requirement | Meaning |
|---|---|---|---|
| `.X` | Dense array or SciPy sparse matrix, `(n_i, p_i)` | Required for every modality | Raw or assembled molecular measurements. The first modality must contain gene expression when CCC inference is enabled. |
| `.obs_names` | Unique string-like index, length `n_i` | Required | Cell/spot identifiers. Modalities are intersected and reordered to the first modality's order. At least two shared identifiers are required. |
| `.var_names` | String-like index, length `p_i` | Required | Feature identifiers. For the first RNA modality, names must match the ligand and receptor gene symbols used by the LR map. |
| `.obsm["spatial"]` | Finite numeric array, `(n_0, 2)` | Required in the first modality | Two-dimensional spatial coordinates. Coordinates are copied to the aligned modalities. |
| `.obs["cell_type"]` | Categorical or string vector, length `n_0` | Optional | Used only when downstream hierarchy visualization groups results by cell type; it does not constrain candidate events or training. |
| `.uns["ligand_receptor_map"]` | Mapping `{ligand: [receptor, ...]}` | Optional alternative to the function argument | Candidate LR events. A receptor complex is encoded with colon-separated subunits, for example `"ITGAV:ITGB3"`. |
| `.uns["edge_type_map"]` | Mapping `{LR_name: integer_channel}` | Optional compatibility metadata | Read by dataset helpers; the model constructs its active channel map from the retained LR candidates. |

`paper.preprocess.cellstic_metadata(rna)` reads the optional LR and
edge-type mappings. The LR map may instead be supplied directly to
`run_cellstic`.

### Handoff

The assembly step produces a sequence of `m` AnnData objects. Their order must
match both `modality_types` and `config.model.feat.encoder_dims`. The first
object is the primary result object and the expression source for CCC.

## 2. Select and validate the configuration

Repository presets are returned as immutable, validated configuration objects:

```python
from model.train import get_cellstic_config

config = get_cellstic_config("scmultisim")
```

Available preset names are `human_skin`, `human_tonsil`,
`human_lymph_node`, `mouse_brain`, `nsf`, `mouse_embryo`,
`axolotl_telencephalon`, and `scmultisim`.

For hierarchy experiments, pass `tree_height=4` to
`get_cellstic_config("human_lymph_node", tree_height=4)`. Omitting it uses the
preset's default. The requested height must be at least two.

For modality `i`, define:

- `d_i = config.model.feat.encoder_dims[i][0]`: preprocessed model-input width;
- `z_i = config.model.feat.encoder_dims[i][-1]`: modality latent width;
- `f = config.model.feat.output_dims[-1]`: fused output width; and
- `h = config.model.ccc.encoder_dims[-1]`: CCC encoder output width.

The validated dimension contract is:

| Component | Required dimensions |
|---|---|
| Modality encoder `i` | `d_i -> ... -> z_i` |
| Modality decoder `i` | `z_i -> ... -> d_i`; every model-input coordinate is reconstructed |
| Multimodal fusion | `sum_i(z_i) == feat.output_dims[0]`, then `... -> f` |
| CCC expression encoder | Its input equals `d_0`, the first-modality model input |
| CCC pair decoder | Its input equals `2 * (f + h)` |

The bundled presets resolve to the following model-ready dimensions:

| Preset | Modality encoders | Full reconstruction decoders | Fused network |
|---|---|---|---|
| `human_skin` | `250->200->120` x2 | `120->250` x2 | `240->100` |
| `human_tonsil` | `250->200->120` x2 | `120->250` x2 | `240->100` |
| `human_lymph_node` | `500->200`; `30->20` | `200->500`; `20->30` | `220->100` |
| `mouse_brain` | `131->100->50` x5 | `50->131` x5 | `250->50` |
| `nsf` | `100->85->60` x3 | `60->100` x3 | `180->60` |
| `mouse_embryo` | `500->300->200` | `200->500` | `200->100` |
| `axolotl_telencephalon` | `500->200` | `200->500` | `200->100` |
| `scmultisim` | `120->100->80` x2 | `80->120` x2 | `160->120->80` |

Raw feature width and model-input width are different quantities. For example,
an AnnData matrix with 2,000 raw features is reduced to the preset value `d_i`
before entering the network. If a custom configuration explicitly sets
`d_i = 2000`, the decoder must end at 2,000 and reconstructs all 2,000
coordinates; no feature-axis truncation is permitted.

Runtime validation also requires one encoder definition per modality,
`knn_top_k < n`, `cluster_top_k < n`, and `2 <= n_clusters <= n`.

### Runtime resources

`resources/` contains local reference data and models used by optional
preprocessing and hierarchy-building paths. Keep this directory beside the
source tree when deploying CellSTIC.

| Resource | Operational role | When required |
|---|---|---|
| [`resources/cellchatdb/`](resources/cellchatdb/README.md) | CellChat metadata used by the biological hierarchy builder; human, mouse, and zebrafish files are included | Runs using `hierarchy_method="biological"` with matching species metadata |
| [`resources/celltypist/`](resources/celltypist/README.md) | Cell-type annotation models used by mouse-embryo preprocessing | The corresponding mouse-embryo assembly workflow |
| `resources/bge-base-en-v1.5/` | Deterministic semantic embeddings for the LLM hierarchy builder | Runs using the local semantic hierarchy path |
| `resources/aliyun/aliyun_config.yaml` | Service configuration for Aliyun-backed hierarchy generation | Runs explicitly selecting that service; users must provide their own credential |

These optional resources are not required by the core feature-integration path.
When `hierarchy_method="llm"` is selected, however, the BGE model is mandatory:
missing or unloadable model files raise an error and are never replaced with
random semantic vectors. CellChat metadata lookup likewise requires an exact
ligand-receptor pair; a missing pair raises an error instead of borrowing
metadata from an interaction sharing only one endpoint.

## 3. Align and preprocess modalities

`prepare_modalities` performs the in-memory handoff from assembled data to
model-ready features:

```python
from pipeline import prepare_modalities

prepared = prepare_modalities(
    [rna, atac],
    config,
    modality_types=["rna", "atac"],
)
```

| Modality type | Transformation | Generated field |
|---|---|---|
| RNA | Preserve `.layers["counts"]`, select variable genes, library-size normalize, log transform, PCA | `.obsm["feat"]`, `(n, d_i)` |
| ADT/protein | Cell-wise CLR normalization, PCA | `.obsm["feat"]`, `(n, d_i)` |
| `adt_standardized` / `protein_standardized` | Validate finite pre-standardized values, then PCA without CLR | `.obsm["feat"]`, `(n, d_i)` |
| ATAC/chromatin/histone mark | LSI, with PCA fallback for degenerate input | `.obsm["feat"]`, `(n, d_i)` |
| Other | PCA | `.obsm["feat"]`, `(n, d_i)` |

If the available numerical rank is below `d_i`, the representation is
zero-padded; if it is wider, it is truncated to `d_i`. Each prepared object
also receives `.uns["cellstic_preprocessing"]` with the modality name and
effective feature dimension. This preparation is in memory and does not create
an intermediate `.h5ad` cache.

### Handoff

The prepared feature matrices `(n, d_i)` and shared spatial coordinates
`(n, 2)` are passed to sparse graph construction. The prepared first-modality
`.X` and `.var_names` are the gene-expression source for ligand-receptor
candidate filtering; RNA normalization may have updated `.X` while
`.layers["counts"]` preserves the assembled values.

## 4. Build sparse candidate graphs

Feature-neighbour, cluster-neighbour, and spatial candidate graphs are stored
as sparse structures. Spatial CCC candidates are restricted by a distance
radius and by `config.model.graph.max_spatial_neighbors` per cell. Candidate
pairs therefore have shape `(E, 2)`, where `E` is bounded by the neighbourhood
cap rather than by all `n^2` ordered pairs.

The graph stage passes per-cell features `(n, d_i)`, coordinates `(n, c)`, and
sparse edges to the feature model. For CCC, it additionally passes LR channel
identifiers and ligand/receptor expression strengths only for retained spatial
candidates.

## 5. Optimize feature integration

Feature integration is unsupervised and dataset-specific; no supervised
train/validation/test split or cross-validation partition is created. The
feature objective combines clustering structure with full-dimensional
reconstruction of each `(n, d_i)` model-input matrix.

Feature optimization uses the complete aligned sparse graphs in every epoch.
Reconstruction is evaluated for every cell, and soft K-means estimates one
dataset-wide set of cluster centers from all fused cell representations. The
feature-pretraining stage does not split cells into mini-batches, so it never
forces each spatially local subset to contain all configured clusters.

This stage produces:

| Output | Format and shape | Destination |
|---|---|---|
| Integrated representation | `float` array, `(n, f)` | Result `.obsm["out"]` |
| Spatial-domain label | Categorical vector, length `n` | Result `.obs["cluster"]` unless `skip_spatial_domain_prediction=True` |
| Feature model state and configuration | PyTorch checkpoint dictionary | `<output_path>/checkpoint/feature.pt` |

## 6. Infer leaf CCC events

CCC optimization runs only when all of the following are true:

- `run_ccc=True`;
- `ligand_receptor_map` is non-empty; and
- a CCC training configuration is present.

The LR map defines the candidate biological events; inference scores these
events on the sparse spatial candidate graph. It does not search arbitrary
ligand-receptor identities outside the supplied map. Hierarchy heads are
optimized from root to leaf as auxiliary multiscale supervision, while only
the deepest LR leaf scores are predicted directly for export.

CCC path supervision uses sparse `(sender, receiver, LR_channel)` samples
rather than a dense edge-by-channel target. For every hierarchy level and
epoch, the number of masked positive edges in an active channel is:

```text
min(channel_edge_count,
    max_masked_edges_per_channel,
    max(min_masked_edges_per_channel,
        ceil(sampling_rate * channel_edge_count)))
```

The same count of spatially eligible negative samples is drawn for that
channel. Positive and negative samples from each channel are kept as matched
pairs and distributed across local graph batches. A batch contains at most
`train.ccc.graph_batch_size` target edges (default: 1,024), and each channel
represented in a batch contributes the same number of positive and negative
targets. Path-loss weights are normalized over the complete epoch so every
active channel and each sampling class have equal total contribution,
independent of how samples are partitioned into local batches.

Each batch graph is constructed before HODGNN execution:

1. The current positive and negative target edges define the seed cells.
2. Incoming graph neighbours are expanded for the number of HODGNN layers.
3. Aligned, multi-hop feature-graph subgraphs are assembled for the same cells,
   and MEI recomputes their multimodal representations with gradients enabled.
4. The current positive target edges are removed from that local graph to
   prevent reconstruction leakage; sampled negative edges are absent by
   construction.
5. HODGNN encoding, path decoding, loss backpropagation, and one joint optimizer
   update of the CCC predictor and MEI are performed on this local batch.

Consequently, the complete hierarchy-level graph is a read-only source for
local subgraphs and is not encoded as one training computation graph. No dense
`(edges, LR_channels)` training tensor is materialized. The defaults are one
and 256 for the per-channel sampling lower and upper bounds, respectively.
`model.ccc.edge_prob_batch_size` is reserved for chunking inference-time
candidate scoring and is independent of the training graph batch size.

Leaf path targets remain binary. At a non-leaf level, a positive target is the
number of active descendant leaf events on that directed cell pair divided by
the total number of real LR leaves. Thus the training target for an ancestor
uses the same normalized descendant sum as the exported ancestor score; graph
topology still uses union semantics to determine which edges participate in
message passing.

Each complete hierarchy-level graph, its normalized edge features, the CPU
incoming-neighbour indices for both CCC and modality graphs, and channel edge
indices are built once. Local graph batches slice these cached structures
without recomputing feature normalization. The multimodal representation is
recomputed for every local batch so that CCC losses fine-tune MEI; the
spatially eligible negative candidate pool is computed once before CCC
optimization.

`hierarchy_method` may be `balanced`, `biological`, or `llm`. The biological
method requires matching CellChat metadata; the LLM method requires its local
semantic model and configured service. Their bundled resources are listed in
the runtime-resources table above.

## 7. Postprocess leaf scores and aggregate the hierarchy

One public argument selects the output path:

```python
result = run_cellstic(
    ...,
    postprocess_before_aggregation=True,  # default
)
```

With `True`, each leaf channel is compared with its in-sample
expression-negative candidate scores. The primary leaf matrices retain scores
whose descriptive reference-tail proportion is at most
`config.postprocess.null_tail_fraction`; unfiltered model scores are exported
with the `_raw` suffix. The `_null_tail` output records that upper-tail
proportion. It is not a p-value and does not imply frequentist FDR control.
Optional MC-dropout stability is enabled only when
`config.postprocess.stability_repeats >= 2`.

With `False`, calibration is skipped and raw leaf predictions become the
primary leaf matrices. In both modes, every non-leaf channel is derived from
the selected leaf source, not independently predicted:

```text
ancestor_score(i, j)
    = sum(descendant_leaf_scores(i, j)) / total_number_of_real_LR_leaves
```

The common denominator preserves cumulative descendant evidence and keeps
scores comparable across hierarchy levels. Null-tail proportions and stability
frequencies are leaf-level quantities and are not aggregated upward.

The same tree-wide denominator is used when descendant ligand/receptor
strengths are combined for hierarchy-level training graphs. Binary graph
topology continues to use union semantics: an ancestor edge exists when at
least one descendant LR edge exists.

## 8. Export contract

`run_cellstic` returns an AnnData object and writes the same object to
`<output_path>/out/out.h5ad`.

```text
<output_path>/
├── checkpoint/
│   ├── feature.pt       # always written
│   └── ccc.pt           # written when run_ccc=True
└── out/
    └── out.h5ad         # always written
```

The CCC checkpoint contains a `trained` flag. `run_ccc=True` requires a
non-empty LR map and fails explicitly when that input is absent;
`run_ccc=False` omits `ccc.pt` entirely.

### `out.h5ad` fields

| Location | Format and shape | Meaning |
|---|---|---|
| `.X`, `.var`, `.obs` | First aligned modality | Primary assay retained from the input; preprocessing may update its values and annotations. |
| `.obsm["spatial"]` | `(n, 2)` | Aligned two-dimensional spatial coordinates. |
| `.obsm["feat"]` | `(n, d_0)` | Preprocessed first-modality model input. |
| `.obsm["out"]` | `(n, f)` | Integrated representation. |
| `.obs["cluster"]` | Length `n`, categorical | Unsupervised spatial-domain assignment; omitted when `skip_spatial_domain_prediction=True`. |
| `.obsp["cellstic_L{level}_lr_{channel}"]` | CSR matrix, `(n, n)` | Primary directed CCC score; rows are senders and columns are receivers. Diagonal entries are removed. |
| `.obsp["..._raw"]` | CSR matrix, `(n, n)` | Unfiltered leaf model score; present only for calibrated leaf output. |
| `.obsp["..._null_tail"]` | CSR matrix, `(n, n)` | Descriptive upper-tail proportion against the in-sample expression-negative reference; not a p-value; present only when postprocessing runs. |
| `.obsp["..._stability"]` | CSR matrix, `(n, n)` | Leaf MC-dropout selection frequency; present only when stability evaluation is enabled. |
| `.uns["cellstic"]` | Mapping of JSON strings and scalar metadata | Schema version, hierarchy, level/channel maps, saved configuration, checkpoint paths, training status, postprocessing, and aggregation provenance. |

Channel names are sanitized for AnnData keys by replacing colons, slashes, and
spaces with underscores. The original names and channel indices are preserved
in `.uns["cellstic"]["tree_levels_json"]`.

## 9. End-to-end use

```python
from model.train import get_cellstic_config
from paper.preprocess import cellstic_metadata, read_modalities
from pipeline import Analysis, run_cellstic

rna, atac = read_modalities("paper/data/ccc_sim/h5ad/domains_11", ["rna", "atac"])
lr_map, _ = cellstic_metadata(rna)
config = get_cellstic_config("scmultisim")
output_dir = "paper/data/ccc_sim/cellstic/tutorial_run"
result = run_cellstic(
    modality_datas=[rna, atac],
    config=config,
    output_path=output_dir,
    ligand_receptor_map=lr_map,
    modality_types=["rna", "atac"],
    run_ccc=True,
    postprocess_before_aggregation=True,
)

analysis = Analysis.from_result(result, output_path=output_dir)
```

For feature integration and domain detection without CCC inference, set
`run_ccc=False`; `ligand_receptor_map` is then unnecessary.

`Analysis.from_result` accepts the returned AnnData object or one loaded from
`out/out.h5ad` with `anndata.read_h5ad`. Plotting methods create `domain/`,
`ccc/`, or hierarchy figure directories only when those methods are called;
figure files are not automatic outputs of `run_cellstic`.

`analysis.plot_hierarchy_spatial_strength(level=2)` uses
`utils.visualization.HierarchySpatialVisualizer` to plot the saved L2 channels
in one row. Color shows `log1p(outgoing + incoming)` after removing self edges,
with one shared scale. It returns the SVG path and writes PNG, PDF and source
CSV files alongside it in `ccc/level2_spatial/`. Optional `channel_titles`
changes display labels; `threshold` filters weights before summation.

## Documentation map

- [Main README](README.md): project overview, installation, and documentation
  navigation.
- [Model card](MODEL_CARD.md): intended use, limitations, environment, and
  research-use considerations.
- `paper/data/<dataset>/README.md`: data sources, measurements, annotations,
  dataset statistics, and data-specific limitations.
- `paper/notebook/`: numbered dataset tutorials with training commented out by
  default and executed analysis outputs embedded in each notebook. Analyses
  reuse saved results and write figures and tables under each dataset's
  `cellstic/` output location, using the shared `domain/`, `ccc/`, and
  `alluvial/` folders. Temporal figures and source tables are grouped under
  `time/lr_counts/`, `time/celltype_strength/`, `time/network_metrics/`,
  `time/celltype_interactions/`, `time/samples/`, and `time/topology/` as applicable.
- `utils/analysis/`: result readers, simulation metrics, temporal statistics,
  and dataset renderers used by the tutorials. Existing plotting
  interfaces in `utils/visualization/` and `utils/metrics/` are reused directly;
  the dataset tutorials do not compute or display UMAP.
- `paper/preprocess/`: raw-to-AnnData assembly entry points.
