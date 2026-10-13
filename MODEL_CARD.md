# Model Card: CellSTIC

## 1. Model details

### Model name

**CellSTIC**  
Decoding Hierarchical Cell-Cell Communication in Spatial Multi-Omics with CellSTIC

### Model type

CellSTIC is a graph-based computational framework for analysis of cell-cell communication (CCC) in spatial multi-omics data. It combines graph representation learning, spatial information, ligand-receptor knowledge, functional-region analysis, and a ligand-receptor semantics tree to infer hierarchical communication programs.

### Brief description

CellSTIC is designed to identify and interpret cell-cell communication in spatially resolved biological tissues. Rather than treating ligand-receptor interactions only as isolated pairwise relationships, CellSTIC organizes inferred communication signals into hierarchical and biologically interpretable functional modules.

The framework integrates:

- Spatial coordinates and spatial-neighborhood information.
- Feature representations derived from one or more molecular modalities.
- Ligand-receptor interaction resources.
- Graph-based representation learning.
- Functional-region segmentation and downstream spatial analyses.
- A ligand-receptor semantics tree for hierarchical interpretation of communication signals.

CellSTIC is intended to facilitate exploratory and comparative analysis of spatial cell-cell communication. Its results should be interpreted as computational hypotheses and independently validated where biological or clinical conclusions are intended.

## 2. Intended use

### Primary intended uses

CellSTIC is intended for research applications including:

- Inferring cell-cell communication from spatial transcriptomics or spatial multi-omics data.
- Identifying spatially organized ligand-receptor signaling patterns.
- Comparing communication patterns across cell types, tissue regions, developmental stages, or experimental conditions.
- Studying hierarchical functional relationships among inferred ligand-receptor communication events.
- Evaluating computational performance on simulated spatial datasets with known ground truth.
- Supporting downstream visualization and interpretation of cell-cell communication results.

### Out-of-scope uses

CellSTIC should not be used as:

- A clinical diagnostic tool.
- A method for selecting or prioritizing patient treatment.
- Direct evidence of causal cell-cell communication.
- A substitute for experimental validation of ligand-receptor interactions.
- A validated predictor of disease outcome, therapeutic response, or patient risk.

## 3. Input scope

CellSTIC operates on spatial molecular measurements with aligned cell or spot
identifiers, spatial coordinates, and feature identifiers. CCC inference also
requires a declared ligand-receptor candidate set. The framework can use one
or more molecular modalities depending on the study design.

This model card intentionally does not define file schemas or API arguments.
The authoritative input fields, dimensions, validation rules, and hand-offs are
specified in [PIPELINE.md](PIPELINE.md).

### Supported analysis contexts

The repository provides tutorial workflows for:

- Simulated spatial domains from aligned RNA, ADT, and ATAC.
- scMultiSim-derived CCC benchmark datasets across multiple scenarios and seeds.
- Mouse embryo Stereo-seq data.
- E12.5 mouse embryo Visium HD data and adjacent-section PLA images.
- Mouse brain spatial data.
- Human lymph node spatial data.
- Human skin and tonsil spatial RNA–ADT data.
- Axolotl telencephalon development data.
- Axolotl telencephalon regeneration data.

These tutorials serve as examples of the intended input structure and analysis workflow. Users should adapt preprocessing and parameter settings to their own data.

## 4. Output interpretation

CellSTIC produces computationally inferred communication scores and integrated
spatial representations. Depending on the selected workflow, these may support:

- Inferred communication relationships between cell types, spatial regions, or biological entities.
- Integrated latent representations derived from spatial multi-omics information.
- Spatial communication patterns and visualizations.
- Hierarchical ligand-receptor semantic structures.
- Cell-type communication heatmaps.
- Functional-region or domain-level communication analyses.
- Time-sequence analyses for developmental or regeneration datasets.
- Downstream exploratory tables and visualizations.

The exact AnnData fields, sparse-matrix orientation, generated files, and
checkpoint contract are defined only in [PIPELINE.md](PIPELINE.md).

## 5. Model and learning scope

CellSTIC combines modality-specific graph encoders, full-dimensional feature
reconstruction, multimodal fusion, spatially restricted ligand-receptor graph
learning, and a ligand-receptor hierarchy. Feature learning and CCC fitting
are unsupervised and dataset-specific. Non-leaf CCC outputs are derived from
leaf predictions through a declared hierarchy rather than treated as
independent biological observations.

Algorithmic step order, local graph batching, loss inputs, and hierarchy
aggregation are operational definitions and are maintained in
[PIPELINE.md](PIPELINE.md).

## 6. Training and evaluation

### Training setting

CellSTIC is applied to each input spatial dataset through the provided analysis workflow. The framework uses graph-based learning and is designed for dataset-specific analysis rather than a fixed supervised model trained once and deployed unchanged across all datasets.

The workflow includes self-supervised graph-learning components. Therefore, conventional train-validation-test splits may not always apply in the same way as they do for supervised prediction tasks.

### Evaluation setting

The repository includes tutorial workflows for simulated and publicly available spatial datasets. The scMultiSim-derived CCC benchmark includes multiple independently seeded datasets across distinct scenarios, enabling repeated evaluation under simulated conditions.

Evaluation should use metrics appropriate to the specific task. Depending on the experiment, these may include measures of communication-inference performance, region-identification performance, clustering agreement, or correspondence with known simulated ground truth.

### Reproducibility

The repository provides source code, tutorial notebooks, configuration files, a Conda environment specification, dataset-specific README files, and a utility for setting global random seeds.

Users should record the exact repository version, environment specification, input-data version, preprocessing procedure, parameter configuration, and random seed used for each analysis.

## 7. Software environment

CellSTIC was implemented in Python 3.10. Its core software environment includes:

- PyTorch 2.1.0
- CUDA 11.8
- Scanpy 1.9.6
- AnnData 0.9.2
- DGL 1.1.2+cu118
- NumPy 1.24.4
- pandas 2.0.3
- SciPy 1.10.1
- scikit-learn 1.3.2
- NetworkX 3.1
- python-igraph 0.11.3

The reference versions above describe the implementation environment. The
repository's [environment.yml](environment.yml) provides dependency ranges for
installation; it is not a lockfile of those exact versions. Record the resolved
package versions for each analysis.

### Hardware

Tutorial analyses were run on a workstation equipped with one NVIDIA GeForce RTX 4090 GPU, a 128-core CPU, and 128 GB RAM. This configuration is reported for reproducibility and does not represent the minimum hardware requirement. Runtime and memory requirements depend on dataset size, number of modalities, graph size, and selected analysis steps.

## 8. Operational documentation

Installation and the documentation map are provided in [README.md](README.md).
The complete usage, configuration, input/output, and artifact specification is
maintained in [PIPELINE.md](PIPELINE.md). Tutorial notebooks and dataset scripts
illustrate dataset-specific applications but do not override that shared
contract.

## 9. Ethical considerations

CellSTIC is a research software framework. When applied to human data, users are responsible for ensuring that data access, use, sharing, and analysis comply with relevant ethical approvals, participant-consent requirements, institutional policies, and data-governance rules.

The software should not be used to make clinical decisions or infer sensitive individual-level attributes without appropriate validation, ethical review, and regulatory oversight.

## 10. Availability and license

The complete source code, documentation, tutorial notebooks, and reproducible environment files are publicly available at:

https://github.com/xuyungang/CellSTIC

CellSTIC source code is distributed under the GNU General Public License v3.0.
Third-party database snapshots and pretrained models have separate provenance
and license scope, documented in the [CellChatDB resource
notice](resources/cellchatdb/README.md) and [CellTypist model
notice](resources/celltypist/README.md).

## 11. Citation

If you use CellSTIC, please cite the associated preprint:

https://doi.org/10.64898/2026.05.27.728114

## 12. Contact and feedback

For bug reports, questions, feature requests, or suggestions, please open an issue in the GitHub repository:

https://github.com/xuyungang/CellSTIC/issues
