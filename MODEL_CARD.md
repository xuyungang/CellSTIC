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

## 3. Input data

### Required inputs

CellSTIC operates on spatial molecular datasets represented as `AnnData` objects. For each modality, the recommended inputs include:

- `obsm["feat"]`: feature representation for cells or spots.
- `obsm["spatial"]`: spatial coordinates.
- `obsp["spatial_distances"]`: spatial distance matrix, recommended for spatial-neighborhood analysis.
- Ligand-receptor mapping information.

The framework can use one or more molecular modalities, depending on the available data and analysis design.

### Supported analysis contexts

The repository provides tutorial workflows for:

- scMultiSim simulated spatial datasets, including eight replicates (`re1`-`re8`).
- Mouse embryo Stereo-seq data.
- Mouse brain spatial data.
- Human lymph node spatial data.
- Axolotl telencephalon development data.
- Axolotl telencephalon regeneration data.

These tutorials serve as examples of the intended input structure and analysis workflow. Users should adapt preprocessing and parameter settings to their own data.

## 4. Outputs

CellSTIC produces inferred cell-cell communication results and supports multiple downstream analyses. Depending on the selected workflow, outputs may include:

- Inferred communication relationships between cell types, spatial regions, or biological entities.
- Integrated latent representations derived from spatial multi-omics information.
- Spatial communication patterns and visualizations.
- Hierarchical ligand-receptor semantic structures.
- Cell-type communication heatmaps.
- Functional-region or domain-level communication analyses.
- Time-sequence analyses for developmental or regeneration datasets.
- Model files, intermediate processed objects, result tables, and visualization outputs.

The repository provides analysis modules including `SingleLevelAnalysis`, `TreeLevelAnalysis`, `TimeSequenceAnalysis`, and `DomainAnalysis`.

## 5. Model workflow

A typical CellSTIC workflow consists of the following stages:

1. **Data loading and preprocessing.** Spatial multi-omics data are processed into the required `AnnData` structure. Feature representations, spatial coordinates, and spatial distance information are prepared.
2. **Graph construction and representation learning.** CellSTIC uses graph-based components to represent relationships among cells or spots, incorporating molecular features and spatial information.
3. **Ligand-receptor knowledge integration.** Ligand-receptor mappings are incorporated to support inference and interpretation of putative intercellular signaling relationships.
4. **Cell-cell communication inference.** The framework estimates communication patterns using integrated representations and biological prior knowledge.
5. **Hierarchical interpretation.** Inferred ligand-receptor communication signals are organized through a semantics tree to support interpretation at multiple biological levels.
6. **Downstream analysis and visualization.** Results can be analyzed at cell-type, spatial-domain, tree, or temporal levels through the provided analysis modules.

The recommended high-level programmatic entry point is:

```python
from pipeline import run_cellstic
```

Step-by-step workflows are provided in the repository's `notebook/` directory.

## 6. Training and evaluation

### Training setting

CellSTIC is applied to each input spatial dataset through the provided analysis workflow. The framework uses graph-based learning and is designed for dataset-specific analysis rather than a fixed supervised model trained once and deployed unchanged across all datasets.

The workflow includes self-supervised graph-learning components. Therefore, conventional train-validation-test splits may not always apply in the same way as they do for supervised prediction tasks.

### Evaluation setting

The repository includes tutorial workflows for simulated and publicly available spatial datasets. In particular, scMultiSim tutorials include eight independently generated replicates, enabling repeated evaluation under simulated conditions.

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

The full environment, including all dependencies and version information, is available at:

https://github.com/xuyungang/CellSTIC/blob/main/environment.yml

### Hardware

Tutorial analyses were run on a workstation equipped with one NVIDIA GeForce RTX 4090 GPU, a 128-core CPU, and 128 GB RAM. This configuration is reported for reproducibility and does not represent the minimum hardware requirement. Runtime and memory requirements depend on dataset size, number of modalities, graph size, and selected analysis steps.

## 8. Installation and use

Create the recommended software environment with:

```bash
conda env create -f environment.yml
conda activate cellstic
```

Tutorial notebooks are available in the `notebook/` directory. Jupyter should be launched with the repository root as the working directory.

A typical workflow is:

```text
Load and preprocess input AnnData objects
        ↓
Prepare feature and spatial representations
        ↓
Run run_cellstic
        ↓
Use the provided analysis modules for interpretation and visualization
```

For detailed usage instructions, refer to https://github.com/xuyungang/CellSTIC.

## 9. Ethical considerations

CellSTIC is a research software framework. When applied to human data, users are responsible for ensuring that data access, use, sharing, and analysis comply with relevant ethical approvals, participant-consent requirements, institutional policies, and data-governance rules.

The software should not be used to make clinical decisions or infer sensitive individual-level attributes without appropriate validation, ethical review, and regulatory oversight.

## 10. Availability and license

The complete source code, documentation, tutorial notebooks, and reproducible environment files are publicly available at:

https://github.com/xuyungang/CellSTIC

CellSTIC is distributed under the GNU General Public License v3.0.

## 11. Citation

If you use CellSTIC, please cite the associated preprint:

https://doi.org/10.64898/2026.05.27.728114

## 12. Contact and feedback

For bug reports, questions, feature requests, or suggestions, please open an issue in the GitHub repository:

https://github.com/xuyungang/CellSTIC/issues
