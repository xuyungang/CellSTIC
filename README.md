# CellSTIC

Decoding Hierarchical Cell-Cell Communication in Spatial Multi-Omics with CellSTIC.
**Preprint:** [https://doi.org/10.64898/2026.05.27.728114](https://doi.org/10.64898/2026.05.27.728114)

<p align="center">
  <img src="assets/overview.png" alt="CellSTIC overview: linking in-situ tissue architecture to intercellular signaling programs" width="900"/>
</p>

## Method

CellSTIC integrates spatial multi-omics evidence, functional region segmentation, and a ligand–receptor semantics tree to infer hierarchical cell–cell communication (CCC).

<p align="center">
  <img src="assets/method_workflow.png" alt="CellSTIC workflow: data input, evidence integration, LR semantics tree building, and CCC application" width="900"/>
</p>

## Installation

```bash
conda env create -f environment.yml
conda activate cellstic
```

Run the tutorials from `notebook/`; each notebook adds the project root to `sys.path` automatically. **Start Jupyter with the repository root as the working directory.**

## Project Structure

| Directory | Description |
|-----------|-------------|
| `model/` | `CellSTIC`, HODGNN, graph construction, hierarchy tree |
| `pipeline/` | Training, evaluation, analysis, and `run_cellstic` entry point |
| `utils/` | Data preprocessing, loaders, analysis tools, metrics, viz |
| `config/` | Experiment YAML configs |
| `data/` | Raw and preprocessed data |
| `component/` | Simulators and component-level modules (e.g. synthetic spatial patterns) |
| `notebook/` | Step-by-step Jupyter tutorials (`.ipynb`) |
| `cache/` | Runtime cache artifacts generated during experiments |

## Dependencies

- Single-cell / spatial: `anndata`, `scanpy`, `celltypist`
- Deep learning: `torch`, `dgl`
- Graphs: `igraph`, `networkx`, `louvain`
- General: `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `PyYAML`, `h5py`, `tqdm`
- Embedding model files: download and place `bge-base-en-v1.5` manually at `component/bge-base-en-v1.5`
- Aliyun LLM config: either manually set `api_key`, `region`, and `base_url` in `config/aliyun_config.yaml` (section `aliyun`), or use the helper API below (do not commit real keys)

See `requirements.txt` and `environment.yml` for details.

## Quick Start

**Workflow**: load / preprocess `AnnData` → `run_cellstic` → `SingleLevelAnalysis.from_adata`. See `notebook/*.ipynb` and `data/<dataset>/README.md` for dataset-specific steps.

Most datasets use **`utils.loader`** (`load_mouse_embryo`, `load_mouse_brain`, `load_human_lymph_node`, …). **scMultiSim** reads `raw/rna.h5ad` and `raw/atac.h5ad` directly and preprocesses in [`notebook/scmultisim.ipynb`](notebook/scmultisim.ipynb).

Each modality needs `obsm["feat"]`, `obsm["spatial"]`, and (recommended) `obsp["spatial_distances"]`.

```python
from pathlib import Path
import torch
from pipeline import run_cellstic
from utils.loader import load_mouse_embryo
from utils.tools.seed_utils import set_global_seed

set_global_seed()

work_dir = Path("data/mouse_embryo/14.5")
rna, lr = load_mouse_embryo(work_dir / "raw", preprocess_path=work_dir / "preprocess" / "Brain")

result = run_cellstic(
    modality_datas=[rna],
    ligand_receptor_map=lr,
    model_path=work_dir / "model" / "Brain",
    output_path=work_dir / "output" / "Brain",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    auto_n_clusters=7,
)

from pipeline.analyzer import SingleLevelAnalysis

analysis = SingleLevelAnalysis.from_adata(result.adata, output_path=work_dir / "output" / "Brain")
analysis.run_cell_type_heatmaps()
```

For scMultiSim: multi-modal `[rna, atac]`, optional `config` / `lr_pair_type_constraints`, outputs under `result/` and `analysis/` — see the notebook.

**Analysis** (`pipeline.analyzer`): `SingleLevelAnalysis`, `TreeLevelAnalysis`, `TimeSequenceAnalysis`, `DomainAnalysis`.

**Low-level API** (optional): `build_config` → `CellSTICTrainer` → `CellSTICEvaluator`. Prefer `run_cellstic`.

### Aliyun LLM configuration (optional)

For Aliyun-based LLM tools (used only in optional helper utilities), edit `config/aliyun_config.yaml` or call `utils.tools.aliyun_utils.set_aliyun_config`. Not required for core experiments. Never commit real API keys.

## Tutorial notebooks

| Notebook | Dataset |
|----------|---------|
| `notebook/scmultisim.ipynb` | scMultiSim replicates re1–re8 |
| `notebook/mouse_embryo.ipynb` | Mouse embryo Stereo-seq |
| `notebook/mouse_brain.ipynb` | Mouse brain 5M |
| `notebook/human_lymph_node.ipynb` | Human lymph node |
| `notebook/axolotl_telencephalon.ipynb` | Axolotl telencephalon |

## Feedback

Bug reports, questions, and feature suggestions: please [open an issue](https://github.com/xuyungang/CellSTIC/issues) on GitHub.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
