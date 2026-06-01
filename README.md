# CellSTIC

Decoding Hierarchical Cell-Cell Communication in Spatial Multi-Omics with CellSTIC.
**Preprint:** [https://doi.org/10.64898/2026.05.27.728114](https://doi.org/10.64898/2026.05.27.728114)

## Installation

```bash
conda env create -f environment.yml
conda activate cellstic
```

Run the tutorials from `notebook/`; each notebook adds the project root to `sys.path` automatically.

## Project Structure

| Directory | Description |
|-----------|-------------|
| `model/` | `CellSTIC`, HODGNN, graph construction, hierarchy tree |
| `pipeline/` | Training, evaluation, analysis, visualization entry points |
| `utils/` | Data preprocessing, loaders, analysis tools, metrics, viz |
| `config/` | Experiment YAML configs |
| `data/` | Raw and preprocessed data |
| `component/` | Simulators and component-level modules (e.g. synthetic spatial patterns) |
| `notebook/` | tep-by-step Jupyter tutorials (`.ipynb`);|
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

**Required workflow**: (1) load `AnnData` with `utils.loader`; (2) call `generate_config` to write YAML from that list.

For step (1), use `utils.loader` (below: `load_mouse_embryo`; also `load_mouse_brain`, `load_human_lymph_node`, `load_human_tonsil`, `load_human_skin`). Point each loader at a dataset root under `data/<dataset>/...` with `raw/` (and optionally `preprocess/`). See `notebook/*.ipynb` and each `data/<dataset>/README.md` for input data sources.

```python
from pathlib import Path
from utils.loader import load_mouse_embryo
from utils.train import generate_config

# Example: mouse embryo E14.5, Brain region (same layout as notebook/mouse_embryo.ipynb)
work_dir = Path("data/mouse_embryo/14.5")
raw_dir = work_dir / "raw"
preprocess_dir = work_dir / "preprocess" / "Brain"
config_path = work_dir / "config" / "mouse_embryos_config.yaml"

# 1) Load data
rna, lr = load_mouse_embryo(raw_dir, preprocess_path=preprocess_dir)

# 2) Generate config from loaded AnnData
generate_config(
    config_path=config_path,
    adatas=[rna],                 # or [rna, adt, atac, ...]
    hierarchy_method="balanced",  # required
)
```

**Training and evaluation**: `load_config` → `CellSTIC` → `CellSTICTrainer.train` → `CellSTICEvaluator`. Reuse `rna` / `lr` and `work_dir` from the snippet above; or open and run `notebook/mouse_embryo.ipynb` (Jupyter, cwd = repo root). Example YAMLs: `config/`.

```python
from pathlib import Path
import torch
from model.cellstic import CellSTIC
from pipeline.trainer import CellSTICTrainer
from pipeline.evaluator import CellSTICEvaluator
from utils.train import load_config

work_dir = Path("data/mouse_embryo/14.5")
config = load_config(work_dir / "config" / "mouse_embryos_config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils.tools.seed_utils import set_global_seed

set_global_seed()  # once before train/eval; pass int to set base (default 42 until set)
model_path = work_dir / "model" / "Brain"
output_path = work_dir / "output" / "Brain"
# rna, lr = load_mouse_embryo(...)  # same as Quick Start

model = CellSTIC(config.model, device)
CellSTICTrainer(model, config, model_path=model_path, device=device, ligand_receptor_map=lr).train(
    modality_datas=[rna], is_train_ccc=True, is_train_feature=True,
)
evaluator = CellSTICEvaluator(model, config, ligand_receptor_map=lr, model_path=model_path, output_path=output_path, device=device)
evaluator.evaluate_mutiple_feature(modality_datas=[rna], auto_n_clusters=7)

# CCC evaluation (single level / last level)
pos_edge_probs_np, edge_type_map, adata_m0 = evaluator.evaluate_ccc_precision(
    modality_datas=[rna],
)

# CCC evaluation (tree levels)
# Returns a list of per-level dicts; each contains "pos_edge_probs_np", "edge_type_map", and "output_dir".
tree_results = evaluator.evaluate_ccc_precision_tree(
    modality_datas=[rna],
)
```

**Analysis** (`pipeline.analyzer`): Use the evaluator outputs to run different analysis entry points.

- `SingleLevelAnalysis`: heatmaps, LR spatial plots, distance/strength summaries, etc. (uses `evaluate_ccc_precision` outputs)
- `TreeLevelAnalysis`: hierarchical CCC views across tree levels (uses `evaluate_ccc_precision_tree` outputs)
- `TimeSequenceAnalysis`: summarize CCC patterns across stages / time points (run after collecting outputs per stage)
- `DomainAnalysis`: compare CCC across spatial domains (requires domain CSV annotations)

See `notebook/*.ipynb` for end-to-end calls.

```python
from pipeline.analyzer import SingleLevelAnalysis

# Uses pos_edge_probs_np, edge_type_map, adata_m0 from evaluate_ccc_precision above
analysis = SingleLevelAnalysis(
    adata_m0,
    pos_edge_probs_np,
    edge_type_map,
    output_path=output_path,
)
analysis.run_cell_type_heatmaps()  # e.g.; also run_strength_vs_distance, run_lr_spatial, ...
```

```python
from pipeline.analyzer import TreeLevelAnalysis

# Uses tree_results from evaluate_ccc_precision_tree above
tree_analysis = TreeLevelAnalysis(tree_results, output_path=output_path)
tree_analysis.run()
```

### Aliyun LLM configuration (optional)

For Aliyun-based LLM tools (used only in optional helper utilities), you can either edit `config/aliyun_config.yaml` by hand:

```yaml
aliyun:
  api_key: "YOUR_API_KEY"
  region: "YOUR_REGION"
  base_url: "YOUR_BASE_URL"
```

or set it programmatically via:

```python
from utils.tools.aliyun_utils import set_aliyun_config

set_aliyun_config(
    api_key="YOUR_API_KEY",
    region="YOUR_REGION",
    base_url="YOUR_BASE_URL",
)  # writes/updates config/aliyun_config.yaml
```

These LLM-related configs are **not required** for reproducing the core CellSTIC experiments; they are used only by optional tooling. Never commit real API keys.

## Tutorial notebooks (CellSTIC)

Step-by-step Jupyter guides for each dataset. **Start Jupyter with the repository root as the working directory.**

| Notebook | Dataset |
|----------|---------|
| `notebook/mouse_embryo.ipynb` | Mouse embryo Stereo-seq |
| `notebook/mouse_brain.ipynb` | Mouse brain 5M |
| `notebook/human_lymph_node.ipynb` | Human lymph node |
| `notebook/scmultisim.ipynb` | scMultiSim replicates |
| `notebook/axolotl_telencephalon.ipynb` | Axolotl telencephalon |

Tutorials are edited under `notebook/` (paths above).

## Feedback

Bug reports, questions, and feature suggestions: please [open an issue](https://github.com/xuyungang/CellSTIC/issues) on GitHub.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
