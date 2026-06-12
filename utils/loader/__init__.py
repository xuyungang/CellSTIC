"""
Data loaders: one function per dataset. Load + preprocess (no cell type annotation).
Usage: from utils.loader import load_human_lymph_node; rna, adt = load_human_lymph_node(Path("data/raw/..."))
"""

from .human_lymph_node import load_human_lymph_node
from .human_skin import load_human_skin
from .human_tonsil import load_human_tonsil
from .mouse_brain import load_mouse_brain, load_mouse_brain_gene_scores
from .mouse_embryo import load_mouse_embryo
from .nsf import load_nsf, load_true_labels
from .axolotl_telencephalon import load_axolotl_telencephalon

__all__ = [
    "load_human_lymph_node",
    "load_human_skin",
    "load_human_tonsil",
    "load_mouse_brain",
    "load_mouse_brain_gene_scores",
    "load_mouse_embryo",
    "load_nsf",
    "load_true_labels",
    "load_axolotl_telencephalon",
]
