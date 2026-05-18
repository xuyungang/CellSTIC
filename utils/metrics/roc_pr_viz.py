import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from pathlib import Path
from typing import Optional

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


class MetricsCurveVisualizer:
    """Utility to draw ROC and PR curves from predictions and labels."""

    @staticmethod
    def plot_roc_pr_combined(pos_edge_probs, label, save_path: str = "roc_pr.png", 
                             title: str = None, save_data_path: Optional[str] = None) -> None:
        # Convert tensors to numpy arrays if necessary
        if isinstance(pos_edge_probs, torch.Tensor):
            pos_edge_probs = pos_edge_probs.detach().cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()

        # Convert to numpy arrays and ensure they have the same shape
        pos_edge_probs = np.asarray(pos_edge_probs)
        label = np.asarray(label)
        
        # Check and ensure dimensions match
        if pos_edge_probs.shape != label.shape:
            raise ValueError(
                f"Shape mismatch: pos_edge_probs shape {pos_edge_probs.shape} != label shape {label.shape}"
            )
        # Flatten to 1D (allow already 1D from evaluator when using eval_mask)
        pos_edge_probs = pos_edge_probs.reshape(-1)
        label = label.reshape(-1)

        # Safety: remove NaNs/Infs
        valid_mask = np.isfinite(pos_edge_probs) & np.isfinite(label)
        pos_edge_probs = pos_edge_probs[valid_mask]
        label = label[valid_mask]

        # Compute ROC
        fpr, tpr, _ = roc_curve(label, pos_edge_probs)
        roc_auc = auc(fpr, tpr)

        # Compute PR
        precision, recall, _ = precision_recall_curve(label, pos_edge_probs)
        ap = average_precision_score(label, pos_edge_probs)

        # Plot combined figure with improved styling
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=600, gridspec_kw={'wspace': 0.15})
        fig.patch.set_facecolor('white')
        
        # Define color scheme
        roc_color = '#2E86AB'  # Dark blue
        pr_color = '#A23B72'   # Deep magenta
        baseline_color = '#6C757D'  # Gray
        fill_alpha = 0.2

        # ROC subplot
        ax = axes[0]
        # Draw diagonal line (random classifier baseline)
        ax.plot([0, 1], [0, 1], color=baseline_color, lw=2, ls="--", 
                label="Random Classifier (AUC = 0.500)", alpha=0.7)
        # Plot ROC curve with thicker line and better color
        ax.plot(fpr, tpr, color=roc_color, lw=3, label=f"ROC Curve (AUC = {roc_auc:.4f})", 
                alpha=0.9, zorder=3)
        # Fill area under the curve
        ax.fill_between(fpr, tpr, alpha=fill_alpha, color=roc_color, zorder=2)
        
        # Beautify axes
        ax.set_xlabel("False Positive Rate", fontsize=12, fontweight='bold')
        ax.set_ylabel("True Positive Rate", fontsize=12, fontweight='bold')
        ax.set_title("ROC Curve", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        # Add border
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')

        # PR subplot
        ax = axes[1]
        # Calculate baseline (positive sample ratio, i.e., AP of random classifier)
        baseline_precision = np.sum(label) / len(label) if len(label) > 0 else 0
        # Draw baseline
        ax.axhline(y=baseline_precision, color=baseline_color, lw=2, ls="--", 
                  label=f"Random Classifier (AP = {baseline_precision:.4f})", alpha=0.7)
        # Plot PR curve
        ax.plot(recall, precision, color=pr_color, lw=3,
                label=f"PR Curve (AP = {ap:.4f})", alpha=0.9, zorder=3)
        # Fill area under the curve
        ax.fill_between(recall, precision, alpha=fill_alpha, color=pr_color, zorder=2)
        
        # Beautify axes
        ax.set_xlabel("Recall", fontsize=12, fontweight='bold')
        ax.set_ylabel("Precision", fontsize=12, fontweight='bold')
        ax.set_title("Precision-Recall Curve", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="lower left", fontsize=10, framealpha=0.9, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        # Add border
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')

        # Add overall title (if provided)
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        # Adjust layout and save
        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=600, bbox_inches="tight", facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"ROC + PR combined curve saved to {save_path}")
        
        # Save plotting data for reproduction
        if save_data_path is None:
            # Auto-generate data path from image path
            save_path_obj = Path(save_path)
            save_data_path = save_path_obj.parent / f"{save_path_obj.stem}_data.npz"
        
        # Prepare data dictionary
        plot_data = {
            'fpr': fpr,
            'tpr': tpr,
            'recall': recall,
            'precision': precision,
            'roc_auc': np.array([roc_auc]),
            'ap': np.array([ap]),
            'baseline_precision': np.array([baseline_precision]),
            'pos_edge_probs': pos_edge_probs,  # Final processed probabilities
            'label': label,  # Final processed labels
        }
        
        # Save data
        np.savez_compressed(save_data_path, **plot_data)
        print(f"Plotting data saved to {save_data_path} for reproduction")
