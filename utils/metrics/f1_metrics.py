"""
F1-score metrics for CCC evaluation: Macro-F1, Macro-Accuracy, per-LR-pair F1/ACC/AUROC/AUPRC.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


class F1MetricsComputer:
    """Compute Macro-F1 and Macro-Accuracy for CCC edge prediction."""

    @staticmethod
    def compute_f1_summary(
        pos_edge_probs: Union[torch.Tensor, np.ndarray],
        label: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        lr_pair_names: Optional[list] = None,
        zero_division: Union[str, float] = 0,
        eval_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute Macro-F1 and Macro-Accuracy from predicted probabilities and binary labels.
        If eval_mask is provided (same shape as probs/label), only mask==True entries are used.
        """
        if isinstance(pos_edge_probs, torch.Tensor):
            pos_edge_probs = pos_edge_probs.detach().cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()
        pos_edge_probs = np.asarray(pos_edge_probs, dtype=np.float64)
        label = np.asarray(label, dtype=np.float64)

        if pos_edge_probs.shape != label.shape or pos_edge_probs.ndim != 3:
            raise ValueError(
                f"Expected same 3D shape (n, n, m); got probs {pos_edge_probs.shape}, label {label.shape}"
            )

        n, _, m = pos_edge_probs.shape
        valid = np.isfinite(pos_edge_probs) & np.isfinite(label)
        if eval_mask is not None and eval_mask.shape == pos_edge_probs.shape:
            valid = valid & eval_mask
        y_true = label.reshape(-1, m)
        y_prob = pos_edge_probs.reshape(-1, m)
        valid_flat = valid.reshape(-1, m)

        y_pred = (y_prob >= threshold).astype(np.float64)
        macro_f1s = []
        macro_accuracies = []

        for k in range(m):
            v = valid_flat[:, k]
            t = y_true[v, k]
            p = y_pred[v, k].astype(np.float64)
            if t.size == 0:
                macro_f1s.append(0.0)
                macro_accuracies.append(0.0)
                continue
            tp = ((p == 1) & (t == 1)).sum()
            fp = ((p == 1) & (t == 0)).sum()
            fn = ((p == 0) & (t == 1)).sum()
            tn = ((p == 0) & (t == 0)).sum()
            if tp + fp + fn > 0:
                f1_k = 2 * tp / (2 * tp + fp + fn)
            else:
                f1_k = 0.0 if zero_division == 0 else float(zero_division)
            total_k = tp + tn + fp + fn
            acc_k = (tp + tn) / total_k if total_k > 0 else (0.0 if zero_division == 0 else float(zero_division))
            macro_f1s.append(float(f1_k))
            macro_accuracies.append(float(acc_k))

        macro_f1 = float(np.mean(macro_f1s)) if macro_f1s else 0.0
        macro_accuracy = float(np.mean(macro_accuracies)) if macro_accuracies else 0.0

        return {
            "macro_f1": macro_f1,
            "macro_accuracy": macro_accuracy,
        }

    @staticmethod
    def find_best_threshold(
        pos_edge_probs: Union[torch.Tensor, np.ndarray],
        label: Union[torch.Tensor, np.ndarray],
        eval_mask: Optional[np.ndarray] = None,
        n_thresholds: int = 13,
        threshold_min: float = 0.2,
        threshold_max: float = 0.8,
    ) -> tuple:
        """Find threshold that maximizes macro F1. Returns (best_threshold, summary_at_best)."""
        if isinstance(pos_edge_probs, torch.Tensor):
            pos_edge_probs = pos_edge_probs.detach().cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()
        thresholds = np.linspace(threshold_min, threshold_max, n_thresholds)
        best_threshold = 0.5
        best_value = -1.0
        best_summary: Optional[Dict] = None
        for t in thresholds:
            s = F1MetricsComputer.compute_f1_summary(
                pos_edge_probs, label, threshold=float(t), eval_mask=eval_mask
            )
            if s["macro_f1"] > best_value:
                best_value = s["macro_f1"]
                best_threshold = float(t)
                best_summary = s
        if best_summary is None:
            best_summary = F1MetricsComputer.compute_f1_summary(
                pos_edge_probs, label, threshold=0.5, eval_mask=eval_mask
            )
        return best_threshold, best_summary

    @staticmethod
    def compute_per_class_metrics(
        pos_edge_probs: Union[torch.Tensor, np.ndarray],
        label: Union[torch.Tensor, np.ndarray],
        threshold: float,
        lr_pair_names: Optional[list] = None,
        eval_mask: Optional[np.ndarray] = None,
        zero_division: Union[str, float] = 0,
    ) -> List[Dict]:
        """
        Compute per-LR-pair F1, accuracy (with given threshold), AUROC, AUPRC.
        Returns list of dicts: [{"lr_pair": "101:2", "f1": 0.7, "accuracy": 0.85, "auroc": 0.9, "auprc": 0.8}, ...]
        """
        if isinstance(pos_edge_probs, torch.Tensor):
            pos_edge_probs = pos_edge_probs.detach().cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()
        pos_edge_probs = np.asarray(pos_edge_probs, dtype=np.float64)
        label = np.asarray(label, dtype=np.float64)

        if pos_edge_probs.shape != label.shape or pos_edge_probs.ndim != 3:
            raise ValueError(
                f"Expected same 3D shape (n, n, m); got probs {pos_edge_probs.shape}, label {label.shape}"
            )

        n, _, m = pos_edge_probs.shape
        valid = np.isfinite(pos_edge_probs) & np.isfinite(label)
        if eval_mask is not None and eval_mask.shape == pos_edge_probs.shape:
            valid = valid & eval_mask
        y_true = label.reshape(-1, m)
        y_prob = pos_edge_probs.reshape(-1, m)
        valid_flat = valid.reshape(-1, m)

        y_pred = (y_prob >= threshold).astype(np.float64)

        per_class_rows = []
        for k in range(m):
            v = valid_flat[:, k]
            t = y_true[v, k]
            p = y_pred[v, k].astype(np.float64)
            pk = y_prob[v, k]
            lr_name = lr_pair_names[k] if lr_pair_names and k < len(lr_pair_names) else str(k)

            # F1 & Accuracy
            if t.size == 0:
                f1_k, acc_k = 0.0, 0.0
            else:
                tp = ((p == 1) & (t == 1)).sum()
                fp = ((p == 1) & (t == 0)).sum()
                fn = ((p == 0) & (t == 1)).sum()
                tn = ((p == 0) & (t == 0)).sum()
                if tp + fp + fn > 0:
                    f1_k = float(2 * tp / (2 * tp + fp + fn))
                else:
                    f1_k = 0.0 if zero_division == 0 else float(zero_division)
                total_k = tp + tn + fp + fn
                acc_k = float((tp + tn) / total_k) if total_k > 0 else (0.0 if zero_division == 0 else float(zero_division))

            # AUROC & AUPRC (threshold-independent)
            valid_k = np.isfinite(pk) & np.isfinite(t)
            if valid_k.sum() > 0 and t[valid_k].sum() > 0 and t[valid_k].sum() < valid_k.sum():
                auroc_k = float(roc_auc_score(t[valid_k], pk[valid_k]))
                auprc_k = float(average_precision_score(t[valid_k], pk[valid_k]))
            else:
                auroc_k = np.nan
                auprc_k = np.nan

            per_class_rows.append({
                "lr_pair": lr_name,
                "f1": f1_k,
                "accuracy": acc_k,
                "auroc": auroc_k,
                "auprc": auprc_k,
            })

        return per_class_rows

    @staticmethod
    def compute_and_save(
        pos_edge_probs: Union[torch.Tensor, np.ndarray],
        label: Union[torch.Tensor, np.ndarray],
        save_dir: Union[str, Path],
        threshold: float = 0.5,
        lr_pair_names: Optional[list] = None,
        title: Optional[str] = None,
        eval_mask: Optional[np.ndarray] = None,
        search_optimal_threshold: bool = True,
    ) -> Dict[str, float]:
        """
        Compute Macro-F1 and Macro-Accuracy, save to save_dir/metrics/f1_summary.csv.
        Optionally finds best threshold. Returns summary dict and per-class rows.
        """
        save_dir = Path(save_dir)
        metrics_dir = save_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        summary = F1MetricsComputer.compute_f1_summary(
            pos_edge_probs, label, threshold=threshold, lr_pair_names=lr_pair_names,
            eval_mask=eval_mask,
        )

        row = {
            "macro_f1": summary["macro_f1"],
            "macro_accuracy": summary["macro_accuracy"],
            "threshold": threshold,
        }
        if title is not None:
            row["name"] = title

        thr_for_per_class = threshold
        if search_optimal_threshold:
            best_thr, best_summary = F1MetricsComputer.find_best_threshold(
                pos_edge_probs, label, eval_mask=eval_mask
            )
            row["best_threshold"] = best_thr
            row["macro_f1_best"] = best_summary["macro_f1"]
            row["macro_accuracy_best"] = best_summary["macro_accuracy"]
            summary["best_threshold"] = best_thr
            summary["macro_f1_best"] = best_summary["macro_f1"]
            summary["macro_accuracy_best"] = best_summary["macro_accuracy"]
            thr_for_per_class = best_thr

        # Per-LR-pair: F1, ACC (with best/fixed threshold), AUROC, AUPRC
        per_class_rows = F1MetricsComputer.compute_per_class_metrics(
            pos_edge_probs, label,
            threshold=thr_for_per_class,
            lr_pair_names=lr_pair_names,
            eval_mask=eval_mask,
        )
        row["macro_auroc"] = float(np.nanmean([r["auroc"] for r in per_class_rows]))
        row["macro_auprc"] = float(np.nanmean([r["auprc"] for r in per_class_rows]))
        summary["macro_auroc"] = row["macro_auroc"]
        summary["macro_auprc"] = row["macro_auprc"]
        pd.DataFrame(per_class_rows).to_csv(metrics_dir / "f1_per_class.csv", index=False)
        pd.DataFrame([row]).to_csv(metrics_dir / "f1_summary.csv", index=False)

        return {"summary": summary, "per_class_rows": per_class_rows}
