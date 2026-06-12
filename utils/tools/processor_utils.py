"""
Spatial preprocessing utilities
This module provides a reusable SpatialPreprocessorUtils that includes:
"""

from sklearn.decomposition import PCA
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import numpy as np
import sklearn.preprocessing
import sklearn.utils.extmath
import scipy.sparse
import anndata
from typing import Optional


class SpatialPreprocessorUtils:
    """
    Spatial data preprocessor utilities.
    """

    @staticmethod
    def pca(adata, use_reps=None, n_comps=10):
        """
        Dimension reduction with PCA algorithm.

        Args:
            adata: AnnData object.
            use_reps: Name of the representation matrix in adata.obsm.
            n_comps: Number of components to keep.
        Returns:
            feat_pca: PCA-transformed features.
        """
        pca = PCA(n_components=n_comps)
        if use_reps is not None:
            data = adata.obsm[use_reps]
        else:
            if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
                data = adata.X.toarray()
            else:
                data = np.array(adata.X)

        n_samples, n_features = data.shape
        # Ensure n_components is valid for sklearn PCA
        max_components = max(1, min(n_samples, n_features))
        n_components = min(n_comps, max_components)
        pca = PCA(n_components=n_components)
        
        # Ensure data is finite (no NaN or inf)
        if np.isnan(data).any() or np.isinf(data).any():
            print(f"  Warning: Found NaN/inf in PCA input, cleaning...")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            if np.isnan(data).any() or np.isinf(data).any():
                raise ValueError("Data contains NaN or inf values that cannot be cleaned for PCA")
        
        feat_pca = pca.fit_transform(data)

        # If downstream code assumes a fixed feature dimension n_comps,
        # pad with zeros when n_components < n_comps to keep the shape consistent.
        if feat_pca.shape[1] < n_comps:
            pad_width = n_comps - feat_pca.shape[1]
            feat_pca = np.pad(feat_pca, ((0, 0), (0, pad_width)))

        return feat_pca

    @staticmethod
    def lsi(
            adata: anndata.AnnData, n_components: int = 20,
            use_highly_variable: Optional[bool] = None, **kwargs
        ) -> None:
        r"""
        LSI analysis (following the Seurat v3 approach)

        Args:
            adata: AnnData object
            n_components: Number of components to keep
            use_highly_variable: Whether to use highly variable genes
            kwargs: Additional arguments for LSI
        Returns:
            None
        """
        print("Performing LSI analysis...")
        if use_highly_variable is None:
            use_highly_variable = "highly_variable" in adata.var
        adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
        X = SpatialPreprocessorUtils.tfidf(adata_use.X)
        #X = adata_use.X
        X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
        X_norm = np.log1p(X_norm * 1e4)
        X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
        X_lsi -= X_lsi.mean(axis=1, keepdims=True)
        # Avoid division by zero: if std is 0, set to 1 to avoid NaN
        std = X_lsi.std(axis=1, ddof=1, keepdims=True)
        std = np.where(std == 0, np.ones_like(std), std)
        X_lsi /= std
        adata.obsm["X_lsi"] = X_lsi[:,1:]

    @staticmethod
    def tfidf(X):
        """
        TF-IDF normalization (following the Seurat v3 approach)
        
        Args:
            X: Feature matrix
        Returns:
            tfidf: TF-IDF-normalized feature matrix
        """
        # Calculate IDF: avoid division by zero
        feature_sums = X.sum(axis=0)
        if scipy.sparse.issparse(X):
            feature_sums = np.array(feature_sums).flatten()
        feature_sums = np.where(feature_sums == 0, np.ones_like(feature_sums), feature_sums)
        idf = X.shape[0] / feature_sums
        
        if scipy.sparse.issparse(X):
            # For sparse matrices: calculate TF and multiply by IDF
            cell_sums = np.array(X.sum(axis=1)).flatten()
            cell_sums = np.where(cell_sums == 0, np.ones_like(cell_sums), cell_sums)
            # Use multiply for row-wise division (TF)
            tf = X.multiply(1.0 / cell_sums[:, np.newaxis])
            # Multiply by IDF (column-wise)
            idf_sparse = scipy.sparse.diags(idf, format=X.format)
            return tf @ idf_sparse
        else:
            # For dense matrices
            cell_sums = X.sum(axis=1, keepdims=True)
            cell_sums = np.where(cell_sums == 0, np.ones_like(cell_sums), cell_sums)
            tf = X / cell_sums
            return tf * idf   

    @staticmethod
    def clr_normalize_each_cell(adata, inplace=True):
        
        """Normalize count vector for each cell, i.e. for each row of .X"""

        def seurat_clr(x):
            s = np.sum(np.log1p(x[x > 0]))
            exp = np.exp(s / len(x))
            return np.log1p(x / exp)

        if not inplace:
            adata = adata.copy()
        
        # apply to dense or sparse matrix, along axis. returns dense matrix
        adata.X = np.apply_along_axis(
            seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
        )
        return adata  
