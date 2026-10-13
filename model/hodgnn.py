"""
Edge-guided Graph Neural Network Layer with Gram-Schmidt orthogonal decomposition.
Uses hyperplane-based decomposition for edge features.
"""

from typing import Tuple

import dgl
import torch
import torch.nn as nn


class HODGNN(nn.Module):
    """
    Edge-guided Graph Neural Network Layer with Gram-Schmidt orthogonal decomposition.
    Uses hyperplane-based decomposition for edge features.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        edge_dim: int = 6,
    ) -> None:
        """
        Initialize HODGNN layer with hyperplane decomposition.

        Args:
            in_features: Input node features dimension
            out_features: Output node features dimension
            dropout: Dropout rate
            edge_dim: Input/output edge features dimension (default 6)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.num_hyperplanes = 3
        self.dropout = dropout
        self.alpha = 0.2

        self.hyperplane_basis = nn.Parameter(
            torch.empty(size=(self.num_hyperplanes, self.edge_dim))
        )
        nn.init.orthogonal_(self.hyperplane_basis.data)

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Cross-subspace attention
        self.cross_subspace_M = nn.Linear(out_features, out_features)
        nn.init.xavier_uniform_(self.cross_subspace_M.weight.data, gain=1.414)
        self.cross_subspace_q = nn.Parameter(torch.empty(out_features))
        nn.init.xavier_uniform_(self.cross_subspace_q.unsqueeze(0).data, gain=1.414)

        # Custom GAT: attention score from [Wh_src || Wh_dst], then × edge_components before softmax
        self.attention_mlps = nn.ModuleList(
            [nn.Linear(2 * out_features, 1) for _ in range(self.num_hyperplanes)]
        )
        for mlp in self.attention_mlps:
            nn.init.xavier_uniform_(mlp.weight.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.elu = nn.ELU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        g: dgl.DGLGraph,
        h: torch.Tensor,
        e: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of EGNN layer with hyperplane decomposition.

        Args:
            g: DGL graph
            h: Input node features (n_nodes, in_features)
            e: Input edge features (n_edges, edge_dim)

        Returns:
            Output node features (n_nodes, out_features)
            Updated edge features (n_edges, edge_dim)
        """
        if h.ndim != 2 or h.shape[1] != self.in_features:
            raise ValueError(
                "HODGNN node feature mismatch: expected shape "
                f"(n_nodes, {self.in_features}), got {tuple(h.shape)}"
            )
        if e.ndim != 2 or e.shape[1] != self.edge_dim:
            raise ValueError(
                "HODGNN edge feature mismatch: expected shape "
                f"(n_edges, {self.edge_dim}), got {tuple(e.shape)}"
            )
        if h.shape[0] != g.num_nodes():
            raise ValueError(
                f"Expected {g.num_nodes()} node-feature rows, got {h.shape[0]}"
            )
        if e.shape[0] != g.num_edges():
            raise ValueError(
                f"Expected {g.num_edges()} edge-feature rows, got {e.shape[0]}"
            )

        # Step 1: Orthogonalize hyperplane basis using Gram-Schmidt
        ortho_basis = self._gram_schmidt_orthogonalize(
            self.hyperplane_basis
        )  # (num_hyperplanes, edge_dim)

        # Step 2: Project edge features onto hyperplanes
        edge_components = self._project_edge_to_hyperplanes(
            e, ortho_basis
        )  # (n_edges, num_hyperplanes)

        # Step 3: Transform node features
        Wh = torch.mm(h, self.W)  # (n_nodes, out_features)
        src, dst = g.edges()
        # Step 4: Custom GAT — attention score × edge_components, then edge_softmax, then aggregate.
        # A linear layer over ``[Wh_src || Wh_dst]`` is algebraically equal to
        # one scalar projection per node followed by edge indexing.  This avoids
        # the O(n_edges * 2 * out_features) concatenated edge tensor while
        # preserving the exact attention scores and gradients.
        attention_scores = []
        for mlp in self.attention_mlps:
            weight = mlp.weight.squeeze(0)
            source_score = Wh @ weight[: self.out_features]
            target_score = Wh @ weight[self.out_features :]
            score = source_score[src] + target_score[dst]
            if mlp.bias is not None:
                score = score + mlp.bias.squeeze(0)
            attention_scores.append(score)
        attention_scores_all = torch.stack(attention_scores, dim=1)
        attention_scores_all = self.leakyrelu(attention_scores_all)
        attention_scores_weighted = (
            attention_scores_all * edge_components
        )  # × edge weight per hyperplane

        attention_weights_all = torch.stack(
            [
                dgl.ops.edge_softmax(g, attention_scores_weighted[:, k])
                for k in range(self.num_hyperplanes)
            ],
            dim=1,
        )

        hyperplane_features_list = []
        for k in range(self.num_hyperplanes):
            # Keep attention-weight gradients sparse: COO torch.sparse.mm
            # backward can materialize an n_nodes x n_nodes dense matrix.
            aggregated = dgl.ops.u_mul_e_sum(
                g, Wh, attention_weights_all[:, k].unsqueeze(-1)
            )
            hyperplane_features_list.append(aggregated)

        # Step 5: Cross-subspace fusion
        hyperplane_features = torch.stack(
            hyperplane_features_list, dim=1
        )  # (n_nodes, num_hyperplanes, out_features) = f̂_u^p
        cross_in = self.cross_subspace_M(
            hyperplane_features
        )  # (n_nodes, num_hyperplanes, out_features)
        cross_tanh = torch.tanh(cross_in)  # (n_nodes, num_hyperplanes, out_features)
        gamma_up = torch.einsum(
            "npo,o->np", cross_tanh, self.cross_subspace_q
        )  # (n_nodes, num_hyperplanes)
        delta_up = torch.softmax(gamma_up, dim=-1)  # (n_nodes, num_hyperplanes) = δ_u^p
        importance_weights = delta_up.unsqueeze(-1)  # (n_nodes, num_hyperplanes, 1)
        weighted_features = (
            hyperplane_features * importance_weights
        )  # (n_nodes, num_hyperplanes, out_features)
        h_prime = weighted_features.sum(dim=1)  # (n_nodes, out_features) = f̂_u
        h_prime = self.elu(h_prime)  # Apply ELU activation
        h_prime = self.dropout_layer(h_prime)  # Apply dropout

        # Step 6: Recover edge features using attention_weights_all: e' = sum_m(attn_m * f_e^m * P_m)
        e_prime = (
            attention_weights_all.unsqueeze(-1)
            * (edge_components.unsqueeze(-1) * ortho_basis.unsqueeze(0))
        ).sum(
            dim=1
        )  # (n_edges, edge_dim)

        return h_prime, e_prime

    def _gram_schmidt_orthogonalize(self, basis: torch.Tensor) -> torch.Tensor:
        """Apply Gram-Schmidt orthogonalization to basis vectors."""
        eps = 1e-8
        ortho_basis_list = []
        for k in range(self.num_hyperplanes):
            vector = basis[k].clone()
            for previous in ortho_basis_list:
                vector = vector - torch.dot(vector, previous) * previous
            norm = torch.norm(vector)
            if norm > eps:
                ortho_basis_list.append(vector / norm)
            else:
                fallback_norm = torch.norm(basis[k])
                ortho_basis_list.append(basis[k] / (fallback_norm + eps))
        return torch.stack(ortho_basis_list, dim=0)

    def _project_edge_to_hyperplanes(
        self, edge_features: torch.Tensor, ortho_basis: torch.Tensor
    ) -> torch.Tensor:
        """
        Project edge features onto num_hyperplanes hyperplanes using orthogonal decomposition.

        Args:
            edge_features: Edge features (n_edges, edge_dim)
            ortho_basis: Orthonormal basis vectors (num_hyperplanes, edge_dim)

        Returns:
            Edge components in each hyperplane (n_edges, num_hyperplanes)
        """
        components = torch.matmul(
            edge_features, ortho_basis.t()
        )  # (n_edges, num_hyperplanes)
        return components
