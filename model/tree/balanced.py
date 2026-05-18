"""
Balanced hierarchy builder

Builds hierarchical label trees using strong balanced node division strategy.
"""

import numpy as np
from typing import Dict, List

from utils.tools.seed_utils import active_base_seed
from .base import HierarchyTree


class BalancedHierarchyBuilder:
    """Builder for balanced hierarchical label trees using strong balanced strategy"""
    
    @staticmethod
    def build(edge_type_map: Dict[str, int]) -> HierarchyTree:
        """
        Build a balanced hierarchical label tree (L=2/3/4 by N).
        Strategy: L=2 (1≤N≤8), L=3 (9≤N≤64), L=4 (N≥65).
        """
        np.random.seed(active_base_seed())
        tree = HierarchyTree("root")
        
        # Get edge type list
        edge_types = list(edge_type_map.keys())
        N = len(edge_types)
        
        if N == 0:
            return tree
        
        # Step 1: Determine number of layers L
        if 1 <= N <= 8:
            L = 2
        elif 9 <= N <= 64:
            L = 3
        else:  # N >= 65
            L = 4
        
        children_counts = BalancedHierarchyBuilder._calculate_children_counts(N, L)
        BalancedHierarchyBuilder._build_tree_structure(tree, children_counts, L)
        BalancedHierarchyBuilder._assign_edge_types(tree, edge_types, edge_type_map, L)
        
        return tree
    
    @staticmethod
    def _calculate_children_counts(N: int, L: int) -> List[int]:
        """
        Calculate children count for each level based on strong balanced strategy
        
        Args:
            N: Total number of edge types
            L: Number of layers
        
        Returns:
            List[int]: Children count for each level [c_1, c_2, ...]
        """
        if L == 2:
            # L=2: Root has N children (all leaves)
            return [N]
        
        elif L == 3:
            # L=3: X×Y=N where X=Y ideally
            # Check if N is a perfect square
            X = int(np.ceil(np.sqrt(N)))
            
            if X * X == N:
                # Perfect square: X=Y
                return [X, X]
            else:
                # Not perfect square: X²>N, Y={⌈N/X⌉,⌈N/X⌉-1}
                Y1 = int(np.ceil(N / X))
                Y2 = Y1 - 1
                
                # Choose Y that minimizes |X*Y - N|
                diff1 = abs(X * Y1 - N)
                diff2 = abs(X * Y2 - N)
                
                if diff1 <= diff2:
                    Y = Y1
                else:
                    Y = Y2
                
                return [X, Y]
        
        else:  # L == 4
            # L=4: X×Y×Z=N where X=Y=Z ideally
            # Check if N is a perfect cube
            X = int(np.ceil(N ** (1/3)))
            
            if X * X * X == N:
                # Perfect cube: X=Y=Z
                return [X, X, X]
            else:
                # Not perfect cube: X³>N, Y=X, Z={⌈N/(X×Y)⌉,⌈N/(X×Y)⌉-1}
                Y = X
                Z1 = int(np.ceil(N / (X * Y)))
                Z2 = Z1 - 1
                
                # Choose Z that minimizes |X*Y*Z - N|
                diff1 = abs(X * Y * Z1 - N)
                diff2 = abs(X * Y * Z2 - N)
                
                if diff1 <= diff2:
                    Z = Z1
                else:
                    Z = Z2
                
                return [X, Y, Z]
    
    @staticmethod
    def _build_tree_structure(tree: HierarchyTree, children_counts: List[int], L: int) -> None:
        """
        Build tree structure with given children counts
        
        Args:
            tree: HierarchyTree instance
            children_counts: List of children count for each level [c_1, c_2, ...]
            L: Number of layers
        """
        current_level_nodes = [tree.root]
        
        for level_idx, children_count in enumerate(children_counts):
            next_level_nodes = []
            
            for parent_node in current_level_nodes:
                for child_idx in range(children_count):
                    child_name = f"node_{parent_node.name}_{child_idx}"
                    child_node = tree.add_node(
                        name=child_name,
                        parent_name=parent_node.name,
                        level=level_idx + 1
                    )
                    next_level_nodes.append(child_node)
            
            current_level_nodes = next_level_nodes
    
    @staticmethod
    def _assign_edge_types(tree: HierarchyTree, edge_types: List[str], 
                           edge_type_map: Dict[str, int], L: int) -> None:
        """
        Assign edge types to leaf nodes using round-robin assignment
        
        Args:
            tree: HierarchyTree instance
            edge_types: List of edge type names
            edge_type_map: Dictionary mapping edge type names to integer indices
            L: Number of layers
        """
        # Get leaf nodes (nodes at the final level)
        final_level = L - 1  # Level is 0-indexed, so final level is L-1
        leaf_nodes = tree.get_nodes_at_level(final_level)
        
        # Assign edge types to leaf nodes using round-robin
        for i, edge_type in enumerate(edge_types):
            target_node = leaf_nodes[i % len(leaf_nodes)]
            
            # Add edge type to node
            target_node.edge_types.append(edge_type)
            target_node.type_indices.append(edge_type_map[edge_type])
            
            # Update edge type to node mapping
            tree._edge_type_to_node[edge_type] = target_node
