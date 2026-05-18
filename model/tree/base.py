"""
Base classes for hierarchical tree structure

Contains TreeNode and HierarchyTree classes used by all hierarchy builders.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class TreeNode:
    """Tree node class for storing hierarchical structure node information"""
    
    name: str
    level: int
    node_id: int
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = None
    edge_types: List[str] = None
    type_indices: List[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.edge_types is None:
            self.edge_types = []
        if self.type_indices is None:
            self.type_indices = []
        if self.metadata is None:
            self.metadata = {}
    
    def add_child(self, child: 'TreeNode') -> None:
        """Add child node"""
        child.parent = self
        self.children.append(child)
    
    def get_path(self) -> List[str]:
        """Get path from root node to current node"""
        path = []
        current = self
        while current is not None:
            path.insert(0, current.name)
            current = current.parent
        return path
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf node"""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if node is root node"""
        return self.parent is None
    
    def get_all_descendants(self) -> List['TreeNode']:
        """Get all descendant nodes."""
        return [c for child in self.children for c in ([child] + child.get_all_descendants())]
    
    def __str__(self) -> str:
        return f"TreeNode(name='{self.name}', level={self.level}, id={self.node_id})"
    
    def __repr__(self) -> str:
        return self.__str__()


class HierarchyTree:
    """Hierarchical tree class providing tree structure operations"""
    
    def __init__(self, name: str = "root"):
        self.root = TreeNode(name=name, level=0, node_id=0)
        self._node_counter = 1
        self._name_to_node: Dict[str, TreeNode] = {name: self.root}
        self._level_nodes: Dict[int, List[TreeNode]] = {0: [self.root]}
        self._edge_type_to_node: Dict[str, TreeNode] = {}
    
    def add_node(self, name: str, parent_name: str, level: int, 
                 edge_types: List[str] = None, type_indices: List[int] = None,
                 metadata: Dict[str, Any] = None) -> TreeNode:
        """Add node to tree
        
        Note: Nodes with the same name can exist in different levels or under different parents.
        get_node() will return the first matching node found.
        """
        parent = self._name_to_node.get(parent_name)
        if parent is None:
            raise ValueError(f"Parent node '{parent_name}' not found")
        
        # Create new node
        node = TreeNode(
            name=name,
            level=level,
            node_id=self._node_counter,
            edge_types=edge_types or [],
            type_indices=type_indices or [],
            metadata=metadata or {}
        )
        
        # Add to tree
        parent.add_child(node)
        # Store node (will overwrite if name already exists, but that's OK for get_node lookup)
        self._name_to_node[name] = node
        
        # Update level index
        if level not in self._level_nodes:
            self._level_nodes[level] = []
        self._level_nodes[level].append(node)
        
        # Update edge type index
        if edge_types:
            for edge_type in edge_types:
                self._edge_type_to_node[edge_type] = node
        
        self._node_counter += 1
        return node
    
    def get_node(self, name: str) -> Optional[TreeNode]:
        """Get node by name"""
        return self._name_to_node.get(name)
    
    def get_nodes_at_level(self, level: int) -> List[TreeNode]:
        """Get all nodes at specified level"""
        return self._level_nodes.get(level, [])
    
    def get_max_depth(self) -> int:
        """Get maximum depth of tree"""
        return max(self._level_nodes.keys()) if self._level_nodes else 0
    
    def get_total_nodes(self) -> int:
        """Get total number of nodes"""
        return len(self._name_to_node)
    
    def get_leaf_nodes(self) -> List[TreeNode]:
        """Get all leaf nodes"""
        return [node for node in self._name_to_node.values() if node.is_leaf()]
    
    def get_edge_type_node(self, edge_type: str) -> Optional[TreeNode]:
        """Get node by edge type"""
        return self._edge_type_to_node.get(edge_type)
    
    def _type_idx(self, node: 'TreeNode', idx: int) -> int:
        """Get type_idx for node at index idx."""
        if not node.type_indices:
            return 0
        return node.type_indices[idx] if idx < len(node.type_indices) else node.type_indices[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary format: level_1, level_2, ... with edge_type / group_name keys."""
        result = {}
        for level in sorted(self._level_nodes.keys()):
            level_key = f'level_{level + 1}'
            result[level_key] = {}
            for node in self._level_nodes[level]:
                if node.is_leaf() and node.edge_types:
                    for idx, edge_type in enumerate(node.edge_types):
                        result[level_key][edge_type] = {
                            'type_idx': self._type_idx(node, idx),
                            'edge_type_name': edge_type,
                        }
                else:
                    result[level_key][node.name] = []
                    for leaf in (d for d in node.get_all_descendants() if d.is_leaf()):
                        for idx, edge_type in enumerate(leaf.edge_types):
                            result[level_key][node.name].append({
                                'edge_type_name': edge_type,
                                'type_idx': self._type_idx(leaf, idx),
                            })
        return result
    
    def __str__(self) -> str:
        return f"HierarchyTree(root='{self.root.name}', nodes={self.get_total_nodes()}, depth={self.get_max_depth()})"
    
    def __repr__(self) -> str:
        return self.__str__()
    