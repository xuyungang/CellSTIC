"""
Base classes for hierarchical tree structure

Contains TreeNode and HierarchyTree classes used by all hierarchy builders.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TreeNode:
    """Tree node class for storing hierarchical structure node information"""

    name: str
    level: int
    node_id: int
    parent: Optional["TreeNode"] = None
    children: List["TreeNode"] = field(default_factory=list)
    edge_types: List[str] = field(default_factory=list)
    type_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "TreeNode") -> None:
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

    def get_all_descendants(self) -> List["TreeNode"]:
        """Get all descendant nodes."""
        return [
            c
            for child in self.children
            for c in ([child] + child.get_all_descendants())
        ]

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

    def add_node(
        self,
        name: str,
        parent_name: str,
        level: int,
        edge_types: Optional[List[str]] = None,
        type_indices: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TreeNode:
        """Add one uniquely named child node to the tree."""
        if name in self._name_to_node:
            raise ValueError(f"Hierarchy node name {name!r} is not unique")
        parent = self._name_to_node.get(parent_name)
        if parent is None:
            raise ValueError(f"Parent node '{parent_name}' not found")
        if level != parent.level + 1:
            raise ValueError(
                f"Node {name!r} level must be parent level + 1 "
                f"({parent.level + 1}), got {level}"
            )
        node_edge_types = list(edge_types or [])
        node_type_indices = list(type_indices or [])
        if len(node_edge_types) != len(node_type_indices):
            raise ValueError(
                "edge_types and type_indices must contain the same number "
                f"of entries for node {name!r}"
            )

        # Create new node
        node = TreeNode(
            name=name,
            level=level,
            node_id=self._node_counter,
            edge_types=node_edge_types,
            type_indices=node_type_indices,
            metadata=dict(metadata or {}),
        )

        # Add to tree
        parent.add_child(node)
        self._name_to_node[name] = node

        # Update level index
        if level not in self._level_nodes:
            self._level_nodes[level] = []
        self._level_nodes[level].append(node)

        # Update edge type index
        if node_edge_types:
            for edge_type in node_edge_types:
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

    def _type_idx(self, node: "TreeNode", idx: int) -> int:
        """Get type_idx for node at index idx."""
        if idx >= len(node.type_indices):
            raise ValueError(
                f"Node {node.name!r} has no type index for edge position {idx}"
            )
        return node.type_indices[idx]

    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary format: level_1, level_2, ... with edge_type / group_name keys."""
        result = {}
        for level in sorted(self._level_nodes.keys()):
            level_key = f"level_{level + 1}"
            result[level_key] = {}
            for node in self._level_nodes[level]:
                if node.is_leaf():
                    if node.edge_types:
                        for idx, edge_type in enumerate(node.edge_types):
                            result[level_key][edge_type] = {
                                "type_idx": self._type_idx(node, idx),
                                "edge_type_name": edge_type,
                            }
                    # Balanced construction may contain unused capacity leaves;
                    # they are structural placeholders, not output channels.
                    continue
                descendants = []
                for leaf in (
                    descendant
                    for descendant in node.get_all_descendants()
                    if descendant.is_leaf()
                ):
                    for idx, edge_type in enumerate(leaf.edge_types):
                        descendants.append(
                            {
                                "edge_type_name": edge_type,
                                "type_idx": self._type_idx(leaf, idx),
                            }
                        )
                if descendants:
                    result[level_key][node.name] = descendants
                else:
                    continue
        return result

    def __str__(self) -> str:
        return f"HierarchyTree(root='{self.root.name}', nodes={self.get_total_nodes()}, depth={self.get_max_depth()})"

    def __repr__(self) -> str:
        return self.__str__()
