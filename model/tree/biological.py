"""
Biological hierarchy builder

Builds hierarchical label trees using biological knowledge from CellChatDB.
"""

from typing import Dict, Optional

from utils.tools import get_metadata_from_db, normalize_edge_type, parse_ligand_receptor

from .base import HierarchyTree


class BiologicalHierarchyBuilder:
    """Builder for biological hierarchical label trees"""
    
    @staticmethod
    def build(edge_type_map: Dict[str, int], 
              cell_chat_db: Optional[Dict] = None) -> HierarchyTree:
        """
        Build hierarchical label tree using biological knowledge.
        
        Hierarchy Design:
        - Level 1: Root node (interaction validity) - all edge_types
        - Level 2: Functional phenotype classification based on annotation + is_neurotransmitter
        - Level 3: Spatial matching filter based on receptor.secreted_type & ligand.secreted_type
        - Level 4: Individual molecular pairs, actual ligand-receptor pairs (leaf nodes)
        
        Args:
            edge_type_map: Dictionary mapping edge type names (format: "ligand:receptor") to integer indices
            cell_chat_db: Optional CellChatDB data. Can be:
                         - pandas DataFrame with columns: ligand, receptor, pathway_name (or annotation), 
                           ligand.secreted_type, receptor.secreted_type, is_neurotransmitter
                         - Dictionary mapping "ligand:receptor" (format: "TGFB1:TGFbR1_R2") to complete row info dict
                           containing all CSV columns (e.g., annotation, pathway_name, ligand.secreted_type,
                           receptor.secreted_type, is_neurotransmitter, etc.)
                         - Dictionary mapping ligand/receptor names to metadata dict (legacy format)
        
        Returns:
            HierarchyTree: Hierarchical label structure
        """
        if not edge_type_map:
            return HierarchyTree("root")
        
        normalized_edge_type_map = {
            normalize_edge_type(et): idx for et, idx in edge_type_map.items()
        }
        tree = HierarchyTree("root")

        edge_type_info = {}
        for edge_type, type_idx in normalized_edge_type_map.items():
            ligand, receptor = parse_ligand_receptor(edge_type)
            receptor_for_db = receptor.replace(':', '_')
            metadata = get_metadata_from_db(ligand, receptor_for_db, cell_chat_db)
            annotation = metadata['annotation']
            
            edge_type_info[edge_type] = {
                'ligand': ligand,
                'receptor': receptor,
                'type_idx': type_idx,
                'annotation': annotation,
                'is_neurotransmitter': metadata['is_neurotransmitter'],
                'ligand_secreted_type': metadata['ligand_secreted_type'],
                'receptor_secreted_type': metadata['receptor_secreted_type']
            }
        
        # Level 1: Root node (already created)
        # All edge types belong to root
        
        # Level 2: Function phenotype classification (annotation + is_neurotransmitter)
        # Group by: annotation + "_neurotransmitter" if is_neurotransmitter else annotation
        level2_groups = {}
        for edge_type, info in edge_type_info.items():
            # Create level 2 group name
            if info['is_neurotransmitter']:
                level2_name = f"{info['annotation']}_neurotransmitter"
            else:
                level2_name = info['annotation']
            
            if level2_name not in level2_groups:
                level2_groups[level2_name] = []
            level2_groups[level2_name].append(edge_type)
        
        # Check if level 2 should be skipped (only one node)
        skip_level2 = len(level2_groups) == 1
        
        # Create level 2 nodes (skip if only one node)
        level2_nodes = {}
        if not skip_level2:
            for level2_name, edge_types in level2_groups.items():
                node = tree.add_node(
                    name=level2_name,
                    parent_name="root",
                    level=1
                )
                level2_nodes[level2_name] = node
        
        # Level 3: Spatial matching filter (receptor.secreted_type & ligand.secreted_type)
        # Group by: "ligand_secreted_type&receptor_secreted_type" (use & for virtual nodes)
        level3_groups = {}
        for level2_name, edge_types in level2_groups.items():
            level3_groups[level2_name] = {}
            for edge_type in edge_types:
                info = edge_type_info[edge_type]
                # Create level 3 group name based on secreted types
                ligand_sec = info['ligand_secreted_type']
                receptor_sec = info['receptor_secreted_type']
                level3_name = f"{ligand_sec}&{receptor_sec}"
                
                if level3_name not in level3_groups[level2_name]:
                    level3_groups[level2_name][level3_name] = []
                level3_groups[level2_name][level3_name].append(edge_type)
        
        level3_nodes = {}
        for level2_name, level3_dict in level3_groups.items():
            level3_nodes[level2_name] = {}
            parent_name = level2_name if not skip_level2 else "root"
            for level3_name, edge_types in level3_dict.items():
                unique_level3_name = level3_name
                existing_node = tree.get_node(level3_name)
                if existing_node is not None and existing_node.parent.name == parent_name:
                    node = existing_node
                else:
                    if existing_node is not None:
                        unique_level3_name = f"{level2_name}_{level3_name}"
                        counter = 1
                        base_name = unique_level3_name
                        while tree.get_node(unique_level3_name) is not None:
                            unique_level3_name = f"{base_name}_{counter}"
                            counter += 1
                    node = tree.add_node(
                        name=unique_level3_name,
                        parent_name=parent_name,
                        level=2 if not skip_level2 else 1
                    )
                level3_nodes[level2_name][level3_name] = node
        
        # Level 4: Individual ligand-receptor pairs (leaf nodes)
        # Each edge_type becomes a leaf node
        for level2_name, level3_dict in level3_groups.items():
            for level3_name, edge_types in level3_dict.items():
                # Connect to level3 - use the actual node name (which may be unique)
                level3_node = level3_nodes[level2_name][level3_name]
                parent_name = level3_node.name
                if skip_level2:
                    leaf_level = 2
                else:
                    leaf_level = 3
                
                for edge_type in edge_types:
                    info = edge_type_info[edge_type]
                    # Create leaf node with edge_type as name
                    leaf_node = tree.add_node(
                        name=edge_type,
                        parent_name=parent_name,
                        level=leaf_level,
                        edge_types=[edge_type],
                        type_indices=[info['type_idx']],
                        metadata={
                            'ligand': info['ligand'],
                            'receptor': info['receptor'],
                            'annotation': info['annotation'],
                            'is_neurotransmitter': info['is_neurotransmitter'],
                            'ligand_secreted_type': info['ligand_secreted_type'],
                            'receptor_secreted_type': info['receptor_secreted_type']
                        }
                    )
                    # Update edge type to node mapping
                    tree._edge_type_to_node[edge_type] = leaf_node
        
        return tree
