"""LLM hierarchy builder using LLM and bge-base-en-v1.5."""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.tools import (
    AliyunLLMClient,
    BGEEmbeddingUtils,
    normalize_edge_type,
    retrieve_from_db,
)

from .base import HierarchyTree, TreeNode


class LLMHierarchyBuilder:
    """Builder for LLM-based hierarchical label trees"""
    
    @staticmethod
    def build(edge_type_map: Dict[str, int], cell_chat_db: Optional[Dict] = None) -> HierarchyTree:
        """Build hierarchical label tree using LLM and bge-base-en-v1.5."""
        if not edge_type_map:
            return HierarchyTree("root")
        
        edge_types = list(edge_type_map.keys())
        N = len(edge_types)
        
        retrieved_data = retrieve_from_db(edge_types, cell_chat_db)
        edge_type_embeddings = LLMHierarchyBuilder._compute_embeddings_with_data(edge_types, retrieved_data)
        
        tree = HierarchyTree("root")
        L, children_counts = LLMHierarchyBuilder._determine_tree_structure(N)
        virtual_nodes_by_level = LLMHierarchyBuilder._generate_virtual_nodes(
            tree, edge_types, L, children_counts, retrieved_data, edge_type_embeddings
        )
        LLMHierarchyBuilder._match_real_to_virtual_nodes(
            tree, edge_types, edge_type_map, virtual_nodes_by_level, L, children_counts
        )
        
        return tree
    
    @staticmethod
    def _determine_tree_structure(N: int) -> Tuple[int, List[int]]:
        """Determine tree structure: L and children_counts per level."""
        if 1 <= N <= 10:
            L = 2
            children_counts = [N]
        elif 11 <= N <= 24:
            L = 3
            X = int(np.ceil(np.sqrt(N)))
            Y1 = int(np.ceil(N / X))
            Y2 = Y1 - 1 if Y1 > 1 else 1
            Y = Y1 if abs(X * Y1 - N) <= abs(X * Y2 - N) else Y2
            children_counts = [X, Y]
        else:
            L = 4
            X = int(np.ceil(N ** (1/3)))
            Y = X
            Z1 = int(np.ceil(N / (X * Y)))
            Z2 = Z1 - 1 if Z1 > 1 else 1
            Z = Z1 if abs(X * Y * Z1 - N) <= abs(X * Y * Z2 - N) else Z2
            children_counts = [X, Y, Z]
        return L, children_counts
    
    @staticmethod
    def _generate_virtual_nodes(
        tree: HierarchyTree,
        edge_types: List[str],
        L: int,
        children_counts: List[int],
        retrieved_data: Dict[str, Dict[str, Any]],
        edge_type_embeddings: Optional[List[np.ndarray]] = None
    ) -> Dict[int, List[TreeNode]]:
        """Generate virtual nodes level by level."""
        virtual_nodes_by_level = {0: [tree.root]}
        if L == 2:
            return virtual_nodes_by_level

        def _sanitize_theme_name(raw: Any) -> str:
            """Sanitize LLM theme for safe, readable node names."""
            if raw is None:
                return ""
            s = str(raw).strip()
            # collapse whitespace/newlines
            s = " ".join(s.split())
            return s

        def _unique_virtual_name(base: str, level: int, parent_node: TreeNode, i: int) -> str:
            """
            Ensure name uniqueness to avoid HierarchyTree._name_to_node overwrite.
            Prefer using the LLM theme as the visible name, append minimal suffix only when needed.
            """
            base = base or f"theme_{level}_{i}"
            candidate = base
            # Avoid collisions with existing nodes (including real leaf edge_type names).
            # Keep suffix stable and short.
            if candidate in tree._name_to_node:
                candidate = f"{base}__L{level}_P{parent_node.node_id}_{i}"
            # In extremely rare cases, keep bumping a counter.
            bump = 1
            while candidate in tree._name_to_node:
                candidate = f"{base}__L{level}_P{parent_node.node_id}_{i}_{bump}"
                bump += 1
            return candidate
        
        for level in range(1, L - 1):
            parent_nodes = virtual_nodes_by_level[level - 1]
            current_level_nodes = []
            
            for parent_node in parent_nodes:
                parent_covered_texts = edge_types if parent_node == tree.root else LLMHierarchyBuilder._get_parent_covered_texts(
                    parent_node, edge_types, edge_type_embeddings
                )
                
                num_children = children_counts[0] if level == 1 else children_counts[level - 1]
                child_themes = LLMHierarchyBuilder._llm_generate_child_themes(
                    parent_node, parent_covered_texts, num_children, level, retrieved_data
                )
                
                for i, theme in enumerate(child_themes):
                    theme_name = _sanitize_theme_name(theme.get("name"))
                    node_name = _unique_virtual_name(theme_name, level, parent_node, i)
                    child_node = tree.add_node(
                        # Use LLM-recommended theme as the virtual node name (human-readable).
                        name=node_name,
                        parent_name=parent_node.name,
                        level=level,
                        metadata={
                            'theme': theme_name or f"theme_{level}_{i}",
                            'description': theme.get('description', ''),
                            'embedding': theme.get('embedding', None)
                        }
                    )
                    current_level_nodes.append(child_node)
            
            virtual_nodes_by_level[level] = current_level_nodes
        
        return virtual_nodes_by_level
    
    @staticmethod
    def _llm_generate_child_themes(
        parent_node: TreeNode,
        parent_covered_texts: List[str],
        num_children: int,
        level: int,
        retrieved_data: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """LLM batch generate child theme virtual nodes."""
        retrieved_texts = {
            normalize_edge_type(et): retrieved_data.get(normalize_edge_type(et), {}
            )
            for et in parent_covered_texts
        }
        
        prompt = LLMHierarchyBuilder._build_prompt(
            parent_node.metadata.get('theme', parent_node.name),
            retrieved_texts,
            len(parent_covered_texts),
            num_children
        )
        
        try:
            response = AliyunLLMClient().get_response([{"role": "user", "content": prompt}])
            themes = LLMHierarchyBuilder._parse_llm_response(response, level, num_children)
        except Exception as e:
            # LLM call failed, using placeholder themes
            themes = [{
                'name': f"theme_{level}_{i}",
                'description': f"Generated theme {i} for level {level}",
                'embedding': None
            } for i in range(num_children)]
        
        for theme in themes:
            if not theme.get('embedding'):
                theme['embedding'] = LLMHierarchyBuilder._compute_embedding(
                    f"{theme['name']} {theme.get('description', '')}"
                )
        
        return themes
    
    @staticmethod
    def _get_parent_covered_texts(
        parent_node: TreeNode,
        all_edge_types: List[str],
        edge_type_embeddings: List[np.ndarray],
        similarity_threshold: float = 0.3
    ) -> List[str]:
        """Get edge_types covered by parent node using preliminary similarity matching."""
        parent_embedding = parent_node.metadata.get('embedding')
        if parent_embedding is None:
            parent_theme = parent_node.metadata.get('theme', parent_node.name)
            parent_embedding = LLMHierarchyBuilder._compute_embedding(parent_theme)
            parent_node.metadata['embedding'] = parent_embedding
        
        covered_texts = []
        similarities = []
        for edge_type, embedding in zip(all_edge_types, edge_type_embeddings):
            similarity = LLMHierarchyBuilder._cosine_similarity(embedding, parent_embedding)
            similarities.append((edge_type, similarity))
            if similarity >= similarity_threshold:
                covered_texts.append(edge_type)
        
        if not covered_texts:
            similarities.sort(key=lambda x: x[1], reverse=True)
            covered_texts = [et for et, _ in similarities[:min(10, len(all_edge_types))]]
        
        return covered_texts
    
    @staticmethod
    def _build_prompt(
        parent_topic: str,
        retrieved_data: Dict[str, Dict[str, Any]],
        real_node_count: int,
        suggested_child_count: int
    ) -> str:
        """Build prompt for LLM based on template from configuration."""
        real_nodes_str_parts = []
        for edge_type, row_info in list(retrieved_data.items())[:50]:
            if row_info:
                row_info_json = json.dumps(row_info, ensure_ascii=False, indent=2)
                real_nodes_str_parts.append(f"- {edge_type}:\n{row_info_json}")
            else:
                real_nodes_str_parts.append(f"- {edge_type} (no database match)")
        
        real_nodes_str = "\n".join(real_nodes_str_parts)
        if len(retrieved_data) > 50:
            real_nodes_str += f"\n... (and {len(retrieved_data) - 50} more nodes)"
        
        prompt_template = AliyunLLMClient.get_prompt_template('generate_virtual_nodes')
        prompt = prompt_template.format(
            parent_topic=parent_topic,
            real_nodes_str=real_nodes_str,
            real_node_count=real_node_count,
            suggested_child_count=suggested_child_count
        )
        
        return prompt
    
    @staticmethod
    def _parse_llm_response(response: str, level: int, num_children: int) -> List[Dict[str, Any]]:
        """Parse LLM JSON response and convert to theme format."""
        themes = []
        try:
            response_clean = response.strip()
            if response_clean.startswith("```"):
                lines = response_clean.split("\n")
                json_start = False
                json_lines = []
                for line in lines:
                    if line.strip().startswith("```"):
                        if json_start:
                            break
                        json_start = True
                        continue
                    if json_start:
                        json_lines.append(line)
                response_clean = "\n".join(json_lines)
            
            data = json.loads(response_clean)
            for i, node_data in enumerate(data.get("virtual_nodes", [])[:num_children]):
                themes.append({
                    'name': node_data.get("node_topic", f"theme_{level}_{i}"),
                    'description': node_data.get("biological_rationale", ""),
                    'embedding': None
                })
            
            while len(themes) < num_children:
                themes.append({
                    'name': f"theme_{level}_{len(themes)}",
                    'description': f"Generated theme {len(themes)} for level {level}",
                    'embedding': None
                })
            
        except Exception:
            # Failed to parse LLM response, using placeholder themes
            themes = [{
                'name': f"theme_{level}_{i}",
                'description': f"Generated theme {i} for level {level}",
                'embedding': None
            } for i in range(num_children)]
        
        return themes
    
    @staticmethod
    def _match_real_to_virtual_nodes(
        tree: HierarchyTree,
        edge_types: List[str],
        edge_type_map: Dict[str, int],
        virtual_nodes_by_level: Dict[int, List[TreeNode]],
        L: int,
        children_counts: List[int]
    ) -> None:
        """Match and associate real nodes with virtual nodes."""
        if L == 2:
            for edge_type in edge_types:
                leaf_node = tree.add_node(
                    name=edge_type,
                    parent_name="root",
                    level=1,
                    edge_types=[edge_type],
                    type_indices=[edge_type_map[edge_type]]
                )
                tree._edge_type_to_node[edge_type] = leaf_node
            return
        
        bottom_level = L - 2
        bottom_virtual_nodes = virtual_nodes_by_level.get(bottom_level, [])
        if not bottom_virtual_nodes:
            return
        
        target_count = max(1, len(edge_types) // len(bottom_virtual_nodes))
        
        edge_type_embeddings = LLMHierarchyBuilder._compute_embeddings(edge_types)
        node_candidates = {node.name: [] for node in bottom_virtual_nodes}
        
        similarity_threshold = 0.5
        for edge_type, embedding in zip(edge_types, edge_type_embeddings):
            best_similarity, best_node = -1, None
            
            for virtual_node in bottom_virtual_nodes:
                virtual_embedding = virtual_node.metadata.get('embedding')
                if virtual_embedding is None:
                    virtual_embedding = LLMHierarchyBuilder._compute_embedding(
                        virtual_node.metadata.get('theme', virtual_node.name)
                    )
                    virtual_node.metadata['embedding'] = virtual_embedding
                
                similarity = LLMHierarchyBuilder._cosine_similarity(embedding, virtual_embedding)
                if similarity >= similarity_threshold and similarity > best_similarity:
                    best_similarity, best_node = similarity, virtual_node
            
            if best_node:
                node_candidates[best_node.name].append((edge_type, best_similarity))
        
        validated_assignments = {
            node.name: LLMHierarchyBuilder._llm_validate_assignments(node, [c[0] for c in node_candidates[node.name]])
            for node in bottom_virtual_nodes
        }
        
        final_assignments = LLMHierarchyBuilder._balance_assignments(
            validated_assignments, edge_types, bottom_virtual_nodes, target_count
        )
        
        for virtual_node in bottom_virtual_nodes:
            for edge_type in final_assignments.get(virtual_node.name, []):
                leaf_node = tree.add_node(
                    name=edge_type,
                    parent_name=virtual_node.name,
                    level=L - 1,
                    edge_types=[edge_type],
                    type_indices=[edge_type_map[edge_type]]
                )
                tree._edge_type_to_node[edge_type] = leaf_node
    
    @staticmethod
    def _compute_embeddings_with_data(
        edge_types: List[Any],
        retrieved_data: Dict[str, Dict[str, Any]]
    ) -> List[np.ndarray]:
        """Compute embeddings using retrieved database information."""
        enriched_texts = []
        for edge_type in edge_types:
            normalized_key = normalize_edge_type(edge_type)
            row_info = retrieved_data.get(normalized_key, {})
            text_parts = [normalized_key]
            if row_info:
                annotation = row_info.get('annotation', '')
                pathway_name = row_info.get('pathway_name', '')
                if annotation and annotation != 'Unknown':
                    text_parts.append(annotation)
                if pathway_name:
                    text_parts.append(pathway_name)
            enriched_texts.append(' '.join(text_parts))
        return LLMHierarchyBuilder._compute_embeddings(enriched_texts)
    
    @staticmethod
    def _compute_embeddings(texts: List[str]) -> List[np.ndarray]:
        """Compute semantic vectors for text list using bge-base-en-v1.5."""
        return BGEEmbeddingUtils.compute_embeddings(texts)

    @staticmethod
    def _compute_embedding(text: str) -> np.ndarray:
        """Compute semantic vector for a single text using bge-base-en-v1.5."""
        return BGEEmbeddingUtils.compute_embedding(text)
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    @staticmethod
    def _llm_validate_assignments(virtual_node: TreeNode, candidate_edge_types: List[str]) -> List[str]:
        """LLM batch validates semantic relevance between candidate nodes and virtual nodes."""
        if not candidate_edge_types:
            return []
        
        virtual_topic = virtual_node.metadata.get('theme', virtual_node.name)
        
        candidate_nodes_str = "\n".join([f"- {et}" for et in candidate_edge_types])
        prompt_template = AliyunLLMClient.get_prompt_template('validate_assignments')
        prompt = prompt_template.format(
            virtual_topic=virtual_topic,
            candidate_nodes_str=candidate_nodes_str
        )
        
        try:
            client = AliyunLLMClient()
            messages = [{"role": "user", "content": prompt}]
            response = client.get_response(messages)
            
            # Parse JSON response
            response_clean = response.strip()
            if response_clean.startswith("```"):
                lines = response_clean.split("\n")
                json_start = False
                json_lines = []
                for line in lines:
                    if line.strip().startswith("```"):
                        if json_start:
                            break
                        json_start = True
                        continue
                    if json_start:
                        json_lines.append(line)
                response_clean = "\n".join(json_lines)
            
            data = json.loads(response_clean)
            validation_results = data.get("validation_results", [])
            
            # Extract valid edge_types
            valid_edge_types = []
            for result in validation_results:
                real_node = result.get("real_node", "")
                is_valid = result.get("is_valid", False)
                if is_valid and real_node in candidate_edge_types:
                    valid_edge_types.append(real_node)
            
            return valid_edge_types if valid_edge_types else candidate_edge_types
            
        except Exception:
            # LLM validation failed, returning all candidates
            return candidate_edge_types
    
    @staticmethod
    def _balance_assignments(
        validated_assignments: Dict[str, List[str]],
        all_edge_types: List[str],
        virtual_nodes: List[TreeNode],
        target_count: int
    ) -> Dict[str, List[str]]:
        """Balance assignments: Ensure difference in real nodes covered by virtual nodes at same level ≤1."""
        assigned = set()
        for edge_types in validated_assignments.values():
            assigned.update(edge_types)
        
        unassigned = [et for et in all_edge_types if et not in assigned]
        if unassigned:
            unassigned_embeddings = LLMHierarchyBuilder._compute_embeddings(unassigned)
            for edge_type, embedding in zip(unassigned, unassigned_embeddings):
                best_node, best_similarity = None, -1
                for virtual_node in virtual_nodes:
                    virtual_embedding = virtual_node.metadata.get('embedding')
                    if virtual_embedding is None:
                        virtual_embedding = LLMHierarchyBuilder._compute_embedding(
                            virtual_node.metadata.get('theme', virtual_node.name)
                        )
                    similarity = LLMHierarchyBuilder._cosine_similarity(embedding, virtual_embedding)
                    if similarity > best_similarity:
                        best_similarity, best_node = similarity, virtual_node
                if best_node:
                    if best_node.name not in validated_assignments:
                        validated_assignments[best_node.name] = []
                    validated_assignments[best_node.name].append(edge_type)
        
        node_names = [node.name for node in virtual_nodes]
        current_counts = {name: len(validated_assignments.get(name, [])) for name in node_names}
        total_assigned = sum(current_counts.values())
        
        if total_assigned == 0:
            per_node = len(all_edge_types) // len(virtual_nodes)
            remainder = len(all_edge_types) % len(virtual_nodes)
            idx = 0
            final_assignments = {}
            for i, node in enumerate(virtual_nodes):
                count = per_node + (1 if i < remainder else 0)
                final_assignments[node.name] = all_edge_types[idx:idx+count]
                idx += count
        else:
            edge_type_list = []
            for name in node_names:
                edge_type_list.extend(validated_assignments.get(name, []))
            per_node = len(edge_type_list) // len(virtual_nodes)
            remainder = len(edge_type_list) % len(virtual_nodes)
            idx = 0
            final_assignments = {}
            for i, node in enumerate(virtual_nodes):
                count = per_node + (1 if i < remainder else 0)
                final_assignments[node.name] = edge_type_list[idx:idx+count]
                idx += count
        
        return {k: v for k, v in final_assignments.items() if len(v) > 0}
