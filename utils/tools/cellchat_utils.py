"""
CellChatDB loader and retrieval utilities. All logic is encapsulated in CellChatDBLoader.

- CellChatDBLoader: load CSV, get species/ligand-receptor maps and static retrieval methods.
- Module-level normalize_edge_type, parse_ligand_receptor, get_metadata_from_db, retrieve_from_db
  are aliases to the same static methods on CellChatDBLoader for backward compatibility (e.g. hierarchy builders).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class CellChatDBLoader:
    """
    A simplified utility class for loading CellChatDB ligand-receptor interaction data.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the CellChatDB loader.
        
        Args:
            data_dir: Path to the CellChatDB directory containing .rda files.
                     If None, uses default path relative to project root: component/cellchatdb
        """
        if data_dir is None:
            # Use relative path from project root
            self.data_dir = project_root / 'component' / 'cellchatdb'
        else:
            # If provided path is absolute, use it; otherwise, treat as relative to project root
            data_dir_path = Path(data_dir)
            if data_dir_path.is_absolute():
                self.data_dir = data_dir_path
            else:
                self.data_dir = project_root / data_dir_path
        self._validate_data_dir()
        
        # Available species and their corresponding files
        self.species_files = {
            'human': 'CellChatDB.human.csv',
            'mouse': 'CellChatDB.mouse.csv', 
            'zebrafish': 'CellChatDB.zebrafish.csv'
        }
        
        # Cache for loaded data
        self._cache = {}
    
    def _validate_data_dir(self) -> None:
        """Validate that the data directory exists and contains expected files."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"CellChatDB directory not found: {self.data_dir}")
        
        # Check for at least one CellChatDB file
        cellchat_files = list(self.data_dir.glob("CellChatDB.*.csv"))
        if not cellchat_files:
            raise FileNotFoundError(f"No CellChatDB .csv files found in {self.data_dir}")
    
    def _load_csv_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a .csv file and return its contents as a DataFrame.
        
        Args:
            file_path: Path to the .csv file
            
        Returns:
            DataFrame containing the CSV data
        """
        result = pd.read_csv(file_path)
        return result
    
    
    def get_available_species(self) -> List[str]:
        """
        Get list of species with available CellChatDB data.
        
        Returns:
            List of species names with available data files
        """
        available = []
        for species, filename in self.species_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                available.append(species)
        return available
    
    def get_ligand_receptor_map(self, species: str) -> Dict[str, Dict[str, Any]]:
        """
        Create a mapping from ligand:receptor pairs to their complete row information for a specific species.
        
        Args:
            species: Species name ('human', 'mouse', or 'zebrafish')
            
        Returns:
            Dictionary mapping "ligand:receptor" strings to dictionaries containing complete row information
        """
        if species not in self.species_files:
            raise ValueError(f"Unsupported species: {species}. Available species: {list(self.species_files.keys())}")
        
        # Check cache first
        cache_key = f"{species}_map"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self.data_dir / self.species_files[species]
        if not file_path.exists():
            raise FileNotFoundError(f"CellChatDB file for {species} not found: {file_path}")
        
        # Load data
        data = self._load_csv_file(file_path)
        
        # Create ligand:receptor -> complete row info mapping
        ligand_receptor_map = {}
        
        ligand_col = None
        receptor_col = None
        
        # Find ligand and receptor columns
        for i, col in enumerate(data.columns):
            col_lower = col.lower()
            if 'ligand' in col_lower and ligand_col is None:
                ligand_col = col
            elif 'receptor' in col_lower and receptor_col is None:
                receptor_col = col
        
        # If we found both columns, build the mapping
        if ligand_col and receptor_col:
            for _, row in data.iterrows():
                ligand = str(row[ligand_col]).strip()
                receptor = str(row[receptor_col]).strip()
                
                if ligand and receptor and ligand != 'nan' and receptor != 'nan':
                    # Create key in format "ligand:receptor"
                    key = f"{ligand}:{receptor}"
                    
                    # Store complete row information as a dictionary
                    row_info = {}
                    for col in data.columns:
                        value = row[col]
                        # Convert to appropriate type, handling NaN
                        if pd.isna(value):
                            row_info[col] = None
                        else:
                            row_info[col] = value
                    
                    ligand_receptor_map[key] = row_info
        
        if not ligand_receptor_map:
            raise ValueError(f"No ligand-receptor interaction data found for species: {species}")
        
        # Cache the result
        self._cache[cache_key] = ligand_receptor_map

        return ligand_receptor_map

    # ----- Retrieval logic for hierarchy builders (DataFrame or dict) -----

    @staticmethod
    def normalize_edge_type(edge_type: Any) -> str:
        """Normalize edge_type to 'ligand_receptor' (first ':' replaced with '_')."""
        if isinstance(edge_type, tuple):
            ligand = edge_type[0]
            receptor = edge_type[1] if len(edge_type) > 1 else edge_type[0]
            return f"{ligand}_{receptor}"
        return str(edge_type).replace(':', '_', 1)

    @staticmethod
    def parse_ligand_receptor(normalized_key: str) -> Tuple[str, str]:
        """Parse normalized key 'ligand_receptor' -> (ligand, receptor)."""
        if '_' not in normalized_key:
            return normalized_key, normalized_key
        parts = normalized_key.split('_', 1)
        return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else parts[0].strip())

    @staticmethod
    def _get_field(data: Any, *field_names: str, default: str = 'Unknown') -> str:
        for name in field_names:
            try:
                val = data.get(name) if hasattr(data, 'get') else None
                if val is None and hasattr(data, '__getitem__'):
                    try:
                        val = data[name]
                    except (KeyError, IndexError):
                        continue
                if val is None:
                    continue
                try:
                    if isinstance(val, float) and pd.isna(val):
                        continue
                except Exception:
                    pass
                s = str(val).strip()
                if s.lower() != 'nan' and s:
                    return s
            except (KeyError, IndexError, AttributeError, TypeError):
                continue
        return default

    @staticmethod
    def _parse_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, str):
            return value.upper() in ('TRUE', 'T', '1', 'YES')
        return bool(value) if value is not None else default

    @staticmethod
    def _receptor_variants(receptor: str) -> List[str]:
        r = receptor.lower()
        out = [r]
        if ':' in receptor:
            out.append(receptor.replace(':', '_').lower())
        if '_' in receptor:
            out.append(receptor.replace('_', ':').lower())
        return list(dict.fromkeys(out))

    @staticmethod
    def _row_to_dict(row: Any) -> Dict[str, Any]:
        if row is None:
            return {}
        if isinstance(row, dict):
            return row.copy()
        if hasattr(row, 'to_dict'):
            return row.to_dict()
        return dict(row)

    @staticmethod
    def _find_row(ligand: str, receptor: str, cell_chat_db: Any) -> Optional[Dict[str, Any]]:
        """Return first matching row as dict, or None. receptor: use '_' for DB."""
        if cell_chat_db is None:
            return None
        ligand_lower = ligand.lower()
        variants = CellChatDBLoader._receptor_variants(receptor)

        if hasattr(cell_chat_db, 'iterrows'):
            ligand_col = cell_chat_db['ligand'].astype(str).str.lower().fillna('')
            receptor_col = cell_chat_db['receptor'].astype(str).str.lower().fillna('')
            mask = (ligand_col == ligand_lower) & (receptor_col == variants[0])
            for v in variants[1:]:
                mask = mask | ((ligand_col == ligand_lower) & (receptor_col == v))
            matching = cell_chat_db[mask]
            if len(matching) > 0:
                return CellChatDBLoader._row_to_dict(matching.iloc[0])
            return None

        if isinstance(cell_chat_db, dict):
            for key in cell_chat_db:
                key_lower = key.lower()
                if ':' in key_lower:
                    parts = key_lower.split(':', 1)
                    if len(parts) == 2 and parts[0] == ligand_lower and parts[1] in variants:
                        return CellChatDBLoader._row_to_dict(cell_chat_db[key])
                elif key_lower == ligand_lower or key_lower in variants:
                    return CellChatDBLoader._row_to_dict(cell_chat_db[key])
            for key in cell_chat_db:
                key_lower = key.lower()
                if ':' in key_lower:
                    parts = key_lower.split(':', 1)
                    if len(parts) == 2 and (parts[0] == ligand_lower or parts[1] in variants):
                        return CellChatDBLoader._row_to_dict(cell_chat_db[key])
                elif key_lower == ligand_lower or key_lower in variants:
                    return CellChatDBLoader._row_to_dict(cell_chat_db[key])
        return None

    @staticmethod
    def get_metadata_from_db(ligand: str, receptor: str, cell_chat_db: Any) -> Dict[str, Any]:
        """
        Get standard metadata for a ligand-receptor pair from CellChatDB.
        receptor: use '_' for DB (e.g. receptor.replace(':', '_')).
        Returns: annotation, is_neurotransmitter, ligand_secreted_type, receptor_secreted_type, pathway_name.
        """
        metadata = {
            'annotation': 'Unknown',
            'is_neurotransmitter': False,
            'ligand_secreted_type': 'Unknown',
            'receptor_secreted_type': 'Unknown',
            'pathway_name': '',
        }
        row = CellChatDBLoader._find_row(ligand, receptor, cell_chat_db)
        if not row:
            return metadata
        row = CellChatDBLoader._row_to_dict(row)

        annotation = CellChatDBLoader._get_field(row, 'annotation', 'pathway_name', 'pathway', default='Unknown')
        if annotation.endswith('_Unknown'):
            annotation = annotation[:-8]
        if annotation != 'Unknown':
            metadata['annotation'] = annotation

        try:
            nt = row.get('is_neurotransmitter', False)
            metadata['is_neurotransmitter'] = metadata['is_neurotransmitter'] or CellChatDBLoader._parse_bool(nt)
        except (KeyError, AttributeError):
            pass

        for key, field in [
            ('ligand.secreted_type', 'ligand_secreted_type'),
            ('receptor.secreted_type', 'receptor_secreted_type'),
        ]:
            val = CellChatDBLoader._get_field(row, key, field.replace('_', '.'), default='Unknown')
            if val != 'Unknown':
                metadata[field] = val

        pathway = row.get('pathway_name', '') or ''
        if pathway:
            metadata['pathway_name'] = str(pathway).strip()

        return metadata

    @staticmethod
    def retrieve_from_db(
        edge_types: List[Any],
        cell_chat_db: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        For each edge_type (tuple or "ligand:receptor" str), normalize to key and return
        dict[normalized_key] = metadata (and merged full row when available).
        """
        result = {}
        for edge_type in edge_types:
            key = CellChatDBLoader.normalize_edge_type(edge_type)
            ligand, receptor = CellChatDBLoader.parse_ligand_receptor(key)
            receptor_for_db = receptor.replace(':', '_')
            metadata = CellChatDBLoader.get_metadata_from_db(ligand, receptor_for_db, cell_chat_db)
            row = CellChatDBLoader._find_row(ligand, receptor_for_db, cell_chat_db)
            if row:
                row = row.copy() if isinstance(row, dict) else dict(row)
                row.update(metadata)
                result[key] = row
            else:
                result[key] = metadata
        return result


# Backward-compatible module-level aliases
normalize_edge_type = CellChatDBLoader.normalize_edge_type
parse_ligand_receptor = CellChatDBLoader.parse_ligand_receptor
get_metadata_from_db = CellChatDBLoader.get_metadata_from_db
retrieve_from_db = CellChatDBLoader.retrieve_from_db
