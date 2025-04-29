import os
import faiss
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

class FaissIndexManager:
    """Manages FAISS indices for efficient vector similarity search."""

    def __init__(self, index_directory: str, dimensions: int, index_factory_string: str = "Flat", normalize_vectors: bool = False):
        """
        Initializes the FAISS index manager.

        Args:
            index_directory (str): Directory to store FAISS index files and ID maps.
            dimensions (int): The dimensionality of the vectors to be indexed.
            index_factory_string (str): FAISS index factory string (e.g., 'Flat', 'IVF100,Flat').
                                        'Flat' uses L2 distance by default. For cosine similarity,
                                        use 'Flat' with normalize_vectors=True (which enables IndexFlatIP),
                                        or consider index types supporting cosine directly.
            normalize_vectors (bool): Whether to normalize vectors before adding (required for cosine
                                      similarity when using IndexFlatIP derived from normalize+Flat).
        """
        self.index_directory = Path(index_directory)
        self.dimensions = dimensions
        self.index_factory_string = index_factory_string
        self.normalize_vectors = normalize_vectors
        self.indices: Dict[str, faiss.Index] = {}
        self.id_maps: Dict[str, Dict[int, str]] = {} # Maps FAISS vector index (int) to original entity ID (str)

        self.index_directory.mkdir(parents=True, exist_ok=True)
        print(f"FAISS Manager initialized. Index directory: {self.index_directory}")

    def _get_index_paths(self, index_name: str) -> Tuple[Path, Path]:
        """Gets the file paths for a given index name."""
        index_file = self.index_directory / f"{index_name}.index"
        id_map_file = self.index_directory / f"{index_name}_id_map.json"
        return index_file, id_map_file

    def _load_index(self, index_name: str) -> bool:
        """Loads an index and its ID map from disk if they exist."""
        index_file, id_map_file = self._get_index_paths(index_name)
        if index_file.exists() and id_map_file.exists():
            try:
                print(f"Loading FAISS index '{index_name}' from {index_file}...")
                self.indices[index_name] = faiss.read_index(str(index_file))
                with open(id_map_file, 'r') as f:
                    # Load JSON map, ensuring keys are converted back to integers
                    loaded_map = json.load(f)
                    self.id_maps[index_name] = {int(k): v for k, v in loaded_map.items()}
                print(f"Loaded index '{index_name}' with {self.indices[index_name].ntotal} vectors.")
                # Sanity check
                if self.indices[index_name].ntotal != len(self.id_maps[index_name]):
                     print(f"Warning: Index '{index_name}' size ({self.indices[index_name].ntotal}) mismatch with ID map size ({len(self.id_maps[index_name])}). Rebuilding might be needed.")
                return True
            except Exception as e:
                print(f"Error loading index '{index_name}' from disk: {e}. Will create a new one.")
                # Clear potentially corrupted partial load
                if index_name in self.indices: del self.indices[index_name]
                if index_name in self.id_maps: del self.id_maps[index_name]
                return False
        return False

    def _create_index(self, index_name: str):
        """Creates a new, empty FAISS index and ID map."""
        print(f"Creating new FAISS index '{index_name}' (Factory: '{self.index_factory_string}')...")
        try:
            # Use IndexFlatIP if normalizing for cosine similarity with Flat factory string
            if self.normalize_vectors and self.index_factory_string == "Flat":
                 # IndexFlatIP performs inner product search, equivalent to cosine on normalized vectors
                 index = faiss.IndexFlatIP(self.dimensions)
                 print(f"Using IndexFlatIP for normalized vectors (cosine similarity).")
            else:
                 index = faiss.index_factory(self.dimensions, self.index_factory_string)
            
            self.indices[index_name] = index
            self.id_maps[index_name] = {}
        except Exception as e:
             print(f"ERROR: Failed to create FAISS index '{index_name}' with factory '{self.index_factory_string}'. Error: {e}")
             raise # Re-raise the error

    def get_index(self, index_name: str) -> Tuple[Optional[faiss.Index], Optional[Dict[int, str]]]:
        """Gets the index and ID map, loading from disk or creating if necessary."""
        if index_name not in self.indices:
            if not self._load_index(index_name):
                self._create_index(index_name)
        
        # Check if index creation failed
        if index_name not in self.indices or index_name not in self.id_maps:
             return None, None 
             
        return self.indices.get(index_name), self.id_maps.get(index_name)

    def add_embeddings(self, index_name: str, ids: List[str], embeddings: np.ndarray):
        """Adds embeddings to the specified index."""
        index, id_map = self.get_index(index_name)
        if index is None or id_map is None:
             print(f"ERROR: Index '{index_name}' could not be obtained. Cannot add embeddings.")
             return
             
        if len(ids) != embeddings.shape[0]:
            print(f"ERROR: Number of IDs ({len(ids)}) does not match number of embeddings ({embeddings.shape[0]}).")
            return
            
        if embeddings.shape[1] != self.dimensions:
             print(f"ERROR: Embedding dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimensions}).")
             return

        vectors_to_add = np.array(embeddings, dtype='float32')

        if self.normalize_vectors:
            faiss.normalize_L2(vectors_to_add) # Normalize vectors for cosine similarity search with IndexFlatIP

        start_index = index.ntotal
        index.add(vectors_to_add)
        
        # Update ID map
        for i, original_id in enumerate(ids):
            id_map[start_index + i] = original_id
            
        print(f"Added {len(ids)} vectors to index '{index_name}'. Total vectors: {index.ntotal}")

    def search(self, index_name: str, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Searches the index for the top_k nearest neighbors."""
        index, id_map = self.get_index(index_name)
        if index is None or id_map is None or index.ntotal == 0:
            return []

        query_vector = np.array(query_vector, dtype='float32').reshape(1, -1)
        if self.normalize_vectors:
            faiss.normalize_L2(query_vector) # Normalize query vector if index uses normalized vectors

        actual_k = min(top_k, index.ntotal)
        if actual_k <= 0: 
             return []

        try:
            distances, indices = index.search(query_vector, actual_k)
        except Exception as e:
            print(f"Error during FAISS search on index '{index_name}': {e}")
            return []

        results = []
        for i in range(actual_k):
            idx = indices[0][i]
            dist = distances[0][i] # This is L2 distance for Flat/IVF, or Inner Product for IndexFlatIP
            if idx != -1:
                original_id = id_map.get(idx)
                if original_id:
                    # For IndexFlatIP on normalized vectors, Inner Product is Cosine Similarity.
                    # Similarity is directly the distance value (higher is better).
                    # For IndexFlatL2, lower distance is better. Convert L2 to similarity (e.g., 1 / (1 + dist)).
                    if self.normalize_vectors and isinstance(index, faiss.IndexFlatIP):
                        similarity = float(dist)
                    else:
                        # Heuristic for L2 distance -> similarity (adjust as needed)
                        similarity = 1.0 / (1.0 + float(dist))
                    results.append((original_id, similarity))
        # Sort by similarity descending (higher is better)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def save_index(self, index_name: str):
        """Saves the specified index and its ID map to disk."""
        index, id_map = self.indices.get(index_name), self.id_maps.get(index_name)
        if index is None or id_map is None:
            print(f"Index '{index_name}' not loaded, cannot save.")
            return
            
        index_file, id_map_file = self._get_index_paths(index_name)
        try:
            print(f"Saving FAISS index '{index_name}' ({index.ntotal} vectors) to {index_file}...")
            faiss.write_index(index, str(index_file))
            # Save ID map with integer keys converted to string for JSON
            with open(id_map_file, 'w') as f:
                json.dump({str(k): v for k, v in id_map.items()}, f, indent=4)
            print(f"Index '{index_name}' saved successfully.")
        except Exception as e:
            print(f"Error saving index '{index_name}' to disk: {e}")

    def save_all_indices(self):
        """Saves all currently loaded indices."""
        print("Saving all loaded FAISS indices...")
        for index_name in list(self.indices.keys()):
            self.save_index(index_name)
        print("Finished saving indices.")

    def remove_index(self, index_name: str):
        """Removes an index from memory and deletes its files from disk."""
        print(f"Removing FAISS index '{index_name}'...")
        if index_name in self.indices:
            del self.indices[index_name]
        if index_name in self.id_maps:
            del self.id_maps[index_name]
            
        index_file, id_map_file = self._get_index_paths(index_name)
        try:
            if index_file.exists():
                os.remove(index_file)
                print(f"Deleted file: {index_file}")
            if id_map_file.exists():
                os.remove(id_map_file)
                print(f"Deleted file: {id_map_file}")
            print(f"Index '{index_name}' removed successfully.")
        except Exception as e:
            print(f"Error removing index files for '{index_name}': {e}")

    def get_index_size(self, index_name: str) -> int:
         """Returns the number of vectors in the specified index."""
         index, _ = self.get_index(index_name)
         return index.ntotal if index else 0 