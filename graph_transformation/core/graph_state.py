# This is already implemented as you shared
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Tuple, Set
from einops import rearrange, reduce

class GraphState:
    """Immutable representation of a graph state."""
    
    def __init__(
        self, 
        node_attrs: Dict[str, jnp.ndarray],
        adj_matrices: Dict[str, jnp.ndarray],
        global_attrs: Optional[Dict[str, Any]] = None
    ):
        # Store as immutable JAX arrays
        self.node_attrs = {k: jnp.asarray(v) for k, v in node_attrs.items()}
        self.adj_matrices = {k: jnp.asarray(v) for k, v in adj_matrices.items()}
        self.global_attrs = global_attrs or {}
        
        # Validate shapes
        self._validate_shapes()
    
    def _validate_shapes(self) -> None:
        """Ensure consistent dimensions across node attributes and adjacency matrices."""
        if not self.node_attrs:
            return
            
        # All node attributes should have same first dimension
        num_nodes = next(iter(self.node_attrs.values())).shape[0]
        for attr_name, attr in self.node_attrs.items():
            assert attr.shape[0] == num_nodes, f"Node attribute '{attr_name}' has inconsistent shape"
            
        # All adjacency matrices should be square with dimension num_nodes
        for rel_name, matrix in self.adj_matrices.items():
            assert matrix.shape == (num_nodes, num_nodes), \
                f"Adjacency matrix '{rel_name}' has inconsistent shape"
    
    def replace(self, 
                node_attrs: Optional[Dict[str, jnp.ndarray]] = None,
                adj_matrices: Optional[Dict[str, jnp.ndarray]] = None,
                global_attrs: Optional[Dict[str, Any]] = None) -> 'GraphState':
        """Create a new GraphState with specified attributes replaced."""
        return GraphState(
            node_attrs=node_attrs if node_attrs is not None else self.node_attrs,
            adj_matrices=adj_matrices if adj_matrices is not None else self.adj_matrices,
            global_attrs=global_attrs if global_attrs is not None else self.global_attrs
        )