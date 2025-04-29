from typing import Set, Type, TypeVar
from .graph_state import GraphState

class Property:
    """Base class for properties that can be verified on a GraphState."""
    
    def check(self, state: GraphState) -> bool:
        """Return True if property holds for the given state."""
        raise NotImplementedError
    
    def __str__(self) -> str:
        return self.__class__.__name__

# Example properties relevant to voting
class ConservesSum(Property):
    """Property checking that the sum of a node attribute remains constant."""
    
    def __init__(self, attribute_name: str):
        self.attribute_name = attribute_name
        
    def check(self, state: GraphState) -> bool:
        """Check if the sum across all nodes for the attribute is 1.0."""
        if self.attribute_name not in state.node_attrs:
            return False
        
        attr = state.node_attrs[self.attribute_name]
        # Check if each row sums to approximately 1
        row_sums = attr.sum(axis=1)
        return bool(((row_sums - 1.0) < 1e-5).all())

class AcyclicRelation(Property):
    """Property checking that a relation (adjacency matrix) has no cycles."""
    
    def __init__(self, relation_name: str):
        self.relation_name = relation_name
    
    def check(self, state: GraphState) -> bool:
        """Check if the relation graph is acyclic."""
        if self.relation_name not in state.adj_matrices:
            return False
            
        # A simple way to check for cycles - raise adjacency matrix to power N
        # If any diagonal element is non-zero, there's a cycle
        import numpy as np
        from scipy.linalg import expm
        
        adj = np.array(state.adj_matrices[self.relation_name])
        n = adj.shape[0]
        
        # Matrix exponentiation is more stable than direct powers
        # This computes I + A + A²/2! + A³/3! + ...
        # If there are cycles, diagonal elements will be non-zero
        m = expm(adj) - np.eye(n)
        
        # Check if any diagonal element is significantly non-zero
        return bool((np.diag(m) < 1e-5).all())