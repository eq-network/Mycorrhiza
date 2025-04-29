from typing import Set, Type, TypeVar
from .graph_state import GraphState
from .properties import Property

T = TypeVar('T', bound='Transformation')

class Transformation:
    """A pure function that transforms a GraphState."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        
    def apply(self, state: GraphState) -> GraphState:
        """Apply the transformation to produce a new state."""
        raise NotImplementedError
        
    @property
    def preserves_properties(self) -> Set[Property]:
        """Return the set of properties preserved by this transformation."""
        return set()
    
    def validate(self, old_state: GraphState, new_state: GraphState) -> bool:
        """Validate that the transformation preserves specified properties."""
        for prop in self.preserves_properties:
            if not prop.check(new_state):
                return False
        return True
        
    def __rshift__(self, other: Type[T]) -> 'CompositeTransformation':
        """Enable composition with the >> operator."""
        return CompositeTransformation([self, other])


class CompositeTransformation(Transformation):
    """A composition of multiple transformations applied sequentially."""
    
    def __init__(self, transformations):
        super().__init__("CompositeTransformation")
        self.transformations = transformations
    
    def apply(self, state: GraphState) -> GraphState:
        """Apply all transformations in sequence."""
        current_state = state
        for transform in self.transformations:
            current_state = transform.apply(current_state)
        return current_state
    
    @property
    def preserves_properties(self) -> Set[Property]:
        """Return intersection of properties preserved by all transformations."""
        if not self.transformations:
            return set()
            
        # Start with all properties of first transformation
        result = set(self.transformations[0].preserves_properties)
        
        # Intersect with properties of remaining transformations
        for transform in self.transformations[1:]:
            result &= transform.preserves_properties
            
        return result