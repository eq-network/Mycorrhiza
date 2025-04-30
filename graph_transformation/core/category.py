"""
Foundational category theory concepts adapted for JAX compatibility.

This module defines the core categorical concepts like morphisms and composition,
designed to work seamlessly with JAX's functional paradigm and transformations.
"""
from typing import TypeVar, Generic, Callable, List, Set, Dict, Any
from functools import wraps
import jax

from .graph import GraphState
from .property import Property

# Type variable for transformation functions
Transform = Callable[[GraphState], GraphState]

def compose(f: Transform, g: Transform) -> Transform:
    """
    Compose two transformations (g ∘ f) to create a new transformation.
    
    This is the fundamental operation in category theory, allowing us to chain
    transformations while preserving their mathematical properties.
    
    Args:
        f: First transformation to apply
        g: Second transformation to apply
    
    Returns:
        A new transformation that applies f followed by g
    """
    @wraps(g)
    def composed(state: GraphState) -> GraphState:
        """Apply f followed by g."""
        return g(f(state))
    
    # Preserve metadata from the original functions
    f_props = getattr(f, 'preserves', set())
    g_props = getattr(g, 'preserves', set())
    composed.preserves = f_props.intersection(g_props)
    
    return composed


def sequential(*transforms: Transform) -> Transform:
    """
    Compose a sequence of transformations into a single transformation.
    
    Args:
        *transforms: A sequence of transformation functions
    
    Returns:
        A single transformation that applies all transforms in sequence
    """
    if not transforms:
        # Return identity transformation
        return lambda state: state
    
    if len(transforms) == 1:
        return transforms[0]
    
    result = transforms[0]
    for t in transforms[1:]:
        result = compose(result, t)
    
    return result


def attach_properties(transform: Transform, properties: Set[Property]) -> Transform:
    """
    Attach a set of preserved properties to a transformation function.
    
    Args:
        transform: The transformation function
        properties: Set of properties preserved by the transformation
    
    Returns:
        The same transformation function with properties attached
    """
    transform.preserves = properties
    return transform


def jit_transform(transform: Transform) -> Transform:
    """
    Apply JAX's JIT compilation to a transformation.
    
    This improves performance by compiling the transformation function.
    The compiled version preserves the same properties as the original.
    
    Args:
        transform: The transformation function to compile
    
    Returns:
        A JIT-compiled version of the transformation
    """
    jitted = jax.jit(transform)
    
    # Preserve properties
    if hasattr(transform, 'preserves'):
        jitted.preserves = transform.preserves
    
    return jitted


def validate_properties(transform: Transform, initial_state: GraphState, final_state: GraphState) -> bool:
    """
    Validate that a transformation preserves its declared properties.
    
    Args:
        transform: The transformation to validate
        initial_state: The state before transformation
        final_state: The state after transformation
    
    Returns:
        True if all preserved properties are maintained, False otherwise
    """
    if not hasattr(transform, 'preserves'):
        return True
    
    for prop in transform.preserves:
        if not prop.check(final_state):
            return False
    
    return True