# transformations/bottom_up/message_passing.py
"""
Message passing transformation - the foundational bottom-up mechanism.

This module defines how agents communicate with connected agents in the network,
without specifying the content of those communications.
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any, Callable, Set

from core.graph import GraphState
from core.property import Property
from core.category import attach_properties

def message_passing(
    state: GraphState,
    connection_type: str = "communication",
    message_generator: Callable = None,
    message_processor: Callable = None
) -> GraphState:
    """
    Fundamental message passing transformation where connected agents communicate.
    
    Rather than specifying the content of communication, this provides a general
    framework where:
    1. Agents generate messages based on their state
    2. Messages are passed along connections
    3. Agents process received messages and update their state
    
    Args:
        state: Current graph state
        connection_type: Edge type representing communication channels
        message_generator: Function to generate messages from node state (optional)
        message_processor: Function to process received messages (optional)
        
    Returns:
        Updated graph state after communication
    """
    if connection_type not in state.adj_matrices:
        return state
    
    # Get adjacency matrix representing connections
    connections = state.adj_matrices[connection_type]
    num_nodes = state.num_nodes
    
    # Default message generator just passes node attributes
    if message_generator is None:
        def message_generator(node_idx, node_attrs):
            return node_attrs
    
    # Default processor averages received messages
    if message_processor is None:
        def message_processor(node_idx, node_attrs, messages):
            if not messages:
                return node_attrs
            # Average all received messages for each attribute
            result = {}
            for attr in node_attrs:
                if all(attr in msg for msg in messages):
                    # Simple averaging of values
                    result[attr] = sum(msg[attr] for msg in messages) / len(messages)
                else:
                    result[attr] = node_attrs[attr]
            return result
    
    # For each node, generate messages
    messages = {}
    for i in range(num_nodes):
        node_attrs = {k: v[i] for k, v in state.node_attrs.items()}
        messages[i] = message_generator(i, node_attrs)
    
    # Determine which nodes receive which messages based on connections
    received_messages = {i: [] for i in range(num_nodes)}
    for sender in range(num_nodes):
        for receiver in range(num_nodes):
            if connections[sender, receiver] > 0:
                received_messages[receiver].append(messages[sender])
    
    # Process received messages and update node states
    new_node_attrs = {k: v.copy() for k, v in state.node_attrs.items()}
    for i in range(num_nodes):
        node_attrs = {k: v[i] for k, v in state.node_attrs.items()}
        updated_attrs = message_processor(i, node_attrs, received_messages[i])
        
        # Update attribute arrays with processed values
        for attr, value in updated_attrs.items():
            if attr in new_node_attrs:
                new_node_attrs[attr] = new_node_attrs[attr].at[i].set(value)
    
    return state.replace(node_attrs=new_node_attrs)


def create_message_passing(
    connection_type: str = "communication",
    message_generator: Callable = None,
    message_processor: Callable = None,
    properties: Set[Property] = None
) -> Callable[[GraphState], GraphState]:
    """
    Create a message passing transformation with custom behaviors.
    
    This factory function allows defining specific types of agent communication
    by providing custom message generation and processing functions.
    
    Args:
        connection_type: Edge type for communication channels
        message_generator: Function to generate messages from node state
        message_processor: Function to process received messages
        properties: Properties preserved by this transformation
        
    Returns:
        A transformation function for message passing
    """
    props = properties or set()
    
    def transform(state: GraphState) -> GraphState:
        return message_passing(
            state, 
            connection_type=connection_type,
            message_generator=message_generator,
            message_processor=message_processor
        )
    
    return attach_properties(transform, props)