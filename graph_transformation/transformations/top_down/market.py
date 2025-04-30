# transformations/top_down/market.py
"""
Market mechanisms as top-down regularization.

This module defines transformations that implement various market-based
mechanisms for resource allocation and coordination.
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any, Callable, Set, Tuple, List
import jax.random as random

from core.graph import GraphState
from core.property import Property
from core.category import attach_properties

def trading_market(
    state: GraphState,
    preference_attr: str,
    resource_attr: str,
    trade_matcher: Callable = None,
    trade_executor: Callable = None,
    key: jnp.ndarray = None
) -> GraphState:
    """
    Implements a trading market where agents exchange resources.
    
    This transformation:
    1. Identifies potential trading pairs based on complementary preferences
    2. Executes trades that are mutually beneficial
    3. Updates resource allocations and records trades
    
    Args:
        state: Current graph state
        preference_attr: Node attribute containing resource preferences
        resource_attr: Node attribute containing resource holdings
        trade_matcher: Function to match trading partners (optional)
        trade_executor: Function to determine and execute trades (optional)
        key: JAX PRNG key for randomization
        
    Returns:
        Updated graph state after market activity
    """
    if preference_attr not in state.node_attrs or resource_attr not in state.node_attrs:
        return state
    
    preferences = state.node_attrs[preference_attr]
    resources = state.node_attrs[resource_attr]
    num_nodes = preferences.shape[0]
    
    # Default to random key if none provided
    if key is None:
        key = random.PRNGKey(0)
    
    # Default trade matcher - pairs agents with complementary preferences
    if trade_matcher is None:
        def trade_matcher(prefs, key):
            # Create potential pairs
            pairs = []
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    # Measure preference complementarity
                    # (agents who value different things are good trade partners)
                    if prefs.ndim > 1:
                        # Multi-resource case
                        diff = jnp.sum(jnp.abs(prefs[i] - prefs[j]))
                    else:
                        # Single resource case
                        diff = jnp.abs(prefs[i] - prefs[j])
                    
                    # Higher difference means more potential gains from trade
                    pairs.append((i, j, float(diff)))
            
            # Sort by potential gain from trade (highest first)
            pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Take top 30% of pairs
            count = max(1, len(pairs) // 3)
            return pairs[:count]
    
    # Default trade executor - simple resource exchange
    if trade_executor is None:
        def trade_executor(i, j, prefs, resources):
            if prefs.ndim > 1:
                # Multi-resource case
                # Find what each agent values most and least
                i_values_most = jnp.argmax(prefs[i])
                i_values_least = jnp.argmin(prefs[i])
                j_values_most = jnp.argmax(prefs[j])
                j_values_least = jnp.argmin(prefs[j])
                
                # Trade if complementary preferences
                if i_values_most == j_values_least and j_values_most == i_values_least:
                    # Check if both have resources to trade
                    if resources[i, i_values_least] > 0 and resources[j, j_values_least] > 0:
                        # Amount to trade (1 unit or half of holdings)
                        amount_i = min(1, resources[i, i_values_least] // 2)
                        amount_j = min(1, resources[j, j_values_least] // 2)
                        
                        # Update resources
                        new_resources = resources.copy()
                        new_resources = new_resources.at[i, i_values_least].add(-amount_i)
                        new_resources = new_resources.at[i, j_values_least].add(amount_j)
                        new_resources = new_resources.at[j, j_values_least].add(-amount_j)
                        new_resources = new_resources.at[j, i_values_least].add(amount_i)
                        
                        # Record trade details
                        trade = {
                            "agents": (i, j),
                            "resources": {
                                "from_i": (i_values_least, amount_i),
                                "from_j": (j_values_least, amount_j)
                            }
                        }
                        return new_resources, trade
            
            # No trade executed
            return resources, None
    
    # Find potential trading pairs
    trade_pairs = trade_matcher(preferences, key)
    
    # Execute trades
    new_resources = resources.copy()
    executed_trades = []
    
    for i, j, _ in trade_pairs:
        # Try to execute a trade between i and j
        updated_resources, trade_info = trade_executor(i, j, preferences, new_resources)
        
        if trade_info is not None:
            # Trade was executed
            new_resources = updated_resources
            executed_trades.append(trade_info)
    
    # Update global attributes with trade information
    new_global_attrs = dict(state.global_attrs)
    
    # Append new trades to history
    trade_history = new_global_attrs.get("trade_history", [])
    trade_history = trade_history + [executed_trades]
    new_global_attrs["trade_history"] = trade_history
    
    # Update trade count
    new_global_attrs["total_trades"] = new_global_attrs.get("total_trades", 0) + len(executed_trades)
    
    # Return updated state
    return state.replace(
        node_attrs={**state.node_attrs, resource_attr: new_resources},
        global_attrs=new_global_attrs
    )


def create_trading_market(
    preference_attr: str,
    resource_attr: str,
    trade_matcher: Callable = None,
    trade_executor: Callable = None,
    seed: int = 0,
    properties: Set[Property] = None
) -> Callable[[GraphState], GraphState]:
    """
    Create a trading market transformation.
    
    Args:
        preference_attr: Node attribute containing preferences
        resource_attr: Node attribute containing resources
        trade_matcher: Function to match trading partners
        trade_executor: Function to determine and execute trades
        seed: Random seed for trade matching
        properties: Properties preserved by this transformation
        
    Returns:
        A transformation function implementing a trading market
    """
    props = properties or set()
    key = random.PRNGKey(seed)
    
    def transform(state: GraphState) -> GraphState:
        nonlocal key
        key, subkey = random.split(key)
        return trading_market(
            state,
            preference_attr=preference_attr,
            resource_attr=resource_attr,
            trade_matcher=trade_matcher,
            trade_executor=trade_executor,
            key=subkey
        )
    
    return attach_properties(transform, props)