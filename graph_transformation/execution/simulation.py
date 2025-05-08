"""
Simulation execution module for the graph transformation framework.

This module provides functions for executing simulations by applying
transformations to graph states over multiple rounds, with support for
termination conditions and state history tracking.
"""
from typing import Dict, List, Tuple, Callable, Any, Optional
import jax
import jax.numpy as jnp
import jax.random as jr

from core.graph import GraphState
from core.category import Transform

# Type alias for PRNG keys
RandomKey = jnp.ndarray


def run_simulation(
    initial_state: GraphState,
    transform: Transform,
    num_rounds: int,
    key: RandomKey,
    termination_condition: Callable[[GraphState], bool] = None
) -> Tuple[GraphState, List[GraphState]]:
    """
    Execute a simulation by repeatedly applying a transformation.
    
    The simulation runs for the specified number of rounds or until
    the termination condition is met. State history is preserved
    for analysis.
    
    Args:
        initial_state: Starting graph state
        transform: Transformation to apply each round
        num_rounds: Maximum number of rounds to simulate
        key: JAX PRNG key for randomization
        termination_condition: Optional function that returns True
                              when simulation should stop
                              
    Returns:
        Tuple of (final_state, state_history)
    """
    # Default termination condition checks resources against threshold
    if termination_condition is None:
        def termination_condition(state: GraphState) -> bool:
            return state.global_attrs.get("total_resources", 0) < \
                   state.global_attrs.get("resource_min_threshold", 0)
    
    # Initialize state history with initial state
    state_history = [initial_state]
    current_state = initial_state
    
    # Split random key for each round
    subkeys = jr.split(key, num_rounds + 1)
    
    # Execute rounds
    for round_num in range(1, num_rounds + 1):
        # Update round counter in state
        current_state = current_state.update_global_attr("round", round_num)
        
        # Apply transformation
        current_state = transform(current_state)
        
        # Append to history
        state_history.append(current_state)
        
        # Check termination condition
        if termination_condition(current_state):
            break
    
    return current_state, state_history


def run_simulation_with_adversary_introduction(
    initial_state: GraphState,
    transform: Transform,
    num_rounds: int,
    key: RandomKey,
    adversarial_schedule: Dict[str, Any]
) -> Tuple[GraphState, List[GraphState]]:
    """
    Execute a simulation with gradual introduction of adversarial agents.
    
    This specialized simulation runner introduces adversarial agents
    according to the provided schedule, allowing for gradual corruption
    of the agent population.
    
    Args:
        initial_state: Starting graph state
        transform: Transformation to apply each round
        num_rounds: Maximum number of rounds to simulate
        key: JAX PRNG key for randomization
        adversarial_schedule: Configuration for adversary introduction
                             {
                                "mode": "linear|exponential|scheduled",
                                "target_proportion": float,
                                "schedule": Optional[List[int]] # For scheduled mode
                             }
                             
    Returns:
        Tuple of (final_state, state_history)
    """
    # Initialize state history with initial state
    state_history = [initial_state]
    current_state = initial_state
    
    # Split random key for rounds and adversary introductions
    key, round_key, adv_key = jr.split(key, 3)
    round_keys = jr.split(round_key, num_rounds)
    adv_keys = jr.split(adv_key, num_rounds)
    
    # Extract adversarial parameters
    mode = adversarial_schedule.get("mode", "linear")
    target_proportion = adversarial_schedule.get("target_proportion", 0.0)
    schedule = adversarial_schedule.get("schedule", [])
    
    # Determine number of agents
    num_agents = current_state.num_nodes
    
    # Current adversarial count
    is_adversarial = current_state.node_attrs.get("is_adversarial", 
                                                 jnp.zeros(num_agents, dtype=jnp.bool_))
    current_count = jnp.sum(is_adversarial)
    target_count = int(num_agents * target_proportion)
    
    # Execute rounds
    for round_num in range(1, num_rounds + 1):
        # Update round counter in state
        current_state = current_state.update_global_attr("round", round_num)
        
        # Determine if we need to introduce adversaries this round
        introduce_adversary = False
        if mode == "linear":
            # Linear increase over simulation duration
            target_for_round = int(target_count * round_num / num_rounds)
            introduce_adversary = target_for_round > current_count
        elif mode == "exponential":
            # Exponential increase (slower at first, faster later)
            ratio = round_num / num_rounds
            target_for_round = int(target_count * (ratio ** 2))
            introduce_adversary = target_for_round > current_count
        elif mode == "scheduled" and schedule:
            # Introduce according to specific schedule
            introduce_adversary = round_num in schedule
        
        # Introduce adversaries if needed
        if introduce_adversary:
            # Find non-adversarial agents
            non_adversarial = jnp.where(~is_adversarial)[0]
            if len(non_adversarial) > 0:
                # Select one random agent to convert
                idx = jr.choice(adv_keys[round_num-1], non_adversarial)
                is_adversarial = is_adversarial.at[idx].set(True)
                current_count += 1
                # Update state
                current_state = current_state.update_node_attrs("is_adversarial", is_adversarial)
        
        # Apply transformation
        current_state = transform(current_state)
        
        # Append to history
        state_history.append(current_state)
        
        # Check termination condition (resources below threshold)
        if current_state.global_attrs.get("total_resources", 0) < \
           current_state.global_attrs.get("resource_min_threshold", 0):
            break
    
    return current_state, state_history


def run_batched_simulations(
    initial_states: List[GraphState],
    transform: Transform,
    num_rounds: int,
    keys: List[RandomKey]
) -> List[Tuple[GraphState, List[GraphState]]]:
    """
    Execute multiple simulations in parallel.
    
    This function runs multiple independent simulations using JAX's
    vectorization capabilities for improved performance.
    
    Args:
        initial_states: List of starting graph states
        transform: Transformation to apply to each state
        num_rounds: Maximum number of rounds to simulate
        keys: List of random keys, one per simulation
        
    Returns:
        List of (final_state, state_history) tuples
    """
    # Run simulations in sequence for now
    # This could be optimized with JAX's vmap in the future
    results = []
    for initial_state, key in zip(initial_states, keys):
        final_state, state_history = run_simulation(
            initial_state, transform, num_rounds, key
        )
        results.append((final_state, state_history))
    
    return results