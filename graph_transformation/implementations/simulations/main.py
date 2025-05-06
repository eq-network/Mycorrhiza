from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr

from core.graph import GraphState
from core.category import sequential, compose, attach_properties, jit_transform
from core.property import ConservesSum, Property

# Type aliases for clean type signatures
Transform = Callable[[GraphState], GraphState]
RandomKey = jnp.ndarray


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration for democratic mechanism experiments."""
    # Mechanism configuration
    mechanism_type: str  # "PDD", "PRD", or "PLD"
    
    # Population parameters
    num_agents: int
    adversarial_proportion: float
    adversarial_introduction: str  # "immediate" or "gradual"
    
    # Environment parameters
    num_rounds: int
    initial_resources: int
    resource_min_threshold: int
    crops: List[str]
    yield_volatility: str  # "stable" or "variable"
    
    # Execution parameters
    random_seed: int
    num_trials: int
    jit_compile: bool = True
    
    # Analysis flags
    track_delegation_metrics: bool = True
    track_resource_metrics: bool = True
    track_information_metrics: bool = True


def run_democratic_experiment(
    config: ExperimentConfig
) -> Dict[str, Any]:
    """
    Orchestrates a complete democratic mechanism experiment according to specified configuration.
    
    This function serves as the compilation layer between the mathematical process definition 
    (transformations) and the computational execution (simulation runs). It maintains strict 
    separation between configuration, process composition, execution strategy, and result collection.
    
    Args:
        config: Immutable experiment configuration dataclass
        
    Returns:
        Dict containing structured experiment results and analysis metrics
    """
    # Initialize master random key from seed
    master_key = jr.PRNGKey(config.random_seed)
    
    # Prepare transformation pipeline based on mechanism type
    transform_pipeline = _construct_mechanism_pipeline(config)
    
    # Apply optimization if requested
    if config.jit_compile:
        transform_pipeline = jit_transform(transform_pipeline)
    
    # Execute trials with different random subkeys
    trial_results = []
    
    for trial in range(config.num_trials):
        # Generate trial-specific key
        master_key, trial_key = jr.split(master_key)
        
        # Initialize graph state for this trial
        initial_state = _initialize_graph_state(config, trial_key)
        
        # Execute simulation for this trial
        final_state, state_history = _execute_simulation(
            initial_state=initial_state,
            transform=transform_pipeline,
            num_rounds=config.num_rounds,
            key=trial_key
        )
        
        # Compute trial metrics
        trial_metrics = _compute_trial_metrics(
            config=config,
            final_state=final_state,
            state_history=state_history
        )
        
        trial_results.append(trial_metrics)
    
    # Aggregate results across trials
    aggregated_results = _aggregate_trial_results(trial_results)
    
    # Return structured results with appropriate metadata
    return {
        "config": config,
        "aggregated_results": aggregated_results,
        "trial_results": trial_results
    }


def _construct_mechanism_pipeline(config: ExperimentConfig) -> Transform:
    """
    Constructs the appropriate transformation pipeline based on mechanism type.
    
    This function composes the elementary transformations into mechanism-specific
    pipelines that implement PDD, PRD, or PLD according to the specification.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Composed transformation implementing the specified democratic mechanism
    """
    # Common transformations across all mechanisms
    information_sharing = _create_information_sharing_transform(config)
    prediction_market = _create_prediction_market_transform(config)
    resource_application = _create_resource_application_transform(config)
    
    # Mechanism-specific transformations and composition
    if config.mechanism_type == "PDD":
        # Predictive Direct Democracy: all agents vote directly
        direct_voting = _create_direct_voting_transform(config)
        
        return sequential(
            information_sharing,
            prediction_market,
            direct_voting,
            resource_application
        )
        
    elif config.mechanism_type == "PRD":
        # Predictive Representative Democracy: fixed representatives vote
        representative_selection = _create_representative_selection_transform(config)
        representative_voting = _create_representative_voting_transform(config)
        
        return sequential(
            information_sharing,
            prediction_market,
            representative_selection,
            representative_voting,
            resource_application
        )
        
    elif config.mechanism_type == "PLD":
        # Predictive Liquid Democracy: delegates vote with weighted power
        delegation = _create_delegation_transform(config)
        voting_power_calculation = _create_voting_power_calculation_transform(config)
        liquid_voting = _create_liquid_voting_transform(config)
        
        # Property: voting power is conserved through the delegation process
        conserves_voting_power = ConservesSum("voting_power")
        voting_power_calculation = attach_properties(
            voting_power_calculation, {conserves_voting_power}
        )
        
        return sequential(
            information_sharing,
            delegation,
            voting_power_calculation,
            prediction_market,
            liquid_voting,
            resource_application
        )
    
    else:
        raise ValueError(f"Unknown mechanism type: {config.mechanism_type}")


def _initialize_graph_state(config: ExperimentConfig, key: RandomKey) -> GraphState:
    """
    Initializes the graph state based on experiment configuration.
    
    Creates the initial population with specified proportion of adversarial agents,
    initializes resource levels, crop distributions, and required adjacency matrices.
    
    Args:
        config: Experiment configuration
        key: Random key for initialization
        
    Returns:
        Initial graph state for simulation
    """
    # This function would call the appropriate initialization services
    # For now, we assume it exists and return the expected interface
    
    # The real implementation would:
    # 1. Create appropriate node attributes for agents
    # 2. Initialize adjacency matrices for communication/delegation
    # 3. Set up global attributes for resources and environment
    
    # Number of adversarial agents
    num_adversarial = int(config.num_agents * config.adversarial_proportion)
    
    # We'll assume this function exists and provides the initialization
    return initialize_democratic_simulation_state(
        num_agents=config.num_agents,
        num_adversarial=num_adversarial,
        crops=config.crops,
        initial_resources=config.initial_resources,
        adversarial_introduction=config.adversarial_introduction,
        yield_volatility=config.yield_volatility,
        random_key=key
    )


def _execute_simulation(
    initial_state: GraphState,
    transform: Transform,
    num_rounds: int,
    key: RandomKey
) -> Tuple[GraphState, List[GraphState]]:
    """
    Executes the simulation for the specified number of rounds.
    
    Applies the transformation pipeline repeatedly, preserving history
    and checking termination conditions.
    
    Args:
        initial_state: Starting graph state
        transform: Composed transformation to apply each round
        num_rounds: Maximum number of rounds
        key: Random key for stochastic processes
        
    Returns:
        Tuple of (final_state, state_history)
    """
    # Initialize state history with initial state
    state_history = [initial_state]
    current_state = initial_state
    
    # Split random key for each round
    subkeys = jr.split(key, num_rounds + 1)
    
    # Execute rounds
    for round_num in range(1, num_rounds + 1):
        # Update round counter in state
        current_state = current_state.update_global_attr("round", round_num)
        
        # Apply transformation with appropriate randomness
        current_state = transform(current_state)
        
        # Append to history
        state_history.append(current_state)
        
        # Check termination condition (resources below threshold)
        if current_state.global_attrs.get("total_resources", 0) < current_state.global_attrs.get("resource_min_threshold", 0):
            break
    
    return current_state, state_history


def _compute_trial_metrics(
    config: ExperimentConfig,
    final_state: GraphState,
    state_history: List[GraphState]
) -> Dict[str, Any]:
    """
    Computes metrics for a single trial based on configuration flags.
    
    Args:
        config: Experiment configuration 
        final_state: Final state of the simulation
        state_history: Complete history of states
        
    Returns:
        Dict of computed metrics
    """
    metrics = {
        "survival_rounds": len(state_history) - 1,
        "survived": final_state.global_attrs.get("total_resources", 0) >= 
                   final_state.global_attrs.get("resource_min_threshold", 0),
        "final_resources": final_state.global_attrs.get("total_resources", 0),
    }
    
    # Extract mechanism-specific metrics
    if config.mechanism_type == "PLD" and config.track_delegation_metrics:
        metrics.update(_compute_delegation_metrics(final_state, state_history))
    
    # Add resource metrics if requested
    if config.track_resource_metrics:
        metrics.update(_compute_resource_metrics(final_state, state_history))
        
    # Add information metrics if requested
    if config.track_information_metrics:
        metrics.update(_compute_information_metrics(final_state, state_history))
    
    return metrics


def _aggregate_trial_results(trial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates results across all trials.
    
    Computes statistics like mean, median, min, max, and standard deviation
    for numerical metrics across all trials.
    
    Args:
        trial_results: List of metrics from each trial
        
    Returns:
        Dict of aggregated metrics
    """
    # This would compute statistics across trials
    # We'd aggregate key metrics like survival rate, average resources, etc.
    return aggregate_experiment_results(trial_results)


# Example usage of the orchestration function
if __name__ == "__main__":
    # Define experiment configuration
    experiment_config = ExperimentConfig(
        mechanism_type="PLD",
        num_agents=100,
        adversarial_proportion=0.3,
        adversarial_introduction="gradual",
        num_rounds=30,
        initial_resources=1000,
        resource_min_threshold=500,
        crops=["Wheat", "Corn", "Rice", "Potatoes", "Soybeans"],
        yield_volatility="variable",
        random_seed=42,
        num_trials=10,
        jit_compile=True,
        track_delegation_metrics=True
    )
    
    # Run experiment
    results = run_democratic_experiment(experiment_config)
    
    # Comparative analysis across mechanisms
    mechanism_comparison = []
    for mechanism in ["PDD", "PRD", "PLD"]:
        # Create configuration variant for this mechanism
        mechanism_config = ExperimentConfig(
            mechanism_type=mechanism,
            num_agents=100,
            adversarial_proportion=0.3,
            adversarial_introduction="gradual",
            num_rounds=30,
            initial_resources=1000,
            resource_min_threshold=500,
            crops=["Wheat", "Corn", "Rice", "Potatoes", "Soybeans"],
            yield_volatility="variable",
            random_seed=42,
            num_trials=10
        )
        
        # Run and store results
        mechanism_results = run_democratic_experiment(mechanism_config)
        mechanism_comparison.append(mechanism_results)
    
    # Generate visualizations and comparative analysis
    generate_mechanism_comparison_report(mechanism_comparison)