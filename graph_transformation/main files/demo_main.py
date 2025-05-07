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

def create_initial_state(config: ExperimentConfig, key: RandomKey) -> GraphState:
    """Map from experiment configuration to initialization parameters."""
    # Map configuration to parameter dictionaries
    adversarial_params = {
        "proportion": config.adversarial_proportion,
        "introduction": config.adversarial_introduction
    }
    
    crop_params = {
        "volatility_params": {
            config.yield_volatility: {"alpha": (5.0, 7.0), "beta": (3.0, 5.0)},
            "other_volatility": {"alpha": (2.0, 5.0), "beta": (1.0, 3.0)}
        }
    }
    
    # Add other parameter mappings as needed
    
    return initialize_democratic_graph_state(
        num_agents=config.num_agents,
        crops=config.crops,
        initial_resources=config.initial_resources,
        resource_min_threshold=config.resource_min_threshold,
        key=key,
        adversarial_params=adversarial_params,
        crop_params=crop_params
    )

def run_democratic_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a complete democratic mechanism experiment."""
    # Import only the domain-specific components needed
    from domains.democracy import construct_mechanism
    from execution import run_simulation
    from graph_transformation.analysis.democracy.democracy import compute_metrics, aggregate_results
    
    # Prepare transformation pipeline
    transform = construct_mechanism(config.mechanism_type, config)
    
    # Run trials
    trial_results = []
    for trial in range(config.num_trials):
        # Run simulation and collect states
        final_state, state_history = run_simulation(
            config=config,
            transform=transform,
            trial=trial
        )
        
        # Compute metrics (in separate analysis module)
        metrics = compute_metrics(config, final_state, state_history)
        trial_results.append(metrics)
    
    # Aggregate results (in separate analysis module)
    aggregated_results = aggregate_results(trial_results)
    
    return {
        "config": config,
        "aggregated_results": aggregated_results,
        "trial_results": trial_results
    }