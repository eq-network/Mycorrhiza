from typing import List, Dict, Any
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass

@dataclass
class FruitTradingConfig:
    """Configuration for the fruit trading simulation."""
    num_agents: int = 10
    num_fruits: int = 5
    num_rounds: int = 10
    fruits: List[str] = None
    initial_endowment_range: tuple = (5, 15)  # Range of initial fruit counts
    
    def __post_init__(self):
        if self.fruits is None:
            self.fruits = ["Apple", "Banana", "Cherry", "Durian", "Elderberry"]
        
        # Validate
        assert len(self.fruits) == self.num_fruits, "Number of fruits must match num_fruits"


def generate_preferences(num_agents: int, num_fruits: int) -> np.ndarray:
    """
    Generate random normalized preferences.
    Shape: [num_agents, num_fruits]
    Values: How much utility an agent gets per unit of each fruit
    """
    # Generate random preference values (1-10 scale)
    preferences = np.random.uniform(1, 10, (num_agents, num_fruits))
    # No need to normalize - these are absolute utility values per fruit
    return preferences


def generate_initial_endowment(num_agents: int, num_fruits: int, 
                              endowment_range: tuple) -> np.ndarray:
    """
    Generate initial fruit endowments for each agent.
    Shape: [num_agents, num_fruits]
    Values: Quantities of each fruit owned
    """
    min_val, max_val = endowment_range
    # Generate random integer quantities of fruits
    endowments = np.random.randint(min_val, max_val+1, (num_agents, num_fruits))
    return endowments


def calculate_utility(preferences: np.ndarray, endowments: np.ndarray) -> np.ndarray:
    """
    Calculate utility for each agent based on preferences and endowments.
    
    Args:
        preferences: [num_agents, num_fruits] array of preference weights
        endowments: [num_agents, num_fruits] array of fruit quantities
    
    Returns:
        [num_agents] array of utility values
    """
    # Utility = sum(preference_i * endowment_i) for each fruit i
    return np.sum(preferences * endowments, axis=1)