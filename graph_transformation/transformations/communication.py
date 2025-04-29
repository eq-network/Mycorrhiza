import numpy as np
import jax.numpy as jnp
from typing import Dict, Any

from core.graph_state import GraphState
from core.transformation import Transformation
from core.properties import Property, ConservesSum


class CommunicationUpdate(Transformation):
    """Determines which agents communicate with each other this round."""
    
    def __init__(self, seed_offset: int = 0):
        super().__init__()
        self.seed_offset = seed_offset
    
    def apply(self, state: GraphState) -> GraphState:
        trust = np.array(state.adj_matrices["trust"])
        
        # Probabilistically create communication links based on trust
        # In this simple model, agents are more likely to talk to agents they trust
        rng = np.random.RandomState(state.global_attrs.get("round", 0) + self.seed_offset)
        probabilities = trust.copy()
        communication = rng.binomial(1, probabilities)
        
        # Ensure diagonal is 0 (agents don't talk to themselves)
        communication = communication * (1 - np.eye(trust.shape[0]))
        
        return state.replace(
            adj_matrices={**state.adj_matrices, "communication": jnp.array(communication)}
        )


class TrustUpdate(Transformation):
    """Updates trust between agents based on their communication."""
    
    def __init__(self, decay_factor: float = 0.95):
        """
        Args:
            decay_factor: Factor by which trust decays when agents don't communicate
        """
        super().__init__()
        self.decay_factor = decay_factor
    
    def apply(self, state: GraphState) -> GraphState:
        trust = np.array(state.adj_matrices["trust"])
        communication = np.array(state.adj_matrices["communication"])
        
        # Apply trust updates based on communication
        # Trust increases for agents who communicated, decays slightly for others
        trust_updates = np.zeros_like(trust)
        trust_updates[communication > 0] = 0.05  # Small increase for communication
        
        # Apply updates and decay
        new_trust = trust * self.decay_factor + trust_updates
        
        # Normalize trust (excluding self)
        for i in range(trust.shape[0]):
            row_sum = np.sum(new_trust[i]) - new_trust[i, i]
            if row_sum > 0:
                new_trust[i] = new_trust[i] / row_sum * (1 - new_trust[i, i])
                new_trust[i, i] = 0
        
        return state.replace(
            adj_matrices={**state.adj_matrices, "trust": jnp.array(new_trust)}
        )