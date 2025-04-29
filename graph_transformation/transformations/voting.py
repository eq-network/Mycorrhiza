import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List, Optional

from core.graph_state import GraphState
from core.transformation import Transformation
from core.properties import Property


class VotingResults(Transformation):
    """Calculate voting results based on agent preferences."""
    
    def apply(self, state: GraphState) -> GraphState:
        preferences = state.node_attrs["fruit_preferences"]
        
        # Sum preferences across all agents
        total_votes = jnp.sum(preferences, axis=0)
        
        # Normalize to get overall vote percentages
        vote_percentages = total_votes / jnp.sum(total_votes)
        
        # Find the winning fruit
        winning_idx = jnp.argmax(vote_percentages)
        winning_fruit = state.global_attrs["fruit_names"][winning_idx]
        
        # Prepare detailed results
        voting_results = {
            "final_votes": np.array(vote_percentages),
            "winning_fruit": winning_fruit,
            "winning_idx": int(winning_idx),
            "vote_distribution": {
                state.global_attrs["fruit_names"][i]: float(vote_percentages[i])
                for i in range(len(vote_percentages))
            }
        }
        
        return state.replace(
            global_attrs={
                **state.global_attrs,
                "voting_results": voting_results
            }
        )


class DelegatedVoting(Transformation):
    """
    Implements a delegated voting mechanism where agents can delegate 
    their voting power to others.
    """
    
    def apply(self, state: GraphState) -> GraphState:
        preferences = np.array(state.node_attrs["fruit_preferences"]) 
        delegation = np.array(state.adj_matrices.get("delegation", 
                                                    np.zeros((preferences.shape[0], preferences.shape[0]))))
        
        # Calculate effective voting power through delegation
        voting_power = np.ones(preferences.shape[0])
        
        # Delegation can be multi-level, so we need to propagate it
        # This is a simplified algorithm that doesn't handle cycles
        for _ in range(preferences.shape[0]):  # Max delegation depth
            new_power = voting_power.copy()
            for i in range(preferences.shape[0]):
                for j in range(preferences.shape[0]):
                    if delegation[i, j] > 0:
                        # Transfer voting power
                        new_power[j] += voting_power[i]
                        new_power[i] = 0
            voting_power = new_power
            
        # Calculate weighted preferences
        weighted_preferences = np.zeros_like(preferences[0])
        for i in range(preferences.shape[0]):
            weighted_preferences += voting_power[i] * preferences[i]
            
        # Normalize
        if np.sum(weighted_preferences) > 0:
            weighted_preferences = weighted_preferences / np.sum(weighted_preferences)
        
        # Find winner
        winning_idx = np.argmax(weighted_preferences)
        winning_fruit = state.global_attrs["fruit_names"][winning_idx]
        
        # Prepare results
        voting_results = {
            "final_votes": weighted_preferences,
            "winning_fruit": winning_fruit,
            "winning_idx": int(winning_idx),
            "vote_distribution": {
                state.global_attrs["fruit_names"][i]: float(weighted_preferences[i])
                for i in range(len(weighted_preferences))
            },
            "voting_power": voting_power
        }
        
        return state.replace(
            global_attrs={
                **state.global_attrs,
                "voting_results": voting_results
            }
        )