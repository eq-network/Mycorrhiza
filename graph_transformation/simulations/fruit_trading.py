# simulations/fruit_trading.py
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import jax.numpy as jnp
import time

from core.graph_state import GraphState
from core.transformation import Transformation, CompositeTransformation
from models.fruit_trading import (
    FruitTradingConfig, generate_preferences, 
    generate_initial_endowment, calculate_utility
)
from transformations.trading import (
    TradeOpportunitySearch, TradeMatching, 
    TradeNegotiation, UtilityCalculation
)
from services.llm import LLMClient
from services.trading_visualization import TradingVisualizer


class FruitTradingSimulation:
    """Orchestrates the fruit trading simulation."""
    
    def __init__(
        self, 
        config: Optional[FruitTradingConfig] = None,
        llm_client: Optional[LLMClient] = None,
        visualizer: Optional[TradingVisualizer] = None
    ):
        """
        Initialize the simulation.
        
        Args:
            config: Configuration for the simulation
            llm_client: Client for LLM negotiations (optional)
            visualizer: Visualization utilities
        """
        self.config = config or FruitTradingConfig()
        self.llm_client = llm_client
        self.visualizer = visualizer or TradingVisualizer()
        self.states = []
    
    def create_initial_state(self) -> GraphState:
        """Create the initial graph state for the simulation."""
        # Create random preferences
        preferences = generate_preferences(
            self.config.num_agents, 
            self.config.num_fruits
        )
        
        # Create initial endowments
        endowments = generate_initial_endowment(
            self.config.num_agents,
            self.config.num_fruits,
            self.config.initial_endowment_range
        )
        
        # Calculate initial utilities
        utilities = calculate_utility(preferences, endowments)
        
        # Create the graph state
        return GraphState(
            node_attrs={
                "fruit_preferences": jnp.array(preferences),
                "fruit_endowments": jnp.array(endowments),
                "utility": jnp.array(utilities)
            },
            adj_matrices={
                # Start with empty adjacency matrices
                # Trading opportunities will be calculated in first round
            },
            global_attrs={
                "round": 0,
                "fruit_names": self.config.fruits,
                "trade_history": [],
                "total_utility": float(np.sum(utilities))
            }
        )
    
    def run(self, verbose: bool = True, visualize: bool = True) -> GraphState:
        """
        Run the complete simulation.
        
        Args:
            verbose: Whether to print progress information
            visualize: Whether to show visualizations
            
        Returns:
            The final state after running all rounds
        """
        if verbose:
            print(f"Starting trading simulation with {self.config.num_agents} agents, "
                  f"{self.config.num_fruits} fruits, and {self.config.num_rounds} rounds")
            print(f"Fruits: {self.config.fruits}")
        
        # Create initial state
        state = self.create_initial_state()
        self.states = [state]
        
        if visualize:
            print("\nInitial state:")
            self.visualizer.visualize_preferences(state)
            self.visualizer.visualize_endowments(state, "Initial Endowments")
        
        # Define round transformation
        round_transformation = CompositeTransformation([
            TradeOpportunitySearch(),
            TradeMatching(),
            TradeNegotiation(self.llm_client),
            UtilityCalculation()
        ])
        
        # Run simulation rounds
        start_time = time.time()
        for round_num in range(self.config.num_rounds):
            if verbose:
                print(f"\n--- Round {round_num + 1} ---")
            
            # Apply round transformation
            state = round_transformation.apply(state)
            self.states.append(state)
            
            # Show visualizations periodically
            if visualize and (round_num + 1) % 3 == 0:
                print(f"\nState after round {round_num + 1}:")
                self.visualizer.visualize_endowments(
                    state, f"Endowments after Round {round_num + 1}"
                )
                self.visualizer.visualize_trading_network(state)
            
            # Report on utility
            if verbose:
                total_utility = state.global_attrs.get("total_utility", 0)
                trades_this_round = len(state.global_attrs["trade_history"][-1])
                print(f"Round {round_num + 1}: {trades_this_round} trades, Total Utility: {total_utility:.2f}")
        
        # Show final results
        execution_time = time.time() - start_time
        if verbose:
            print(f"\n=== Simulation completed in {execution_time:.2f} seconds ===")
            initial_utility = self.states[0].global_attrs.get("total_utility", 0)
            final_utility = state.global_attrs.get("total_utility", 0)
            utility_gain = final_utility - initial_utility
            utility_gain_pct = (utility_gain / initial_utility) * 100 if initial_utility > 0 else 0
            
            print(f"Initial Total Utility: {initial_utility:.2f}")
            print(f"Final Total Utility: {final_utility:.2f}")
            print(f"Utility Gain: {utility_gain:.2f} ({utility_gain_pct:.1f}%)")
            
            # Count total trades
            total_trades = sum(len(round_trades) for round_trades in state.global_attrs["trade_history"])
            print(f"Total Trades: {total_trades}")
        
        if visualize:
            print("\nFinal state:")
            self.visualizer.visualize_endowments(state, "Final Endowments")
            self.visualizer.visualize_trading_network(state)
            self.visualizer.visualize_utility_evolution(self.states)
            self.visualizer.visualize_trade_summary(self.states)
        
        return state
    
    def calculate_trading_efficiency(self) -> Dict[str, float]:
        """
        Calculate how efficient the trading was.
        
        Returns:
            Dictionary with efficiency metrics
        """
        if not self.states:
            return {"error": "No simulation data available"}
        
        initial_state = self.states[0]
        final_state = self.states[-1]
        
        initial_utility = initial_state.global_attrs.get("total_utility", 0)
        final_utility = final_state.global_attrs.get("total_utility", 0)
        
        # Calculate utility gain
        utility_gain = final_utility - initial_utility
        utility_gain_pct = (utility_gain / initial_utility) * 100 if initial_utility > 0 else 0
        
        # Count trades
        total_trades = sum(len(round_trades) for round_trades in final_state.global_attrs["trade_history"])
        
        # Calculate utility per trade
        utility_per_trade = utility_gain / total_trades if total_trades > 0 else 0
        
        # Calculate Pareto improvements
        # A trade is Pareto if both agents benefit (all our trades should be)
        agent_utilities_initial = np.array(initial_state.node_attrs["utility"])
        agent_utilities_final = np.array(final_state.node_attrs["utility"])
        agents_better_off = np.sum(agent_utilities_final > agent_utilities_initial)
        agents_worse_off = np.sum(agent_utilities_final < agent_utilities_initial)
        agents_same = np.sum(agent_utilities_final == agent_utilities_initial)
        
        return {
            "initial_utility": float(initial_utility),
            "final_utility": float(final_utility),
            "utility_gain": float(utility_gain),
            "utility_gain_pct": float(utility_gain_pct),
            "total_trades": total_trades,
            "utility_per_trade": float(utility_per_trade),
            "agents_better_off": int(agents_better_off),
            "agents_worse_off": int(agents_worse_off),
            "agents_same": int(agents_same)
        }
    
    def get_state_history(self) -> List[GraphState]:
        """Get the history of states from the simulation."""
        return self.states
    
    def get_trade_history(self) -> List[List[Dict]]:
        """Get the history of trades from the simulation."""
        if not self.states:
            return []
        
        return self.states[-1].global_attrs.get("trade_history", [])
    
    def save_results(self, filename: str):
        """Save simulation results to a file."""
        if not self.states:
            print("No simulation results to save.")
            return
        
        # Extract key information from states
        results = {
            "config": {
                "num_agents": self.config.num_agents,
                "num_fruits": self.config.num_fruits,
                "num_rounds": self.config.num_rounds,
                "fruits": self.config.fruits
            },
            "efficiency": self.calculate_trading_efficiency(),
            "rounds": []
        }
        
        # Add data for each round
        for i, state in enumerate(self.states):
            utilities = np.array(state.node_attrs["utility"])
            round_data = {
                "round": i,
                "total_utility": float(np.sum(utilities)),
                "agent_utilities": utilities.tolist(),
            }
            
            # Add trade details for non-initial rounds
            if i > 0 and i <= len(state.global_attrs.get("trade_history", [])):
                round_trades = state.global_attrs["trade_history"][i-1]
                round_data["trades"] = round_trades
            
            results["rounds"].append(round_data)
        
        # Save to file
        import json
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")


def calculate_optimal_allocation(
    preferences: np.ndarray, 
    endowments: np.ndarray
) -> np.ndarray:
    """
    Calculate the theoretically optimal allocation of fruits.
    
    This is a simplification that assumes:
    1. Total endowments remain fixed
    2. Each fruit should go entirely to agents who value it most
    
    Args:
        preferences: [num_agents, num_fruits] preference matrix
        endowments: [num_agents, num_fruits] initial endowments
        
    Returns:
        [num_agents, num_fruits] optimal allocation matrix
    """
    num_agents, num_fruits = preferences.shape
    
    # Calculate total supply of each fruit
    total_supply = np.sum(endowments, axis=0)
    
    # Create empty optimal allocation
    optimal = np.zeros_like(endowments)
    
    # For each fruit, allocate to agents in order of preference
    for fruit_idx in range(num_fruits):
        # Sort agents by preference for this fruit (highest first)
        agent_indices = np.argsort(-preferences[:, fruit_idx])
        
        # Allocate all supply to the agents who value it most
        remaining = total_supply[fruit_idx]
        agent_idx = 0
        
        while remaining > 0 and agent_idx < num_agents:
            # Give current agent as much as possible
            optimal[agent_indices[agent_idx], fruit_idx] = remaining
            remaining = 0
            agent_idx += 1
    
    return optimal