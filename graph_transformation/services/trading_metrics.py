# services/trading_metrics.py
import numpy as np
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

from core.graph_state import GraphState


class TradingMetricsAnalyzer:
    """
    Class for analyzing the results of trading simulations using metrics
    inspired by information theory and economic analysis.
    """
    
    @staticmethod
    def analyze_simulation(states: List[GraphState]) -> Dict[str, Any]:
        """
        Analyze a completed simulation, calculating key metrics.
        
        Args:
            states: List of graph states from the simulation
            
        Returns:
            Dictionary containing all calculated metrics
        """
        if not states:
            return {"error": "No simulation data provided"}
        
        # Calculate basic metrics
        basic_metrics = TradingMetricsAnalyzer.calculate_basic_metrics(states)
        
        # Calculate allocation efficiency
        allocation_metrics = TradingMetricsAnalyzer.calculate_allocation_efficiency(states)
        
        # Calculate trading network metrics
        network_metrics = TradingMetricsAnalyzer.analyze_trading_network(states)
        
        # Calculate information-theoretic metrics
        info_metrics = TradingMetricsAnalyzer.calculate_information_metrics(states)
        
        # Combine all metrics
        return {
            **basic_metrics,
            **allocation_metrics,
            **network_metrics,
            **info_metrics
        }
    
    @staticmethod
    def calculate_basic_metrics(states: List[GraphState]) -> Dict[str, float]:
        """
        Calculate basic metrics for the simulation.
        
        Args:
            states: List of graph states from the simulation
            
        Returns:
            Dictionary of metrics
        """
        initial_state = states[0]
        final_state = states[-1]
        num_agents = initial_state.node_attrs["utility"].shape[0]
        
        # Calculate utility metrics
        initial_utilities = np.array(initial_state.node_attrs["utility"])
        final_utilities = np.array(final_state.node_attrs["utility"])
        
        utility_gain = np.sum(final_utilities) - np.sum(initial_utilities)
        utility_gain_pct = utility_gain / np.sum(initial_utilities) if np.sum(initial_utilities) > 0 else 0
        
        # Count trades
        trade_history = final_state.global_attrs.get("trade_history", [])
        total_trades = sum(len(round_trades) for round_trades in trade_history)
        trades_per_agent = total_trades / num_agents if num_agents > 0 else 0
        utility_per_trade = utility_gain / total_trades if total_trades > 0 else 0
        
        # Calculate Gini coefficient for final utility distribution
        gini_coefficient = TradingMetricsAnalyzer.gini(final_utilities)
        
        return {
            "initial_total_utility": float(np.sum(initial_utilities)),
            "final_total_utility": float(np.sum(final_utilities)),
            "utility_gain": float(utility_gain),
            "utility_gain_pct": float(utility_gain_pct * 100),  # As percentage
            "utility_per_trade": float(utility_per_trade),
            "total_trades": total_trades,
            "trades_per_agent": float(trades_per_agent),
            "gini_coefficient": float(gini_coefficient)
        }
    
    @staticmethod
    def calculate_allocation_efficiency(states: List[GraphState]) -> Dict[str, float]:
        """
        Calculate metrics related to allocation efficiency.
        
        Args:
            states: List of graph states from the simulation
            
        Returns:
            Dictionary of efficiency metrics
        """
        initial_state = states[0]
        final_state = states[-1]
        
        # Extract preferences and endowments
        preferences = np.array(initial_state.node_attrs["fruit_preferences"])
        initial_endowments = np.array(initial_state.node_attrs["fruit_endowments"])
        final_endowments = np.array(final_state.node_attrs["fruit_endowments"])
        
        # Calculate total supply of each fruit (should remain constant)
        initial_supply = np.sum(initial_endowments, axis=0)
        final_supply = np.sum(final_endowments, axis=0)
        
        # Verify conservation of total supply (for consistency check)
        supply_conserved = np.allclose(initial_supply, final_supply)
        
        # Calculate a simple allocation efficiency metric:
        # For each fruit, what percentage is held by the top 20% of agents who value it most?
        
        num_agents, num_fruits = preferences.shape
        top_agent_count = max(1, int(num_agents * 0.2))  # Top 20% of agents
        
        initial_top_allocation = np.zeros(num_fruits)
        final_top_allocation = np.zeros(num_fruits)
        
        for fruit_idx in range(num_fruits):
            # Sort agents by preference for this fruit (highest first)
            sorted_agents = np.argsort(-preferences[:, fruit_idx])
            top_agents = sorted_agents[:top_agent_count]
            
            # Calculate what percentage of this fruit is held by top agents
            initial_top_holding = np.sum(initial_endowments[top_agents, fruit_idx])
            final_top_holding = np.sum(final_endowments[top_agents, fruit_idx])
            
            initial_top_allocation[fruit_idx] = initial_top_holding / initial_supply[fruit_idx]
            final_top_allocation[fruit_idx] = final_top_holding / final_supply[fruit_idx]
        
        # Average across all fruits
        avg_initial_top_allocation = np.mean(initial_top_allocation)
        avg_final_top_allocation = np.mean(final_top_allocation)
        
        # Calculate preference-weighted allocation improvement
        allocation_improvement = float(avg_final_top_allocation - avg_initial_top_allocation)
        
        # Calculate a "preference alignment" score
        # This measures how well fruit allocation correlates with preferences
        initial_alignment = 0.0
        final_alignment = 0.0
        
        for agent_idx in range(num_agents):
            # Calculate correlation between preferences and holdings
            # (Higher correlation means fruits agents prefer more are the ones they have more of)
            initial_corr = np.corrcoef(preferences[agent_idx], initial_endowments[agent_idx])[0, 1]
            final_corr = np.corrcoef(preferences[agent_idx], final_endowments[agent_idx])[0, 1]
            
            # Handle NaN cases (can occur if all preferences or endowments are identical)
            if np.isnan(initial_corr):
                initial_corr = 0.0
            if np.isnan(final_corr):
                final_corr = 0.0
                
            initial_alignment += initial_corr
            final_alignment += final_corr
        
        # Average across all agents
        initial_alignment /= num_agents
        final_alignment /= num_agents
        alignment_improvement = final_alignment - initial_alignment
        
        return {
            "supply_conserved": supply_conserved,
            "initial_top_allocation_pct": float(avg_initial_top_allocation * 100),  # As percentage
            "final_top_allocation_pct": float(avg_final_top_allocation * 100),      # As percentage
            "allocation_improvement_pct": float(allocation_improvement * 100),       # As percentage
            "initial_preference_alignment": float(initial_alignment),
            "final_preference_alignment": float(final_alignment),
            "alignment_improvement": float(alignment_improvement)
        }