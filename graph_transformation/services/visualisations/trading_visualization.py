# services/trading_visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import networkx as nx
from matplotlib.ticker import MaxNLocator

from graph_transformation.core.graph import GraphState


class TradingVisualizer:
    """Visualization tools for the fruit trading simulation."""
    
    @staticmethod
    def visualize_endowments(state: GraphState, title: Optional[str] = None):
        """Visualize current fruit endowments."""
        endowments = np.array(state.node_attrs["fruit_endowments"])
        fruit_names = state.global_attrs["fruit_names"]
        num_agents = endowments.shape[0]
        
        plt.figure(figsize=(12, 8))
        
        # Plot individual endowments
        plt.subplot(1, 2, 1)
        sns.heatmap(endowments, annot=True, fmt="d", cmap="YlGnBu",
                    xticklabels=fruit_names, 
                    yticklabels=[f"Agent {i}" for i in range(num_agents)])
        plt.title("Individual Agent Endowments")
        
        # Plot aggregate endowments (total supply)
        plt.subplot(1, 2, 2)
        aggregate = np.sum(endowments, axis=0)
        plt.bar(fruit_names, aggregate)
        plt.title("Total Fruit Supply")
        
        if title:
            plt.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_preferences(state: GraphState):
        """Visualize agent preferences for fruits."""
        preferences = np.array(state.node_attrs["fruit_preferences"])
        fruit_names = state.global_attrs["fruit_names"]
        num_agents = preferences.shape[0]
        
        plt.figure(figsize=(12, 8))
        
        # Plot preference heatmap
        sns.heatmap(preferences, annot=True, fmt=".2f", cmap="Reds",
                    xticklabels=fruit_names,
                    yticklabels=[f"Agent {i}" for i in range(num_agents)])
        plt.title("Agent Preferences (Utility per Fruit)")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_trading_network(state: GraphState):
        """Visualize the trading network for the current round."""
        if "trading_matches" not in state.adj_matrices:
            print("No trading matches data in this state.")
            return
            
        matches = np.array(state.adj_matrices["trading_matches"])
        num_agents = matches.shape[0]
        
        # Create network graph
        G = nx.Graph()
        for i in range(num_agents):
            # Include utility in node label
            utility = state.node_attrs["utility"][i] if "utility" in state.node_attrs else 0
            G.add_node(i, label=f"Agent {i}\nU={utility:.1f}")
        
        # Add edges where trading occurs
        for i in range(num_agents):
            for j in range(i+1, num_agents):  # Only add each edge once
                if matches[i, j] > 0:
                    G.add_edge(i, j)
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes with utility-based size
        if "utility" in state.node_attrs:
            utilities = np.array(state.node_attrs["utility"])
            # Normalize utilities for node size (min 300, max 1500)
            if np.max(utilities) > np.min(utilities):
                norm_utils = (utilities - np.min(utilities)) / (np.max(utilities) - np.min(utilities))
                node_sizes = 300 + norm_utils * 1200
            else:
                node_sizes = [800] * num_agents
        else:
            node_sizes = [800] * num_agents
            
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightblue")
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color="darkblue")
        
        # Draw labels
        labels = {i: f"Agent {i}" for i in range(num_agents)}
        nx.draw_networkx_labels(G, pos, labels=labels)
        
        plt.title(f"Trading Network (Round {state.global_attrs.get('round', 0)})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_utility_evolution(states: List[GraphState]):
        """Visualize how utility evolved over the simulation."""
        if not states:
            print("No states to visualize.")
            return
            
        rounds = len(states)
        
        # Extract utility evolution
        utilities = np.zeros((rounds, states[0].node_attrs["utility"].shape[0]))
        total_utilities = np.zeros(rounds)
        
        for i, state in enumerate(states):
            utilities[i] = np.array(state.node_attrs["utility"])
            total_utilities[i] = state.global_attrs.get("total_utility", 0)
        
        # Plot individual utilities
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        for i in range(utilities.shape[1]):
            plt.plot(range(rounds), utilities[:, i], label=f"Agent {i}")
        
        plt.xlabel("Round")
        plt.ylabel("Utility")
        plt.title("Individual Agent Utility Evolution")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot total utility (social welfare)
        plt.subplot(2, 1, 2)
        plt.plot(range(rounds), total_utilities, 'b-', linewidth=2)
        plt.xlabel("Round")
        plt.ylabel("Total Utility")
        plt.title("Social Welfare Evolution")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Ensure y-axis starts at 0 for better visual comparison
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylim(bottom=0)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_trade_summary(states: List[GraphState]):
        """Visualize a summary of all trades that occurred."""
        if not states or not states[-1].global_attrs.get("trade_history"):
            print("No trade history available.")
            return
        
        trade_history = states[-1].global_attrs["trade_history"]
        fruit_names = states[-1].global_attrs["fruit_names"]
        
        # Count trades per round
        trades_per_round = [len(round_trades) for round_trades in trade_history]
        
        # Count trades per fruit
        fruits_traded = {fruit: 0 for fruit in fruit_names}
        
        for round_trades in trade_history:
            for trade in round_trades:
                fruits_traded[trade["fruit_i_name"]] += trade["amount_i"]
                fruits_traded[trade["fruit_j_name"]] += trade["amount_j"]
        
        # Plot trades per round
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.bar(range(1, len(trades_per_round) + 1), trades_per_round)
        plt.xlabel("Round")
        plt.ylabel("Number of Trades")
        plt.title("Trades Per Round")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(range(1, len(trades_per_round) + 1))
        
        # Plot fruits traded
        plt.subplot(2, 1, 2)
        fruits = list(fruits_traded.keys())
        counts = list(fruits_traded.values())
        
        plt.bar(fruits, counts)
        plt.xlabel("Fruit")
        plt.ylabel("Units Traded")
        plt.title("Total Units Traded by Fruit Type")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()