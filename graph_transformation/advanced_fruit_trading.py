# advanced_fruit_trading.py
"""
Extended demonstration of the fruit trading simulation with advanced metrics.

This script runs multiple configurations of the trading simulation and analyzes
the results using information-theoretic and economic metrics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from models.fruit_trading import FruitTradingConfig
from services.llm import LLMClient
from services.trading_visualization import TradingVisualizer
from services.trading_metrics import TradingMetricsAnalyzer
from simulations.fruit_trading import FruitTradingSimulation


def run_simulation(config, llm_client=None, verbose=True, visualize=True):
    """Run a single simulation with the given configuration."""
    # Create visualizer
    visualizer = TradingVisualizer()
    
    # Create and run simulation
    simulation = FruitTradingSimulation(
        config=config,
        llm_client=llm_client,
        visualizer=visualizer
    )
    
    # Run the simulation
    final_state = simulation.run(verbose=verbose, visualize=visualize)
    
    # Analyze simulation metrics
    metrics = TradingMetricsAnalyzer.analyze_simulation(simulation.get_state_history())
    
    # If visualization is enabled, show metric summary
    if visualize:
        TradingMetricsAnalyzer.visualize_metrics(metrics, 
                                                 f"Trading Metrics: {config.num_agents} Agents, {config.num_rounds} Rounds")
        TradingMetricsAnalyzer.visualize_efficiency_comparison(metrics)
    
    return simulation, metrics


def compare_simulations(simulations, metrics_list, labels, title="Simulation Comparison"):
    """Compare metrics across multiple simulations."""
    # Select key metrics to compare
    key_metrics = [
        "utility_gain_pct",
        "final_preference_alignment",
        "final_information_efficiency",
        "network_density",
        "trades_per_agent"
    ]
    
    # Extract metrics
    comparison_data = {}
    for metric in key_metrics:
        comparison_data[metric] = [metrics.get(metric, 0) for metrics in metrics_list]
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Make a separate subplot for each metric
    for i, metric in enumerate(key_metrics):
        plt.subplot(1, len(key_metrics), i+1)
        plt.bar(labels, comparison_data[metric])
        plt.title(metric.replace("_", " ").title())
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
    plt.show()


def main():
    """Run multiple fruit trading simulations and compare results."""
    # Get LLM client if API key is available
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    
    if openrouter_api_key:
        print("Using OpenRouter for trade negotiations")
        llm_client = LLMClient(
            provider="openrouter",
            api_key=openrouter_api_key,
            model="anthropic/claude-3-sonnet"
        )
    else:
        print("OPENROUTER_API_KEY not found, using algorithmic trading")
        llm_client = None
    
    # Define simulation configurations to compare
    configs = [
        # Base configuration
        FruitTradingConfig(
            num_agents=8,
            num_fruits=5,
            num_rounds=10,
            fruits=["Apple", "Banana", "Cherry", "Durian", "Elderberry"],
            initial_endowment_range=(5, 15)
        ),
        
        # Larger network
        FruitTradingConfig(
            num_agents=16,
            num_fruits=5,
            num_rounds=10,
            fruits=["Apple", "Banana", "Cherry", "Durian", "Elderberry"],
            initial_endowment_range=(5, 15)
        ),
        
        # More trading rounds
        FruitTradingConfig(
            num_agents=8,
            num_fruits=5,
            num_rounds=20,
            fruits=["Apple", "Banana", "Cherry", "Durian", "Elderberry"],
            initial_endowment_range=(5, 15)
        )
    ]
    
    config_labels = ["Base (8 agents)", "Large (16 agents)", "Extended (20 rounds)"]
    
    # Run simulations
    print("=== Running Multiple Fruit Trading Simulations ===")
    
    simulations = []
    metrics_list = []
    
    for i, config in enumerate(configs):
        print(f"\n\n=== Configuration {i+1}: {config_labels[i]} ===")
        # Run with full visualization only for first simulation
        visualize = (i == 0)
        sim, metrics = run_simulation(config, llm_client, verbose=True, visualize=visualize)
        simulations.append(sim)
        metrics_list.append(metrics)
        
        # Save results for this configuration
        sim.save_results(f"fruit_trading_results_{i+1}.json")