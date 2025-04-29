# main_fruit_trading.py
"""
Example usage of the fruit trading simulation.

Run this script to see a complete simulation of the fruit trading game.
"""

import os
from models.fruit_trading import FruitTradingConfig
from services.llm import LLMClient
from services.trading_visualization import TradingVisualizer
from simulations.fruit_trading import FruitTradingSimulation

def main():
    """Run a demonstration of the fruit trading simulation."""
    # Create configuration
    config = FruitTradingConfig(
        num_agents=8,
        num_fruits=5,
        num_rounds=10,
        fruits=["Apple", "Banana", "Cherry", "Durian", "Elderberry"],
        initial_endowment_range=(5, 15)  # Initial amount of each fruit agents have
    )
    
    # Create LLM client with OpenRouter for more realistic trade negotiations
    # Get API key from environment variable or set it directly
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    
    if openrouter_api_key:
        print("Using OpenRouter for trade negotiations")
        llm_client = LLMClient(
            provider="openrouter",
            api_key=openrouter_api_key,
            model="anthropic/claude-3-sonnet"  # Change to your preferred model
        )
    else:
        print("OPENROUTER_API_KEY not found, using algorithmic trading")
        llm_client = None
    
    # Create visualizer
    visualizer = TradingVisualizer()
    
    # Create and run simulation
    simulation = FruitTradingSimulation(
        config=config,
        llm_client=llm_client,
        visualizer=visualizer
    )
    
    print("=== Running Fruit Trading Simulation ===")
    final_state = simulation.run(verbose=True, visualize=True)
    
    # Calculate and display efficiency metrics
    efficiency = simulation.calculate_trading_efficiency()
    print("\n=== Trading Efficiency Metrics ===")
    for metric, value in efficiency.items():
        print(f"{metric}: {value}")
    
    # Save results
    simulation.save_results("fruit_trading_results.json")
    
    print("\nSimulation complete. Results saved to fruit_trading_results.json")
    
    return simulation

if __name__ == "__main__":
    simulation = main()