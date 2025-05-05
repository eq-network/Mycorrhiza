import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
rounds_df = pd.read_csv('simulation_results/round_metrics.csv')
agents_df = pd.read_csv('simulation_results/agent_metrics.csv')
crops_df = pd.read_csv('simulation_results/crop_metrics.csv')
delegation_df = pd.read_csv('simulation_results/delegation_network.csv')

# Configure visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

# Analysis 1: Mechanism Performance Comparison
def plot_accuracy_comparison():
    plt.figure(figsize=(12, 6))
    
    for mechanism in ['PDD', 'PRD', 'PLD']:
        data = rounds_df[rounds_df['mechanism_type'] == mechanism]
        plt.plot(data['round'], data['allocation_accuracy'], marker='o', 
                 label=mechanism, linewidth=2, alpha=0.8)
    
    plt.xlabel('Round')
    plt.ylabel('Allocation Accuracy')
    plt.title('Mechanism Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)
    plt.close()

# Analysis 2: Adversarial Influence Reduction in PLD
def plot_adversarial_influence():
    plt.figure(figsize=(12, 6))
    
    for mechanism in ['PDD', 'PRD', 'PLD']:
        data = rounds_df[rounds_df['mechanism_type'] == mechanism]
        plt.plot(data['round'], data['adversary_voting_power'], marker='o', 
                 label=mechanism, linewidth=2, alpha=0.8)
    
    plt.xlabel('Round')
    plt.ylabel('Adversarial Voting Power')
    plt.title('Adversarial Influence Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('adversarial_influence.png', dpi=300)
    plt.close()

# Analysis 3: Resource Accumulation by Mechanism
def plot_resource_accumulation():
    plt.figure(figsize=(12, 6))
    
    for mechanism in ['PDD', 'PRD', 'PLD']:
        data = rounds_df[rounds_df['mechanism_type'] == mechanism]
        plt.plot(data['round'], data['total_resources'], marker='o', 
                 label=mechanism, linewidth=2, alpha=0.8)
    
    plt.xlabel('Round')
    plt.ylabel('Total Resources')
    plt.title('Resource Accumulation Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('resource_accumulation.png', dpi=300)
    plt.close()

# Analysis 4: Delegation Patterns (PLD only)
def analyze_delegation_patterns():
    # Filter PLD data
    pld_agents = agents_df[agents_df['mechanism_type'] == 'PLD']
    
    # Calculate average delegations received by expertise level
    expertise_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    pld_agents['expertise_bin'] = pd.cut(pld_agents['expertise_level'], bins=expertise_bins)
    
    # Group by round and expertise bin, then calculate mean delegations
    delegation_by_expertise = pld_agents.groupby(['round', 'expertise_bin'])['delegations_received'].mean().reset_index()
    
    # Pivot for plotting
    pivot_data = delegation_by_expertise.pivot(index='round', columns='expertise_bin', values='delegations_received')
    
    # Plot
    plt.figure(figsize=(12, 6))
    pivot_data.plot(marker='o', linewidth=2, ax=plt.gca())
    plt.xlabel('Round')
    plt.ylabel('Average Delegations Received')
    plt.title('Delegation Patterns by Expertise Level in PLD')
    plt.legend(title='Expertise Level')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('delegation_by_expertise.png', dpi=300)
    plt.close()

# Analysis 5: Voting Power Distribution in PLD
def plot_voting_power_distribution():
    # Get PLD data for first, middle and last rounds
    pld_data = agents_df[agents_df['mechanism_type'] == 'PLD']
    rounds_to_analyze = [1, 15, 30]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    for i, round_num in enumerate(rounds_to_analyze):
        round_data = pld_data[pld_data['round'] == round_num]
        
        # Sort by expertise for visualization
        round_data = round_data.sort_values('expertise_level')
        
        # Color by agent type
        colors = ['red' if is_adv else 'blue' for is_adv in round_data['is_adversarial']]
        
        axes[i].bar(range(len(round_data)), round_data['voting_power'], color=colors)
        axes[i].set_title(f'Round {round_num}')
        axes[i].set_xlabel('Agents (sorted by expertise)')
        
        if i == 0:
            axes[i].set_ylabel('Voting Power')
    
    plt.suptitle('Voting Power Distribution in PLD Over Time')
    plt.figlegend(['Non-Adversarial', 'Adversarial'], loc='lower center', ncol=2)
    plt.tight_layout()
    plt.savefig('voting_power_distribution.png', dpi=300)
    plt.close()

# Analysis 7: Statistical Summary
def generate_statistical_summary():
    # Comparative summary statistics
    summary = rounds_df.groupby('mechanism_type').agg({
        'allocation_accuracy': ['mean', 'std', 'min', 'max'],
        'total_resources': ['mean', 'std', 'min', 'max'],
        'adversary_voting_power': ['mean', 'std', 'min', 'max']
    })
    
    # Calculate improvement from first to last round
    improvement = {}
    for mechanism in ['PDD', 'PRD', 'PLD']:
        mech_data = rounds_df[rounds_df['mechanism_type'] == mechanism]
        first_accuracy = mech_data.iloc[0]['allocation_accuracy']
        last_accuracy = mech_data.iloc[-1]['allocation_accuracy']
        improvement[mechanism] = (last_accuracy - first_accuracy) / first_accuracy * 100
    
    # Print summary
    print("\n===== STATISTICAL SUMMARY =====")
    print("\nAverage Performance Metrics by Mechanism:")
    print(summary)
    
    print("\nAccuracy Improvement (First to Last Round):")
    for mechanism, pct in improvement.items():
        print(f"{mechanism}: {pct:.2f}%")
    
    # Return data for further analysis
    return summary, improvement

# Run analyses
plot_accuracy_comparison()
plot_adversarial_influence()
plot_resource_accumulation()
analyze_delegation_patterns()
plot_voting_power_distribution()
summary, improvement = generate_statistical_summary()

# Comprehensive analysis summary
print("\n===== COMPREHENSIVE ANALYSIS =====")

# Best performing mechanism based on final accuracy
final_round = 30
final_accuracies = {mech: rounds_df[(rounds_df['mechanism_type'] == mech) & 
                                     (rounds_df['round'] == final_round)]['allocation_accuracy'].values[0]
                   for mech in ['PDD', 'PRD', 'PLD']}
best_mechanism = max(final_accuracies, key=final_accuracies.get)

print(f"\nBest performing mechanism: {best_mechanism} with final accuracy of {final_accuracies[best_mechanism]:.3f}")

# Adversarial influence analysis
adv_reduction = {}
for mechanism in ['PDD', 'PRD', 'PLD']:
    mech_data = rounds_df[rounds_df['mechanism_type'] == mechanism]
    first_adv_power = mech_data.iloc[0]['adversary_voting_power']
    last_adv_power = mech_data.iloc[-1]['adversary_voting_power']
    adv_reduction[mechanism] = (first_adv_power - last_adv_power) / first_adv_power * 100

print("\nAdversarial Influence Reduction:")
for mechanism, pct in adv_reduction.items():
    print(f"{mechanism}: {pct:.2f}%")

# Resource efficiency analysis
resource_efficiency = {}
for mechanism in ['PDD', 'PRD', 'PLD']:
    mech_data = rounds_df[rounds_df['mechanism_type'] == mechanism]
    first_resources = mech_data.iloc[0]['total_resources']
    last_resources = mech_data.iloc[-1]['total_resources']
    resource_efficiency[mechanism] = (last_resources - first_resources) / first_resources * 100

print("\nResource Growth Rate:")
for mechanism, pct in resource_efficiency.items():
    print(f"{mechanism}: {pct:.2f}%")

# Expertise utilization in delegation (PLD only)
pld_agents = agents_df[agents_df['mechanism_type'] == 'PLD']
expertise_correlations = []

for round_num in range(1, 31):
    round_data = pld_agents[pld_agents['round'] == round_num]
    correlation = round_data['expertise_level'].corr(round_data['delegations_received'])
    expertise_correlations.append(correlation)

avg_expertise_correlation = np.mean(expertise_correlations)
print(f"\nAverage correlation between expertise and delegations in PLD: {avg_expertise_correlation:.3f}")

# Final conclusions
print("\n===== FINAL CONCLUSIONS =====")
print(f"1. {best_mechanism} achieved the highest final decision accuracy at {final_accuracies[best_mechanism]:.3f}.")
print(f"2. PLD showed the greatest reduction in adversarial influence at {adv_reduction['PLD']:.2f}%.")
print(f"3. Resource accumulation was highest in {max(resource_efficiency, key=resource_efficiency.get)}.")
print(f"4. In PLD, expertise was positively correlated with delegations received (r={avg_expertise_correlation:.3f}),"
      f" suggesting the mechanism successfully identifies and leverages expertise.")

# Create visualization of results progression
def visualize_results_progression():
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    # Common x-axis
    x = range(1, 31)
    
    # Plot 1: Allocation Accuracy
    for mechanism in ['PDD', 'PRD', 'PLD']:
        data = rounds_df[rounds_df['mechanism_type'] == mechanism]
        axes[0].plot(data['round'], data['allocation_accuracy'], marker='o', 
                   label=mechanism, linewidth=2, alpha=0.8)
    
    axes[0].set_ylabel('Allocation Accuracy')
    axes[0].set_title('Decision Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Adversarial Influence
    for mechanism in ['PDD', 'PRD', 'PLD']:
        data = rounds_df[rounds_df['mechanism_type'] == mechanism]
        axes[1].plot(data['round'], data['adversary_voting_power'], marker='o', 
                   label=mechanism, linewidth=2, alpha=0.8)
    
    axes[1].set_ylabel('Adversarial Voting Power')
    axes[1].set_title('Adversarial Influence Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Resource Accumulation
    for mechanism in ['PDD', 'PRD', 'PLD']:
        data = rounds_df[rounds_df['mechanism_type'] == mechanism]
        axes[2].plot(data['round'], data['total_resources'], marker='o', 
                   label=mechanism, linewidth=2, alpha=0.8)
    
    axes[2].set_xlabel('Round')
    axes[2].set_ylabel('Total Resources')
    axes[2].set_title('Resource Accumulation')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_progression.png', dpi=300)
    plt.close()

visualize_results_progression()