# transformations/trading.py
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Tuple, List, Optional
import random

from core.graph_state import GraphState
from core.transformation import Transformation
from services.llm import LLMClient


class TradeOpportunitySearch(Transformation):
    """
    Identifies potential trading partners based on complementary preferences.
    Creates a temporary 'trading_opportunity' adjacency matrix.
    """
    
    def apply(self, state: GraphState) -> GraphState:
        preferences = np.array(state.node_attrs["fruit_preferences"]) 
        endowments = np.array(state.node_attrs["fruit_endowments"])
        num_agents = preferences.shape[0]
        
        # Create a matrix where entry [i,j] represents potential gain from i trading with j
        trading_opportunity = np.zeros((num_agents, num_agents))
        
        # A simple heuristic: agents who value fruits differently have more to gain from trade
        for i in range(num_agents):
            for j in range(i+1, num_agents):  # Only check each pair once
                # Calculate preference difference vector
                pref_diff = preferences[i] - preferences[j]
                
                # Higher absolute differences suggest more potential gains from trade
                # Sum of absolute differences as a simple metric
                trade_potential = np.sum(np.abs(pref_diff))
                
                # Store symmetrically
                trading_opportunity[i, j] = trade_potential
                trading_opportunity[j, i] = trade_potential
        
        # Zero out diagonal (no self-trading)
        np.fill_diagonal(trading_opportunity, 0)
        
        # Normalize to represent probabilities (if desired)
        row_sums = trading_opportunity.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_opportunities = trading_opportunity / row_sums
        
        return state.replace(
            adj_matrices={
                **state.adj_matrices, 
                "trading_opportunity": jnp.array(normalized_opportunities)
            }
        )


class TradeMatching(Transformation):
    """
    Determines which agents will trade with each other in this round.
    Creates a 'trading_matches' adjacency matrix.
    """
    
    def __init__(self, avg_trades_per_agent: float = 1.0):
        """
        Args:
            avg_trades_per_agent: Average number of trading partners per agent
        """
        super().__init__()
        self.avg_trades_per_agent = avg_trades_per_agent
    
    def apply(self, state: GraphState) -> GraphState:
        if "trading_opportunity" not in state.adj_matrices:
            # Run opportunity search first if not already done
            state = TradeOpportunitySearch().apply(state)
            
        opportunities = np.array(state.adj_matrices["trading_opportunity"])
        num_agents = opportunities.shape[0]
        
        # Create empty trading matches matrix
        trading_matches = np.zeros((num_agents, num_agents))
        
        # Convert opportunity scores to probabilities of trading
        # Higher opportunity score = higher probability of being selected
        
        # Total number of trades to create (undirected, so each counts twice)
        total_trades = int(num_agents * self.avg_trades_per_agent / 2)
        
        # Create a flattened list of agent pairs, excluding self-pairs
        agent_pairs = []
        pair_weights = []
        
        for i in range(num_agents):
            for j in range(i+1, num_agents):  # Only include each pair once
                agent_pairs.append((i, j))
                pair_weights.append(opportunities[i, j])
        
        # Normalize weights to probabilities
        if sum(pair_weights) > 0:
            pair_weights = np.array(pair_weights) / sum(pair_weights)
        else:
            # If all weights are zero, use uniform distribution
            pair_weights = np.ones(len(agent_pairs)) / len(agent_pairs)
        
        # Sample pairs without replacement
        num_pairs = min(total_trades, len(agent_pairs))
        selected_indices = np.random.choice(
            len(agent_pairs), 
            size=num_pairs, 
            replace=False, 
            p=pair_weights
        )
        
        # Create trading matches
        for idx in selected_indices:
            i, j = agent_pairs[idx]
            trading_matches[i, j] = 1
            trading_matches[j, i] = 1  # Symmetric - both agents are matched
        
        return state.replace(
            adj_matrices={
                **state.adj_matrices, 
                "trading_matches": jnp.array(trading_matches)
            }
        )


class TradeNegotiation(Transformation):
    """
    For matched trading pairs, negotiate trades that improve utility for both.
    Can use LLM to simulate realistic negotiation or algorithmic approach.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Args:
            llm_client: Optional LLM client for simulating negotiations
        """
        super().__init__()
        self.llm_client = llm_client
        self.use_llm = llm_client is not None
    
    def apply(self, state: GraphState) -> GraphState:
        if "trading_matches" not in state.adj_matrices:
            # No matches to process
            return state
            
        # Get current state
        preferences = np.array(state.node_attrs["fruit_preferences"])
        endowments = np.array(state.node_attrs["fruit_endowments"])
        trading_matches = np.array(state.adj_matrices["trading_matches"])
        fruit_names = state.global_attrs["fruit_names"]
        current_round = state.global_attrs.get("round", 0)
        
        # Create a copy of endowments to modify
        new_endowments = endowments.copy()
        
        # Track trades for history
        trades_this_round = []
        
        # Find trading pairs
        num_agents = trading_matches.shape[0]
        for i in range(num_agents):
            for j in range(i+1, num_agents):  # Process each pair only once
                if trading_matches[i, j] > 0:
                    # These agents are matched for trading
                    if self.use_llm:
                        # Use LLM to simulate negotiation
                        trade = self._negotiate_trade_with_llm(
                            i, j, preferences, new_endowments, fruit_names
                        )
                    else:
                        # Use algorithmic approach
                        trade = self._negotiate_trade_algorithmic(
                            i, j, preferences, new_endowments
                        )
                    
                    # If a valid trade was found, apply it
                    if trade is not None:
                        agent_i, agent_j, fruit_i, fruit_j, amount_i, amount_j = trade
                        
                        # Update endowments
                        new_endowments[agent_i, fruit_i] -= amount_i
                        new_endowments[agent_j, fruit_i] += amount_i
                        new_endowments[agent_j, fruit_j] -= amount_j
                        new_endowments[agent_i, fruit_j] += amount_j
                        
                        # Record the trade
                        trades_this_round.append({
                            "agent_i": int(agent_i),
                            "agent_j": int(agent_j),
                            "fruit_i": int(fruit_i),
                            "fruit_i_name": fruit_names[fruit_i],
                            "amount_i": int(amount_i),
                            "fruit_j": int(fruit_j),
                            "fruit_j_name": fruit_names[fruit_j],
                            "amount_j": int(amount_j)
                        })
        
        # Update trade history
        trade_history = state.global_attrs.get("trade_history", [])
        updated_history = trade_history.copy()
        updated_history.append(trades_this_round)
        
        # Calculate new utilities based on updated endowments
        new_utilities = np.sum(preferences * new_endowments, axis=1)
        
        # Calculate total utility (social welfare)
        total_utility = np.sum(new_utilities)
        
        # Create new state
        return state.replace(
            node_attrs={
                **state.node_attrs,
                "fruit_endowments": jnp.array(new_endowments),
                "utility": jnp.array(new_utilities)
            },
            global_attrs={
                **state.global_attrs,
                "trade_history": updated_history,
                "round": current_round + 1,
                "total_utility": float(total_utility)
            }
        )
    
    def _negotiate_trade_algorithmic(
        self, 
        agent_i: int, 
        agent_j: int, 
        preferences: np.ndarray, 
        endowments: np.ndarray
    ) -> Optional[Tuple[int, int, int, int, int, int]]:
        """
        Algorithmically find a trade that improves utility for both agents.
        
        Returns:
            Tuple of (agent_i, agent_j, fruit_i, fruit_j, amount_i, amount_j)
            or None if no beneficial trade found
        """
        # Get current utilities
        utility_i_before = np.sum(preferences[agent_i] * endowments[agent_i])
        utility_j_before = np.sum(preferences[agent_j] * endowments[agent_j])
        
        num_fruits = preferences.shape[1]
        best_trade = None
        best_gain = 0
        
        # Try trading each pair of fruits
        for fruit_i in range(num_fruits):
            for fruit_j in range(num_fruits):
                if fruit_i == fruit_j:
                    continue  # Skip same fruit trades
                
                # Skip if either agent doesn't have the fruit they would give
                if endowments[agent_i, fruit_i] <= 0 or endowments[agent_j, fruit_j] <= 0:
                    continue
                
                # Calculate marginal utilities
                marg_util_i_gives = preferences[agent_i, fruit_i]
                marg_util_i_gets = preferences[agent_i, fruit_j]
                marg_util_j_gives = preferences[agent_j, fruit_j]
                marg_util_j_gets = preferences[agent_j, fruit_i]
                
                # Only consider trades where both benefit
                if (marg_util_i_gets <= marg_util_i_gives) or (marg_util_j_gets <= marg_util_j_gives):
                    continue
                
                # Simple case: 1-for-1 trade
                # Each agent gives 1 unit of one fruit and gets 1 unit of another
                
                # Check if this trade improves utility for both
                new_endowments_i = endowments[agent_i].copy()
                new_endowments_j = endowments[agent_j].copy()
                
                new_endowments_i[fruit_i] -= 1
                new_endowments_i[fruit_j] += 1
                new_endowments_j[fruit_j] -= 1
                new_endowments_j[fruit_i] += 1
                
                utility_i_after = np.sum(preferences[agent_i] * new_endowments_i)
                utility_j_after = np.sum(preferences[agent_j] * new_endowments_j)
                
                # Both must benefit
                if utility_i_after > utility_i_before and utility_j_after > utility_j_before:
                    # Calculate total gain
                    total_gain = (utility_i_after - utility_i_before) + (utility_j_after - utility_j_before)
                    
                    # Keep track of best trade
                    if total_gain > best_gain:
                        best_gain = total_gain
                        best_trade = (agent_i, agent_j, fruit_i, fruit_j, 1, 1)
        
        return best_trade
    
    def _negotiate_trade_with_llm(
            self, 
            agent_i: int, 
            agent_j: int, 
            preferences: np.ndarray, 
            endowments: np.ndarray, 
            fruit_names: List[str]
        ) -> Optional[Tuple[int, int, int, int, int, int]]:
            """
            Use LLM to negotiate a trade between two agents.
            
            Returns:
                Tuple of (agent_i, agent_j, fruit_i, fruit_j, amount_i, amount_j)
                or None if no beneficial trade found
            """
            if self.llm_client is None:
                return self._negotiate_trade_algorithmic(agent_i, agent_j, preferences, endowments)
            
            # Format agent preferences and endowments
            agent_i_prefs = {fruit_names[idx]: float(preferences[agent_i, idx]) 
                            for idx in range(len(fruit_names))}
            agent_j_prefs = {fruit_names[idx]: float(preferences[agent_j, idx]) 
                            for idx in range(len(fruit_names))}
            
            agent_i_endow = {fruit_names[idx]: int(endowments[agent_i, idx]) 
                            for idx in range(len(fruit_names))}
            agent_j_endow = {fruit_names[idx]: int(endowments[agent_j, idx]) 
                            for idx in range(len(fruit_names))}
            
            # Build prompt
            prompt = f"""
            Simulate a trading negotiation between two people with different preferences for fruits.

            Person A (Agent {agent_i}):
            - Preferences (utility per fruit): {agent_i_prefs}
            - Current fruit inventory: {agent_i_endow}
            - Current total utility: {sum(agent_i_prefs[f] * agent_i_endow[f] for f in fruit_names)}

            Person B (Agent {agent_j}):
            - Preferences (utility per fruit): {agent_j_prefs}
            - Current fruit inventory: {agent_j_endow}
            - Current total utility: {sum(agent_j_prefs[f] * agent_j_endow[f] for f in fruit_names)}

            These people want to trade fruits to increase their utility. They can only trade if BOTH benefit.
            Utility is calculated as (preference for fruit) * (quantity of fruit).

            Analyze their preferences and inventories, then propose ONE specific trade that would benefit both parties.
            If no beneficial trade is possible, state that clearly.

            Output your response in this format:
            [TRADE]
            fruit_from_A: [fruit name]
            amount_from_A: [integer amount]
            fruit_from_B: [fruit name]
            amount_from_B: [integer amount]
            [/TRADE]

            Or if no beneficial trade is possible:
            [TRADE]
            no_trade_possible
            [/TRADE]
            """
            
            # Call the LLM
            response = self.llm_client.generate(prompt)
            
            # Parse response
            trade_text = self._extract_between(response, "[TRADE]", "[/TRADE]").strip()
            
            if "no_trade_possible" in trade_text:
                return None
            
            # Parse the trade details
            try:
                lines = [line.strip() for line in trade_text.split('\n') if line.strip()]
                trade_details = {}
                
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        trade_details[key.strip()] = value.strip()
                
                # Extract the trade information
                fruit_from_a = trade_details.get('fruit_from_A')
                amount_from_a = int(trade_details.get('amount_from_A', 0))
                fruit_from_b = trade_details.get('fruit_from_B')
                amount_from_b = int(trade_details.get('amount_from_B', 0))
                
                # Map fruit names to indices
                fruit_i = fruit_names.index(fruit_from_a)
                fruit_j = fruit_names.index(fruit_from_b)
                
                # Validate trade
                if (endowments[agent_i, fruit_i] >= amount_from_a and 
                    endowments[agent_j, fruit_j] >= amount_from_b):
                    
                    # Verify this trade benefits both agents
                    new_endowments_i = endowments[agent_i].copy()
                    new_endowments_j = endowments[agent_j].copy()
                    
                    new_endowments_i[fruit_i] -= amount_from_a
                    new_endowments_i[fruit_j] += amount_from_b
                    new_endowments_j[fruit_j] -= amount_from_b
                    new_endowments_j[fruit_i] += amount_from_a
                    
                    utility_i_before = np.sum(preferences[agent_i] * endowments[agent_i])
                    utility_j_before = np.sum(preferences[agent_j] * endowments[agent_j])
                    utility_i_after = np.sum(preferences[agent_i] * new_endowments_i)
                    utility_j_after = np.sum(preferences[agent_j] * new_endowments_j)
                    
                    if utility_i_after > utility_i_before and utility_j_after > utility_j_before:
                        return (agent_i, agent_j, fruit_i, fruit_j, amount_from_a, amount_from_b)
            
            except Exception as e:
                print(f"Error parsing trade: {e}")
            
            # If parsing failed or trade is invalid, fall back to algorithmic approach
            return self._negotiate_trade_algorithmic(agent_i, agent_j, preferences, endowments)
        
    def _extract_between(self, text, start_marker, end_marker):
        """Extract text between two markers."""
        start = text.find(start_marker) + len(start_marker)
        end = text.find(end_marker, start)
        if start == -1 or end == -1:
            return ""
        return text[start:end].strip()


class UtilityCalculation(Transformation):
    """
    Calculate utility for each agent based on current endowments and preferences.
    """
    
    def apply(self, state: GraphState) -> GraphState:
        preferences = np.array(state.node_attrs["fruit_preferences"])
        endowments = np.array(state.node_attrs["fruit_endowments"])
        
        # Calculate utility for each agent
        utilities = np.sum(preferences * endowments, axis=1)
        
        # Calculate total utility (social welfare)
        total_utility = np.sum(utilities)
        
        # Calculate efficiency metric - how close to optimal allocation
        # This requires knowing the optimal allocation, which is complex
        # For now we just track raw total utility
        
        return state.replace(
            node_attrs={
                **state.node_attrs,
                "utility": jnp.array(utilities)
            },
            global_attrs={
                **state.global_attrs,
                "total_utility": float(total_utility)
            }
        )