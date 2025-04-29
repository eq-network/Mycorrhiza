import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List
import json

from core.graph_state import GraphState
from core.transformation import Transformation
from core.properties import Property, ConservesSum
from services.llm import LLMClient


class PreferenceDiscussion(Transformation):
    """
    Simulates agents discussing their fruit preferences using LLM.
    This updates preferences based on agent discussions.
    """
    
    def __init__(self, llm_client: LLMClient):
        super().__init__()
        self.llm_client = llm_client
        
    @property
    def preserves_properties(self) -> set:
        """This transformation preserves the sum of preferences."""
        return {ConservesSum("fruit_preferences")}
    
    def apply(self, state: GraphState) -> GraphState:
        # Get current state information
        preferences = np.array(state.node_attrs["fruit_preferences"])
        communication = np.array(state.adj_matrices["communication"])
        trust = np.array(state.adj_matrices["trust"])
        fruit_names = state.global_attrs["fruit_names"]
        current_round = state.global_attrs.get("round", 0)
        
        # For each pair of communicating agents, simulate a discussion
        new_preferences = preferences.copy()
        discussion_history = state.global_attrs.get("discussion_history", [])
        round_discussions = []
        
        # Find all communicating pairs
        communicating_pairs = []
        for agent_i in range(preferences.shape[0]):
            for agent_j in range(preferences.shape[0]):
                if communication[agent_i, agent_j] == 1:
                    communicating_pairs.append((agent_i, agent_j))
        
        # For each pair, get the discussion and preference influence
        for agent_i, agent_j in communicating_pairs:
            discussion, pref_influence, trust_update = self._simulate_discussion(
                agent_i, agent_j, preferences, fruit_names
            )
            
            # Record the discussion
            round_discussions.append({
                "agent_i": int(agent_i),
                "agent_j": int(agent_j),
                "discussion": discussion
            })
            
            # Update preferences based on discussion (apply influence)
            new_preferences[agent_i] += pref_influence
        
        # Normalize preferences to sum to 1 for each agent
        new_preferences = new_preferences / np.sum(new_preferences, axis=1, keepdims=True)
        
        # Record discussions
        updated_history = discussion_history.copy()
        updated_history.append(round_discussions)
        
        # Create new state
        return state.replace(
            node_attrs={
                **state.node_attrs,
                "fruit_preferences": jnp.array(new_preferences)
            },
            global_attrs={
                **state.global_attrs,
                "discussion_history": updated_history,
                "round": current_round + 1
            }
        )
    
    def _simulate_discussion(self, agent_i, agent_j, preferences, fruit_names):
        """
        Use LLM to simulate a discussion between two agents about fruit preferences.
        Returns:
            - discussion text
            - preference influence (how agent_i's preferences change)
            - trust update (how agent_i's trust in agent_j changes)
        """
        agent_i_prefs = {fruit: float(preferences[agent_i, idx]) for idx, fruit in enumerate(fruit_names)}
        agent_j_prefs = {fruit: float(preferences[agent_j, idx]) for idx, fruit in enumerate(fruit_names)}
        
        # Construct prompt for the LLM
        prompt = self._build_discussion_prompt(agent_i_prefs, agent_j_prefs, fruit_names)
        
        # Call the LLM
        response = self.llm_client.generate(prompt)
        
        # Parse the response
        conversation = self._extract_between(response, "[CONVERSATION]", "[/CONVERSATION]")
        pref_influence_str = self._extract_between(response, "[PREFERENCE_INFLUENCE]", "[/PREFERENCE_INFLUENCE]")
        trust_update_str = self._extract_between(response, "[TRUST_UPDATE]", "[/TRUST_UPDATE]")
        
        # Parse numerical values
        try:
            pref_influence = np.array(eval(pref_influence_str.strip()))
            trust_update = float(trust_update_str.strip())
        except Exception as e:
            # Fallback in case parsing fails
            print(f"Warning: Failed to parse LLM response: {e}. Using default values.")
            pref_influence = np.zeros(len(fruit_names))
            trust_update = 0.0
        
        return conversation, pref_influence, trust_update
    
    def _build_discussion_prompt(self, agent_i_prefs, agent_j_prefs, fruit_names):
        """Build the prompt for the LLM."""
        return f"""
        Simulate a conversation between two people discussing their fruit preferences.

        Person A's preferences: {agent_i_prefs}
        Person B's preferences: {agent_j_prefs}

        Write a short, realistic conversation where they discuss why they like or dislike these fruits.
        Then as an expert summarizer, provide two numerical assessments:
        1. PREFERENCE_INFLUENCE: A list of {len(fruit_names)} small numbers (between -0.1 and 0.1) indicating how Person A's preferences for {fruit_names} might change after this conversation.
        2. TRUST_UPDATE: A single number between -0.1 and 0.1 indicating how Person A's trust in Person B might change after this conversation.

        Output your response in the format:
        [CONVERSATION]
        Person A: ...
        Person B: ...
        ...
        [/CONVERSATION]

        [PREFERENCE_INFLUENCE]
        [value1, value2, value3, value4, value5]
        [/PREFERENCE_INFLUENCE]

        [TRUST_UPDATE]
        value
        [/TRUST_UPDATE]
        """
    
    def _extract_between(self, text, start_marker, end_marker):
        """Extract text between two markers."""
        start = text.find(start_marker) + len(start_marker)
        end = text.find(end_marker, start)
        if start == -1 or end == -1:
            return ""
        return text[start:end].strip()