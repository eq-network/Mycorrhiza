# services/llm.py
from typing import Dict, Any, Optional, List
import os
import json
import random
import requests

class LLMClient:
    """Client for interacting with Large Language Models via different providers."""
    
    def __init__(
        self, 
        provider: str = "mock", 
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: The LLM provider to use ('openai', 'anthropic', 'openrouter', 'mock')
            api_key: The API key for the provider
            model: Specific model to use (especially important for OpenRouter)
        """
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        
        # Set default models based on provider
        if model is None:
            if self.provider == "openai":
                self.model = "gpt-4"
            elif self.provider == "anthropic":
                self.model = "claude-3-sonnet-20240229"
            elif self.provider == "openrouter":
                self.model = "anthropic/claude-3-sonnet"  # Default OpenRouter model
            else:
                self.model = "mock"
        else:
            self.model = model
        
        # Initialize provider-specific client if needed
        if self.provider == "openai":
            import openai
            openai.api_key = self.api_key
            self.client = openai
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "openrouter":
            # OpenRouter uses direct API calls
            self.openrouter_api_url = "https://openrouter.ai/api/v1/chat/completions"
            if not self.api_key:
                raise ValueError("API key required for OpenRouter")
        elif self.provider == "mock":
            self.client = None
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            The generated text response
        """
        if self.provider == "openai":
            return self._generate_openai(prompt, temperature)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, temperature)
        elif self.provider == "openrouter":
            return self._generate_openrouter(prompt, temperature)
        elif self.provider == "mock":
            return self._generate_mock(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _generate_openai(self, prompt: str, temperature: float) -> str:
        """Generate response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model, 
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(self, prompt: str, temperature: float) -> str:
        """Generate response using Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _generate_openrouter(self, prompt: str, temperature: float) -> str:
        """Generate response using OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-site.com",  # Replace with your site
            "X-Title": "Fruit Preference Simulation"  # Name of your app/project
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(
                self.openrouter_api_url,
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()  # Raise exception for HTTP errors
            response_json = response.json()
            
            # Extract the content from the response
            return response_json["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling OpenRouter API: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response content: {e.response.text}")
            # Return a simple error message for the mock
            return f"[ERROR: OpenRouter API call failed - {str(e)}]"
    
    def _generate_mock(self, prompt: str) -> str:
        """Generate a mock response for testing."""
        # Extract fruit names from prompt if possible
        fruit_names = []
        if "preferences for [" in prompt:
            start = prompt.find("preferences for [") + len("preferences for [")
            end = prompt.find("]", start)
            if end > start:
                fruit_list = prompt[start:end]
                fruit_names = [f.strip() for f in fruit_list.split(",")]
        
        if not fruit_names:
            fruit_names = ["Apple", "Banana", "Cherry", "Durian", "Elderberry"]
        
        # Generate mock conversation
        conversations = [
            f"Person A: I enjoy {fruit_names[0]} for their crispness, though they're a bit plain.\nPerson B: Have you tried different varieties? I prefer {fruit_names[1]} for convenience though.\nPerson A: You make a good point about convenience.",
            f"Person A: {fruit_names[min(3, len(fruit_names)-1)]} has such a strong smell, I can't get past it.\nPerson B: The smell is intense, but the taste is incredible! Creamy and sweet.\nPerson A: I'm not convinced, but maybe I judged too quickly.",
            f"Person A: {fruit_names[min(2, len(fruit_names)-1)]} are perfect little packages of flavor.\nPerson B: They're good but so seasonal and expensive!\nPerson A: That's true, I hadn't considered the cost factor."
        ]
        
        # Random preference changes
        pref_changes = [round(random.uniform(-0.1, 0.1), 2) for _ in range(len(fruit_names))]
        trust_change = round(random.uniform(-0.1, 0.1), 2)
        
        return f"""[CONVERSATION]
{random.choice(conversations)}
[/CONVERSATION]

[PREFERENCE_INFLUENCE]
{pref_changes}
[/PREFERENCE_INFLUENCE]

[TRUST_UPDATE]
{trust_change}
[/TRUST_UPDATE]"""

    def list_available_models(self) -> List[str]:
        """
        Return a list of available models based on the provider.
        For OpenRouter, this is a curated list of popular models.
        """
        if self.provider == "openrouter":
            return [
                # Anthropic models
                "anthropic/claude-3-opus",
                "anthropic/claude-3-sonnet",
                "anthropic/claude-3-haiku",
                "anthropic/claude-2",
                # OpenAI models
                "openai/gpt-4-turbo",
                "openai/gpt-4",
                "openai/gpt-3.5-turbo",
                # Google models
                "google/gemini-pro",
                "google/gemini-1.5-pro",
                # Meta models
                "meta-llama/llama-3-70b-instruct",
                "meta-llama/llama-3-8b-instruct",
                "meta-llama/llama-2-70b-chat",
                # Mistral models
                "mistral/mistral-large",
                "mistral/mistral-medium",
                "mistral/mistral-small",
                # More models available at: https://openrouter.ai/docs#models
            ]
        elif self.provider == "openai":
            return ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        elif self.provider == "anthropic":
            return ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-2"]
        else:
            return ["mock"]