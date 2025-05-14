# services/llm.py
class LLMService:
    """Interface for interacting with large language models."""
    
    def __init__(self, api_key: str, model: str = "default"):
        self.api_key = api_key
        self.model = model
    
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        raise NotImplementedError("Subclasses must implement")

class OpenAIService(LLMService):
    """OpenAI implementation of LLM service."""
    
    def generate(self, prompt: str) -> str:
        """Generate text using OpenAI API."""
        # Implementation details
        import openai
        openai.api_key = self.api_key
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content