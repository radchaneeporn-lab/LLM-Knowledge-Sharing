import os
from dotenv import load_dotenv
load_dotenv()

PRICING = {
    # Gemini
    "gemini-3.1-flash-lite":  {"input": 0.10,  "output": 0.40,  "thinking": 0.0},
    "gemini-3-flash-preview": {"input": 0.075, "output": 0.30,  "thinking": 3.50},
    # Claude
    "claude-haiku-4-5":       {"input": 1.00,  "output": 5.00,  "thinking": 0.0},
    "claude-sonnet-4-6":      {"input": 3.00,  "output": 15.00, "thinking": 0.0},
    "claude-opus-4-7":        {"input": 5.00,  "output": 25.00, "thinking": 0.0},
    # GPT
    "gpt-4.1-nano":           {"input": 0.10,  "output": 0.40,  "thinking": 0.0},
    "gpt-4.1-mini":           {"input": 0.40,  "output": 1.60,  "thinking": 0.0},
}


class TokenCounter:
    """Count tokens and calculate cost for Gemini, Claude, or GPT models.

    Usage:
        counter = TokenCounter(model="claude-haiku-4-5")
        tokens  = counter.count_tokens(prompt="Hello", system="You are helpful.")
        cost    = counter.calculate_cost(response.usage)
    """

    def __init__(self, model: str):
        self.model = model
        self._provider = self._detect_provider(model)
        self._client = self._init_client()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_provider(self, model: str) -> str:
        if model.startswith("gemini"):
            return "gemini"
        if model.startswith("claude"):
            return "claude"
        if model.startswith("gpt"):
            return "gpt"
        raise ValueError(f"Cannot detect provider for model: {model!r}")

    def _init_client(self):
        if self._provider == "gemini":
            from google import genai
            return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        if self._provider == "claude":
            import anthropic
            return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        if self._provider == "gpt":
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_tokens(
        self,
        prompt: str = None,
        system: str = None,
        messages: list = None,
    ) -> int:
        
        """Return estimated input token count without generating a response.

        Args:
            prompt:   Plain-text prompt (Gemini / GPT) or user message content (Claude).
            system:   System/instruction text (Claude system param, GPT instructions param).
            messages: Full message list for Claude (overrides prompt when provided).
        """
        
        if self._provider == "gemini":
            from google.genai import types
            kwargs = {"model": self.model, "contents": prompt}
            if system:
                kwargs["config"] = types.GenerateContentConfig(system_instruction=system)
            result = self._client.models.count_tokens(**kwargs)
            return result.total_tokens

        elif self._provider == "claude":
            msgs = messages or [{"role": "user", "content": prompt}]
            kwargs = {"model": self.model, "messages": msgs}
            if system:
                kwargs["system"] = system
            return self._client.messages.count_tokens(**kwargs).input_tokens

        elif self._provider == "gpt":
            result = self._client.responses.input_tokens.count(
                model=self.model,
                instructions=system or "",
                input=prompt or "",
            )
            return result.input_tokens

    def calculate_cost(self, usage) -> dict:
        """Return token counts and USD cost from an API usage object.

        Accepts:
            Gemini  → response.usage_metadata
            Claude  → response.usage
            GPT     → response.usage
        """
        if self.model not in PRICING:
            raise ValueError(f"No pricing found for model: {self.model!r}")
        pricing = PRICING[self.model]

        if self._provider == "gemini":
            input_tokens   = usage.prompt_token_count or 0
            output_tokens  = usage.candidates_token_count or 0
            thinking_tokens = usage.thoughts_token_count or 0
        else:
            input_tokens   = usage.input_tokens or 0
            output_tokens  = usage.output_tokens or 0
            thinking_tokens = 0

        input_cost   = (input_tokens   / 1_000_000) * pricing["input"]
        output_cost  = (output_tokens  / 1_000_000) * pricing["output"]
        thinking_cost = (thinking_tokens / 1_000_000) * pricing["thinking"]
        total_cost   = input_cost + output_cost + thinking_cost

        return {
            "model": self.model,
            "tokens": {
                "input":    input_tokens,
                "output":   output_tokens,
                "thinking": thinking_tokens,
                "total":    input_tokens + output_tokens + thinking_tokens,
            },
            "cost_usd": {
                "input":    round(input_cost,    6),
                "output":   round(output_cost,   6),
                "thinking": round(thinking_cost, 6),
                "total":    round(total_cost,    6),
            },
        }


