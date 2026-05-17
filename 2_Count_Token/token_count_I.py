import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from google import genai
import anthropic
from openai import OpenAI
from google.genai import types

import json
from token_utils import TokenCounter

# =======================================================================
# Benefit 1. Accurate Cost Estimation, Tuning Prompt, and Model Decision
# =======================================================================
system = "You are an expert in Natural Language Processing (NLP) and Large Language Models (LLMs)."
prompt = "Compare transformer-based LLMs and RNN-based models. Which architecture is better for long-context tasks and why?"

# ==================== Gemini ===========================
# count token --> model variant decision --> tune your system prompt & user prompt
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Gemini 3.1 Flash Lite ---
counter = TokenCounter(model="gemini-3.1-flash-lite")
print(f"[gemini-3.1-flash-lite] estimated input tokens: {counter.count_tokens(prompt=prompt, system=system)}")

response_gemini_flash_lite = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=prompt,
    config=types.GenerateContentConfig(system_instruction=system),
)

print(json.dumps(counter.calculate_cost(response_gemini_flash_lite.usage_metadata), indent=2))

# --- Gemini 3 Flash Preview ---
counter_2 = TokenCounter(model="gemini-3-flash-preview")
print(f"[gemini-3-flash-preview] estimated input tokens: {counter_2.count_tokens(prompt=prompt, system=system)}")

response_gemini_flash_preview = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
    config=types.GenerateContentConfig(system_instruction=system),
)
print(json.dumps(counter_2.calculate_cost(response_gemini_flash_preview.usage_metadata), indent=2))


# ==================== Claude ===========================
# count token & calculate the price --> model variant decision --> tune your system prompt & user prompt
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

messages = [{"role": "user", "content": prompt}]

# --- Count input tokens before sending (free, no generation) ---
counter = TokenCounter(model="claude-haiku-4-5")
print(f"[claude-haiku-4-5] estimated input tokens: {counter.count_tokens(system=system, messages=messages)}")

# --- Generate and calculate full cost ---
models_to_compare = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"]

print("\n--- Cost Comparison ---")
for model in models_to_compare:
    response = client.messages.create(
        model=model,
        max_tokens=200,
        system=system,
        messages=messages,
    )
    counter = TokenCounter(model=model)
    print(f"\nquestion : {prompt}")
    print(f"response : {response.content[0].text}")
    print(json.dumps(counter.calculate_cost(response.usage), indent=2))
    print('--------------------------------------------------')


# ==================== Claude (Extended Thinking) ===========================
# thinking mode — Claude reasons internally before answering
# thinking tokens are included inside output_tokens (not reported separately in usage)
# but the thinking content is visible as a separate block in response.content

response_thinking = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    thinking={
        "type": "enabled",
        "budget_tokens": 1024,
    },
    system=system,
    messages=[{"role": "user", "content": prompt}],
)

for block in response_thinking.content:
    if block.type == "thinking":
        print(f"[Thinking]\n{block.thinking}\n")
    elif block.type == "text":
        print(f"[Response]\n{block.text}\n")

counter_thinking = TokenCounter(model="claude-sonnet-4-6")
print(json.dumps(counter_thinking.calculate_cost(response_thinking.usage), indent=2))
print('--------------------------------------------------')


# ==================== GPT ===========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Count input tokens before sending (free) ---
counter = TokenCounter(model="gpt-4.1-nano")
print(f"[gpt-4.1-nano] estimated input tokens: {counter.count_tokens(prompt=prompt, system=system)}")

# --- Generate and calculate full cost ---
models_to_compare = ["gpt-4.1-nano", "gpt-4.1-mini"]

print("\n--- Cost Comparison ---")
for model in models_to_compare:
    response = client.responses.create(
        model=model,
        instructions=system,
        max_output_tokens=50,
        input=prompt,
    )
    counter = TokenCounter(model=model)
    print(f"\nquestion : {prompt}")
    print(f"response : {response.output[0].content[0].text}")
    print(json.dumps(counter.calculate_cost(response.usage), indent=2))
    print('--------------------------------------------')
