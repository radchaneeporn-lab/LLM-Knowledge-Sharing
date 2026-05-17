#concept  #implementation

#token-count #cost-optimization 

**Scope:** 
- token counting from LLM API provider
- 3 LLM provider: Gemini, Claude, GPT
- text only input
# I) Benefits of Token Counting for Accurate Cost Estimation and Budget Forecasting 

## 1.1 Counting Input Tokens Before Inference

### Google 's Gemini

```python
from google import genai

client = genai.Client(api_key = 'your_api_key')

response = client.models.count_tokens(
        model=model,
        contents=prompt)
        
print(result.total_tokens)
```

### Anthropic 's Claude

- general mode
```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.count_tokens(
        model=model,
        messages=messages,
    )

print(response.input_tokens)

```

- thinking mode enable
```python
response_thinking_mode = client.messages.create(

    model=model,
    max_tokens=4096,
    thinking={
        "type": "enabled",
        "budget_tokens": 1024,
    },
    system=system,
    messages=[{"role": "user", "content": prompt}]
)
```


### OpenAI 's GPT

```python 
from openai import OpenAI

client = OpenAI()
response = client.responses.input_tokens.count(
    model= model,
    input=prompt
)
print(response.input_tokens)
```

## 1.2 Counting Overall Token Usage After Sending an API Request

### Google 's Gemini

- usage metadata from API response object

`response.usage_metadata`

```md
{
  "cache_tokens_details": null,
  "cached_content_token_count": null,
  "candidates_token_count": 992,
  "candidates_tokens_details": null,
  "prompt_token_count": 45,
  "prompt_tokens_details": [
    {
      "modality": "TEXT",
      "token_count": 45
    }
  ],
  "thoughts_token_count": null,
  "tool_use_prompt_token_count": null,
  "tool_use_prompt_tokens_details": null,
  "total_token_count": 1037,
  "traffic_type": null
}
```

- token counting
```python
prompt_token_count = response.usage_metadata.prompt_token_count
cached_content_token_count = response.usage_metadata.cached_content_token_count
candidates_token_count = response.usage_metadata.candidates_token_count
thoughts_token_count = response.usage_metadata.thoughts_token_count
```

- cost calculation
```python
input_cost    = (prompt_token_count / 1_000_000) * input_cost_per_1M
cached_cost   = (cached_content_token_count / 1_000_000) * cached_cost_per_1M 
output_cost   = (candidates_token_count / 1_000_000) * output_cost_per_1M
thinking_cost = (thoughts_token_count / 1_000_000) * thinking_cost_per_1M

total_cost = input_cost + cached_cost + output_cost + thinking_cost
```
### Anthropic 's Claude

- usage metadata from API response object
	
`response.usage`

```python
# print(json.dumps(response.usage.model_dump(), indent=2))
{
  "cache_creation": {
    "ephemeral_1h_input_tokens": 0,
    "ephemeral_5m_input_tokens": 0
  },
  "cache_creation_input_tokens": 0,
  "cache_read_input_tokens": 0,
  "inference_geo": "not_available",
  "input_tokens": 54,
  "output_tokens": 200,
  "server_tool_use": null,
  "service_tier": "standard"
}
# same structure for thinking mode enabled
```

- token counting

```python
input_tokens         = response.usage.input_tokens
cache_read_tokens    = response.usage.cache_read_input_tokens
cache_creation_tokens= response.usage.cache_creation_input_tokens
output_tokens        = response.usage.output_tokens
thinking_tokens      = response.usage.output_tokens  
# thinking tokens are included in output_tokens
```

- cost calculation

```python
input_cost    = (input_tokens / 1_000_000) * input_cost_per_1M
cache_read_cost    = (cache_read_tokens  / 1_000_000) * cache_read_cost_per_1M
cache_write_cost   = (cache_creation_tokens / 1_000_000) * cache_write_cost_per_1M
output_cost   = (output_tokens / 1_000_000) * output_cost_per_1M

total_cost = input_cost + cache_read_cost + cache_write_cost + output_cost
```

### OpenAI 's GPT

- usage metadata from API response object

```md
{
  "input_tokens": 54,
  "input_tokens_details": {
    "cached_tokens": 0
  },
  "output_tokens": 50,
  "output_tokens_details": {
    "reasoning_tokens": 0
  },
  "total_tokens": 104
}
```

- token counting

```python
input_tokens     = response.usage.input_tokens
cached_tokens    = response.usage.input_tokens_details.cached_tokens
output_tokens    = response.usage.output_tokens
reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
total_tokens     = response.usage.total_tokens
```

- cost calculation

```python
non_cached_input  = input_tokens - cached_tokens

input_cost     = (non_cached_input  / 1_000_000) * input_cost_per_1M
cached_cost    = (cached_tokens     / 1_000_000) * cached_cost_per_1M
output_cost    = (output_tokens     / 1_000_000) * output_cost_per_1M

total_cost = input_cost + cached_cost + output_cost
```


## 1.3 Summarization

- **Token input:** system prompt, user input, conversation history, retrieved information, tool calls 
- **Token output:** LLM response (thinking tokens in Gemini are separate; reasoning tokens in GPT are a subset of output)


| |**Gemini**|**Claude**|**GPT**|
|---|---|---|---|
|**Input**|`usage_metadata.prompt_token_count`|`usage.input_tokens`|`usage.input_tokens`|
|**Cached input**|`usage_metadata.cached_content_token_count`|`usage.cache_read_input_tokens`|`usage.input_tokens_details.cached_tokens`|
|**Cache write**|❌|`usage.cache_creation_input_tokens`|❌|
|**Output**|`usage_metadata.candidates_token_count`|`usage.output_tokens`|`usage.output_tokens`|
|**Thinking/Reasoning**|`usage_metadata.thoughts_token_count` — separate field|bundled inside `output_tokens`, not split|`usage.output_tokens_details.reasoning_tokens` — `0` for standard models; non-zero only on o-series (o1, o3)|
|**Total**|`usage_metadata.total_token_count`|sum manually `input + output`|`usage.total_tokens`|
|**Pre-flight count**|`client.models.count_tokens(model=..., contents=...)`|`client.messages.count_tokens(model=..., messages=...)`|`client.responses.input_tokens.count(model=..., input=..., instructions=...)`|
|**Thinking visible**|Yes — extract via part variables if `includeThoughts=True`|Yes — `block.type == "thinking"` in `response.content`|No|
|**Enable thinking**|Use thinking model e.g. `gemini-2.5-flash`, configure `thinking_budget`|`thinking={"type": "enabled", "budget_tokens": N}` or `"adaptive"`|Use o-series models (o1, o3-mini); optionally set `reasoning_effort`|

> - Gemini thinking tokens are **additive** — billed separately on top of output tokens
> - Claude thinking tokens are **bundled** — no separate field, priced as regular output
> - GPT reasoning tokens are a **subset** of output tokens — visible only on o-series models, billed at output rate
> - Only Claude charges for **cache write** (prompt caching creation cost)
> - GPT cached tokens are subtracted from input; Claude and Gemini treat them as separate fields

  

