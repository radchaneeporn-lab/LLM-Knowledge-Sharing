#gemini
from google import genai
import anthropic
from openai import OpenAI

client = genai.Client()
prompt = "The quick brown fox jumps over the lazy dog."

total_tokens = client.models.count_tokens(
    model="gemini-3-flash-preview", contents=prompt
)
print("total_tokens: ", total_tokens)

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents=prompt
)

print(response.usage_metadata)


#anthropic
client = anthropic.Anthropic()

response = client.messages.count_tokens(
    model="claude-opus-4-7",
    system="You are a scientist",
    messages=[{"role": "user", "content": "Hello, Claude"}],
)

print(response.json())


#gpt
client = OpenAI()

response = client.responses.input_tokens.count(
    model="gpt-5",
    input="Tell me a joke."
)
print(response.input_tokens)