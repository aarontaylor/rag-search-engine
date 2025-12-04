import os
from dotenv import load_dotenv
from google import genai

# Load .env file
load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Did you load .env?")

print(f"Using key {api_key[:6]}...")

# Initialize client
client = genai.Client(api_key=api_key)

# Call Gemini model
response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
)

# Print model response
print(response.text)

# Print token usage
usage = response.usage_metadata
print(f"Prompt Tokens: {usage.prompt_token_count}")
print(f"Response Tokens: {usage.candidates_token_count}")
