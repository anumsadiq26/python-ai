
"""Task 2.1: Make the first AI call using Ollama."""

from langchain.chat_models import init_chat_model
import time

topic = input("Enter a topic: ").strip()
level = input("Choose level (beginner/expert): ").strip().lower()

# why: The system message guides the AI's role and answer length.
if level == "expert":
    system_message = "You are an expert tutor. Give a detailed technical explanation."
else:
    system_message = "You are a beginner-friendly tutor. Explain briefly using simple words."

llm = init_chat_model(
    "ollama:llama3.2:1b",
    temperature=0.2,
)

start_time = time.time()

response = llm.invoke([
    ("system", system_message),
    ("human", f"Explain this topic: {topic}"),
])

duration = time.time() - start_time

print("\nAI Answer:")
print(response.text)

# why: Usage metadata helps us measure the tokens used by the AI.
usage = response.usage_metadata or {}
print(f"\nTotal tokens: {usage.get('total_tokens', 'Not available')}")
print(f"Response time: {duration:.1f} seconds")
