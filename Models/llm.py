
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the HuggingFace endpoint with conversational task
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
    max_new_tokens=512,
    temperature=0.7
)

# Wrap it with ChatHuggingFace for proper conversational handling
chat_model = ChatHuggingFace(llm=llm)

# Use the chat model
result = chat_model.invoke("what is langchain in simple words?")

print("Result:", result.content)