from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm= HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
    temperature=0.7
)
chat_history=[
    SystemMessage(content="You are a helpful assistant."),
]


chat_model=ChatHuggingFace(llm=llm)

while True:
   user_message=input("User: ")
   chat_history.append(HumanMessage(content=user_message))
   if(user_message.lower()=='exit'):
    break
   response=chat_model.invoke(chat_history)
   chat_history.append(AIMessage(content=response.content))
   print("Bot:",response.content) 

   
print(chat_history);   

