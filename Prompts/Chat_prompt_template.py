# Chat_prompt_template is used to give a structure to multiple turn conversations 
# used for role-based interactions with the model,
# like Chatbots,support-agents etc.

# MessagesPlaceholder is used to handle dynamic message history in conversations.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm= HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
    temperature=0.7
    max_new_tokens=50
)

chat_Model=ChatHuggingFace(llm=llm)


chat_template=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chat_history=[]

while True:
    user_input=input("User: ")
    chat_prompt=chat_template.format_messages(
            chat_history=chat_history,
            input=user_input
    )
    if(user_input.lower()=='exit'):
        break
    response=chat_Model.invoke(chat_prompt)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))
    print("Bot:",response.content)

print(chat_history)    
