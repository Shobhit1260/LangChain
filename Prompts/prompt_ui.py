from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st 
load_dotenv()
import os

llm= HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
    max_new_tokens=512,
    temperature=0.7
)

st.header("Chat with HuggingFace Model")
input_text=st.text_input("Enter your Prompt here")
chat_Model=ChatHuggingFace(llm=llm)
if st.button("Submit"):
    result=chat_Model.invoke(input_text)
    st.write(result.content)
