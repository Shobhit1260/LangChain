# We can use String literal but used PromptTemplate for reusability,
# it will help in Validation (like any missing input variables), and maintainability (easy to update prompt structure).
# And in Chaining 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st 
load_dotenv()
import os
from langchain_core.prompts import PromptTemplate,load_prompt

llm= HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
    temperature=0.7
)

st.header("Research Paper Summarizer")

paper_input = st.selectbox(
"Select a paper to summarize",
["Attention is All You Need", "Word2Vec", "BERT", "ResNet", "Transformer"]
)

style_input = st.selectbox(
"Select explanation style",
["Simple and Intuitive", "Code Heavy", "Maths Heavy"]
)

length_input = st.selectbox(
"Select explanation length",
["Short", "Medium", "Long"]
)

template=load_prompt("template.json")

formatted_prompt =template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

chat_Model=ChatHuggingFace(llm=llm)
if st.button("Submit"):
    result=chat_Model.invoke(formatted_prompt)
    st.write(result.content)



