import streamlit as st
from llama_index.llms.openai import OpenAI

def load_llm(openai_api_key):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    st.session_state.api_key = openai_api_key
    st.success("Clave ingresada correctamente")
    return llm

st.set_page_config(page_title="Chatbot")
st.title("Chatbot")

openai_api_key = st.text_input(label="OpenAI API Key", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
if openai_api_key.startswith("sk-"):
    llm = load_llm(openai_api_key)

prompt = st.chat_input(
    "Say something and/or attach an image",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"],
)
if prompt and prompt.text:
    st.markdown(prompt.text)
if prompt and prompt["files"]:
    st.image(prompt["files"][0])