import streamlit as st
from llama_index.llms.openai import OpenAI

def load_llm(openai_api_key):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    st.session_state.api_key = openai_api_key
    st.session_state.messages = []
    st.success("Clave ingresada correctamente")
    return llm



def generate_response(question):
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    st.chat_message("user").write(question)

    # Generar respuesta con el modelo real
    response = llm.stream_chat(question)  # Aquí llamamos al modelo de OpenAI
    assistant_message = {"role": "assistant", "content": response}
    st.session_state.messages.append(assistant_message)
    st.chat_message("assistant").write(response)
    return response

st.set_page_config(page_title="Chatbot")
st.title("Chatbot")

openai_api_key = st.text_input(label="OpenAI API Key", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
if openai_api_key.startswith("sk-"):
    llm = load_llm(openai_api_key)

question = st.chat_input("Di algo")
if question:
    st.write(f"User has sent the following prompt: {question}")

generate_response(question)