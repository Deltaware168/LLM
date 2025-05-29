import streamlit as st, pandas as pd, tempfile, os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="PDF Analyzer")
st.title("PDF Analyzer")
st.markdown("## Ingrese su Key de la API de OpenAI")
openai_api_key = st.text_input("OpenAI API Key", "Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
if openai_api_key.startswith("sk-"): llm = OpenAI(temperature=0, openai_api_key=openai_api_key); st.success("Clave ingresada correctamente")

import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI

st.set_page_config(page_title="PDF Analyzer")
st.title("PDF Analyzer")
st.markdown("## Ingrese su Key de la API de OpenAI")

openai_api_key = st.text_input("OpenAI API Key", "Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
if openai_api_key.startswith("sk-"):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    st.success("Clave ingresada correctamente")

uploaded_file = st.file_uploader("Carga un archivo PDF", type=["pdf"])

if uploaded_file is not None:
    # Crear un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    # Cargar el PDF usando PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Mostrar el contenido extraído sin dividirlo
    text = "\n\n".join([doc.page_content for doc in documents])
    st.text_area("Contenido del PDF:", text)
