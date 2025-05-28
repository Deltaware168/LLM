import streamlit as st
import pandas as pd
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="PDF Analyzer"
)

st.title("PDF Analyzer")

# Campo para ingresar la API Key
st.markdown("## Ingrese su Key de la API de OpenAI")
openai_api_key = st.text_input(label="OpenAI API Key", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")

# Mostrar otros elementos incluso si la API Key no está presente
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Procesamiento del archivo
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

# Solo inicializar el modelo si la API Key está presente
if openai_api_key:
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    st.success("Modelo LLM inicializado correctamente")

print("parte 2")
