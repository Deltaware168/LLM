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
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

st.set_page_config(
    page_title = "PDF Analyzer"
)

st.title("PDF Analyzer")

#funcion que retorna el modelo llm con su api key
def load_LLM(openai_api_key):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm

st.markdown("## Ingrese su Key de la API de OpenAI")

#funcion que retorna la api key de openai
def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text



openai_api_key = get_openai_api_key()
if openai_api_key:
    llm = load_LLM(openai_api_key)
    st.write("LLM iniciado")