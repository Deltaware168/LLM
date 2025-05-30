import streamlit as st, pandas as pd, tempfile, os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="PDF Analyzer")
st.title("PDF Analyzer")
st.markdown("## Ingrese su Key de la API de OpenAI")

# Inicializar valores en session_state
if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = None

def load_llm(openai_api_key):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    st.session_state.api_key = openai_api_key
    st.success("Clave ingresada correctamente")
    return llm

openai_api_key = st.text_input(label="OpenAI API Key", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")

if openai_api_key.startswith("sk-"):
    llm = load_llm(openai_api_key)

uploaded_file = st.file_uploader("Carga un archivo PDF", type="pdf", key="pdf_uploader")

if uploaded_file:
    st.session_state.file_uploaded = uploaded_file

# El campo de texto se muestra pero está deshabilitado si no se ha subido un archivo o ingresado la clave
txt_input = st.text_input("Ingrese su pregunta:", disabled=not (st.session_state.api_key and st.session_state.file_uploaded))

def generate_response(question, uploaded_file):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        loaded_data = loader.load()
        recursive_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

        textos = []
        for page in loaded_data:
            textos.extend(recursive_splitter.create_documents([page.page_content]))

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.from_documents(textos, embeddings)
        retriever = db.as_retriever(search_kwargs={"k":10})

        template = """Responde a la pregunta basándote solo en la información que se te brinda. Si no sabes la respuesta, no te la inventes.

        {pregunta}

        Contexto:
        {contexto}
        """

        prompt = PromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        chain = (
            {"contexto": retriever | format_docs, "pregunta": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(question)
        return response

if txt_input and uploaded_file and openai_api_key:
    response = generate_response(txt_input, uploaded_file)
    st.write(response)