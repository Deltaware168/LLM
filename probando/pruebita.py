import streamlit as st
import pandas as pd
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os  # Para eliminar el archivo temporal después de usarlo

st.set_page_config(page_title="PDF Analyzer")
st.title("PDF Analyzer")

# Campo para ingresar la API Key
st.markdown("## Ingrese su Key de la API de OpenAI")
openai_api_key = st.text_input(label="OpenAI API Key", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")

# Solo inicializar el modelo si la API Key está presente
if openai_api_key.startswith("sk-"):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    st.success("Clave ingresada correctamente")

# **Archivo PDF**
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

def generate_response(question, uploaded_file):
    if uploaded_file is None:
        return "Por favor, sube un archivo PDF antes de hacer una pregunta."

    # Guardar el archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        # Cargar el PDF con PyPDFLoader
        loader = PyPDFLoader(temp_path)
        loaded_data = loader.load()

        # Aplicar la técnica RAG con RecursiveCharacterTextSplitter
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=0,
            separators=["\n\n", "\n", r"(?<=\.)", " ", ""]
        )

        # **Corrección: Procesar TODAS las páginas correctamente**
        textos = recursive_splitter.split_documents(loaded_data)

        # Crear la base de datos de embeddings para recuperación
        db = FAISS.from_documents(textos, OpenAIEmbeddings(openai_api_key=openai_api_key))
        retriever = db.as_retriever(search_kwargs={"k": 3})

        template = """
        Responde a la pregunta basándote solo en la información proporcionada. Si no sabes la respuesta, no la inventes.

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

        return chain.invoke(question)

    finally:
        # **Eliminar archivo temporal después de procesarlo**
        os.remove(temp_path)

# **Ahora la función recibe `uploaded_file` para procesarlo**
txt_input = st.text_input("Ingrese su pregunta: ")

if txt_input and uploaded_file:
    response = generate_response(txt_input, uploaded_file)
    st.write(response)