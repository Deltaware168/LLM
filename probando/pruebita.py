import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="PDF Analyzer")
st.title("PDF Analyzer")

# Campo para ingresar la API Key
st.markdown("## Ingrese su Key de la API de OpenAI")
openai_api_key = st.text_input(label="OpenAI API Key", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")

# Subida de archivos
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

def generate_response(question, uploaded_file):
    if uploaded_file is None:
        return "Por favor, sube un archivo PDF antes de hacer una pregunta."

    # Leer el archivo PDF en memoria con pdfplumber
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    # Aplicar la técnica RAG con RecursiveCharacterTextSplitter
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=0,
        separators=["\n\n", "\n", "(?<=\.)", " ", ""]
    )

    textos = recursive_splitter.create_documents([text])

    # Crear la base de datos de embeddings para recuperación
    db = Chroma.from_documents(textos, OpenAIEmbeddings())
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
        | OpenAI(temperature=0, openai_api_key=st.session_state.get("openai_api_key", ""))
        | StrOutputParser()
    )

    return chain.invoke(question)

# Captura la pregunta del usuario
txt_input = st.text_input("Ingrese su pregunta: ")

if txt_input and uploaded_file:
    response = generate_response(txt_input, uploaded_file)
    st.write(response)