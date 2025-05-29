import streamlit as st, pandas as pd, tempfile, os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="PDF Analyzer")
st.title("PDF Analyzer")
st.markdown("## Ingrese su Key de la API de OpenAI")
openai_api_key = st.text_input("OpenAI API Key", "Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
if openai_api_key.startswith("sk-"): 
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key); st.success("Clave ingresada correctamente")

uploaded_file = st.file_uploader("Carga un archivo PDF", type=["pdf"])
txt_input = st.text_input("Ingrese su pregunta: ")

if uploaded_file is not None:
    # Crear un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    # Cargar el PDF usando PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    loaded_data = loader.load()

    recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=0,
            separators=["\n\n", "\n", "(?<=\.)", " ", ""]
    )
    
    textos = []
    for page in loaded_data:
        textos.extend(recursive_splitter.create_documents([page.page_content]))

    db = Chroma.from_documents(textos, OpenAIEmbeddings())
    retriever = db.as_retriever(search_kwargs={"k":5})

    template = """
    Responde a la pregunta basandote solo con la informacion que se te brinda, si no sabes la respuesta no te la inventes

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


    response = chain.invoke(txt_input)
    st.write(response)


