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

openai_api_key = st.text_input("OpenAI API Key", "Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
if openai_api_key.startswith("sk-"): 
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key); st.success("Clave ingresada correctamente")





def generate_response(question):
    uploaded_file = st.file_uploader("Carga un archivo PDF", type="pdf")
    if uploaded_file is not None:
    # Crear un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        st.write("archivo temporal creado")
    # Cargar el PDF usando PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        loaded_data = loader.load()
        st.write("creado loaded_data")
        recursive_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=50
        )
        st.write("creado splitter")
        textos = []
        for page in loaded_data:
            textos.extend(recursive_splitter.create_documents([page.page_content]))
        
        st.write("cargado elementos de texto")

        embeddings = OpenAIEmbeddings(
            openai_api_key= openai_api_key
        )

        st.write("creacion de embedding")

        db = FAISS.from_documents(textos, embeddings)

        st.write("USO DE BASE DE DATOS CHROMA")
        retriever = db.as_retriever(search_kwargs={"k":5})

        st.write("creacion de retriever")
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
        response = chain.invoke(question)
    return response

txt_input = st.text_input("Ingrese su pregunta: ")
response = generate_response(txt_input)
st.write(response)
#-------------------------------------------------------------------------




