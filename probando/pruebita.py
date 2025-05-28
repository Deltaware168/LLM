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

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

def generate_response(question, uploaded_file):
    if uploaded_file is None: return "Por favor, sube un archivo PDF antes de hacer una pregunta."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file: temp_file.write(uploaded_file.getvalue()); temp_path = temp_file.name
    try:
        loader = PyPDFLoader(temp_path)
        loaded_data = loader.load()
        recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0, separators=["\n\n", "\n", r"(?<=\.)", " ", ""])
        textos = [page.page_content for page in loaded_data]
        documentos = recursive_splitter.create_documents(textos)
        db = FAISS.from_documents(documentos, OpenAIEmbeddings(openai_api_key=openai_api_key))
        retriever = db.as_retriever(search_kwargs={"k": 10})
        retrieved_docs = retriever.get_relevant_documents(question)
        st.write("Documentos recuperados:")
        for doc in retrieved_docs: st.write(doc.page_content)
        template = """Responde a la pregunta basándote solo en la información proporcionada. Si no sabes la respuesta, no la inventes.\n\n{pregunta}\n\nContexto:\n{contexto}"""
        prompt = PromptTemplate.from_template(template)
        chain = ({"contexto": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "pregunta": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        return chain.invoke(question)
    finally: os.remove(temp_path)

txt_input = st.text_input("Ingrese su pregunta: ")
if txt_input and uploaded_file: st.write(generate_response(txt_input, uploaded_file))