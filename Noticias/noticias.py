import streamlit as st

# Título de la aplicación
st.title("Configuración de API Key y Noticias Personalizadas 🔑📰")

# Entrada de texto para la API Key
api_key = st.text_input("Ingrese su API Key aquí:", type="password")

# Línea divisoria para organización visual
st.write("---")

# Crear columnas para los recuadros
col1, col2, col3, col4 = st.columns(4)

# Definir categorías y sus descripciones
categorias = ["Tecnología", "Salud", "OpenSource", "Investigaciones"]
descripciones = [
    "Últimas noticias sobre avances tecnológicos e inteligencia artificial.",
    "Información relevante sobre medicina, bienestar y estudios clínicos.",
    "Noticias y novedades sobre proyectos de código abierto.",
    "Descubrimientos científicos y avances en distintas áreas de investigación."
]

# Mostrar recuadros con títulos y descripciones
for col, categoria, descripcion in zip([col1, col2, col3, col4], categorias, descripciones):
    with col:
        st.subheader(categoria)
        st.write(descripcion)