import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Noticias Personalizadas", layout="wide")

# Título de la aplicación
st.title("Configuración de API Key y Noticias Personalizadas 🔑📰")

# 📌 Entrada de texto para la API Key
api_key = st.text_input("Ingrese su API Key aquí:", type="password")

# Línea divisoria
st.write("---")

# 📌 Panel lateral para sitios bloqueados
st.sidebar.title("🚫 Sitios Bloqueados")

# Inicializar `session_state` para almacenar sitios bloqueados
if "blocked_sites" not in st.session_state:
    st.session_state["blocked_sites"] = ["www.ejemplo1.com", "www.ejemplo2.com", "www.ejemplo3.com"]

# Input para agregar un nuevo sitio bloqueado
new_site = st.sidebar.text_input("Ingrese un sitio a bloquear:")

# Botón para añadir el sitio a la lista
if st.sidebar.button("Bloquear sitio"):
    if new_site and new_site not in st.session_state["blocked_sites"]:
        st.session_state["blocked_sites"].append(new_site)

# Mostrar lista actualizada de sitios bloqueados
for site in st.session_state["blocked_sites"]:
    st.sidebar.write(f"❌ {site}")

# 📌 Crear columnas para los recuadros
col1, col2, col3, col4 = st.columns(4)

# 📌 Definir categorías y sus descripciones
categorias = ["Tecnología", "Salud", "OpenSource", "Investigaciones"]
descripciones = [
    "Últimas noticias sobre avances tecnológicos e inteligencia artificial.",
    "Información relevante sobre medicina, bienestar y estudios clínicos.",
    "Noticias y novedades sobre proyectos de código abierto.",
    "Descubrimientos científicos y avances en distintas áreas de investigación."
]

# 📌 Mostrar recuadros con efecto hover
for col, categoria, descripcion in zip([col1, col2, col3, col4], categorias, descripciones):
    with col:
        st.markdown(
            f"""
            <div style="padding: 15px; border: 2px solid #4CAF50; border-radius: 10px; text-align: center; background-color: #f9f9f9; transition: 0.3s;">
                <h3 style="color: #2E7D32;">{categoria}</h3>
                <p style="color: #555;">{descripcion}</p>
            </div>
            <style>
            div:hover {{
                background-color: #FFA726;  /* Cambia a anaranjado */
                border-color: #FF9800;  /* Borde anaranjado */
                transform: scale(1.05); /* Efecto ligero de zoom */
            }}
            </style>
            """,
            unsafe_allow_html=True
        )