import streamlit as st
from chatbot import execute as execute_chatbot
from utils import limpiar_memoria_conversacion_previa, inicializar_base_conocimiento_vectorizada


st.set_page_config(
    page_title="LangChain Chatbot",
)

# Configuración de la interfaz de Streamlit
st.title("Chatbot con LangChain")
st.write("¡Hola! Soy tu asistente, ¿en qué te puedo ayudar?")


# Preparamos una lista para almacenar los mensajes de la conversación
if "messages" not in st.session_state:
    st.session_state.messages = []
    limpiar_memoria_conversacion_previa()


st.sidebar.title("Configuración")

st.sidebar.text_input("Instrucciones / Personalidad", key="system_message",
                      help="Indicar instrucciones del comportamiento del chatbot.",
                      placeholder="Eres un asistente de inteligencia artificial enfocado a responder a las cuestiones del usuario.")

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Temperatura',
    0.0, 2.0, (0.0),
    step=0.1,
    key="temperature"
)

st.sidebar.checkbox("Integrar base de conocimientos", value=False, key="checkbox_base_conocimiento",
                    help="Marcar solo para consultas sobre la base de conocimientos sobre la historia de España.")

if st.session_state["checkbox_base_conocimiento"]:
    with st.spinner("Generando base de conocimiento..."):
        inicializar_base_conocimiento_vectorizada()

# Mostrar mensajes previos
for message in st.session_state.messages:
    st.markdown(message, unsafe_allow_html=True)


# Inicializa el estado de la sesión para almacenar el valor del campo de texto
if "text_input" not in st.session_state:
    st.session_state.text_input = ""


# Función para limpiar el campo de texto
def limpiar_entrada_texto_usuario():
    st.session_state["user_input"] = ""


# Función para añadir mensaje al historial de conversación
def enviar_entrada_texto_usuario():
    message = st.session_state["user_input"]
    st.session_state.messages.append(f'<div style="background-color:#c0e8ff;padding:5pt;width: 75%;margin-left: 25%;">{message}</div>')
    limpiar_entrada_texto_usuario()
    # Obtener la respuesta del chatbot
    chatbot_response = execute_chatbot(message, st.session_state.system_message, st.session_state.temperature, st.session_state.checkbox_base_conocimiento)
    st.session_state.messages.append(f'<div style="padding:5pt;width: 75%;">{chatbot_response}</div>')


# Interfaz de usuario
st.text_input("Escribe tu mensaje", value=st.session_state.text_input, on_change=enviar_entrada_texto_usuario, key="user_input",
              placeholder="Escribe tu mensaje", label_visibility='hidden')
