import streamlit as st
# from langchain.llms import OpenAI
# from langchain.agents import initialize_agent, Tool, AgentType
# from langchain.agents import AgentExecutor
# from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from chatbot import execute as execute_chatbot
import os


# Configurar la clave API de OpenAI
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



# Configuración de la interfaz de Streamlit
st.title("Chatbot con LangChain")
st.write("¡Hola! Soy tu chatbot, ¿en qué te puedo ayudar?")


# Crear una lista para almacenar los mensajes de la conversación
if "messages" not in st.session_state:
    st.session_state.messages = []
    if os.path.exists('storage/memory.pkl'):
        os.remove('storage/memory.pkl')


st.sidebar.title("Configuración")

st.sidebar.text_input("Instrucciones", key="system_message", placeholder="Eres un asistente de inteligencia artificial enfocado a responder a las cuestiones del usuario.")

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Temperatura',
    0.0, 2.0, (0.1),
    step=0.1,
    key="temperature"
)

# Mostrar mensajes previos
for message in st.session_state.messages:
    st.markdown(message, unsafe_allow_html=True)


# Inicializa el estado de la sesión para almacenar el valor del campo de texto
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# Función para limpiar el campo de texto
def limpiar_campo():
    # st.session_state.text_input = ""  # Restablece el valor a vacío
    st.session_state["user_input"] = ""


# Función para añadir mensaje al historial de conversación
def add_message():
    message = st.session_state["user_input"]
    st.session_state.messages.append(f'<div style="background-color:#c0e8ff;padding:5pt;width: 75%;margin-left: 25%;">{message}</div>')
    limpiar_campo()
    # Obtener la respuesta del chatbot
    chatbot_response = execute_chatbot(message, st.session_state.system_message, st.session_state.temperature)
    st.session_state.messages.append(f'<div style="padding:5pt;width: 75%;">{chatbot_response}</div>')
    # st.write(chatbot_response)

# Interfaz de usuario
st.text_input("", value=st.session_state.text_input, on_change=add_message, key="user_input", placeholder="Escribe tu mensaje")
